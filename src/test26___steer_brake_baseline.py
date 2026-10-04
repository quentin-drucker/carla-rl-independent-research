"""test26___steer_brake_baseline.py

Deterministic steering-plus-braking baseline under CARLA DEFAULT physics
(Week 4 decision). For each (speed, onset TTC) case it runs three matched
controller modes on the same scenario -- see steer_brake_baseline.py:

    no_intervention | brake_only | brake_steer

Scenario: Town04_Opt route from spawn 242 (test3), stationary pedestrian at
route center ("stationary" walker), ClearSunset with the sun at the horizon.
Onset ("oracle onset"): run_scenario's trigger, which fires when the ego
center is speed x onset_ttc_s from the pedestrian; every mode reacts from
that same tick using ground truth, never LiDAR detection timing.

Scored by encounter_metrics.compute_encounter_metrics over an identical
post-onset horizon, with oriented-footprint contact/clearance and per-tick
footprint drivability (all four corners on a Driving lane).

Usage (CARLA running with default physics; from src/ in the venv):
    python -X utf8 test26___steer_brake_baseline.py --smoke
    python -X utf8 test26___steer_brake_baseline.py --mph 35 45 --ttc 0.8 1.0 1.2 1.4 1.6 1.8
    python -X utf8 test26___steer_brake_baseline.py --rescore <sweep_dir>   # re-score saved traces

Drivability uses map_drivability.classify_point_drivability_seam_tolerant
(since 2026-10-02). Sweep 20261002_004906 was scored with the strict check,
which flagged a lane-seam false positive; --rescore recomputes drivability
and metrics from its saved traces into *_v2 files without re-simulating.

Evidence (git-ignored, never overwritten):
    src/runs/steer_brake_baseline/<timestamp>_steer_brake_baseline_default/
        summary.csv, sweep_manifest.json, <case>/<mode>/{trace.csv,
        drivability.json, result.json, metrics.json, manifest.json}
"""

import argparse
import csv
import datetime
import json
import os
import platform
import subprocess
import sys

import carla

from encounter_metrics import EncounterProtocol, compute_encounter_metrics, summarize_for_console
from map_drivability import classify_point_drivability_seam_tolerant
from physics_backend import BACKEND_DEFAULT, create_run_dir
from reactive_avoidance import compute_required_clearance_offset_m
from scenario_config import ScenarioConfig
from steer_brake_baseline import (
    MODES,
    footprint_corners_xy,
    make_hazard_command_fn,
    mode_uses_steering,
    onset_gap_m,
)
from test3___ped_intrusion_scenario import FIXED_DT, run_scenario
from test5___scripted_pedestrian_steering import build_evasive_offset_fn
from trace_schema import read_trace_csv
from trajectory_recording import TrajectoryRecorder

RUN_FAMILY = "steer_brake_baseline"
RUNS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", RUN_FAMILY)

# Predeclared scenario/controller parameters (recorded in every manifest).
ENCOUNTER_DISTANCE_M = 160.0      # long enough to reach 45 mph before onset
BRAKE_TARGET = 1.0                # both braking modes: the same full brake
BRAKE_RAMP_UP_PER_S = 20.0        # "slam" end of ScenarioConfig's range: each
                                  # strategy's physical best case, not comfort
SWERVE_SIDE_SIGN = +1             # route-right (more room at this location, Week 3)
SWERVE_MARGIN_M = 0.6             # test18's buffered clearance margin
SWERVE_SHIFT_DURATION_S = 0.5     # test18's live-validated shift duration
PROTOCOL = EncounterProtocol(horizon_s=8.0)
SMOKE_CASES = [(35.0, 1.2)]


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="steering-plus-braking baseline (see docstring)")
    p.add_argument("--mph", type=float, nargs="+", default=[35.0, 45.0])
    p.add_argument("--ttc", type=float, nargs="+", default=[0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--smoke", action="store_true", help=f"run only {SMOKE_CASES}")
    p.add_argument("--rescore", metavar="SWEEP_DIR",
                   help="recompute drivability + metrics from a sweep's saved traces (no simulation)")
    return p.parse_args(argv)


def _footprint_drivable(carla_map, x, y, yaw_deg, z):
    """(flag, seam_corrections): flag is True if all four footprint corners
    are on a Driving lane, False if any is not, None if any query failed."""
    statuses, corrections = [], 0
    for cx, cy in footprint_corners_xy(x, y, yaw_deg):
        status, _, corrected = classify_point_drivability_seam_tolerant(
            carla_map, carla.Location(x=cx, y=cy, z=z))
        statuses.append(status)
        corrections += int(corrected)
    if "non_drivable" in statuses:
        return False, corrections
    return (None if "unknown" in statuses else True), corrections


def _run_one(*, mph, ttc, mode, swerve_offset_m, case_dir, client):
    cfg = ScenarioConfig(
        walker_cross="stationary", walker_side="left", trigger_ttc_s=ttc,
        target_mph=mph, encounter_distance_m=ENCOUNTER_DISTANCE_M,
        braking_ramp_up_per_s=BRAKE_RAMP_UP_PER_S, brake_profile="step_constant",
        weather_preset="ClearSunset", sun_altitude_deg=0,
        sim_seconds=ENCOUNTER_DISTANCE_M / (mph * 0.44704) + 30.0,
        run_id=f"{mph:g}mph_ttc{ttc:g}_{mode}",
    )
    recorder = TrajectoryRecorder(dt_s=FIXED_DT)
    drivable_flags = []
    state = {"tick": 0, "map": None, "onset_marked": False, "seam_corrections": 0}

    def observer(sim_time_s, triggered, telemetry):
        index = state["tick"]
        state["tick"] += 1
        if telemetry is None:
            return
        marker = None
        if triggered and not state["onset_marked"]:
            marker, state["onset_marked"] = "onset", True
        recorder.record(tick_index=index, sim_time_s=sim_time_s, telemetry=telemetry, event_marker=marker)
        if state["map"] is None:
            state["map"] = client.get_world().get_map()
        flag, corrections = _footprint_drivable(state["map"], telemetry["pos_x_m"], telemetry["pos_y_m"],
                                                telemetry["yaw_deg"], telemetry.get("pos_z_m") or 0.0)
        drivable_flags.append(flag)
        state["seam_corrections"] += corrections

    steering_fn = (build_evasive_offset_fn(peak_offset_m=swerve_offset_m, shift_duration_s=SWERVE_SHIFT_DURATION_S)
                   if mode_uses_steering(mode) else None)
    result = run_scenario(
        cfg, plot_after=False, lateral_offset_fn=steering_fn, tick_observer=observer,
        post_crossing_settle_s=PROTOCOL.horizon_s + 1.0,
        hazard_command_fn=make_hazard_command_fn(mode, brake_target=BRAKE_TARGET),
    )
    recorder.tag_closest_approach()
    metrics = compute_encounter_metrics(recorder.ticks, onset_time_s=result.trigger_time_s,
                                        protocol=PROTOCOL, footprint_drivable=drivable_flags)

    os.makedirs(case_dir, exist_ok=False)
    recorder.write(os.path.join(case_dir, "trace.csv"))
    with open(os.path.join(case_dir, "drivability.json"), "x") as f:
        json.dump(drivable_flags, f)
    with open(os.path.join(case_dir, "result.json"), "x") as f:
        json.dump(result.to_dict(), f, indent=2)
    with open(os.path.join(case_dir, "metrics.json"), "x") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    with open(os.path.join(case_dir, "manifest.json"), "x") as f:
        json.dump({
            "mode": mode, "target_mph": mph, "onset_ttc_s": ttc,
            "onset_gap_m_at_target_speed": onset_gap_m(speed_mps=mph * 0.44704, onset_ttc_s=ttc),
            "brake_target": BRAKE_TARGET, "brake_ramp_up_per_s": BRAKE_RAMP_UP_PER_S,
            "steering": ({"peak_offset_m": swerve_offset_m, "side_sign": SWERVE_SIDE_SIGN,
                          "margin_m": SWERVE_MARGIN_M, "shift_duration_s": SWERVE_SHIFT_DURATION_S,
                          "recovered": steering_fn.recovered,
                          "used_fallback_timeout": steering_fn.used_fallback_timeout}
                         if steering_fn else None),
            "config": cfg.to_dict(),
            "drivability_check": "seam_tolerant",
            "drivability_seam_corrections": state["seam_corrections"],
            "physics_backend": BACKEND_DEFAULT,
        }, f, indent=2)
    return result, metrics


def main(argv=None):
    args = _parse_args(argv)
    cases = SMOKE_CASES if args.smoke else [(m, t) for m in args.mph for t in args.ttc]

    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    client_version, server_version = client.get_client_version(), client.get_server_version()
    if client_version != server_version:
        print(f"[test26] client {client_version} != server {server_version}; refusing", file=sys.stderr)
        return 2

    if args.rescore:
        return _rescore(args.rescore, client)

    swerve_offset_m = compute_required_clearance_offset_m(  # already signed by side_sign
        pedestrian_lateral_m=0.0, side_sign=SWERVE_SIDE_SIGN, safety_margin_m=SWERVE_MARGIN_M)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = create_run_dir(runs_root=RUNS_ROOT, timestamp=timestamp, run_family=RUN_FAMILY,
                               backend=BACKEND_DEFAULT)
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    with open(os.path.join(sweep_dir, "sweep_manifest.json"), "x") as f:
        json.dump({
            "created_at_iso": datetime.datetime.now().isoformat(),
            "git_commit": _git("rev-parse", "HEAD"), "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_tracked_changes": dirty, "git_dirty": bool(dirty),
            "physics_backend": BACKEND_DEFAULT,
            "physics_backend_note": "CARLA default vehicle physics; Chrono never enabled (Week 4 decision).",
            "carla_client_version": client_version, "carla_server_version": server_version,
            "python_version": sys.version, "platform": platform.platform(),
            "cases": cases, "modes": args.modes, "protocol": PROTOCOL.__dict__,
            "encounter_distance_m": ENCOUNTER_DISTANCE_M, "brake_target": BRAKE_TARGET,
            "brake_ramp_up_per_s": BRAKE_RAMP_UP_PER_S, "swerve_peak_offset_m": swerve_offset_m,
            "swerve_shift_duration_s": SWERVE_SHIFT_DURATION_S, "random_seed": None,
            "seed_note": "fully scripted and deterministic; no random sampling",
        }, f, indent=2)
    print(f"[test26] {len(cases)} case(s) x {len(args.modes)} mode(s); swerve offset {swerve_offset_m:+.3f} m")
    print(f"[test26] evidence: {sweep_dir}")

    rows = []
    for mph, ttc in cases:
        for mode in args.modes:
            case_dir = os.path.join(sweep_dir, f"{mph:g}mph_ttc{ttc:g}", mode)
            print(f"\n[test26] ===== {mph:g} mph, onset TTC {ttc:g} s, {mode} =====")
            result, m = _run_one(mph=mph, ttc=ttc, mode=mode, swerve_offset_m=swerve_offset_m,
                                 case_dir=case_dir, client=client)
            for line in summarize_for_console(m):
                print("    " + line)
            rows.append(_summary_row(m, mph=mph, ttc=ttc, mode=mode,
                                     collision_sensor=result.collision_detected,
                                     trigger_time_s=result.trigger_time_s))
            with open(os.path.join(sweep_dir, "summary.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
                w.writeheader()
                w.writerows(rows)

    _print_summary(rows)
    print(f"\n[test26] evidence: {sweep_dir}")
    return 0


SUMMARY_COLUMNS = ["mph", "onset_ttc_s", "mode", "outcome", "safe_success", "window_complete",
                   "speed_at_onset_mps", "min_clearance_m", "contact_speed_mps", "time_to_stop_s",
                   "stopping_distance_m", "speed_when_passing_mps", "max_abs_route_lateral_m",
                   "final_route_lateral_m", "returned_to_route", "drivable_violation_ticks",
                   "peak_decel_normal_speed_mps2", "max_abs_jerk_normal_speed_mps3",
                   "max_abs_lateral_accel_mps2", "carla_collision_sensor", "trigger_time_s"]


def _summary_row(m, *, mph, ttc, mode, collision_sensor, trigger_time_s):
    d = m.to_dict()
    return {**{c: d.get(c) for c in SUMMARY_COLUMNS}, "mph": mph, "onset_ttc_s": ttc, "mode": mode,
            "carla_collision_sensor": collision_sensor, "trigger_time_s": trigger_time_s}


def _rescore(sweep_dir, client):
    """Recompute seam-tolerant drivability and metrics from saved traces.
    Writes drivability_v2.json / metrics_v2.json per run and
    rescored_summary.csv; never overwrites the original files."""
    carla_map = client.get_world().get_map()
    rows = []
    for case in sorted(os.listdir(sweep_dir)):
        for mode in MODES:
            run_dir = os.path.join(sweep_dir, case, mode)
            if not os.path.isfile(os.path.join(run_dir, "trace.csv")):
                continue
            ticks = read_trace_csv(os.path.join(run_dir, "trace.csv"))
            with open(os.path.join(run_dir, "result.json")) as f:
                result = json.load(f)
            with open(os.path.join(run_dir, "manifest.json")) as f:
                manifest = json.load(f)
            flags, corrections = [], 0
            for t in ticks:
                flag, c = _footprint_drivable(carla_map, t.pos_x_m, t.pos_y_m, t.yaw_deg, t.pos_z_m)
                flags.append(flag)
                corrections += c
            m = compute_encounter_metrics(ticks, onset_time_s=result["trigger_time_s"],
                                          protocol=PROTOCOL, footprint_drivable=flags)
            with open(os.path.join(run_dir, "drivability_v2.json"), "x") as f:
                json.dump({"check": "seam_tolerant", "seam_corrections": corrections, "flags": flags}, f)
            with open(os.path.join(run_dir, "metrics_v2.json"), "x") as f:
                json.dump(m.to_dict(), f, indent=2)
            rows.append(_summary_row(m, mph=manifest["target_mph"], ttc=manifest["onset_ttc_s"], mode=mode,
                                     collision_sensor=result["collision_detected"],
                                     trigger_time_s=result["trigger_time_s"]))
            print(f"[test26] rescored {case}/{mode}: {m.outcome} safe={m.safe_success} "
                  f"violations={m.drivable_violation_ticks} seam_corrections={corrections}")
    rows.sort(key=lambda r: (r["mph"], r["onset_ttc_s"], MODES.index(r["mode"])))
    with open(os.path.join(sweep_dir, "rescored_summary.csv"), "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    _print_summary(rows)
    return 0


def _print_summary(rows):
    print("\n[test26] SUMMARY")
    print(f"{'mph':>4} {'ttc':>4} {'mode':16} {'outcome':22} {'safe':5} {'v_onset':>7} "
          f"{'min_clr':>7} {'v_contact':>9} {'stop_m':>7} {'max_lat':>7} {'drv_viol':>8}")

    def f(v, spec):
        return "-" if v is None else format(v, spec)
    for r in rows:
        print(f"{r['mph']:4g} {r['onset_ttc_s']:4g} {r['mode']:16} {r['outcome']:22} {str(r['safe_success']):5} "
              f"{f(r['speed_at_onset_mps'], '7.2f')} {f(r['min_clearance_m'], '7.2f')} "
              f"{f(r['contact_speed_mps'], '9.2f')} {f(r['stopping_distance_m'], '7.2f')} "
              f"{f(r['max_abs_route_lateral_m'], '7.2f')} {f(r['drivable_violation_ticks'], '8')}")


if __name__ == "__main__":
    sys.exit(main())
