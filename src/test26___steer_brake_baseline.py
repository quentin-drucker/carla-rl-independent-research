"""test26___steer_brake_baseline.py

Deterministic steering-plus-braking baseline under CARLA DEFAULT physics
(Week 4 decision). For each (speed, onset TTC) case it runs matched
controller modes on the same scenario -- see steer_brake_baseline.py:

    no_intervention | brake_only | brake_steer | brake_passage_edge (Phase 2: Izmirli's
    passage-edge rule, passage.py; the passage is computed once from the stationary
    pedestrian's position and recorded in each manifest)

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
    python -X utf8 test26___steer_brake_baseline.py --passage-u 0.25 0.5 0.75 1.0   # brake_passage_edge once per aim position
    python -u -X utf8 test26___steer_brake_baseline.py --resume <sweep_dir>  # continue after a crash
    python -X utf8 test26___steer_brake_baseline.py --table <sweep_dir>     # print the best-route table (no CARLA needed)
    add --relaunch-carla to restart a crashed CARLA server automatically (renderless)

Drivability uses map_drivability.classify_point_drivability_seam_tolerant
(since 2026-10-02). Sweep 20261002_004906 was scored with the strict check,
which flagged a lane-seam false positive; --rescore recomputes drivability
and metrics from its saved traces into *_v2 files without re-simulating.

Evidence (git-ignored, never overwritten):
    src/runs/steer_brake_baseline/<timestamp>_steer_brake_baseline_default/
        summary.csv, sweep_manifest.json, progress.log, best_route_table.md,
        <case>/<run label>/{trace.csv, drivability.json, result.json,
        metrics.json, manifest.json, row.json}
    The run label is the mode, plus _u<value> for brake_passage_edge.
"""

import argparse
import csv
import dataclasses
import datetime
import glob
import json
import os
import platform
import subprocess
import sys
import time

import carla

from encounter_metrics import EncounterProtocol, compute_encounter_metrics, summarize_for_console
from fixed_rescore_protocol import eta_text
from map_drivability import classify_point_drivability_seam_tolerant
from physics_backend import BACKEND_DEFAULT, create_run_dir
from reactive_avoidance import compute_required_clearance_offset_m
from scenario_config import ScenarioConfig
from passage_carla import passage_at_location
from steer_brake_baseline import (
    MODE_BRAKE_PASSAGE_EDGE,
    MODES,
    PassageEdgeOffset,
    expand_runs,
    footprint_corners_xy,
    format_best_route_table,
    make_hazard_command_fn,
    max_abs_yaw_change_deg,
    mode_uses_steering,
    onset_gap_m,
)
from test27___fixed_profile_rescore import (  # crash recovery, proven in the Week 5 overnight run
    MAX_ATTEMPTS_PER_RUN,
    MAX_CONSECUTIVE_FAILED_RUNS,
    _carla_port_open,
    _relaunch_carla,
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
# Reported beside the primary 0.3 m radius: CARLA's measured walker box (MASTER, Phase 2 decision 5).
ALT_RADIUS_PROTOCOL = dataclasses.replace(PROTOCOL, pedestrian_radius_m=0.188)
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
    p.add_argument("--passage-u", type=float, nargs="+", default=[1.0],
                   help="aim positions for brake_passage_edge, one run each (0 = lane path, 1 = right passage edge)")
    p.add_argument("--smoke", action="store_true", help=f"run only {SMOKE_CASES}")
    p.add_argument("--rescore", metavar="SWEEP_DIR",
                   help="recompute drivability + metrics from a sweep's saved traces (no simulation)")
    p.add_argument("--resume", metavar="SWEEP_DIR", help="continue an interrupted sweep (grid read from its manifest)")
    p.add_argument("--relaunch-carla", action="store_true",
                   help="if CARLA stops answering, restart it renderless (default physics, no --chrono) and retry")
    p.add_argument("--table", metavar="SWEEP_DIR", help="print the best-route table from a sweep's finished runs")
    args = p.parse_args(argv)
    if any(not 0.0 <= u <= 1.0 for u in args.passage_u):  # right swerves only for now
        p.error("--passage-u values must be in [0, 1]")
    return args


def _connect():
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    return client


def _load_rows(sweep_dir):
    rows = []
    for path in glob.glob(os.path.join(glob.escape(sweep_dir), "*", "*", "row.json")):
        with open(path) as f:
            rows.append(json.load(f))
    return sorted(rows, key=lambda r: (r["mph"], r["onset_ttc_s"], MODES.index(r["mode"]), r["passage_u"] or 0.0))


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


def _run_one(*, mph, ttc, mode, passage_u, label, swerve_offset_m, case_dir, client):
    cfg = ScenarioConfig(
        walker_cross="stationary", walker_side="left", trigger_ttc_s=ttc,
        target_mph=mph, encounter_distance_m=ENCOUNTER_DISTANCE_M,
        braking_ramp_up_per_s=BRAKE_RAMP_UP_PER_S, brake_profile="step_constant",
        weather_preset="ClearSunset", sun_altitude_deg=0,
        sim_seconds=ENCOUNTER_DISTANCE_M / (mph * 0.44704) + 30.0,
        run_id=f"{mph:g}mph_ttc{ttc:g}_{label}",
    )
    recorder = TrajectoryRecorder(dt_s=FIXED_DT)
    drivable_flags = []
    state = {"tick": 0, "map": None, "onset_marked": False, "seam_corrections": 0, "passage_frame": None}

    if mode == MODE_BRAKE_PASSAGE_EDGE:
        steering_fn = PassageEdgeOffset(u=passage_u, u_min=0.0, u_max=1.0)  # right-swerve-only
    elif mode_uses_steering(mode):
        steering_fn = build_evasive_offset_fn(peak_offset_m=swerve_offset_m, shift_duration_s=SWERVE_SHIFT_DURATION_S)
    else:
        steering_fn = None

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
        # Stationary pedestrian: the passage at its position never changes, so
        # computing it once from the first tick's position is exact and lets
        # the swerve start on the onset tick, like brake_steer's.
        if (isinstance(steering_fn, PassageEdgeOffset) and steering_fn.passage is None
                and telemetry.get("pedestrian_x_m") is not None):
            ped = (telemetry["pedestrian_x_m"], telemetry["pedestrian_y_m"])
            passage, frame = passage_at_location(state["map"], carla.Location(x=ped[0], y=ped[1], z=telemetry.get("pos_z_m") or 0.0), [ped])
            steering_fn.set_passage(passage)
            state["passage_frame"] = frame
    result = run_scenario(
        cfg, plot_after=False, lateral_offset_fn=steering_fn, tick_observer=observer,
        post_crossing_settle_s=PROTOCOL.horizon_s + 1.0,
        hazard_command_fn=make_hazard_command_fn(mode, brake_target=BRAKE_TARGET),
    )
    recorder.tag_closest_approach()
    metrics = compute_encounter_metrics(recorder.ticks, onset_time_s=result.trigger_time_s,
                                        protocol=PROTOCOL, footprint_drivable=drivable_flags)
    alt = compute_encounter_metrics(recorder.ticks, onset_time_s=result.trigger_time_s,
                                    protocol=ALT_RADIUS_PROTOCOL, footprint_drivable=drivable_flags)
    onset = result.trigger_time_s
    p = getattr(steering_fn, "passage", None)
    row = _summary_row(
        metrics, mph=mph, ttc=ttc, mode=mode, collision_sensor=result.collision_detected, trigger_time_s=onset,
        passage_u=passage_u, contact_r0188=alt.outcome == "contact", min_clearance_r0188_m=alt.min_clearance_m,
        passage_left_edge_m=p and p.left_edge_m, passage_right_edge_m=p and p.right_edge_m,
        target_offset_m=getattr(steering_fn, "last_target_m", None),
        missing_passage_ticks=getattr(steering_fn, "missing_passage_ticks", None),
        max_abs_yaw_change_deg=max_abs_yaw_change_deg(
            [t.yaw_deg for t in recorder.ticks
             if onset is not None and onset <= t.sim_time_s <= onset + PROTOCOL.horizon_s]))

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
            "steering": _steering_manifest(steering_fn, swerve_offset_m, state["passage_frame"]),
            "config": cfg.to_dict(),
            "drivability_check": "seam_tolerant",
            "drivability_seam_corrections": state["seam_corrections"],
            "physics_backend": BACKEND_DEFAULT,
        }, f, indent=2)
    with open(os.path.join(case_dir, "row.json"), "x") as f:  # written last: marks the run complete for --resume
        json.dump(row, f, indent=2)
    return row, metrics


def _steering_manifest(steering_fn, swerve_offset_m, passage_frame):
    if steering_fn is None:
        return None
    if isinstance(steering_fn, PassageEdgeOffset):
        p = steering_fn.passage
        return {"rule": "passage_edge (Izmirli): aim at passage coordinate u (1 = the right edge)", "u": steering_fn.u,
                "u_bounds": [steering_fn.u_min, steering_fn.u_max],
                "target_offset_m": steering_fn.last_target_m,
                "missing_passage_ticks": steering_fn.missing_passage_ticks,
                "passage": None if p is None else {
                    "road_left_m": p.road_left_m, "road_right_m": p.road_right_m,
                    "intervals": [list(i) for i in p.intervals],
                    "left_edge_m": p.left_edge_m, "right_edge_m": p.right_edge_m},
                "passage_frame": passage_frame}
    return {"peak_offset_m": swerve_offset_m, "side_sign": SWERVE_SIDE_SIGN,
            "margin_m": SWERVE_MARGIN_M, "shift_duration_s": SWERVE_SHIFT_DURATION_S,
            "recovered": steering_fn.recovered, "used_fallback_timeout": steering_fn.used_fallback_timeout}


def main(argv=None):
    args = _parse_args(argv)
    if args.table:
        print(format_best_route_table(_load_rows(args.table)))
        return 0
    cases = SMOKE_CASES if args.smoke else [(m, t) for m in args.mph for t in args.ttc]
    modes, passage_u = args.modes, args.passage_u

    client = _connect()
    client_version, server_version = client.get_client_version(), client.get_server_version()
    if client_version != server_version:
        print(f"[test26] client {client_version} != server {server_version}; refusing", file=sys.stderr)
        return 2

    if args.rescore:
        return _rescore(args.rescore, client)

    swerve_offset_m = compute_required_clearance_offset_m(  # already signed by side_sign
        pedestrian_lateral_m=0.0, side_sign=SWERVE_SIDE_SIGN, safety_margin_m=SWERVE_MARGIN_M)
    if args.resume:
        sweep_dir = args.resume
        with open(os.path.join(sweep_dir, "sweep_manifest.json")) as f:
            manifest = json.load(f)
        cases, modes, passage_u = manifest["cases"], manifest["modes"], manifest["passage_u"]
    else:
        sweep_dir = _start_sweep(cases, modes, passage_u, swerve_offset_m, client_version, server_version)
    runs = [(mph, ttc, *run) for mph, ttc in cases for run in expand_runs(modes, passage_u)]
    log_path = os.path.join(sweep_dir, "progress.log")

    def log(line):
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"[test26] {len(runs)} run(s); swerve offset {swerve_offset_m:+.3f} m | evidence: {sweep_dir}")
    rows, resumed, consecutive_failed, start = [], 0, 0, time.time()
    for i, (mph, ttc, mode, u, label) in enumerate(runs, start=1):
        case_dir = os.path.join(sweep_dir, f"{mph:g}mph_ttc{ttc:g}", label)
        tag = f"[{i:3d}/{len(runs)}] {mph:g} mph ttc {ttc:g} {label:24s}"
        if os.path.isfile(os.path.join(case_dir, "row.json")):  # finished before a --resume
            with open(os.path.join(case_dir, "row.json")) as f:
                rows.append(json.load(f))
            resumed += 1
            continue
        row = None
        for attempt in range(1, MAX_ATTEMPTS_PER_RUN + 1):
            if os.path.exists(case_dir):  # partial run from a crash: keep it, never overwrite
                os.rename(case_dir, f"{case_dir}_incomplete_{datetime.datetime.now():%H%M%S%f}")
            try:
                print(f"\n[test26] ===== {tag.strip()} =====")
                row, m = _run_one(mph=mph, ttc=ttc, mode=mode, passage_u=u, label=label,
                                  swerve_offset_m=swerve_offset_m, case_dir=case_dir, client=_connect())
                break
            except Exception as exc:  # one failed run must not lose the rest
                log(f"{tag} FAILED (attempt {attempt}): {type(exc).__name__}: {exc}")
                if args.relaunch_carla and attempt < MAX_ATTEMPTS_PER_RUN:
                    time.sleep(15)
                    if not _carla_port_open() or attempt >= 2:
                        _relaunch_carla(lambda line: log(line.replace("[test27]", "[test26]")))
        if row is None:
            consecutive_failed += 1
            if consecutive_failed >= MAX_CONSECUTIVE_FAILED_RUNS:
                log(f"[test26] STOPPING: {consecutive_failed} runs in a row failed; resume with --resume {sweep_dir}")
                break
            continue
        consecutive_failed = 0
        for line in summarize_for_console(m):
            print("    " + line)
        rows.append(row)
        with open(os.path.join(sweep_dir, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        elapsed_s = time.time() - start
        result_text = (f"contact at {row['contact_speed_mps']:.1f} m/s" if row["outcome"] == "contact"
                       else f"{row['outcome']}, clearance {row['min_clearance_m']:.2f} m")
        log(f"{tag} -> {result_text} | {elapsed_s / 60:5.1f} min, "
            f"{eta_text(i - resumed, len(runs) - resumed, elapsed_s)}")

    _print_summary(rows)
    table = format_best_route_table(rows)
    with open(os.path.join(sweep_dir, "best_route_table.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\n" + table)
    log(f"[test26] DONE {len(rows)}/{len(runs)} runs | evidence: {sweep_dir}")
    return 0


def _start_sweep(cases, modes, passage_u, swerve_offset_m, client_version, server_version):
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
            "cases": cases, "modes": modes, "passage_u": passage_u, "protocol": PROTOCOL.__dict__,
            "encounter_distance_m": ENCOUNTER_DISTANCE_M, "brake_target": BRAKE_TARGET,
            "brake_ramp_up_per_s": BRAKE_RAMP_UP_PER_S, "swerve_peak_offset_m": swerve_offset_m,
            "swerve_shift_duration_s": SWERVE_SHIFT_DURATION_S, "random_seed": None,
            "seed_note": "fully scripted and deterministic; no random sampling",
        }, f, indent=2)
    return sweep_dir


SUMMARY_COLUMNS = ["mph", "onset_ttc_s", "mode", "passage_u", "outcome", "safe_success", "window_complete",
                   "speed_at_onset_mps", "min_clearance_m", "contact_speed_mps", "first_contact_time_s",
                   "contact_r0188", "min_clearance_r0188_m", "time_to_stop_s",
                   "stopping_distance_m", "speed_when_passing_mps", "max_abs_route_lateral_m",
                   "final_route_lateral_m", "returned_to_route", "drivable_violation_ticks",
                   "peak_decel_normal_speed_mps2", "max_abs_jerk_normal_speed_mps3",
                   "max_abs_lateral_accel_mps2", "max_abs_yaw_change_deg",
                   "requested_side_reversals", "requested_offset_reversals",
                   "target_offset_m", "passage_left_edge_m", "passage_right_edge_m", "missing_passage_ticks",
                   "carla_collision_sensor", "trigger_time_s"]


def _summary_row(m, *, mph, ttc, mode, collision_sensor, trigger_time_s, **extra):
    d = m.to_dict()
    return {**{c: d.get(c) for c in SUMMARY_COLUMNS}, "mph": mph, "onset_ttc_s": ttc, "mode": mode,
            "carla_collision_sensor": collision_sensor, "trigger_time_s": trigger_time_s, **extra}


def _rescore(sweep_dir, client):
    """Recompute seam-tolerant drivability and metrics from saved traces.
    Writes drivability_v2.json / metrics_v2.json per run and
    rescored_summary.csv; never overwrites the original files."""
    carla_map = client.get_world().get_map()
    rows = []
    for case in sorted(os.listdir(sweep_dir)):
        if not os.path.isdir(os.path.join(sweep_dir, case)):
            continue
        for label in sorted(os.listdir(os.path.join(sweep_dir, case))):
            run_dir = os.path.join(sweep_dir, case, label)
            if not os.path.isfile(os.path.join(run_dir, "manifest.json")):  # written after the trace: a finished run
                continue
            ticks = read_trace_csv(os.path.join(run_dir, "trace.csv"))
            with open(os.path.join(run_dir, "result.json")) as f:
                result = json.load(f)
            with open(os.path.join(run_dir, "manifest.json")) as f:
                manifest = json.load(f)
            mode = manifest["mode"]
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
                                     trigger_time_s=result["trigger_time_s"],
                                     passage_u=(manifest["steering"] or {}).get("u")))
            print(f"[test26] rescored {case}/{label}: {m.outcome} safe={m.safe_success} "
                  f"violations={m.drivable_violation_ticks} seam_corrections={corrections}")
    rows.sort(key=lambda r: (r["mph"], r["onset_ttc_s"], MODES.index(r["mode"]), r["passage_u"] or 0.0))
    with open(os.path.join(sweep_dir, "rescored_summary.csv"), "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    _print_summary(rows)
    return 0


def _print_summary(rows):
    print("\n[test26] SUMMARY")
    print(f"{'mph':>4} {'ttc':>4} {'mode':18} {'u':>4} {'outcome':22} {'safe':5} {'v_onset':>7} "
          f"{'min_clr':>7} {'v_contact':>9} {'stop_m':>7} {'max_lat':>7} {'drv_viol':>8}")

    def f(v, spec):
        return "-" if v is None else format(v, spec)
    for r in rows:
        print(f"{r['mph']:4g} {r['onset_ttc_s']:4g} {r['mode']:18} {f(r.get('passage_u'), '4g'):>4} "
              f"{r['outcome']:22} {str(r['safe_success']):5} "
              f"{f(r['speed_at_onset_mps'], '7.2f')} {f(r['min_clearance_m'], '7.2f')} "
              f"{f(r['contact_speed_mps'], '9.2f')} {f(r['stopping_distance_m'], '7.2f')} "
              f"{f(r['max_abs_route_lateral_m'], '7.2f')} {f(r['drivable_violation_ticks'], '8')}")


if __name__ == "__main__":
    sys.exit(main())
