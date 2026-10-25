"""run_circuit.py

City circuit (Week 8, 2026-10-25): the ego drives laps of a loop route in a
CARLA town; pedestrians appear at random points ahead of it, several per lap.
Each one is an encounter handled by the policy (or a scripted action) plus
merge-back (merge_back.py), after which the car keeps driving to the next.
Every encounter is scored with the same protocol as eval_policy_encounters.py
(encounter_metrics v4, 8 s window from the hazard's oracle onset).

Hazards: the walkers of steer_brake_baseline (right_edge / left_edge /
stationary), onset TTC 0.6-1.5 s, placed only on straight, non-junction
stretches (circuit.straight_flags) at least --min-gap-m apart. The passage is
limited to same-direction lanes (no swerving into oncoming traffic).

Usage (CARLA running; from src/ in the venv):
    python -u -X utf8 run_circuit.py --model runs/steering_train/<run>/final_model.zip --steer-commit per_tick --u-min -1
    python -u -X utf8 run_circuit.py --fixed-u 1 --steer-commit per_tick        # scripted: full brake + full right swerve
    add --watch to follow the car with the spectator camera (visible CARLA)

Evidence (git-ignored): src/runs/circuit/<timestamp>_circuit_default/
    manifest.json, hazards.csv, hz_NN/{trace.csv, metrics.json, hazard.json}
"""

import argparse
import csv
import datetime
import json
import math
import os
import random
import sys

import numpy as np

from circuit import curve_speed_limit_mps, encounter_done, place_hazards, sample_hazard, straight_flags
from encounter_metrics import compute_encounter_metrics
from eval_policy_encounters import PROTOCOL, _FixedAction, _git, _versions
from physics_backend import BACKEND_DEFAULT, create_run_dir
from rl_eval_protocol import file_sha256, resolve_model_path, trace_telemetry
from rl_steering import STEER_COMMIT_MODES

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_ROOT = os.path.join(SRC_DIR, "runs", "circuit")
HAZARD_COLUMNS = ["hazard", "lap", "route_m", "walker", "ttc_s", "onset_time_s", "speed_at_onset_mps", "outcome",
                  "outcome_v4", "contact_cause", "ego_safe_success", "safe_success", "contact_speed_mps",
                  "min_clearance_m", "max_abs_route_lateral_m", "drivable_violation_ticks", "merge_back_state",
                  "drive_on_count", "encounter_s", "end_lateral_m", "end_speed_mps"]


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="city circuit with random pedestrians (see docstring)")
    p.add_argument("--map", default="Town05_Opt")
    p.add_argument("--laps", type=int, default=2)
    p.add_argument("--hazards-per-lap", type=int, default=4)
    p.add_argument("--min-gap-m", type=float, default=300.0, help="minimum route distance between hazards")
    p.add_argument("--mph", type=float, default=35.0, help="cruise speed (the policy was trained at 30-50 mph)")
    p.add_argument("--seed", type=int, default=0, help="route-independent hazard plan seed")
    p.add_argument("--model", help="SAC model ZIP")
    p.add_argument("--fixed-u", type=float, help="no model: a constant [--fixed-brake, u] action")
    p.add_argument("--fixed-brake", type=float, default=1.0)
    p.add_argument("--steer-commit", choices=STEER_COMMIT_MODES, default="per_tick")
    p.add_argument("--lock-delay-s", type=float, default=0.5)
    p.add_argument("--u-min", type=float, default=-1.0, help="-1 = both sides (default), 0 = right only")
    p.add_argument("--watch", action="store_true", help="follow the ego with the spectator camera")
    args = p.parse_args(argv)
    if (args.model is None) == (args.fixed_u is None):
        p.error("give exactly one of --model and --fixed-u")
    return args


def loop_route(world, laps):
    """(ego spawn Transform, route waypoints): a loop through 4 far-apart spawn
    points, repeated `laps` times (city_map_probe.py's route)."""
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    points = world.get_map().get_spawn_points()
    stops = [points[0]]
    while len(stops) < 4:
        stops.append(max(points, key=lambda p: min(p.location.distance(s.location) for s in stops)))
    grp = GlobalRoutePlanner(world.get_map(), 2.0)
    lap = []
    for a, b in zip(stops, stops[1:] + stops[:1]):
        lap += [wp for wp, _ in grp.trace_route(a.location, b.location)]
    return stops[0], lap * laps


def plan_hazards(route_wps, args):
    """([(route_m, ScenarioConfig)] for the whole run, route distances, route yaws)."""
    from steer_brake_baseline import baseline_scenario_config

    s, total = [0.0], 0.0
    for a, b in zip(route_wps, route_wps[1:]):
        total += a.transform.location.distance(b.transform.location)
        s.append(total)
    # Headings from the waypoints' POSITIONS: the planner's lane changes jump ~7 m sideways within ~10 m while
    # each waypoint's own yaw still reads "straight" (the first city lap crashed into one at full speed).
    yaws = []
    for a, b in zip(route_wps, route_wps[1:]):
        dx, dy = b.transform.location.x - a.transform.location.x, b.transform.location.y - a.transform.location.y
        yaws.append(math.degrees(math.atan2(dy, dx)) if dx * dx + dy * dy > 1e-4 else (yaws[-1] if yaws else 0.0))
    yaws.append(yaws[-1])
    ok = straight_flags(yaws, [wp.is_junction for wp in route_wps])
    rng = random.Random(args.seed)
    positions = place_hazards(s, ok, n=args.laps * args.hazards_per_lap, min_gap_m=args.min_gap_m, start_m=200.0,
                              end_margin_m=150.0, rng=rng)
    plan = []
    for i, pos in enumerate(positions):
        h = sample_hazard(rng)
        cfg = baseline_scenario_config(args.mph, round(h["ttc_s"], 2), walker=h["walker"],
                                       run_id=f"hz{i:02d}_{h['walker']}_ttc{h['ttc_s']:.2f}")
        cfg.encounter_distance_m, cfg.sim_seconds = pos, 1e6  # the episode is the whole circuit
        plan.append((pos, cfg))
    return plan, s, yaws


def main(argv=None):
    args = _parse_args(argv)
    from carla_aeb_env import FIXED_DT, CarlaAEBEnv
    from map_drivability import footprint_drivable
    from steer_brake_baseline import baseline_scenario_config
    from trajectory_recording import TrajectoryRecorder

    env = CarlaAEBEnv(cfg=baseline_scenario_config(args.mph, 1.0), steer_commit=args.steer_commit,
                      lock_delay_s=args.lock_delay_s, merge_back=True, u_bounds=(args.u_min, 1.0),
                      map_name=args.map, same_direction_passage=True, trigger_by_line_distance=True)
    route = loop_route(env._world, args.laps)
    env._route_builder = lambda world: route
    plan, route_s, route_yaws = plan_hazards(route[1], args)
    route_len = route_s[-1]
    if not plan:
        raise SystemExit("no straight, non-junction stretch long enough for a hazard on this route")
    lap_len = route_len / args.laps

    if args.model:
        from stable_baselines3 import SAC
        model_path = resolve_model_path(args.model, search_dir=SRC_DIR)
        policy = SAC.load(model_path)
        if policy.observation_space.shape != env.observation_space.shape:
            raise SystemExit(f"model observation shape {policy.observation_space.shape} != env {env.observation_space.shape}")
    else:
        model_path, policy = f"fixed_action:brake={args.fixed_brake:g},u={args.fixed_u:g}", _FixedAction(args.fixed_brake, args.fixed_u)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = create_run_dir(runs_root=RUNS_ROOT, timestamp=stamp, run_family="circuit", backend=BACKEND_DEFAULT)
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    with open(os.path.join(out_dir, "manifest.json"), "x") as f:
        json.dump({"created_at_iso": datetime.datetime.now().isoformat(), "git_commit": _git("rev-parse", "HEAD"),
                   "git_dirty": bool(dirty), "git_tracked_changes": dirty, "argv": sys.argv, "args": vars(args),
                   "model_path": model_path, "model_sha256": file_sha256(model_path) if args.model else None,
                   "map": args.map, "route_m": route_len, "lap_m": lap_len, "physics_backend": BACKEND_DEFAULT,
                   "protocol": PROTOCOL.__dict__, "merge_back": True, "same_direction_passage": True,
                   "hazards": [{"route_m": pos, "config": cfg.to_dict()} for pos, cfg in plan],
                   "versions": _versions()}, f, indent=2)
    print(f"[circuit] {args.map}: {args.laps} lap(s) of {lap_len:.0f} m, {len(plan)} hazards | policy {os.path.basename(model_path)}")
    print(f"[circuit] evidence: {out_dir}")

    env.cfg = plan[0][1]
    rows, hz, recorder, drivable, onset_marked = [], 0, TrajectoryRecorder(dt_s=FIXED_DT), [], False
    spec = None
    try:
        obs, _ = env.reset()
        carla_map = env._world.get_map()
        if args.watch:
            from spectator import SpectatorController
            spec = SpectatorController(env._world, env._vehicle, debug_draw=False)
        max_ticks = int((1.6 * route_len / (args.mph * 0.44704) + 25.0 * len(plan) + 60.0) / FIXED_DT)
        stopped_ticks, end_reason, route_i = 0, "max_ticks", 0
        route_xy = [(wp.transform.location.x, wp.transform.location.y) for wp in route[1]]
        for _ in range(max_ticks):
            ego = env._vehicle.get_location()
            route_i = min(range(route_i, min(route_i + 60, len(route_xy))),
                          key=lambda k: (route_xy[k][0] - ego.x) ** 2 + (route_xy[k][1] - ego.y) ** 2)
            env._target_speed_mps = curve_speed_limit_mps(route_s, route_yaws, route_i, v_max_mps=args.mph * 0.44704)
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, _, _, info = env.step(np.asarray(action))
            if spec is not None:
                spec.tick(fixed_dt=FIXED_DT)
            marker = "onset" if info["triggered"] and not onset_marked else None
            onset_marked = onset_marked or info["triggered"]
            rec = recorder.record(tick_index=info["tick"], sim_time_s=info["sim_time_s"],
                                  telemetry=trace_telemetry(env._last_telemetry or None, info), event_marker=marker)
            if rec is not None:
                drivable.append(footprint_drivable(carla_map, rec.pos_x_m, rec.pos_y_m, rec.yaw_deg, rec.pos_z_m)[0])
            if info["triggered"] and encounter_done(info["sim_time_s"] - info["trigger_time_s"], info.get("merge_back_state")):
                rows.append(_finish(out_dir, hz, plan[hz], recorder, drivable, info, env, lap_len))
                _write_rows(out_dir, rows)
                hz += 1
                if hz == len(plan):
                    end_reason = "all hazards done"
                    break
                env.next_encounter(plan[hz][1])
                recorder, drivable, onset_marked = TrajectoryRecorder(dt_s=FIXED_DT), [], False
            stopped_ticks = stopped_ticks + 1 if info["ego_speed_mps"] < 0.3 and not info["triggered"] else 0
            if stopped_ticks * FIXED_DT > 60.0:
                end_reason = "stuck: stopped for 60 s outside an encounter"
                recorder.write(os.path.join(out_dir, "stuck_trace.csv"))
                with open(os.path.join(out_dir, "stuck_state.json"), "x") as f:
                    json.dump({k: v for k, v in env._speed_state.items() if isinstance(v, (int, float, str, bool))}
                              | {"telemetry": {k: v for k, v in (env._last_telemetry or {}).items()
                                               if isinstance(v, (int, float, str, bool))}}, f, indent=2)
                break
    finally:
        if spec is not None:
            spec.close()
        env.close()
    _print_summary(rows, end_reason, len(plan))
    print(f"[circuit] evidence: {out_dir}")
    return 0 if end_reason == "all hazards done" else 1


def _finish(out_dir, index, plan_item, recorder, drivable, info, env, lap_len):
    pos, cfg = plan_item
    m = compute_encounter_metrics(recorder.ticks, onset_time_s=info["trigger_time_s"], protocol=PROTOCOL,
                                  footprint_drivable=drivable)
    mb = env.merge_back_summary or {}
    hz_dir = os.path.join(out_dir, f"hz_{index:02d}")
    os.makedirs(hz_dir, exist_ok=False)
    recorder.write(os.path.join(hz_dir, "trace.csv"))
    with open(os.path.join(hz_dir, "metrics.json"), "x") as f:
        json.dump(m.to_dict(), f, indent=2)
    with open(os.path.join(hz_dir, "hazard.json"), "x") as f:
        json.dump({"route_m": pos, "config": cfg.to_dict(), "merge_back": mb,
                   "walker_transitions": env.walker_transitions}, f, indent=2)
    d = m.to_dict()
    row = {c: d.get(c) for c in HAZARD_COLUMNS if c in d}
    row.update(hazard=index, lap=int(pos // lap_len) + 1, route_m=round(pos, 1), walker=cfg.run_id.split("_", 1)[1].rsplit("_", 1)[0],
               ttc_s=cfg.trigger_ttc_s, merge_back_state=mb.get("state"), drive_on_count=mb.get("drive_on_count"),
               encounter_s=round(info["sim_time_s"] - info["trigger_time_s"], 2),
               end_lateral_m=round(info.get("ego_lateral_m") or 0.0, 2), end_speed_mps=round(info["ego_speed_mps"], 2))
    print(f"[circuit] hazard {index:2d} lap {row['lap']} at {pos:7.1f} m | {row['walker']:10s} ttc {cfg.trigger_ttc_s:.2f} | "
          f"{d['outcome_v4']:28s} clr {d['min_clearance_m'] if d['min_clearance_m'] is None else round(d['min_clearance_m'], 2)} "
          f"| merge-back {mb.get('state')} | {row['encounter_s']:.1f} s")
    return {c: row.get(c) for c in HAZARD_COLUMNS}


def _write_rows(out_dir, rows):
    with open(os.path.join(out_dir, "hazards.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HAZARD_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def _print_summary(rows, end_reason, n_planned):
    safe = sum(1 for r in rows if r["ego_safe_success"])
    back = sum(1 for r in rows if r["merge_back_state"] == "returned")
    print(f"\n[circuit] SUMMARY: {len(rows)}/{n_planned} hazards encountered ({end_reason}) | safe (v4) {safe} | "
          f"back in lane {back} | contacts by the car {sum(1 for r in rows if r['contact_cause'] == 'ego')}")


if __name__ == "__main__":
    sys.exit(main())
