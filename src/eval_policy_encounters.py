"""eval_policy_encounters.py

Evaluate a trained RL policy with the common encounter protocol (Phase 1,
2026-10-04): the same onset + fixed 8 s window and the same encounter_metrics
scoring (oriented-footprint contact/clearance, onset-referenced stopping) as
the test26 scripted baselines. Replaces eval_sac.py / eval_sac_on_sweep.py for
any reported number; see rl_eval_protocol.py for why their labels are not
trusted.

Every episode records a TraceTick trace and a per-step log (full action
vector, reward, env termination flags, every collision component), and runs
past the env's own termination until onset + horizon. The legacy label that
eval_sac_on_sweep.py would have produced is kept beside the protocol outcome,
for auditing only.

Suites:
    sweep   the matched configs from a fixed-profile sweep_summary.csv
            (same parsing as eval_sac_on_sweep.py; --sweep-csv required)
    random  configs from train_sac.sample_config, sampled up front with --seed

Usage (CARLA running with default physics; from src/ in the venv):
    python -X utf8 eval_policy_encounters.py --model sac_v3_1600k --suite sweep \
        --sweep-csv runs/20260417_202355/sweep_summary.csv --limit 5
    python -X utf8 eval_policy_encounters.py --model sac_v3_1600k --suite random --n 100 --seed 42

Evidence (git-ignored, never overwritten):
    src/runs/rl_encounter_eval/<timestamp>_rl_encounter_eval_default/
        eval_manifest.json, summary.csv, ep_NNNN/{trace.csv, steps.csv,
        metrics.json, episode.json}
"""

import argparse
import csv
import datetime
import json
import os
import platform
import random
import subprocess
import sys

import numpy as np

from encounter_metrics import EncounterProtocol, compute_encounter_metrics, summarize_for_console
from physics_backend import BACKEND_DEFAULT, create_run_dir
from rl_collision_signal import COLLISION_SIGNAL_GEOMETRIC, COLLISION_SIGNAL_MODES
from rl_eval_protocol import (
    EVAL_PROTOCOL_VERSION,
    RUN_FAMILY,
    SUMMARY_COLUMNS,
    LegacyLabelTracker,
    check_manifest,
    extended_sim_seconds,
    file_sha256,
    legacy_truncated,
    load_sweep_configs,
    resolve_model_path,
    summary_row,
    trace_telemetry,
    window_decision,
)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_ROOT = os.path.join(SRC_DIR, "runs", RUN_FAMILY)
PROTOCOL = EncounterProtocol(horizon_s=8.0)  # identical to test26
STEP_COLUMNS = ["tick", "sim_time_s", "action", "reward", "terminated", "truncated", "drive_mode",
                "triggered", "collision", "collision_sensor", "collision_legacy_proximity",
                "collision_geometric_contact", "collision_clearance_m", "full_stop_achieved", "ped_crossed"]


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10, cwd=SRC_DIR).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _versions():
    out = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__}
    for name in ("stable_baselines3", "torch", "gymnasium"):
        try:
            out[name] = __import__(name).__version__
        except Exception as exc:  # recorded, never fatal
            out[name] = f"unavailable: {exc}"
    return out


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="encounter-protocol evaluation of an RL policy (see docstring)")
    p.add_argument("--model", required=True, help="model ZIP (no default: stale defaults caused confusion before)")
    p.add_argument("--suite", choices=("sweep", "random"), required=True)
    p.add_argument("--sweep-csv", help="sweep_summary.csv for --suite sweep")
    p.add_argument("--n", type=int, default=100, help="episodes for --suite random")
    p.add_argument("--seed", type=int, default=42, help="config-sampling seed for --suite random")
    p.add_argument("--limit", type=int, help="evaluate only the first N configs (smoke runs)")
    p.add_argument("--collision-signal", choices=COLLISION_SIGNAL_MODES, default=COLLISION_SIGNAL_GEOMETRIC,
                   help="env signal for reward/termination; scoring never depends on it")
    p.add_argument("--no-drivability", action="store_true", help="skip per-tick footprint drivability")
    p.add_argument("--watch", action="store_true", help="follow the ego with the spectator camera")
    return p.parse_args(argv)


def _build_configs(args):
    if args.suite == "sweep":
        if not args.sweep_csv:
            raise SystemExit("--suite sweep needs --sweep-csv")
        configs = load_sweep_configs(args.sweep_csv)
        source = {"sweep_csv": os.path.abspath(args.sweep_csv), "sweep_csv_sha256": file_sha256(args.sweep_csv)}
    else:
        from train_sac import sample_config  # imports the env module; deferred
        random.seed(args.seed)
        np.random.seed(args.seed)
        configs = [sample_config() for _ in range(args.n)]
        source = {"sampler": "train_sac.sample_config", "seed": args.seed}
    if args.limit is not None:
        configs = configs[: args.limit]
    return configs, source


def _footprint_drivable(carla_map, x, y, yaw_deg, z):
    """True if all four footprint corners are on a Driving lane, False if any
    is not, None if any query failed (same rule as test26)."""
    import carla
    from map_drivability import classify_point_drivability_seam_tolerant
    from steer_brake_baseline import footprint_corners_xy

    statuses = [classify_point_drivability_seam_tolerant(carla_map, carla.Location(x=cx, y=cy, z=z))[0]
                for cx, cy in footprint_corners_xy(x, y, yaw_deg)]
    if "non_drivable" in statuses:
        return False
    return None if "unknown" in statuses else True


def _run_episode(env, model, cfg, *, episode, ep_dir, carla_map, watch):
    from avoidability import compute_avoidability
    from carla_aeb_env import FIXED_DT, POST_TRIGGER_TIMEOUT_S
    from trajectory_recording import TrajectoryRecorder

    original_sim_seconds = cfg.sim_seconds
    sim_seconds_used = extended_sim_seconds(original_sim_seconds, PROTOCOL.horizon_s, POST_TRIGGER_TIMEOUT_S)
    cfg.sim_seconds = sim_seconds_used
    env.cfg = cfg
    obs, _ = env.reset()

    spec = None
    if watch:
        from spectator import SpectatorController
        spec = SpectatorController(env._world, env._vehicle, debug_draw=False)

    recorder = TrajectoryRecorder(dt_s=FIXED_DT)
    legacy = LegacyLabelTracker()
    drivable, steps = [], []
    counts = {"sensor": 0, "legacy_proximity": 0, "geometric_contact": 0}
    first_env_termination = None
    onset_marked = False

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        tick = info["tick"]
        if spec is not None:
            spec.tick(fixed_dt=FIXED_DT)

        marker = None
        if info["triggered"] and not onset_marked:
            marker, onset_marked = "onset", True
        tel = trace_telemetry(env._last_telemetry or None, info)
        recorded = recorder.record(tick_index=tick, sim_time_s=info["sim_time_s"], telemetry=tel, event_marker=marker)
        if recorded is not None and carla_map is not None:
            drivable.append(_footprint_drivable(carla_map, recorded.pos_x_m, recorded.pos_y_m,
                                                recorded.yaw_deg, recorded.pos_z_m))
        elif recorded is not None:
            drivable.append(None)

        legacy.update(tick=tick, info=info, truncated=legacy_truncated(
            tick=tick, original_sim_seconds=original_sim_seconds, fixed_dt_s=FIXED_DT,
            post_trigger_timeout=info.get("post_trigger_timeout")))
        for key in counts:
            counts[key] += int(bool(info.get(f"collision_{key}")))
        if terminated and first_env_termination is None:
            first_env_termination = {"tick": tick, "sim_time_s": info["sim_time_s"], "collision": info["collision"],
                                     "full_stop_achieved": info["full_stop_achieved"],
                                     "ped_crossed": info["ped_crossed"]}
        steps.append({**{k: info.get(k) for k in STEP_COLUMNS}, "tick": tick, "reward": float(reward),
                      "terminated": terminated, "truncated": truncated,
                      "action": json.dumps(np.asarray(action, dtype=float).ravel().tolist())})

        decision = window_decision(sim_time_s=info["sim_time_s"], trigger_time_s=info["trigger_time_s"],
                                   horizon_s=PROTOCOL.horizon_s, truncated=truncated, fixed_dt_s=FIXED_DT,
                                   legacy_ended=legacy.ended)
        if decision.stop:
            break

    if spec is not None:
        spec.close()
    recorder.tag_closest_approach()
    onset = info["trigger_time_s"]
    metrics = compute_encounter_metrics(recorder.ticks, onset_time_s=onset, protocol=PROTOCOL,
                                        footprint_drivable=drivable if carla_map is not None else None)
    cfg.sim_seconds = original_sim_seconds
    avoid = compute_avoidability(cfg)

    os.makedirs(ep_dir, exist_ok=False)
    recorder.write(os.path.join(ep_dir, "trace.csv"))
    with open(os.path.join(ep_dir, "steps.csv"), "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STEP_COLUMNS)
        w.writeheader()
        w.writerows(steps)
    with open(os.path.join(ep_dir, "metrics.json"), "x") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    with open(os.path.join(ep_dir, "episode.json"), "x") as f:
        json.dump({
            "episode": episode, "config": cfg.to_dict(),
            "sim_seconds_original": original_sim_seconds,
            "sim_seconds_used": sim_seconds_used,
            "onset_time_s": onset, "stop_reason": decision.reason,
            "first_env_termination": first_env_termination,
            "legacy_label": legacy.label, "legacy_end_reason": legacy.end_reason, "legacy_end_tick": legacy.end_tick,
            "signal_tick_counts": counts, "drivability_check": None if carla_map is None else "seam_tolerant",
            "avoidability": {k: (v if v != float("inf") else "inf") for k, v in avoid.items()},
        }, f, indent=2)

    row = summary_row(episode=episode, cfg=cfg, metrics=metrics.to_dict(), stop_reason=decision.reason,
                      legacy_label=legacy.label, legacy_end_reason=legacy.end_reason, signal_counts=counts,
                      physics_label=avoid.get("label"))
    return metrics, row


def main(argv=None):
    args = _parse_args(argv)
    model_path = resolve_model_path(args.model, search_dir=SRC_DIR)
    configs, source = _build_configs(args)
    if not configs:
        raise SystemExit("no configs to evaluate")

    from stable_baselines3 import SAC
    from carla_aeb_env import CarlaAEBEnv

    model = SAC.load(model_path)
    env = CarlaAEBEnv(cfg=configs[0], collision_signal=args.collision_signal)
    try:
        if model.observation_space.shape != env.observation_space.shape:
            raise SystemExit(f"model observation shape {model.observation_space.shape} != env "
                             f"{env.observation_space.shape}; this model predates the current env")
        client_v, server_v = env._client.get_client_version(), env._client.get_server_version()
        if client_v != server_v:
            raise SystemExit(f"CARLA client {client_v} != server {server_v}; refusing")
        carla_map = None if args.no_drivability else env._world.get_map()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = create_run_dir(runs_root=RUNS_ROOT, timestamp=timestamp, run_family=RUN_FAMILY,
                                 backend=BACKEND_DEFAULT)
        dirty = _git("status", "--porcelain", "--untracked-files=no")
        manifest = check_manifest({
            "created_at_iso": datetime.datetime.now().isoformat(),
            "eval_protocol_version": EVAL_PROTOCOL_VERSION,
            "git_commit": _git("rev-parse", "HEAD"), "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_tracked_changes": dirty, "git_dirty": bool(dirty),
            "model_path": model_path, "model_sha256": file_sha256(model_path),
            "model_size_bytes": os.path.getsize(model_path),
            "model_observation_shape": list(model.observation_space.shape),
            "model_action_shape": list(model.action_space.shape),
            "physics_backend": BACKEND_DEFAULT,
            "physics_backend_note": "CARLA default vehicle physics; Chrono never enabled (Week 4 decision).",
            "carla_client_version": client_v, "carla_server_version": server_v,
            "collision_signal": args.collision_signal,
            "collision_signal_note": "env reward/termination only; outcomes come from encounter_metrics",
            "protocol": PROTOCOL.__dict__, "suite": args.suite, "suite_source": source,
            "seed": args.seed if args.suite == "random" else None,
            "n_episodes": len(configs), "limit": args.limit, "deterministic_policy": True,
            "drivability": not args.no_drivability, "versions": _versions(), "argv": sys.argv,
        })
        with open(os.path.join(out_dir, "eval_manifest.json"), "x") as f:
            json.dump(manifest, f, indent=2)
        print(f"[eval] {len(configs)} episode(s) | model {os.path.basename(model_path)} "
              f"sha256 {manifest['model_sha256'][:12]} | signal {args.collision_signal}")
        print(f"[eval] evidence: {out_dir}")

        rows = []
        for i, cfg in enumerate(configs, start=1):
            print(f"\n[eval] ===== episode {i}/{len(configs)}: {cfg.walker_cross} {cfg.walker_side} "
                  f"{cfg.target_mph:.1f} mph ttc={cfg.trigger_ttc_s} =====")
            metrics, row = _run_episode(env, model, cfg, episode=i, ep_dir=os.path.join(out_dir, f"ep_{i:04d}"),
                                        carla_map=carla_map, watch=args.watch)
            for line in summarize_for_console(metrics):
                print("    " + line)
            print(f"    legacy label (audit only): {row['legacy_label']} ({row['legacy_end_reason']})")
            rows.append(row)
            with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
                w.writeheader()
                w.writerows(rows)
    finally:
        env.close()

    _print_summary(rows)
    print(f"\n[eval] evidence: {out_dir}")
    return 0


def _print_summary(rows):
    from collections import Counter

    n = len(rows)
    outcomes = Counter(r["outcome"] for r in rows)
    legacy = Counter(r["legacy_label"] for r in rows)
    sensor_missed = sum(1 for r in rows if r["outcome"] == "contact" and not r["sensor_collision_ticks"])
    print(f"\n[eval] SUMMARY ({n} episodes, protocol outcome)")
    for k, v in sorted(outcomes.items()):
        print(f"    {k:24s} {v}")
    print(f"    safe_success             {sum(1 for r in rows if r['safe_success'])}")
    print(f"    contacts the sensor missed: {sensor_missed}/{outcomes.get('contact', 0)}")
    print("[eval] legacy labels (audit only): " + ", ".join(f"{k}={v}" for k, v in sorted(legacy.items(), key=str)))


if __name__ == "__main__":
    sys.exit(main())
