"""test27___fixed_profile_rescore.py

Re-run last semester's final fixed-profile matched sweep (src/runs/20260417_202355:
4 profiles x 200 configs) with trace recording, and score every run with the
common encounter protocol -- the same onset, 8 s window and encounter_metrics
used for the test26 baselines and the SAC v3 re-audit. Only then can fixed
profiles and SAC be compared. See fixed_rescore_protocol.py for why.

Each run uses its archived config.json unchanged and run_scenario exactly as
sweep.py did, except that the post-crossing settle is extended (3 s -> 9 s)
and the sim_seconds ceiling raised so the 8 s window is always covered. Both
only move the run's end; nothing before it changes. A reproduction check
compares pre-end RunResult values with the archive for every run.

CARLA default physics; renderless is fine (-RenderOffScreen): LiDAR is
ray-cast and does not need a window.

Usage (from src/ in the venv; CARLA running):
    python -u -X utf8 test27___fixed_profile_rescore.py                # all 800 runs
    python -u -X utf8 test27___fixed_profile_rescore.py --limit 4      # smoke
    python -u -X utf8 test27___fixed_profile_rescore.py --resume <out_dir>   # continue after a crash
    add --relaunch-carla to restart a crashed CARLA server automatically (renderless)

Evidence (git-ignored, never overwritten; --resume only adds missing runs):
    src/runs/fixed_profile_rescore/<timestamp>_fixed_profile_rescore_default/
        manifest.json, progress.log, summary.csv,
        run_XXXX/{trace.csv, metrics.json, result.json, run.json, console.log}
"""

import argparse
import contextlib
import csv
import datetime
import json
import os
import platform
import subprocess
import sys
import socket
import time

from encounter_metrics import EncounterProtocol, compute_encounter_metrics
from fixed_rescore_protocol import RUN_FAMILY, compare_to_archived, load_archived_runs, progress_line
from physics_backend import BACKEND_DEFAULT, create_run_dir
from rl_eval_protocol import file_sha256

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_ROOT = os.path.join(SRC_DIR, "runs", RUN_FAMILY)
DEFAULT_SWEEP = os.path.join(SRC_DIR, "runs", "20260417_202355")
PROTOCOL = EncounterProtocol(horizon_s=8.0)  # identical to test26 and the SAC re-audit
SETTLE_S = PROTOCOL.horizon_s + 1.0           # archive used 3.0
CARLA_EXE = r"C:\Users\qdruc\OneDrive\Desktop\CARLA_0.9.16\CarlaUE4.exe"
MAX_ATTEMPTS_PER_RUN = 3
MAX_CONSECUTIVE_FAILED_RUNS = 5
PROFILES = ("proportional_ramp", "step_constant", "cautious_ramp", "exponential")
SUMMARY_COLUMNS = [
    "run_id", "brake_profile", "walker_cross", "walker_side", "target_mph", "trigger_ttc_s", "walker_speed_mps",
    "outcome", "safe_success", "window_complete", "speed_at_onset_mps", "min_clearance_m", "contact_speed_mps",
    "first_contact_time_s", "time_to_stop_s", "stopping_distance_m", "peak_decel_normal_speed_mps2",
    "max_abs_jerk_normal_speed_mps3", "onset_time_s", "archived_outcome", "archived_collision", "rerun_outcome",
    "rerun_collision_sensor_or_fallback", "reproduced",
]


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10, cwd=SRC_DIR).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="fixed-profile matched-sweep re-score (see docstring)")
    p.add_argument("--sweep-dir", default=DEFAULT_SWEEP)
    p.add_argument("--profiles", nargs="+", default=list(PROFILES))
    p.add_argument("--limit", type=int, help="only the first N runs (smoke)")
    p.add_argument("--resume", metavar="OUT_DIR", help="continue an interrupted re-score in OUT_DIR")
    p.add_argument("--relaunch-carla", action="store_true",
                   help="if CARLA stops answering, restart it renderless (default physics, no --chrono) and retry")
    return p.parse_args(argv)


def _run_one(entry, run_dir):
    from scenario_config import ScenarioConfig
    from test3___ped_intrusion_scenario import FIXED_DT, run_scenario
    from trajectory_recording import TrajectoryRecorder

    cfg = ScenarioConfig.from_dict(entry["config"])
    original_sim_seconds = cfg.sim_seconds
    cfg.sim_seconds = original_sim_seconds + SETTLE_S  # ceiling only
    recorder = TrajectoryRecorder(dt_s=FIXED_DT)
    state = {"tick": 0, "onset": False}

    def observer(sim_time_s, triggered, telemetry):
        index = state["tick"]
        state["tick"] += 1
        if telemetry is None:
            return
        marker = None
        if triggered and not state["onset"]:
            marker, state["onset"] = "onset", True
        recorder.record(tick_index=index, sim_time_s=sim_time_s, telemetry=telemetry, event_marker=marker)

    os.makedirs(run_dir, exist_ok=False)
    with open(os.path.join(run_dir, "console.log"), "x", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        result = run_scenario(cfg, plot_after=False, tick_observer=observer, post_crossing_settle_s=SETTLE_S)
    recorder.tag_closest_approach()
    metrics = compute_encounter_metrics(recorder.ticks, onset_time_s=result.trigger_time_s, protocol=PROTOCOL)
    result_dict = result.to_dict()
    reproduction = compare_to_archived(result_dict, entry["archived"])

    recorder.write(os.path.join(run_dir, "trace.csv"))
    with open(os.path.join(run_dir, "metrics.json"), "x") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    with open(os.path.join(run_dir, "result.json"), "x") as f:
        json.dump(result_dict, f, indent=2)
    with open(os.path.join(run_dir, "run.json"), "x") as f:
        json.dump({"run_id": entry["run_id"], "config": entry["config"], "sim_seconds_original": original_sim_seconds,
                   "sim_seconds_used": cfg.sim_seconds, "post_crossing_settle_s": SETTLE_S,
                   "archived_result": entry["archived"], "reproduction": reproduction,
                   "physics_backend": BACKEND_DEFAULT}, f, indent=2)

    m = metrics.to_dict()
    row = {c: m.get(c) for c in SUMMARY_COLUMNS if c in m}
    row.update(run_id=entry["run_id"], brake_profile=cfg.brake_profile, walker_cross=cfg.walker_cross,
               walker_side=cfg.walker_side, target_mph=cfg.target_mph, trigger_ttc_s=cfg.trigger_ttc_s,
               walker_speed_mps=cfg.walker_speed_mps, archived_outcome=entry["archived"].get("outcome"),
               archived_collision=entry["archived"].get("collision_detected"), rerun_outcome=result.outcome,
               rerun_collision_sensor_or_fallback=result.collision_detected, reproduced=reproduction["all_match"])
    return {c: row.get(c) for c in SUMMARY_COLUMNS}


def _carla_port_open(timeout_s=2.0):
    try:
        with socket.create_connection(("localhost", 2000), timeout=timeout_s):
            return True
    except OSError:
        return False


def _relaunch_carla(log):
    """Kill any CARLA processes and start a fresh renderless server with
    default physics. Returns True once the RPC port answers."""
    subprocess.run(["taskkill", "/F", "/IM", "CarlaUE4-Win64-Shipping.exe"], capture_output=True)
    subprocess.run(["taskkill", "/F", "/IM", "CarlaUE4.exe"], capture_output=True)
    time.sleep(10)
    subprocess.Popen([CARLA_EXE, "-RenderOffScreen"],
                     creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    for _ in range(90):
        time.sleep(2)
        if _carla_port_open():
            time.sleep(20)  # let the server finish loading before the next load_world
            log("[test27] CARLA relaunched (renderless, default physics)")
            return True
    log("[test27] CARLA relaunch FAILED: port 2000 never opened")
    return False


def _read_summary(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main(argv=None):
    args = _parse_args(argv)
    entries = load_archived_runs(args.sweep_dir, args.profiles)
    if args.limit is not None:
        entries = entries[: args.limit]
    if not entries:
        raise SystemExit("no archived runs found")

    if args.resume:
        out_dir = args.resume
        if not os.path.isfile(os.path.join(out_dir, "manifest.json")):
            raise SystemExit(f"--resume: no manifest.json in {out_dir}")
    else:
        import carla  # noqa: F401  (fail fast if the PythonAPI is missing)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = create_run_dir(runs_root=RUNS_ROOT, timestamp=timestamp, run_family=RUN_FAMILY,
                                 backend=BACKEND_DEFAULT)
        dirty = _git("status", "--porcelain", "--untracked-files=no")
        with open(os.path.join(out_dir, "manifest.json"), "x") as f:
            json.dump({
                "created_at_iso": datetime.datetime.now().isoformat(),
                "git_commit": _git("rev-parse", "HEAD"), "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                "git_tracked_changes": dirty, "git_dirty": bool(dirty),
                "physics_backend": BACKEND_DEFAULT,
                "physics_backend_note": "CARLA default vehicle physics; server launched without --chrono.",
                "sweep_dir": os.path.abspath(args.sweep_dir),
                "sweep_summary_sha256": file_sha256(os.path.join(args.sweep_dir, "sweep_summary.csv")),
                "profiles": args.profiles, "n_runs": len(entries), "limit": args.limit,
                "protocol": PROTOCOL.__dict__, "post_crossing_settle_s": SETTLE_S, "archived_settle_s": 3.0,
                "python_version": sys.version, "platform": platform.platform(), "random_seed": None,
                "seed_note": "scripted and deterministic; configs reused unchanged from the archive",
            }, f, indent=2)

    summary_path = os.path.join(out_dir, "summary.csv")
    rows = _read_summary(summary_path)
    done_ids = {r["run_id"] for r in rows}
    log_path = os.path.join(out_dir, "progress.log")
    start = time.time()

    def log(line):
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"[test27] {len(entries)} runs ({len(done_ids)} already done) | evidence: {out_dir}")
    consecutive_failed = 0
    for i, entry in enumerate(entries, start=1):
        if entry["run_id"] in done_ids:
            continue
        row = None
        for attempt in range(1, MAX_ATTEMPTS_PER_RUN + 1):
            run_dir = os.path.join(out_dir, entry["run_id"])
            if os.path.exists(run_dir):  # partial run from a crash: keep it, never overwrite
                os.rename(run_dir, f"{run_dir}_incomplete_{datetime.datetime.now():%H%M%S%f}")
            try:
                row = _run_one(entry, run_dir)
                break
            except Exception as exc:  # one failed run must not lose the rest
                log(f"[{i:3d}/{len(entries)}] {entry['run_id']} FAILED (attempt {attempt}): "
                    f"{type(exc).__name__}: {exc}")
                if args.relaunch_carla and attempt < MAX_ATTEMPTS_PER_RUN:
                    time.sleep(15)
                    if not _carla_port_open() or attempt >= 2:
                        _relaunch_carla(log)
        if row is None:
            consecutive_failed += 1
            if consecutive_failed >= MAX_CONSECUTIVE_FAILED_RUNS:
                log(f"[test27] STOPPING: {consecutive_failed} runs in a row failed; "
                    f"resume later with --resume {out_dir}")
                break
            continue
        consecutive_failed = 0
        rows.append(row)
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        log(progress_line(index=i, total=len(entries), run_id=entry["run_id"], config=entry["config"],
                          outcome=row["outcome"], archived_outcome=row["archived_outcome"],
                          reproduced=row["reproduced"], elapsed_s=time.time() - start))

    n_contact = sum(r["outcome"] == "contact" for r in rows)
    n_repro = sum(str(r["reproduced"]) == "True" for r in rows)
    log(f"[test27] DONE {len(rows)}/{len(entries)} runs | protocol contacts {n_contact} | "
        f"reproduced pre-end values {n_repro}/{len(rows)} | evidence: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
