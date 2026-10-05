"""
fixed_rescore_protocol.py

Pure helpers for re-running last semester's fixed-profile matched sweep with
trace recording and scoring it with the common encounter protocol (Phase 1
follow-up, 2026-10-05). Used by test27___fixed_profile_rescore.py.

Why: the archived fixed-profile collision rates (19-23%) came from
test3.RunResult, which relies on CARLA's collision sensor plus a near-cross
proximity fallback that compares 3D distance with a 2D threshold -- the same
checks that hid 33 of SAC v3's 73 contacts. The sweep saved no traces, so it
cannot be re-scored offline; it has to be re-run.

Reproduction check: values RunResult records BEFORE a run ends (trigger time,
speed/distance at trigger, hazard engagement, time to stop) must match the
archive exactly if the re-run is the same simulation. The outcome label may
legitimately differ, because the re-run keeps simulating long enough to cover
the protocol's 8 s window (the archive stopped 3 s after the pedestrian
finished).

Pure: no CARLA imports. Offline-tested in tests/test_fixed_rescore_protocol.py.
"""

import json
import os
from typing import List, Optional

RUN_FAMILY = "fixed_profile_rescore"

# RunResult fields fixed before the run's end, compared for exact reproduction.
REPRODUCTION_FIELDS = (
    "hazard_triggered", "trigger_time_s", "ego_speed_at_trigger_mps", "ego_dist_at_trigger_m", "time_to_stop_s",
)
REPRODUCTION_TOLERANCE = 1e-6


def load_archived_runs(sweep_dir: str, profiles: Optional[List[str]] = None) -> List[dict]:
    """[{run_id, config, archived}] for every run_XXXX folder, in run_id order,
    optionally filtered to the given brake profiles."""
    runs = []
    for name in sorted(os.listdir(sweep_dir)):
        run_dir = os.path.join(sweep_dir, name)
        cfg_path = os.path.join(run_dir, "config.json")
        res_path = os.path.join(run_dir, "result.json")
        if not (name.startswith("run_") and os.path.isfile(cfg_path) and os.path.isfile(res_path)):
            continue
        with open(cfg_path) as f:
            config = json.load(f)
        with open(res_path) as f:
            archived = json.load(f)
        if profiles and config.get("brake_profile") not in profiles:
            continue
        runs.append({"run_id": name, "config": config, "archived": archived})
    return runs


def compare_to_archived(new_result: dict, archived: dict) -> dict:
    """{field: {archived, new, match}} for REPRODUCTION_FIELDS, plus
    all_match. Floats match within REPRODUCTION_TOLERANCE; None only matches None."""
    out, all_match = {}, True
    for field in REPRODUCTION_FIELDS:
        a, b = archived.get(field), new_result.get(field)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            match = abs(a - b) <= REPRODUCTION_TOLERANCE
        else:
            match = a == b
        out[field] = {"archived": a, "new": b, "match": match}
        all_match &= match
    out["all_match"] = all_match
    return out


def eta_text(done: int, total: int, elapsed_s: float) -> str:
    if done <= 0:
        return "eta --"
    remaining_s = elapsed_s / done * (total - done)
    h, rem = divmod(int(remaining_s), 3600)
    return f"eta {h}h{rem // 60:02d}m"


def progress_line(*, index: int, total: int, run_id: str, config: dict, outcome: str, archived_outcome: str,
                  reproduced: bool, elapsed_s: float) -> str:
    flag = "" if reproduced else "  [REPRODUCTION MISMATCH]"
    ttc = config.get("trigger_ttc_s")
    return (f"[{index:3d}/{total}] {run_id} {config.get('brake_profile', '?'):18s} {config.get('walker_cross', '?'):4s} "
            f"{config.get('target_mph', 0):4.0f} mph ttc={ttc}  ->  {outcome:22s} (archived: {archived_outcome})"
            f"  | {elapsed_s / 60:5.1f} min, {eta_text(index, total, elapsed_s)}{flag}")
