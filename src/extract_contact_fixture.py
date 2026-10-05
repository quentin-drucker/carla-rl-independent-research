"""extract_contact_fixture.py

Builds tests/fixtures/test26_contact_cases.json from the Week 4 test26 evidence
(git-ignored under src/runs/steer_brake_baseline/), so the RL collision-signal
tests can check the geometric contact signal against known outcomes on any
clone. Offline only; reads traces, never re-simulates.

For each run it stores the recorded outcome (metrics.json), CARLA's sensor flag
(result.json), whole-trace geometric-contact facts, and a small tick sample:
  - contact runs: the onset tick plus 5 ticks before to 3 ticks after first contact;
  - clear runs:   the 5 ticks either side of minimum clearance.

Usage (from src/):
    ..\\venv\\Scripts\\python.exe -X utf8 extract_contact_fixture.py
"""

import glob
import json
import os
import sys

from rl_collision_signal import geometric_clearance_m, is_geometric_contact
from trace_schema import read_trace_csv

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
SWEEP_GLOB = os.path.join(SRC_DIR, "runs", "steer_brake_baseline", "*_steer_brake_baseline_default")
OUT_PATH = os.path.join(SRC_DIR, "..", "tests", "fixtures", "test26_contact_cases.json")
TICK_FIELDS = ("tick_index", "sim_time_s", "pos_x_m", "pos_y_m", "yaw_deg", "speed_mps",
               "pedestrian_x_m", "pedestrian_y_m")


def _clearance(t):
    if t.pedestrian_x_m is None or t.pedestrian_y_m is None:
        return None
    return geometric_clearance_m(ego_x_m=t.pos_x_m, ego_y_m=t.pos_y_m, ego_yaw_deg=t.yaw_deg,
                                 pedestrian_x_m=t.pedestrian_x_m, pedestrian_y_m=t.pedestrian_y_m)


def _run_case(run_dir, sweep_name):
    ticks = read_trace_csv(os.path.join(run_dir, "trace.csv"))
    with open(os.path.join(run_dir, "metrics.json")) as f:
        metrics = json.load(f)
    with open(os.path.join(run_dir, "result.json")) as f:
        result = json.load(f)
    with open(os.path.join(run_dir, "manifest.json")) as f:
        manifest = json.load(f)
    onset = result["trigger_time_s"]
    clearances = [_clearance(t) for t in ticks]
    contact_idx = [i for i, c in enumerate(clearances) if is_geometric_contact(c)]
    measured = [(c, i) for i, c in enumerate(clearances) if c is not None]
    min_clearance, min_idx = min(measured)

    onset_idx = next(i for i, t in enumerate(ticks) if t.sim_time_s >= onset - 1e-6)
    if contact_idx:
        first = contact_idx[0]
        keep = sorted({onset_idx, *range(max(0, first - 5), min(len(ticks), first + 4))})
    else:
        keep = list(range(max(0, min_idx - 5), min(len(ticks), min_idx + 6)))

    return {
        "sweep": sweep_name,
        "case": os.path.relpath(run_dir, os.path.dirname(os.path.dirname(run_dir))).replace("\\", "/"),
        "mode": manifest["mode"], "target_mph": manifest["target_mph"], "onset_ttc_s": manifest["onset_ttc_s"],
        "onset_time_s": onset,
        "recorded_outcome": metrics["outcome"],
        "recorded_first_contact_time_s": metrics["first_contact_time_s"],
        "recorded_min_clearance_m": metrics["min_clearance_m"],
        "carla_collision_sensor": result["collision_detected"],
        "trace_ticks": len(ticks),
        "trace_geometric_contact_ticks": len(contact_idx),
        "trace_first_contact_time_s": ticks[contact_idx[0]].sim_time_s if contact_idx else None,
        "trace_min_clearance_m": min_clearance,
        "ticks": [[getattr(ticks[i], k) for k in TICK_FIELDS] for i in keep],
    }


def main():
    sweeps = sorted(glob.glob(SWEEP_GLOB))
    if not sweeps:
        print(f"no test26 sweeps under {SWEEP_GLOB}", file=sys.stderr)
        return 1
    cases = []
    for sweep in sweeps:
        for trace in sorted(glob.glob(os.path.join(sweep, "*", "*", "trace.csv"))):
            cases.append(_run_case(os.path.dirname(trace), os.path.basename(sweep)))
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"tick_fields": list(TICK_FIELDS), "source": "src/runs/steer_brake_baseline (test26, CARLA default physics)",
                   "generator": "src/extract_contact_fixture.py", "cases": cases}, f, separators=(",", ":"))
    n_contact = sum(c["recorded_outcome"] == "contact" for c in cases)
    print(f"wrote {len(cases)} cases ({n_contact} recorded contacts) -> {os.path.normpath(OUT_PATH)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
