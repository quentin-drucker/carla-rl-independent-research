"""
resume_sweep.py
Quentin | Junior Spring Research, 2026.

Resumes a partially-completed sweep by re-running only the rows whose
error column contains "CRASHED". Reads the existing sweep_summary.csv,
reruns the failed configs, and rewrites the CSV with the patched results.

Usage:
    python resume_sweep.py                        <- latest sweep folder
    python resume_sweep.py runs/20260417_202355   <- explicit folder
"""

import os
import sys
import csv
import glob
import traceback

CARLA_ROOT = r"C:\Users\qdruc\OneDrive\Desktop\CARLA_0.9.16"
_carla_pythonapi_root  = os.path.join(CARLA_ROOT, "PythonAPI")
_carla_pythonapi_carla = os.path.join(CARLA_ROOT, "PythonAPI", "carla")
if _carla_pythonapi_root not in sys.path:
    sys.path.append(_carla_pythonapi_root)
if _carla_pythonapi_carla not in sys.path:
    sys.path.append(_carla_pythonapi_carla)

from scenario_config import ScenarioConfig
from test3___ped_intrusion_scenario import run_scenario

# ---------------------------------------------------------------------------
# Locate sweep folder
# ---------------------------------------------------------------------------

def _latest_sweep_dir():
    run_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    dirs = sorted(glob.glob(os.path.join(run_root, "*")))
    for d in reversed(dirs):
        if os.path.exists(os.path.join(d, "sweep_summary.csv")):
            return d
    raise FileNotFoundError("No sweep folder with sweep_summary.csv found.")

sweep_dir = sys.argv[1] if len(sys.argv) > 1 else _latest_sweep_dir()
csv_path  = os.path.join(sweep_dir, "sweep_summary.csv")
print(f"\n[resume] Sweep folder : {os.path.abspath(sweep_dir)}")
print(f"[resume] CSV          : {csv_path}\n")

# ---------------------------------------------------------------------------
# Load all rows; identify crashed ones
# ---------------------------------------------------------------------------

with open(csv_path, newline="") as f:
    reader    = csv.DictReader(f)
    fieldnames = reader.fieldnames
    all_rows  = list(reader)

crashed = [r for r in all_rows if r.get("error", "").strip() == "CRASHED"]
print(f"[resume] Total rows   : {len(all_rows)}")
print(f"[resume] Crashed rows : {len(crashed)}")

if not crashed:
    print("[resume] Nothing to resume — no CRASHED rows found.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Rebuild ScenarioConfig from each crashed row and rerun
# ---------------------------------------------------------------------------

def _cfg_from_row(r):
    return ScenarioConfig(
        target_mph            = float(r["target_mph"]),
        encounter_distance_m  = float(r["encounter_distance_m"]),
        walker_speed_mps      = float(r["walker_speed_mps"]),
        walker_side           = r["walker_side"],
        walker_cross          = r["walker_cross"],
        walker_startup_s      = float(r.get("walker_startup_s", 0.5)),
        trigger_ttc_s         = float(r["trigger_ttc_s"]) if r.get("trigger_ttc_s") else None,
        trigger_delay_s       = float(r.get("trigger_delay_s", 0.0)),
        braking_ramp_up_per_s = float(r.get("braking_ramp_up_per_s", 4.0)),
        brake_headway_s       = float(r.get("brake_headway_s", 2.5)),
        brake_profile         = r["brake_profile"],
        weather_preset        = r.get("weather_preset", "ClearSunset"),
        sim_seconds           = float(r.get("sim_seconds", 30.0)),
    )

# Build a lookup: run_id -> row index in all_rows
row_index = {r["run_id"]: i for i, r in enumerate(all_rows)}

fixed = 0
still_failed = 0

for ci, row in enumerate(crashed):
    run_id = row["run_id"]
    print(f"\n[resume] Re-running {run_id} ({ci+1}/{len(crashed)})  "
          f"profile={row['brake_profile']}  "
          f"target={row['target_mph']}mph  ttc={row.get('trigger_ttc_s','')}s")

    cfg = _cfg_from_row(row)
    cfg.run_id = run_id

    # Reuse existing run directory
    run_dir = os.path.join(sweep_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    cfg.to_json(os.path.join(run_dir, "config.json"))

    try:
        result = run_scenario(cfg, plot_after=False)
        result.to_json(os.path.join(run_dir, "result.json"))

        # Patch the row in all_rows
        idx = row_index[run_id]
        all_rows[idx].update({
            "hazard_triggered":         result.hazard_triggered,
            "collision_detected":       result.collision_detected,
            "outcome":                  result.outcome,
            "min_ped_distance_m":       f"{result.min_ped_distance_m:.3f}",
            "min_ttc_s":                f"{result.min_ttc_s:.3f}" if result.min_ttc_s >= 0 else "",
            "max_jerk_mps3":            f"{result.max_jerk_mps3:.2f}",
            "mean_jerk_mps3":           f"{result.mean_jerk_mps3:.2f}",
            "ego_speed_at_trigger_mph": f"{result.ego_speed_at_trigger_mph:.2f}",
            "ego_speed_at_trigger_mps": f"{result.ego_speed_at_trigger_mps:.3f}",
            "ego_dist_at_trigger_m":    f"{result.ego_dist_at_trigger_m:.1f}",
            "trigger_time_s":           f"{result.trigger_time_s:.2f}" if result.trigger_time_s is not None else "",
            "time_to_stop_s":           f"{result.time_to_stop_s:.2f}" if result.time_to_stop_s is not None else "",
            "error":                    "",
        })
        print(f"[resume] {run_id} -> {result.outcome}  "
              f"collision={result.collision_detected}  "
              f"min_ped={result.min_ped_distance_m:.2f}m")
        fixed += 1

    except Exception:
        print(f"[resume] ERROR in {run_id} -- still failing, keeping as CRASHED.")
        traceback.print_exc()
        still_failed += 1

# ---------------------------------------------------------------------------
# Rewrite CSV with patched rows
# ---------------------------------------------------------------------------

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(all_rows)

print(f"\n[resume] Done. Fixed: {fixed}  Still failed: {still_failed}")
print(f"[resume] Updated CSV -> {csv_path}")
