"""
eval_sac_on_sweep.py
Quentin | Junior Spring Research, 2026.

Runs the trained SAC model on the SAME scenario configs used in the
hand-designed braking profile sweep. This creates a matched dataset
where SAC and all four profiles faced identical scenario conditions,
enabling a rigorous head-to-head comparison.

How it works:
  1. Load the unique scenario configs from the latest (or specified)
     sweep_summary.csv. Uses only the proportional_ramp rows as the
     canonical config list — each scenario appears once.
  2. For each config, run the SAC agent deterministically.
  3. Record the same metrics saved by sweep.py:
     outcome, min_ped_distance_m, min_ttc_s, max_jerk_mps3,
     mean_jerk_mps3, ego_speed_at_trigger, time_to_stop_s, etc.
  4. Save to sac_on_sweep_results.csv in the same folder as the sweep.

The output CSV can be merged with sweep_summary.csv (same columns) for
unified comparison plots. Use brake_profile = "SAC_800k" as the label.

Usage:
    python eval_sac_on_sweep.py
    python eval_sac_on_sweep.py sac_aeb_rand_800k
    python eval_sac_on_sweep.py sac_aeb_rand_800k runs/20260405_025448/sweep_summary.csv

Prerequisite: CARLA must be running (CarlaUE4.exe).
"""

import os
import sys
import csv
import glob
import math

import numpy as np
import carla as _carla

from stable_baselines3 import SAC
from carla_aeb_env import CarlaAEBEnv
from scenario_config import ScenarioConfig
from avoidability import compute_avoidability
from math_utils import get_speed_mps

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

MODEL_PATH  = sys.argv[1] if len(sys.argv) > 1 else "sac_aeb_rand_1600k"
SWEEP_PATH  = sys.argv[2] if len(sys.argv) > 2 else None

def _find_latest_sweep_csv():
    run_root   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    candidates = sorted(glob.glob(os.path.join(run_root, "*", "sweep_summary.csv")))
    if not candidates:
        raise FileNotFoundError("No sweep_summary.csv found under runs/")
    return candidates[-1]

sweep_csv_path = SWEEP_PATH or _find_latest_sweep_csv()
print(f"Model      : {MODEL_PATH}")
print(f"Sweep CSV  : {sweep_csv_path}")

# ---------------------------------------------------------------------------
# Load scenario configs from sweep (deduplicate — one config per scenario)
# We use only the proportional_ramp rows as the canonical list because every
# scenario in the sweep was run under all four profiles; proportional_ramp
# rows give us the unique config set with no repetition.
# ---------------------------------------------------------------------------

CANONICAL_PROFILE = "proportional_ramp"
sweep_configs = []

with open(sweep_csv_path, newline="") as f:
    for r in csv.DictReader(f):
        if r.get("brake_profile") != CANONICAL_PROFILE:
            continue
        try:
            cfg = ScenarioConfig(
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
                weather_preset        = r.get("weather_preset", "ClearSunset"),
                sim_seconds           = float(r.get("sim_seconds", 18)),
                brake_profile         = "proportional_ramp",  # ignored in RL env
            )
            cfg.run_id = r.get("run_id", "")
            sweep_configs.append(cfg)
        except (ValueError, KeyError) as e:
            print(f"  [warn] skipping row: {e}")

print(f"Loaded {len(sweep_configs)} unique scenario configs from sweep")

# ---------------------------------------------------------------------------
# Set up environment and model
# ---------------------------------------------------------------------------

env   = CarlaAEBEnv(cfg=sweep_configs[0])
model = SAC.load(MODEL_PATH, env=env)

# ---------------------------------------------------------------------------
# Run SAC on each config
# ---------------------------------------------------------------------------

out_rows = []

for i, cfg in enumerate(sweep_configs):
    cfg.brake_profile = "proportional_ramp"  # env ignores it; field must be valid

    # Inject config into env and reset
    env.cfg = cfg
    obs, _ = env.reset()
    avoid  = compute_avoidability(cfg)

    done = False
    last_info = {}

    # Per-episode accumulators
    # jerk_ticks: skip first JERK_WARMUP_TICKS to avoid LP-filter initialization
    # spike from the spawned vehicle's velocity being nonzero while the filter
    # starts at zero. The spike always dominates max_jerk and is not real.
    JERK_WARMUP_TICKS = 10
    jerk_ticks        = []
    _tick_count       = 0
    min_ped_dist_m    = float("inf")
    min_ttc_obs_s     = float("inf")
    hazard_start_s    = None
    stop_time_s       = None
    ego_speed_at_trig = None
    last_terminated   = False
    collision_out     = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, last_info = env.step(action)
        done = terminated or truncated
        last_terminated = terminated

        _tick_count += 1
        if _tick_count > JERK_WARMUP_TICKS:
            jerk_ticks.append(last_info.get("jerk_mps3", 0.0))

        ttc_tick = last_info.get("ttc_s", float("inf"))
        if ttc_tick < min_ttc_obs_s:
            min_ttc_obs_s = ttc_tick

        if env._walker is not None and env._vehicle is not None:
            try:
                d = env._vehicle.get_location().distance(env._walker.get_location())
                if d < min_ped_dist_m:
                    min_ped_dist_m = d
            except Exception:
                pass

        sim_t = last_info.get("sim_time_s", 0.0)
        mode  = last_info.get("drive_mode", "CRUISE")

        if hazard_start_s is None and mode in ("HAZARD_BRAKE", "STOP_HOLD"):
            hazard_start_s = sim_t
            try:
                ego_speed_at_trig = get_speed_mps(env._vehicle)
            except Exception:
                ego_speed_at_trig = None

        if (hazard_start_s is not None and stop_time_s is None
                and env._vehicle is not None):
            try:
                if get_speed_mps(env._vehicle) < 0.3:
                    stop_time_s = sim_t
            except Exception:
                pass

        if last_info.get("collision", False):
            collision_out = True

    # Determine outcome
    hit       = collision_out
    full_stop = last_terminated and not hit
    if hit:
        outcome = "collision"
    elif full_stop:
        outcome = "full_stop"
    else:
        outcome = "slowed_avoided"

    max_jerk  = max(jerk_ticks) if jerk_ticks else 0.0
    mean_jerk = sum(jerk_ticks) / len(jerk_ticks) if jerk_ticks else 0.0
    time_to_stop = (round(stop_time_s - hazard_start_s, 3)
                    if (stop_time_s is not None and hazard_start_s is not None)
                    else None)
    v_trig_mps = ego_speed_at_trig if ego_speed_at_trig is not None else 0.0
    v_trig_mph = v_trig_mps / 0.44704

    out_rows.append({
        "run_id":                   cfg.run_id or f"sac_{i:04d}",
        "brake_profile":            "SAC_" + MODEL_PATH.replace(".zip", "").removeprefix("sac_aeb_rand_").removeprefix("sac_"),
        "target_mph":               cfg.target_mph,
        "encounter_distance_m":     cfg.encounter_distance_m,
        "walker_speed_mps":         cfg.walker_speed_mps,
        "walker_side":              cfg.walker_side,
        "walker_cross":             cfg.walker_cross,
        "trigger_ttc_s":            cfg.trigger_ttc_s if cfg.trigger_ttc_s else "",
        "braking_ramp_up_per_s":    cfg.braking_ramp_up_per_s,
        "brake_headway_s":          cfg.brake_headway_s,
        "weather_preset":           cfg.weather_preset,
        "hazard_triggered":         hazard_start_s is not None,
        "collision_detected":       hit,
        "outcome":                  outcome,
        "min_ped_distance_m":       round(min_ped_dist_m, 3) if min_ped_dist_m != float("inf") else "",
        "min_ttc_s":                round(min_ttc_obs_s, 3) if min_ttc_obs_s != float("inf") else "",
        "max_jerk_mps3":            round(max_jerk, 2),
        "mean_jerk_mps3":           round(mean_jerk, 2),
        "ego_speed_at_trigger_mph": round(v_trig_mph, 2),
        "ego_speed_at_trigger_mps": round(v_trig_mps, 3),
        "time_to_stop_s":           time_to_stop if time_to_stop is not None else "",
        "physics_label":            avoid["label"],
        "margin_m":                 round(avoid["margin_m"], 2) if avoid["margin_m"] != float("inf") else "inf",
    })

    print(f"  [{i+1:3d}/{len(sweep_configs)}] {cfg.target_mph:.0f}mph "
          f"ttc={cfg.trigger_ttc_s}s  outcome={outcome:<14}  "
          f"jerk_max={max_jerk:.0f}  physics={avoid['label']}")

env.close()

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

out_dir  = os.path.dirname(sweep_csv_path)
out_path = os.path.join(out_dir, "sac_on_sweep_results.csv")

with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=out_rows[0].keys())
    writer.writeheader()
    writer.writerows(out_rows)

print(f"\nSaved {len(out_rows)} rows -> {out_path}")
print(f"\nTo build unified comparison: merge {os.path.basename(sweep_csv_path)}")
print(f"  + sac_on_sweep_results.csv on matching config columns,")
print(f"  then run plot_unified_comparison.py (next script to write).")

# Summary
n_col   = sum(1 for r in out_rows if r["collision_detected"])
n_stop  = sum(1 for r in out_rows if r["outcome"] == "full_stop")
n_total = len(out_rows)
avoidable_rows = [r for r in out_rows if r["physics_label"] in ("avoidable", "borderline-avoidable")]
n_col_avoid = sum(1 for r in avoidable_rows if r["collision_detected"])
print(f"\n{'='*50}")
print(f"  SAC on matched sweep ({n_total} episodes)")
print(f"  Collisions       : {n_col}/{n_total}")
print(f"  Full stops       : {n_stop}/{n_total}")
print(f"  Collisions on avoidable scenarios: {n_col_avoid}/{len(avoidable_rows)}")
print(f"{'='*50}")
