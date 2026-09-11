"""
brake_calibration.py
Quentin | Junior Spring Research, 2026.

Finds the optimal (shortest-stopping) constant brake force for each speed
by running short CARLA experiments at different brake levels.

Why this matters:
  Without ABS, applying full brake (1.0) locks the wheels. Locked wheels
  slide on kinetic friction, which is LOWER than rolling static friction.
  The optimal brake force is just below the lock threshold — this gives
  maximum deceleration. step_constant overshoots this and is therefore NOT
  the upper bound on braking performance.

  avoidability.py needs a_max = v0² / (2 * d_stop_best), where d_stop_best
  comes from the best-performing brake level at each speed, not brake=1.0.

What this script does:
  For each test speed in TEST_SPEEDS_MPH:
    For each brake level in BRAKE_LEVELS:
      - Spawn ego at test speed (use high encounter distance so it reaches cruise)
      - Apply constant brake at that level from t=0
      - Measure distance traveled until v < 0.5 m/s (effectively stopped)
      - Record stopping distance
  Then find which brake level minimized stopping distance at each speed.
  Report effective a_max = v0² / (2 * d_stop_best) for each speed.
  Write results to brake_calibration_results.csv.

Usage:
    python brake_calibration.py

Output:
    brake_calibration_results.csv   <- full results
    Prints recommended A_MAX_DEFAULT for avoidability.py

Notes:
  - CARLA must be running (headless or windowed)
  - Each test run is ~5-10 seconds of sim time
  - Total runtime: ~15-30 minutes depending on number of combos
"""

import os
import sys
import csv
import time

import numpy as np

CARLA_ROOT = r"C:\Users\qdruc\OneDrive\Desktop\CARLA_0.9.16"
for _p in [
    os.path.join(CARLA_ROOT, "PythonAPI"),
    os.path.join(CARLA_ROOT, "PythonAPI", "carla"),
]:
    if _p not in sys.path:
        sys.path.append(_p)

import carla

from math_utils import get_speed_mps
from spawning import prepare_spawn_context, spawn_ego_vehicle
from carla_session import connect_and_load_world, enable_sync_mode, restore_async_mode

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

TEST_SPEEDS_MPH = [25.0, 35.0, 45.0, 50.0]
BRAKE_LEVELS    = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

FIXED_DT        = 0.02      # 50 Hz sync
EGO_SPAWN_INDEX = 242
WARMUP_DISTANCE_M = 100.0   # drive this far before starting brake test (reach cruise speed)
MAX_TEST_TICKS  = 1000       # safety limit (20 seconds)
STOP_THRESHOLD_MPS = 0.5    # consider stopped below this speed

TARGET_MAP = "Town04_Opt"
OUTPUT_CSV = "brake_calibration_results.csv"

# ---------------------------------------------------------------------------
# Connect
# ---------------------------------------------------------------------------

print("[brake_calibration] Connecting to CARLA...")
client, world = connect_and_load_world(
    host="localhost", port=2000, timeout_s=15.0, target_map=TARGET_MAP
)
enable_sync_mode(world, fixed_dt=FIXED_DT)
bp_lib = world.get_blueprint_library()
print("[brake_calibration] Connected.\n")

# ---------------------------------------------------------------------------
# Run one test: spawn at target speed, apply constant brake, measure stop dist
# ---------------------------------------------------------------------------

def run_brake_test(target_mph: float, brake_level: float) -> dict:
    """
    Spawn ego vehicle, cruise to target speed, then apply constant brake.
    Returns dict with stopping distance, effective a_max, and raw data.
    """
    target_mps = target_mph * 0.44704
    spawn_points, ego_tf, _ = prepare_spawn_context(world, SPAWN_INDEX=EGO_SPAWN_INDEX)
    vehicle: carla.Vehicle = spawn_ego_vehicle(world, bp_lib=bp_lib, spawn_tf=ego_tf)  # type: ignore[assignment]
    world.tick()

    # --- Cruise to target speed ---
    dist_traveled = 0.0
    prev_loc = vehicle.get_transform().location
    speed_state = {"i_term": 0.0, "kp": 0.50, "ki": 0.10, "brake_prev": 0.0}

    warmup_ticks = 0
    while dist_traveled < WARMUP_DISTANCE_M:
        world.tick()
        warmup_ticks += 1
        loc_now = vehicle.get_transform().location
        dist_traveled += loc_now.distance(prev_loc)
        prev_loc = loc_now

        speed = get_speed_mps(vehicle)
        error = target_mps - speed
        throttle = min(max(0.5 + 0.5 * error / target_mps, 0.0), 1.0)
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=0.0))

        if warmup_ticks > 2000:  # safety: don't loop forever
            break

    # Record actual speed at brake start
    v0_mps = get_speed_mps(vehicle)
    v0_mph = v0_mps / 0.44704

    # --- Apply constant brake until stopped ---
    brake_start_loc = vehicle.get_transform().location
    brake_dist = 0.0
    prev_loc = brake_start_loc
    ticks = 0

    while ticks < MAX_TEST_TICKS:
        world.tick()
        ticks += 1
        loc_now = vehicle.get_transform().location
        brake_dist += loc_now.distance(prev_loc)
        prev_loc = loc_now

        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=brake_level))

        if get_speed_mps(vehicle) < STOP_THRESHOLD_MPS:
            break

    vehicle.destroy()
    world.tick()

    # Effective a_max from measured stopping distance
    a_max_eff = (v0_mps ** 2) / (2.0 * brake_dist) if brake_dist > 0.01 else 0.0

    return {
        "target_mph":    target_mph,
        "v0_mps":        round(v0_mps, 2),
        "v0_mph":        round(v0_mph, 1),
        "brake_level":   brake_level,
        "stop_dist_m":   round(brake_dist, 2),
        "a_max_eff":     round(a_max_eff, 3),
        "ticks":         ticks,
    }


# ---------------------------------------------------------------------------
# Run all combinations
# ---------------------------------------------------------------------------

all_results = []
n_total = len(TEST_SPEEDS_MPH) * len(BRAKE_LEVELS)
n_done  = 0

print(f"Running {n_total} brake tests ({len(TEST_SPEEDS_MPH)} speeds × {len(BRAKE_LEVELS)} brake levels)...\n")

for mph in TEST_SPEEDS_MPH:
    print(f"  Speed: {mph:.0f} mph")
    speed_results = []
    for brake in BRAKE_LEVELS:
        result = run_brake_test(mph, brake)
        speed_results.append(result)
        all_results.append(result)
        n_done += 1
        print(f"    brake={brake:.1f} -> stop_dist={result['stop_dist_m']:.1f}m  "
              f"a_max_eff={result['a_max_eff']:.2f} m/s²  "
              f"(actual v0={result['v0_mph']:.1f} mph)")
        time.sleep(0.5)  # brief pause between spawns

    # Find optimal brake for this speed
    best = min(speed_results, key=lambda r: r["stop_dist_m"])
    print(f"  -> BEST at {mph:.0f} mph: brake={best['brake_level']:.1f}  "
          f"stop_dist={best['stop_dist_m']:.1f}m  a_max={best['a_max_eff']:.2f} m/s²\n")

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
    writer.writeheader()
    writer.writerows(all_results)
print(f"Results saved -> {OUTPUT_CSV}\n")

# ---------------------------------------------------------------------------
# Summary: recommended a_max for avoidability.py
# ---------------------------------------------------------------------------

print("=" * 60)
print("  SUMMARY — Optimal brake level by speed")
print("=" * 60)
print(f"  {'Speed':>8}  {'Best brake':>10}  {'Stop dist':>10}  {'a_max_eff':>10}")
print(f"  {'-'*50}")

best_by_speed = []
for mph in TEST_SPEEDS_MPH:
    speed_rows = [r for r in all_results if r["target_mph"] == mph]
    best = min(speed_rows, key=lambda r: r["stop_dist_m"])
    best_by_speed.append(best)
    print(f"  {mph:>7.0f}mph  {best['brake_level']:>10.1f}  "
          f"{best['stop_dist_m']:>9.1f}m  {best['a_max_eff']:>9.2f} m/s²")

# Conservative a_max = minimum across all speeds (worst case)
conservative_a = min(r["a_max_eff"] for r in best_by_speed)
mean_a         = sum(r["a_max_eff"] for r in best_by_speed) / len(best_by_speed)

print(f"\n  Conservative a_max (minimum): {conservative_a:.2f} m/s²")
print(f"  Mean a_max across speeds    : {mean_a:.2f} m/s²")
print(f"\n  Recommended setting for avoidability.py:")
print(f"    A_MAX_DEFAULT = {conservative_a:.1f}   # calibrated from CARLA brake test")
print(f"\n  NOTE: use conservative (minimum) value — avoidability checks should")
print(f"  err toward 'avoidable' rather than mis-labeling things as impossible.")
print("=" * 60)

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

restore_async_mode(world)
print("\n[brake_calibration] Done.")
