"""
validate_avoidability.py
Quentin | Junior Spring Research, 2026.

Cross-validates avoidability.py predictions against actual sweep results.

Method:
  - Load the most recent sweep_summary.csv
  - Filter for step_constant brake_profile (full brake from trigger — closest to
    "best possible braking" achievable with this controller)
  - For each near-cross run, compute what avoidability.py predicts
  - Compare prediction vs actual outcome
  - Report mismatches and compute implied a_max from actual stopping data

Why step_constant only:
  proportional_ramp and other profiles don't apply full brake immediately, so
  a collision with those profiles doesn't tell us the scenario is physically
  impossible — it might just mean the profile was too gentle. step_constant
  applies brake=1.0 from the first hazard tick, which is the closest we have
  to the theoretical maximum braking used in the physics model.

NOTE on ego_speed_at_trigger_mps:
  avoidability.py uses cfg.target_mph as v0, but the actual speed at trigger
  can be lower if encounter_distance_m is short (ego still accelerating).
  This script uses the ACTUAL logged speed for a more accurate validation.

Usage:
    python validate_avoidability.py
    python validate_avoidability.py runs/20260405_025448/sweep_summary.csv
"""

import sys
import os
import math
import pandas as pd
from avoidability import compute_avoidability, A_MAX_DEFAULT
from scenario_config import ScenarioConfig


# ---------------------------------------------------------------------------
# Find the most recent sweep CSV, or use one from command line
# ---------------------------------------------------------------------------

def find_latest_csv(base="runs"):
    """Return path to most recently modified sweep_summary.csv."""
    candidates = []
    for root, _, files in os.walk(base):
        for f in files:
            if f == "sweep_summary.csv":
                p = os.path.join(root, f)
                candidates.append((os.path.getmtime(p), p))
    if not candidates:
        raise FileNotFoundError(f"No sweep_summary.csv found under {base}/")
    return sorted(candidates)[-1][1]


csv_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_csv()
print(f"\nLoading: {csv_path}")
df = pd.read_csv(csv_path)
print(f"Total rows: {len(df)}")


# ---------------------------------------------------------------------------
# Filter: step_constant + near-cross + hazard triggered + no error
# ---------------------------------------------------------------------------

mask = (
    (df["brake_profile"] == "step_constant") &
    (df["walker_cross"] == "near") &
    (df["hazard_triggered"] == True) &
    (df["error"].isna() | (df["error"] == ""))
)
sc = df[mask].copy()
print(f"step_constant + near-cross rows: {len(sc)}")

if len(sc) == 0:
    print("\nNo step_constant near-cross rows found. Check your sweep CSV.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Run avoidability check on each row
# ---------------------------------------------------------------------------

results = []
implied_a_max_list = []

for _, row in sc.iterrows():
    # Use actual logged speed at trigger (more accurate than target_mph)
    v0_mps = row["ego_speed_at_trigger_mps"]

    # Reconstruct a minimal ScenarioConfig for avoidability check
    # (only fields used by compute_avoidability matter)
    cfg = ScenarioConfig(
        target_mph=v0_mps / 0.44704,     # back-convert actual speed
        trigger_ttc_s=row["trigger_ttc_s"] if not pd.isna(row["trigger_ttc_s"]) else None,
        walker_cross=row["walker_cross"],
        braking_ramp_up_per_s=row["braking_ramp_up_per_s"],
        walker_startup_s=row.get("walker_startup_s", 0.5),
    )

    avoid = compute_avoidability(cfg)
    actual_collision = (row["outcome"] == "collision") or (row["collision_detected"] == True)

    predicted_avoidable = avoid["avoidable"]
    d_available = avoid["d_available_m"]
    d_stop_pred = avoid["d_stop_m"]
    margin_pred = avoid["margin_m"]

    # Classify match/mismatch
    if predicted_avoidable and not actual_collision:
        verdict = "CORRECT (predicted avoidable, no collision)"
    elif predicted_avoidable and actual_collision:
        verdict = "WRONG: predicted avoidable but collision occurred -> a_max may be OVERESTIMATED"
    elif not predicted_avoidable and actual_collision:
        verdict = "CORRECT (predicted impossible, collision occurred)"
    else:  # not predicted_avoidable and not actual_collision
        verdict = "WRONG: predicted impossible but no collision -> a_max may be UNDERESTIMATED"

    # Implied a_max from actual data (only meaningful for clean stops)
    implied_a = None
    if not actual_collision and not pd.isna(row.get("min_ped_distance_m", float("nan"))):
        # d_stop_actual ≈ d_available - min_ped_distance
        d_stop_actual = d_available - row["min_ped_distance_m"]
        if d_stop_actual > 0 and v0_mps > 0:
            # Simplified: a_max = v0² / (2 * d_stop)
            implied_a = (v0_mps ** 2) / (2.0 * d_stop_actual)
            implied_a_max_list.append(implied_a)

    results.append({
        "run_id":           row["run_id"],
        "target_mph":       row["target_mph"],
        "v0_actual_mps":    round(v0_mps, 2),
        "trigger_ttc_s":    row["trigger_ttc_s"],
        "d_available_m":    round(d_available, 1),
        "d_stop_pred_m":    round(d_stop_pred, 1),
        "margin_pred_m":    round(margin_pred, 1),
        "actual_outcome":   row["outcome"],
        "min_ped_dist_m":   row.get("min_ped_distance_m", None),
        "implied_a_max":    round(implied_a, 2) if implied_a else None,
        "verdict":          verdict,
    })


# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------

results_df = pd.DataFrame(results)

print(f"\n{'='*70}")
print(f"  AVOIDABILITY VALIDATION — step_constant near-cross ({len(results)} runs)")
print(f"{'='*70}")

# Summary counts
correct   = sum(1 for r in results if r["verdict"].startswith("CORRECT"))
wrong_oa  = sum(1 for r in results if "OVERESTIMATED" in r["verdict"])
wrong_ue  = sum(1 for r in results if "UNDERESTIMATED" in r["verdict"])

print(f"\n  Correct predictions : {correct}/{len(results)}")
print(f"  Wrong (a_max too high — said avoidable, got collision): {wrong_oa}")
print(f"  Wrong (a_max too low  — said impossible, no collision): {wrong_ue}")

# Implied a_max from actual stopping data
if implied_a_max_list:
    mean_a = sum(implied_a_max_list) / len(implied_a_max_list)
    min_a  = min(implied_a_max_list)
    max_a  = max(implied_a_max_list)
    print(f"\n  Implied a_max from actual stops (simplified, ignores ramp):")
    print(f"    Mean : {mean_a:.2f} m/s²")
    print(f"    Min  : {min_a:.2f} m/s²  (conservative — use this for avoidability.py)")
    print(f"    Max  : {max_a:.2f} m/s²")
    print(f"    Current A_MAX_DEFAULT in avoidability.py: {A_MAX_DEFAULT} m/s²")
    if min_a < A_MAX_DEFAULT * 0.85:
        print(f"\n  *** WARNING: actual a_max is significantly lower than assumed.")
        print(f"      Consider setting A_MAX_DEFAULT = {min_a:.1f} in avoidability.py")
    elif min_a > A_MAX_DEFAULT * 1.15:
        print(f"\n  *** NOTE: actual a_max is higher than assumed — predictions are conservative.")
        print(f"      Could raise A_MAX_DEFAULT to {min_a:.1f} for tighter predictions.")
    else:
        print(f"\n  A_MAX_DEFAULT looks well-calibrated.")

# Detail table for mismatches
mismatches = [r for r in results if not r["verdict"].startswith("CORRECT")]
if mismatches:
    print(f"\n  MISMATCHES ({len(mismatches)}):")
    print(f"  {'run_id':<10} {'mph':>5} {'v0':>6} {'ttc':>5} {'d_avail':>8} "
          f"{'d_stop':>7} {'margin':>7} {'outcome':<14} {'implied_a':>9}")
    print(f"  {'-'*85}")
    for r in mismatches:
        print(f"  {r['run_id']:<10} {r['target_mph']:>5.0f} "
              f"{r['v0_actual_mps']:>6.1f} {r['trigger_ttc_s']:>5.1f} "
              f"{r['d_available_m']:>8.1f} {r['d_stop_pred_m']:>7.1f} "
              f"{r['margin_pred_m']:>7.1f} {r['actual_outcome']:<14} "
              f"{str(r['implied_a_max']):>9}")
else:
    print(f"\n  No mismatches — avoidability.py predictions match actual outcomes.")

print(f"\n{'='*70}\n")
