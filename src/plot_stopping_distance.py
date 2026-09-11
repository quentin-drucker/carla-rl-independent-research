"""
plot_stopping_distance.py
Quentin | Junior Spring Research, 2026.

Stopping distance reference plot: raw vehicle capability vs effective system
(including ramp-limiter overhead), over the project's speed range.

Reads:
    brake_calibration_results.csv   — direct apply_control measurements
                                      (brake_calibration.py output)

Outputs:
    stopping_distance_reference.png

What this shows:
  Line A  — Raw vehicle capability at full brake (brake=1.0), theoretical:
             d = v²/(2 * a_vehicle)  where a_vehicle ≈ 7.2 m/s² (median from calibration)
  Scatter — Measured calibration points at brake_level=1.0 (each target speed)
  Line B  — Effective system through full controller pipeline:
             d = v²/(2 * A_MAX_DEFAULT)  where A_MAX_DEFAULT = 3.5 m/s²
             This is the line used by avoidability.py for the physics boundary
  Band    — Borderline uncertainty band: ±8 m around line B
             (same band shown in the speed×TTC figures)

The gap between lines A and B is the RAMP LIMITER OVERHEAD — the extra stopping
distance the controller pays because braking ramps up gradually rather than
slamming immediately to brake=1.0. At 35 mph this overhead is roughly 17 m
(raw: 16m, effective: 32m).

Usage:
    python plot_stopping_distance.py
    python plot_stopping_distance.py brake_calibration_results.csv
"""

import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

A_MAX_DEFAULT  = 3.5    # m/s²  effective system (ramp-limiter inclusive) — avoidability.py
BORDERLINE_M   = 8.0    # m     avoidability uncertainty margin
MPH_TO_MPS     = 0.44704

# ---------------------------------------------------------------------------
# Load calibration data
# ---------------------------------------------------------------------------

_script_dir = os.path.dirname(os.path.abspath(__file__))
_cal_path = (sys.argv[1] if len(sys.argv) > 1
             else os.path.join(_script_dir, "brake_calibration_results.csv"))

cal_rows = []
with open(_cal_path, newline="") as f:
    for r in csv.DictReader(f):
        cal_rows.append({
            "v0_mph":    float(r["v0_mph"]),
            "v0_mps":    float(r["v0_mps"]),
            "brake":     float(r["brake_level"]),
            "stop_m":    float(r["stop_dist_m"]),
            "a_eff":     float(r["a_max_eff"]),
        })

# Full-brake points only (brake_level = 1.0)
full_brake = [r for r in cal_rows if abs(r["brake"] - 1.0) < 1e-9]
print(f"Loaded {len(cal_rows)} calibration rows ({len(full_brake)} at brake=1.0)")

# Median raw effective a_max at full brake (vehicle capability, before ramp)
a_raw_median = float(np.median([r["a_eff"] for r in full_brake]))
print(f"  Median raw a_max (full brake): {a_raw_median:.2f} m/s²")
print(f"  Effective a_max (ramp system): {A_MAX_DEFAULT:.2f} m/s²")

# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------

v_mph  = np.linspace(15, 57, 400)
v_mps  = v_mph * MPH_TO_MPS

# Line A: raw vehicle theoretical (full brake, no ramp delay)
d_raw  = v_mps**2 / (2.0 * a_raw_median)

# Line B: effective system (through ramp limiter) — used by avoidability.py
d_eff  = v_mps**2 / (2.0 * A_MAX_DEFAULT)

# Borderline band around line B
d_upper = d_eff + BORDERLINE_M
d_lower = np.maximum(0.0, d_eff - BORDERLINE_M)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))

# Borderline band
ax.fill_between(v_mph, d_lower, d_upper,
                alpha=0.18, color="orange",
                label=f"Borderline band (\u00b1{BORDERLINE_M:.0f} m around line B)")

# Line B: effective system
ax.plot(v_mph, d_eff, color="#e74c3c", linewidth=2.2, linestyle="-",
        label=f"B — Effective system (a_eff = {A_MAX_DEFAULT} m/s², ramp-limited)\n"
              "     Used by avoidability.py as the physics boundary")

# Line A: raw vehicle at full brake
ax.plot(v_mph, d_raw, color="#2c3e50", linewidth=2.0, linestyle="--",
        label=f"A — Raw vehicle capability (a_raw \u2248 {a_raw_median:.1f} m/s², brake=1.0, no ramp)")

# Calibration scatter at full brake
cal_mph  = [r["v0_mph"]  for r in full_brake]
cal_dist = [r["stop_m"]  for r in full_brake]
ax.scatter(cal_mph, cal_dist, s=90, color="#2c3e50", zorder=5,
           label="Measured: direct apply_control at brake=1.0")

# Overhead annotation arrows at reference speeds
for v_ref_mph, label_offset in [(25.1, 3), (35.1, 4), (45.1, 5)]:
    v_ref_mps = v_ref_mph * MPH_TO_MPS
    d_r = v_ref_mps**2 / (2.0 * a_raw_median)
    d_e = v_ref_mps**2 / (2.0 * A_MAX_DEFAULT)
    overhead = d_e - d_r
    mid = (d_r + d_e) / 2.0
    ax.annotate(
        f"+{overhead:.0f} m\nramp\noverhead",
        xy=(v_ref_mph, mid),
        fontsize=7.5,
        color="#888888",
        ha="left",
        va="center",
        xytext=(v_ref_mph + label_offset, mid),
        arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.8),
    )
    ax.annotate("", xy=(v_ref_mph - 0.3, d_r + 0.3),
                xytext=(v_ref_mph - 0.3, d_e - 0.3),
                arrowprops=dict(arrowstyle="<->", color="#aaaaaa", lw=1.0))

# Reference speed vertical lines
for v_ref, col in [(25, "#3498db"), (35, "#2ecc71"), (45, "#e67e22")]:
    ax.axvline(v_ref, color=col, linewidth=0.7, linestyle=":", alpha=0.6)

ax.set_xlabel("Initial Speed (mph)", fontsize=12)
ax.set_ylabel("Stopping Distance (m)", fontsize=12)
ax.set_title(
    "Stopping Distance: Raw Vehicle Capability vs Effective Controller System\n"
    "The Gap Is the Ramp-Limiter Overhead",
    fontsize=12,
)
ax.legend(fontsize=8.5, loc="upper left", framealpha=0.92)
ax.set_xlim(14, 58)
ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
ax.grid(alpha=0.28)

note = (
    "Road friction: CARLA default dry asphalt (friction not varied in current setup). "
    "Ramp-up rate: 4.0 brake units/s (braking_ramp_up_per_s in ScenarioConfig). "
    "A_MAX_DEFAULT = 3.5 m/s\u00b2 back-calculated from controller pipeline sweep data."
)
fig.text(0.5, 0.01, note, ha="center", fontsize=7.5, color="#555555")
fig.tight_layout(rect=[0, 0.05, 1, 1])

out_path = os.path.join(_script_dir, "stopping_distance_reference.png")
fig.savefig(out_path, dpi=150)
print(f"Saved -> {out_path}")
