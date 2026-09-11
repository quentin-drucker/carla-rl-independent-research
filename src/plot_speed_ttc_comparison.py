"""
plot_speed_ttc_comparison.py
Quentin | Junior Spring Research, 2026.

5-panel Speed vs TTC figure: SAC (800k) + 4 hand-designed braking profiles.
Each panel = one controller. Physics avoidability boundary overlaid on every panel.

Reads:
    eval_results.csv                        — SAC 100-episode eval (from eval_sac.py)
    runs/<latest>/sweep_summary.csv         — profile sweep (from sweep.py)

Outputs:
    speed_ttc_5panel.png

Point encoding:
    Color   = physics avoidability label (avoidable / borderline / impossible)
    Marker  = circle (no collision) or X (collision)
    Each panel = one controller; shared physics boundary line + borderline band

SCOPE NOTE:
  SAC was evaluated on randomized scenarios (22–50 mph, TTC 1.5–4.0 s).
  Profiles were evaluated on a fixed grid (25/35/45 mph, TTC 2.0/2.8 s).
  This figure shows WHERE each controller places collision points relative
  to the physics boundary — not a matched head-to-head comparison.
  Use eval_sac_on_sweep.py to run SAC on the same 48 configs for a
  directly matched comparison (needed for B4 / unified bar chart).

Usage:
    python plot_speed_ttc_comparison.py
    python plot_speed_ttc_comparison.py eval_results.csv runs/20260405_025448/sweep_summary.csv
"""

import os
import sys
import csv
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Physics constants (mirrors avoidability.py)
# ---------------------------------------------------------------------------

A_MAX_DEFAULT    = 3.5   # m/s²  effective system deceleration (ramp-limiter-inclusive)
BORDERLINE_M     = 8.0   # m     avoidability uncertainty band

def _physics_label(target_mph, trigger_ttc_s, walker_speed_mps, walker_cross,
                   walker_startup_s=0.5):
    """
    Inline version of compute_avoidability() that works without importing
    ScenarioConfig. Returns one of: avoidable, borderline-avoidable,
    borderline-impossible, impossible.
    """
    FAR_CROSS_LANE_M = 3.3
    v0  = target_mph * 0.44704
    ttc = trigger_ttc_s if trigger_ttc_s is not None else (walker_startup_s + 1.5)

    if str(walker_cross).lower() == "far":
        if walker_speed_mps > 0:
            time_to_clear = FAR_CROSS_LANE_M / walker_speed_mps
            if ttc >= time_to_clear:
                return "avoidable"

    d_avail    = v0 * ttc
    d_stop     = v0**2 / (2.0 * A_MAX_DEFAULT)
    margin     = d_avail - d_stop
    avoidable  = margin >= 0.0
    borderline = abs(margin) < BORDERLINE_M

    if avoidable and borderline:  return "borderline-avoidable"
    if avoidable:                 return "avoidable"
    if borderline:                return "borderline-impossible"
    return "impossible"


# ---------------------------------------------------------------------------
# Locate input files
# ---------------------------------------------------------------------------

def _find_latest_sweep_csv():
    run_root   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    candidates = sorted(glob.glob(os.path.join(run_root, "*", "sweep_summary.csv")))
    if not candidates:
        raise FileNotFoundError(
            "No sweep_summary.csv found under runs/. "
            "Pass the path explicitly: python plot_speed_ttc_comparison.py <sac_csv> <sweep_csv>"
        )
    return candidates[-1]


if len(sys.argv) >= 3:
    _sac_path   = sys.argv[1]
    _sweep_path = sys.argv[2]
elif len(sys.argv) == 2:
    _sac_path   = sys.argv[1]
    _sweep_path = _find_latest_sweep_csv()
else:
    _sac_path   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.csv")
    _sweep_path = _find_latest_sweep_csv()

print(f"SAC data   : {_sac_path}")
print(f"Sweep data : {_sweep_path}")


# ---------------------------------------------------------------------------
# Load SAC results (eval_results.csv)
# ---------------------------------------------------------------------------

sac_rows = []
with open(_sac_path, newline="") as f:
    for r in csv.DictReader(f):
        if not r.get("ttc_s"):
            continue
        sac_rows.append({
            "speed_mph":     float(r["speed_mph"]),
            "ttc_s":         float(r["ttc_s"]),
            "physics_label": r["physics_label"],
            "collision":     int(r["collision"]),
        })
print(f"Loaded {len(sac_rows)} SAC episodes")


# ---------------------------------------------------------------------------
# Load profile sweep + compute avoidability labels
# ---------------------------------------------------------------------------

PROFILE_ORDER = ["proportional_ramp", "step_constant", "cautious_ramp", "exponential"]
profile_rows: dict[str, list] = {p: [] for p in PROFILE_ORDER}

with open(_sweep_path, newline="") as f:
    for r in csv.DictReader(f):
        profile = r.get("brake_profile", "")
        if profile not in PROFILE_ORDER:
            continue
        ttc_str = r.get("trigger_ttc_s", "")
        if not ttc_str:
            continue
        try:
            ttc_val   = float(ttc_str)
            mph_val   = float(r["target_mph"])
            spd_val   = float(r.get("walker_speed_mps", "1.8"))
            cross_val = r.get("walker_cross", "near")
            su_val    = float(r.get("walker_startup_s", "0.5"))
            label     = _physics_label(mph_val, ttc_val, spd_val, cross_val, su_val)
            col_val   = 1 if str(r.get("collision_detected", "False")).lower() == "true" else 0
        except (ValueError, KeyError):
            continue

        profile_rows[profile].append({
            "speed_mph":     mph_val,
            "ttc_s":         ttc_val,
            "physics_label": label,
            "collision":     col_val,
        })

for p in PROFILE_ORDER:
    print(f"  {p}: {len(profile_rows[p])} runs")


# ---------------------------------------------------------------------------
# Color + style scheme
# ---------------------------------------------------------------------------

LABEL_COLORS = {
    "avoidable":             "#2ecc71",
    "borderline-avoidable":  "#f1c40f",
    "borderline-impossible": "#e67e22",
    "impossible":            "#e74c3c",
}

PROFILE_DISPLAY = {
    "SAC (800k steps)":  "SAC (800k steps)",
    "proportional_ramp": "Proportional Ramp\n(baseline)",
    "step_constant":     "Step Constant",
    "cautious_ramp":     "Cautious Ramp",
    "exponential":       "Exponential",
}


# ---------------------------------------------------------------------------
# Physics boundary curve
# ---------------------------------------------------------------------------

_speeds_mph = np.linspace(17, 60, 400)
_speeds_mps = _speeds_mph * 0.44704
_ttc_bound  = _speeds_mps / (2.0 * A_MAX_DEFAULT)           # boundary: margin = 0
_ttc_upper  = _ttc_bound + BORDERLINE_M / _speeds_mps       # +8m band edge
_ttc_lower  = np.maximum(0, _ttc_bound - BORDERLINE_M / _speeds_mps)  # -8m band edge


# ---------------------------------------------------------------------------
# Build 2×3 figure (5 panels + 1 legend panel)
# ---------------------------------------------------------------------------

panel_data = [("SAC (800k steps)", sac_rows)] + [
    (p, profile_rows[p]) for p in PROFILE_ORDER
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True, sharex=True)
axes_flat  = axes.flatten()

for idx, (key, rows) in enumerate(panel_data):
    ax = axes_flat[idx]

    for r in rows:
        col    = LABEL_COLORS.get(r["physics_label"], "#aaaaaa")
        marker = "x" if r["collision"] else "o"
        size   = 55 if r["collision"] else 28
        lw     = 1.8 if r["collision"] else 0.4
        ax.scatter(r["ttc_s"], r["speed_mph"],
                   c=col, marker=marker, s=size, linewidths=lw, zorder=3, alpha=0.88)

    # Physics boundary
    ax.plot(_ttc_bound, _speeds_mph, "k--", linewidth=1.6, zorder=5)
    ax.fill_betweenx(_speeds_mph, 0, _ttc_bound,
                     alpha=0.06, color="red", zorder=1)
    ax.fill_betweenx(_speeds_mph, _ttc_lower, _ttc_upper,
                     alpha=0.11, color="orange", zorder=2)

    n_col = sum(r["collision"] for r in rows)
    n_tot = len(rows)
    pct   = 100 * n_col / n_tot if n_tot > 0 else 0.0
    title = PROFILE_DISPLAY.get(key, key.replace("_", " ").title())
    ax.set_title(f"{title}\n{n_col}/{n_tot} collisions ({pct:.0f}%)",
                 fontsize=10, fontweight="bold")

    ax.set_xlim(1.0, 4.6)
    ax.set_ylim(17, 60)
    ax.grid(alpha=0.22)

    if idx in (0, 3):
        ax.set_ylabel("Ego Speed (mph)", fontsize=9)
    if idx in (3, 4):
        ax.set_xlabel("Trigger TTC (s)", fontsize=9)


# ---------------------------------------------------------------------------
# Legend panel (axes_flat[5])
# ---------------------------------------------------------------------------

ax_leg = axes_flat[5]
ax_leg.set_axis_off()

legend_elements = []
for lbl, col in LABEL_COLORS.items():
    legend_elements.append(mpatches.Patch(facecolor=col, label=lbl))
legend_elements.append(
    mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                  markersize=7, label="No collision"))
legend_elements.append(
    mlines.Line2D([], [], color="gray", marker="x", linestyle="None",
                  markersize=7, markeredgewidth=1.8, label="Collision"))
legend_elements.append(
    mlines.Line2D([], [], color="black", linestyle="--", linewidth=1.6,
                  label=f"Physics limit\n(a_eff = {A_MAX_DEFAULT} m/s²)"))
legend_elements.append(
    mpatches.Patch(facecolor="red", alpha=0.15, label="Impossible zone"))
legend_elements.append(
    mpatches.Patch(facecolor="orange", alpha=0.22,
                   label=f"Borderline band\n(\u00b1{BORDERLINE_M:.0f} m)"))

ax_leg.legend(handles=legend_elements, loc="center", fontsize=9,
              framealpha=0.92, title="Legend", title_fontsize=10,
              labelspacing=0.7)


# ---------------------------------------------------------------------------
# Shared title + scope note
# ---------------------------------------------------------------------------

fig.suptitle(
    "Speed vs Trigger TTC — Collision Outcomes and Physics Avoidability Boundary\n"
    "SAC (800k) vs Hand-Designed Braking Profiles",
    fontsize=13, fontweight="bold",
)

scope_note = (
    "SAC: randomized eval (22\u201350 mph, TTC 1.5\u20134.0 s, n=100). "
    "Profiles: fixed grid (25/35/45 mph, TTC 2.0/2.8 s). "
    "Not a matched comparison \u2014 see eval_sac_on_sweep.py for head-to-head on identical configs."
)
fig.text(0.5, 0.005, scope_note, ha="center", fontsize=7.5, color="#555555")

fig.tight_layout(rect=[0, 0.025, 1, 1])

out_path = os.path.join(os.path.dirname(os.path.abspath(_sac_path)), "speed_ttc_5panel.png")
fig.savefig(out_path, dpi=150)
print(f"\nSaved -> {out_path}")
