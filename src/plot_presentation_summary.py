"""
plot_presentation_summary.py
Quentin | Junior Spring Research, 2026.

Single presentation-ready figure summarizing SAC v2 vs four hand-designed
braking profiles across the matched 200-scenario sweep.

Four panels:
  1. Overall collision rate (bar chart)
  2. Full-stop rate by speed band (line chart) — shows where SAC wins/loses
  3. Collision rate in avoidable + borderline-avoidable zones only
  4. Key comfort/safety metrics (jerk, ped margin, time-to-stop) as dot+bar

Reads:
  sweep_summary.csv        — profile results
  sac_on_sweep_results.csv — SAC v2 results

Usage:
    python plot_presentation_summary.py
    python plot_presentation_summary.py runs/20260417_202355/sweep_summary.csv
"""

import os, sys, csv, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ---------------------------------------------------------------------------
# Locate files
# ---------------------------------------------------------------------------

def _latest_sweep_dir():
    run_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    dirs = sorted(glob.glob(os.path.join(run_root, "*")))
    for d in reversed(dirs):
        if os.path.exists(os.path.join(d, "sac_on_sweep_results.csv")):
            return d
    raise FileNotFoundError("No sweep folder with sac_on_sweep_results.csv found.")

if len(sys.argv) > 1:
    sweep_dir = os.path.dirname(os.path.abspath(sys.argv[1]))
else:
    sweep_dir = _latest_sweep_dir()

sweep_path = os.path.join(sweep_dir, "sweep_summary.csv")
sac_path   = os.path.join(sweep_dir, "sac_on_sweep_results.csv")

# ---------------------------------------------------------------------------
# Controller styling
# ---------------------------------------------------------------------------

PROFILE_ORDER = ["proportional_ramp", "step_constant", "cautious_ramp", "exponential"]

def _detect_sac_label(path):
    try:
        with open(path, newline="") as f:
            r = next(csv.DictReader(f))
            lbl = r.get("brake_profile", "SAC_v2_1600k")
            if lbl not in PROFILE_ORDER:
                return lbl
    except Exception:
        pass
    return "SAC_v2_1600k"

SAC_LABEL = _detect_sac_label(sac_path)
CONTROLLER_ORDER = [SAC_LABEL] + PROFILE_ORDER

DISPLAY_NAMES = {
    SAC_LABEL:           f"SAC v2\n({SAC_LABEL.replace('SAC_','').replace('_',' ')})",
    "proportional_ramp": "Proportional\nRamp",
    "step_constant":     "Step\nConstant",
    "cautious_ramp":     "Cautious\nRamp",
    "exponential":       "Exponential",
}
COLORS = {
    SAC_LABEL:           "#1a1a2e",
    "proportional_ramp": "#4C72B0",
    "step_constant":     "#DD8452",
    "cautious_ramp":     "#55A868",
    "exponential":       "#C44E52",
}

# ---------------------------------------------------------------------------
# Physics label (inline)
# ---------------------------------------------------------------------------

A_MAX = 3.5
BORDERLINE_M = 8.0

def _physics_label(r):
    try:
        v0  = float(r["target_mph"]) * 0.44704
        ttc = float(r["trigger_ttc_s"]) if r.get("trigger_ttc_s") else 2.0
        if str(r.get("walker_cross", "")).lower() == "far":
            wspd = float(r.get("walker_speed_mps", 0))
            if wspd > 0 and ttc >= 3.3 / wspd:
                return "avoidable"
        margin = v0 * ttc - v0 ** 2 / (2.0 * A_MAX)
        av = margin >= 0.0
        bl = abs(margin) < BORDERLINE_M
        if av and bl:  return "borderline-avoidable"
        if av:         return "avoidable"
        if bl:         return "borderline-impossible"
        return "impossible"
    except Exception:
        return ""

def _float(v, default=None):
    try:
        return float(v) if v not in ("", None) else default
    except (ValueError, TypeError):
        return default

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

all_rows = defaultdict(list)

with open(sweep_path, newline="") as f:
    for r in csv.DictReader(f):
        p = r.get("brake_profile", "")
        if p not in PROFILE_ORDER:
            continue
        all_rows[p].append({
            "collision":    r["collision_detected"] == "True",
            "outcome":      r["outcome"],
            "physics":      _physics_label(r),
            "speed_mph":    _float(r.get("target_mph")),
            "mean_jerk":    _float(r.get("mean_jerk_mps3")),
            "min_ped_dist": _float(r.get("min_ped_distance_m")),
            "time_to_stop": _float(r.get("time_to_stop_s")),
        })

with open(sac_path, newline="") as f:
    for r in csv.DictReader(f):
        all_rows[SAC_LABEL].append({
            "collision":    r["collision_detected"] == "True",
            "outcome":      r["outcome"],
            "physics":      r.get("physics_label") or _physics_label(r),
            "speed_mph":    _float(r.get("target_mph")),
            "mean_jerk":    _float(r.get("mean_jerk_mps3")),
            "min_ped_dist": _float(r.get("min_ped_distance_m")),
            "time_to_stop": _float(r.get("time_to_stop_s")),
        })

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("#f9f9f9")

gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.32,
                      left=0.07, right=0.97, top=0.88, bottom=0.10)
ax1 = fig.add_subplot(gs[0, 0])   # overall collision rate
ax2 = fig.add_subplot(gs[0, 1])   # full-stop rate by speed
ax3 = fig.add_subplot(gs[1, 0])   # avoidable-zone collision rate
ax4 = fig.add_subplot(gs[1, 1])   # comfort/safety metrics

x = np.arange(len(CONTROLLER_ORDER))
bar_w = 0.55

# ---------------------------------------------------------------------------
# Panel 1: Overall collision rate
# ---------------------------------------------------------------------------

col_rates = [
    100 * sum(1 for r in all_rows[c] if r["collision"]) / len(all_rows[c])
    for c in CONTROLLER_ORDER
]

bars = ax1.bar(x, col_rates, bar_w,
               color=[COLORS[c] for c in CONTROLLER_ORDER],
               alpha=0.88, edgecolor="white", linewidth=0.8)

for i, (bar, val) in enumerate(zip(bars, col_rates)):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
             f"{val:.0f}%", ha="center", va="bottom", fontsize=9,
             fontweight="bold", color=COLORS[CONTROLLER_ORDER[i]])

ax1.set_xticks(x)
ax1.set_xticklabels([DISPLAY_NAMES[c] for c in CONTROLLER_ORDER], fontsize=8)
ax1.get_xticklabels()[0].set_fontweight("bold")
ax1.set_ylabel("Collision rate (%)", fontsize=10)
ax1.set_title("Overall Collision Rate\n(all 200 matched scenarios)", fontsize=10, fontweight="bold")
ax1.set_ylim(0, 35)
ax1.axhline(col_rates[0], color=COLORS[SAC_LABEL], lw=0.8, ls="--", alpha=0.4)
ax1.grid(axis="y", alpha=0.3)
ax1.set_facecolor("white")

# ---------------------------------------------------------------------------
# Panel 2: Full-stop rate by speed (line chart)
# ---------------------------------------------------------------------------

speed_groups = [22.0, 28.0, 35.0, 40.0, 45.0]
speed_labels = ["22", "28", "35", "40", "45"]

for ctrl in CONTROLLER_ORDER:
    rows = all_rows[ctrl]
    rates = []
    for spd in speed_groups:
        sub = [r for r in rows if r["speed_mph"] is not None and abs(r["speed_mph"] - spd) < 0.5]
        rates.append(100 * sum(1 for r in sub if r["outcome"] == "full_stop") / len(sub) if sub else 0)
    lw  = 2.8 if ctrl == SAC_LABEL else 1.4
    zo  = 4   if ctrl == SAC_LABEL else 2
    ms  = 8   if ctrl == SAC_LABEL else 5
    ax2.plot(speed_labels, rates, color=COLORS[ctrl], lw=lw, zorder=zo,
             marker="o", markersize=ms, label=DISPLAY_NAMES[ctrl].replace("\n", " "))

ax2.set_ylabel("Full-stop rate (%)", fontsize=10)
ax2.set_xlabel("Target speed (mph)", fontsize=9)
ax2.set_title("Full-Stop Rate by Speed\n(SAC leads at high speed)", fontsize=10, fontweight="bold")
ax2.set_ylim(0, 110)
ax2.axhline(100, color="green", lw=0.7, ls="--", alpha=0.4)
ax2.grid(alpha=0.3)
ax2.set_facecolor("white")
ax2.legend(fontsize=7, loc="lower left", framealpha=0.9)

# ---------------------------------------------------------------------------
# Panel 3: Collision rate in avoidable + borderline-avoidable only
# ---------------------------------------------------------------------------

AVOID_LABELS = ["avoidable", "borderline-avoidable"]
avoid_bar_w  = 0.13
n_ctrl       = len(CONTROLLER_ORDER)
offsets      = np.linspace(-(n_ctrl - 1) * avoid_bar_w / 2,
                            (n_ctrl - 1) * avoid_bar_w / 2, n_ctrl)

for ci, ctrl in enumerate(CONTROLLER_ORDER):
    rows  = all_rows[ctrl]
    rates = []
    for lbl in AVOID_LABELS:
        sub = [r for r in rows if r["physics"] == lbl]
        rates.append(100 * sum(1 for r in sub if r["collision"]) / len(sub) if sub else 0)
    xi = np.arange(len(AVOID_LABELS)) + offsets[ci]
    ax3.bar(xi, rates, avoid_bar_w,
            color=COLORS[ctrl], alpha=0.88,
            edgecolor="white", linewidth=0.5,
            label=DISPLAY_NAMES[ctrl].replace("\n", " "))

# Sample counts
for li, lbl in enumerate(AVOID_LABELS):
    n = len([r for r in all_rows[SAC_LABEL] if r["physics"] == lbl])
    ax3.text(li, -6, f"n={n}", ha="center", fontsize=7.5, color="gray")

ax3.set_xticks(np.arange(len(AVOID_LABELS)))
ax3.set_xticklabels(["Avoidable\n(margin ≥ 8m)", "Borderline-avoidable\n(margin 0–8m)"], fontsize=9)
ax3.set_ylabel("Collision rate (%)", fontsize=10)
ax3.set_title("Collision Rate — Physically Avoidable Scenarios\n(where agent decisions matter most)",
              fontsize=10, fontweight="bold")
ax3.set_ylim(-12, 65)
ax3.axhline(0, color="black", lw=0.5)
ax3.grid(axis="y", alpha=0.3)
ax3.set_facecolor("white")
ax3.legend(fontsize=7, loc="upper right", ncol=2, framealpha=0.9)

# ---------------------------------------------------------------------------
# Panel 4: Comfort/safety metric dots
# ---------------------------------------------------------------------------

METRICS = [
    ("mean_jerk",    "noncollision", "Mean Jerk (m/s³)\nlower = smoother",       False),
    ("min_ped_dist", "all",          "Min Ped Distance (m)\nhigher = safer",      True),
    ("time_to_stop", "fullstop",     "Time to Full Stop (s)\nlower = faster stop", False),
]

def _mean(ctrl, key, subset):
    rows = all_rows[ctrl]
    if subset == "noncollision":
        rows = [r for r in rows if not r["collision"]]
    elif subset == "fullstop":
        rows = [r for r in rows if r["outcome"] == "full_stop"]
    vals = [r[key] for r in rows if r[key] is not None]
    return np.mean(vals) if vals else 0.0

metric_y   = np.arange(len(METRICS))
metric_bar_w = 0.12
m_offsets  = np.linspace(-(n_ctrl - 1) * metric_bar_w / 2,
                           (n_ctrl - 1) * metric_bar_w / 2, n_ctrl)

for ci, ctrl in enumerate(CONTROLLER_ORDER):
    vals = [_mean(ctrl, key, sub) for key, sub, _, _ in METRICS]
    # Normalize each metric 0-1 across controllers for visual comparison
    yi   = metric_y + m_offsets[ci]
    ax4.barh(yi, vals, metric_bar_w,
             color=COLORS[ctrl], alpha=0.88,
             edgecolor="white", linewidth=0.5,
             label=DISPLAY_NAMES[ctrl].replace("\n", " "))

# Annotate SAC values
sac_vals = [_mean(SAC_LABEL, key, sub) for key, sub, _, _ in METRICS]
for mi, val in enumerate(sac_vals):
    ax4.text(val + 0.1, metric_y[mi] + m_offsets[0],
             f"{val:.2f}", va="center", fontsize=7.5,
             color=COLORS[SAC_LABEL], fontweight="bold")

ax4.set_yticks(metric_y)
ax4.set_yticklabels([lbl for _, _, lbl, _ in METRICS], fontsize=8.5)
ax4.set_xlabel("Metric value (raw units)", fontsize=9)
ax4.set_title("Comfort & Safety Metrics\n(SAC value annotated in dark)",
              fontsize=10, fontweight="bold")
ax4.grid(axis="x", alpha=0.3)
ax4.set_facecolor("white")
ax4.legend(fontsize=7, loc="upper right", framealpha=0.9)

# ---------------------------------------------------------------------------
# Title and save
# ---------------------------------------------------------------------------

fig.suptitle(
    f"{SAC_LABEL.replace('_', ' ')} vs Hand-Designed Braking Profiles — Matched 200-Scenario Sweep",
    fontsize=13, fontweight="bold", y=0.95
)

note = ("All controllers evaluated on identical scenario configs "
        "(22–45 mph × TTC 1.8–3.5s × near/far walker × 60/120m encounter × 2 headways). "
        "Physics label based on a_eff = 3.5 m/s² ramp-limited deceleration.")
fig.text(0.5, 0.02, note, ha="center", fontsize=7.5, color="#666666")

out = os.path.join(sweep_dir, "presentation_summary.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved -> {out}")
