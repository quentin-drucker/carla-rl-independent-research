"""
plot_comfort_safety.py
Quentin | Junior Spring Research, 2026.

Compares SAC 1600k vs the four hand-designed braking profiles across
three dimensions on the matched 200-scenario sweep:
  - Comfort     : mean jerk (m/s³) during braking — lower = smoother
  - Safety margin: min pedestrian distance (m) — higher = more buffer
  - Response     : time to full stop (s) — lower = faster, but trades comfort

Produces two figures:
  comfort_safety_distributions.png  — violin + strip plots, 3 metrics × 5 controllers
  comfort_safety_radar.png          — radar chart normalizing all metrics per controller

Usage:
    python plot_comfort_safety.py
    python plot_comfort_safety.py runs/20260417_202355/sweep_summary.csv
"""

import os, sys, csv, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ---------------------------------------------------------------------------
# Locate files (mirrors plot_unified_comparison.py)
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
print(f"Sweep : {sweep_path}")
print(f"SAC   : {sac_path}")

# ---------------------------------------------------------------------------
# Controller order / colors / labels
# ---------------------------------------------------------------------------

PROFILE_ORDER = ["proportional_ramp", "step_constant", "cautious_ramp", "exponential"]

def _detect_sac_label(path):
    try:
        with open(path, newline="") as f:
            r = next(csv.DictReader(f))
            lbl = r.get("brake_profile", "SAC_1600k")
            if lbl not in PROFILE_ORDER:
                return lbl
    except Exception:
        pass
    return "SAC_1600k"

SAC_LABEL = _detect_sac_label(sac_path)
CONTROLLER_ORDER = [SAC_LABEL] + PROFILE_ORDER

CONTROLLER_LABELS = {
    SAC_LABEL:           f"SAC\n({SAC_LABEL.replace('SAC_','')} steps)",
    "proportional_ramp": "Proportional\nRamp",
    "step_constant":     "Step\nConstant",
    "cautious_ramp":     "Cautious\nRamp",
    "exponential":       "Exponential",
}
CONTROLLER_COLORS = {
    SAC_LABEL:           "#1a1a2e",
    "proportional_ramp": "#4C72B0",
    "step_constant":     "#DD8452",
    "cautious_ramp":     "#55A868",
    "exponential":       "#C44E52",
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def _float(val, default=None):
    try:
        return float(val) if val not in ("", None) else default
    except (ValueError, TypeError):
        return default

# Inline physics label — sweep_summary.csv doesn't store this column
A_MAX = 3.5
BORDERLINE_M = 8.0

def _physics_label(r):
    try:
        v0  = float(r["target_mph"]) * 0.44704
        ttc = float(r["trigger_ttc_s"]) if r.get("trigger_ttc_s") else (float(r.get("walker_startup_s", 0.5)) + 1.5)
        if str(r.get("walker_cross", "")).lower() == "far":
            wspd = float(r.get("walker_speed_mps", 0))
            if wspd > 0 and ttc >= 3.3 / wspd:
                return "avoidable"
        margin = v0 * ttc - v0**2 / (2.0 * A_MAX)
        av = margin >= 0.0
        bl = abs(margin) < BORDERLINE_M
        if av and bl:  return "borderline-avoidable"
        if av:         return "avoidable"
        if bl:         return "borderline-impossible"
        return "impossible"
    except Exception:
        return ""

all_rows = defaultdict(list)

with open(sweep_path, newline="") as f:
    for r in csv.DictReader(f):
        p = r.get("brake_profile", "")
        if p not in PROFILE_ORDER:
            continue
        all_rows[p].append({
            "collision":       r["collision_detected"] == "True",
            "outcome":         r["outcome"],
            "mean_jerk":       _float(r.get("mean_jerk_mps3")),
            "max_jerk":        _float(r.get("max_jerk_mps3")),
            "min_ped_dist":    _float(r.get("min_ped_distance_m")),
            "time_to_stop":    _float(r.get("time_to_stop_s")),
            "physics_label":   _physics_label(r),
        })

with open(sac_path, newline="") as f:
    for r in csv.DictReader(f):
        all_rows[SAC_LABEL].append({
            "collision":       r["collision_detected"] == "True",
            "outcome":         r["outcome"],
            "mean_jerk":       _float(r.get("mean_jerk_mps3")),
            "max_jerk":        _float(r.get("max_jerk_mps3")),
            "min_ped_dist":    _float(r.get("min_ped_distance_m")),
            "time_to_stop":    _float(r.get("time_to_stop_s")),
            "physics_label":   r.get("physics_label", ""),
        })

# ---------------------------------------------------------------------------
# Extract per-controller metric arrays
# (only non-collision episodes for jerk/time; all episodes for ped distance)
# ---------------------------------------------------------------------------

def _vals(ctrl, key, subset="all"):
    rows = all_rows[ctrl]
    if subset == "noncollision":
        rows = [r for r in rows if not r["collision"]]
    elif subset == "fullstop":
        rows = [r for r in rows if r["outcome"] == "full_stop"]
    return [r[key] for r in rows if r[key] is not None]

np.random.seed(42)

# ---------------------------------------------------------------------------
# Figure 1: Distribution plots — 3 metrics, violin + strip
# ---------------------------------------------------------------------------

fig1, axes = plt.subplots(1, 3, figsize=(15, 6))
fig1.suptitle(
    f"Comfort, Safety Margin & Response — {SAC_LABEL.replace('_',' ')} vs Hand-Designed Profiles\n"
    "Matched 200-scenario sweep (identical conditions for all controllers)",
    fontsize=12, fontweight="bold"
)

METRICS = [
    ("mean_jerk",    "noncollision", "Mean Braking Jerk (m/s³)",
     "Lower = smoother, more comfortable braking\n(non-collision episodes only)",
     False),   # lower_better
    ("min_ped_dist", "all",          "Min Pedestrian Distance (m)",
     "Higher = greater safety buffer to pedestrian\n(all episodes)",
     True),
    ("time_to_stop", "fullstop",     "Time to Full Stop (s)",
     "Lower = faster emergency stop\n(full-stop episodes only)",
     False),
]

for ax, (key, subset, ylabel, subtitle, higher_better) in zip(axes, METRICS):
    positions = np.arange(len(CONTROLLER_ORDER))

    for ci, ctrl in enumerate(CONTROLLER_ORDER):
        vals = _vals(ctrl, key, subset)
        if not vals:
            continue
        arr = np.array(vals)
        col = CONTROLLER_COLORS[ctrl]

        # Violin
        vp = ax.violinplot([arr], positions=[ci], widths=0.6,
                           showmedians=False, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(col)
            body.set_alpha(0.3)
            body.set_edgecolor(col)

        # IQR box
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        ax.vlines(ci, q1, q3, color=col, linewidth=5, alpha=0.6, zorder=3)
        ax.scatter([ci], [med], color="white", s=40, zorder=4,
                   edgecolors=col, linewidths=1.5)

        # Mean marker
        ax.scatter([ci], [arr.mean()], color=col, s=60, marker="D",
                   zorder=5, edgecolors="white", linewidths=0.8)

        # Jitter strip
        jx = np.random.uniform(-0.15, 0.15, len(arr))
        ax.scatter(np.full(len(arr), ci) + jx, arr,
                   color=col, s=12, alpha=0.35, zorder=2)

        # Mean annotation
        ax.text(ci, arr.max() + (arr.max() - arr.min()) * 0.04,
                f"μ={arr.mean():.1f}", ha="center", fontsize=7,
                color=col, fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels([CONTROLLER_LABELS[c] for c in CONTROLLER_ORDER],
                       fontsize=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(subtitle, fontsize=9, color="#444444")
    ax.grid(axis="y", alpha=0.3)
    ax.get_xticklabels()[0].set_fontweight("bold")  # highlight SAC

# Shared legend
legend_elements = [
    plt.Line2D([0], [0], marker="D", color="gray", markersize=7,
               linestyle="None", label="Mean"),
    plt.Line2D([0], [0], marker="o", color="white", markersize=6,
               markeredgecolor="gray", linestyle="None", label="Median"),
    mpatches.Patch(facecolor="gray", alpha=0.5, label="IQR (25th–75th %ile)"),
    mpatches.Patch(facecolor="gray", alpha=0.25, label="Distribution (violin)"),
    plt.Line2D([0], [0], marker="o", color="gray", markersize=4,
               alpha=0.4, linestyle="None", label="Individual episodes (jittered)"),
]
fig1.legend(handles=legend_elements, loc="lower center", ncol=5,
            fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
fig1.tight_layout(rect=[0, 0.07, 1, 1])

out1 = os.path.join(sweep_dir, "comfort_safety_distributions.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved -> {out1}")

# ---------------------------------------------------------------------------
# Figure 2: Radar chart — 5 normalized metrics per controller
# ---------------------------------------------------------------------------

def _collision_rate(ctrl):
    rows = all_rows[ctrl]
    return sum(1 for r in rows if r["collision"]) / len(rows) if rows else 1.0

def _avoidable_safety(ctrl):
    rows = [r for r in all_rows[ctrl]
            if r["physics_label"] in ("avoidable", "borderline-avoidable")]
    if not rows:
        return 0.0
    return 1 - sum(1 for r in rows if r["collision"]) / len(rows)

def _mean_val(ctrl, key, subset="all"):
    v = _vals(ctrl, key, subset)
    return np.mean(v) if v else 0.0

# Raw metric values per controller
raw = {
    "safety":          {c: 1 - _collision_rate(c)               for c in CONTROLLER_ORDER},
    "avoidable_perf":  {c: _avoidable_safety(c)                  for c in CONTROLLER_ORDER},
    "ped_margin":      {c: _mean_val(c, "min_ped_dist", "all")   for c in CONTROLLER_ORDER},
    "comfort":         {c: 1 / (1 + _mean_val(c, "mean_jerk", "noncollision")) for c in CONTROLLER_ORDER},
    "response":        {c: 1 / (0.01 + _mean_val(c, "time_to_stop", "fullstop")) for c in CONTROLLER_ORDER},
}

RADAR_AXES = [
    ("safety",         "Overall\nSafety\n(1−collision rate)"),
    ("avoidable_perf", "Avoidable\nZone\nPerformance"),
    ("ped_margin",     "Pedestrian\nMargin\n(mean min dist)"),
    ("comfort",        "Braking\nComfort\n(1/(1+jerk))"),
    ("response",       "Response\nSpeed\n(1/time-to-stop)"),
]

# Normalize each axis 0→1 across controllers
def _normalize(metric_dict):
    vals = list(metric_dict.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 0.5 for k in metric_dict}
    return {k: (v - lo) / (hi - lo) for k, v in metric_dict.items()}

norm = {ax_key: _normalize(raw[ax_key]) for ax_key, _ in RADAR_AXES}

N = len(RADAR_AXES)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the polygon

fig2, ax_r = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

for ctrl in CONTROLLER_ORDER:
    values = [norm[ax_key][ctrl] for ax_key, _ in RADAR_AXES]
    values += values[:1]
    col = CONTROLLER_COLORS[ctrl]
    ax_r.plot(angles, values, color=col, linewidth=2, label=CONTROLLER_LABELS[ctrl].replace("\n", " "))
    ax_r.fill(angles, values, color=col, alpha=0.08)
    ax_r.scatter(angles[:-1], values[:-1], color=col, s=55, zorder=5,
                 edgecolors="white", linewidths=0.8)

# Axis labels
ax_r.set_xticks(angles[:-1])
ax_r.set_xticklabels([lbl for _, lbl in RADAR_AXES], fontsize=9)
ax_r.set_ylim(0, 1)
ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
ax_r.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="gray")
ax_r.grid(color="gray", alpha=0.3)

ax_r.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
            fontsize=9, framealpha=0.9)

note_r = ("All axes normalized 0–1 across controllers (outer edge = best on that metric). "
          "Larger polygon area = better overall. Metrics computed on matched 200-scenario sweep.")
fig2.text(0.5, 0.01, note_r, ha="center", fontsize=8, color="#555555")
fig2.suptitle(
    f"Controller Comparison — Comfort, Safety & Response\n"
    f"{SAC_LABEL.replace('_',' ')} vs Hand-Designed Braking Profiles",
    fontsize=12, fontweight="bold", y=1.01
)
fig2.tight_layout()

out2 = os.path.join(sweep_dir, "comfort_safety_radar.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved -> {out2}")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

print(f"\n{'Controller':<22} {'Collision%':>10} {'MeanJerk':>10} {'MinPedDist':>12} "
      f"{'TimeToStop':>12} {'AvoidPerf':>10}")
print("-" * 80)
for c in CONTROLLER_ORDER:
    cr   = 100 * _collision_rate(c)
    jerk = _mean_val(c, "mean_jerk", "noncollision")
    ped  = _mean_val(c, "min_ped_dist", "all")
    tts  = _mean_val(c, "time_to_stop", "fullstop")
    avd  = 100 * _avoidable_safety(c)
    print(f"  {c:<20} {cr:>9.1f}% {jerk:>10.2f} {ped:>12.2f} {tts:>12.2f} {avd:>9.1f}%")
