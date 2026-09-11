"""
plot_unified_comparison.py
Quentin | Junior Spring Research, 2026.

Unified head-to-head comparison of SAC vs the four hand-designed braking
profiles on IDENTICAL scenario configs (same 48 runs from sweep.py).

Data sources:
    sweep_summary.csv           — 4 profiles x 48 configs
    sac_on_sweep_results.csv    — SAC 800k x same 48 configs

Produces two figures:
    unified_outcomes.png        — outcome distribution + collision breakdown
    unified_by_label.png        — collision rate by physics label per controller

Usage:
    python plot_unified_comparison.py
    python plot_unified_comparison.py runs/20260405_025448/sweep_summary.csv
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
print(f"Sweep : {sweep_path}")
print(f"SAC   : {sac_path}")

# ---------------------------------------------------------------------------
# Physics label (inline — mirrors avoidability.py)
# ---------------------------------------------------------------------------

A_MAX        = 3.5
BORDERLINE_M = 8.0

def _label(target_mph, trigger_ttc_s, walker_speed_mps, walker_cross,
           walker_startup_s=0.5):
    FAR_LANE = 3.3
    v0  = float(target_mph) * 0.44704
    ttc = float(trigger_ttc_s) if trigger_ttc_s else (float(walker_startup_s) + 1.5)
    if str(walker_cross).lower() == "far":
        if float(walker_speed_mps) > 0 and ttc >= FAR_LANE / float(walker_speed_mps):
            return "avoidable"
    d_avail = v0 * ttc
    d_stop  = v0**2 / (2.0 * A_MAX)
    margin  = d_avail - d_stop
    av      = margin >= 0.0
    bl      = abs(margin) < BORDERLINE_M
    if av and bl:  return "borderline-avoidable"
    if av:         return "avoidable"
    if bl:         return "borderline-impossible"
    return "impossible"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

PROFILE_ORDER = ["proportional_ramp", "step_constant", "cautious_ramp", "exponential"]

# Auto-detect SAC controller label from sac_on_sweep_results.csv
def _detect_sac_label(sac_csv):
    try:
        with open(sac_csv, newline="") as f:
            r = next(csv.DictReader(f))
            lbl = r.get("brake_profile", "SAC_1600k")
            if lbl not in PROFILE_ORDER:
                return lbl
    except Exception:
        pass
    return "SAC_1600k"

SAC_LABEL = _detect_sac_label(sac_path)
CONTROLLER_ORDER = [SAC_LABEL] + PROFILE_ORDER

_step_label = SAC_LABEL.replace("SAC_", "").replace("k", "k steps")
CONTROLLER_LABELS = {
    SAC_LABEL:           f"SAC\n({_step_label})",
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

# Load profile sweep
profile_rows = defaultdict(list)
with open(sweep_path, newline="") as f:
    for r in csv.DictReader(f):
        p = r.get("brake_profile","")
        if p not in PROFILE_ORDER:
            continue
        r["physics_label"] = _label(
            r["target_mph"], r.get("trigger_ttc_s",""),
            r["walker_speed_mps"], r["walker_cross"],
            r.get("walker_startup_s", 0.5)
        )
        r["collision"] = (r["collision_detected"] == "True")
        r["controller"] = p
        profile_rows[p].append(r)

# Load SAC
sac_rows = []
with open(sac_path, newline="") as f:
    for r in csv.DictReader(f):
        r["collision"] = (r["collision_detected"] == "True")
        r["controller"] = SAC_LABEL
        sac_rows.append(r)

all_rows = {**profile_rows, SAC_LABEL: sac_rows}

print(f"\n{'Controller':<22} {'Total':>6} {'Collision':>10} {'Full stop':>10} {'Slowed':>8}")
print("-" * 60)
for c in CONTROLLER_ORDER:
    rows = all_rows[c]
    n    = len(rows)
    col  = sum(1 for r in rows if r["collision"])
    stop = sum(1 for r in rows if r["outcome"] == "full_stop")
    slow = sum(1 for r in rows if r["outcome"] == "slowed_avoided")
    print(f"  {c:<20} {n:>6} {col:>8} ({100*col/n:.0f}%) "
          f"{stop:>7} ({100*stop/n:.0f}%) {slow:>5} ({100*slow/n:.0f}%)")

# ---------------------------------------------------------------------------
# Figure 1: Outcome distribution + collision breakdown (2 panels side by side)
# ---------------------------------------------------------------------------

LABEL_ORDER  = ["avoidable", "borderline-avoidable", "borderline-impossible", "impossible"]
LABEL_SHORT  = {
    "avoidable":             "Avoidable",
    "borderline-avoidable":  "Borderline\navoidable",
    "borderline-impossible": "Borderline\nimpossible",
    "impossible":            "Impossible",
}
LABEL_COLORS = {
    "avoidable":             "#2ecc71",
    "borderline-avoidable":  "#f1c40f",
    "borderline-impossible": "#e67e22",
    "impossible":            "#e74c3c",
}

fig1, (ax_out, ax_lbl) = plt.subplots(1, 2, figsize=(14, 6))
x      = np.arange(len(CONTROLLER_ORDER))
width  = 0.55

# Left panel: stacked outcome bars
stop_vals  = [sum(1 for r in all_rows[c] if r["outcome"]=="full_stop")     for c in CONTROLLER_ORDER]
slow_vals  = [sum(1 for r in all_rows[c] if r["outcome"]=="slowed_avoided") for c in CONTROLLER_ORDER]
col_vals   = [sum(1 for r in all_rows[c] if r["collision"])                 for c in CONTROLLER_ORDER]
totals     = [len(all_rows[c]) for c in CONTROLLER_ORDER]

bars_stop = ax_out.bar(x, stop_vals, width, color="#2ecc71", label="Full stop")
bars_slow = ax_out.bar(x, slow_vals, width, bottom=stop_vals, color="#f39c12", label="Slowed & avoided")
bars_col  = ax_out.bar(x, col_vals,  width,
                       bottom=[s+l for s,l in zip(stop_vals,slow_vals)],
                       color="#e74c3c", label="Collision")

for i, (c, t) in enumerate(zip(col_vals, totals)):
    pct = 100 * c / t
    ax_out.text(i, t + 0.4, f"{pct:.0f}%\ncoll.", ha="center", va="bottom",
                fontsize=8, color="#c0392b", fontweight="bold")

ax_out.set_xticks(x)
ax_out.set_xticklabels([CONTROLLER_LABELS[c] for c in CONTROLLER_ORDER], fontsize=9)
n_per_ctrl = max(totals)
ax_out.set_ylabel(f"Episode count (out of {n_per_ctrl})", fontsize=11)
ax_out.set_title(f"Outcome Distribution — Matched {n_per_ctrl} Scenarios", fontsize=11, fontweight="bold")
ax_out.legend(fontsize=9, loc="lower right")
ax_out.set_ylim(0, max(totals) * 1.18)
ax_out.grid(axis="y", alpha=0.3)

# Highlight SAC bar
ax_out.get_xticklabels()[0].set_fontweight("bold")

# Right panel: collision rate per physics label per controller (grouped bars)
n_labels = len(LABEL_ORDER)
n_ctrl   = len(CONTROLLER_ORDER)
bar_w    = 0.13
offsets  = np.linspace(-(n_ctrl-1)*bar_w/2, (n_ctrl-1)*bar_w/2, n_ctrl)

for ci, ctrl in enumerate(CONTROLLER_ORDER):
    rows = all_rows[ctrl]
    rates = []
    for lbl in LABEL_ORDER:
        sub = [r for r in rows if r.get("physics_label","") == lbl]
        rates.append(100 * sum(1 for r in sub if r["collision"]) / len(sub) if sub else 0)
    xi = np.arange(n_labels) + offsets[ci]
    bars = ax_lbl.bar(xi, rates, bar_w,
                      color=CONTROLLER_COLORS[ctrl],
                      label=CONTROLLER_LABELS[ctrl].replace("\n", " "),
                      alpha=0.88, edgecolor="white", linewidth=0.5)

# Sample counts under each label
for li, lbl in enumerate(LABEL_ORDER):
    n = len([r for r in all_rows[SAC_LABEL] if r.get("physics_label","") == lbl])
    ax_lbl.text(li, -7, f"n={n}", ha="center", fontsize=7.5, color="gray")

ax_lbl.set_xticks(np.arange(n_labels))
ax_lbl.set_xticklabels([LABEL_SHORT[l] for l in LABEL_ORDER], fontsize=9)
ax_lbl.set_ylabel("Collision rate (%)", fontsize=11)
ax_lbl.set_title("Collision Rate by Physics Label — All Controllers", fontsize=11, fontweight="bold")
ax_lbl.set_ylim(-12, 115)
ax_lbl.axhline(0, color="black", linewidth=0.5)
ax_lbl.legend(fontsize=7.5, loc="upper left", ncol=2)
ax_lbl.grid(axis="y", alpha=0.3)

note = (f"n={n_per_ctrl} matched scenarios per controller "
        "(22/28/35/40/45 mph × TTC 1.8/2.2/2.6/3.0/3.5 s × near/far walker × 60/120 m encounter × 2 headways). "
        "Physics label = avoidability given a_eff=3.5 m/s² (ramp-limited). "
        "All controllers share the same scenario set.")
fig1.text(0.5, 0.01, note, ha="center", fontsize=7.5, color="#555555")
fig1.suptitle(f"{SAC_LABEL.replace('_', ' ')} vs Hand-Designed Braking Profiles — Identical Scenario Conditions",
              fontsize=13, fontweight="bold")
fig1.tight_layout(rect=[0, 0.045, 1, 1])

out1 = os.path.join(sweep_dir, "unified_outcomes.png")
fig1.savefig(out1, dpi=150)
print(f"\nSaved -> {out1}")

# ---------------------------------------------------------------------------
# Figure 2: Full-stop rate (comfort / effectiveness proxy) per controller
# across speed groups — cleaner single-panel summary
# ---------------------------------------------------------------------------

fig2, ax2 = plt.subplots(figsize=(10, 5))

speed_groups = [(22.0, "22 mph"), (28.0, "28 mph"), (35.0, "35 mph"),
                (40.0, "40 mph"), (45.0, "45 mph")]
n_speeds = len(speed_groups)
bar_w2   = 0.12
offsets2 = np.linspace(-(n_ctrl-1)*bar_w2/2, (n_ctrl-1)*bar_w2/2, n_ctrl)

for ci, ctrl in enumerate(CONTROLLER_ORDER):
    rows = all_rows[ctrl]
    stop_rates = []
    for spd, _ in speed_groups:
        sub = [r for r in rows if abs(float(r["target_mph"]) - spd) < 0.5]
        stop_rates.append(100 * sum(1 for r in sub if r["outcome"]=="full_stop") / len(sub) if sub else 0)
    xi = np.arange(n_speeds) + offsets2[ci]
    ax2.bar(xi, stop_rates, bar_w2,
            color=CONTROLLER_COLORS[ctrl],
            label=CONTROLLER_LABELS[ctrl].replace("\n"," "),
            alpha=0.88, edgecolor="white", linewidth=0.5)

# n= count per speed (same for all controllers)
for si, (spd, _) in enumerate(speed_groups):
    n = len([r for r in all_rows[SAC_LABEL] if abs(float(r["target_mph"]) - spd) < 0.5])
    ax2.text(si, -4, f"n={n}", ha="center", fontsize=7.5, color="gray")

ax2.set_xticks(np.arange(n_speeds))
ax2.set_xticklabels([s for _, s in speed_groups], fontsize=11)
ax2.set_ylabel("Full-stop rate (%)", fontsize=11)
ax2.set_title("Full-Stop Rate by Speed — All Controllers on Matched Scenarios\n"
              "(Full stop = ego fully stopped before reaching pedestrian — ideal outcome)",
              fontsize=11, fontweight="bold")
ax2.set_ylim(-10, 115)
ax2.legend(fontsize=8.5, loc="upper right", ncol=2)
ax2.grid(axis="y", alpha=0.3)
ax2.axhline(0, color="black", linewidth=0.5)
ax2.axhline(100, color="green", linewidth=0.8, linestyle="--", alpha=0.5)

note2 = ("n= shows unique scenario configs per speed (5 TTC values × near/far walker × 60/120 m encounter × 2 headways = 40 per speed). "
         "Full-stop rate tends to drop at higher speeds as more scenarios enter the borderline-impossible zone.")
fig2.text(0.5, 0.01, note2, ha="center", fontsize=7.5, color="#555555")
fig2.tight_layout(rect=[0, 0.05, 1, 1])

out2 = os.path.join(sweep_dir, "unified_fullstop_by_speed.png")
fig2.savefig(out2, dpi=150)
print(f"Saved -> {out2}")
