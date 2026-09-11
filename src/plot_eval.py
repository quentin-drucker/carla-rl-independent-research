"""
plot_eval.py
Quentin | Junior Spring Research, 2026.

Reads eval_results.csv (produced by eval_sac.py) and generates four figures
showing what the SAC agent is doing across the randomized evaluation scenarios.

Usage:
    python plot_eval.py                  <- reads eval_results.csv
    python plot_eval.py my_results.csv   <- reads a specific file

Physics labels used: avoidable, borderline-avoidable, borderline-impossible, impossible.

Figures produced:
    1. speed_ttc_scatter.png  — Speed vs TTC scatter; each point = one episode,
                                colored by outcome, physics boundary overlaid
    2. outcome_by_label.png   — Stacked bar: collision vs safe per physics label
    3. reward_by_label.png    — Box/strip plot of reward distribution per label
    4. success_by_speed.png   — Agent success rate across speed bands
"""

import sys
import os
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ── Load CSV ────────────────────────────────────────────────────────────────
csv_path = sys.argv[1] if len(sys.argv) > 1 else "eval_results.csv"
out_prefix = os.path.splitext(os.path.basename(csv_path))[0] + "_"
rows = []
with open(csv_path, newline="") as f:
    for r in csv.DictReader(f):
        rows.append({
            "episode":       int(r["episode"]),
            "reward":        float(r["reward"]),
            "collision":     int(r["collision"]),
            "full_stop":     int(r["full_stop"]),
            "speed_mph":     float(r["speed_mph"]),
            "ttc_s":         float(r["ttc_s"]) if r["ttc_s"] else None,
            "walker_cross":  r["walker_cross"],
            "physics_label": r["physics_label"],
            "margin_m":      float(r["margin_m"]) if r["margin_m"] != "inf" else float("inf"),
        })

print(f"Loaded {len(rows)} episodes from {csv_path}")

# ── Color scheme ─────────────────────────────────────────────────────────────
LABEL_COLORS = {
    "avoidable":             "#2ecc71",   # green
    "borderline-avoidable":  "#f1c40f",   # yellow
    "borderline-impossible": "#e67e22",   # orange
    "impossible":            "#e74c3c",   # red
}
OUTCOME_COLORS = {
    "safe":      "#2ecc71",
    "collision": "#e74c3c",
}

# ── Figure 1: Speed vs TTC scatter with avoidability boundary ────────────────
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Marker scheme: shape = pedestrian mode, fill = outcome
# near + safe: filled circle  near + collision: X
# far  + safe: filled square  far  + collision: filled diamond
MARKER_MAP = {
    ("near", False): ("o",  55,  1.0),   # circle, filled
    ("near", True):  ("x",  75,  1.5),   # X (collision)
    ("far",  False): ("s",  55,  1.0),   # square, filled
    ("far",  True):  ("D",  55,  1.0),   # diamond (collision)
}

for r in rows:
    if r["ttc_s"] is None:
        continue
    cross  = r["walker_cross"]
    col    = r["collision"]
    color  = LABEL_COLORS.get(r["physics_label"], "#888888")
    mk, sz, lw = MARKER_MAP.get((cross, bool(col)), ("o", 50, 1.0))
    ec = "none" if mk not in ("x", "+") else color
    ax1.scatter(r["ttc_s"], r["speed_mph"], c=color, marker=mk,
                s=sz, linewidths=lw, edgecolors=ec, zorder=3)

# Physics boundary
A_MAX = 3.5
speeds_mph = np.linspace(20, 55, 200)
ttc_boundary = (speeds_mph * 0.44704) / (2.0 * A_MAX)
ax1.plot(ttc_boundary, speeds_mph, "k--", linewidth=1.5, label="Physics limit (a_max=3.5 m/s²)")
ax1.fill_betweenx(speeds_mph, 0, ttc_boundary, alpha=0.08, color="red", label="Impossible zone")

BORDERLINE_M = 8.0
ttc_upper = ttc_boundary + BORDERLINE_M / (speeds_mph * 0.44704)
ttc_lower = ttc_boundary - BORDERLINE_M / (speeds_mph * 0.44704)
ax1.fill_betweenx(speeds_mph, ttc_lower, ttc_upper,
                  alpha=0.12, color="orange", label=f"Borderline band (±{BORDERLINE_M:.0f}m)")

# Legend — shape section
ax1.scatter([], [], c="gray", marker="o", s=50, label="Near-cross, safe")
ax1.scatter([], [], c="gray", marker="x", s=70, linewidths=1.5, label="Near-cross, collision")
ax1.scatter([], [], c="gray", marker="s", s=50, label="Far-cross, safe")
ax1.scatter([], [], c="gray", marker="D", s=50, label="Far-cross, collision")
# Legend — color section
for lbl, col in LABEL_COLORS.items():
    ax1.scatter([], [], c=col, marker="o", s=50, label=lbl)

ax1.set_xlabel("Trigger TTC (s)", fontsize=12)
ax1.set_ylabel("Ego Speed (mph)", fontsize=12)
ax1.set_title("SAC Agent: Speed vs TTC — Outcomes and Physics Boundary\n"
              "Shape = pedestrian mode (circle/X = near, square/diamond = far)  |  Color = physics label",
              fontsize=10)
ax1.legend(fontsize=7, loc="upper right", framealpha=0.88,
           handlelength=1.2, handletextpad=0.5, borderpad=0.6, labelspacing=0.3)
ax1.set_xlim(1.0, 4.0)
ax1.set_ylim(18, 57)
ax1.grid(alpha=0.3)
fig1.tight_layout()
fig1.savefig(f"{out_prefix}speed_ttc_scatter.png", dpi=150)
print(f"Saved {out_prefix}speed_ttc_scatter.png")

# ── Figure 2: Collision vs safe normalized bar by physics label ───────────────
label_order = ["avoidable", "borderline-avoidable", "borderline-impossible", "impossible"]
label_totals   = defaultdict(int)
label_collide  = defaultdict(int)
for r in rows:
    lbl = r["physics_label"]
    label_totals[lbl]  += 1
    label_collide[lbl] += r["collision"]

present_labels = [l for l in label_order if label_totals[l] > 0]
totals   = [label_totals[l]  for l in present_labels]
collides = [label_collide[l] for l in present_labels]
safes    = [t - c for t, c in zip(totals, collides)]

# Normalize to percentage so all bars are uniform height
safe_pcts = [100 * s / t if t > 0 else 0 for s, t in zip(safes, totals)]
coll_pcts = [100 * c / t if t > 0 else 0 for c, t in zip(collides, totals)]

fig2, ax2 = plt.subplots(figsize=(9, 5))
x = np.arange(len(present_labels))
bars_s = ax2.bar(x, safe_pcts, color=OUTCOME_COLORS["safe"],      label="Safe (no collision)")
bars_c = ax2.bar(x, coll_pcts, bottom=safe_pcts, color=OUTCOME_COLORS["collision"], label="Collision")

# Annotate: collision rate always shown above bar in red, n= inside safe section
for i, (sp, cp, t) in enumerate(zip(safe_pcts, coll_pcts, totals)):
    ax2.text(i, 102, f"{cp:.0f}% collision rate", ha="center", va="bottom",
             fontsize=8, color="#c0392b", fontweight="bold")
    if cp > 5:
        ax2.text(i, sp + cp / 2, f"{cp:.0f}%", ha="center", va="center",
                 fontsize=9, color="white", fontweight="bold")
    ax2.text(i, sp / 2, f"n={t}", ha="center", va="center",
             fontsize=8, color="white", fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(present_labels, rotation=15, ha="right", fontsize=10)
ax2.set_ylabel("% of episodes", fontsize=12)
ax2.set_ylim(0, 110)
ax2.set_title("SAC Agent: Outcome by Physics Label", fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(axis="y", alpha=0.3)
fig2.tight_layout()
fig2.savefig(f"{out_prefix}outcome_by_label.png", dpi=150)
print(f"Saved {out_prefix}outcome_by_label.png")

# ── Figure 3: Reward distribution by physics label — split safe vs collision ───
# Collision rewards are always ~-200 (fixed penalty); safe rewards carry the
# nuance. We separate them so both bands are legible on the same plot.
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5),
                                   gridspec_kw={"width_ratios": [1, 1]})

np.random.seed(0)  # reproducible jitter
SAFE_COL = "#2ecc71"
COLL_COL = "#e74c3c"
MEAN_SAFE = "#1a8a4a"
MEAN_COLL = "#a93226"

for i, lbl in enumerate(present_labels):
    lbl_color = LABEL_COLORS.get(lbl, "#888")
    safe_r = [r["reward"] for r in rows if r["physics_label"] == lbl and not r["collision"]]
    coll_r = [r["reward"] for r in rows if r["physics_label"] == lbl and r["collision"]]

    # Alternating column background
    for ax in (ax3a, ax3b):
        if i % 2 == 0:
            ax.axvspan(i - 0.45, i + 0.45, color="#f5f5f5", zorder=0)

    # Left panel: safe episode rewards
    if safe_r:
        arr = np.array(safe_r)
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        # IQR box behind dots
        ax3a.fill_between([i - 0.28, i + 0.28], q1, q3,
                          color=lbl_color, alpha=0.18, zorder=1)
        ax3a.hlines(med, i - 0.28, i + 0.28, colors=lbl_color, linewidth=1.5,
                    linestyle="--", alpha=0.7, zorder=2)
        jx = np.random.uniform(-0.18, 0.18, len(arr))
        ax3a.scatter(np.full(len(arr), i) + jx, arr,
                     c=SAFE_COL, s=32, alpha=0.8, zorder=3, edgecolors="white", linewidths=0.4)
        # Mean line
        ax3a.hlines(arr.mean(), i - 0.32, i + 0.32,
                    colors=MEAN_SAFE, linewidth=2.8, zorder=4)
        ax3a.text(i, 23, f"n={len(arr)}\nμ={arr.mean():.1f}",
                  ha="center", va="top", fontsize=7.5, color="#1a6a35")

    # Right panel: collision episode rewards
    if coll_r:
        arr = np.array(coll_r)
        jx = np.random.uniform(-0.18, 0.18, len(arr))
        ax3b.fill_between([i - 0.28, i + 0.28],
                          arr.min(), arr.max(),
                          color=lbl_color, alpha=0.15, zorder=1)
        ax3b.scatter(np.full(len(arr), i) + jx, arr,
                     c=COLL_COL, s=32, alpha=0.8, zorder=3, edgecolors="white", linewidths=0.4)
        ax3b.hlines(arr.mean(), i - 0.32, i + 0.32,
                    colors=MEAN_COLL, linewidth=2.8, zorder=4)
        ax3b.text(i, -285, f"n={len(arr)}\nμ={arr.mean():.0f}",
                  ha="center", va="top", fontsize=7.5, color="#7b241c")
    else:
        ax3b.text(i, -305, "n=0", ha="center", va="center",
                  fontsize=8, color="gray", style="italic")

for ax in (ax3a, ax3b):
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(range(len(present_labels)))
    ax.set_xticklabels(present_labels, rotation=15, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(-0.5, len(present_labels) - 0.5)

ax3a.set_ylabel("Episode reward", fontsize=11)
ax3a.set_title("Safe episodes (no collision)", fontsize=11, fontweight="bold")
ax3b.set_ylabel("Episode reward", fontsize=11)
ax3b.set_title("Collision episodes", fontsize=11, fontweight="bold")
ax3a.set_ylim(-15, 25)
ax3b.set_ylim(-325, 15)

# Shared legend panel — explain every visual element
legend_elements = [
    mpatches.Patch(facecolor=SAFE_COL, alpha=0.85, label="Safe episode (dot = one episode)"),
    mpatches.Patch(facecolor=COLL_COL, alpha=0.85, label="Collision episode (dot = one episode)"),
    plt.Line2D([0], [0], color=MEAN_SAFE, linewidth=2.5, label="Mean reward (safe)"),
    plt.Line2D([0], [0], color=MEAN_COLL, linewidth=2.5, label="Mean reward (collision)"),
    plt.Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--", alpha=0.7,
               label="Median reward (safe panel only)"),
    mpatches.Patch(facecolor="#aaaaaa", alpha=0.2,
                   label="IQR box — middle 50% of safe rewards\n(Q1–Q3, colored by physics label)"),
    mpatches.Patch(facecolor="#aaaaaa", alpha=0.15,
                   label="Range band — min to max of collision rewards"),
    plt.Line2D([0], [0], color="gray", linewidth=0.8, linestyle="--", alpha=0.6,
               label="Reward = 0 reference line"),
]
fig3.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=8,
            framealpha=0.92, edgecolor="#cccccc",
            bbox_to_anchor=(0.5, -0.01),
            title="How to read this chart", title_fontsize=9)

note = ("Left panel y-axis: −15 to +25  (safe reward detail visible).  "
        "Right panel y-axis: −325 to +15  (collision penalty ≈ −300 always).\n"
        "Dots are jittered horizontally for visibility.  n= and μ= annotations per group shown above each column.")
fig3.text(0.5, -0.13, note, ha="center", fontsize=7.5, color="#555555",
          transform=fig3.transFigure)

fig3.suptitle("SAC Agent: Reward Distribution by Physics Label", fontsize=13, fontweight="bold")
fig3.tight_layout()
fig3.subplots_adjust(bottom=0.32)
fig3.savefig(f"{out_prefix}reward_by_label.png", dpi=150, bbox_inches="tight")
print(f"Saved {out_prefix}reward_by_label.png")

# ── Figure 4: Agent success rate by speed band ────────────────────────────────
# Only count episodes where a stop was physically possible (avoidable or borderline)
# "Success" = no collision
fig4, ax4 = plt.subplots(figsize=(8, 5))

bands = [(20, 30, "<30 mph"), (30, 40, "30-40 mph"), (40, 50, "40-50 mph"), (50, 60, "50+ mph")]
band_labels, success_rates, episode_counts = [], [], []

for lo, hi, lbl in bands:
    subset = [r for r in rows
              if lo <= r["speed_mph"] < hi
              and r["physics_label"] in ("avoidable", "borderline-avoidable")]
    if not subset:
        continue
    n_safe = sum(1 for r in subset if not r["collision"])
    band_labels.append(lbl)
    success_rates.append(100 * n_safe / len(subset))
    episode_counts.append(len(subset))

x = np.arange(len(band_labels))
bars = ax4.bar(x, success_rates, color="#3498db", alpha=0.8, edgecolor="black")
ax4.bar_label(bars, labels=[f"{v:.0f}%" for v in success_rates],
              padding=3, fontsize=11, fontweight="bold")
for i, n in enumerate(episode_counts):
    # Place n= inside the bar near the bottom, white text for contrast
    ax4.text(i, 4, f"n={n}", ha="center", va="bottom", fontsize=9,
             color="white", fontweight="bold")

ax4.set_xticks(x)
ax4.set_xticklabels(band_labels, fontsize=12)
ax4.set_ylim(0, 115)
ax4.set_ylabel("Success rate (% no collision)", fontsize=12)
ax4.set_title("SAC Agent: Success Rate on Avoidable Scenarios by Speed Band", fontsize=13)
ax4.axhline(100, color="green", linewidth=0.8, linestyle="--", alpha=0.5)
ax4.grid(axis="y", alpha=0.3)
fig4.tight_layout()
fig4.savefig(f"{out_prefix}success_by_speed.png", dpi=150)
print(f"Saved {out_prefix}success_by_speed.png")

print("\nAll figures saved. Run: python plot_eval.py")
