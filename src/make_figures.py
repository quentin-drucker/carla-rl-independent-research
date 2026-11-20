"""make_figures.py

Regenerates the final-presentation results figures from the circuit evaluation logs, so they can be rebuilt
after every retrain (Week 10 start, 2026-11-09). Every number is read from each run's hazards.csv
(ego_safe_success, the run's own "safe (v4)" score; checked against the run's SUMMARY line).

Each controller is a set of run_circuit.py logs named <Town>_p<plan>_<tag>.log in artifacts/runs/circuit/<folder>
(the w9/w10 queue scripts write them). A seed counts only when all 15 plans (Town10HD p0-5, Town05 p0-3,
Town02 p0-4) have finished; a controller with no finished seed is left out, so this runs at any time.
To add a controller: one CONTROLLERS line.

Usage (from src/): ../venv/Scripts/python.exe make_figures.py
Output: artifacts/figures/ (circuit_results.csv, fig_safe_overall.png, fig_safe_by_town.png)
"""
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from project_paths import RUNS_DIR  # noqa: E402

CIRCUIT_DIR = RUNS_DIR / "circuit"
FIG_DIR = RUNS_DIR.parent / "figures"
PLANS = {"Town05": range(4), "Town02": range(5), "Town10HD": range(6)}
TOWN_LABEL = {"Town05": "Town05 (trained on)", "Town02": "Town02 (never seen, narrow)",
              "Town10HD": "Town10HD (never seen)"}

# (label, folder, log tag with {s} for the seed, seeds); the rule is the reference line, not a row.
RULE = ("Rule: full brake + swerve to the roomier side", "w9_sidebal", "rule", [None])
CONTROLLERS = [
    ("First multi-town models", "w9_sidebal", "mixed_s{s}", [1, 2, 3]),
    ("+ map-gap fix (control)", "w9_overnight", "ctrl_s{s}", [1, 2, 3]),
    ("+ left-lane training (side-balanced)", "w9_sidebal", "sidebal_s{s}", [1, 2, 3]),
    ("Side-balanced + brake-first", "w9_sidebal_bf", "sidebal_s{s}_bf", [1, 2, 3]),
    ("Side-balanced, no LiDAR / hazard flag", "w9_overnight", "blank_s{s}", [1, 2, 3]),
    ("Side-balanced, raw inputs", "w10_raw", "raw_s{s}", [1, 2, 3]),
    ("Side-balanced + realism limits", "w11_realism", "real_s{s}", [1, 2, 3]),  # steer 2.0/s, brake <= 0.75
    ("Rule + realism limits", "w11_realism", "rule_real", [None]),
]

# fig_hw7_inputs.png: (CONTROLLERS label, presentation label), top to bottom after the rule.
PRESENTATION = [
    ("+ left-lane training (side-balanced)", "SAC, calculated inputs\n(predicted pedestrian position)"),
    ("Side-balanced, raw inputs", "SAC, raw inputs\n(current position, plain road edges)"),
    ("Side-balanced, no LiDAR / hazard flag", "SAC, no LiDAR"),
]

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SEED, MEAN, RULE_C = "#2a78d6", "#0b0b0b", "#eb6834"  # palette from the input-importance chart

_EVIDENCE = re.compile(r"evidence: .*?(\d{8}_\d{6}_circuit_default)")
_SAFE = re.compile(r"SUMMARY: .*safe \(v4\) (\d+)")


def run_outcome(log_path):
    """(safe, encounters) of one circuit run, or None if it has not finished."""
    if not os.path.isfile(log_path):
        return None
    text = open(log_path, errors="replace").read()
    safe, run = _SAFE.findall(text), _EVIDENCE.findall(text)
    if not safe or not run:
        return None
    with open(CIRCUIT_DIR / run[-1] / "hazards.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    n_safe = sum(r["ego_safe_success"] == "True" for r in rows)
    if n_safe != int(safe[-1]):
        raise ValueError(f"{log_path}: hazards.csv says {n_safe} safe, SUMMARY says {safe[-1]}")
    return n_safe, len(rows)


def load(controller):
    """{seed: {town: (safe, encounters)}} for the seeds whose 15 plans have all finished."""
    _, folder, tag, seeds = controller
    out = {}
    for s in seeds:
        towns = {}
        for town, plans in PLANS.items():
            got = [run_outcome(CIRCUIT_DIR / folder / f"{town}_p{p}_{tag.format(s=s)}.log") for p in plans]
            if None in got:
                break
            towns[town] = (sum(g[0] for g in got), sum(g[1] for g in got))
        if len(towns) == len(PLANS):
            out[s] = towns
    return out


def _style(ax):
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)


def dot_rows(ax, rows, rule_pct, *, labels=True, show_mean_text=True):
    """One row per controller: a dot per seed, a bar for the seed mean, the rule as a dashed line."""
    for i, (_, pcts) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.scatter(pcts, [y] * len(pcts), s=46, color=SEED, edgecolor="white", linewidth=1.2, zorder=3)
        m = float(np.mean(pcts))
        ax.plot([m, m], [y - 0.28, y + 0.28], color=MEAN, lw=2.2, zorder=4, solid_capstyle="round")
        if show_mean_text:
            ax.text(102, y, f"{m:.0f}%", va="center", fontsize=9, color=INK)
    ax.axvline(rule_pct, color=RULE_C, lw=1.6, ls=(0, (4, 3)), zorder=2)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(0, 100)
    if labels:  # shared y-axis: set the labels on one panel only, or the others wipe them
        ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    _style(ax)
    ax.tick_params(axis="y", labelleft=labels)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rule = load(RULE)
    if not rule:
        raise SystemExit("the rule's 15 runs are missing; nothing to compare against")
    rule = rule[None]
    data = []
    for c in CONTROLLERS:
        got = load(c)
        if not got:
            print(f"[figures] skipped (no finished seed yet): {c[0]}")
            continue
        missing = [s for s in c[3] if s not in got]
        if missing:
            print(f"[figures] {c[0]}: seed(s) {missing} not finished; plotting {sorted(got)}")
        data.append((c[0], got))

    rows = [("Rule", "-", t, *rule[t]) for t in PLANS]
    rows += [(label, s, t, *got[s][t]) for label, got in data for s in sorted(got) for t in PLANS]
    with open(FIG_DIR / "circuit_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["controller", "seed", "town", "safe", "encounters"])
        w.writerows(rows)
    for label, _, t, safe, n in rows:  # every controller must face the same encounters as the rule
        if n != rule[t][1]:
            print(f"[figures] WARNING {label} {t}: {n} encounters, rule had {rule[t][1]}")

    def pct(towns, keys):
        return 100.0 * sum(towns[k][0] for k in keys) / sum(towns[k][1] for k in keys)

    total = sum(n for _, n in rule.values())
    fig, ax = plt.subplots(figsize=(10, 0.62 * len(data) + 1.9))
    dot_rows(ax, [(lab, [pct(t, PLANS) for t in got.values()]) for lab, got in data], pct(rule, PLANS))
    ax.set_xlabel(f"encounters handled safely (% of {total}, 15 circuit routes in 3 towns)", fontsize=9, color=MUTED)
    ax.text(pct(rule, PLANS), len(data) - 0.35, f" rule {pct(rule, PLANS):.0f}%", color=RULE_C, fontsize=9,
            va="bottom")
    ax.tick_params(axis="y", labelsize=10, colors=INK)
    fig.suptitle("Safe encounters by controller (dots = training seeds, bar = mean, dashed = simple rule)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 0.97, 0.95))
    fig.savefig(FIG_DIR / "fig_safe_overall.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(PLANS), figsize=(14, 0.62 * len(data) + 2.1), sharey=True)
    for k, (ax, town) in enumerate(zip(axes, PLANS)):
        dot_rows(ax, [(lab, [pct(t, [town]) for t in got.values()]) for lab, got in data], pct(rule, [town]),
                 labels=k == 0, show_mean_text=False)
        ax.set_title(f"{TOWN_LABEL[town]}\nrule {pct(rule, [town]):.0f}% of {rule[town][1]}", fontsize=10,
                     loc="left", color=INK)
        ax.set_xlabel("% safe", fontsize=9, color=MUTED)
    axes[0].tick_params(axis="y", labelsize=10, colors=INK)
    fig.suptitle("Safe encounters by town (dots = seeds, bar = mean, dashed = simple rule)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIR / "fig_safe_by_town.png", dpi=160)
    plt.close(fig)
    # Presentation version (HW7): the input question only, short labels, the rule as its own row.
    by_label = dict(data)
    simple = [("Simple rule\n(full brake, swerve to roomier side)", [pct(rule, PLANS)])]
    simple += [(short, [pct(t, PLANS) for t in by_label[lab].values()]) for lab, short in PRESENTATION
               if lab in by_label]
    fig, ax = plt.subplots(figsize=(9, 0.8 * len(simple) + 1.6))
    dot_rows(ax, simple, pct(rule, PLANS))
    ax.set_xlabel(f"pedestrian encounters handled safely (% of {total}, 3 towns, 2 never seen in training)",
                  fontsize=9, color=MUTED)
    ax.tick_params(axis="y", labelsize=11, colors=INK)
    fig.suptitle("What does the learned policy need to see?  (dots = 3 training seeds, bar = mean)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 0.97, 0.93))
    fig.savefig(FIG_DIR / "fig_hw7_inputs.png", dpi=200)
    plt.close(fig)
    print(f"[figures] wrote {FIG_DIR}: circuit_results.csv, fig_safe_overall.png, fig_safe_by_town.png, "
          "fig_hw7_inputs.png")


if __name__ == "__main__":
    main()
