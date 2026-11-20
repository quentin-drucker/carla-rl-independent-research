"""analyze_speed_sweep.py

Does the first reaction change with the pedestrian's walking speed? Reads the w10_speed sweep
(_scratch/w10_speed.sh: same Town04 spot, edge walkers at 0.8-2.4 m/s) and labels each episode with
analyze_strategy's rule (mean [brake, u] over the first WINDOW_S s after onset).

Usage (from src/): ../venv/Scripts/python.exe analyze_speed_sweep.py [VARIANT]   (e.g. _eqroom)
Output: artifacts/figures/speed_sweep{VARIANT}.csv, fig_speed_sweep{VARIANT}.png, and a printed table.
"""
import csv
import json
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyze_strategy import COLORS, NAMES, STRATEGIES, WINDOW_S, label  # noqa: E402
from make_figures import FIG_DIR, INK, MUTED  # noqa: E402
from project_paths import RUNS_DIR  # noqa: E402

EVAL_DIR = RUNS_DIR / "rl_encounter_eval"
VARIANT = sys.argv[1] if len(sys.argv) > 1 else ""
TAGS = ["sidebal_s1", "sidebal_s2", "sidebal_s3", "rule"]
BEHIND = {"left": "left", "right": "right"}  # walker_side -> the side that passes behind the walker


def early(steps_csv):
    with open(steps_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["triggered"] == "True"]
    if not rows:
        return None
    t0 = float(rows[0]["sim_time_s"])
    acts = [json.loads(r["action"]) for r in rows if float(r["sim_time_s"]) < t0 + WINDOW_S]
    return sum(a[0] for a in acts) / len(acts), sum(a[1] for a in acts) / len(acts)


def load():
    out = []
    for tag in TAGS:
        log = EVAL_DIR / f"w10_speed{VARIANT}_{tag}.log"
        if not log.is_file():
            continue
        ev = re.findall(r"evidence: .*?(\d{8}_\d{6}_rl_encounter_eval_default)", log.read_text(errors="replace"))
        if not ev or "SUMMARY" not in log.read_text(errors="replace"):
            print(f"[speed] {tag}: not finished, skipped")
            continue
        run = EVAL_DIR / ev[-1]
        with open(run / "summary.csv", newline="") as f:
            for s in csv.DictReader(f):
                act = early(run / f"ep_{int(s['episode']):04d}" / "steps.csv")
                if act is None:
                    continue
                strat = label(act[1], act[0])
                out.append({"controller": tag, "walker_side": s["walker_side"],
                            "walker_speed_mps": float(s["walker_speed_mps"]), "mph": float(s["target_mph"]),
                            "ttc_s": float(s["trigger_ttc_s"]), "brake": round(act[0], 3), "u": round(act[1], 3),
                            "strategy": strat, "behind": strat == BEHIND[s["walker_side"]],
                            "safe": s["ego_safe_success"] == "True"})
    return out


def main():
    rows = load()
    if not rows:
        raise SystemExit("no finished sweep runs")
    with open(FIG_DIR / f"speed_sweep{VARIANT}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["controller"], r["walker_side"], r["walker_speed_mps"])].append(r)
    speeds = sorted({r["walker_speed_mps"] for r in rows})
    print("controller  walker from  speed   L/R/brake first/neither %   pass behind   safe")
    for (c, side, v), g in sorted(groups.items()):
        n = len(g)
        mix = "/".join(f"{100 * sum(r['strategy'] == s for r in g) / n:.0f}" for s in STRATEGIES)
        print(f"  {NAMES[c]:11s} {side:5s} {v:4.1f} m/s   {mix:18s}  {sum(r['behind'] for r in g):2d}/{n}"
              f"        {sum(r['safe'] for r in g):2d}/{n}")

    ctrls = [c for c in TAGS if any(r["controller"] == c for r in rows)]
    fig, axes = plt.subplots(len(ctrls), 2, figsize=(10, 1.5 * len(ctrls) + 1.2), sharex=True)
    for i, c in enumerate(ctrls):
        for j, side in enumerate(("left", "right")):
            ax = axes[i][j]
            for k, v in enumerate(speeds):
                g = groups[(c, side, v)]
                left = 0.0
                for s in STRATEGIES:
                    share = 100 * sum(r["strategy"] == s for r in g) / len(g) if g else 0
                    ax.barh(k, share, left=left, color=COLORS[s], edgecolor="white", height=0.75)
                    left += share
                ax.text(101, k, f"{sum(r['safe'] for r in g)}/{len(g)} safe", va="center", fontsize=8, color=MUTED)
            ax.set_yticks(range(len(speeds)), [f"{v} m/s" for v in speeds])
            ax.set_xlim(0, 100)
            ax.tick_params(colors=MUTED, labelsize=8, length=0)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            if i == 0:
                ax.set_title(f"Walker from the {side}", fontsize=10, loc="left", color=INK)
            if j == 0:
                ax.set_ylabel(NAMES[c], fontsize=10, color=INK)
    for ax in axes[-1]:
        ax.set_xlabel("% of episodes (first 0.5 s)", fontsize=9, color=MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[s]) for s in STRATEGIES]
    fig.legend(handles, ["swerve left", "swerve right", "brake first (may swerve later)", "neither"], loc="lower center",
               ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Same spot, different walking speed (models trained only at 1.8 m/s)" + (" - equal room both sides" if VARIANT else ""), x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.05, 0.97, 0.95))
    fig.savefig(FIG_DIR / f"fig_speed_sweep{VARIANT}.png", dpi=180)
    plt.close(fig)
    print(f"[speed] wrote {FIG_DIR / f'fig_speed_sweep{VARIANT}.png'}")


if __name__ == "__main__":
    main()
