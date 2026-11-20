"""analyze_strategy.py

How does the policy choose to swerve left, swerve right, or just brake, and does it choose the same way when the
same encounter is repeated? (Izmirli, Nov 10; Week 10 Priority 2/3.)

Reads the circuit runs of the side-balanced models and the rule: w9_sidebal (repeat 1) and w10_repeat (repeats 2, 3).
For each encounter, takes the mean action over the first WINDOW_S seconds after onset (first `triggered` tick in
policy_ticks.csv) and labels it:
    left / right   |mean u| >= SIDE_U   (u > 0 aims right)
    brake first    otherwise, mean brake >= BRAKE  (only the first reaction: many of these swerve later)
    neither        otherwise
Each row also carries the whole-encounter max_abs_route_lateral_m (hazards.csv) and stayed_in_lane (< STAY_M), so
"brake and stay in lane" can be checked instead of read off the first-0.5 s label.
An encounter is identified by (town, plan, hazard index); the same plan seed places the same hazards.

Usage (from src/): ../venv/Scripts/python.exe analyze_strategy.py
Output: artifacts/figures/strategy_encounters.csv (one row per run x encounter) and a printed summary.
"""
import csv
import json
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from make_figures import _EVIDENCE, CIRCUIT_DIR, FIG_DIR, INK, MUTED, PLANS  # noqa: E402
from rl_steering import LATERAL_SCALE_M  # noqa: E402

WINDOW_S, SIDE_U, BRAKE = 0.5, 0.2, 0.5
STAY_M = 0.5  # max sideways offset from the route over the whole encounter to count as "stayed in lane"
CONTROLLERS = ["sidebal_s1", "sidebal_s2", "sidebal_s3", "rule"]
REPEATS = {1: ("w9_sidebal", ""), 2: ("w10_repeat", "_r2"), 3: ("w10_repeat", "_r3")}


def label(u, brake):
    if abs(u) >= SIDE_U:
        return "right" if u > 0 else "left"
    return "brake first" if brake >= BRAKE else "neither"


def early_action(ticks_csv):
    """(mean brake, mean u, onset observation) over the first WINDOW_S s after onset, or None if never triggered.
    The onset observation is the 2nd triggered tick: the 1st still shows the pre-onset state (hazard_active 0)."""
    with open(ticks_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["triggered"] == "True"]
    if len(rows) < 2:
        return None
    t0 = float(rows[0]["sim_time_s"])
    win = [r for r in rows if float(r["sim_time_s"]) < t0 + WINDOW_S]
    obs = rows[1]
    onset = {"left_room_m": round(-float(obs["passage_left_edge_norm"]) * LATERAL_SCALE_M, 2),
             "right_room_m": round(float(obs["passage_right_edge_norm"]) * LATERAL_SCALE_M, 2),
             "ped_lateral_m": round(float(obs["pedestrian_lateral_norm"]) * LATERAL_SCALE_M, 2)}
    return (sum(float(r["action_brake"]) for r in win) / len(win),
            sum(float(r["action_u"]) for r in win) / len(win), onset)


def load_rows():
    out = []
    for rep, (folder, suffix) in REPEATS.items():
        for town, plans in PLANS.items():
            for p in plans:
                for c in CONTROLLERS:
                    log = CIRCUIT_DIR / folder / f"{town}_p{p}_{c}{suffix}.log"
                    if not log.is_file():
                        continue
                    text = open(log, errors="replace").read()
                    run = _EVIDENCE.findall(text)
                    if "SUMMARY" not in text or not run:
                        continue  # unfinished or crashed run
                    run_dir = CIRCUIT_DIR / run[-1]
                    with open(run_dir / "hazards.csv", newline="") as f:
                        for h in csv.DictReader(f):
                            hz_dir = run_dir / f"hz_{int(h['hazard']):02d}"
                            act = early_action(hz_dir / "policy_ticks.csv")
                            if act is None:
                                continue
                            # scripted speed: the observed one is ~0 at onset (0.5 s walker startup)
                            with open(hz_dir / "hazard.json") as hf:
                                act[2]["walker_speed_mps"] = json.load(hf)["config"]["walker_speed_mps"]
                            out.append({"town": town, "plan": p, "hazard": int(h["hazard"]), "controller": c,
                                        "repeat": rep, "walker": h["walker"], "ttc_s": float(h["ttc_s"]),
                                        "brake": round(act[0], 3), "u": round(act[1], 3),
                                        "strategy": label(act[1], act[0]),
                                        "safe": h["ego_safe_success"] == "True",
                                        "min_clearance_m": h["min_clearance_m"],
                                        "max_abs_route_lateral_m": float(h["max_abs_route_lateral_m"]),
                                        "stayed_in_lane": float(h["max_abs_route_lateral_m"]) < STAY_M,
                                        **act[2]})
    return out


def main():
    rows = load_rows()
    if not rows:
        raise SystemExit("no finished runs found")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(FIG_DIR / "strategy_encounters.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"[strategy] {len(rows)} run-encounters, repeats found: {sorted({r['repeat'] for r in rows})}")
    print(f"\nStrategy by controller and walker type (all repeats; window {WINDOW_S}s, |u|>={SIDE_U}, brake>={BRAKE}):")
    by = defaultdict(Counter)
    for r in rows:
        by[(r["controller"], r["walker"])][r["strategy"]] += 1
    for (c, wk), cnt in sorted(by.items()):
        n = sum(cnt.values())
        print(f"  {c:11s} {wk:10s} n={n:3d}  " + "  ".join(f"{s} {100 * cnt[s] / n:3.0f}%"
                                                         for s in STRATEGIES))

    print(f"\nStayed in lane (whole encounter, max offset < {STAY_M} m), repeat 1:")
    for c in CONTROLLERS:
        r1 = [r for r in rows if r["controller"] == c and r["repeat"] == 1]
        stay = [r for r in r1 if r["stayed_in_lane"]]
        bf = [r for r in r1 if r["strategy"] == "brake first"]
        print(f"  {c:11s} stayed {len(stay):2d}/{len(r1)} (safe {sum(r['safe'] for r in stay)});  "
              f"'brake first' {len(bf)}, of which stayed {sum(r['stayed_in_lane'] for r in bf)}")

    # Repeatability: same controller, same encounter, different repeats.
    reps = defaultdict(list)
    for r in rows:
        reps[(r["controller"], r["town"], r["plan"], r["hazard"])].append(r)
    print("\nRepeatability (encounters with >= 2 repeats): same strategy / same outcome every time")
    for c in CONTROLLERS:
        groups = [g for k, g in reps.items() if k[0] == c and len(g) >= 2]
        if not groups:
            continue
        same_s = sum(len({x["strategy"] for x in g}) == 1 for g in groups)
        same_o = sum(len({x["safe"] for x in g}) == 1 for g in groups)
        spread = max(max(x["u"] for x in g) - min(x["u"] for x in g) for g in groups)
        print(f"  {c:11s} {len(groups):3d} encounters  strategy {100 * same_s / len(groups):3.0f}%  "
              f"outcome {100 * same_o / len(groups):3.0f}%  max u spread {spread:.2f}")

    # Seed agreement: repeat 1, the three models on the same encounter.
    seeds = defaultdict(dict)
    for r in rows:
        if r["repeat"] == 1 and r["controller"] != "rule":
            seeds[(r["town"], r["plan"], r["hazard"])][r["controller"]] = r["strategy"]
    full = [v for v in seeds.values() if len(v) == 3]
    if full:
        agree = sum(len(set(v.values())) == 1 for v in full)
        print(f"\nSeed agreement (repeat 1): all 3 models pick the same strategy on {agree}/{len(full)} encounters")
    drivers([r for r in rows if r["repeat"] == 1])


STRATEGIES = ("left", "right", "brake first", "neither")
COLORS = {"left": "#2a78d6", "right": "#eb6834", "brake first": "#52514e", "neither": "#d6d5d0"}
NAMES = {"rule": "Rule", "sidebal_s1": "SAC seed 1", "sidebal_s2": "SAC seed 2", "sidebal_s3": "SAC seed 3"}
WALKERS = {"left_edge": "Walker from the left", "right_edge": "Walker from the right",
           "stationary": "Standing in the road"}


def pct_str(part, whole):
    return f"{part}/{whole} ({100 * part / whole:.0f}%)" if whole else "-"


def drivers(rows):
    """What goes with each choice (repeats are identical, so repeat 1 only), plus the strategy figure."""
    roomier = lambda r: "right" if r["right_room_m"] > r["left_room_m"] else "left"  # noqa: E731 (rule: left on tie)

    print("\nSide picks that go to the roomier side (the rule's choice), by room difference at onset:")
    for c in CONTROLLERS:
        sides = [r for r in rows if r["controller"] == c and r["strategy"] in ("left", "right")]
        near = [r for r in sides if abs(r["right_room_m"] - r["left_room_m"]) < 1.0]
        clear = [r for r in sides if abs(r["right_room_m"] - r["left_room_m"]) >= 1.0]
        print(f"  {NAMES[c]:11s} near tie (<1 m): {pct_str(sum(r['strategy'] == roomier(r) for r in near), len(near)):14s}"
              f"  clear (>=1 m): {pct_str(sum(r['strategy'] == roomier(r) for r in clear), len(clear))}")

    behind_side = {"left_edge": "left", "right_edge": "right"}  # the side the walker starts from
    print("\nWalking pedestrians, side picks that pass BEHIND them (toward the side they came from):")
    for c in CONTROLLERS:
        mv = [r for r in rows if r["controller"] == c and r["walker"] in behind_side
              and r["strategy"] in ("left", "right")]
        print(f"  {NAMES[c]:11s} {pct_str(sum(r['strategy'] == behind_side[r['walker']] for r in mv), len(mv))}")

    print("\nWalker speed (Izmirli's slow vs fast): pass-behind share by scripted walking speed")
    speeds = sorted({r["walker_speed_mps"] for r in rows if r["walker"] in behind_side})
    for c in CONTROLLERS:
        parts = []
        for v in speeds:
            mv = [r for r in rows if r["controller"] == c and r["walker"] in behind_side
                  and r["walker_speed_mps"] == v and r["strategy"] in ("left", "right")]
            parts.append(f"{v} m/s: {pct_str(sum(r['strategy'] == behind_side[r['walker']] for r in mv), len(mv))}")
        print(f"  {NAMES[c]:11s} " + "   ".join(parts))

    print("\nStrategy by warning time (onset TTC):")
    bins = (("< 0.9 s", 0, 0.9), ("0.9-1.2 s", 0.9, 1.2), (">= 1.2 s", 1.2, 99))
    for c in CONTROLLERS:
        parts = []
        for name, lo, hi in bins:
            b = [r for r in rows if r["controller"] == c and lo <= r["ttc_s"] < hi]
            cnt = Counter(r["strategy"] for r in b)
            parts.append(f"{name}: " + "/".join(f"{100 * cnt[s] / len(b):.0f}" for s in STRATEGIES) + f" n={len(b)}")
        print(f"  {NAMES[c]:11s} (L/R/brake first/neither %)  " + "  ".join(parts))

    print("\nMatched outcomes vs the rule (same encounter): model safe & rule not / rule safe & model not")
    rule = {(r["town"], r["plan"], r["hazard"]): r for r in rows if r["controller"] == "rule"}
    for c in CONTROLLERS[:-1]:
        mine = [r for r in rows if r["controller"] == c]
        win = [r for r in mine if r["safe"] and not rule[(r["town"], r["plan"], r["hazard"])]["safe"]]
        lose = [r for r in mine if not r["safe"] and rule[(r["town"], r["plan"], r["hazard"])]["safe"]]
        diff_w = sum(r["strategy"] != rule[(r["town"], r["plan"], r["hazard"])]["strategy"] for r in win)
        diff_l = sum(r["strategy"] != rule[(r["town"], r["plan"], r["hazard"])]["strategy"] for r in lose)
        print(f"  {NAMES[c]:11s} wins {len(win):2d} (different strategy from rule in {diff_w})   "
              f"losses {len(lose):2d} (different strategy in {diff_l})   "
              f"wins by walker {dict(Counter(r['walker'] for r in win))}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, (wk, title) in zip(axes, WALKERS.items()):
        for i, c in enumerate(CONTROLLERS[::-1]):
            sub = [r for r in rows if r["controller"] == c and r["walker"] == wk]
            left = 0.0
            for s in STRATEGIES:
                share = 100 * sum(r["strategy"] == s for r in sub) / len(sub)
                ax.barh(i, share, left=left, color=COLORS[s], edgecolor="white", height=0.7,
                        label=s if (wk == "left_edge" and i == 0) else None)
                if share >= 12:
                    ax.text(left + share / 2, i, f"{share:.0f}", ha="center", va="center", fontsize=9,
                            color="white" if s != "neither" else INK)
                left += share
        ax.set_title(f"{title} (n={len(sub)})", fontsize=10, loc="left", color=INK)
        ax.set_xlim(0, 100)
        ax.set_yticks(range(len(CONTROLLERS)), [NAMES[c] for c in CONTROLLERS[::-1]])
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("% of encounters", fontsize=9, color=MUTED)
    axes[0].tick_params(axis="y", labelsize=10, colors=INK)
    fig.legend(axes[0].get_legend_handles_labels()[0], ["swerve left", "swerve right", "brake first (may swerve later)",
               "neither"], loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.suptitle(f"First reaction to the pedestrian (mean action over the first {WINDOW_S} s); "
                 "identical on every repeat run", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(FIG_DIR / "fig_strategy_by_walker.png", dpi=200)
    plt.close(fig)
    print(f"\n[strategy] wrote {FIG_DIR / 'fig_strategy_by_walker.png'}")


if __name__ == "__main__":
    main()
