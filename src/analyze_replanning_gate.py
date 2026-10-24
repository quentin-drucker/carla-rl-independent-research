"""analyze_replanning_gate.py

Week 7 scenario gate: does a scripted controller that re-decides after the
pedestrian changes behavior beat every onset-fixed maneuver somewhere?

Input: rl_encounter_eval folders from eval_policy_encounters --suite test26
with DYNAMIC_WALKER_CASES walkers (src/_scratch/w7_table.sh). Each folder's
manifest names its controller: a fixed [brake, u] from onset, or a scripted
aim rule run per_tick or locked.

For every case (walker, mph, onset TTC) and pair of walkers that look the same
until the change (steer_brake_baseline.DYNAMIC_WALKER_PAIRS):
  separating      both members avoidable by some fixed option, but no single
                  fixed option is safe in both (an onset decision cannot be
                  right for both, since they look identical when it is made)
  replanner wins  ...and the replanner is safe in both members
  strong win      the replanner is safe in a member where EVERY fixed option fails
Safety is reported under encounter protocol v4 (ego_safe_success: a pedestrian
walking into an already-stopped car is not an ego hit) and v3 (safe_success,
every contact counts).

Usage (from src/):
    python -X utf8 analyze_replanning_gate.py runs/rl_encounter_eval/<dir> [...] [--out report.md]
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict

from steer_brake_baseline import DYNAMIC_WALKER_CASES, DYNAMIC_WALKER_PAIRS

SAFETY_KEYS = {"v4": "ego_safe_success", "v3": "safe_success"}


def controller_name(manifest: dict, fixed_u) -> str:
    if manifest.get("scripted_aim"):
        commit = manifest["steer_commit"]
        lock = f"_{manifest['lock_delay_s']:g}s" if commit == "lock_at_onset" else ""
        return f"rule_{manifest['scripted_aim']}_{commit}{lock}"
    if manifest.get("fixed_u"):
        return f"{'coast' if manifest['fixed_brake'] == 0 else 'brake'}_u{float(fixed_u):g}"
    return f"sac_{manifest.get('steer_commit')}"  # ponytail: two models of one commit mode would share a column


def walker_of(run_id: str) -> str:
    rest = run_id.split("_", 2)[2] if run_id.count("_") >= 2 else ""
    for name in sorted(DYNAMIC_WALKER_CASES, key=len, reverse=True):
        if rest == name or rest.startswith(name + "_"):
            return name
    return rest


def load_table(dirs):
    """{(walker, mph, ttc): {controller: summary row}}, controllers in first-seen order."""
    table, controllers = defaultdict(dict), []
    for d in dirs:
        with open(os.path.join(d, "eval_manifest.json")) as f:
            manifest = json.load(f)
        with open(os.path.join(d, "summary.csv"), newline="") as f:
            for r in csv.DictReader(f):
                name = controller_name(manifest, r["fixed_u"])
                if name not in controllers:
                    controllers.append(name)
                table[(walker_of(r["run_id"]), float(r["target_mph"]), float(r["trigger_ttc_s"]))][name] = r
    return table, controllers


def _safe(row, key) -> bool:
    return str(row.get(key)) == "True"


def gate(table, *, fixed, replanner, pairs=DYNAMIC_WALKER_PAIRS, key="ego_safe_success"):
    """Gate facts per pair: lists of (mph, ttc) cases. Pure."""
    out = {}
    cases = sorted({(m, t) for (_, m, t) in table})
    for a, b in pairs:
        res = {"separating": [], "replanner_wins": [], "strong_wins": [], "replanner_loses": [], "cases": 0}
        for mph, ttc in cases:
            ra, rb = table.get((a, mph, ttc)), table.get((b, mph, ttc))
            if not ra or not rb:
                continue
            res["cases"] += 1
            sa = {c for c in fixed if c in ra and _safe(ra[c], key)}
            sb = {c for c in fixed if c in rb and _safe(rb[c], key)}
            rep = [replanner in r and _safe(r[replanner], key) for r in (ra, rb)]
            if sa and sb and not (sa & sb):
                res["separating"].append((mph, ttc))
                if all(rep):
                    res["replanner_wins"].append((mph, ttc))
            for s, ok, w in ((sa, rep[0], a), (sb, rep[1], b)):
                if not s and ok:
                    res["strong_wins"].append((w, mph, ttc))
                if s and not ok:
                    res["replanner_loses"].append((w, mph, ttc))
        res["passes"] = bool(res["replanner_wins"] or res["strong_wins"])
        out[(a, b)] = res
    return out


def _cell(r) -> str:
    if r is None:
        return "-"
    if r["outcome"] == "contact":
        tag = "walked-into" if r.get("contact_cause") == "pedestrian" else "hit"
        return f"{tag} {float(r['contact_speed_mps']):.1f}"
    off = " OFF" if r.get("drivable_violation_ticks") not in (None, "", "0") else ""
    return f"{'pass' if r['outcome'] == 'passed_clear' else 'stop' if r['outcome'] == 'stopped_clear' else r['outcome']}" \
           f" {float(r['min_clearance_m']):.2f}{off}"


def report(table, controllers, *, fixed, replanner) -> str:
    out = ["# Week 7 replanning gate", ""]
    for version, key in SAFETY_KEYS.items():
        out += [f"## Gate, protocol {version} ({key})", "",
                "| Pair | Cases | Separating | Replanner safe in both | Strong wins | Replanner fails where a fixed option is safe | PASS |",
                "|---|---|---|---|---|---|---|"]
        for (a, b), res in gate(table, fixed=fixed, replanner=replanner, key=key).items():
            if not res["cases"]:
                continue
            fmt = lambda xs: ", ".join(f"{x[-2]:g}/{x[-1]:g}" + (f" ({x[0]})" if len(x) == 3 else "") for x in xs) or "none"
            out.append(f"| {a} vs {b} | {res['cases']} | {len(res['separating'])}: {fmt(res['separating'])} "
                       f"| {fmt(res['replanner_wins'])} | {fmt(res['strong_wins'])} "
                       f"| {len(res['replanner_loses'])} | {'YES' if res['passes'] else 'no'} |")
        out.append("")
    out += ["## Safe counts per walker (v4 ego_safe_success / v3 safe_success)", "",
            "| Walker | " + " | ".join(controllers) + " |", "|---" * (len(controllers) + 1) + "|"]
    for w in sorted({k[0] for k in table}):
        rows = [v for k, v in table.items() if k[0] == w]
        out.append(f"| {w} ({len(rows)} cases) | " + " | ".join(
            f"{sum(_safe(r[c], 'ego_safe_success') for r in rows if c in r)} / "
            f"{sum(_safe(r[c], 'safe_success') for r in rows if c in r)}" for c in controllers) + " |")
    out += ["", "## Per-case results (hit/walked-into = impact m/s; pass/stop = min clearance m)", ""]
    for w in sorted({k[0] for k in table}):
        out += [f"### {w}", "", "| mph | TTC | " + " | ".join(controllers) + " |", "|---" * (len(controllers) + 2) + "|"]
        for (ww, mph, ttc) in sorted(k for k in table if k[0] == w):
            out.append(f"| {mph:g} | {ttc:g} | " + " | ".join(_cell(table[(ww, mph, ttc)].get(c)) for c in controllers) + " |")
        out.append("")
    out += ["## Comfort and commitment (medians over all cases)", "",
            "| Controller | peak decel m/s^2 | max jerk m/s^3 | max route lateral m | aim reversals while moving |",
            "|---|---|---|---|---|"]
    for c in controllers:
        rows = [v[c] for v in table.values() if c in v]

        def med(k):
            vals = sorted(float(r[k]) for r in rows if r.get(k) not in (None, ""))
            return f"{vals[len(vals) // 2]:.1f}" if vals else "-"
        out.append(f"| {c} | {med('peak_decel_normal_speed_mps2')} | {med('max_abs_jerk_normal_speed_mps3')} "
                   f"| {med('max_abs_route_lateral_m')} | {med('requested_offset_reversals_moving')} |")
    return "\n".join(out) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description="Week 7 replanning gate (see docstring)")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--replanner", default="rule_pass_behind_per_tick")
    p.add_argument("--out")
    args = p.parse_args(argv)
    table, controllers = load_table(args.dirs)
    fixed = [c for c in controllers if re.match(r"(brake|coast)_u", c)]
    text = report(table, controllers, fixed=fixed, replanner=args.replanner)
    if args.out:
        with open(args.out, "x", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
