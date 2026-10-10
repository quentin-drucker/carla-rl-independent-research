"""analyze_controller_comparison.py

SAC versus the four fixed brake profiles on last semester's 200-scenario
matched grid, everything scored by the SAME common encounter protocol
(Phase 1 follow-up, 2026-10-05). Offline: reads saved evidence only.

Inputs:
    --fixed-dir  a test27___fixed_profile_rescore.py output directory
    --sac-dir    an eval_policy_encounters.py output directory (SAC re-audit)

Scenarios are matched by their configuration (everything except
brake_profile/run_id), so each controller is compared on identical
scenarios. Every trace is also re-scored at a second pedestrian radius
(default 0.188 m, CARLA's measured walker collision half-extent; the
protocol's primary radius is 0.3 m) as a sensitivity check.

Paired comparison: for each fixed profile vs. SAC, the scenarios where
exactly one of the two made contact, with an exact two-sided McNemar
(binomial) p-value. This is a descriptive test on one deterministic pass
per controller, not a claim about policy seeds.

Usage (from src/):
    ..\\venv\\Scripts\\python.exe -X utf8 analyze_controller_comparison.py --fixed-dir <dir> --sac-dir <dir> --out <dir>
Writes <out>/controller_comparison.csv and <out>/controller_comparison.json (never overwrites).
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

from avoidability import compute_avoidability
from encounter_metrics import EncounterProtocol, compute_encounter_metrics
from scenario_config import ScenarioConfig
from trace_schema import read_trace_csv

KEY_FIELDS = ("target_mph", "encounter_distance_m", "walker_speed_mps", "walker_side", "walker_cross",
              "walker_startup_s", "trigger_ttc_s", "trigger_delay_s", "braking_ramp_up_per_s", "brake_headway_s",
              "weather_preset")
SAC_LABEL = "SAC_v3"


def scenario_key(config: dict) -> tuple:
    return tuple(round(config[k], 6) if isinstance(config.get(k), float) else config.get(k) for k in KEY_FIELDS)


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _score(run_dir, metrics, protocol, alt_protocol):
    ticks = read_trace_csv(os.path.join(run_dir, "trace.csv"))
    re = compute_encounter_metrics(ticks, onset_time_s=metrics["onset_time_s"], protocol=protocol)
    if re.outcome != metrics["outcome"]:
        raise RuntimeError(f"{run_dir}: re-score {re.outcome} != saved {metrics['outcome']}")
    return compute_encounter_metrics(ticks, onset_time_s=metrics["onset_time_s"], protocol=alt_protocol)


def _record(controller, config, metrics, alt, archived_outcome):
    avoid = compute_avoidability(ScenarioConfig.from_dict(config))
    return {
        "controller": controller, "key": scenario_key(config), "walker_cross": config["walker_cross"],
        "target_mph": config["target_mph"], "trigger_ttc_s": config["trigger_ttc_s"],
        "physics_label": avoid["label"], "outcome": metrics["outcome"],
        "contact": metrics["outcome"] == "contact", "contact_speed_mps": metrics["contact_speed_mps"],
        "min_clearance_m": metrics["min_clearance_m"], "time_to_stop_s": metrics["time_to_stop_s"],
        "max_abs_jerk_normal_speed_mps3": metrics["max_abs_jerk_normal_speed_mps3"],
        "peak_decel_normal_speed_mps2": metrics["peak_decel_normal_speed_mps2"],
        "alt_outcome": alt.outcome, "alt_contact": alt.outcome == "contact", "archived_outcome": archived_outcome,
    }


def load_fixed(fixed_dir, protocol, alt_protocol):
    out = []
    for name in sorted(os.listdir(fixed_dir)):
        run_dir = os.path.join(fixed_dir, name)
        if not (name.startswith("run_") and os.path.isfile(os.path.join(run_dir, "metrics.json"))):
            continue
        with open(os.path.join(run_dir, "metrics.json")) as f:
            m = json.load(f)
        with open(os.path.join(run_dir, "run.json")) as f:
            run = json.load(f)
        alt = _score(run_dir, m, protocol, alt_protocol)
        out.append(_record(run["config"]["brake_profile"], run["config"], m, alt, run["archived_result"]["outcome"]))
    return out


def load_sac(sac_dir, protocol, alt_protocol, archived_csv):
    archived = {}
    if archived_csv and os.path.isfile(archived_csv):
        with open(archived_csv, newline="") as f:
            archived = {r["run_id"]: r["outcome"] for r in csv.DictReader(f)}
    out = []
    for name in sorted(os.listdir(sac_dir)):
        ep_dir = os.path.join(sac_dir, name)
        if not (name.startswith("ep_") and os.path.isfile(os.path.join(ep_dir, "metrics.json"))):
            continue
        with open(os.path.join(ep_dir, "metrics.json")) as f:
            m = json.load(f)
        with open(os.path.join(ep_dir, "episode.json")) as f:
            ep = json.load(f)
        alt = _score(ep_dir, m, protocol, alt_protocol)
        out.append(_record(SAC_LABEL, ep["config"], m, alt, archived.get(ep["config"].get("run_id"))))
    return out


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def build_report(records, alt_radius_m):
    by_ctrl = defaultdict(list)
    for r in records:
        by_ctrl[r["controller"]].append(r)
    common = set.intersection(*(set(r["key"] for r in rs) for rs in by_ctrl.values())) if by_ctrl else set()

    controllers = {}
    for ctrl, rs in sorted(by_ctrl.items()):
        rs = [r for r in rs if r["key"] in common]
        clear = [r for r in rs if not r["contact"]]
        controllers[ctrl] = {
            "n": len(rs),
            "contact": sum(r["contact"] for r in rs),
            f"contact_at_radius_{alt_radius_m}": sum(r["alt_contact"] for r in rs),
            "outcomes": dict(Counter(r["outcome"] for r in rs)),
            "archived_outcomes": dict(Counter(r["archived_outcome"] for r in rs)),
            "contact_by_cross": {c: sum(r["contact"] for r in rs if r["walker_cross"] == c)
                                 for c in sorted({r["walker_cross"] for r in rs})},
            "contact_by_mph": {f"{m:g}": sum(r["contact"] for r in rs if r["target_mph"] == m)
                               for m in sorted({r["target_mph"] for r in rs})},
            "contact_by_physics_label": {lbl: sum(r["contact"] for r in rs if r["physics_label"] == lbl)
                                         for lbl in sorted({r["physics_label"] for r in rs})},
            "median_contact_speed_mps": _median(r["contact_speed_mps"] for r in rs),
            "median_min_clearance_clear_runs_m": _median(r["min_clearance_m"] for r in clear),
            "median_time_to_stop_s": _median(r["time_to_stop_s"] for r in rs),
            "median_max_jerk_normal_speed_mps3": _median(r["max_abs_jerk_normal_speed_mps3"] for r in rs),
        }

    paired = {}
    if SAC_LABEL in by_ctrl:
        sac = {r["key"]: r for r in by_ctrl[SAC_LABEL] if r["key"] in common}
        for ctrl, rs in sorted(by_ctrl.items()):
            if ctrl == SAC_LABEL:
                continue
            other = {r["key"]: r for r in rs if r["key"] in common}
            b = sum(1 for k in common if sac[k]["contact"] and not other[k]["contact"])
            c = sum(1 for k in common if other[k]["contact"] and not sac[k]["contact"])
            paired[ctrl] = {"sac_contact_profile_clear": b, "profile_contact_sac_clear": c,
                            "both_contact": sum(1 for k in common if sac[k]["contact"] and other[k]["contact"]),
                            "mcnemar_exact_p": mcnemar_exact_p(b, c)}
    return {"n_common_scenarios": len(common), "alt_radius_m": alt_radius_m,
            "controllers": controllers, "paired_vs_sac": paired}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fixed-dir", required=True)
    p.add_argument("--sac-dir", required=True)
    p.add_argument("--out", required=True, help="directory for the comparison files")
    p.add_argument("--alt-radius", type=float, default=0.188)
    p.add_argument("--archived-sac-csv",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "20260417_202355",
                                        "sac_on_sweep_results.csv"))
    args = p.parse_args(argv)

    protocol = EncounterProtocol(horizon_s=8.0)
    alt_protocol = EncounterProtocol(horizon_s=8.0, pedestrian_radius_m=args.alt_radius)
    records = load_fixed(args.fixed_dir, protocol, alt_protocol) + load_sac(args.sac_dir, protocol, alt_protocol,
                                                                           args.archived_sac_csv)
    report = build_report(records, args.alt_radius)
    report["inputs"] = {"fixed_dir": os.path.abspath(args.fixed_dir), "sac_dir": os.path.abspath(args.sac_dir)}

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "controller_comparison.csv"), "x", newline="") as f:
        cols = [k for k in records[0] if k != "key"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({k: r[k] for k in cols} for r in records)
    with open(os.path.join(args.out, "controller_comparison.json"), "x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
