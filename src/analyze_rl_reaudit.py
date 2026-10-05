"""analyze_rl_reaudit.py

Offline analysis of an eval_policy_encounters.py run on the matched grid
(Phase 1 re-audit of last semester's SAC results). Reads the saved evidence
only; never re-simulates, never overwrites.

For each episode it lines up three labels:
    archived  -- the outcome in last semester's sac_on_sweep_results.csv;
    legacy    -- what eval_sac_on_sweep.py's rule gives on THIS run (a
                 reproduction check: agreement with archived means the
                 old pipeline's numbers are reproducible from current code);
    protocol  -- encounter_metrics' outcome (the trustworthy one).
and re-scores every trace at a second pedestrian radius (default 0.188 m,
CARLA's measured walker collision half-extent; the protocol default is a
conservative 0.3 m) to show how sensitive the contact count is to it.

Usage (from src/):
    ..\\venv\\Scripts\\python.exe -X utf8 analyze_rl_reaudit.py runs/rl_encounter_eval/<dir>
Writes <dir>/reaudit_comparison.csv and <dir>/reaudit_report.json.
"""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter

from encounter_metrics import EncounterProtocol, compute_encounter_metrics
from trace_schema import read_trace_csv

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARCHIVED = os.path.join(SRC_DIR, "runs", "20260417_202355", "sac_on_sweep_results.csv")
COLUMNS = ["episode", "run_id", "walker_cross", "target_mph", "trigger_ttc_s", "physics_label",
           "archived_outcome", "legacy_label", "protocol_outcome", "contact_speed_mps", "first_contact_time_s",
           "time_to_stop_s", "min_clearance_m", "contact_after_first_stop", "alt_radius_outcome",
           "alt_radius_contact_speed_mps", "alt_radius_min_clearance_m", "sensor_collision_ticks"]


def _episode_rows(eval_dir, alt_radius_m, archived):
    with open(os.path.join(eval_dir, "eval_manifest.json")) as f:
        manifest = json.load(f)
    protocol = EncounterProtocol(**manifest["protocol"])
    alt_protocol = EncounterProtocol(**{**manifest["protocol"], "pedestrian_radius_m": alt_radius_m})
    with open(os.path.join(eval_dir, "summary.csv"), newline="") as f:
        summary = list(csv.DictReader(f))
    rows = []
    for s in summary:
        ep_dir = os.path.join(eval_dir, f"ep_{int(s['episode']):04d}")
        with open(os.path.join(ep_dir, "metrics.json")) as f:
            m = json.load(f)
        ticks = read_trace_csv(os.path.join(ep_dir, "trace.csv"))
        recomputed = compute_encounter_metrics(ticks, onset_time_s=m["onset_time_s"], protocol=protocol)
        if recomputed.outcome != m["outcome"]:
            raise RuntimeError(f"episode {s['episode']}: trace re-score {recomputed.outcome} != saved {m['outcome']}")
        alt = compute_encounter_metrics(ticks, onset_time_s=m["onset_time_s"], protocol=alt_protocol)
        after_stop = (m["outcome"] == "contact" and m["time_to_stop_s"] is not None
                      and m["first_contact_time_s"] > m["time_to_stop_s"])
        rows.append({
            "episode": int(s["episode"]), "run_id": s["run_id"], "walker_cross": s["walker_cross"],
            "target_mph": float(s["target_mph"]), "trigger_ttc_s": s["trigger_ttc_s"],
            "physics_label": s["physics_label"],
            "archived_outcome": archived.get(s["run_id"], {}).get("outcome"),
            "legacy_label": s["legacy_label"], "protocol_outcome": m["outcome"],
            "contact_speed_mps": m["contact_speed_mps"], "first_contact_time_s": m["first_contact_time_s"],
            "time_to_stop_s": m["time_to_stop_s"], "min_clearance_m": m["min_clearance_m"],
            "contact_after_first_stop": after_stop,
            "alt_radius_outcome": alt.outcome, "alt_radius_contact_speed_mps": alt.contact_speed_mps,
            "alt_radius_min_clearance_m": alt.min_clearance_m,
            "sensor_collision_ticks": int(s["sensor_collision_ticks"] or 0),
        })
    return manifest, rows


def _crosstab(rows, a, b):
    return {f"{k[0]} -> {k[1]}": v for k, v in sorted(Counter((r[a], r[b]) for r in rows).items(), key=str)}


def build_report(rows, alt_radius_m, manifest):
    contacts = [r for r in rows if r["protocol_outcome"] == "contact"]
    speeds = [r["contact_speed_mps"] for r in contacts if r["contact_speed_mps"] is not None]
    by_cross = {}
    for cross in sorted({r["walker_cross"] for r in rows}):
        sub = [r for r in rows if r["walker_cross"] == cross]
        by_cross[cross] = {"n": len(sub), "protocol": dict(Counter(r["protocol_outcome"] for r in sub)),
                           "archived": dict(Counter(r["archived_outcome"] for r in sub)),
                           f"protocol_at_radius_{alt_radius_m}": dict(Counter(r["alt_radius_outcome"] for r in sub))}
    return {
        "eval_dir_manifest": {k: manifest.get(k) for k in ("git_commit", "git_dirty", "model_path", "model_sha256",
                                                           "collision_signal", "protocol", "physics_backend")},
        "n_episodes": len(rows),
        "protocol_outcomes": dict(Counter(r["protocol_outcome"] for r in rows)),
        "archived_outcomes": dict(Counter(r["archived_outcome"] for r in rows)),
        "legacy_labels_this_run": dict(Counter(r["legacy_label"] for r in rows)),
        "legacy_matches_archived": sum(r["legacy_label"] == r["archived_outcome"] for r in rows),
        "archived_to_protocol": _crosstab(rows, "archived_outcome", "protocol_outcome"),
        "by_walker_cross": by_cross,
        "contacts": {
            "n": len(contacts),
            "sensor_missed": sum(1 for r in contacts if r["sensor_collision_ticks"] == 0),
            "contact_after_first_stop": sum(1 for r in contacts if r["contact_after_first_stop"]),
            "contact_speed_mps": ({"median": statistics.median(speeds), "max": max(speeds), "min": min(speeds),
                                   "n_below_1_mps": sum(v < 1.0 for v in speeds),
                                   "n_below_3_mps": sum(v < 3.0 for v in speeds)} if speeds else None),
        },
        "alt_radius_m": alt_radius_m,
        "alt_radius_outcomes": dict(Counter(r["alt_radius_outcome"] for r in rows)),
        "protocol_to_alt_radius": _crosstab(rows, "protocol_outcome", "alt_radius_outcome"),
        "safe_success_protocol": sum(r["protocol_outcome"] in ("passed_clear", "stopped_clear") for r in rows),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("eval_dir")
    p.add_argument("--archived-csv", default=DEFAULT_ARCHIVED)
    p.add_argument("--alt-radius", type=float, default=0.188)
    args = p.parse_args(argv)

    with open(args.archived_csv, newline="") as f:
        archived = {r["run_id"]: r for r in csv.DictReader(f)}
    manifest, rows = _episode_rows(args.eval_dir, args.alt_radius, archived)
    report = build_report(rows, args.alt_radius, manifest)

    with open(os.path.join(args.eval_dir, "reaudit_comparison.csv"), "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.eval_dir, "reaudit_report.json"), "x") as f:
        json.dump({**report, "archived_csv": os.path.abspath(args.archived_csv)}, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
