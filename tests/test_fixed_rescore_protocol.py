import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fixed_rescore_protocol import compare_to_archived, eta_text, load_archived_runs, progress_line  # noqa: E402

ARCHIVED = {"hazard_triggered": True, "trigger_time_s": 5.54, "ego_speed_at_trigger_mps": 9.737252689248136,
            "ego_dist_at_trigger_m": 42.5, "time_to_stop_s": 0.94, "outcome": "full_stop"}


class CompareTests(unittest.TestCase):
    def test_identical_pre_end_values_reproduce_even_if_outcome_differs(self):
        new = dict(ARCHIVED, outcome="slowed_avoided")
        self.assertTrue(compare_to_archived(new, ARCHIVED)["all_match"])

    def test_any_pre_end_difference_is_flagged(self):
        r = compare_to_archived(dict(ARCHIVED, trigger_time_s=5.56), ARCHIVED)
        self.assertFalse(r["all_match"])
        self.assertFalse(r["trigger_time_s"]["match"])
        self.assertTrue(r["ego_speed_at_trigger_mps"]["match"])

    def test_none_only_matches_none(self):
        self.assertTrue(compare_to_archived(dict(ARCHIVED, trigger_time_s=None),
                                            dict(ARCHIVED, trigger_time_s=None))["all_match"])
        self.assertFalse(compare_to_archived(dict(ARCHIVED, trigger_time_s=None), ARCHIVED)["all_match"])

    def test_time_to_stop_is_reported_but_not_a_reproduction_failure(self):
        r = compare_to_archived(dict(ARCHIVED, time_to_stop_s=8.1), dict(ARCHIVED, time_to_stop_s=None))
        self.assertTrue(r["all_match"])
        self.assertFalse(r["time_to_stop_s"]["match"])


class LoadTests(unittest.TestCase):
    def test_loads_run_folders_in_order_and_filters_profiles(self):
        with tempfile.TemporaryDirectory() as d:
            for rid, prof in (("run_0001", "exponential"), ("run_0000", "step_constant"), ("notes", None)):
                os.makedirs(os.path.join(d, rid))
                if prof:
                    with open(os.path.join(d, rid, "config.json"), "w") as f:
                        json.dump({"brake_profile": prof}, f)
                    with open(os.path.join(d, rid, "result.json"), "w") as f:
                        json.dump(ARCHIVED, f)
            self.assertEqual([r["run_id"] for r in load_archived_runs(d)], ["run_0000", "run_0001"])
            self.assertEqual([r["run_id"] for r in load_archived_runs(d, ["exponential"])], ["run_0001"])


class ProgressTests(unittest.TestCase):
    def test_eta_and_mismatch_flag(self):
        self.assertEqual(eta_text(0, 10, 5.0), "eta --")
        self.assertEqual(eta_text(1, 3, 1800.0), "eta 1h00m")
        line = progress_line(index=2, total=800, run_id="run_0001", config={"brake_profile": "exponential",
                             "walker_cross": "far", "target_mph": 35.0, "trigger_ttc_s": 1.8},
                             outcome="contact", archived_outcome="full_stop", reproduced=False, elapsed_s=60.0)
        self.assertIn("[  2/800] run_0001", line)
        self.assertIn("REPRODUCTION MISMATCH", line)


if __name__ == "__main__":
    unittest.main()
