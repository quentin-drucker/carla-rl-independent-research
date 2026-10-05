import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analyze_rl_reaudit import build_report  # noqa: E402


def _row(cross, archived, legacy, protocol, alt, speed=None, after_stop=False, sensor=0):
    return dict(walker_cross=cross, archived_outcome=archived, legacy_label=legacy, protocol_outcome=protocol,
                alt_radius_outcome=alt, contact_speed_mps=speed, contact_after_first_stop=after_stop,
                sensor_collision_ticks=sensor)


class BuildReportTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            _row("near", "full_stop", "full_stop", "contact", "contact", speed=3.7),
            _row("near", "collision", "collision", "contact", "contact", speed=6.0, sensor=4),
            _row("far", "full_stop", "full_stop", "passed_clear", "passed_clear"),
            _row("far", "full_stop", "slowed_avoided", "contact", "stopped_clear", speed=0.4, after_stop=True),
        ]
        self.report = build_report(self.rows, 0.188, {"model_sha256": "abc"})

    def test_counts_and_reproduction_check(self):
        r = self.report
        self.assertEqual(r["protocol_outcomes"], {"contact": 3, "passed_clear": 1})
        self.assertEqual(r["legacy_matches_archived"], 3)
        self.assertEqual(r["archived_to_protocol"]["full_stop -> contact"], 2)
        self.assertEqual(r["safe_success_protocol"], 1)

    def test_contact_details(self):
        c = self.report["contacts"]
        self.assertEqual((c["n"], c["sensor_missed"], c["contact_after_first_stop"]), (3, 2, 1))
        self.assertEqual(c["contact_speed_mps"]["n_below_1_mps"], 1)
        self.assertEqual(c["contact_speed_mps"]["max"], 6.0)

    def test_radius_sensitivity(self):
        self.assertEqual(self.report["protocol_to_alt_radius"]["contact -> stopped_clear"], 1)
        self.assertEqual(self.report["by_walker_cross"]["far"]["protocol_at_radius_0.188"],
                         {"passed_clear": 1, "stopped_clear": 1})


if __name__ == "__main__":
    unittest.main()
