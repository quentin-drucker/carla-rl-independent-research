import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analyze_controller_comparison import SAC_LABEL, build_report, mcnemar_exact_p, scenario_key  # noqa: E402


def _rec(ctrl, key, contact, cross="near", mph=35.0):
    return dict(controller=ctrl, key=key, walker_cross=cross, target_mph=mph, trigger_ttc_s=1.8,
                physics_label="avoidable", outcome="contact" if contact else "stopped_clear", contact=contact,
                contact_speed_mps=5.0 if contact else None, min_clearance_m=-0.1 if contact else 1.0,
                time_to_stop_s=1.5, max_abs_jerk_normal_speed_mps3=100.0, peak_decel_normal_speed_mps2=-8.0,
                alt_outcome="contact" if contact else "stopped_clear", alt_contact=contact, archived_outcome="full_stop")


class McNemarTests(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(mcnemar_exact_p(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact_p(0, 5), 2 / 32)
        self.assertAlmostEqual(mcnemar_exact_p(3, 3), 1.0)
        self.assertLess(mcnemar_exact_p(20, 2), 0.001)


class ReportTests(unittest.TestCase):
    def test_paired_counts_on_common_scenarios_only(self):
        recs = [_rec(SAC_LABEL, "a", True), _rec(SAC_LABEL, "b", True), _rec(SAC_LABEL, "c", False),
                _rec("step_constant", "a", False), _rec("step_constant", "b", True), _rec("step_constant", "c", True),
                _rec("step_constant", "z", True)]  # z has no SAC counterpart: excluded
        r = build_report(recs, 0.188)
        self.assertEqual(r["n_common_scenarios"], 3)
        self.assertEqual(r["controllers"]["step_constant"]["contact"], 2)
        p = r["paired_vs_sac"]["step_constant"]
        self.assertEqual((p["sac_contact_profile_clear"], p["profile_contact_sac_clear"], p["both_contact"]), (1, 1, 1))

    def test_scenario_key_ignores_profile_and_float_noise(self):
        base = dict(target_mph=35.0, encounter_distance_m=60.0, walker_speed_mps=1.8, walker_side="left",
                    walker_cross="far", walker_startup_s=0.5, trigger_ttc_s=1.8, trigger_delay_s=0.0,
                    braking_ramp_up_per_s=4.0, brake_headway_s=2.5, weather_preset="ClearSunset")
        other = dict(base, brake_profile="exponential", run_id="run_0400", target_mph=35.0000000001)
        self.assertEqual(scenario_key(base), scenario_key(other))


if __name__ == "__main__":
    unittest.main()
