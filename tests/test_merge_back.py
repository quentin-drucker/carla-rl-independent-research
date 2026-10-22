import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from merge_back import MergeBack, hazard_behind_ego  # noqa: E402
from pedestrian_contact import EGO_HALF_LENGTH_M  # noqa: E402

DT = 0.02


def drive(mb, behind_fn, target_m=7.4, seconds=10.0):
    """[(t, target, hazard_active)] with the policy holding target_m."""
    return [(k * DT, *mb.update(k * DT, hazard_behind=behind_fn(k * DT), target_m=target_m))
            for k in range(int(seconds / DT))]


class HazardBehindTests(unittest.TestCase):
    def test_behind_means_past_the_rear_bumper_plus_the_hazard_and_a_margin(self):
        edge = EGO_HALF_LENGTH_M + 0.3 + 1.0  # pedestrian radius 0.3, margin 1.0
        kw = dict(ego_x_m=0.0, ego_y_m=0.0, ego_yaw_deg=0.0, hazard_y_m=5.0, hazard_half_length_m=0.3, margin_m=1.0)
        self.assertFalse(hazard_behind_ego(hazard_x_m=-edge + 0.05, **kw))
        self.assertTrue(hazard_behind_ego(hazard_x_m=-edge - 0.05, **kw))
        self.assertFalse(hazard_behind_ego(hazard_x_m=10.0, **kw))  # ahead

    def test_a_longer_hazard_must_be_further_behind(self):  # e.g. a parked car later
        kw = dict(ego_x_m=0.0, ego_y_m=0.0, ego_yaw_deg=0.0, hazard_x_m=-5.0, hazard_y_m=3.0, margin_m=1.0)
        self.assertTrue(hazard_behind_ego(hazard_half_length_m=0.3, **kw))
        self.assertFalse(hazard_behind_ego(hazard_half_length_m=2.4, **kw))

    def test_heading_is_respected(self):  # ego heading -x, as on the Town04 test road
        self.assertTrue(hazard_behind_ego(ego_x_m=0.0, ego_y_m=0.0, ego_yaw_deg=180.0, hazard_x_m=10.0,
                                          hazard_y_m=0.0, hazard_half_length_m=0.3, margin_m=1.0))


class MergeBackTests(unittest.TestCase):
    def test_holds_the_policy_route_and_hazard_mode_until_clear(self):
        trace = drive(MergeBack(), lambda t: False)
        self.assertTrue(all(target == 7.4 and active for _, target, active in trace))

    def test_a_brief_clear_reading_does_not_start_the_return(self):
        mb = MergeBack(clear_confirm_s=0.5)
        trace = drive(mb, lambda t: 2.0 <= t < 2.3)
        self.assertTrue(all(active for _, _, active in trace))
        self.assertIsNone(mb.return_start_s)

    def test_confirmed_clear_ends_hazard_mode_and_returns_smoothly_to_the_lane(self):
        mb = MergeBack(clear_confirm_s=0.5, return_rate_mps=1.5, min_return_s=2.0)
        trace = drive(mb, lambda t: t >= 2.0)
        self.assertAlmostEqual(mb.clear_since_s, 2.0)
        self.assertAlmostEqual(mb.return_start_s, 2.5, delta=DT + 1e-9)
        self.assertAlmostEqual(mb.return_duration_s, 7.4 / 1.5)
        before = [x for x in trace if x[0] < mb.return_start_s - 1e-9]
        after = [x for x in trace if x[0] >= mb.return_start_s - 1e-9]
        self.assertTrue(all(active for *_, active in before) and not any(active for *_, active in after))
        targets = [target for _, target, _ in after]
        self.assertTrue(all(a >= b >= 0.0 for a, b in zip(targets, targets[1:])))   # monotonic, never overshoots
        self.assertLess(max(abs(a - b) for a, b in zip(targets, targets[1:])), 7.4 * 2 * DT / mb.return_duration_s)
        end = mb.return_start_s + mb.return_duration_s
        self.assertTrue(all(target == 0.0 for t, target, _ in after if t >= end))
        self.assertEqual(mb.state, "returned")

    def test_small_offsets_use_the_minimum_return_time(self):
        mb = MergeBack(min_return_s=2.0, return_rate_mps=1.5)
        drive(mb, lambda t: True, target_m=0.5)
        self.assertAlmostEqual(mb.return_duration_s, 2.0)

    def test_once_returning_it_keeps_returning(self):
        # ponytail-tested limit: a hazard behind a moving car cannot get back in front of it.
        mb = MergeBack(clear_confirm_s=0.5)
        trace = drive(mb, lambda t: 1.0 <= t < 2.0)
        self.assertEqual(trace[-1][1:], (0.0, False))


if __name__ == "__main__":
    unittest.main()
