import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from merge_back import MergeBack, hazard_behind_ego, path_clear_of_hazard  # noqa: E402
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


class PathClearTests(unittest.TestCase):
    """The car's CURRENT path (from where it is to where it aims) is what is checked."""

    def test_a_pedestrian_in_the_lane_does_not_block_a_car_holding_far_right(self):
        self.assertTrue(path_clear_of_hazard(path_lo_m=7.0, path_hi_m=7.4, hazard_lateral_m=0.0,
                                             hazard_lateral_speed_mps=0.0, hazard_radius_m=0.3))

    def test_a_hazard_between_the_car_and_its_aim_blocks_the_path(self):
        self.assertFalse(path_clear_of_hazard(path_lo_m=0.0, path_hi_m=7.4, hazard_lateral_m=2.55,
                                              hazard_lateral_speed_mps=0.0, hazard_radius_m=0.3))

    def test_margin_is_measured_from_the_car_body_and_the_hazard_edge(self):
        # car body edge at 3.0 - 1.08 = 1.92; pedestrian edge at 0.3: gap 1.62 m
        kw = dict(path_lo_m=3.0, path_hi_m=3.0, hazard_lateral_m=0.0, hazard_lateral_speed_mps=0.0, hazard_radius_m=0.3)
        self.assertTrue(path_clear_of_hazard(margin_m=1.5, **kw))
        self.assertFalse(path_clear_of_hazard(margin_m=1.7, **kw))

    def test_walking_toward_the_path_blocks_it_walking_away_does_not(self):
        kw = dict(path_lo_m=5.0, path_hi_m=5.5, hazard_lateral_m=1.0, hazard_radius_m=0.3, horizon_s=2.0)
        self.assertTrue(path_clear_of_hazard(hazard_lateral_speed_mps=0.0, **kw))
        self.assertFalse(path_clear_of_hazard(hazard_lateral_speed_mps=1.8, **kw))   # reaches ~4.6 m in 2 s
        self.assertTrue(path_clear_of_hazard(hazard_lateral_speed_mps=-1.8, **kw))


class DriveOnTests(unittest.TestCase):
    """A car that STOPPED beside the hazard drives on along its swerve path once
    that path is clear, then returns once the hazard is behind it."""

    def run_mb(self, mb, *, stopped_fn, clear_fn, behind_fn, seconds=12.0, target_m=7.4):
        return [(k * DT, *mb.update(k * DT, hazard_behind=behind_fn(k * DT), path_clear=clear_fn(k * DT),
                                     ego_stopped=stopped_fn(k * DT), target_m=target_m))
                for k in range(int(seconds / DT))]

    def test_stopped_and_clear_drives_on_holding_the_swerve_then_returns(self):
        mb = MergeBack(clear_confirm_s=0.5)
        trace = self.run_mb(mb, stopped_fn=lambda t: 2.0 <= t < 3.0, clear_fn=lambda t: t >= 2.0,
                            behind_fn=lambda t: t >= 5.0)
        at = {round(t, 2): (target, active) for t, target, active in trace}
        self.assertEqual(at[2.0], (7.4, True))             # stopped, clear not yet confirmed
        self.assertEqual(at[2.6], (7.4, False))            # drives on: hazard mode off, swerve path held
        self.assertEqual(at[4.9], (7.4, False))
        self.assertAlmostEqual(mb.drive_on_start_s, 2.5)
        self.assertAlmostEqual(mb.return_start_s, 5.5)     # behind for 0.5 s
        self.assertEqual(trace[-1][1:], (0.0, False))

    def test_a_moving_car_keeps_policy_control_even_with_a_clear_path(self):
        mb = MergeBack()
        trace = self.run_mb(mb, stopped_fn=lambda t: False, clear_fn=lambda t: True, behind_fn=lambda t: False)
        self.assertTrue(all(active for *_, active in trace))
        self.assertIsNone(mb.drive_on_start_s)

    def test_path_blocked_again_while_driving_on_goes_back_to_hazard_mode(self):
        # The hook for a second hazard on the swerve path: brake, policy back in control.
        mb = MergeBack(clear_confirm_s=0.5)
        trace = self.run_mb(mb, stopped_fn=lambda t: t >= 2.0, clear_fn=lambda t: 2.0 <= t < 4.0,
                            behind_fn=lambda t: False)
        at = {round(t, 2): active for t, _, active in trace}
        self.assertFalse(at[3.0])
        self.assertTrue(at[4.1])
        self.assertEqual(mb.state, "avoiding")
        self.assertEqual(mb.drive_on_count, 1)


if __name__ == "__main__":
    unittest.main()
