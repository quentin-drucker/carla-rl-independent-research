import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from merge_back import MergeBack, hazard_ahead_of_ego, hazard_behind_ego, path_clear_of_hazard  # noqa: E402
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


class BesideTheCarTests(unittest.TestCase):
    """2026-10-26: a person standing right beside the stopped car no longer blocks
    every path. Car body half-width 1.08 m + pedestrian 0.3 m: centers 1.88 m apart
    = a 0.5 m gap."""

    def test_holding_the_lane_past_a_person_beside_the_car_needs_only_no_contact(self):
        kw = dict(path_lo_m=-3.9, path_hi_m=-3.9, hazard_lateral_m=-5.78, hazard_lateral_speed_mps=0.0, hazard_radius_m=0.3)
        self.assertTrue(path_clear_of_hazard(hazard_ahead=False, **kw))   # beside or behind: driving on keeps the gap
        self.assertFalse(path_clear_of_hazard(hazard_ahead=True, **kw))   # ahead: the full 1 m margin still applies

    def test_moving_away_from_a_person_needs_only_no_contact(self):
        kw = dict(path_lo_m=-3.9, hazard_lateral_m=-5.78, hazard_lateral_speed_mps=0.0, hazard_radius_m=0.3)
        self.assertTrue(path_clear_of_hazard(path_hi_m=0.0, **kw))       # back to the lane, away from them
        self.assertFalse(path_clear_of_hazard(path_hi_m=-7.0, **kw))     # toward them

    def test_too_close_or_walking_into_the_car_still_blocks(self):
        self.assertFalse(path_clear_of_hazard(path_lo_m=-3.9, path_hi_m=0.0, hazard_lateral_m=-5.4,  # 0.12 m gap
                                              hazard_lateral_speed_mps=0.0, hazard_radius_m=0.3))
        self.assertFalse(path_clear_of_hazard(path_lo_m=-3.9, path_hi_m=-3.9, hazard_lateral_m=-5.78, hazard_ahead=False,
                                              hazard_lateral_speed_mps=1.0, hazard_radius_m=0.3))  # walking toward the car

    def test_ahead_means_past_the_front_bumper_plus_the_hazard(self):
        kw = dict(ego_x_m=0.0, ego_y_m=0.0, ego_yaw_deg=0.0, hazard_y_m=3.0, hazard_half_length_m=0.3)
        self.assertFalse(hazard_ahead_of_ego(hazard_x_m=1.0, **kw))   # beside the car
        self.assertTrue(hazard_ahead_of_ego(hazard_x_m=4.0, **kw))


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


class DirectReturnTests(unittest.TestCase):
    """A stopped car whose swerve path is blocked but whose original lane is clear
    returns to the lane directly, starting from where the car actually is."""

    def run_mb(self, mb, *, path_clear, lane_clear, stopped=True, seconds=10.0, ego_lateral_m=3.0):
        return [(k * DT, *mb.update(k * DT, hazard_behind=False, path_clear=path_clear, lane_clear=lane_clear,
                                     ego_stopped=stopped, target_m=7.4, ego_lateral_m=ego_lateral_m))
                for k in range(int(seconds / DT))]

    def test_blocked_swerve_but_clear_lane_returns_from_the_cars_position(self):
        mb = MergeBack(clear_confirm_s=0.5, return_rate_mps=1.5, min_return_s=2.0)
        trace = self.run_mb(mb, path_clear=False, lane_clear=True)
        self.assertAlmostEqual(mb.return_start_s, 0.5)
        self.assertTrue(mb.direct_return)
        after = [x for x in trace if x[0] >= 0.5 - 1e-9]
        self.assertAlmostEqual(after[0][1], 3.0)                 # from the car, not from the 7.4 m aim
        self.assertTrue(all(t <= 3.0 and not active for _, t, active in after))
        self.assertEqual(trace[-1][1:], (0.0, False))

    def test_both_blocked_waits(self):
        mb = MergeBack()
        trace = self.run_mb(mb, path_clear=False, lane_clear=False)
        self.assertTrue(all(active for *_, active in trace))
        self.assertEqual(mb.state, "avoiding")

    def test_swerve_path_clear_is_preferred_over_returning(self):
        mb = MergeBack()
        self.run_mb(mb, path_clear=True, lane_clear=True, seconds=2.0)
        self.assertEqual((mb.state, mb.direct_return), ("driving_on", False))

    def test_a_moving_car_never_returns_directly(self):
        mb = MergeBack()
        trace = self.run_mb(mb, path_clear=False, lane_clear=True, stopped=False)
        self.assertTrue(all(active for *_, active in trace))


class AfterContactTests(unittest.TestCase):
    def test_after_a_contact_it_holds_hazard_mode_and_its_lateral_target(self):
        mb = MergeBack(clear_confirm_s=0.5)
        mb.update(0.0, hazard_behind=False, target_m=4.0)
        mb.note_contact()
        for k in range(1, 300):  # hazard behind, path clear, stopped: none of it may resume driving
            out = mb.update(k * DT, hazard_behind=True, path_clear=True, lane_clear=True, ego_stopped=True,
                            target_m=7.4)
            self.assertEqual(out, (4.0, True))
        self.assertEqual(mb.state, "holding_after_contact")

    def test_a_contact_during_the_return_freezes_it_where_it_is(self):
        mb = MergeBack(clear_confirm_s=0.0, min_return_s=2.0)
        target = None
        for k in range(60):
            target, _ = mb.update(k * DT, hazard_behind=True, target_m=7.4)
        mb.note_contact()
        self.assertEqual(mb.update(2.0, hazard_behind=True, target_m=7.4), (target, True))


if __name__ == "__main__":
    unittest.main()
