import math
import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from steer_brake_baseline import (  # noqa: E402
    MODE_BRAKE_ONLY,
    MODE_BRAKE_STEER,
    MODE_NO_INTERVENTION,
    footprint_corners_xy,
    make_hazard_command_fn,
    mode_uses_steering,
    MODE_BRAKE_PASSAGE_EDGE,
    PassageEdgeOffset,
    best_option,
    expand_runs,
    format_best_route_table,
    max_abs_yaw_change_deg,
    onset_gap_m,
)


class HazardCommandTests(unittest.TestCase):
    def test_no_intervention_never_brakes_even_after_onset(self):
        fn = make_hazard_command_fn(MODE_NO_INTERVENTION)
        self.assertEqual(fn(5.0, True, 4.0), (False, 0.0))
        self.assertEqual(fn(1.0, False, None), (False, 0.0))

    def test_braking_modes_are_identical_and_start_exactly_at_onset(self):
        for mode in (MODE_BRAKE_ONLY, MODE_BRAKE_STEER):
            fn = make_hazard_command_fn(mode, brake_target=0.8)
            # Before onset: explicitly False (not None), so LiDAR braking is
            # disabled and pre-onset driving matches no_intervention.
            self.assertEqual(fn(1.0, False, None), (False, 0.8))
            self.assertEqual(fn(4.0, True, 4.0), (True, 0.8))

    def test_only_steering_modes_steer(self):
        self.assertTrue(mode_uses_steering(MODE_BRAKE_STEER))
        self.assertTrue(mode_uses_steering(MODE_BRAKE_PASSAGE_EDGE))
        self.assertFalse(mode_uses_steering(MODE_BRAKE_ONLY))
        self.assertFalse(mode_uses_steering(MODE_NO_INTERVENTION))

    def test_rejects_bad_mode_and_brake_target(self):
        with self.assertRaises(ValueError):
            make_hazard_command_fn("swerve_only")
        with self.assertRaises(ValueError):
            make_hazard_command_fn(MODE_BRAKE_ONLY, brake_target=1.5)


class GeometryTests(unittest.TestCase):
    def test_corners_heading_plus_x(self):
        c = footprint_corners_xy(0.0, 0.0, 0.0, half_length_m=2.0, half_width_m=1.0)
        # front-left, front-right, rear-right, rear-left; CARLA right = +y.
        expected = [(2.0, -1.0), (2.0, 1.0), (-2.0, 1.0), (-2.0, -1.0)]
        for (x, y), (ex, ey) in zip(c, expected):
            self.assertAlmostEqual(x, ex)
            self.assertAlmostEqual(y, ey)

    def test_corners_rotate_with_yaw(self):
        # Heading -180 deg (the Town04 spawn-242 direction): front is -x,
        # and the vehicle's right is -y.
        c = footprint_corners_xy(10.0, 5.0, -180.0, half_length_m=2.0, half_width_m=1.0)
        fr = c[1]
        self.assertAlmostEqual(fr[0], 8.0)
        self.assertAlmostEqual(fr[1], 4.0)
        for x, y in c:
            self.assertAlmostEqual(math.hypot(x - 10.0, y - 5.0), math.sqrt(5.0))

    def test_onset_gap_subtracts_half_length_and_pedestrian_radius(self):
        self.assertAlmostEqual(onset_gap_m(speed_mps=10.0, onset_ttc_s=2.0,
                                           half_length_m=2.4, pedestrian_radius_m=0.3), 17.3)



class PassageEdgeOffsetTests(unittest.TestCase):
    def _passage(self):
        from passage import Obstacle, compute_passage
        return compute_passage(lambda lat: -5.0 <= lat <= 3.5, [Obstacle(0.0, 0.3)], ego_half_width_m=1.0, margin_m=0.25)

    def test_zero_before_onset_then_right_edge(self):
        fn = PassageEdgeOffset()
        fn.set_passage(self._passage())
        self.assertEqual(fn(1.0, False, None), 0.0)
        self.assertAlmostEqual(fn(1.02, True, 1.02), 2.25)  # right road edge 3.5 - 1.0 - 0.25

    def test_left_actions_clipped_in_right_only_mode(self):
        fn = PassageEdgeOffset(u=-1.0)
        fn.set_passage(self._passage())
        self.assertEqual(fn(1.0, True, 1.0), 0.0)

    def test_missing_passage_is_counted_not_guessed(self):
        fn = PassageEdgeOffset()
        self.assertEqual(fn(1.0, True, 1.0), 0.0)
        self.assertEqual(fn(1.02, True, 1.0), 0.0)
        self.assertEqual(fn.missing_passage_ticks, 2)

    def test_brakes_like_the_other_braking_modes(self):
        cmd = make_hazard_command_fn(MODE_BRAKE_PASSAGE_EDGE, brake_target=1.0)
        self.assertEqual(cmd(1.0, True, 1.0), (True, 1.0))
        self.assertEqual(cmd(0.5, False, None), (False, 1.0))


def _row(mode, u=None, *, outcome="contact", clearance=-0.1, v=None, off_road=0, mph=35.0, ttc=0.8):
    return {"mph": mph, "onset_ttc_s": ttc, "mode": mode, "passage_u": u, "outcome": outcome,
            "safe_success": outcome != "contact" and not off_road, "min_clearance_m": clearance,
            "contact_speed_mps": v, "drivable_violation_ticks": off_road, "passage_left_edge_m": -3.87,
            "passage_right_edge_m": 7.42, "missing_passage_ticks": 0, "target_offset_m": (u or 0) * 7.42,
            "max_abs_route_lateral_m": (u or 0) * 7.0, "max_abs_yaw_change_deg": 10.0,
            "contact_r0188": outcome == "contact" and clearance < -0.112}


class PassageSweepTests(unittest.TestCase):
    def test_expand_runs_repeats_only_the_passage_mode(self):
        self.assertEqual(expand_runs([MODE_BRAKE_ONLY, MODE_BRAKE_PASSAGE_EDGE], [0.25, 1.0]), [
            (MODE_BRAKE_ONLY, None, "brake_only"),
            (MODE_BRAKE_PASSAGE_EDGE, 0.25, "brake_passage_edge_u0.25"),
            (MODE_BRAKE_PASSAGE_EDGE, 1.0, "brake_passage_edge_u1")])

    def test_yaw_change_wraps_across_180(self):
        self.assertAlmostEqual(max_abs_yaw_change_deg([-179.0, 179.0, 170.0]), 11.0)
        self.assertIsNone(max_abs_yaw_change_deg([]))

    def test_best_option_prefers_safe_then_missed_then_slowest_hit(self):
        hit_fast, hit_slow = _row(MODE_BRAKE_ONLY, v=10.0), _row(MODE_BRAKE_PASSAGE_EDGE, 0.25, v=6.0)
        off_road = _row(MODE_BRAKE_PASSAGE_EDGE, 1.0, outcome="passed_clear", clearance=2.0, off_road=5)
        safe_small = _row(MODE_BRAKE_PASSAGE_EDGE, 0.5, outcome="passed_clear", clearance=0.4)
        safe_big = _row(MODE_BRAKE_PASSAGE_EDGE, 0.75, outcome="passed_clear", clearance=0.9)
        self.assertEqual(best_option([hit_fast, hit_slow]), ("all_contact", hit_slow))
        self.assertEqual(best_option([hit_fast, off_road]), ("no_contact_unsafe", off_road))
        self.assertEqual(best_option([hit_fast, off_road, safe_small, safe_big]), ("safe", safe_big))

    def test_table_reports_best_option_and_gate_facts(self):
        text = format_best_route_table([
            _row(MODE_BRAKE_ONLY, v=10.2),
            _row(MODE_BRAKE_PASSAGE_EDGE, 0.5, outcome="passed_clear", clearance=0.4),
            _row(MODE_BRAKE_PASSAGE_EDGE, 1.0, outcome="passed_clear", clearance=2.0, off_road=5),
            _row(MODE_BRAKE_ONLY, v=12.3, ttc=0.6),
            _row(MODE_BRAKE_PASSAGE_EDGE, 0.5, v=12.1, clearance=-0.2, ttc=0.6),
            _row(MODE_BRAKE_PASSAGE_EDGE, 1.0, v=12.0, clearance=-0.05, ttc=0.6)])
        self.assertIn("| 0.8 | hit 10.2 m/s | passed 0.40 m | passed 2.00 m OFF-ROAD | u=0.5 (0.40 m) | u=0.5 (0.40 m) |", text)
        self.assertIn("| all hit; least bad u=1 at 12.0 m/s | all hit; least bad u=1 at 12.0 m/s |", text)
        self.assertIn("(-3.87, 7.42)", text)
        self.assertIn("| 1 | 7.42 | 7.00 | 10.0 | 1 of 2 |", text)
        self.assertIn("Contacts: 4 of 6 runs at the 0.3 m pedestrian radius; 1 at 0.188 m.", text)


if __name__ == "__main__":
    unittest.main()
