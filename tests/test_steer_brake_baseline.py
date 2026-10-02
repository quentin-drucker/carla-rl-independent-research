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

    def test_only_brake_steer_steers(self):
        self.assertTrue(mode_uses_steering(MODE_BRAKE_STEER))
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


if __name__ == "__main__":
    unittest.main()
