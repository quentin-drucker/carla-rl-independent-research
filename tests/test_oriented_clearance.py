import math
import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from oriented_clearance import (  # noqa: E402
    oriented_rectangle_circle_clearance_m,
    yaw_deg_to_forward_xy,
)


class OrientedRectangleCircleClearanceTests(unittest.TestCase):
    def clearance(self, *, circle_x, circle_y, forward_x=1.0, forward_y=0.0):
        return oriented_rectangle_circle_clearance_m(
            rectangle_x_m=0.0,
            rectangle_y_m=0.0,
            forward_x=forward_x,
            forward_y=forward_y,
            half_length_m=2.396,
            half_width_m=1.082,
            circle_x_m=circle_x,
            circle_y_m=circle_y,
            circle_radius_m=0.3,
        )

    def test_lateral_pass_uses_half_width(self):
        self.assertAlmostEqual(self.clearance(circle_x=0.0, circle_y=1.68), 0.298)

    def test_front_approach_uses_half_length(self):
        self.assertAlmostEqual(self.clearance(circle_x=3.0, circle_y=0.0), 0.304)

    def test_overlap_is_negative(self):
        self.assertLess(self.clearance(circle_x=0.0, circle_y=1.0), 0.0)

    def test_corner_clearance_uses_euclidean_distance(self):
        expected = math.hypot(3.0 - 2.396, 2.0 - 1.082) - 0.3
        self.assertAlmostEqual(self.clearance(circle_x=3.0, circle_y=2.0), expected)

    def test_rotating_vehicle_rotates_footprint(self):
        # With the vehicle facing +Y, a circle at +X is lateral.
        self.assertAlmostEqual(
            self.clearance(circle_x=1.68, circle_y=0.0, forward_x=0.0, forward_y=1.0),
            0.298,
        )

    def test_forward_vector_is_normalized(self):
        self.assertAlmostEqual(
            self.clearance(circle_x=0.0, circle_y=1.68, forward_x=10.0, forward_y=0.0),
            0.298,
        )

    def test_zero_forward_vector_is_rejected(self):
        with self.assertRaises(ValueError):
            self.clearance(circle_x=0.0, circle_y=1.68, forward_x=0.0, forward_y=0.0)

    def test_yaw_conversion_matches_cardinal_directions(self):
        x, y = yaw_deg_to_forward_xy(90.0)
        self.assertAlmostEqual(x, 0.0, places=12)
        self.assertAlmostEqual(y, 1.0)


if __name__ == "__main__":
    unittest.main()
