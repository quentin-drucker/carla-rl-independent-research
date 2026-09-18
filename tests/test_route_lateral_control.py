import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from route_lateral_control import (  # noqa: E402
    closest_route_index_xy,
    offset_point_xy,
    route_unit_right_xy,
    signed_lateral_offset_m,
)


def point(x, y):
    return SimpleNamespace(x=float(x), y=float(y))


class RouteLateralControlTests(unittest.TestCase):
    def test_right_of_eastbound_route_is_positive_y(self):
        route = [point(0, 0), point(5, 0), point(10, 0)]
        self.assertEqual(route_unit_right_xy(route, 1), (0.0, 1.0))
        self.assertAlmostEqual(signed_lateral_offset_m(route, point(5, 2)), 2.0)
        self.assertAlmostEqual(signed_lateral_offset_m(route, point(5, -2)), -2.0)

    def test_offset_point_uses_positive_route_right_convention(self):
        route = [point(0, 0), point(5, 0), point(10, 0)]
        x, y = offset_point_xy(route, route[1], 0.75)
        self.assertAlmostEqual(x, 5.0)
        self.assertAlmostEqual(y, 0.75)

    def test_diagonal_route_right_vector_is_normalized(self):
        route = [point(0, 0), point(1, 1), point(2, 2)]
        right_x, right_y = route_unit_right_xy(route, 1)
        self.assertAlmostEqual(math.hypot(right_x, right_y), 1.0)
        self.assertAlmostEqual(right_x, -math.sqrt(0.5))
        self.assertAlmostEqual(right_y, math.sqrt(0.5))

    def test_closest_route_index_uses_xy_distance(self):
        route = [point(0, 0), point(10, 0), point(20, 0)]
        self.assertEqual(closest_route_index_xy(route, point(8, 3)), 1)

    def test_duplicate_neighbor_is_skipped(self):
        route = [point(0, 0), point(0, 0), point(5, 0)]
        self.assertEqual(route_unit_right_xy(route, 0), (0.0, 1.0))

    def test_degenerate_route_is_rejected(self):
        route = [point(1, 1), point(1, 1)]
        with self.assertRaises(ValueError):
            route_unit_right_xy(route, 0)


if __name__ == "__main__":
    unittest.main()
