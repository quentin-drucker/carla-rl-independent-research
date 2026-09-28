import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ego_clearance_override import (  # noqa: E402
    compute_ego_pedestrian_distance_m,
    is_ego_geometrically_clear_of_pedestrian,
    is_pedestrian_behind_ego,
)
from pedestrian_contact import compute_ego_pedestrian_contact_radius_m  # noqa: E402


class ComputeEgoPedestrianDistanceTests(unittest.TestCase):
    def test_directly_ahead(self):
        d = compute_ego_pedestrian_distance_m(
            ego_x_m=0.0, ego_y_m=0.0, pedestrian_x_m=0.0, pedestrian_y_m=5.0
        )
        self.assertAlmostEqual(d, 5.0)

    def test_directly_to_the_side(self):
        d = compute_ego_pedestrian_distance_m(
            ego_x_m=0.0, ego_y_m=0.0, pedestrian_x_m=3.0, pedestrian_y_m=0.0
        )
        self.assertAlmostEqual(d, 3.0)

    def test_diagonal_uses_full_euclidean_distance(self):
        # 3-4-5 triangle -- catches a lateral-only projection regression
        # (test18 case A's real bug: lateral clearance alone was satisfied
        # while still longitudinally alongside the pedestrian).
        d = compute_ego_pedestrian_distance_m(
            ego_x_m=0.0, ego_y_m=0.0, pedestrian_x_m=3.0, pedestrian_y_m=4.0
        )
        self.assertAlmostEqual(d, 5.0)

    def test_alongside_pedestrian_with_large_lateral_but_zero_longitudinal_gap(self):
        # This is exactly the scenario that produced test18's 21-tick
        # contact: 2.0m sideways, but still squarely alongside (0 forward/
        # back separation). Distance here (2.0m) is correctly LESS than the
        # default contact-based clearance requirement, unlike a lateral-only
        # check which would have already called this "clear".
        d = compute_ego_pedestrian_distance_m(
            ego_x_m=0.0, ego_y_m=10.0, pedestrian_x_m=2.0, pedestrian_y_m=10.0
        )
        self.assertAlmostEqual(d, 2.0)
        required = compute_ego_pedestrian_contact_radius_m(safety_margin_m=1.0)
        self.assertFalse(
            is_ego_geometrically_clear_of_pedestrian(distance_m=d, required_clearance_m=required)
        )


class IsPedestrianBehindEgoTests(unittest.TestCase):
    def test_pedestrian_far_ahead_is_not_behind(self):
        # This is the exact case that produced test18's second bug: a
        # pedestrian 35m ahead read as "clear" by distance alone.
        self.assertFalse(
            is_pedestrian_behind_ego(
                ego_x_m=0.0, ego_y_m=0.0, ego_forward_x=0.0, ego_forward_y=1.0,
                pedestrian_x_m=0.0, pedestrian_y_m=35.0,
            )
        )

    def test_pedestrian_behind_is_behind(self):
        self.assertTrue(
            is_pedestrian_behind_ego(
                ego_x_m=0.0, ego_y_m=10.0, ego_forward_x=0.0, ego_forward_y=1.0,
                pedestrian_x_m=0.0, pedestrian_y_m=0.0,
            )
        )

    def test_pedestrian_directly_alongside_is_not_behind(self):
        # Zero forward/back separation -- squarely beside the ego, dot
        # product is exactly zero, not negative.
        self.assertFalse(
            is_pedestrian_behind_ego(
                ego_x_m=0.0, ego_y_m=10.0, ego_forward_x=0.0, ego_forward_y=1.0,
                pedestrian_x_m=2.0, pedestrian_y_m=10.0,
            )
        )


class IsEgoGeometricallyClearTests(unittest.TestCase):
    def test_clear_when_well_beyond_required_clearance(self):
        self.assertTrue(
            is_ego_geometrically_clear_of_pedestrian(distance_m=5.0, required_clearance_m=3.696)
        )

    def test_not_clear_when_within_required_clearance(self):
        self.assertFalse(
            is_ego_geometrically_clear_of_pedestrian(distance_m=2.0, required_clearance_m=3.696)
        )

    def test_boundary_exactly_equal_is_not_clear(self):
        self.assertFalse(
            is_ego_geometrically_clear_of_pedestrian(distance_m=3.696, required_clearance_m=3.696)
        )

    def test_default_required_clearance_matches_contact_radius_plus_margin(self):
        # compute_ego_pedestrian_contact_radius_m() default = 2.396+0.3 = 2.696;
        # + DEFAULT_CLEAR_MARGIN_M (1.0) = 3.696
        self.assertTrue(is_ego_geometrically_clear_of_pedestrian(distance_m=3.7))
        self.assertFalse(is_ego_geometrically_clear_of_pedestrian(distance_m=3.6))


if __name__ == "__main__":
    unittest.main()
