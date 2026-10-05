import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from passage import (  # noqa: E402
    Obstacle,
    admissible_center_intervals,
    compute_passage,
    drivable_extent_m,
    passage_coordinate_to_offset_m,
)

HW = 1.0   # ego half-width used in these tests
MG = 0.25  # margin


def road(left, right, seams=(), unknown=()):
    """Fake probe: drivable on [left, right] except tiny seam gaps."""
    def probe(lat):
        if any(abs(lat - u) < 1e-9 for u in unknown):
            return None
        if any(abs(lat - s) < 0.011 for s in seams):
            return False
        return left - 1e-9 <= lat <= right + 1e-9
    return probe


class DrivableExtentTests(unittest.TestCase):
    def test_plain_road(self):
        left, right = drivable_extent_m(road(-5.25, 3.0))
        self.assertAlmostEqual(left, -5.25)
        self.assertAlmostEqual(right, 3.0)

    def test_bridges_hairline_lane_seam(self):
        # Town04_Opt: ~2 cm gaps between adjacent driving lanes.
        left, right = drivable_extent_m(road(-5.25, 3.0, seams=(1.5, -1.5)))
        self.assertAlmostEqual((left, right), (-5.25, 3.0))

    def test_does_not_bridge_a_real_gap(self):
        probe = lambda lat: -2.0 <= lat <= 1.0 or 1.5 <= lat <= 4.0  # 0.5 m median strip  # noqa: E731
        left, right = drivable_extent_m(probe)
        self.assertAlmostEqual(right, 1.0)

    def test_unknown_probe_ends_the_scan_conservatively(self):
        left, right = drivable_extent_m(road(-5.0, 5.0, unknown=(2.0,)))
        self.assertAlmostEqual(right, 1.95)

    def test_route_center_off_road(self):
        self.assertEqual(drivable_extent_m(lambda lat: False), (None, None))


class IntervalTests(unittest.TestCase):
    def test_road_shrunk_by_half_width_and_margin(self):
        self.assertEqual(admissible_center_intervals(-4.0, 4.0, ego_half_width_m=HW, margin_m=MG), [(-2.75, 2.75)])

    def test_pedestrian_at_lane_center_splits_the_passage(self):
        iv = admissible_center_intervals(-4.0, 4.0, [Obstacle(0.0, 0.3)], ego_half_width_m=HW, margin_m=MG)
        self.assertEqual(len(iv), 2)
        self.assertAlmostEqual(iv[0][1], -1.55)
        self.assertAlmostEqual(iv[1][0], 1.55)

    def test_second_obstacle_narrows_further(self):
        iv = admissible_center_intervals(-4.0, 6.0, [Obstacle(0.0, 0.3), Obstacle(4.0, 1.0)],
                                         ego_half_width_m=HW, margin_m=MG)
        self.assertEqual(len(iv), 2)
        self.assertAlmostEqual(iv[1][0], 1.55)
        self.assertAlmostEqual(iv[1][1], 1.75)  # right gap now ends where the second obstacle's zone starts

    def test_too_narrow_road_has_no_passage(self):
        self.assertEqual(admissible_center_intervals(-1.0, 1.0, ego_half_width_m=HW, margin_m=MG), [])


class PassageAndActionTests(unittest.TestCase):
    def setUp(self):
        self.p = compute_passage(road(-5.25, 3.0), [Obstacle(0.0, 0.3)], ego_half_width_m=HW, margin_m=MG)

    def test_edges_are_the_extreme_admissible_center_positions(self):
        self.assertAlmostEqual(self.p.left_edge_m, -4.0)
        self.assertAlmostEqual(self.p.right_edge_m, 1.75)
        self.assertTrue(self.p.admits(1.7))
        self.assertFalse(self.p.admits(0.0))  # straight ahead is blocked by the pedestrian

    def test_u_maps_proportionally_to_each_edge(self):
        self.assertEqual(passage_coordinate_to_offset_m(0.0, self.p), 0.0)
        self.assertAlmostEqual(passage_coordinate_to_offset_m(1.0, self.p), 1.75)
        self.assertAlmostEqual(passage_coordinate_to_offset_m(0.5, self.p), 0.875)
        self.assertAlmostEqual(passage_coordinate_to_offset_m(-1.0, self.p), -4.0)

    def test_right_only_mode_clips_left_actions_to_straight(self):
        self.assertEqual(passage_coordinate_to_offset_m(-1.0, self.p, u_min=0.0), 0.0)
        self.assertAlmostEqual(passage_coordinate_to_offset_m(2.0, self.p, u_min=0.0), 1.75)

    def test_side_without_room_maps_to_zero(self):
        p = compute_passage(road(-5.0, 1.2), [Obstacle(0.0, 0.3)], ego_half_width_m=HW, margin_m=MG)
        self.assertLess(p.right_edge_m, 0.0)  # only the left gap exists
        self.assertEqual(passage_coordinate_to_offset_m(1.0, p), 0.0)

    def test_empty_passage_means_no_swerve(self):
        p = compute_passage(road(-1.0, 1.0), ego_half_width_m=HW, margin_m=MG)
        self.assertTrue(p.empty)
        self.assertEqual(passage_coordinate_to_offset_m(1.0, p), 0.0)

    def test_bad_bounds_rejected(self):
        with self.assertRaises(ValueError):
            passage_coordinate_to_offset_m(0.0, self.p, u_min=0.5, u_max=0.0)


if __name__ == "__main__":
    unittest.main()
