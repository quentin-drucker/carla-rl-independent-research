import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pedestrian_contact import (  # noqa: E402
    compute_ego_pedestrian_contact_radius_m,
    detect_contact_ticks,
)


class ComputeContactRadiusTests(unittest.TestCase):
    def test_uses_larger_half_extent_not_average(self):
        radius = compute_ego_pedestrian_contact_radius_m(
            ego_half_length_m=2.4, ego_half_width_m=1.0, pedestrian_radius_m=0.0, safety_margin_m=0.0
        )
        self.assertAlmostEqual(radius, 2.4)

    def test_adds_pedestrian_radius_and_margin(self):
        radius = compute_ego_pedestrian_contact_radius_m(
            ego_half_length_m=2.4, ego_half_width_m=1.0, pedestrian_radius_m=0.3, safety_margin_m=0.2
        )
        self.assertAlmostEqual(radius, 2.9)

    def test_default_matches_documented_tesla_model3_extents(self):
        radius = compute_ego_pedestrian_contact_radius_m()
        self.assertAlmostEqual(radius, 2.396 + 0.3)


class DetectContactTicksTests(unittest.TestCase):
    def test_no_contact_when_far_apart(self):
        ego = [(0.0, 0.0), (10.0, 0.0)]
        ped = [(50.0, 50.0), (60.0, 60.0)]
        self.assertEqual(detect_contact_ticks(ego, ped, contact_radius_m=2.7), [])

    def test_contact_detected_within_radius(self):
        ego = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
        ped = [(10.0, 10.0), (1.0, 1.0), (10.0, 10.0)]
        # distance at index 1 = sqrt(2) ~= 1.414, within a 2.7m radius
        self.assertEqual(detect_contact_ticks(ego, ped, contact_radius_m=2.7), [1])

    def test_exact_boundary_is_not_contact(self):
        # distance exactly equal to the radius should not count (strict <)
        ego = [(0.0, 0.0)]
        ped = [(3.0, 4.0)]  # distance = 5.0
        self.assertEqual(detect_contact_ticks(ego, ped, contact_radius_m=5.0), [])
        self.assertEqual(detect_contact_ticks(ego, ped, contact_radius_m=5.001), [0])

    def test_none_entries_are_skipped_not_treated_as_contact(self):
        ego = [None, (0.0, 0.0), (0.0, 0.0)]
        ped = [(0.0, 0.0), None, (0.0, 0.0)]
        self.assertEqual(detect_contact_ticks(ego, ped, contact_radius_m=2.7), [2])

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            detect_contact_ticks([(0.0, 0.0)], [(0.0, 0.0), (1.0, 1.0)], contact_radius_m=1.0)

    def test_multiple_contact_windows_all_reported(self):
        ego = [(0.0, 0.0)] * 5
        ped = [(0.0, 0.0), (10.0, 10.0), (0.5, 0.0), (10.0, 10.0), (0.1, 0.1)]
        self.assertEqual(detect_contact_ticks(ego, ped, contact_radius_m=2.7), [0, 2, 4])


if __name__ == "__main__":
    unittest.main()
