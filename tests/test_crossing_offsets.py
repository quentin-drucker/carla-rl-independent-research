import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scenario_config import crossing_lateral_offsets_m  # noqa: E402

LANE = 3.0


class CrossingOffsetTests(unittest.TestCase):
    """Walker placement shared by test3 and CarlaAEBEnv."""

    def test_near_and_far_unchanged(self):
        # The env's pre-Phase-1 rule: start 0.85 lane widths out (negative =
        # left), near ends at lane center, far ends mirrored across it.
        self.assertEqual(crossing_lateral_offsets_m(side="left", cross="near", lane_width_m=LANE), (-2.55, 0.0))
        self.assertEqual(crossing_lateral_offsets_m(side="right", cross="near", lane_width_m=LANE), (2.55, 0.0))
        self.assertEqual(crossing_lateral_offsets_m(side="left", cross="far", lane_width_m=LANE), (-2.55, 2.55))
        self.assertEqual(crossing_lateral_offsets_m(side="right", cross="far", lane_width_m=LANE), (2.55, -2.55))

    def test_case_insensitive(self):
        self.assertEqual(crossing_lateral_offsets_m(side="LEFT", cross="Far", lane_width_m=LANE), (-2.55, 2.55))

    def test_stationary_stands_at_lane_center(self):
        # Previously the env silently treated "stationary" as "far".
        for side in ("left", "right"):
            self.assertEqual(crossing_lateral_offsets_m(side=side, cross="stationary", lane_width_m=LANE), (0.0, 0.0))

    def test_unknown_values_raise(self):
        with self.assertRaises(ValueError):
            crossing_lateral_offsets_m(side="centre", cross="near", lane_width_m=LANE)
        with self.assertRaises(ValueError):
            crossing_lateral_offsets_m(side="left", cross="halfway", lane_width_m=LANE)


if __name__ == "__main__":
    unittest.main()
