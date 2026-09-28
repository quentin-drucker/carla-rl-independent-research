import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reactive_avoidance import compute_required_clearance_offset_m  # noqa: E402


class ComputeRequiredClearanceOffsetTests(unittest.TestCase):
    def test_pedestrian_at_lane_center_passing_right(self):
        offset = compute_required_clearance_offset_m(
            pedestrian_lateral_m=0.0, side_sign=+1,
            ego_half_width_m=1.082, pedestrian_radius_m=0.3, safety_margin_m=0.3,
        )
        self.assertAlmostEqual(offset, 1.682)

    def test_pedestrian_at_lane_center_passing_left(self):
        offset = compute_required_clearance_offset_m(
            pedestrian_lateral_m=0.0, side_sign=-1,
            ego_half_width_m=1.082, pedestrian_radius_m=0.3, safety_margin_m=0.3,
        )
        self.assertAlmostEqual(offset, -1.682)

    def test_offset_pedestrian_passing_right_adds_on_top(self):
        # Pedestrian already 1.0m right of center -- passing them on the
        # right must clear the pedestrian's OWN position, not lane center.
        offset = compute_required_clearance_offset_m(
            pedestrian_lateral_m=1.0, side_sign=+1,
            ego_half_width_m=1.082, pedestrian_radius_m=0.3, safety_margin_m=0.3,
        )
        self.assertAlmostEqual(offset, 2.682)

    def test_offset_pedestrian_passing_left_subtracts(self):
        # Pedestrian 1.0m right of center, passing on the LEFT means going
        # even further left than center, past the pedestrian entirely.
        offset = compute_required_clearance_offset_m(
            pedestrian_lateral_m=1.0, side_sign=-1,
            ego_half_width_m=1.082, pedestrian_radius_m=0.3, safety_margin_m=0.3,
        )
        self.assertAlmostEqual(offset, -0.682)

    def test_reproduces_test16_undershoot_diagnosis(self):
        # test16's far-cross pedestrian ends up at +2.55m; the script used a
        # fixed peak_offset_m=1.5, well short of what's needed to clear
        # them. This function, given that final position, would have
        # correctly called for a much larger offset.
        offset = compute_required_clearance_offset_m(
            pedestrian_lateral_m=2.55, side_sign=+1,
        )
        self.assertGreater(offset, 1.5)  # the value actually used was not enough
        self.assertAlmostEqual(offset, 2.55 + 1.082 + 0.3 + 0.3)

    def test_zero_pedestrian_radius_and_margin_still_adds_ego_half_width(self):
        offset = compute_required_clearance_offset_m(
            pedestrian_lateral_m=0.0, side_sign=+1,
            ego_half_width_m=1.082, pedestrian_radius_m=0.0, safety_margin_m=0.0,
        )
        self.assertAlmostEqual(offset, 1.082)


if __name__ == "__main__":
    unittest.main()
