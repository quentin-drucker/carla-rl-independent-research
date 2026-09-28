import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from route_recovery import PhysicalRouteReturnTracker  # noqa: E402


class PhysicalRouteReturnTrackerTests(unittest.TestCase):
    def test_zero_offset_without_prior_departure_is_not_recovery(self):
        tracker = PhysicalRouteReturnTracker(required_return_ticks=2)
        tracker.update(requested_offset_m=0.0, actual_offset_m=0.0)
        tracker.update(requested_offset_m=0.0, actual_offset_m=0.0)
        self.assertFalse(tracker.departed_route)
        self.assertFalse(tracker.physically_returned)

    def test_requested_return_while_ego_remains_shifted_is_not_recovery(self):
        tracker = PhysicalRouteReturnTracker(required_return_ticks=2)
        tracker.update(requested_offset_m=4.8, actual_offset_m=4.7)
        tracker.update(requested_offset_m=0.0, actual_offset_m=4.7)
        tracker.update(requested_offset_m=0.0, actual_offset_m=4.7)
        self.assertTrue(tracker.departed_route)
        self.assertFalse(tracker.physically_returned)
        self.assertAlmostEqual(tracker.final_actual_offset_m, 4.7)

    def test_physical_return_requires_stable_measured_centering(self):
        tracker = PhysicalRouteReturnTracker(required_return_ticks=3)
        tracker.update(requested_offset_m=1.5, actual_offset_m=1.4)
        self.assertFalse(
            tracker.update(requested_offset_m=0.0, actual_offset_m=0.2)
        )
        self.assertFalse(
            tracker.update(requested_offset_m=0.0, actual_offset_m=0.1)
        )
        self.assertTrue(
            tracker.update(requested_offset_m=0.0, actual_offset_m=0.05)
        )

    def test_requested_offset_must_also_return_to_center(self):
        tracker = PhysicalRouteReturnTracker(required_return_ticks=2)
        tracker.update(requested_offset_m=1.5, actual_offset_m=1.4)
        tracker.update(requested_offset_m=1.5, actual_offset_m=0.1)
        tracker.update(requested_offset_m=1.5, actual_offset_m=0.1)
        self.assertFalse(tracker.physically_returned)

    def test_later_departure_invalidates_an_earlier_return(self):
        tracker = PhysicalRouteReturnTracker(required_return_ticks=2)
        tracker.update(requested_offset_m=1.5, actual_offset_m=1.4)
        tracker.update(requested_offset_m=0.0, actual_offset_m=0.1)
        tracker.update(requested_offset_m=0.0, actual_offset_m=0.1)
        self.assertTrue(tracker.physically_returned)

        tracker.update(requested_offset_m=0.0, actual_offset_m=0.8)
        self.assertFalse(tracker.physically_returned)
        self.assertEqual(tracker.consecutive_return_ticks, 0)

    def test_missing_samples_do_not_create_or_erase_recovery(self):
        tracker = PhysicalRouteReturnTracker(required_return_ticks=1)
        self.assertFalse(
            tracker.update(requested_offset_m=None, actual_offset_m=None)
        )
        tracker.update(requested_offset_m=1.0, actual_offset_m=1.0)
        self.assertTrue(
            tracker.update(requested_offset_m=0.0, actual_offset_m=0.0)
        )
        self.assertTrue(
            tracker.update(requested_offset_m=None, actual_offset_m=None)
        )


if __name__ == "__main__":
    unittest.main()
