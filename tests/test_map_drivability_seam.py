import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import carla  # noqa: E402

from map_drivability import classify_point_drivability_seam_tolerant  # noqa: E402


class FakeWaypoint:
    def __init__(self, lane_type, lane_id=-1):
        self.lane_type = lane_type
        self.road_id = 1
        self.lane_id = lane_id


class FunctionMap:
    """get_waypoint(project_to_road=False) answered by a function of (x, y)."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def get_waypoint(self, location, project_to_road=True, lane_type=None):
        assert project_to_road is False
        self.calls += 1
        result = self.fn(round(location.x, 6), round(location.y, 6))
        if result == "raise":
            raise RuntimeError("simulated failed map query")
        return result


DRIVING = FakeWaypoint(carla.LaneType.Driving)
SIDEWALK = FakeWaypoint(carla.LaneType.Sidewalk)


class SeamTolerantDrivabilityTests(unittest.TestCase):
    def test_drivable_point_needs_no_probes(self):
        fake = FunctionMap(lambda x, y: DRIVING)
        status, wp, corrected = classify_point_drivability_seam_tolerant(fake, carla.Location(x=0, y=0, z=0))
        self.assertEqual((status, corrected), ("drivable", False))
        self.assertEqual(fake.calls, 1)

    def test_hairline_seam_between_two_driving_lanes_is_drivable(self):
        # 2 cm gap at y in [0.99, 1.01] between lane -3 (y < 1) and lane -2 (y > 1).
        def fn(x, y):
            if 0.99 <= y <= 1.01:
                return None
            return FakeWaypoint(carla.LaneType.Driving, lane_id=-3 if y < 1 else -2)
        status, wp, corrected = classify_point_drivability_seam_tolerant(
            FunctionMap(fn), carla.Location(x=5.0, y=1.0, z=0.0))
        self.assertEqual(status, "drivable")
        self.assertTrue(corrected)

    def test_point_just_past_the_road_edge_stays_non_drivable(self):
        fake = FunctionMap(lambda x, y: DRIVING if y < 1.0 else None)
        status, _, corrected = classify_point_drivability_seam_tolerant(fake, carla.Location(x=0, y=1.02, z=0))
        self.assertEqual((status, corrected), ("non_drivable", False))

    def test_fully_off_road_stays_non_drivable(self):
        status, _, corrected = classify_point_drivability_seam_tolerant(
            FunctionMap(lambda x, y: None), carla.Location(x=0, y=0, z=0))
        self.assertEqual((status, corrected), ("non_drivable", False))

    def test_wrong_lane_type_is_never_corrected(self):
        def fn(x, y):
            return SIDEWALK if (x, y) == (0, 0) else DRIVING
        status, wp, corrected = classify_point_drivability_seam_tolerant(
            FunctionMap(fn), carla.Location(x=0, y=0, z=0))
        self.assertEqual((status, corrected), ("non_drivable", False))
        self.assertIs(wp, SIDEWALK)

    def test_failed_query_stays_unknown(self):
        status, _, corrected = classify_point_drivability_seam_tolerant(
            FunctionMap(lambda x, y: "raise"), carla.Location(x=0, y=0, z=0))
        self.assertEqual((status, corrected), ("unknown", False))


if __name__ == "__main__":
    unittest.main()
