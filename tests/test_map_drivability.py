import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import carla  # noqa: E402

from map_drivability import (  # noqa: E402
    classify_point_drivability,
    check_corridor_drivability,
)


class FakeWaypoint:
    def __init__(self, lane_type, road_id=1, lane_id=-1):
        self.lane_type = lane_type
        self.road_id = road_id
        self.lane_id = lane_id


class FakeMap:
    """Duck-typed stand-in for carla.Map.get_waypoint(project_to_road=False).

    `responses` maps a location's x-coordinate to either:
      - a FakeWaypoint (query found a matching-or-mismatched lane), or
      - None (off-road: no waypoint at all), or
      - the string "raise" (simulate a failed/errored map query).
    Keyed on x only since these tests only vary points along one axis.
    """

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get_waypoint(self, location, project_to_road=True, lane_type=None):
        self.calls.append((location, project_to_road, lane_type))
        assert project_to_road is False, (
            "drivability checks must query with project_to_road=False -- "
            "True would silently snap off-road points onto a road"
        )
        response = self.responses[location.x]
        if response == "raise":
            raise RuntimeError("simulated failed map query")
        return response


class ClassifyPointDrivabilityTests(unittest.TestCase):
    def test_driving_lane_is_drivable(self):
        wp = FakeWaypoint(lane_type=carla.LaneType.Driving)
        fake_map = FakeMap({0.0: wp})
        status, returned_wp = classify_point_drivability(
            fake_map, carla.Location(x=0.0, y=0.0, z=0.0)
        )
        self.assertEqual(status, "drivable")
        self.assertIs(returned_wp, wp)

    def test_no_waypoint_is_non_drivable(self):
        fake_map = FakeMap({1.0: None})
        status, returned_wp = classify_point_drivability(
            fake_map, carla.Location(x=1.0, y=0.0, z=0.0)
        )
        self.assertEqual(status, "non_drivable")
        self.assertIsNone(returned_wp)

    def test_wrong_lane_type_is_non_drivable(self):
        wp = FakeWaypoint(lane_type=carla.LaneType.Sidewalk)
        fake_map = FakeMap({2.0: wp})
        status, returned_wp = classify_point_drivability(
            fake_map, carla.Location(x=2.0, y=0.0, z=0.0)
        )
        self.assertEqual(status, "non_drivable")
        self.assertIs(returned_wp, wp)

    def test_failed_query_is_unknown(self):
        fake_map = FakeMap({3.0: "raise"})
        status, returned_wp = classify_point_drivability(
            fake_map, carla.Location(x=3.0, y=0.0, z=0.0)
        )
        self.assertEqual(status, "unknown")
        self.assertIsNone(returned_wp)

    def test_queries_with_project_to_road_false(self):
        # FakeMap itself asserts this, but make the intent explicit here too:
        # project_to_road=True would silently snap an off-road point onto
        # the nearest road, defeating the entire point of this check.
        fake_map = FakeMap({4.0: None})
        classify_point_drivability(fake_map, carla.Location(x=4.0, y=0.0, z=0.0))
        _, project_to_road, lane_type = fake_map.calls[0]
        self.assertFalse(project_to_road)
        self.assertEqual(lane_type, carla.LaneType.Driving)


class CheckCorridorDrivabilityTests(unittest.TestCase):
    def test_empty_corridor_is_unknown_not_drivable(self):
        result = check_corridor_drivability(FakeMap({}), None)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["samples"], [])

    def test_all_drivable_samples_report_drivable(self):
        wp = FakeWaypoint(lane_type=carla.LaneType.Driving)
        fake_map = FakeMap({0.0: wp, 1.0: wp, 2.0: wp})
        points = [carla.Location(x=x, y=0.0, z=0.0) for x in (0.0, 1.0, 2.0)]
        result = check_corridor_drivability(fake_map, points)
        self.assertEqual(result["status"], "drivable")
        self.assertIsNone(result["first_non_drivable_index"])
        self.assertEqual(len(result["samples"]), 3)
        self.assertTrue(all(s["status"] == "drivable" for s in result["samples"]))

    def test_single_off_road_sample_marks_whole_corridor_non_drivable(self):
        wp = FakeWaypoint(lane_type=carla.LaneType.Driving)
        fake_map = FakeMap({0.0: wp, 1.0: None, 2.0: wp})
        points = [carla.Location(x=x, y=0.0, z=0.0) for x in (0.0, 1.0, 2.0)]
        result = check_corridor_drivability(fake_map, points)
        self.assertEqual(result["status"], "non_drivable")
        self.assertEqual(result["first_non_drivable_index"], 1)

    def test_non_drivable_takes_precedence_over_unknown(self):
        wp = FakeWaypoint(lane_type=carla.LaneType.Driving)
        fake_map = FakeMap({0.0: wp, 1.0: "raise", 2.0: None})
        points = [carla.Location(x=x, y=0.0, z=0.0) for x in (0.0, 1.0, 2.0)]
        result = check_corridor_drivability(fake_map, points)
        # A confirmed off-road sample is a stronger conclusion than an
        # unresolved query elsewhere in the same corridor.
        self.assertEqual(result["status"], "non_drivable")
        self.assertEqual(result["first_non_drivable_index"], 2)

    def test_unknown_sample_without_any_non_drivable_reports_unknown(self):
        wp = FakeWaypoint(lane_type=carla.LaneType.Driving)
        fake_map = FakeMap({0.0: wp, 1.0: "raise"})
        points = [carla.Location(x=x, y=0.0, z=0.0) for x in (0.0, 1.0)]
        result = check_corridor_drivability(fake_map, points)
        self.assertEqual(result["status"], "unknown")
        self.assertIsNone(result["first_non_drivable_index"])

    def test_samples_preserve_road_and_lane_identifiers(self):
        wp = FakeWaypoint(lane_type=carla.LaneType.Driving, road_id=42, lane_id=-2)
        fake_map = FakeMap({0.0: wp})
        result = check_corridor_drivability(
            fake_map, [carla.Location(x=0.0, y=0.0, z=0.0)]
        )
        sample = result["samples"][0]
        self.assertEqual(sample["road_id"], 42)
        self.assertEqual(sample["lane_id"], -2)


if __name__ == "__main__":
    unittest.main()
