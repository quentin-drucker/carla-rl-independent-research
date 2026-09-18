import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hazard_governance import (  # noqa: E402
    select_hazard_governing_distance,
    is_original_corridor_confirmed_clear,
)

DRIVABLE = {"status": "drivable"}
NON_DRIVABLE = {"status": "non_drivable"}
UNKNOWN = {"status": "unknown"}


class SelectHazardGoverningDistanceTests(unittest.TestCase):
    def test_defaults_to_original_when_not_monitoring_corridors(self):
        distance, source = select_hazard_governing_distance(
            d_min_original_path_m=10.0,
            d_min_transition_path_m=100.0,
            transition_path_drivability=DRIVABLE,
            requested_lateral_offset_m=1.5,
            trigger_distance_m=20.0,
            monitor_lateral_corridors=False,
        )
        self.assertEqual(distance, 10.0)
        self.assertEqual(source, "original")

    def test_defaults_to_original_when_offset_is_negligible(self):
        distance, source = select_hazard_governing_distance(
            d_min_original_path_m=10.0,
            d_min_transition_path_m=100.0,
            transition_path_drivability=DRIVABLE,
            requested_lateral_offset_m=0.05,  # below the commit threshold
            trigger_distance_m=20.0,
            monitor_lateral_corridors=True,
        )
        self.assertEqual(distance, 10.0)
        self.assertEqual(source, "original")

    def test_shifts_to_transition_when_clear_and_drivable_and_committed(self):
        distance, source = select_hazard_governing_distance(
            d_min_original_path_m=10.0,   # would be a hazard on its own
            d_min_transition_path_m=100.0,  # transition path is far clearer
            transition_path_drivability=DRIVABLE,
            requested_lateral_offset_m=1.5,
            trigger_distance_m=20.0,
            monitor_lateral_corridors=True,
        )
        self.assertEqual(distance, 100.0)
        self.assertEqual(source, "transition")

    def test_none_transition_reading_counts_as_clear(self):
        # No LiDAR return on the transition corridor at all -- same
        # "nothing detected = clear" convention the original corridor uses.
        distance, source = select_hazard_governing_distance(
            d_min_original_path_m=10.0,
            d_min_transition_path_m=None,
            transition_path_drivability=DRIVABLE,
            requested_lateral_offset_m=1.5,
            trigger_distance_m=20.0,
            monitor_lateral_corridors=True,
        )
        self.assertIsNone(distance)
        self.assertEqual(source, "transition")

    def test_falls_back_to_original_when_transition_lidar_not_clear(self):
        distance, source = select_hazard_governing_distance(
            d_min_original_path_m=10.0,
            d_min_transition_path_m=15.0,  # still inside trigger_distance_m
            transition_path_drivability=DRIVABLE,
            requested_lateral_offset_m=1.5,
            trigger_distance_m=20.0,
            monitor_lateral_corridors=True,
        )
        self.assertEqual(distance, 10.0)
        self.assertEqual(source, "original")

    def test_falls_back_to_original_when_transition_non_drivable(self):
        distance, source = select_hazard_governing_distance(
            d_min_original_path_m=10.0,
            d_min_transition_path_m=100.0,  # LiDAR-clear
            transition_path_drivability=NON_DRIVABLE,  # but off-road
            requested_lateral_offset_m=1.5,
            trigger_distance_m=20.0,
            monitor_lateral_corridors=True,
        )
        self.assertEqual(distance, 10.0)
        self.assertEqual(source, "original")

    def test_falls_back_to_original_when_drivability_unknown(self):
        # An unresolved map query is not proof of safety.
        distance, source = select_hazard_governing_distance(
            d_min_original_path_m=10.0,
            d_min_transition_path_m=100.0,
            transition_path_drivability=UNKNOWN,
            requested_lateral_offset_m=1.5,
            trigger_distance_m=20.0,
            monitor_lateral_corridors=True,
        )
        self.assertEqual(distance, 10.0)
        self.assertEqual(source, "original")

    def test_falls_back_to_original_when_drivability_missing(self):
        distance, source = select_hazard_governing_distance(
            d_min_original_path_m=10.0,
            d_min_transition_path_m=100.0,
            transition_path_drivability=None,
            requested_lateral_offset_m=1.5,
            trigger_distance_m=20.0,
            monitor_lateral_corridors=True,
        )
        self.assertEqual(distance, 10.0)
        self.assertEqual(source, "original")

    def test_negative_offset_also_commits(self):
        # A route-left maneuver (negative offset) must commit just as a
        # route-right one does -- this must not be sign-sensitive.
        distance, source = select_hazard_governing_distance(
            d_min_original_path_m=10.0,
            d_min_transition_path_m=100.0,
            transition_path_drivability=DRIVABLE,
            requested_lateral_offset_m=-1.5,
            trigger_distance_m=20.0,
            monitor_lateral_corridors=True,
        )
        self.assertEqual(distance, 100.0)
        self.assertEqual(source, "transition")


class IsOriginalCorridorConfirmedClearTests(unittest.TestCase):
    def test_none_reading_is_not_clear(self):
        # The core fix: unlike lane_follow_step()'s own hazard_clear check
        # (which treats None as "nothing detected = clear" outside
        # STOP_HOLD), this must always treat a missing LiDAR return as
        # unknown -- never as confirmation of safety.
        self.assertFalse(
            is_original_corridor_confirmed_clear(None, 20.0, 1.0)
        )

    def test_confirmed_distant_reading_is_clear(self):
        self.assertTrue(
            is_original_corridor_confirmed_clear(30.0, 20.0, 1.0)
        )

    def test_reading_inside_margin_is_not_clear(self):
        self.assertFalse(
            is_original_corridor_confirmed_clear(20.5, 20.0, 1.0)
        )

    def test_reading_exactly_at_margin_boundary_is_not_clear(self):
        self.assertFalse(
            is_original_corridor_confirmed_clear(21.0, 20.0, 1.0)
        )


if __name__ == "__main__":
    unittest.main()
