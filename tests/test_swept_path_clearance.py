import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swept_path_clearance import (  # noqa: E402
    build_ego_rooted_path_xy,
    classify_swept_path_lidar_observation,
    is_active_path_release_blocked,
    is_swept_path_confirmed_clear,
    pedestrian_clearance_from_swept_path_m,
    point_to_polyline_distance_m,
    should_hold_for_blocked_active_path,
    update_active_path_authority,
    update_swept_path_clearance_decision,
)


class SweptPathGeometryTests(unittest.TestCase):
    def test_distance_projects_to_segment_not_only_samples(self):
        self.assertAlmostEqual(
            point_to_polyline_distance_m((5.0, 2.0), [(0.0, 0.0), (10.0, 0.0)]),
            2.0,
        )

    def test_ego_root_is_exact_and_duplicate_is_removed(self):
        path = build_ego_rooted_path_xy(
            ego_x_m=1.0,
            ego_y_m=2.0,
            path_points=[SimpleNamespace(x=1.0, y=2.0), SimpleNamespace(x=5.0, y=3.0)],
        )
        self.assertEqual(path, [(1.0, 2.0), (5.0, 3.0)])

    def test_stationary_pedestrian_is_clear_of_sufficient_offset_path(self):
        clearance = pedestrian_clearance_from_swept_path_m(
            pedestrian_x_m=10.0,
            pedestrian_y_m=0.0,
            path_points=[(0.0, 1.682), (20.0, 1.682)],
        )
        self.assertAlmostEqual(clearance, 0.3)
        self.assertTrue(is_swept_path_confirmed_clear(clearance_m=clearance))

    def test_undersized_offset_path_is_not_clear(self):
        clearance = pedestrian_clearance_from_swept_path_m(
            pedestrian_x_m=10.0,
            pedestrian_y_m=0.0,
            path_points=[(0.0, 0.5), (20.0, 0.5)],
        )
        self.assertLess(clearance, 0.0)
        self.assertFalse(is_swept_path_confirmed_clear(clearance_m=clearance))


class SweptPathClearanceDecisionTests(unittest.TestCase):
    def evaluate(self, **overrides):
        values = {
            "previous_clear_ticks": 0,
            "required_clear_ticks": 10,
            "requested_lateral_offset_m": 1.7,
            "actual_lateral_offset_m": 1.4,
            "clearance_m": 0.3,
            "path_drivability_status": "drivable",
            "path_occupancy_status": "clear",
            "lidar_distance_m": None,
            "trigger_distance_m": 30.0,
        }
        values.update(overrides)
        return update_swept_path_clearance_decision(**values)

    def test_requires_consecutive_ticks_before_confirmation(self):
        ninth = self.evaluate(previous_clear_ticks=8)
        tenth = self.evaluate(previous_clear_ticks=9)
        self.assertFalse(ninth.confirmed_clear)
        self.assertTrue(tenth.confirmed_clear)

    def test_lidar_hazard_on_swept_path_vetoes_clearance(self):
        result = self.evaluate(previous_clear_ticks=9, lidar_distance_m=12.0)
        self.assertFalse(result.clear_now)
        self.assertEqual(result.consecutive_clear_ticks, 0)

    def test_unsafe_geometry_resets_latch(self):
        result = self.evaluate(previous_clear_ticks=9, clearance_m=-0.1)
        self.assertFalse(result.confirmed_clear)
        self.assertEqual(result.consecutive_clear_ticks, 0)

    def test_uncommitted_path_cannot_suppress_braking(self):
        self.assertFalse(
            self.evaluate(
                requested_lateral_offset_m=0.1,
                actual_lateral_offset_m=0.1,
            ).clear_now
        )

    def test_command_without_physical_tracking_cannot_suppress_braking(self):
        self.assertFalse(
            self.evaluate(actual_lateral_offset_m=0.0).clear_now
        )
        self.assertFalse(
            self.evaluate(actual_lateral_offset_m=-1.4).clear_now
        )

    def test_large_tracking_error_is_encoded_by_ego_rooted_path_not_rejected_twice(self):
        self.assertTrue(
            self.evaluate(actual_lateral_offset_m=0.1).clear_now
        )

    def test_unknown_or_blocked_path_cannot_suppress_braking(self):
        self.assertFalse(
            self.evaluate(path_drivability_status="unknown").clear_now
        )
        self.assertFalse(
            self.evaluate(path_occupancy_status="occupied").clear_now
        )


class ActivePathAuthorityTests(unittest.TestCase):
    def evaluate(self, **overrides):
        values = {
            "authority_was_active": False,
            "previous_commit_ticks": 0,
            "required_commit_ticks": 5,
            "requested_lateral_offset_m": 4.8,
            "actual_lateral_offset_m": 4.6,
            "pedestrian_clearance_m": 2.0,
            "pedestrian_clearance_required": True,
            "path_drivability_status": "drivable",
        }
        values.update(overrides)
        return update_active_path_authority(**values)

    def test_requires_stable_commitment_before_handoff(self):
        fourth = self.evaluate(previous_commit_ticks=3)
        fifth = self.evaluate(previous_commit_ticks=4)
        self.assertFalse(fourth.active)
        self.assertTrue(fifth.active)

    def test_unsafe_pedestrian_geometry_prevents_handoff(self):
        result = self.evaluate(previous_commit_ticks=4, pedestrian_clearance_m=-0.1)
        self.assertFalse(result.active)
        self.assertEqual(result.consecutive_commit_ticks, 0)

    def test_wrong_direction_or_non_drivable_path_prevents_handoff(self):
        self.assertFalse(self.evaluate(actual_lateral_offset_m=-4.0).active)
        self.assertFalse(
            self.evaluate(path_drivability_status="non_drivable").active
        )

    def test_authority_stays_stable_during_committed_maneuver(self):
        result = self.evaluate(
            authority_was_active=True,
            previous_commit_ticks=5,
            actual_lateral_offset_m=0.0,
            pedestrian_clearance_m=None,
            path_drivability_status="unknown",
        )
        self.assertTrue(result.active)

    def test_occupied_or_unknown_active_path_blocks_brake_release(self):
        self.assertTrue(
            is_active_path_release_blocked(
                active_path_authority=True,
                path_occupancy_status="occupied",
            )
        )
        self.assertTrue(
            is_active_path_release_blocked(
                active_path_authority=True,
                path_occupancy_status="unknown",
            )
        )

    def test_clear_or_inactive_path_does_not_block_release(self):
        self.assertFalse(
            is_active_path_release_blocked(
                active_path_authority=True,
                path_occupancy_status="clear",
            )
        )
        self.assertFalse(
            is_active_path_release_blocked(
                active_path_authority=False,
                path_occupancy_status="occupied",
            )
        )

    def test_blocked_path_enters_hold_only_at_low_speed(self):
        self.assertTrue(
            should_hold_for_blocked_active_path(
                release_blocked=True, speed_mps=0.8
            )
        )
        self.assertFalse(
            should_hold_for_blocked_active_path(
                release_blocked=True, speed_mps=2.0
            )
        )
        self.assertFalse(
            should_hold_for_blocked_active_path(
                release_blocked=False, speed_mps=0.1
            )
        )

    def test_authority_returns_to_route_when_maneuver_request_ends(self):
        result = self.evaluate(
            authority_was_active=True,
            previous_commit_ticks=5,
            requested_lateral_offset_m=0.0,
        )
        self.assertFalse(result.active)
        self.assertEqual(result.consecutive_commit_ticks, 0)

    def test_pedestrian_free_maneuver_can_own_active_path(self):
        result = self.evaluate(
            previous_commit_ticks=4,
            pedestrian_clearance_m=None,
            pedestrian_clearance_required=False,
        )
        self.assertTrue(result.active)


class SweptPathLidarObservationTests(unittest.TestCase):
    def test_no_return_is_distinct_from_detected_clear(self):
        self.assertEqual(
            classify_swept_path_lidar_observation(None, 30.0), "no_return"
        )

    def test_detection_beyond_dynamic_trigger_is_reported(self):
        self.assertEqual(
            classify_swept_path_lidar_observation(34.0, 25.0),
            "detected_beyond_trigger",
        )

    def test_detection_inside_dynamic_trigger_is_hazard(self):
        self.assertEqual(
            classify_swept_path_lidar_observation(24.0, 25.0), "hazard"
        )

    def test_far_cross_pedestrian_on_escape_path_is_not_clear(self):
        clearance = pedestrian_clearance_from_swept_path_m(
            pedestrian_x_m=10.0,
            pedestrian_y_m=2.55,
            path_points=[(0.0, 1.5), (20.0, 1.5)],
        )
        self.assertLess(clearance, 0.0)
        self.assertFalse(is_swept_path_confirmed_clear(clearance_m=clearance))

    def test_ego_root_prevents_skipping_current_transition(self):
        path = build_ego_rooted_path_xy(
            ego_x_m=0.0,
            ego_y_m=0.0,
            path_points=[(10.0, 2.0), (20.0, 2.0)],
        )
        # Pedestrian lies on the transition from the actual ego position to
        # the first future path sample, even though it is far from y=2.
        clearance = pedestrian_clearance_from_swept_path_m(
            pedestrian_x_m=5.0,
            pedestrian_y_m=1.0,
            path_points=path,
        )
        self.assertLess(clearance, 0.0)


if __name__ == "__main__":
    unittest.main()
