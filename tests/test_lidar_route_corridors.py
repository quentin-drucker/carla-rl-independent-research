import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import carla  # noqa: E402

from lidar_utils import (  # noqa: E402
    lidar_min_distances_along_route_corridors,
    lidar_min_distance_along_transition_corridor,
    lidar_min_along_path_distance_in_swept_tube,
)


class FakeLidar:
    def __init__(self):
        self._transform = carla.Transform()

    def get_transform(self):
        return self._transform


class MultiCorridorLidarTests(unittest.TestCase):
    def test_separates_original_and_shifted_route_returns(self):
        route = [
            carla.Location(x=0, y=0, z=0),
            carla.Location(x=10, y=0, z=0),
            carla.Location(x=20, y=0, z=0),
        ]
        frame = [
            SimpleNamespace(point=carla.Location(x=8, y=0, z=0)),
            SimpleNamespace(point=carla.Location(x=12, y=1.5, z=0)),
        ]

        distances, polylines = lidar_min_distances_along_route_corridors(
            FakeLidar(),
            frame,
            route,
            carla.Location(x=0, y=0, z=0),
            lateral_offsets_m=(0.0, -1.5, 1.5),
            half_width_m=0.2,
            z_min=-1.0,
            z_max=2.5,
            max_dist_m=30.0,
            x_min_m=1.0,
        )

        self.assertAlmostEqual(distances[0.0], 8.0)
        self.assertIsNone(distances[-1.5])
        self.assertAlmostEqual(distances[1.5], math.hypot(12.0, 1.5))
        self.assertAlmostEqual(polylines[1.5][0].y, 1.5)


class TransitionCorridorTests(unittest.TestCase):
    def test_corridor_starts_at_actual_offset_and_blends_to_target(self):
        route = [
            carla.Location(x=0, y=0, z=0),
            carla.Location(x=10, y=0, z=0),
            carla.Location(x=20, y=0, z=0),
            carla.Location(x=30, y=0, z=0),
        ]

        _, polyline = lidar_min_distance_along_transition_corridor(
            FakeLidar(),
            [],
            route,
            carla.Location(x=0, y=0, z=0),
            start_offset_m=0.3,
            target_offset_m=1.5,
            blend_distance_m=20.0,
            half_width_m=0.2,
            z_min=-1.0,
            z_max=2.5,
            max_dist_m=30.0,
            x_min_m=1.0,
        )

        self.assertIsNotNone(polyline)
        # At s=0 the corridor center matches the ego's actual offset.
        self.assertAlmostEqual(polyline[0].y, 0.3)
        # Well beyond the blend distance the corridor holds the target offset.
        self.assertAlmostEqual(polyline[3].y, 1.5)
        # Monotonic blend: the middle point sits strictly between start/target.
        self.assertGreater(polyline[1].y, polyline[0].y)
        self.assertLess(polyline[1].y, polyline[3].y)

    def test_detects_return_only_near_blended_center(self):
        route = [
            carla.Location(x=0, y=0, z=0),
            carla.Location(x=10, y=0, z=0),
            carla.Location(x=20, y=0, z=0),
        ]
        # At s=10 (halfway through a 20m blend from 0.0 -> 2.0) smoothstep(0.5)=0.5,
        # so the corridor center is at y=1.0. A return sitting at y=1.0 there
        # should be picked up; one sitting at the eventual target y=2.0 should not.
        frame = [
            SimpleNamespace(point=carla.Location(x=10, y=1.0, z=0)),
            SimpleNamespace(point=carla.Location(x=10, y=2.0, z=0)),
        ]

        d_min, _ = lidar_min_distance_along_transition_corridor(
            FakeLidar(),
            frame,
            route,
            carla.Location(x=0, y=0, z=0),
            start_offset_m=0.0,
            target_offset_m=2.0,
            blend_distance_m=20.0,
            half_width_m=0.3,
            z_min=-1.0,
            z_max=2.5,
            max_dist_m=30.0,
            x_min_m=1.0,
        )

        self.assertAlmostEqual(d_min, math.hypot(10.0, 1.0))


class EgoRootedSweptTubeTests(unittest.TestCase):
    def test_reports_arc_length_not_sensor_euclidean_distance(self):
        path = [
            carla.Location(x=0, y=0, z=0),
            carla.Location(x=10, y=0, z=0),
            carla.Location(x=10, y=10, z=0),
        ]
        frame = [SimpleNamespace(point=carla.Location(x=10, y=5, z=0))]
        distance = lidar_min_along_path_distance_in_swept_tube(
            FakeLidar(),
            frame,
            path,
            tube_half_width_m=0.5,
            z_min=-1.0,
            z_max=2.5,
            min_along_path_m=1.0,
            max_along_path_m=30.0,
        )
        self.assertAlmostEqual(distance, 15.0)

    def test_turning_path_does_not_require_positive_sensor_x(self):
        path = [
            carla.Location(x=0, y=0, z=0),
            carla.Location(x=0, y=10, z=0),
        ]
        # Sensor-local x is zero, so the old forward-X gate would discard it.
        frame = [SimpleNamespace(point=carla.Location(x=0, y=5, z=0))]
        distance = lidar_min_along_path_distance_in_swept_tube(
            FakeLidar(),
            frame,
            path,
            tube_half_width_m=0.5,
            z_min=-1.0,
            z_max=2.5,
            min_along_path_m=1.0,
            max_along_path_m=30.0,
        )
        self.assertAlmostEqual(distance, 5.0)

    def test_ignores_points_outside_tube_or_behind_minimum_distance(self):
        path = [
            carla.Location(x=0, y=0, z=0),
            carla.Location(x=20, y=0, z=0),
        ]
        frame = [
            SimpleNamespace(point=carla.Location(x=0.5, y=0, z=0)),
            SimpleNamespace(point=carla.Location(x=5, y=2, z=0)),
        ]
        distance = lidar_min_along_path_distance_in_swept_tube(
            FakeLidar(),
            frame,
            path,
            tube_half_width_m=1.0,
            z_min=-1.0,
            z_max=2.5,
            min_along_path_m=1.0,
            max_along_path_m=30.0,
        )
        self.assertIsNone(distance)


if __name__ == "__main__":
    unittest.main()
