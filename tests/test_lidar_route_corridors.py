import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import carla  # noqa: E402

from lidar_utils import lidar_min_distances_along_route_corridors  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
