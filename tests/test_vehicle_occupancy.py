import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import carla  # noqa: E402

from vehicle_occupancy import (  # noqa: E402
    OccupancyActor,
    compute_actor_occupancy_radius_m,
    check_corridor_occupancy,
)


def _corridor(*xs, y=0.0):
    return [carla.Location(x=x, y=y, z=0.0) for x in xs]


def _actor(x_m, y_m, radius_m=1.0, actor_id=1, type_id="vehicle.tesla.model3"):
    return OccupancyActor(actor_id=actor_id, type_id=type_id, x_m=x_m, y_m=y_m, occupancy_radius_m=radius_m)


class ComputeActorOccupancyRadiusTests(unittest.TestCase):
    def test_uses_larger_extent_not_average(self):
        # A long, narrow vehicle (large x extent, small y extent) must not
        # have its footprint underestimated by averaging the two.
        radius = compute_actor_occupancy_radius_m(extent_x_m=2.5, extent_y_m=1.0, safety_margin_m=0.0)
        self.assertAlmostEqual(radius, 2.5)

    def test_adds_safety_margin(self):
        radius = compute_actor_occupancy_radius_m(extent_x_m=2.0, extent_y_m=1.0, safety_margin_m=0.3)
        self.assertAlmostEqual(radius, 2.3)


class CheckCorridorOccupancyTests(unittest.TestCase):
    def test_no_adjacent_vehicle_is_clear(self):
        corridor = _corridor(0.0, 10.0, 20.0, y=1.5)
        result = check_corridor_occupancy(corridor, [], lateral_half_width_m=1.4)
        self.assertEqual(result["status"], "clear")
        self.assertEqual(result["occupying_actors"], [])

    def test_actor_clearly_inside_candidate_path_is_occupied(self):
        # Corridor centered at y=1.5 (the candidate/commanded offset), an
        # actor sitting right on that centerline should be flagged.
        corridor = _corridor(0.0, 10.0, 20.0, y=1.5)
        actor = _actor(x_m=10.0, y_m=1.5, radius_m=1.0)
        result = check_corridor_occupancy(corridor, [actor], lateral_half_width_m=1.4)
        self.assertEqual(result["status"], "occupied")
        self.assertEqual(len(result["occupying_actors"]), 1)
        self.assertEqual(result["occupying_actors"][0]["actor_id"], actor.actor_id)

    def test_actor_just_outside_candidate_path_is_clear(self):
        # Half-width 1.4m + actor radius 1.0m = 2.4m threshold. Placing the
        # actor 3.0m laterally away should read clear, not occupied.
        corridor = _corridor(0.0, 10.0, 20.0, y=0.0)
        actor = _actor(x_m=10.0, y_m=3.0, radius_m=1.0)
        result = check_corridor_occupancy(corridor, [actor], lateral_half_width_m=1.4)
        self.assertEqual(result["status"], "clear")

    def test_actor_at_exact_boundary_is_not_occupied(self):
        # min_dist must be STRICTLY LESS THAN threshold to count as
        # occupied -- a boundary-exact distance is the "just clear" case,
        # not a false negative in the other direction.
        corridor = _corridor(0.0, y=0.0)
        actor = _actor(x_m=0.0, y_m=2.4, radius_m=1.0)  # exactly 2.4m away
        result = check_corridor_occupancy(corridor, [actor], lateral_half_width_m=1.4)
        self.assertEqual(result["status"], "clear")

    def test_deliberately_unavailable_side_reports_unknown_not_clear(self):
        # An empty/None corridor (e.g. a candidate side that was never
        # computed this tick) must never be silently promoted to "clear".
        result = check_corridor_occupancy(None, [_actor(0.0, 0.0)], lateral_half_width_m=1.4)
        self.assertEqual(result["status"], "unknown")
        result_empty = check_corridor_occupancy([], [_actor(0.0, 0.0)], lateral_half_width_m=1.4)
        self.assertEqual(result_empty["status"], "unknown")

    def test_uses_nearest_corridor_sample_not_first(self):
        # An actor near the END of a long corridor must still be detected
        # -- occupancy must check every sample, not just the first.
        corridor = _corridor(0.0, 10.0, 20.0, 30.0, y=0.0)
        actor = _actor(x_m=30.0, y_m=0.0, radius_m=1.0)
        result = check_corridor_occupancy(corridor, [actor], lateral_half_width_m=1.4)
        self.assertEqual(result["status"], "occupied")

    def test_multiple_actors_all_reported_when_occupying(self):
        corridor = _corridor(0.0, 10.0, y=0.0)
        actors = [_actor(x_m=0.0, y_m=0.0, actor_id=1), _actor(x_m=10.0, y_m=0.0, actor_id=2)]
        result = check_corridor_occupancy(corridor, actors, lateral_half_width_m=1.4)
        self.assertEqual(result["status"], "occupied")
        self.assertEqual({a["actor_id"] for a in result["occupying_actors"]}, {1, 2})

    def test_clear_actor_among_occupying_actors_is_not_listed(self):
        corridor = _corridor(0.0, y=0.0)
        occupying = _actor(x_m=0.0, y_m=0.0, actor_id=1)
        clear = _actor(x_m=100.0, y_m=100.0, actor_id=2)
        result = check_corridor_occupancy(corridor, [occupying, clear], lateral_half_width_m=1.4)
        self.assertEqual(result["status"], "occupied")
        self.assertEqual([a["actor_id"] for a in result["occupying_actors"]], [1])


if __name__ == "__main__":
    unittest.main()
