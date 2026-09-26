"""vehicle_occupancy.py

Conservative dynamic-actor occupancy check for candidate lateral paths
(commanded offset / swept transition corridors) -- Week 3 Workstream 1.2.

This is deliberately separate from LiDAR clearance (lidar_utils.py) and
map drivability (map_drivability.py):
- LiDAR clearance answers "is any material return sitting in this
  corridor right now?"
- Map drivability answers "is this corridor geometrically on a driving
  lane at all?"
- This module answers "does another VEHICLE currently occupy this
  corridor?" -- the specific question of whether an escape path is
  blocked by another car, independent of whether the road surface itself
  is present or drivable.

None of the three substitutes for the others; all three are kept as
separate telemetry fields. This check, like the LiDAR candidate/
transition corridors and the map-drivability check before it, is
observational only in its first version: it does not gate braking or
steering, and no other-vehicle actors exist in the current test scenarios
yet -- this module only becomes exercisable once a scenario actually
spawns one (see test13's live validation cases).
"""

import math
from dataclasses import dataclass
from typing import List


@dataclass
class OccupancyActor:
    """One dynamic actor considered for occupancy, already filtered to the
    actor classes that count (see gather_occupancy_actors) and reduced to
    a conservative circular footprint. This module does no CARLA API calls
    and no actor-class filtering itself -- both belong to the live wrapper
    below, keeping the geometry pure and offline-testable.
    """

    actor_id: int
    type_id: str
    x_m: float
    y_m: float
    occupancy_radius_m: float


def compute_actor_occupancy_radius_m(
    *, extent_x_m: float, extent_y_m: float, safety_margin_m: float = 0.3
) -> float:
    """Conservative circular occupancy radius for a vehicle actor, from its
    bounding-box half-extents (carla.BoundingBox.extent.x/.y).

    Uses the LARGER half-extent, not an average or the diagonal, plus a
    safety margin -- this never under-covers the actor's actual footprint
    regardless of its heading relative to the corridor, at the cost of
    being somewhat conservative when the actor is aligned favorably. That
    conservatism is intentional: an "occupied" false-negative (missing a
    real blocker) is a far worse failure mode here than an "occupied"
    false-positive (an overly cautious corridor read), same reasoning
    already applied to the map-drivability and LiDAR corridor checks.
    """
    return max(extent_x_m, extent_y_m) + safety_margin_m


def check_corridor_occupancy(
    corridor_points_world, actors: List[OccupancyActor], *, lateral_half_width_m: float
) -> dict:
    """Classify whether any actor occupies a candidate corridor.

    corridor_points_world: list of points with .x/.y attributes (e.g.
    carla.Location), or None/empty -- same convention as
    map_drivability.check_corridor_drivability, so both checks can be run
    against the exact same corridor sample list.
    actors: list of OccupancyActor (already gathered/filtered).
    lateral_half_width_m: the corridor's own half-width (the same value
    used for its LiDAR/drivability geometry), so all three signals
    describe the same physical band rather than three different widths.

    Returns:
      {
        "status": "clear" | "occupied" | "unknown",
        "occupying_actors": [ {"actor_id", "type_id", "distance_m"}, ... ],
      }

    Conservative, not majority-vote, same spirit as check_corridor_
    drivability:
    - "occupied" if ANY actor's circular footprint comes within
      (lateral_half_width_m + that actor's own occupancy_radius_m) of the
      NEAREST corridor sample point.
    - otherwise "unknown" if the corridor itself has no samples -- absence
      of a corridor is not evidence it is clear (an empty/unset transition
      corridor must never be silently promoted to "clear").
    - otherwise "clear".
    """
    if not corridor_points_world:
        return {"status": "unknown", "occupying_actors": []}

    occupying = []
    for actor in actors:
        threshold_m = lateral_half_width_m + actor.occupancy_radius_m
        min_dist_m = min(
            math.hypot(actor.x_m - point.x, actor.y_m - point.y)
            for point in corridor_points_world
        )
        if min_dist_m < threshold_m:
            occupying.append(
                {"actor_id": actor.actor_id, "type_id": actor.type_id, "distance_m": min_dist_m}
            )

    return {
        "status": "occupied" if occupying else "clear",
        "occupying_actors": occupying,
    }


def gather_occupancy_actors(
    world, ego_vehicle, *, actor_filter: str = "vehicle.*", safety_margin_m: float = 0.3
) -> List[OccupancyActor]:
    """Live wrapper: queries world actors matching actor_filter, excludes
    the ego itself, and reduces each surviving actor to an OccupancyActor.

    Thin and not offline-tested directly (mirrors other live-only harness
    helpers in this codebase, e.g. physics_harness.accelerate_to_matched_
    entry_speed) -- the geometry it feeds is what check_corridor_occupancy
    tests. actor_filter defaults to "vehicle.*" per the advisor's original
    framing ("check whether a car is in its way to swerve"); pedestrians
    and the ego's own sensors are excluded by construction.
    """
    actors = []
    for actor in world.get_actors().filter(actor_filter):
        if actor.id == ego_vehicle.id:
            continue
        transform = actor.get_transform()
        bbox = actor.bounding_box
        radius_m = compute_actor_occupancy_radius_m(
            extent_x_m=bbox.extent.x, extent_y_m=bbox.extent.y, safety_margin_m=safety_margin_m
        )
        actors.append(
            OccupancyActor(
                actor_id=actor.id,
                type_id=actor.type_id,
                x_m=transform.location.x,
                y_m=transform.location.y,
                occupancy_radius_m=radius_m,
            )
        )
    return actors
