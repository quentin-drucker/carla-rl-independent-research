"""map_drivability.py

Conservative CARLA map-based drivable-surface check for candidate lateral
paths (commanded offset / swept transition corridors).

This is deliberately separate from LiDAR clearance (lidar_utils.py):
- LiDAR clearance answers "is anything sitting in this corridor right now?"
- This module answers "is this corridor geometrically on a driving lane at
  all?" (e.g. it would leave the road, cross a curb, or run off the paved
  surface into non-Driving lane types such as sidewalk/shoulder/parking).

A path can be LiDAR-clear and still not be drivable (an empty sidewalk), and
a path can be entirely drivable and still be LiDAR-blocked. Treat these as
two independent requirements -- neither substitutes for the other, and
this check, like the LiDAR candidate/transition corridors, is observational
only in its first version: it does not gate braking or steering.

Uses project_to_road=False for every query. CARLA's default
project_to_road=True silently snaps an off-road query point onto the
nearest road waypoint, which would make an off-road candidate path look
drivable -- exactly the failure mode this check exists to catch.
"""

import carla  # pyright: reportMissingImports=false


def classify_point_drivability(carla_map, location, lane_type=carla.LaneType.Driving):
    """Classify one world point against the map's drivable-lane geometry.

    Returns (status, waypoint):
      "drivable"     -- the point lies on a lane of `lane_type` (waypoint is
                         the matching carla.Waypoint).
      "non_drivable" -- the point does not lie on any lane of `lane_type`
                         (waypoint is None, or a waypoint was found but its
                         lane_type does not match).
      "unknown"      -- the map query itself failed; this is not evidence of
                         either drivability or non-drivability.
    """
    try:
        waypoint = carla_map.get_waypoint(
            location, project_to_road=False, lane_type=lane_type
        )
    except RuntimeError:
        return "unknown", None

    if waypoint is None:
        return "non_drivable", None

    if waypoint.lane_type != lane_type:
        return "non_drivable", waypoint

    return "drivable", waypoint


def check_corridor_drivability(carla_map, corridor_points_world, lane_type=carla.LaneType.Driving):
    """Classify every sample point of a candidate lateral path.

    Returns a dict:
      {
        "status": "drivable" | "non_drivable" | "unknown",
        "samples": [ {location, status, road_id, lane_id, lane_type}, ... ],
        "first_non_drivable_index": int | None,
      }

    The overall status is conservative, not a majority vote:
    - "non_drivable" if ANY sample is non_drivable (a proposed escape path
      must stay on the driving surface along its whole length, not just
      mostly).
    - otherwise "unknown" if ANY sample could not be classified (an unknown
      sample is not proof the path is safe -- same spirit as the LiDAR
      corridors being observational rather than a safety guarantee).
    - otherwise "drivable" only if every sample was classified drivable.

    An empty/None corridor (e.g. the swept-transition corridor when its
    LiDAR precondition wasn't met this tick) reports "unknown" rather than
    "drivable" -- absence of a corridor is not evidence it is safe.
    """
    if not corridor_points_world:
        return {"status": "unknown", "samples": [], "first_non_drivable_index": None}

    samples = []
    saw_unknown = False
    first_non_drivable_index = None

    for index, location in enumerate(corridor_points_world):
        status, waypoint = classify_point_drivability(carla_map, location, lane_type=lane_type)
        samples.append({
            "location": location,
            "status": status,
            "road_id": waypoint.road_id if waypoint is not None else None,
            "lane_id": waypoint.lane_id if waypoint is not None else None,
            "lane_type": str(waypoint.lane_type) if waypoint is not None else None,
        })

        if status == "non_drivable" and first_non_drivable_index is None:
            first_non_drivable_index = index
        elif status == "unknown":
            saw_unknown = True

    if first_non_drivable_index is not None:
        overall = "non_drivable"
    elif saw_unknown:
        overall = "unknown"
    else:
        overall = "drivable"

    return {
        "status": overall,
        "samples": samples,
        "first_non_drivable_index": first_non_drivable_index,
    }
