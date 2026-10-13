"""
passage_carla.py

CARLA-side adapter for passage.py: builds the lateral drivability probe from
the map and computes the passage across the road at a world location
(Phase 2, 2026-10-05).

The lateral frame is the Driving-lane waypoint nearest the location (the
pedestrian's position for the encounter): origin at its lane center, +right
along its right vector. The ego route follows lane centers, so this matches
the route-relative offsets lane_follow_step uses at that station. Each probe
point uses map_drivability.classify_point_drivability_seam_tolerant, so the
Town04_Opt hairline lane seams are handled at the point level too.
"""

import carla

from map_drivability import classify_point_drivability_seam_tolerant
from passage import Obstacle, compute_passage

_STATUS_TO_PROBE = {"drivable": True, "non_drivable": False}


def passage_frame(carla_map, location):
    """(origin Location, right unit (x, y)) of the nearest Driving-lane center."""
    wp = carla_map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
    right = wp.transform.get_right_vector()
    norm = (right.x ** 2 + right.y ** 2) ** 0.5
    return wp.transform.location, (right.x / norm, right.y / norm)


_CORRIDOR_STYLE = {  # kind -> (thickness m, color); same colors as trajectory_plot's corridor
    "road": (0.03, carla.Color(150, 150, 150)),
    "open": (0.08, carla.Color(0, 200, 0)),   # 0.12 at full brightness glared in the dusk lighting
    "blocked": (0.08, carla.Color(200, 0, 0)),
}


def draw_corridor(debug, segments, *, z, life_time):
    """Draw passage.corridor_segments as CARLA debug lines at height z. Debug
    lines are render-only: they do not touch physics or control."""
    for kind, (x1, y1), (x2, y2) in segments:
        thickness, color = _CORRIDOR_STYLE[kind]
        line_z = z - 0.15 if kind == "road" else z  # road under the bar, or its grey washes the colors out
        debug.draw_line(carla.Location(x=x1, y=y1, z=line_z), carla.Location(x=x2, y=y2, z=line_z),
                        thickness=thickness, color=color, life_time=life_time)


def passage_at_location(carla_map, location, obstacles_xy=(), *, obstacle_laterals_m=(), obstacle_radius_m=None,
                        **passage_kwargs):
    """Passage across the road at `location`. obstacles_xy: world (x, y)
    points (e.g. the pedestrian); they are converted to lateral offsets in
    the same frame. obstacle_laterals_m: obstacles given directly as lateral
    offsets, e.g. passage.predicted_lateral_at_arrival for a walking
    pedestrian. Returns (passage, frame_info dict for manifests).

    For a pedestrian that is not on the ego's lane center, `location` must be
    the point on the ego's route at the pedestrian's line, not the
    pedestrian's own position: the frame snaps to the nearest lane center, so
    a pedestrian in the next lane would shift every offset by a lane width."""
    origin, (rx, ry) = passage_frame(carla_map, location)

    def probe(lateral_m):
        point = carla.Location(x=origin.x + rx * lateral_m, y=origin.y + ry * lateral_m, z=origin.z)
        status = classify_point_drivability_seam_tolerant(carla_map, point)[0]
        return _STATUS_TO_PROBE.get(status)  # "unknown" -> None

    laterals = [(ox - origin.x) * rx + (oy - origin.y) * ry for ox, oy in obstacles_xy] + list(obstacle_laterals_m)
    obstacles = [Obstacle(lat) if obstacle_radius_m is None else Obstacle(lat, obstacle_radius_m) for lat in laterals]
    passage = compute_passage(probe, obstacles, **passage_kwargs)
    frame = {"origin_xyz": [origin.x, origin.y, origin.z], "right_xy": [rx, ry],
             "obstacle_laterals_m": [o.lateral_m for o in obstacles]}
    return passage, frame
