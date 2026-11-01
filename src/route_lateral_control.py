"""Pure geometry helpers for route-relative lateral control.

The CARLA route is represented as objects with ``x`` and ``y`` attributes.
Keeping this module independent of CARLA makes the sign convention and offset
math easy to test without launching the simulator.

Sign convention:
    positive offset -> right of the route's direction of travel
    negative offset -> left of the route's direction of travel
"""

import math


ROUTE_WINDOW_BACK = 10   # route points (~2 m apart) searched behind near_index
ROUTE_WINDOW_AHEAD = 25  # ... and ahead of it (~50 m: far more than a tick's travel, less than a block)


def route_window(n_points, near_index=None):
    """Indices to search for the closest route point: all of them, or only a
    window around near_index (the last match). A route that comes back along
    the same street in the other lane (~4 m away, as on Town01) otherwise snaps
    the car onto the opposite leg and turns it around (2026-10-26)."""
    if near_index is None:
        return range(n_points)
    return range(max(0, near_index - ROUTE_WINDOW_BACK), min(n_points, near_index + ROUTE_WINDOW_AHEAD))


def closest_route_index_xy(route_points, location, near_index=None):
    """Return the index of the route point closest to ``location`` in XY
    (within route_window around near_index when given)."""
    if not route_points:
        return None

    best_index = None
    best_distance_sq = None
    for index in route_window(len(route_points), near_index):
        point = route_points[index]
        dx = point.x - location.x
        dy = point.y - location.y
        distance_sq = dx * dx + dy * dy
        if best_distance_sq is None or distance_sq < best_distance_sq:
            best_index = index
            best_distance_sq = distance_sq

    return best_index


def route_unit_right_xy(route_points, anchor_index):
    """Return the route-relative unit-right vector at ``anchor_index``.

    A centered difference is used inside the route and a one-sided difference
    is used at either endpoint. Degenerate duplicate points are skipped.
    """
    if len(route_points) < 2:
        raise ValueError("At least two route points are required.")
    if not 0 <= anchor_index < len(route_points):
        raise IndexError("anchor_index is outside route_points.")

    for radius in range(1, len(route_points)):
        before_index = max(0, anchor_index - radius)
        after_index = min(len(route_points) - 1, anchor_index + radius)
        before = route_points[before_index]
        after = route_points[after_index]

        tangent_x = after.x - before.x
        tangent_y = after.y - before.y
        magnitude = math.hypot(tangent_x, tangent_y)
        if magnitude > 1e-9:
            tangent_x /= magnitude
            tangent_y /= magnitude
            # CARLA/Unreal uses a left-handed XY coordinate system. For an
            # eastbound tangent (1, 0), route-right is therefore (0, 1).
            return -tangent_y, tangent_x

    raise ValueError("Route points do not define a nonzero XY direction.")


def offset_point_xy(route_points, route_point, lateral_offset_m):
    """Return the XY coordinates of ``route_point`` shifted route-right."""
    anchor_index = closest_route_index_xy(route_points, route_point)
    if anchor_index is None:
        raise ValueError("route_points is empty.")

    right_x, right_y = route_unit_right_xy(route_points, anchor_index)
    return (
        route_point.x + right_x * lateral_offset_m,
        route_point.y + right_y * lateral_offset_m,
    )


def signed_lateral_offset_m(route_points, location, near_index=None):
    """Measure ``location`` relative to the nearest route point (within
    route_window around near_index when given).

    Positive values are to route-right and negative values are to route-left.
    """
    anchor_index = closest_route_index_xy(route_points, location, near_index)
    if anchor_index is None:
        raise ValueError("route_points is empty.")

    route_point = route_points[anchor_index]
    right_x, right_y = route_unit_right_xy(route_points, anchor_index)
    displacement_x = location.x - route_point.x
    displacement_y = location.y - route_point.y
    return displacement_x * right_x + displacement_y * right_y
