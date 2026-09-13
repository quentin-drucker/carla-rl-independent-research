"""Pure geometry helpers for route-relative lateral control.

The CARLA route is represented as objects with ``x`` and ``y`` attributes.
Keeping this module independent of CARLA makes the sign convention and offset
math easy to test without launching the simulator.

Sign convention:
    positive offset -> right of the route's direction of travel
    negative offset -> left of the route's direction of travel
"""

import math


def closest_route_index_xy(route_points, location):
    """Return the index of the route point closest to ``location`` in XY."""
    if not route_points:
        return None

    best_index = None
    best_distance_sq = None
    for index, point in enumerate(route_points):
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


def signed_lateral_offset_m(route_points, location):
    """Measure ``location`` relative to the nearest route point.

    Positive values are to route-right and negative values are to route-left.
    """
    anchor_index = closest_route_index_xy(route_points, location)
    if anchor_index is None:
        raise ValueError("route_points is empty.")

    route_point = route_points[anchor_index]
    right_x, right_y = route_unit_right_xy(route_points, anchor_index)
    displacement_x = location.x - route_point.x
    displacement_y = location.y - route_point.y
    return displacement_x * right_x + displacement_y * right_y
