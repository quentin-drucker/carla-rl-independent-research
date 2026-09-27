"""Pure 2D footprint-clearance geometry.

The ego vehicle is represented by an oriented rectangle and a pedestrian by
a circle.  Unlike the older conservative circular approximation, this keeps
the Model 3's half-length and half-width distinct, so a lateral pass is judged
against the vehicle's width while front/rear approaches still use its length.
"""

import math


def oriented_rectangle_circle_clearance_m(
    *,
    rectangle_x_m: float,
    rectangle_y_m: float,
    forward_x: float,
    forward_y: float,
    half_length_m: float,
    half_width_m: float,
    circle_x_m: float,
    circle_y_m: float,
    circle_radius_m: float,
) -> float:
    """Return signed edge-to-edge clearance between a rectangle and circle.

    Positive values are separation, zero is touching, and negative values are
    overlap. ``forward_x``/``forward_y`` need not already be normalized.
    """
    if half_length_m < 0.0 or half_width_m < 0.0 or circle_radius_m < 0.0:
        raise ValueError("footprint dimensions must be non-negative")

    forward_norm = math.hypot(forward_x, forward_y)
    if forward_norm <= 1e-9:
        raise ValueError("forward vector must have non-zero length")

    unit_forward_x = forward_x / forward_norm
    unit_forward_y = forward_y / forward_norm
    # Either perpendicular convention is valid because the local lateral
    # coordinate is used through abs().
    unit_lateral_x = -unit_forward_y
    unit_lateral_y = unit_forward_x

    relative_x = circle_x_m - rectangle_x_m
    relative_y = circle_y_m - rectangle_y_m
    local_longitudinal_m = (
        relative_x * unit_forward_x + relative_y * unit_forward_y
    )
    local_lateral_m = relative_x * unit_lateral_x + relative_y * unit_lateral_y

    # Standard signed-distance function for an axis-aligned rectangle, after
    # rotating the circle center into the rectangle's local frame.
    q_longitudinal = abs(local_longitudinal_m) - half_length_m
    q_lateral = abs(local_lateral_m) - half_width_m
    outside_distance_m = math.hypot(
        max(q_longitudinal, 0.0), max(q_lateral, 0.0)
    )
    inside_distance_m = min(max(q_longitudinal, q_lateral), 0.0)
    center_to_rectangle_clearance_m = outside_distance_m + inside_distance_m
    return center_to_rectangle_clearance_m - circle_radius_m


def yaw_deg_to_forward_xy(yaw_deg: float):
    """Return a unit XY forward vector for a CARLA/world yaw in degrees."""
    yaw_rad = math.radians(yaw_deg)
    return math.cos(yaw_rad), math.sin(yaw_rad)
