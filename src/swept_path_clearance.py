"""Pure geometry for clearance from an ego-rooted intended path.

This module deliberately does not decide which LiDAR corridor "owns" braking.
It answers one continuous question: does a pedestrian circle overlap the
vehicle-width tube swept by the ego's currently intended center path?
"""

import math
from dataclasses import dataclass

from pedestrian_contact import EGO_HALF_WIDTH_M, DEFAULT_PEDESTRIAN_RADIUS_M


DEFAULT_PATH_CLEAR_MARGIN_M = 0.25
PATH_COMMIT_THRESHOLD_M = 0.2
ACTUAL_MOTION_THRESHOLD_M = 0.05


@dataclass(frozen=True)
class SweptPathClearanceDecision:
    clear_now: bool
    consecutive_clear_ticks: int
    confirmed_clear: bool


@dataclass(frozen=True)
class ActivePathAuthorityDecision:
    """Stable handoff from the original route to the maneuver path."""

    active: bool
    consecutive_commit_ticks: int


def is_active_path_release_blocked(
    *, active_path_authority: bool, path_occupancy_status
) -> bool:
    """Keep braking latched while a committed path is not positively unoccupied."""
    return active_path_authority and path_occupancy_status != "clear"


def should_hold_for_blocked_active_path(
    *, release_blocked: bool, speed_mps: float, hold_entry_speed_mps: float = 1.0
) -> bool:
    """Enter a positive brake hold once a still-blocked path is nearly stopped."""
    if speed_mps < 0.0 or hold_entry_speed_mps <= 0.0:
        raise ValueError("speeds must be non-negative/positive")
    return release_blocked and speed_mps <= hold_entry_speed_mps


def classify_swept_path_lidar_observation(distance_m, trigger_distance_m: float) -> str:
    """Separate sensor detection from the speed-dependent braking decision."""
    if distance_m is None:
        return "no_return"
    if distance_m < trigger_distance_m:
        return "hazard"
    return "detected_beyond_trigger"


def _xy(point):
    if hasattr(point, "x") and hasattr(point, "y"):
        return float(point.x), float(point.y)
    return float(point[0]), float(point[1])


def point_to_segment_distance_m(point_xy, start_xy, end_xy) -> float:
    """Return the shortest 2D distance from one point to a line segment."""
    px, py = _xy(point_xy)
    ax, ay = _xy(start_xy)
    bx, by = _xy(end_xy)
    segment_x = bx - ax
    segment_y = by - ay
    denominator = segment_x * segment_x + segment_y * segment_y
    if denominator <= 1e-12:
        return math.hypot(px - ax, py - ay)
    projection = ((px - ax) * segment_x + (py - ay) * segment_y) / denominator
    projection = max(0.0, min(1.0, projection))
    closest_x = ax + projection * segment_x
    closest_y = ay + projection * segment_y
    return math.hypot(px - closest_x, py - closest_y)


def point_to_polyline_distance_m(point_xy, path_points) -> float:
    """Return shortest 2D distance to a non-empty path polyline."""
    if not path_points:
        raise ValueError("path_points must not be empty")
    if len(path_points) == 1:
        point_x, point_y = _xy(point_xy)
        path_x, path_y = _xy(path_points[0])
        return math.hypot(point_x - path_x, point_y - path_y)
    return min(
        point_to_segment_distance_m(point_xy, start, end)
        for start, end in zip(path_points, path_points[1:])
    )


def build_ego_rooted_path_xy(*, ego_x_m: float, ego_y_m: float, path_points):
    """Prepend the ego's exact current center to an intended path.

    Existing transition-corridor samples are route-relative approximations;
    this makes the safety query explicitly start at the actual ego position.
    A duplicate first sample is omitted.
    """
    rooted = [(float(ego_x_m), float(ego_y_m))]
    for point in path_points or ():
        point_xy = _xy(point)
        if math.hypot(point_xy[0] - rooted[-1][0], point_xy[1] - rooted[-1][1]) > 1e-6:
            rooted.append(point_xy)
    return rooted


def pedestrian_clearance_from_swept_path_m(
    *,
    pedestrian_x_m: float,
    pedestrian_y_m: float,
    path_points,
    ego_half_width_m: float = EGO_HALF_WIDTH_M,
    pedestrian_radius_m: float = DEFAULT_PEDESTRIAN_RADIUS_M,
) -> float:
    """Return signed clearance from pedestrian circle to ego swept tube.

    Positive is separated, zero is touching, negative is overlap. The path
    describes the ego center, so the tube radius is the ego half-width; the
    pedestrian radius is then subtracted from centerline distance.
    """
    center_distance_m = point_to_polyline_distance_m(
        (pedestrian_x_m, pedestrian_y_m), path_points
    )
    return center_distance_m - ego_half_width_m - pedestrian_radius_m


def is_swept_path_confirmed_clear(
    *, clearance_m: float, required_margin_m: float = DEFAULT_PATH_CLEAR_MARGIN_M
) -> bool:
    """True when measured swept-path clearance meets the safety margin."""
    return clearance_m >= required_margin_m


def update_active_path_authority(
    *,
    authority_was_active: bool,
    previous_commit_ticks: int,
    required_commit_ticks: int,
    requested_lateral_offset_m: float,
    actual_lateral_offset_m: float,
    pedestrian_clearance_m,
    pedestrian_clearance_required: bool,
    path_drivability_status,
    required_margin_m: float = DEFAULT_PATH_CLEAR_MARGIN_M,
    commit_threshold_m: float = PATH_COMMIT_THRESHOLD_M,
    actual_motion_threshold_m: float = ACTUAL_MOTION_THRESHOLD_M,
) -> ActivePathAuthorityDecision:
    """Choose one braking path after a safe maneuver-path commitment.

    Before commitment, the original route retains braking authority. Once the
    intended swept path has been physically entered and is geometrically clear
    of the original pedestrian for several ticks, that swept path becomes the
    sole braking path until the lateral maneuver request ends.

    Vehicle occupancy and swept-path LiDAR are intentionally *not* commitment
    gates. They describe hazards on the new active path and must be handled by
    that path's own distance signal; otherwise a parked car on the new path
    incorrectly preserves braking authority for an unrelated old-lane hazard.
    """
    if previous_commit_ticks < 0 or required_commit_ticks <= 0:
        raise ValueError("commit tick counts must be non-negative/positive")

    maneuver_requested = abs(requested_lateral_offset_m) > commit_threshold_m
    if not maneuver_requested:
        return ActivePathAuthorityDecision(active=False, consecutive_commit_ticks=0)

    # Do not reconsider the source every tick after handoff. The earlier
    # corridor-governance experiment visibly chattered because it did exactly
    # that. Authority lasts until the maneuver request returns to center.
    if authority_was_active:
        return ActivePathAuthorityDecision(
            active=True,
            consecutive_commit_ticks=max(previous_commit_ticks, required_commit_ticks),
        )

    tracking_requested_direction = (
        abs(actual_lateral_offset_m) > actual_motion_threshold_m
        and requested_lateral_offset_m * actual_lateral_offset_m > 0.0
    )
    pedestrian_clear = (
        not pedestrian_clearance_required
        or (
            pedestrian_clearance_m is not None
            and is_swept_path_confirmed_clear(
                clearance_m=pedestrian_clearance_m,
                required_margin_m=required_margin_m,
            )
        )
    )
    commit_ready = (
        tracking_requested_direction
        and pedestrian_clear
        and path_drivability_status == "drivable"
    )
    consecutive_commit_ticks = previous_commit_ticks + 1 if commit_ready else 0
    return ActivePathAuthorityDecision(
        active=consecutive_commit_ticks >= required_commit_ticks,
        consecutive_commit_ticks=consecutive_commit_ticks,
    )


def update_swept_path_clearance_decision(
    *,
    previous_clear_ticks: int,
    required_clear_ticks: int,
    requested_lateral_offset_m: float,
    actual_lateral_offset_m: float,
    clearance_m,
    path_drivability_status,
    path_occupancy_status,
    lidar_distance_m,
    trigger_distance_m: float,
    required_margin_m: float = DEFAULT_PATH_CLEAR_MARGIN_M,
    commit_threshold_m: float = PATH_COMMIT_THRESHOLD_M,
    actual_motion_threshold_m: float = ACTUAL_MOTION_THRESHOLD_M,
) -> SweptPathClearanceDecision:
    """Apply all positive-clearance gates and temporal hysteresis.

    ``lidar_distance_m=None`` means no return inside the swept tube. It is
    accepted here only because this experimental gate also requires positive
    ground-truth pedestrian clearance plus explicit map/vehicle checks. It
    must not be reused as a general "no return means safe" object tracker.
    """
    if previous_clear_ticks < 0 or required_clear_ticks <= 0:
        raise ValueError("clear tick counts must be non-negative/positive")

    path_committed = (
        abs(requested_lateral_offset_m) > commit_threshold_m
        and abs(actual_lateral_offset_m) > actual_motion_threshold_m
        and requested_lateral_offset_m * actual_lateral_offset_m > 0.0
    )
    geometry_clear = (
        clearance_m is not None
        and is_swept_path_confirmed_clear(
            clearance_m=clearance_m, required_margin_m=required_margin_m
        )
    )
    lidar_clear = lidar_distance_m is None or lidar_distance_m > trigger_distance_m
    clear_now = (
        path_committed
        and geometry_clear
        and path_drivability_status == "drivable"
        and path_occupancy_status == "clear"
        and lidar_clear
    )
    consecutive_clear_ticks = previous_clear_ticks + 1 if clear_now else 0
    return SweptPathClearanceDecision(
        clear_now=clear_now,
        consecutive_clear_ticks=consecutive_clear_ticks,
        confirmed_clear=consecutive_clear_ticks >= required_clear_ticks,
    )
