"""pedestrian_contact.py

Ground-truth ego-pedestrian contact check, independent of CARLA's own
collision sensor.

Built 2026-09-27 after a live-confirmed collision (visually observed:
pedestrian geometry displaced/shoved as the ego drove through it) that
CARLA's `sensor.other.collision` did NOT report (`collision_detected=
False` across two reproducible automated runs). This is a second,
independent confirmation of the same category of gap test3's own code
already documents for near-cross proximity checks ("CARLA's collision
sensor requires a minimum impulse and will NOT fire..."), now shown to
also affect far-cross scenarios, which is exactly the case that existing
comment says the CARLA sensor is "the only reliable detector" for. It
is not.

Two checks are retained. The original conservative circular approximation
is useful when yaw is unavailable, but can over-report lateral contact because
it uses the ego half-length in every direction. The preferred oriented check
uses the vehicle's actual rectangular half-length/half-width and yaw against a
pedestrian circle, resolving that directional false-positive.

This module is purely for POST-HOC / offline verification of whether
contact plausibly occurred -- it does not gate any control decision and
is not itself a proposal to change braking or steering behavior.
"""

import math
from typing import List, Optional, Tuple

from oriented_clearance import (
    oriented_rectangle_circle_clearance_m,
    yaw_deg_to_forward_xy,
)

# Tesla Model 3 bounding-box half-extents, confirmed live via
# vehicle.bounding_box.extent (2026-09-26, Week 3 occupancy work):
# extent.x=2.396m (half-length), extent.y=1.082m (half-width).
EGO_HALF_LENGTH_M = 2.396
EGO_HALF_WIDTH_M = 1.082

# CARLA pedestrian blueprints do not expose a bounding-box radius as
# conveniently as vehicles; 0.3m is a standard adult-body-width estimate,
# matching the radius already used in test3's own near-cross proximity
# fallback (_MAX_LATERAL_M reasoning: "vehicle half-width ~1.0m + ped
# radius ~0.2-0.3m").
DEFAULT_PEDESTRIAN_RADIUS_M = 0.3


def compute_ego_pedestrian_contact_radius_m(
    *,
    ego_half_length_m: float = EGO_HALF_LENGTH_M,
    ego_half_width_m: float = EGO_HALF_WIDTH_M,
    pedestrian_radius_m: float = DEFAULT_PEDESTRIAN_RADIUS_M,
    safety_margin_m: float = 0.0,
) -> float:
    """Conservative circular contact radius: the larger ego half-extent
    (never under-covers regardless of relative heading) plus the
    pedestrian's own radius plus an optional safety margin. Matches
    vehicle_occupancy.compute_actor_occupancy_radius_m's reasoning exactly,
    applied to a pedestrian actor instead of a vehicle actor.
    """
    return max(ego_half_length_m, ego_half_width_m) + pedestrian_radius_m + safety_margin_m


def detect_contact_ticks(
    ego_positions: List[Optional[Tuple[float, float]]],
    pedestrian_positions: List[Optional[Tuple[float, float]]],
    *,
    contact_radius_m: float,
) -> List[int]:
    """Returns the indices where ego and pedestrian centers are within
    contact_radius_m of each other -- a plausible-contact ground truth,
    independent of whether CARLA's own collision sensor fired.

    Either list may contain None entries (position unavailable that tick);
    those ticks are skipped, never treated as contact or as clear.
    ego_positions and pedestrian_positions must be the same length (one
    entry per tick, same indexing).
    """
    if len(ego_positions) != len(pedestrian_positions):
        raise ValueError("ego_positions and pedestrian_positions must be the same length")

    contact_ticks = []
    for i, (ego_pos, ped_pos) in enumerate(zip(ego_positions, pedestrian_positions)):
        if ego_pos is None or ped_pos is None:
            continue
        dist = math.hypot(ego_pos[0] - ped_pos[0], ego_pos[1] - ped_pos[1])
        if dist < contact_radius_m:
            contact_ticks.append(i)
    return contact_ticks


def compute_oriented_ego_pedestrian_clearance_m(
    *,
    ego_x_m: float,
    ego_y_m: float,
    ego_yaw_deg: float,
    pedestrian_x_m: float,
    pedestrian_y_m: float,
    ego_half_length_m: float = EGO_HALF_LENGTH_M,
    ego_half_width_m: float = EGO_HALF_WIDTH_M,
    pedestrian_radius_m: float = DEFAULT_PEDESTRIAN_RADIUS_M,
) -> float:
    """Signed clearance using the ego's actual oriented rectangular body."""
    forward_x, forward_y = yaw_deg_to_forward_xy(ego_yaw_deg)
    return oriented_rectangle_circle_clearance_m(
        rectangle_x_m=ego_x_m,
        rectangle_y_m=ego_y_m,
        forward_x=forward_x,
        forward_y=forward_y,
        half_length_m=ego_half_length_m,
        half_width_m=ego_half_width_m,
        circle_x_m=pedestrian_x_m,
        circle_y_m=pedestrian_y_m,
        circle_radius_m=pedestrian_radius_m,
    )


def detect_oriented_contact_ticks(
    ego_poses: List[Optional[Tuple[float, float, float]]],
    pedestrian_positions: List[Optional[Tuple[float, float]]],
    *,
    contact_tolerance_m: float = 0.0,
) -> List[int]:
    """Return ticks whose oriented footprint clearance indicates contact.

    Each ego pose is ``(x_m, y_m, yaw_deg)``. Missing samples are skipped.
    ``contact_tolerance_m`` may be positive to conservatively count very
    small positive separations as contact.
    """
    if len(ego_poses) != len(pedestrian_positions):
        raise ValueError("ego_poses and pedestrian_positions must be the same length")

    contact_ticks = []
    for index, (ego_pose, pedestrian_position) in enumerate(
        zip(ego_poses, pedestrian_positions)
    ):
        if ego_pose is None or pedestrian_position is None:
            continue
        clearance_m = compute_oriented_ego_pedestrian_clearance_m(
            ego_x_m=ego_pose[0],
            ego_y_m=ego_pose[1],
            ego_yaw_deg=ego_pose[2],
            pedestrian_x_m=pedestrian_position[0],
            pedestrian_y_m=pedestrian_position[1],
        )
        if clearance_m <= contact_tolerance_m:
            contact_ticks.append(index)
    return contact_ticks
