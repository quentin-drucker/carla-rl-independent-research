"""
steer_brake_baseline.py

Pure (no CARLA) pieces of the deterministic steering-plus-braking baseline
(test26___steer_brake_baseline.py), offline-tested in
tests/test_steer_brake_baseline.py.

Design ("oracle onset", chosen 2026-10-02): every controller starts reacting
at the SAME predeclared moment -- the scenario trigger, which fires when the
ego is `onset_ttc_s` seconds (at its current speed) from the pedestrian --
using ground truth instead of LiDAR detection. This isolates the physical
question that must be answered before steering enters the RL action space:
is a braking-plus-steering maneuver achievable, and in which conditions does
it succeed where braking alone fails? Perception timing is a separate,
later question (LiDAR-in-the-loop follow-up).

Three matched controller modes:
  no_intervention -- hazard braking disabled for the whole run; the ego keeps
                     cruising (establishes that the encounter is dangerous).
  brake_only      -- from onset: scripted brake target, wheel on the route.
  brake_steer     -- from onset: the same scripted brake target PLUS the
                     existing scripted swerve (test5's
                     HazardClearRecoveryController).
LiDAR hazard detection is disabled in all three (the scripted decision is
False before onset), so pre-onset driving is identical across modes.
"""

import math
from typing import List, Tuple

MODE_NO_INTERVENTION = "no_intervention"
MODE_BRAKE_ONLY = "brake_only"
MODE_BRAKE_STEER = "brake_steer"
MODES = (MODE_NO_INTERVENTION, MODE_BRAKE_ONLY, MODE_BRAKE_STEER)

# Tesla Model 3 footprint, same constants as pedestrian_contact.py.
from pedestrian_contact import EGO_HALF_LENGTH_M, EGO_HALF_WIDTH_M  # noqa: E402


def make_hazard_command_fn(mode: str, *, brake_target: float = 1.0):
    """Returns a run_scenario hazard_command_fn for a mode:
    (sim_time_s, triggered, trigger_time_s) -> (hazard_active, brake_target)."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if not 0.0 <= brake_target <= 1.0:
        raise ValueError("brake_target must be in [0, 1]")

    def hazard_command(sim_time_s, triggered, trigger_time_s):
        if mode == MODE_NO_INTERVENTION:
            return False, 0.0
        return bool(triggered), brake_target

    return hazard_command


def mode_uses_steering(mode: str) -> bool:
    return mode == MODE_BRAKE_STEER


def footprint_corners_xy(
    x_m: float, y_m: float, yaw_deg: float,
    *, half_length_m: float = EGO_HALF_LENGTH_M, half_width_m: float = EGO_HALF_WIDTH_M,
) -> List[Tuple[float, float]]:
    """World-frame (x, y) of the ego footprint's four corners: front-left,
    front-right, rear-right, rear-left. CARLA is left-handed (y to the
    right of +x when viewed from above), so 'right' = forward rotated +90 deg."""
    yaw = math.radians(yaw_deg)
    fx, fy = math.cos(yaw), math.sin(yaw)
    rx, ry = -fy, fx  # right vector in CARLA's left-handed frame
    corners = []
    for lon, lat in ((1, -1), (1, 1), (-1, 1), (-1, -1)):
        corners.append((
            x_m + lon * half_length_m * fx + lat * half_width_m * rx,
            y_m + lon * half_length_m * fy + lat * half_width_m * ry,
        ))
    return corners


def onset_gap_m(*, speed_mps: float, onset_ttc_s: float,
                half_length_m: float = EGO_HALF_LENGTH_M, pedestrian_radius_m: float = 0.3) -> float:
    """Free distance between the ego's front bumper and the pedestrian's
    near edge at onset. run_scenario's trigger measures ego-CENTER distance
    to the encounter point (speed x TTC), so the bumper gap is shorter by
    the half-length plus the pedestrian radius."""
    return speed_mps * onset_ttc_s - half_length_m - pedestrian_radius_m
