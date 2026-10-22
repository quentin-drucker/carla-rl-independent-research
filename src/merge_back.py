"""
merge_back.py

Return to the original lane after avoiding a hazard ("merge-back", Option A,
Quentin 2026-10-22). The learned policy decides only the avoidance (brake and
where to aim); once the hazard is confirmed clear BEHIND the car, the
classical controller takes over again:
  - hazard mode ends, so lane_follow_step releases the brake and resumes
    cruise (its HAZARD_BRAKE/STOP_HOLD -> RECOVER -> CRUISE path);
  - the lateral target returns from wherever the avoidance left it to the
    route center (0) along a smoothstep, no faster than return_rate_mps.
Same idea as test5's HazardClearRecoveryController (confirm clear, then a
smooth return), which instead also schedules its own fixed swerve.

Hazard-generic: "clear" means the hazard is past the ego's rear bumper by its
own half-length plus a margin, so a parked car (half-length ~2.4 m) works
the same way as a pedestrian (radius 0.3 m). A hazard the car stops short of
never gets behind it, so a stopped car waits (it does not drive on around).

Pure (no CARLA); offline-tested in tests/test_merge_back.py.
"""

from typing import Optional, Tuple

from oriented_clearance import yaw_deg_to_forward_xy
from pedestrian_contact import EGO_HALF_LENGTH_M

DEFAULT_CLEAR_MARGIN_M = 1.0


def hazard_behind_ego(*, ego_x_m: float, ego_y_m: float, ego_yaw_deg: float, hazard_x_m: float, hazard_y_m: float,
                      hazard_half_length_m: float, margin_m: float = DEFAULT_CLEAR_MARGIN_M) -> bool:
    """True once the hazard's nearest point is more than margin_m behind the
    ego's rear bumper, measured along the ego's heading."""
    fx, fy = yaw_deg_to_forward_xy(ego_yaw_deg)
    along = (hazard_x_m - ego_x_m) * fx + (hazard_y_m - ego_y_m) * fy
    return along < -(EGO_HALF_LENGTH_M + hazard_half_length_m + margin_m)


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


class MergeBack:
    """One episode. update() each tick after onset with the policy's lateral
    target; returns (lateral target to use, hazard mode still on)."""

    def __init__(self, *, clear_confirm_s: float = 0.5, return_rate_mps: float = 1.5, min_return_s: float = 2.0):
        if clear_confirm_s < 0.0 or return_rate_mps <= 0.0 or min_return_s <= 0.0:
            raise ValueError("bad merge-back timing")
        self.clear_confirm_s, self.return_rate_mps, self.min_return_s = clear_confirm_s, return_rate_mps, min_return_s
        self.clear_since_s: Optional[float] = None
        self.return_start_s: Optional[float] = None
        self.return_duration_s: Optional[float] = None
        self.return_from_m = 0.0

    @property
    def state(self) -> str:
        if self.return_start_s is None:
            return "avoiding"
        return "returning" if self._fraction < 1.0 else "returned"

    def update(self, t_s: float, *, hazard_behind: bool, target_m: float) -> Tuple[float, bool]:
        if self.return_start_s is None:
            if not hazard_behind:
                self.clear_since_s = None
                return target_m, True
            if self.clear_since_s is None:
                self.clear_since_s = t_s
            if t_s - self.clear_since_s < self.clear_confirm_s - 1e-9:
                return target_m, True
            # ponytail: once returning it keeps returning; a hazard behind a moving car cannot get back
            # in front of it. Re-arm (as test5 does) if a hazard that moves faster than the car is added.
            self.return_start_s, self.return_from_m = t_s, target_m
            self.return_duration_s = max(self.min_return_s, abs(target_m) / self.return_rate_mps)
        self._fraction = (t_s - self.return_start_s) / self.return_duration_s
        return self.return_from_m * (1.0 - _smoothstep(self._fraction)), False
