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

Drive on (Quentin, 2026-10-22): a car that has STOPPED beside or before the
hazard drives on along its current swerve path once that path is clear of the
hazard by a margin (now and over the next couple of seconds, so a hazard
walking toward the path still blocks it). Hazard mode ends and the swerve
offset is held until the hazard is behind the car; then the return above. If
the path becomes blocked again while driving on, hazard mode comes back
(brake, policy in control): the hook for a second hazard on the swerve path,
which later may branch into a new swerve. While the car is still moving the
policy keeps control, so the rule never overrides an avoidance in progress.

Hazard-generic: "behind" means past the ego's rear bumper by the hazard's own
half-length plus a margin, and the path check uses the hazard's lateral
extent, so a parked car (half-length ~2.4 m) works the same way as a
pedestrian (radius 0.3 m).

Pure (no CARLA); offline-tested in tests/test_merge_back.py.
"""

from typing import Optional, Tuple

from oriented_clearance import yaw_deg_to_forward_xy
from pedestrian_contact import EGO_HALF_LENGTH_M, EGO_HALF_WIDTH_M

DEFAULT_CLEAR_MARGIN_M = 1.0
DRIVE_ON_SPEED_MPS = 0.5  # "stopped" for driving on (same threshold as protocol v4's stopped car)


def hazard_behind_ego(*, ego_x_m: float, ego_y_m: float, ego_yaw_deg: float, hazard_x_m: float, hazard_y_m: float,
                      hazard_half_length_m: float, margin_m: float = DEFAULT_CLEAR_MARGIN_M) -> bool:
    """True once the hazard's nearest point is more than margin_m behind the
    ego's rear bumper, measured along the ego's heading."""
    fx, fy = yaw_deg_to_forward_xy(ego_yaw_deg)
    along = (hazard_x_m - ego_x_m) * fx + (hazard_y_m - ego_y_m) * fy
    return along < -(EGO_HALF_LENGTH_M + hazard_half_length_m + margin_m)


def path_clear_of_hazard(*, path_lo_m: float, path_hi_m: float, hazard_lateral_m: float,
                         hazard_lateral_speed_mps: float, hazard_radius_m: float,
                         margin_m: float = DEFAULT_CLEAR_MARGIN_M, horizon_s: float = 2.0, step_s: float = 0.25) -> bool:
    """True if a hazard stays more than margin_m (body edge to hazard edge) from
    the lateral strip the ego's center sweeps between path_lo_m and path_hi_m
    (route-relative, +right), now and at constant lateral speed for horizon_s."""
    lo, hi = min(path_lo_m, path_hi_m), max(path_lo_m, path_hi_m)
    reach = EGO_HALF_WIDTH_M + hazard_radius_m + margin_m
    for k in range(int(round(horizon_s / step_s)) + 1):
        lateral = hazard_lateral_m + hazard_lateral_speed_mps * k * step_s
        if lo - reach < lateral < hi + reach:
            return False
    return True


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
        self.clear_since_s: Optional[float] = None       # hazard behind the car since
        self._path_clear_since_s: Optional[float] = None
        self._drive_on_target_m: Optional[float] = None
        self.drive_on_start_s: Optional[float] = None    # latest start of driving on
        self.drive_on_count = 0
        self.return_start_s: Optional[float] = None
        self.return_duration_s: Optional[float] = None
        self.return_from_m = 0.0

    @property
    def state(self) -> str:
        if self.return_start_s is not None:
            return "returning" if self._fraction < 1.0 else "returned"
        return "avoiding" if self._drive_on_target_m is None else "driving_on"

    def aim_m(self, target_m: float) -> float:
        """The lateral target the car is following: the held swerve while driving on."""
        return target_m if self._drive_on_target_m is None else self._drive_on_target_m

    def summary(self) -> dict:
        return {"state": self.state, "drive_on_start_s": self.drive_on_start_s, "drive_on_count": self.drive_on_count,
                "return_start_s": self.return_start_s, "return_duration_s": self.return_duration_s}

    def _confirmed(self, since: Optional[float], t_s: float) -> bool:
        return since is not None and t_s - since >= self.clear_confirm_s - 1e-9

    def update(self, t_s: float, *, hazard_behind: bool, target_m: float, path_clear: bool = False,
               ego_stopped: bool = False) -> Tuple[float, bool]:
        if self.return_start_s is None:
            held = target_m if self._drive_on_target_m is None else self._drive_on_target_m
            self.clear_since_s = (self.clear_since_s if self.clear_since_s is not None else t_s) if hazard_behind else None
            if self._confirmed(self.clear_since_s, t_s):
                # ponytail: once returning it keeps returning; a hazard behind a moving car cannot get back
                # in front of it. Check the return path too (as driving on does) when a second hazard is added.
                self.return_start_s, self.return_from_m = t_s, held
                self.return_duration_s = max(self.min_return_s, abs(held) / self.return_rate_mps)
            elif self._drive_on_target_m is not None:
                if path_clear:
                    return held, False
                self._drive_on_target_m = self._path_clear_since_s = None  # blocked again: hazard mode
                return target_m, True
            else:
                stopped_and_clear = ego_stopped and path_clear
                self._path_clear_since_s = (self._path_clear_since_s if self._path_clear_since_s is not None
                                            else t_s) if stopped_and_clear else None
                if self._confirmed(self._path_clear_since_s, t_s):
                    self._drive_on_target_m, self.drive_on_start_s = target_m, t_s
                    self.drive_on_count += 1
                    return target_m, False
                return target_m, True
        self._fraction = (t_s - self.return_start_s) / self.return_duration_s
        return self.return_from_m * (1.0 - _smoothstep(self._fraction)), False
