"""
rl_steering.py

Pure (no CARLA) pieces of the steering RL environment (Phase 2, Week 5 plan
step 4a, 2026-10-10). CarlaAEBEnv wires them in as an opt-in mode in step 4b;
the default braking-only mode does not use this file.

Design (MASTER, "Phase 2 design decisions"):
  Action       two numbers per tick: brake in [0, 1] and the passage
               coordinate u (passage.py). Right swerves only for now
               (u in [0, 1]); the bounds are parameters, so [-1, 1] is a
               setting change.
  Commitment   "per_tick": the passage and u are re-decided every tick.
               "lock_at_onset": the ROUTE is decided once, at the first tick
               after the hazard starts, and frozen: both the passage and the
               lateral target stay fixed for the rest of the episode. Only
               the route is locked; braking stays per tick (decision 6).
  Observation  last semester's 6 values plus 8 steering values.
  Reward       ordered by weight, most important first:
                 1. no contact with the pedestrian
                 2. if contact, the lower the impact speed the better
                 3. the whole car stays on the road
                 4. enough clearance, counted only up to
                    enough_clearance_m (more than enough earns nothing, so
                    the gentlest swerve that clears is preferred)
                 5. ride comfort (jerk, brake changes, lateral acceleration)
               1-4 are "safety", 5 is "comfort"; they are reported separately.
               The ordering is by weight size, not strictly lexicographic.
               PROPOSED 2026-10-10, not yet confirmed by Quentin / Prof.
               Izmirli: every number is a SteeringRewardConfig field.

Offline-tested in tests/test_rl_steering.py.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from passage import DEFAULT_SCAN_MAX_M, DEFAULT_SIDE_MARGIN_M, Obstacle, Passage, passage_coordinate_to_offset_m
from pedestrian_contact import EGO_HALF_WIDTH_M
from rl_reward_design import STATE_DIM, W_JERK, W_SMOOTH_ACTION

STEER_COMMIT_PER_TICK = "per_tick"
STEER_COMMIT_LOCK_AT_ONSET = "lock_at_onset"
STEER_COMMIT_MODES = (STEER_COMMIT_PER_TICK, STEER_COMMIT_LOCK_AT_ONSET)

ACTION_DIM = 2  # [brake, u]


# =============================================================================
# Action
# =============================================================================

def split_action(action: Sequence[float], *, u_min: float = 0.0, u_max: float = 1.0) -> Tuple[float, float]:
    """(brake, u) from the policy's 2-number action, each clipped to its range.
    A non-finite value raises: it means training has diverged, and clipping
    it would hide that while steering the car on garbage."""
    brake, u = float(action[0]), float(action[1])
    if not (math.isfinite(brake) and math.isfinite(u)):
        raise ValueError(f"non-finite action {list(action)!r}")
    return min(max(brake, 0.0), 1.0), min(max(u, u_min), u_max)


# =============================================================================
# Commitment (the lock)
# =============================================================================

class SteeringCommitment:
    """Turns the policy's u into the lateral target in meters, one episode.

    Before the hazard starts the target is 0 (the normal lane path). After:
      per_tick       target = u mapped through the CURRENT passage, every tick.
      lock_at_onset  on the first post-onset tick that has a passage, u and
                     the passage are recorded and the target computed once;
                     later u values and later passages are ignored.
                     lock_delay_ticks postpones that tick: until then the
                     target stays 0 (a walking pedestrian has zero speed at
                     onset, so a route locked on that tick is chosen blind).
    A post-onset tick with no passage keeps the last target (0 if there has
    never been one) and is counted in missing_passage_ticks, so the car never
    silently swerves on nothing.
    """

    def __init__(self, mode: str, *, u_min: float = 0.0, u_max: float = 1.0, lock_delay_ticks: int = 0):
        if mode not in STEER_COMMIT_MODES:
            raise ValueError(f"steer_commit must be one of {STEER_COMMIT_MODES}, got {mode!r}")
        if lock_delay_ticks < 0:
            raise ValueError("lock_delay_ticks must be >= 0")
        self.mode, self.u_min, self.u_max = mode, u_min, u_max
        self._ticks_until_lock = lock_delay_ticks if mode == STEER_COMMIT_LOCK_AT_ONSET else 0
        self.locked = False
        self.u_applied = 0.0          # the u behind the current target ("previous u" in the observation)
        self.passage: Optional[Passage] = None  # the passage behind the current target
        self.target_m = 0.0
        self.missing_passage_ticks = 0

    def target_offset_m(self, u: float, passage: Optional[Passage], *, triggered: bool) -> float:
        if not triggered or self.locked:
            return self.target_m
        if self._ticks_until_lock > 0:
            self._ticks_until_lock -= 1
            return self.target_m
        if passage is None:
            self.missing_passage_ticks += 1
            return self.target_m
        self.u_applied = min(max(float(u), self.u_min), self.u_max)
        self.passage = passage
        self.target_m = passage_coordinate_to_offset_m(self.u_applied, passage, u_min=self.u_min, u_max=self.u_max)
        self.locked = self.mode == STEER_COMMIT_LOCK_AT_ONSET
        return self.target_m

    def observed_passage(self, current_passage: Optional[Passage]) -> Optional[Passage]:
        """The passage the policy should see: the frozen one once locked,
        otherwise the current one."""
        return self.passage if self.locked else current_passage


# =============================================================================
# Passage-frame geometry (CARLA is left-handed: right = forward rotated +90 deg)
# =============================================================================

def frame_coordinates_m(origin_xy, right_xy, point_xy) -> Tuple[float, float]:
    """(along_m, lateral_m) of a world point in the passage frame: along is
    measured in the direction of travel from the pedestrian's line (negative =
    not there yet), lateral is +right of the lane center."""
    dx, dy = point_xy[0] - origin_xy[0], point_xy[1] - origin_xy[1]
    rx, ry = right_xy
    return dx * ry + dy * -rx, dx * rx + dy * ry


def heading_error_rad(ego_yaw_deg: float, right_xy) -> float:
    """Signed angle between the ego's heading and the road's direction at the
    passage frame: 0 = parallel, positive = pointing to the right."""
    ex, ey = math.cos(math.radians(ego_yaw_deg)), math.sin(math.radians(ego_yaw_deg))
    rx, ry = right_xy
    return math.atan2(ex * rx + ey * ry, ex * ry + ey * -rx)


# =============================================================================
# Observation
# =============================================================================

LATERAL_SCALE_M = DEFAULT_SCAN_MAX_M        # 12 m: how far the passage looks either side
PEDESTRIAN_SPEED_SCALE_MPS = 3.0            # a fast walk; the scripted walker does 1.0-1.8
DISTANCE_SCALE_M = 80.0                     # the LiDAR corridor length
HEADING_SCALE_RAD = math.pi / 2.0

# Appended, in this order, after rl_reward_design.STATE_SPEC. Changing the
# order or length requires retraining. All values are in [-1, 1].
STEERING_STATE_NAMES = (
    "passage_left_edge_norm",        # <= 0; 0 when there is no room on the left
    "passage_right_edge_norm",       # >= 0; 0 when there is no room on the right
    "pedestrian_lateral_norm",       # predicted at the ego's arrival (passage.predicted_lateral_at_arrival)
    "pedestrian_lateral_speed_norm",
    "distance_to_pedestrian_line_norm",  # negative once the ego has passed the line
    "ego_lateral_norm",
    "heading_error_norm",
    "previous_u",
)
STEERING_STATE_DIM = STATE_DIM + len(STEERING_STATE_NAMES)  # 6 + 8 = 14


def _unit(value: float, scale: float) -> float:
    return min(max(value / scale, -1.0), 1.0)


def build_steering_observation(
    base_state: Sequence[float],
    *,
    passage: Optional[Passage],
    pedestrian_lateral_m: float,
    pedestrian_lateral_speed_mps: float,
    distance_to_pedestrian_line_m: float,
    ego_lateral_m: float,
    heading_error_rad: float,
    previous_u: float,
) -> List[float]:
    """The 14-number observation: base_state (rl_reward_design.build_state_vector,
    6 values) followed by STEERING_STATE_NAMES. Lateral values are +right of
    the lane center, in the passage frame. An unknown or empty passage shows
    as 0 / 0: no room either side."""
    if len(base_state) != STATE_DIM:
        raise ValueError(f"base_state must have {STATE_DIM} values, got {len(base_state)}")
    left = right = 0.0
    if passage is not None and not passage.empty:
        left, right = min(passage.left_edge_m, 0.0), max(passage.right_edge_m, 0.0)
    return list(base_state) + [
        _unit(left, LATERAL_SCALE_M),
        _unit(right, LATERAL_SCALE_M),
        _unit(pedestrian_lateral_m, LATERAL_SCALE_M),
        _unit(pedestrian_lateral_speed_mps, PEDESTRIAN_SPEED_SCALE_MPS),
        _unit(distance_to_pedestrian_line_m, DISTANCE_SCALE_M),
        _unit(ego_lateral_m, LATERAL_SCALE_M),
        _unit(heading_error_rad, HEADING_SCALE_RAD),
        min(max(float(previous_u), -1.0), 1.0),
    ]


def clear_predicted_u(observation: Sequence[float]) -> float:
    """A hand-written aim rule for the scripted reference runs: aim just past
    where the pedestrian is predicted to be when the car arrives, on the
    right; stay in lane (u = 0) if they will not be in the car's way there,
    or if there is no room to their right. Reads only what the policy sees."""
    right_edge_m = observation[STATE_DIM + STEERING_STATE_NAMES.index("passage_right_edge_norm")] * LATERAL_SCALE_M
    predicted_m = observation[STATE_DIM + STEERING_STATE_NAMES.index("pedestrian_lateral_norm")] * LATERAL_SCALE_M
    reach_m = Obstacle(0.0).radius_m + EGO_HALF_WIDTH_M + DEFAULT_SIDE_MARGIN_M  # the passage's exclusion zone
    target_m = predicted_m + reach_m
    if abs(predicted_m) >= reach_m or right_edge_m <= 0.0 or target_m > right_edge_m:
        return 0.0
    return target_m / right_edge_m


PASS_BEHIND_SPEED_MPS = 0.3  # sideways speed below this counts as standing


def pass_behind_u(observation: Sequence[float]) -> float:
    """Week 7 scripted replanner (run per_tick, it re-decides as the
    pedestrian changes): pass BEHIND the pedestrian with right-only steering.
    In order: already right of where they will be -> u = 1 (never cut back
    across them); walking left -> u = 1 if they will reach the car's path
    (the far right opens up behind them), else hold the lane; walking right,
    toward the escape side -> u = 0 (the lane opens up behind them); standing
    -> swerve right only if they stand in the car's path. Reads only what the
    policy sees."""
    def value(name, scale):
        return observation[STATE_DIM + STEERING_STATE_NAMES.index(name)] * scale

    speed = value("pedestrian_lateral_speed_norm", PEDESTRIAN_SPEED_SCALE_MPS)
    predicted_m = value("pedestrian_lateral_norm", LATERAL_SCALE_M)
    ego_m = value("ego_lateral_norm", LATERAL_SCALE_M)
    reach_m = Obstacle(0.0).radius_m + EGO_HALF_WIDTH_M + DEFAULT_SIDE_MARGIN_M
    if ego_m > predicted_m:
        return 1.0
    if speed < -PASS_BEHIND_SPEED_MPS:
        return 1.0 if predicted_m < ego_m + reach_m else 0.0
    if speed > PASS_BEHIND_SPEED_MPS:
        return 0.0
    return 1.0 if abs(predicted_m) < reach_m else 0.0


# =============================================================================
# Reward
# =============================================================================

@dataclass(frozen=True)
class SteeringRewardConfig:
    """Every number that shapes the steering reward (recorded with a run so
    two policies can only be compared if they were trained on the same one)."""

    w_contact: float = 300.0
    # One-time penalty for touching the pedestrian (same size as last semester's).
    w_contact_per_mps: float = 15.0
    # Added per m/s of impact speed: a 20 m/s hit costs 600, a 5 m/s hit 375.
    w_off_road_per_s: float = 25.0
    # While any corner of the car is off the driving lanes. Over a whole 8 s
    # window this is 200, less than the lightest contact: leaving the road is
    # bad but better than hitting a person.
    w_clearance: float = 50.0
    # One-time bonus at the end of a contact-free window, scaled by
    # min clearance / enough_clearance_m and capped at 1. Only paid if the car
    # never left the road, so road-keeping outranks clearance.
    enough_clearance_m: float = 1.0
    # Quentin, 2026-10-12 (was 0.5 m): about the US 3-ft (0.91 m) minimum for
    # passing a cyclist; MASTER "Phase 2 design decisions" 9.
    w_jerk: float = W_JERK
    w_smooth_brake: float = W_SMOOTH_ACTION
    # The two comfort terms last semester's reward already had, same weights.
    w_lateral_accel: float = 0.1
    # Per m/s^2 of lateral acceleration per second, after onset: makes a
    # gentler swerve preferable when two swerves are equally safe.
    w_aim_change: float = 0.07
    # Per unit change of the applied u from one tick to the next, after onset.
    # One full re-aim costs 0.07; flipping every tick for a whole window ~28,
    # still below the clearance bonus. Without it per_tick learned to dither
    # (the car's dynamics hide the wheel buzz from the other comfort terms).
    window_s: float = 8.0
    # The scoring window (encounter_metrics); used only to check the ordering.
    pedestrian_contact_free: bool = True
    # 2026-10-23: a contact the PEDESTRIAN causes (car at rest, pedestrian moving: contact_is_pedestrians)
    # is neither penalized nor episode-ending, matching encounter protocol v4. Before, it cost -300
    # although the evaluation does not count it (Option B investigation).

    def __post_init__(self):
        if self.enough_clearance_m <= 0.0:
            raise ValueError("enough_clearance_m must be positive")
        if self.w_contact <= self.w_off_road_per_s * self.window_s + self.w_clearance:
            raise ValueError("the lightest contact must cost more than a whole window off the road "
                             "plus the lost clearance bonus")
        if self.w_aim_change * self.window_s / 0.02 >= self.w_clearance:  # 50 Hz ticks, full swing every tick
            raise ValueError("re-aiming every tick for a whole window must cost less than the clearance bonus")


def compute_steering_reward(
    *,
    contact_this_tick: bool,
    ego_speed_mps: float,
    off_road_this_tick: bool,
    window_ended: bool,            # True on the last tick of a contact-free window
    min_clearance_m: Optional[float],  # smallest footprint-to-pedestrian gap so far this episode
    went_off_road: bool,           # any off-road tick so far this episode (this tick included)
    jerk_mps3: float,
    brake_cmd_this_tick: float,
    brake_cmd_prev_tick: float,
    lateral_accel_mps2: float,
    hazard_active: bool,
    u_this_tick: float = 0.0,      # applied u (constant once locked, so lock_at_onset pays only its one choice)
    u_prev_tick: float = 0.0,
    fixed_dt: float = 0.02,
    config: SteeringRewardConfig = SteeringRewardConfig(),
) -> Tuple[float, dict]:
    """Per-tick reward for the steering policy: (total, breakdown).
    Positive = good. See the module docstring for the ordering."""
    r_contact = -(config.w_contact + config.w_contact_per_mps * max(0.0, ego_speed_mps)) if contact_this_tick else 0.0
    r_off_road = -config.w_off_road_per_s * fixed_dt if off_road_this_tick else 0.0
    r_clearance = 0.0
    if window_ended and not contact_this_tick and not went_off_road and min_clearance_m is not None:
        r_clearance = config.w_clearance * min(max(min_clearance_m, 0.0) / config.enough_clearance_m, 1.0)

    r_comfort = 0.0
    if hazard_active:  # before onset the cruise controller drives, not the policy
        r_comfort = -(config.w_jerk * abs(jerk_mps3)
                      + config.w_smooth_brake * abs(brake_cmd_this_tick - brake_cmd_prev_tick)
                      + config.w_lateral_accel * abs(lateral_accel_mps2)) * fixed_dt
        r_comfort -= config.w_aim_change * abs(u_this_tick - u_prev_tick)

    r_safety = r_contact + r_off_road + r_clearance
    return r_safety + r_comfort, {
        "r_total": r_safety + r_comfort, "r_safety": r_safety, "r_comfort": r_comfort,
        "r_contact": r_contact, "r_off_road": r_off_road, "r_clearance": r_clearance,
    }


def contact_is_pedestrians(*, ego_speed_mps: float, pedestrian_speed_mps: float,
                           stopped_mps: float = 0.5, moving_mps: float = 0.3) -> bool:
    """Per-tick stand-in for encounter protocol v4's contact_cause: the car is at
    rest (<= stopped_mps, as v4) and the pedestrian is moving. A slow creep into a
    standing pedestrian stays the car's contact.
    ponytail: v4 compares who closed the gap over 0.5 s; this uses current speeds
    only. Switch to the closing comparison if the two ever disagree on a run."""
    return ego_speed_mps <= stopped_mps and pedestrian_speed_mps > moving_mps


# =============================================================================
# Training helper
# =============================================================================

def policy_should_act(*, dist_travelled_m: float, encounter_distance_m: float, trigger_ttc_s: float,
                      speed_mps: float, lead_s: float = 1.0) -> bool:
    """False while the hazard is still more than lead_s seconds from starting.
    The steering environment ignores the policy before onset, so training
    fast-forwards those ticks instead of filling the replay buffer with
    decisions that change nothing. Mirrors the env's trigger rule:
    onset when dist_travelled >= encounter_distance - speed * trigger_ttc."""
    trigger_at_m = encounter_distance_m - speed_mps * trigger_ttc_s
    return dist_travelled_m >= trigger_at_m - speed_mps * lead_s
