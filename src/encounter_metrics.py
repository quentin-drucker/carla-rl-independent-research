"""
encounter_metrics.py

Common, controller-independent evaluation protocol for one ego-pedestrian
encounter (steering-plus-braking baseline milestone, 2026-10-02).

Why this exists -- the protocol audit of test3.run_scenario() found that its
RunResult cannot fairly compare controllers that steer:
  - min_ped_distance_m is center-to-center and only counted while the
    pedestrian is AHEAD of the ego, so the closest point of a swerve (the
    pedestrian beside the car) is never measured;
  - pedestrian contact relies on CARLA's collision sensor (proven to miss
    pedestrian contact, test16) except in "near" crossings;
  - time_to_stop_s starts at each controller's OWN first hazard-brake tick,
    not at a shared event, so it is not comparable across controllers;
  - there is no explicit outcome reason and no fixed post-event horizon.

Every metric here is computed from the shared TraceTick format
(trace_schema.py, written by trajectory_recording.TrajectoryRecorder) over
ONE window that is identical for every controller:

    [onset_time_s, onset_time_s + protocol.horizon_s]

where onset_time_s is a scenario event that does not depend on the
controller (e.g. the scripted pedestrian trigger). Contact and clearance use
the ego's oriented rectangular footprint against a pedestrian circle
(pedestrian_contact.py), in every direction -- not only ahead.

Pure: no CARLA imports, offline-tested in tests/test_encounter_metrics.py.
Drivable-surface checks need the CARLA map, so the caller computes them per
tick and passes them in as `footprint_drivable` (None = not measured).
"""

import math
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence

from oriented_clearance import yaw_deg_to_forward_xy
from pedestrian_contact import (
    DEFAULT_PEDESTRIAN_RADIUS_M,
    EGO_HALF_LENGTH_M,
    compute_oriented_ego_pedestrian_clearance_m,
)
from physics_harness import LOW_SPEED_ARTIFACT_THRESHOLD_MPS
from route_recovery import PhysicalRouteReturnTracker

PROTOCOL_VERSION = 2
# v2 (2026-10-05): adds the steering-commitment metrics; v1 outcomes and
# metrics are unchanged.

OUTCOME_NO_ONSET = "no_onset"
OUTCOME_CONTACT = "contact"
OUTCOME_PASSED_CLEAR = "passed_clear"
OUTCOME_STOPPED_CLEAR = "stopped_clear"
OUTCOME_UNRESOLVED = "unresolved_at_horizon"
OUTCOME_INCOMPLETE = "incomplete_window"


@dataclass(frozen=True)
class EncounterProtocol:
    """Every threshold that affects a reported number, recorded with the
    result so two runs can only be compared if their protocols match."""

    horizon_s: float = 8.0
    # Post-onset window length; identical for every controller.
    stop_speed_mps: float = 0.3
    # "Stopped" threshold -- same value as RunResult/test3 for continuity.
    contact_tolerance_m: float = 0.0
    # Oriented footprint clearance at or below this counts as contact.
    pedestrian_radius_m: float = DEFAULT_PEDESTRIAN_RADIUS_M
    route_return_tolerance_m: float = 0.25
    route_return_ticks: int = 10
    # Same defaults as route_recovery.PhysicalRouteReturnTracker.
    normal_speed_threshold_mps: float = LOW_SPEED_ARTIFACT_THRESHOLD_MPS
    # Peak deceleration and jerk are reported only over ticks at or above
    # this speed: below it, CARLA's default physics produces a documented
    # ~-27 m/s^2 stopping snap (MASTER summary, Week 3 braking finding).
    commitment_deadband_m: float = 0.25
    # Requested lateral targets within this distance of the route center
    # count as "no side chosen" for the side-reversal count.
    offset_reversal_threshold_m: float = 0.05
    # The requested target must move back by at least this much to count as
    # a change of direction (hysteresis; filters tick-level jitter).
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self):
        if self.horizon_s <= 0.0:
            raise ValueError("horizon_s must be positive")
        if self.stop_speed_mps <= 0.0:
            raise ValueError("stop_speed_mps must be positive")


@dataclass
class EncounterMetrics:
    outcome: str
    # One of the OUTCOME_* constants. Priority: contact > passed_clear >
    # stopped_clear > unresolved_at_horizon. A run whose trace ends before
    # the horizon is "incomplete_window" unless contact already happened.
    safe_success: bool
    # No contact, no drivable-surface violation (when measured), window
    # complete, and the encounter resolved (passed clear or stopped clear).
    window_complete: bool
    onset_time_s: Optional[float]
    horizon_s: float

    speed_at_onset_mps: Optional[float] = None
    min_speed_mps: Optional[float] = None

    # Clearance: oriented ego footprint vs. pedestrian circle, any direction.
    min_clearance_m: Optional[float] = None
    min_clearance_time_s: Optional[float] = None  # seconds after onset
    speed_at_min_clearance_mps: Optional[float] = None
    contact_ticks: int = 0
    first_contact_time_s: Optional[float] = None  # seconds after onset
    contact_speed_mps: Optional[float] = None

    # Stopping, measured from the SHARED onset, not a controller's own event.
    stopped: bool = False
    time_to_stop_s: Optional[float] = None
    stopping_distance_m: Optional[float] = None

    # Passing: pedestrian fully behind the ego's rear bumper.
    passed: bool = False
    pass_time_s: Optional[float] = None
    speed_when_passing_mps: Optional[float] = None

    # Lateral motion, from the route-relative lateral offset.
    max_abs_route_lateral_m: Optional[float] = None
    final_route_lateral_m: Optional[float] = None
    departed_route: bool = False
    returned_to_route: bool = False

    # Drivable surface (None = not measured by the caller).
    drivable_violation_ticks: Optional[int] = None
    first_drivable_violation_time_s: Optional[float] = None

    # Dynamics (normal-speed ticks only for decel/jerk; see protocol).
    peak_decel_normal_speed_mps2: Optional[float] = None
    max_abs_jerk_normal_speed_mps3: Optional[float] = None
    max_abs_lateral_accel_mps2: Optional[float] = None

    # Steering commitment (advisor's reading (c): measure, don't constrain).
    # Counted on the REQUESTED lateral target after onset, not the steering
    # command, because any aim-point controller counter-steers normally.
    requested_side_reversals: Optional[int] = None
    # Switches of the chosen side (right <-> left), ignoring targets inside
    # the deadband. None = no requested target recorded.
    requested_offset_reversals: Optional[int] = None
    # Changes of direction of the requested target (outward <-> inward),
    # with hysteresis. A committed swerve-and-hold scores 0; a swerve then a
    # deliberate return to the route scores 1.

    protocol: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _pedestrian_longitudinal_m(tick) -> float:
    """Pedestrian position along the ego's heading, from the ego center
    (positive = ahead)."""
    fx, fy = yaw_deg_to_forward_xy(tick.yaw_deg)
    return (tick.pedestrian_x_m - tick.pos_x_m) * fx + (tick.pedestrian_y_m - tick.pos_y_m) * fy


def count_side_reversals(values: Sequence[float], deadband: float) -> int:
    """Number of right<->left switches, ignoring values within the deadband."""
    reversals, side = 0, 0
    for v in values:
        s = 1 if v > deadband else (-1 if v < -deadband else 0)
        if s and side and s != side:
            reversals += 1
        if s:
            side = s
    return reversals


def count_direction_reversals(values: Sequence[float], threshold: float) -> int:
    """Number of direction changes in a signal, with hysteresis: a reversal
    counts only once the signal has moved back by more than `threshold`
    from its latest extreme. The first move away from the start must also
    exceed `threshold` before any direction is established."""
    if not values:
        return 0
    reversals, direction, extreme = 0, 0, values[0]
    for v in values[1:]:
        if direction == 0:
            if v - extreme > threshold:
                direction, extreme = 1, v
            elif extreme - v > threshold:
                direction, extreme = -1, v
        elif direction == 1:
            if v > extreme:
                extreme = v
            elif extreme - v > threshold:
                reversals, direction, extreme = reversals + 1, -1, v
        else:
            if v < extreme:
                extreme = v
            elif v - extreme > threshold:
                reversals, direction, extreme = reversals + 1, 1, v
    return reversals


def compute_encounter_metrics(
    ticks: Sequence,
    *,
    onset_time_s: Optional[float],
    protocol: EncounterProtocol = EncounterProtocol(),
    footprint_drivable: Optional[Sequence[Optional[bool]]] = None,
) -> EncounterMetrics:
    """Score one encounter. ticks: TraceTick rows in time order (sim_time_s
    on the same clock as onset_time_s). footprint_drivable: optional per-tick
    flags, same length as ticks (True = whole ego footprint on a driving
    lane, False = some part off it, None = not measured that tick)."""
    rows = list(ticks)
    if footprint_drivable is not None and len(footprint_drivable) != len(rows):
        raise ValueError("footprint_drivable must be the same length as ticks")

    base = dict(onset_time_s=onset_time_s, horizon_s=protocol.horizon_s, protocol=asdict(protocol))
    if onset_time_s is None or not rows:
        return EncounterMetrics(outcome=OUTCOME_NO_ONSET, safe_success=False, window_complete=False, **base)

    end_time_s = onset_time_s + protocol.horizon_s
    eps = 1e-6
    window = [
        (i, r) for i, r in enumerate(rows) if onset_time_s - eps <= r.sim_time_s <= end_time_s + eps
    ]
    if not window:
        return EncounterMetrics(outcome=OUTCOME_NO_ONSET, safe_success=False, window_complete=False, **base)
    dt_guess = (rows[-1].sim_time_s - rows[0].sim_time_s) / max(1, len(rows) - 1)
    window_complete = rows[-1].sim_time_s >= end_time_s - 0.5 * dt_guess

    m = EncounterMetrics(outcome=OUTCOME_UNRESOLVED, safe_success=False, window_complete=window_complete, **base)
    first = window[0][1]
    m.speed_at_onset_mps = first.speed_mps
    m.min_speed_mps = min(r.speed_mps for _, r in window)

    tracker = PhysicalRouteReturnTracker(
        return_tolerance_m=protocol.route_return_tolerance_m,
        required_return_ticks=protocol.route_return_ticks,
    )
    behind_threshold_m = -(EGO_HALF_LENGTH_M + protocol.pedestrian_radius_m)
    path_m = 0.0
    prev = None
    lateral_values = []
    requested_values = []
    decels, jerks, lat_accels = [], [], []
    prev_accel_normal = None
    violation_ticks = 0 if footprint_drivable is not None else None

    for i, r in window:
        t_rel = r.sim_time_s - onset_time_s
        if prev is not None:
            path_m += math.hypot(r.pos_x_m - prev.pos_x_m, r.pos_y_m - prev.pos_y_m)
        prev = r

        if r.pedestrian_x_m is not None and r.pedestrian_y_m is not None:
            clearance = compute_oriented_ego_pedestrian_clearance_m(
                ego_x_m=r.pos_x_m, ego_y_m=r.pos_y_m, ego_yaw_deg=r.yaw_deg,
                pedestrian_x_m=r.pedestrian_x_m, pedestrian_y_m=r.pedestrian_y_m,
                pedestrian_radius_m=protocol.pedestrian_radius_m,
            )
            if m.min_clearance_m is None or clearance < m.min_clearance_m:
                m.min_clearance_m = clearance
                m.min_clearance_time_s = t_rel
                m.speed_at_min_clearance_mps = r.speed_mps
            if clearance <= protocol.contact_tolerance_m:
                m.contact_ticks += 1
                if m.first_contact_time_s is None:
                    m.first_contact_time_s = t_rel
                    m.contact_speed_mps = r.speed_mps
            if not m.passed and _pedestrian_longitudinal_m(r) < behind_threshold_m:
                m.passed = True
                m.pass_time_s = t_rel
                m.speed_when_passing_mps = r.speed_mps

        if not m.stopped and r.speed_mps < protocol.stop_speed_mps:
            m.stopped = True
            m.time_to_stop_s = t_rel
            m.stopping_distance_m = path_m

        if r.requested_lateral_offset_m is not None:
            requested_values.append(r.requested_lateral_offset_m)
        if r.route_lateral_m is not None:
            lateral_values.append(r.route_lateral_m)
            requested = r.requested_lateral_offset_m if r.requested_lateral_offset_m is not None else 0.0
            tracker.update(requested_offset_m=requested, actual_offset_m=r.route_lateral_m)

        if footprint_drivable is not None and footprint_drivable[i] is False:
            violation_ticks += 1
            if m.first_drivable_violation_time_s is None:
                m.first_drivable_violation_time_s = t_rel

        if r.yaw_rate_dps is not None:
            lat_accels.append(abs(r.speed_mps * math.radians(r.yaw_rate_dps)))
        if r.speed_mps >= protocol.normal_speed_threshold_mps and r.accel_mps2 is not None:
            decels.append(r.accel_mps2)
            if prev_accel_normal is not None:
                jerks.append(abs(r.accel_mps2 - prev_accel_normal[1]) / max(eps, r.sim_time_s - prev_accel_normal[0]))
            prev_accel_normal = (r.sim_time_s, r.accel_mps2)
        else:
            prev_accel_normal = None  # never difference across a low-speed gap

    if requested_values:
        m.requested_side_reversals = count_side_reversals(requested_values, protocol.commitment_deadband_m)
        m.requested_offset_reversals = count_direction_reversals(requested_values, protocol.offset_reversal_threshold_m)
    if lateral_values:
        m.max_abs_route_lateral_m = max(abs(v) for v in lateral_values)
        m.final_route_lateral_m = lateral_values[-1]
        m.departed_route = tracker.departed_route
        m.returned_to_route = tracker.physically_returned
    m.drivable_violation_ticks = violation_ticks
    if decels and min(decels) < 0.0:
        m.peak_decel_normal_speed_mps2 = min(decels)
    m.max_abs_jerk_normal_speed_mps3 = max(jerks) if jerks else None
    m.max_abs_lateral_accel_mps2 = max(lat_accels) if lat_accels else None

    if m.contact_ticks > 0:
        m.outcome = OUTCOME_CONTACT
    elif not window_complete:
        m.outcome = OUTCOME_INCOMPLETE
    elif m.passed:
        m.outcome = OUTCOME_PASSED_CLEAR
    elif m.stopped:
        m.outcome = OUTCOME_STOPPED_CLEAR
    else:
        m.outcome = OUTCOME_UNRESOLVED

    m.safe_success = (
        m.outcome in (OUTCOME_PASSED_CLEAR, OUTCOME_STOPPED_CLEAR)
        and not (violation_ticks or 0)
    )
    return m


def summarize_for_console(m: EncounterMetrics) -> List[str]:
    """Short human-readable lines (descriptive; the dict is the evidence)."""
    def f(v, spec=".2f"):
        return "n/a" if v is None else format(v, spec)

    return [
        f"outcome={m.outcome} safe_success={m.safe_success} window_complete={m.window_complete}",
        f"onset speed {f(m.speed_at_onset_mps)} m/s | min clearance {f(m.min_clearance_m)} m "
        f"at +{f(m.min_clearance_time_s)} s ({f(m.speed_at_min_clearance_mps)} m/s)",
        f"contact ticks {m.contact_ticks} (first +{f(m.first_contact_time_s)} s at {f(m.contact_speed_mps)} m/s)",
        f"stop: {m.stopped} in {f(m.time_to_stop_s)} s over {f(m.stopping_distance_m)} m | "
        f"passed: {m.passed} at +{f(m.pass_time_s)} s ({f(m.speed_when_passing_mps)} m/s)",
        f"lateral max {f(m.max_abs_route_lateral_m)} m final {f(m.final_route_lateral_m, '+.2f')} m "
        f"returned={m.returned_to_route} | drivable violations {m.drivable_violation_ticks}",
        f"peak decel (>= normal speed) {f(m.peak_decel_normal_speed_mps2)} m/s^2 | "
        f"max jerk {f(m.max_abs_jerk_normal_speed_mps3, '.1f')} m/s^3 | "
        f"max lateral accel {f(m.max_abs_lateral_accel_mps2)} m/s^2",
    ]
