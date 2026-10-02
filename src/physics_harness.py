"""
physics_harness.py

Shared harness for the Week 3 CARLA vehicle physical-limits test suite
(steering lock, rollover, braking, throttle/brake response) -- see
Workstream 2 of plans/Week-3_2026-09-26_1052_physical-limits-trajectory-plan.md.

Design goals per the plan:
  - One shared harness/schema instead of four unrelated scripts.
  - Direct low-level VehicleControl so route-following, hazard logic, and
    SAC never confound a vehicle-model measurement.
  - Pure calculation/classification functions are kept separate from live
    CARLA calls so they can be offline-tested before any live run
    (tests/test_physics_harness.py exercises exactly these functions).
  - Evidence-status vocabulary throughout: "confirmed" / "not_observed_in_
    tested_range" / "inconclusive" / "not_measurable" -- never a bare
    True/False claim about a physical phenomenon CARLA may not expose.

CARLA does not model production ABS (recorded in MASTER_CARLA_RESEARCH_
SUMMARY.md and scenario_config.py already) -- this is a stated study
limitation, not something this module re-derives or works around.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import carla

from math_utils import get_speed_mps, mps_to_mph  # noqa: F401  (mps_to_mph: convenience re-export)
from trace_schema import TraceTick

MPH_TO_MPS = 0.44704

# CARLA 0.9.16's Python API does not expose a validated per-wheel slip-ratio
# or wheel-lock signal. If a future CARLA version adds one, wire it in here
# and update the manifest/trace docstrings -- do not infer lock/skid from
# speed or steering angle alone.
WHEEL_SLIP_SIGNAL_AVAILABLE = False

# Two independent live findings (test9 braking, 2026-09-26; test10 steering
# lock, 2026-09-26) found reproducible, physically implausible single-tick
# discontinuities in this CARLA vehicle model below roughly 5 m/s residual
# speed: test9 found a deceleration transient (~-27 m/s^2, ~9x a real
# vehicle's sustained braking capability) independent of commanded brake
# level; test10 found a velocity-DIRECTION discontinuity after a hard
# full-lock turn (speed 6.38 -> 2.02 m/s in one 0.02s tick, implying
# -218 m/s^2 -- over 20g, not physically possible) that corrupts any
# velocity-heading-based metric (body slip angle) computed from it. Treat
# this as a general "low-speed artifact zone" for this vehicle model and
# exclude it from any derivative/heading-based metric by default, rather
# than re-discovering it per test family.
#
# Third occurrence, and a resolved mystery (2026-09-27): the same test10
# matrix's "near_stop_onset" timing looked asymmetric between left/right
# turns at the same speed (e.g. 45mph_left_step stopped early, 45mph_right_
# step never did). detect_low_speed_snap_events(), run across all 12 traces,
# found this is fully explained by the same snap phenomenon occurring in 4
# of the 5 early-stop cases, at DIFFERENT residual speeds (4.56-7.82 m/s)
# and in BOTH directions (3 left, 1 right) -- ruling out a fixed speed
# threshold and a real left/right vehicle-dynamics bias. The two directions'
# trajectories are numerically near-identical mirror images right up to the
# tick before one of them snaps, consistent with a sharp, narrow-window tire/
# vehicle-model transition that a trajectory either crosses or narrowly
# avoids depending on its exact evolving state -- not a systematic asymmetry.
# The 5th early-stop case (15mph_left_step) showed NO snap event at all; it
# simply lacked enough speed to stay above the near-stop threshold for the
# full 5s window under ordinary cornering drag -- a second, unrelated and
# unremarkable mechanism, not a third variant of the snap.
LOW_SPEED_ARTIFACT_THRESHOLD_MPS = 5.0


# ----------------------------------------------------------------------
# Pure calculation helpers (no CARLA object access -- offline-testable)
# ----------------------------------------------------------------------

def compute_derived_kinematics(
    *,
    prev_speed_mps: Optional[float],
    curr_speed_mps: float,
    prev_yaw_deg: Optional[float],
    curr_yaw_deg: float,
    dt_s: float,
) -> Tuple[Optional[float], Optional[float]]:
    """Finite-difference longitudinal accel (m/s^2) and yaw rate (deg/s).

    Returns (None, None) if there is no previous sample (first tick of a
    run) or if dt_s is non-positive (would divide by zero / be meaningless).
    Yaw wrap-around (e.g. 179 -> -179 deg) is unwrapped to the shortest
    signed turn before differencing.
    """
    if prev_speed_mps is None or prev_yaw_deg is None or dt_s <= 0:
        return None, None

    accel_mps2 = (curr_speed_mps - prev_speed_mps) / dt_s

    yaw_delta_deg = curr_yaw_deg - prev_yaw_deg
    # Unwrap to [-180, 180] so a wrap-around doesn't look like a huge spike.
    while yaw_delta_deg > 180.0:
        yaw_delta_deg -= 360.0
    while yaw_delta_deg < -180.0:
        yaw_delta_deg += 360.0
    yaw_rate_dps = yaw_delta_deg / dt_s

    return accel_mps2, yaw_rate_dps


def compute_lateral_displacement_m(
    *,
    start_x_m: float,
    start_y_m: float,
    start_yaw_deg: float,
    curr_x_m: float,
    curr_y_m: float,
) -> float:
    """Signed perpendicular distance of (curr_x, curr_y) from the straight
    reference line through (start_x, start_y) at heading start_yaw_deg.

    Positive = to the right of the original heading (consistent with the
    route-right-positive convention already used in route_lateral_control.py
    for scenario runs). Used by the physical-limits tests, which have no
    planned route to measure against -- the initial heading line is the
    reference instead.
    """
    yaw_rad = math.radians(start_yaw_deg)
    forward_x, forward_y = math.cos(yaw_rad), math.sin(yaw_rad)
    # Right-hand perpendicular of (forward_x, forward_y) in a left-handed,
    # z-up world (CARLA convention) is (-forward_y, forward_x) rotated to
    # match route_lateral_control's existing "route-right" sign convention.
    right_x, right_y = -forward_y, forward_x

    dx = curr_x_m - start_x_m
    dy = curr_y_m - start_y_m
    return dx * right_x + dy * right_y


def compute_body_slip_angle_deg(
    *, vel_x_mps: float, vel_y_mps: float, yaw_deg: float,
    min_speed_mps: float = LOW_SPEED_ARTIFACT_THRESHOLD_MPS,
) -> Optional[float]:
    """Whole-body sideslip angle: the angle between the vehicle's velocity
    vector and its heading (yaw), in degrees, signed and wrapped to
    [-180, 180].

    This is a standard vehicle-dynamics quantity computable purely from
    position/velocity and yaw -- it does NOT require or imply any per-wheel
    slip/lock signal, and this module does not claim it as evidence of tire
    skid. It is exposed as the plan's "observable loss-of-control proxy":
    a large body slip angle means the car is travelling substantially
    sideways relative to where it is pointed, which is measurable and
    reportable without a skid/wheel-lock label.

    Returns None below min_speed_mps (default LOW_SPEED_ARTIFACT_THRESHOLD_MPS)
    -- a live steering-lock run (2026-09-26, test10) found a physically
    implausible single-tick velocity-direction discontinuity (speed
    6.38 -> 2.02 m/s in one 0.02s tick, implying over 20g of deceleration)
    in exactly this low-speed regime, which corrupted this angle to >150 deg.
    A bogus large angle from that artifact must not be reported as if it
    were a real measurement of vehicle motion.
    """
    speed_mps = math.sqrt(vel_x_mps * vel_x_mps + vel_y_mps * vel_y_mps)
    if speed_mps < min_speed_mps:
        return None
    velocity_heading_deg = math.degrees(math.atan2(vel_y_mps, vel_x_mps))
    slip_deg = velocity_heading_deg - yaw_deg
    while slip_deg > 180.0:
        slip_deg -= 360.0
    while slip_deg < -180.0:
        slip_deg += 360.0
    return slip_deg


def compute_turn_radius_m(
    *, speed_mps: float, yaw_rate_dps: Optional[float], min_yaw_rate_dps: float = 0.5
) -> Optional[float]:
    """Instantaneous turn radius from speed and yaw rate (radius = v / omega).

    Returns None when yaw_rate_dps is None or near zero (not meaningfully
    turning) -- a near-zero yaw rate would otherwise blow up to a
    meaningless enormous radius rather than reporting "not turning".
    """
    if yaw_rate_dps is None or abs(yaw_rate_dps) < min_yaw_rate_dps:
        return None
    yaw_rate_rad_s = math.radians(yaw_rate_dps)
    return speed_mps / abs(yaw_rate_rad_s)


def compute_lateral_accel_from_yaw_rate(
    *, speed_mps: float, yaw_rate_dps: Optional[float]
) -> Optional[float]:
    """Centripetal lateral acceleration approximation: a_lat = v * omega.

    This is the standard planar kinematic approximation (ignores body slip
    angle rate), adequate as a coarse proxy per the plan -- it is not a
    claim about actual tire lateral force.
    """
    if yaw_rate_dps is None:
        return None
    return speed_mps * math.radians(yaw_rate_dps)


def classify_stop_outcome(
    *, final_speed_mps: float, stopped_threshold_mps: float = 0.15
) -> str:
    """"stopped" / "not_stopped" -- a plain, predeclared threshold check.

    Deliberately does not attempt "smooth stop" vs "abrupt stop" labels
    here; that is a jerk-based judgment made by the caller from the full
    trace, not this single-value classifier.
    """
    return "stopped" if final_speed_mps < stopped_threshold_mps else "not_stopped"


@dataclass
class RolloverClassification:
    rollover_detected: bool
    evidence_status: str
    # "confirmed" | "not_observed_in_tested_range"
    max_abs_roll_deg: float
    threshold_deg: float


def classify_rollover(
    *, max_abs_roll_deg: float, threshold_deg: float = 60.0
) -> RolloverClassification:
    """Conservative rollover classification from a predeclared roll-angle
    threshold, not from visual appearance of the run.

    A run that never crosses the threshold is reported as
    "not_observed_in_tested_range" for the tested conditions, not as proof
    rollover is impossible under different conditions.
    """
    detected = max_abs_roll_deg >= threshold_deg
    return RolloverClassification(
        rollover_detected=detected,
        evidence_status="confirmed" if detected else "not_observed_in_tested_range",
        max_abs_roll_deg=max_abs_roll_deg,
        threshold_deg=threshold_deg,
    )


def detect_sustained_near_stop_onset_s(
    *, speed_time_pairs, threshold_mps: float = 0.5
) -> Optional[float]:
    """Returns the sim_time_s of the first tick after which speed stays
    below threshold_mps for the REST of the run, or None if the vehicle
    never sustainedly drops below it.

    Used to detect a maneuver that scrubbed off essentially all speed well
    before its nominal duration ended (e.g. a live steering-lock run,
    2026-09-26, found several full-lock cases came to a near-stop within
    ~1.2-1.5s of a 5s maneuver via cornering drag alone) -- a "steady
    state" window measured against the nominal duration would otherwise
    silently summarize a near-stationary vehicle as if it were still
    turning. Requires the LAST sample to also be below threshold (a
    transient dip that recovers does not count), avoiding a false trigger
    from a momentary noisy low reading.
    """
    pairs = list(speed_time_pairs)
    if not pairs or pairs[-1][0] >= threshold_mps:
        return None
    for i in range(len(pairs) - 1, -1, -1):
        speed, sim_time = pairs[i]
        if speed >= threshold_mps:
            return pairs[i + 1][1] if i + 1 < len(pairs) else None
    return pairs[0][1]  # every sample was already below threshold


def detect_low_speed_snap_events(
    accel_time_pairs, *, accel_threshold_mps2: float = 100.0
):
    """Flags ticks where |accel_mps2| exceeds accel_threshold_mps2 -- a
    single-tick deceleration/acceleration magnitude no real vehicle can
    produce (100 m/s^2 is roughly 10g; the default is set well above the
    milder ~27 m/s^2 low-speed braking transient already characterized by
    summarize_deceleration(), so this flags the more extreme, distinct
    "snap" phenomenon rather than double-counting that one).

    Built to programmatically confirm a pattern found by hand while
    investigating why test10's steering-lock matrix hit `near_stop_onset`
    asymmetrically between left/right at the same speed (2026-09-27): the
    two cases inspected directly (45mph_left_step, 45mph_right_ramp) each
    showed a SINGLE 0.02s tick with accel around -167 to -218 m/s^2 (>15g),
    at DIFFERENT residual speeds (6.38 m/s and 4.68 m/s) and in BOTH
    steering directions -- ruling out both "a fixed speed threshold" and "a
    left/right vehicle-dynamics asymmetry" as the explanation. Returns the
    list of (sim_time_s, accel_mps2) pairs for every tick that crosses the
    threshold, so a caller can confirm how many events occurred and when,
    rather than just a yes/no flag.

    accel_time_pairs: iterable of (accel_mps2, sim_time_s) tuples; accel_mps2
    may be None (first tick of a run) and is skipped.
    """
    events = []
    for accel_mps2, sim_time_s in accel_time_pairs:
        if accel_mps2 is not None and abs(accel_mps2) > accel_threshold_mps2:
            events.append((sim_time_s, accel_mps2))
    return events


@dataclass
class DecelerationSummary:
    normal_speed_peak_decel_mps2: Optional[float]
    normal_speed_mean_decel_mps2: Optional[float]
    low_speed_transient_peak_decel_mps2: Optional[float]
    threshold_mps: float


def summarize_deceleration(
    *, speed_accel_pairs, threshold_mps: float = 5.0
) -> DecelerationSummary:
    """Splits accel samples into a "normal-speed" regime (speed >=
    threshold_mps) and a "low-speed" regime (speed < threshold_mps) before
    computing peak/mean deceleration.

    Exists because a live braking test (2026-09-26, test9) found a sharp,
    reproducible deceleration transient below ~5 m/s that landed near the
    same ~-27 m/s^2 value across commanded brake levels 0.25-1.00, while the
    normal-speed peak scaled with brake level as expected (-6.1, -6.8,
    -22.3, -27.1 m/s^2 for 0.25/0.50/0.75/1.00). A single un-split peak/mean
    metric would silently mix the two regimes and hide that partial brake
    commands do not scale linearly all the way to a stop. This is reported
    as an observed CARLA vehicle-model characteristic, not a skid/wheel-lock
    claim -- no validated slip signal supports that label.

    speed_accel_pairs: iterable of (speed_mps, accel_mps2) tuples; accel_mps2
    may be None (e.g. the first tick of a run) and is skipped.
    """
    normal = [a for s, a in speed_accel_pairs if a is not None and s >= threshold_mps]
    low = [a for s, a in speed_accel_pairs if a is not None and s < threshold_mps]
    return DecelerationSummary(
        normal_speed_peak_decel_mps2=min(normal) if normal else None,
        normal_speed_mean_decel_mps2=(sum(normal) / len(normal)) if normal else None,
        low_speed_transient_peak_decel_mps2=min(low) if low else None,
        threshold_mps=threshold_mps,
    )


@dataclass
class AccelerationSummary:
    normal_speed_peak_accel_mps2: Optional[float]
    normal_speed_mean_accel_mps2: Optional[float]
    low_speed_transient_peak_accel_mps2: Optional[float]
    threshold_mps: float


def summarize_acceleration(
    *, speed_accel_pairs, threshold_mps: float = LOW_SPEED_ARTIFACT_THRESHOLD_MPS
) -> AccelerationSummary:
    """Throttle-side counterpart of summarize_deceleration(): splits accel
    samples into "normal-speed" (>= threshold_mps) and "low-speed"
    (< threshold_mps) regimes before computing peak/mean acceleration.
    Uses max() (most positive), not min(), since throttle response is
    positive acceleration -- otherwise identical rationale: a standing
    start begins inside the same low-speed artifact zone documented for
    braking (test9) and steering (test10), so any acceleration-from-rest
    measurement must not blindly average across it without checking
    whether a similar artifact appears here too.

    speed_accel_pairs: iterable of (speed_mps, accel_mps2) tuples; accel_mps2
    may be None (e.g. the first tick of a run) and is skipped.
    """
    normal = [a for s, a in speed_accel_pairs if a is not None and s >= threshold_mps]
    low = [a for s, a in speed_accel_pairs if a is not None and s < threshold_mps]
    return AccelerationSummary(
        normal_speed_peak_accel_mps2=max(normal) if normal else None,
        normal_speed_mean_accel_mps2=(sum(normal) / len(normal)) if normal else None,
        low_speed_transient_peak_accel_mps2=max(low) if low else None,
        threshold_mps=threshold_mps,
    )


def compute_rise_time_s(
    *, time_speed_value_pairs, target_fraction: float = 0.9
) -> Optional[float]:
    """First sim_time_s at which |value| reaches target_fraction of the
    final sample's |value|, or None if the final value is ~0 (nothing to
    rise to) or the input is empty.

    A coarse, descriptive "how fast does the response build up" metric per
    the plan's "time constants, saturation" requirement -- assumes a
    roughly monotonic approach to the final value; does not attempt to fit
    an actual first-order time constant.

    time_speed_value_pairs: iterable of (sim_time_s, value) tuples, in
    chronological order.
    """
    pairs = list(time_speed_value_pairs)
    if not pairs:
        return None
    final_value = pairs[-1][1]
    if abs(final_value) < 1e-9:
        return None
    target = abs(final_value) * target_fraction
    for sim_time_s, value in pairs:
        if abs(value) >= target:
            return sim_time_s
    return None


def summarize_stationary_hold(ticks, *, early_window_s=(1.0, 3.0), late_window_s=2.0) -> dict:
    """Descriptive motion summary for a vehicle that is SUPPOSED to stay
    still (Week 4 Chrono hold diagnostic). ticks: TraceTick rows of the
    hold window only, in order.

    Reports how far it moved (net and along its path), how much it turned,
    which way it moved relative to its own heading, and whether the motion
    is dying out (early-window vs. late-window mean speed) -- no pass/fail
    label, since "how much creep is acceptable" is a protocol decision.

    motion_direction_rel_heading_deg: angle of the net displacement vector
    relative to the starting heading, wrapped to [-180, 180]: ~0 = rolled
    forward, ~+/-180 = rolled backward, ~+/-90 = slid sideways (positive =
    toward the vehicle's right, matching compute_lateral_displacement_m).
    None if net displacement is under 1 cm.
    """
    rows = list(ticks)
    if not rows:
        return {}
    first, last = rows[0], rows[-1]
    t0 = first.sim_time_s

    path_m = 0.0
    turned_deg = 0.0
    for a, b in zip(rows, rows[1:]):
        path_m += math.hypot(b.pos_x_m - a.pos_x_m, b.pos_y_m - a.pos_y_m)
        d = b.yaw_deg - a.yaw_deg
        while d > 180.0:
            d -= 360.0
        while d < -180.0:
            d += 360.0
        turned_deg += d

    dx, dy = last.pos_x_m - first.pos_x_m, last.pos_y_m - first.pos_y_m
    net_m = math.hypot(dx, dy)
    direction = None
    if net_m >= 0.01:
        rel = math.degrees(math.atan2(dy, dx)) - first.yaw_deg
        while rel > 180.0:
            rel -= 360.0
        while rel < -180.0:
            rel += 360.0
        direction = rel

    def _mean_speed(lo, hi):
        vals = [r.speed_mps for r in rows if lo <= r.sim_time_s - t0 < hi]
        return sum(vals) / len(vals) if vals else None

    duration = last.sim_time_s - t0
    return {
        "duration_s": duration,
        "net_displacement_m": net_m,
        "path_length_m": path_m,
        "heading_change_deg": turned_deg,
        "motion_direction_rel_heading_deg": direction,
        "early_mean_speed_mps": _mean_speed(*early_window_s),
        "late_mean_speed_mps": _mean_speed(duration - late_window_s, duration + 1e-9),
        "final_speed_mps": last.speed_mps,
        "max_speed_mps": max(r.speed_mps for r in rows),
        "max_abs_pitch_change_deg": max(abs(r.pitch_deg - first.pitch_deg) for r in rows),
    }


def classify_upright_recovery(*, final_abs_roll_deg: float, upright_threshold_deg: float = 15.0) -> str:
    """"upright" / "not_upright" -- a plain, predeclared threshold check on
    the vehicle's roll angle after a maneuver has ended and it has settled.
    Deliberately coarse (matches the plan's "whether the vehicle recovers
    upright" requirement) -- does not attempt a "how close to tipping"
    gradation.
    """
    return "upright" if final_abs_roll_deg < upright_threshold_deg else "not_upright"


def classify_throttle_brake_symmetry(
    *, accel_response_mps2: float, decel_response_mps2: float, tolerance_ratio: float = 0.15
) -> str:
    """Compares magnitude of an acceleration response against a matched-
    magnitude deceleration response for the same commanded [0,1] value.

    Returns "symmetric" if the two magnitudes are within tolerance_ratio of
    the larger one, otherwise "asymmetric". Does not claim which direction
    is "better" -- only whether equal numeric commands produced comparable
    physical magnitudes, per the plan's RL-design framing.
    """
    a = abs(accel_response_mps2)
    d = abs(decel_response_mps2)
    if a == 0.0 and d == 0.0:
        return "inconclusive"
    larger = max(a, d)
    diff = abs(a - d)
    return "symmetric" if (diff / larger) <= tolerance_ratio else "asymmetric"


def get_wheel_steer_angle_deg(vehicle) -> Optional[float]:
    """Front-left wheel steer angle in degrees, or None if the API call
    fails. Wrapped in try/except because this is a live-actor call whose
    availability should never crash an otherwise-valid tick capture --
    callers must treat None as "not measurable this tick", not as zero.

    NOTE: FL alone is direction-biased under Ackermann steering geometry --
    see get_front_wheel_steer_angles_deg() for a direction-fair pair.
    """
    try:
        return vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)
    except Exception:
        return None


def get_front_wheel_steer_angles_deg(vehicle) -> Tuple[Optional[float], Optional[float]]:
    """(front_left, front_right) steer angles in degrees, each None if its
    API call fails independently.

    A live steering-lock run (2026-09-26, test10) found FL and FR reading
    very different magnitudes at full lock in opposite turn directions
    (e.g. ~46.7 deg one way, ~68.8 deg the other) despite every other
    measured quantity (yaw rate, turn radius, slip angle, speed loss) being
    symmetric between directions. This is Ackermann steering geometry, not
    an asymmetric vehicle response: the inner front wheel of a turn steers
    further than the outer one. A single "achieved wheel angle" summary
    should use max(|FL|, |FR|) (the inner wheel), not FL alone, to avoid
    manufacturing a false left/right asymmetry finding.
    """
    fl = get_wheel_steer_angle_deg(vehicle)
    try:
        fr = vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel)
    except Exception:
        fr = None
    return fl, fr


def inner_wheel_steer_angle_deg(
    *, front_left_deg: Optional[float], front_right_deg: Optional[float]
) -> Optional[float]:
    """Signed steer angle of whichever front wheel has the larger magnitude
    (the geometrically inner wheel of the current turn), or None if both
    inputs are None. See get_front_wheel_steer_angles_deg() for why FL alone
    is not a fair direction-independent summary.
    """
    candidates = [a for a in (front_left_deg, front_right_deg) if a is not None]
    if not candidates:
        return None
    return max(candidates, key=abs)


def apply_uniform_tire_friction(vehicle, friction: Optional[float]) -> None:
    """Applies one friction coefficient to all four wheels, or leaves
    CARLA's default physics untouched if friction is None. Mirrors
    brake_test.py's existing _apply_friction() so both the harness and the
    older diagnostic report the same physical configuration.
    """
    if friction is None:
        return
    phys = vehicle.get_physics_control()
    wheels = phys.wheels
    for w in wheels:
        w.tire_friction = friction
    phys.wheels = wheels
    vehicle.apply_physics_control(phys)


# ----------------------------------------------------------------------
# Live-CARLA helpers (thin -- not offline-testable, kept minimal)
# ----------------------------------------------------------------------

def read_physics_settings(world) -> dict:
    """Reads the world's current substepping configuration for manifest
    provenance. Returns raw settings values, no interpretation.
    """
    settings = world.get_settings()
    return {
        "fixed_delta_seconds": settings.fixed_delta_seconds,
        "substepping_enabled": settings.substepping,
        "max_substep_delta_time": settings.max_substep_delta_time,
        "max_substeps": settings.max_substeps,
    }


def capture_tick_from_actor(
    *,
    vehicle,
    tick_index: int,
    sim_time_s: float,
    requested_control,
    prev_speed_mps: Optional[float],
    prev_yaw_deg: Optional[float],
    dt_s: float,
    start_pose: Optional[Tuple[float, float, float]] = None,
    sim_frame: Optional[int] = None,
    wall_tick_s: Optional[float] = None,
    event_marker: Optional[str] = None,
) -> TraceTick:
    """Builds one TraceTick from a live CARLA vehicle actor.

    requested_control: the carla.VehicleControl this tick's caller applied
    (used for the requested_* fields; applied_* is read back separately via
    vehicle.get_control(), since CARLA can clamp/modify it internally).
    start_pose: (x, y, yaw_deg) of the first tick, used to compute
    lateral_displacement_m against the vehicle's own initial heading. None
    skips lateral-displacement computation (leaves it None on the tick).
    sim_frame / wall_tick_s / event_marker: optional caller-known values
    passed straight through to the TraceTick (None if not supplied).
    """
    transform = vehicle.get_transform()
    loc = transform.location
    rot = transform.rotation
    speed_mps = get_speed_mps(vehicle)
    velocity = vehicle.get_velocity()
    body_slip_angle_deg = compute_body_slip_angle_deg(
        vel_x_mps=velocity.x, vel_y_mps=velocity.y, yaw_deg=rot.yaw
    )

    accel_mps2, yaw_rate_dps = compute_derived_kinematics(
        prev_speed_mps=prev_speed_mps,
        curr_speed_mps=speed_mps,
        prev_yaw_deg=prev_yaw_deg,
        curr_yaw_deg=rot.yaw,
        dt_s=dt_s,
    )

    lateral_displacement_m = None
    if start_pose is not None:
        start_x, start_y, start_yaw = start_pose
        lateral_displacement_m = compute_lateral_displacement_m(
            start_x_m=start_x,
            start_y_m=start_y,
            start_yaw_deg=start_yaw,
            curr_x_m=loc.x,
            curr_y_m=loc.y,
        )

    applied = vehicle.get_control()
    fl_wheel_angle_deg, fr_wheel_angle_deg = get_front_wheel_steer_angles_deg(vehicle)
    api_accel = vehicle.get_acceleration()
    angular_vel = vehicle.get_angular_velocity()

    return TraceTick(
        tick_index=tick_index,
        sim_time_s=sim_time_s,
        pos_x_m=loc.x,
        pos_y_m=loc.y,
        pos_z_m=loc.z,
        yaw_deg=rot.yaw,
        pitch_deg=rot.pitch,
        roll_deg=rot.roll,
        speed_mps=speed_mps,
        vel_x_mps=velocity.x,
        vel_y_mps=velocity.y,
        accel_mps2=accel_mps2,
        yaw_rate_dps=yaw_rate_dps,
        body_slip_angle_deg=body_slip_angle_deg,
        requested_throttle=requested_control.throttle,
        requested_brake=requested_control.brake,
        requested_steer=requested_control.steer,
        applied_throttle=applied.throttle,
        applied_brake=applied.brake,
        applied_steer=applied.steer,
        lateral_displacement_m=lateral_displacement_m,
        front_wheel_steer_angle_deg=fl_wheel_angle_deg,
        front_right_wheel_steer_angle_deg=fr_wheel_angle_deg,
        event_marker=event_marker,
        vel_z_mps=velocity.z,
        api_accel_x_mps2=api_accel.x,
        api_accel_y_mps2=api_accel.y,
        api_accel_z_mps2=api_accel.z,
        angular_vel_x_dps=angular_vel.x,
        angular_vel_y_dps=angular_vel.y,
        angular_vel_z_dps=angular_vel.z,
        sim_frame=sim_frame,
        wall_tick_s=wall_tick_s,
    )


def accelerate_to_matched_entry_speed(
    *,
    world,
    vehicle,
    target_mps: float,
    fixed_dt: float,
    max_ticks: int = 2000,
    speed_fraction: float = 0.97,
) -> float:
    """Accelerates under full throttle until reaching speed_fraction of
    target_mps, then returns the ACTUAL speed achieved at that moment (not
    target_mps) -- per the plan's "use matched actual entry speeds rather
    than requested speeds." Callers should log the returned value as the
    test's true entry speed, not the requested target.
    """
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=0.0))
    for _ in range(max_ticks):
        world.tick()
        current = get_speed_mps(vehicle)
        if current >= target_mps * speed_fraction:
            return current
    return get_speed_mps(vehicle)


def compute_forward_velocity_components(*, yaw_deg: float, speed_mps: float) -> Tuple[float, float]:
    """World-frame (vx, vy) for a velocity of speed_mps pointed along yaw_deg.

    Pure trig, offline-testable -- factored out of set_instant_entry_velocity
    so the actual CARLA call (live-only) carries no untested math. Uses the
    same yaw convention as carla.Rotation.yaw (degrees, CARLA's left-handed
    world frame), matching compute_lateral_displacement_m elsewhere in this
    module.
    """
    yaw_rad = math.radians(yaw_deg)
    return speed_mps * math.cos(yaw_rad), speed_mps * math.sin(yaw_rad)


def set_instant_entry_velocity(
    *, world, vehicle, yaw_deg: float, target_mps: float, settle_ticks: int = 15
) -> float:
    """Sets the vehicle's velocity directly to target_mps along yaw_deg via
    carla.Actor.set_target_velocity(), then ticks settle_ticks times and
    returns the actual achieved speed.

    Use this INSTEAD OF accelerate_to_matched_entry_speed() when the test
    location does not have enough straight-line distance for a full-throttle
    ramp -- found live (2026-09-27) that accelerate_to_matched_entry_speed
    can cover 100+ meters reaching 90 mph, which silently carries a coasting
    maneuver's start point far past wherever a caller checked for open space
    around the nominal spawn. This settles to within ~1% of target_mps in
    about 3 ticks (0.06s at 50 Hz) with a smooth ramp, not a discontinuity --
    verified live, distinct from the physically-impossible single-tick drops
    already documented as LOW_SPEED_ARTIFACT_THRESHOLD_MPS. Not a substitute
    for accelerate_to_matched_entry_speed() where the throttle-achieved
    acceleration phase itself is part of what's being measured (e.g. test9,
    test12) -- only appropriate for coasting-only tests (steering lock,
    rollover) where the entry speed is a precondition, not the measurement.
    """
    vx, vy = compute_forward_velocity_components(yaw_deg=yaw_deg, speed_mps=target_mps)
    vehicle.set_target_velocity(carla.Vector3D(x=vx, y=vy, z=0.0))
    for _ in range(settle_ticks):
        world.tick()
    return get_speed_mps(vehicle)


def attach_collision_sensor(world, bp_lib, vehicle) -> Tuple[object, dict]:
    """Attaches a collision sensor to vehicle. Returns (sensor_actor,
    flag_dict); flag_dict["hit"] becomes True if any collision fires.
    Mirrors test3___ped_intrusion_scenario.py's `_attach_collision_sensor`
    so both use the same pattern -- kept here as a shared harness utility
    since the rollover substudy needs it and future substudies may too.
    """
    collision_bp = bp_lib.find("sensor.other.collision")
    sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
    flag = {"hit": False, "other_actor": None}

    def _on_collision(event):
        other = event.other_actor
        flag["hit"] = True
        flag["other_actor"] = other.type_id if other else "unknown"

    sensor.listen(_on_collision)
    return sensor, flag
