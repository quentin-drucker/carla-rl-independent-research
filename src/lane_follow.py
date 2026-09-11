"""
lane_follow.py

Step 8 modularization:
- Extract lane_follow_step() out of the main test2 script.
- Keep behavior identical; only move code into a helper module.
"""

import carla 
# pyright: reportMissingImports=false
import math
from math_utils import clamp, wrap_to_pi, yaw_deg_to_rad, get_speed_mps
from lidar_utils import (
    lidar_min_distance_ahead,
    lidar_min_distance_in_lane_noodle,
    lidar_min_distance_along_route_noodle,
)


# -------------------------------------------------
# Helpers for lane-following control step (called from main loop)
# -------------------------------------------------
def _find_closest_route_index(route_points_world, loc):
    """
    Return the index of the route point closest to the current vehicle location.
    """
    if not route_points_world:
        return None

    best_i = None
    best_d2 = None

    for i, p in enumerate(route_points_world):
        dx = p.x - loc.x
        dy = p.y - loc.y
        dz = p.z - loc.z
        d2 = dx * dx + dy * dy + dz * dz

        if (best_d2 is None) or (d2 < best_d2):
            best_d2 = d2
            best_i = i

    return best_i


def _get_route_target_point(route_points_world, loc, lookahead_m):
    """
    Pick a target point lookahead_m ahead along the planned route polyline.
    """
    if not route_points_world:
        return None, None

    start_i = _find_closest_route_index(route_points_world, loc)
    if start_i is None:
        return None, None

    traveled = 0.0
    prev = route_points_world[start_i]

    for i in range(start_i + 1, len(route_points_world)):
        curr = route_points_world[i]
        seg = prev.distance(curr)

        if traveled + seg >= lookahead_m:
            return curr, start_i

        traveled += seg
        prev = curr

    return route_points_world[-1], start_i


# -------------------------------------------------
# lane-follow control step
# -------------------------------------------------

def lane_follow_step(world, vehicle, lookahead_m, steer_gain,
                     target_speed_mps, fixed_dt, speed_state,
                     lidar_frame, FIXED_DT,
                     # -------------------------------------------------
                     # LiDAR braking "knobs" (defined in main() and passed in)
                     # -------------------------------------------------
                     base_distance_m,      # meters (minimum clearance even at low speed)
                     headway_seconds,      # seconds of time-gap scaling
                     panic_distance_m=0.0, # meters: inside this, slam full brake immediately (0 disables)
                     ramp_up_per_s=0.5,    # how fast brake increases per second
                     ramp_down_per_s=3.0,  # how fast brake decreases per second
                     brake_profile="proportional_ramp",  # see ScenarioConfig.brake_profile for options
                     rl_brake_override=None,  # float [0,1] set by RL agent; bypasses profile logic when not None
                     ):
    """
    Lane-follow "brain" for one simulation step (meaning one tick).

    What it does each time it gets called:
    1) Figure out which lane waypoint (lane center) the car is currently on.
    2) Pick a target waypoint some distance AHEAD on that lane (param lookahead_m).
       - lookahead makes steering smoother: so vehicle isn't aiming "at its feet", but rather ahead.
    3) Compute the angle from the car -> target point (where we WANT to face from our current position to match lookahead waypoint).
    4) Compare this angle to the car's current yaw (where we ARE facing).
       The difference is a heading/steering error.
    5) Convert that error into a steering command and apply control (so the car steers toward the target point).

    Important note:
    - This is NOT using a camera. It's using CARLA's map waypoints (ground truth lane center).
    - We now use a target-speed controller instead of a fixed throttle, so "10 mph" is easy to set.
    """


    carla_map = world.get_map() # used below to query waypoints "under" car

    # current pose of vehicle (position + rotation)
    tf = vehicle.get_transform()
    loc = tf.location # 3D position of car
    yaw_rad = yaw_deg_to_rad(tf.rotation.yaw) # convert yaw to radians (of car's rotation)

    # waypoint "under" the car (lane centerline)
    wp = carla_map.get_waypoint(
        loc,
        project_to_road=True, # if the car is slightly off lane, CARLA snaps the query to nearest road lane.
        lane_type=carla.LaneType.Driving # only consider driving lanes (ignore shoulders, sidewalks, etc.)
    )

    # -------------------------------------------------
    # Route-consistent steering target
    # -------------------------------------------------
    # If main() provided a planned route polyline, use it for steering too.
    # This keeps:
    # - route debug line
    # - pedestrian encounter waypoint
    # - lane-follow target
    # all on the SAME path.
    route_points_world = speed_state.get("route_points_world", None)

    if route_points_world:
        target_loc, route_closest_i = _get_route_target_point(route_points_world, loc, lookahead_m)

        if target_loc is None:
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            print("lane_follow_step: no valid target point found on planned route!")
            return None
    else:
        # fallback: old local lane-follow behavior
        next_list = wp.next(lookahead_m)

        if not next_list:
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            print("lane_follow_step: no next waypoint found ahead!")
            return None

        target_wp = next_list[0]
        target_loc = target_wp.transform.location

    # -------------------------------------------------
    # METRIC: cross-track error (CTE)
    # -------------------------------------------------
    # CTE = distance from the car's current position to the lane center point (waypoint under the car).
    # This is a direct measure of how well we are staying centered in the lane.
    # Units: meters.
    cte_m = loc.distance(wp.transform.location)

    DEBUG_POINT_Z_OFFSET = 0.5
    # Pink dot = target waypoint the car is steering toward.
    # life_time slightly > FIXED_DT so it is refreshed each tick and disappears
    # immediately when the episode ends (no persistent trail).
    world.debug.draw_point(
        target_loc + carla.Location(z=DEBUG_POINT_Z_OFFSET),
        size=0.15,
        color=carla.Color(255,0,255), # pink
        life_time=FIXED_DT * 1.05
    )

    # Compute direction from car -> target point in the ground plane (ignore z).
    # dx, dy are "how far target is from me" in x and y: so like the difference between target and current location.
    dx = target_loc.x - loc.x
    dy = target_loc.y - loc.y

    # atan2(dy, dx) gives the angle (radians) pointing from car to the target in world coordinates.
    # This is the direction we WANT the car to face.
    target_angle = math.atan2(dy, dx)

    # Heading error = desired direction - current direction.
    # wrap_to_pi makes sure the error is the smallest-turn direction (no weird 358° turns and such).
    heading_error = wrap_to_pi(target_angle - yaw_rad)

    # Convert heading error into a steering command.
    # Bigger error => bigger steering magnitude.
    # steer_gain is just "how aggressive the steering corrections should be".
    steer_cmd = clamp(steer_gain * heading_error, -1.0, 1.0)  # CARLA steer must be within range: [-1, 1]
    # steer_cmd is [-1, +1], so map it to [-MAX_CONE_YAW_DEG, +MAX_CONE_YAW_DEG]
    cone_yaw_deg = steer_cmd * speed_state.get("max_cone_yaw_deg", 25.0) 

    
    # -------------------------------------------------
    # LiDAR hazard check (LANE-NOODLE corridor):
    # -------------------------------------------------
    # NOTE: This defines "ahead" relative to the lane centerline waypoints,
    # so curves don't accidentally include roadside objects inside a straight cone.

    # Keep these defined no matter which hazard method is active,
    # so telemetry/debug return at end never crashes.
    cone_half = speed_state.get("cone_half_angle_deg", 15.0)
    max_lat_m = speed_state.get("cone_max_lateral_m", None)

    NOODLE_ENABLE = speed_state.get("noodle_enable", True)
    NOODLE_STEP_M = speed_state.get("noodle_step_m", 1.25)
    NOODLE_HALF_WIDTH_M = speed_state.get(
        "noodle_half_width_m",
        speed_state.get("cone_max_lateral_m", 2.0)
    )
    NOODLE_MAX_DIST_M = speed_state.get("noodle_max_dist_m", 80.0)
    NOODLE_X_MIN_M = speed_state.get("noodle_x_min_m", 1.0)

    d_min_ahead = None
    noodle_points_world = None

    if NOODLE_ENABLE:
        lidar_actor = speed_state.get("lidar_actor", None)
        route_points_world = speed_state.get("route_points_world", None)

        if route_points_world:
            d_min_ahead, noodle_points_world = lidar_min_distance_along_route_noodle(
                lidar_actor,
                lidar_frame,
                route_points_world,
                loc,
                half_width_m=NOODLE_HALF_WIDTH_M,
                z_min=-1.0,
                z_max=2.5,
                max_dist_m=NOODLE_MAX_DIST_M,
                x_min_m=NOODLE_X_MIN_M
            )
        else:
            d_min_ahead, noodle_points_world = lidar_min_distance_in_lane_noodle(
                world,
                vehicle,
                lidar_actor,
                lidar_frame,
                lookahead_m=lookahead_m,
                step_m=NOODLE_STEP_M,
                half_width_m=NOODLE_HALF_WIDTH_M,
                z_min=-1.0,
                z_max=2.5,
                max_dist_m=NOODLE_MAX_DIST_M,
                x_min_m=NOODLE_X_MIN_M
            )

    # stash points for debug drawing in main()
    speed_state["noodle_points_world"] = noodle_points_world
    
    # -------------------------------------------------
    # SPEED + HAZARD CONTROL (MODE-BASED / HIERARCHICAL)
    # -------------------------------------------------
    # Goal:
    # - throttle and brake NEVER fight each other
    # - if hazard is active, throttle is fully suppressed
    # - if we stop due to hazard, we hold position without jitter
    # - once hazard clears, smoothly transition back to cruise
    #
    # NOTE: we keep your existing braking math (trigger distance, brake_target, ramp smoothing)
    # exactly the same, but we decide who "owns" the throttle/brake using a simple mode variable.

    # Always compute these so telemetry return stays consistent:
    speed_mps = get_speed_mps(vehicle)  #  measure current speed (m/s) so we can compare to target
    speed_error = target_speed_mps - speed_mps  #  positive = too slow, negative = too fast

    # -------------------------------------------------
    # binary LiDAR braking logic below:
    # -------------------------------------------------
    # NOTE: base_distance_m / headway_seconds are defined in main() and passed into this function.
    # trigger_distance = base_distance_m + speed_mps * headway_seconds
    trigger_distance_m = base_distance_m + speed_mps * headway_seconds
    # trigger_distance_m is the distance at which we want to start braking based on LiDAR hazard detection.
    hazard_active = (
        d_min_ahead is not None and
        d_min_ahead < trigger_distance_m
    )

    # -------------------------------------------------
    # Simple brake ramp (optional comfort layer)
    # -------------------------------------------------
    # Goal: reduce 'on/off' / '0/1' jitter, but still allow emergency full brake close-in.
    # NOTE: panic_distance_m is defined in main() and passed into this function.

    # -------------------------------------------------
    # Brake profile: compute brake_target from obstacle distance
    # -------------------------------------------------
    # All profiles share the same ramp-rate limiter below, so comparisons
    # are fair -- only the *target* differs, not the ramp speed.
    #
    # penetration = 0.0 at the trigger edge (just entered hazard zone)
    #             = 1.0 at panic_distance (worst-case, closest point)
    #
    # Profiles:
    #   proportional_ramp  -- brake_target = penetration        (linear, baseline)
    #   step_constant      -- brake_target = 1.0 immediately    (slam-brakes, max jerk)
    #   cautious_ramp      -- brake_target = min(penetration, 0.5) (half-force cap)
    #   exponential        -- brake_target = penetration^2      (gentle->urgent curve)

    brake_target = 0.0
    if hazard_active and d_min_ahead is not None:
        if panic_distance_m > 0.0 and d_min_ahead <= panic_distance_m:
            brake_target = 1.0
        else:
            denom = max(1e-3, (trigger_distance_m - max(0.0, panic_distance_m)))
            penetration = clamp((trigger_distance_m - d_min_ahead) / denom, 0.0, 1.0)

            if brake_profile == "step_constant":
                brake_target = 1.0
            elif brake_profile == "cautious_ramp":
                brake_target = clamp(penetration, 0.0, 0.5)
            elif brake_profile == "exponential":
                brake_target = penetration * penetration
            else:
                # "proportional_ramp" (default) and any unknown value
                brake_target = penetration

    # RL override: agent's chosen brake force replaces profile-computed target.
    # The ramp limiter still applies, so the agent doesn't need to learn smoothing.
    if rl_brake_override is not None:
        brake_target = float(max(0.0, min(1.0, rl_brake_override)))

    # Ramp brake_prev toward brake_target
    b_prev = speed_state.get("brake_prev", 0.0)

    if brake_target > b_prev:
        b_new = min(brake_target, b_prev + ramp_up_per_s * fixed_dt)
    else:
        b_new = max(brake_target, b_prev - ramp_down_per_s * fixed_dt)

    # NOTE: we don't commit brake_prev until we know which mode we're in (below).


    # -------------------------------------------------
    # Drive modes (simple state machine)
    # -------------------------------------------------
    # This prevents the PI speed controller from "fighting" hazard braking at low speeds.

    # --- small "near stop" threshold ---
    STOP_SPEED_EPS_MPS = 0.20   # ~0.45 mph (tune if needed)

    # --- hazard exit tuning (prevents mode flipping due to LiDAR noise) ---
    HAZARD_CLEAR_MARGIN_M = 1.0     # must be this much ABOVE trigger distance to consider "clear"
    HAZARD_CLEAR_TICKS_REQ = 10     # must be clear for this many ticks in a row (~0.2s at dt=0.02)

    # --- hold tuning ---
    HOLD_BRAKE = 0.25              # steady brake to prevent creeping at standstill (tune)
    USE_HANDBRAKE_IN_HOLD = True   # optional: makes the stop rock-solid
    
    # --- CRUISE smoothing knobs (prevents snap throttle<->brake transitions) ---
    OVERSPEED_COAST_START_MPS = 0.3*0.44704    # if we're >0.3mph too fast, stop throttling (coast)
    OVERSPEED_BRAKE_START_MPS = 0.8*0.44704   # if we're >0.8mph too fast, start gentle braking
    # was 2mph -- matched CRUISE_BRAKE_DISABLE_BAND exactly, so overspeed braking NEVER fired for small
    # overshoots. At 35mph+ the car had enough momentum that coasting alone couldn't damp oscillations.

    # --- Disable cruise braking near target (prevents mystery braking) ---
    CRUISE_BRAKE_DISABLE_BAND_MPS = 0.5*0.44704  # +/-~0.5mph band -- was 2mph which masked all overspeed braking

    # --- Slew limiters (smooth commands) ---
    THROTTLE_SLEW_PER_S = 1.5
    CRUISE_BRAKE_SLEW_PER_S = 2.5

    # --- PI integrator bleed when overspeeding (reduces ping-pong) ---
    # 0.95 was too aggressive (7.7% remaining after 1s → chronic undershoot).
    # 0.999 was too gentle (95% remaining → no damping of overshoot).
    # 0.98 per tick = ~36% remaining after 1s: drains integrator during overshoot without causing undershoot.
    ITERM_DECAY_WHEN_OVERSPEED = 0.98

    # Defaults (these will be applied this tick)
    throttle_cmd = 0.0
    brake_cmd = 0.0
    hand_brake_cmd = False

    # Mode memory (stored in speed_state so it persists tick-to-tick)
    prev_mode = speed_state.get("drive_mode", "CRUISE")
    mode = prev_mode  # CRUISE | HAZARD_BRAKE | STOP_HOLD | RECOVER

    # Decide if hazard is "really clear" (more conservative than hazard_active False).
    #
    # When in STOP_HOLD: require a POSITIVE confirmation that the obstacle is far
    # away. d_min_ahead=None means the LiDAR returned nothing — this happens when
    # a stopped pedestrian falls into a scan gap or their mesh settles at very close
    # range. Treating None as "clear" here caused the ego to resume and drive into
    # a stopped pedestrian. In STOP_HOLD, None = "unknown" not "safe".
    #
    # In all other modes: None still means nothing detected = clear (normal cruise).
    if prev_mode == "STOP_HOLD":
        hazard_clear = (
            d_min_ahead is not None and
            d_min_ahead > (trigger_distance_m + HAZARD_CLEAR_MARGIN_M)
        )
    else:
        hazard_clear = (
            (d_min_ahead is None) or
            (d_min_ahead > (trigger_distance_m + HAZARD_CLEAR_MARGIN_M))
        )

    # Count consecutive clear ticks (for stability)
    if hazard_clear:
        speed_state["hazard_clear_ticks"] = speed_state.get("hazard_clear_ticks", 0) + 1
    else:
        speed_state["hazard_clear_ticks"] = 0

    hazard_clear_stable = (speed_state["hazard_clear_ticks"] >= HAZARD_CLEAR_TICKS_REQ)

    # ----------------
    # MODE TRANSITIONS
    # ----------------
    if mode == "CRUISE":
        if hazard_active:
            mode = "HAZARD_BRAKE"

    elif mode == "HAZARD_BRAKE":
        if hazard_active and speed_mps <= STOP_SPEED_EPS_MPS:
            mode = "STOP_HOLD"
        elif hazard_clear_stable:
            mode = "RECOVER"

    elif mode == "STOP_HOLD":
        if hazard_clear_stable:
            mode = "RECOVER"

    elif mode == "RECOVER":
        if hazard_active:
            mode = "HAZARD_BRAKE"  # re-arm immediately if hazard returns while recovering
        elif speed_state.get("brake_prev", 0.0) <= 0.02:
            mode = "CRUISE"

    # Zero integrator on first tick of RECOVER so PI starts fresh when cruise resumes.
    if prev_mode != "RECOVER" and mode == "RECOVER":
        speed_state["i_term"] = 0.0

    speed_state["drive_mode"] = mode


    # ---------------------
    # ACTIONS (PER MODE)
    # ---------------------
    if mode == "CRUISE":
        # -------------------------------------------------
        # target-speed controller (smoothed + coast band + taper)
        # -------------------------------------------------
        KP = speed_state["kp"]
        KI = speed_state["ki"]

        # PI integration with clamped anti-windup.
        # Back-calculation anti-windup was removed: when KP alone saturates the output
        # (KP * large_error >> 1.0), back-calculation sets i_term deeply negative, destroying
        # accumulated error memory and causing slow/no convergence to target speed.
        # Simple clamp is the right tool: i_term is bounded so its max contribution is I_TERM_MAX * KI.
        # During the hard-acceleration phase KP dominates (full throttle); once within ~2mph of target
        # the integrator contribution takes over for steady-state tracking.
        speed_state["i_term"] += speed_error * fixed_dt
        I_TERM_MAX = 5.0   # KI * I_TERM_MAX = max I contribution to throttle (0.10 * 5.0 = 0.50)
        speed_state["i_term"] = clamp(speed_state["i_term"], -I_TERM_MAX, I_TERM_MAX)

        throttle_raw = clamp(KP * speed_error + KI * speed_state["i_term"], 0.0, 1.0)

        brake_raw = 0.0

        # -------------------------------------------------
        # Overspeed logic (coast band + brake band)
        # -------------------------------------------------
        # NOTE: approach taper was removed. It scaled down both P and I contributions near target,
        # creating a false equilibrium ~3mph below target speed. The i_term clamp above now limits
        # overshoot without sacrificing steady-state tracking.
        overspeed_mps = -speed_error  # positive if too fast

        if overspeed_mps > OVERSPEED_COAST_START_MPS:
            # bleed integrator so throttle doesn't snap back
            speed_state["i_term"] *= ITERM_DECAY_WHEN_OVERSPEED

            # cut throttle first
            throttle_raw = 0.0

            if overspeed_mps > OVERSPEED_BRAKE_START_MPS:
                brake_raw = clamp(
                    (overspeed_mps - OVERSPEED_BRAKE_START_MPS) * 0.35,
                    0.0,
                    1.0
                )
            else:
                brake_raw = 0.0  # coast only

        # -------------------------------------------------
        # Guard: never cruise-brake if we're close enough to target speed
        # -------------------------------------------------
        if abs(speed_error) <= CRUISE_BRAKE_DISABLE_BAND_MPS:
            brake_raw = 0.0

        # -------------------------------------------------
        # Slew-rate limiting (prevents snap transitions)
        # -------------------------------------------------

        # --- throttle smoothing ---
        t_prev = speed_state.get("throttle_prev", 0.0)
        max_t_step = THROTTLE_SLEW_PER_S * fixed_dt
        throttle_cmd = clamp(throttle_raw, t_prev - max_t_step, t_prev + max_t_step)
        throttle_cmd = clamp(throttle_cmd, 0.0, 1.0)

        # --- brake smoothing (cruise only) ---
        b_prev_cruise = speed_state.get("brake_prev_cruise", 0.0)

        if brake_raw <= 1e-6:
            # clear aggressively so no lingering brake near target
            brake_cmd = 0.0
            speed_state["brake_prev_cruise"] = 0.0
        else:
            max_b_step = CRUISE_BRAKE_SLEW_PER_S * fixed_dt
            brake_cmd = clamp(brake_raw, b_prev_cruise - max_b_step, b_prev_cruise + max_b_step)
            brake_cmd = clamp(brake_cmd, 0.0, 1.0)
            speed_state["brake_prev_cruise"] = brake_cmd

        # store previous throttle for next tick
        speed_state["throttle_prev"] = throttle_cmd

        # In cruise, clear the hazard brake ramp state
        speed_state["brake_prev"] = 0.0
        b_new = 0.0


    elif mode == "HAZARD_BRAKE":
        # IMPORTANT: stop PI from "charging up" while hazard braking
        speed_state["i_term"] *= 0.90

        # If hazard is active, throttle is fully suppressed (no fighting)
        throttle_cmd = 0.0

        # Apply ramped brake (your same ramp logic)
        speed_state["brake_prev"] = b_new
        brake_cmd = b_new
        
        # --- keep slew-limiter state sane across mode transitions ---
        speed_state["throttle_prev"] = throttle_cmd
        speed_state["brake_prev_cruise"] = 0.0


    elif mode == "STOP_HOLD":
        # When stopped due to hazard: do NOT let PI request throttle at all
        speed_state["i_term"] = 0.0

        throttle_cmd = 0.0

        # Hold steady brake to prevent creeping / micro-movement
        b_new = max(speed_state.get("brake_prev", 0.0), HOLD_BRAKE)
        speed_state["brake_prev"] = b_new
        brake_cmd = b_new

        # For telemetry, just say the ramp target is what we're holding
        brake_target = b_new

        if USE_HANDBRAKE_IN_HOLD:
            hand_brake_cmd = True
        
        # --- keep slew-limiter state sane across mode transitions ---
        speed_state["throttle_prev"] = throttle_cmd
        speed_state["brake_prev_cruise"] = 0.0


    elif mode == "RECOVER":
        # Smoothly ramp brake down to zero, THEN allow cruise to take over
        speed_state["i_term"] = 0.0  # hold at zero so PI is fresh when CRUISE resumes (no windup burst)

        throttle_cmd = 0.0

        # Ramp brake down toward 0 using your ramp_down_per_s
        b_prev = speed_state.get("brake_prev", 0.0)
        b_new = max(0.0, b_prev - ramp_down_per_s * fixed_dt)
        speed_state["brake_prev"] = b_new
        brake_cmd = b_new

        brake_target = 0.0

        # --- keep slew-limiter state sane across mode transitions ---
        speed_state["throttle_prev"] = throttle_cmd
        speed_state["brake_prev_cruise"] = 0.0


    # FINAL HARD GUARANTEE: if braking, no throttle (prevents controls fighting)
    if brake_cmd > 0.001:
        throttle_cmd = 0.0


    # -------------------------------------------------
    # Apply the control to the vehicle.
    # -------------------------------------------------
    control = carla.VehicleControl(
        throttle=throttle_cmd,  #  throttle comes from speed controller now
        steer=steer_cmd,
        brake=brake_cmd,  #  brake comes from speed controller when overspeeding
        hand_brake=hand_brake_cmd
    )
    vehicle.apply_control(control)

    # Return "telemetry" so main() can log metrics without recomputing everything.
    return {
        "cte_m": cte_m,                     # lane centering error (meters)
        "heading_error_rad": heading_error, # heading misalignment (radians)
        "steer_cmd": steer_cmd,             # applied steering command [-1,1]
        "speed_mps": speed_mps,             # measured speed (m/s)
        "speed_error_mps": speed_error,     # target - current (m/s)
        
        # LiDAR hazard info:
        "d_min_ahead_m": d_min_ahead,       # minimum LiDAR distance ahead (meters)
        "trigger_distance_m": trigger_distance_m,  # computed safety trigger distance for THIS tick (meters)
        "hazard_brake_cmd": 1.0 if hazard_active else 0.0, # 0 or 1 depending on whether hazard is active--no ramp for now.
        "brake_target": brake_target,       # what the ramp is trying to move toward [0..1]

        # Actual commands applied this tick (for terminal dashboard logging):
        "throttle_cmd": throttle_cmd,
        "brake_cmd": brake_cmd,
        "brake_prev_state": b_new,  # ramped brake state (this is what you were tuning)

        "cone_yaw_deg": cone_yaw_deg,       # current yaw/rotation of the hazard detection cone based on steering (steering-aware hazard detection)
        "cone_half_angle_deg": cone_half,   # current half-angle of the hazard detection cone (steering-aware hazard detection)

        # Debug / sanity: which mode "owned" throttle/brake this tick
        "drive_mode": mode,

        # Braking profile in use (for sweep CSV / comparison plots)
        "brake_profile": brake_profile,
    }