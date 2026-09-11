"""
test2___lane_follow.py

Step 8 modularization:
- Extract lane_follow_step() out of the main test2 script.
- Keep behavior identical; only move code into a helper module.
"""

import carla 
# pyright: reportMissingImports=false
import math
from test2___math_utils import clamp, wrap_to_pi, yaw_deg_to_rad, get_speed_mps
from test2___lidar_utils import lidar_min_distance_ahead


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
                     ramp_down_per_s=3.0   # how fast brake decreases per second
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

    # get waypoint ahead along this lane
    next_list = wp.next(lookahead_m)
    
    if not next_list:
        # end of road / weird edge case, just brake so it doesn't fly off
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
        # Print the error for debug:
        print("lane_follow_step: no next waypoint found ahead!")
        return None # no telemetry this tick

    # wp.next(distance) returns a list because intersections can branch (multiple "next" lanes).
    # So, for this baseline, just take the first candidate target @ index 0.
    # Later can choose more specifically based on route planning, etc.
    target_wp = next_list[0]
    target_loc = target_wp.transform.location

    # -------------------------------------------------
    # METRIC: cross-track error (CTE)
    # -------------------------------------------------
    # CTE = distance from the car's current position to the lane center point (waypoint under the car).
    # This is a direct measure of how well we are staying centered in the lane.
    # Units: meters.
    cte_m = loc.distance(wp.transform.location)

    DEBUG_POINT_Z_OFFSET = 0.5  # Cosmetic: to lift debug points above ground (+Z axis value) for better visibility
    # Debug visualization (optional but helpful):
    # - teal dots = waypoint under car
    # - pink dot = target waypoint we're aiming at
    world.debug.draw_point(
        wp.transform.location + carla.Location(z=DEBUG_POINT_Z_OFFSET),
        size=0.10,
        color=carla.Color(0,255,255), # teal
        life_time= 10.0
    )
    world.debug.draw_point(
        target_loc + carla.Location(z=DEBUG_POINT_Z_OFFSET),
        size=0.15,
        color=carla.Color(255,0,255), # pink
        life_time= FIXED_DT * 1.05
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
    # LiDAR hazard check (ahead only):
    # -------------------------------------------------
    cone_half = speed_state.get("cone_half_angle_deg", 15.0)

    # lane-width lateral cap (meters)
    max_lat_m = speed_state.get("cone_max_lateral_m", None)

    d_min_ahead = lidar_min_distance_ahead(
        lidar_frame,
        cone_half_angle_deg=cone_half,  # width of cone, "vision"
        cone_yaw_deg=cone_yaw_deg,      # rotate cone based on steering to look for hazards in the direction we're turning toward, not just straight ahead (steering-aware hazard detection)
        z_min=-1.0,             # ignore points that are too low (e.g. ground)
        # ^ (Don't go below ~-1.15m or the car will consider the ground as an obstacle and brake all the time?)
        z_max=2.5,              # ignore points that are too high (e.g. overpasses, signs)
        max_dist=80.0,          # how far ahead to look for hazards (tune based on speed and braking distance.
        x_min=1.0,               # ignore points that are too close (e.g. from roof or noise
        max_lateral_m=max_lat_m  # optional: ignore points that are too far left/right from the centerline (e.g. if we want to focus on points in our lane and ignore adjacent lanes, environmental distractions around the road)
    )
    
    
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

    brake_target = 0.0
    if hazard_active and d_min_ahead is not None: # this second Not None check here is redundant, but I added so Pylance doesn't complain.
        if panic_distance_m > 0.0 and d_min_ahead <= panic_distance_m:
            brake_target = 1.0
        else:
            # simple linear scaling: farther away => smaller brake
            # (not a fancy curve; just a straight line)
            # When d_min == trigger_distance => ~0 brake
            # When d_min == panic_distance  => ~1 brake
            denom = max(1e-3, (trigger_distance_m - max(0.0, panic_distance_m)))
            brake_target = (trigger_distance_m - d_min_ahead) / denom
            brake_target = clamp(brake_target, 0.0, 1.0)

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
    OVERSPEED_COAST_START_MPS = 0.20   # if we're >0.20 m/s too fast, stop throttling (coast)
    OVERSPEED_BRAKE_START_MPS = 0.80   # if we're >0.80 m/s too fast, start gentle braking
    APPROACH_FRAC = 0.10          # 0.10 = taper begins within last 10% of target speed
    APPROACH_MIN_SCALE = 0.25     # even near target, allow up to 25% throttle authority (prevents “stuck below target”)
    CRUISE_BRAKE_DISABLE_BAND_MPS = 0.44704  # within +/-0.44704 m/s (~1 mph) of target, never apply cruise brake
    THROTTLE_SLEW_PER_S = 1.5          # max throttle change per second (~0.03 per tick at dt=0.02)
    CRUISE_BRAKE_SLEW_PER_S = 2.5      # max cruise-brake change per second (~0.05 per tick at dt=0.02)
    ITERM_DECAY_WHEN_OVERSPEED = 0.95  # bleed PI integrator when overspeeding (reduces ping-pong)

    # Defaults (these will be applied this tick)
    throttle_cmd = 0.0
    brake_cmd = 0.0
    hand_brake_cmd = False

    # Mode memory (stored in speed_state so it persists tick-to-tick)
    mode = speed_state.get("drive_mode", "CRUISE")  # CRUISE | HAZARD_BRAKE | STOP_HOLD | RECOVER

    # Decide if hazard is "really clear" (more conservative than hazard_active False)
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
        # once brake ramp is basically at zero, go back to cruise
        if speed_state.get("brake_prev", 0.0) <= 0.02:
            mode = "CRUISE"

    speed_state["drive_mode"] = mode


    # ---------------------
    # ACTIONS (PER MODE)
    # ---------------------
    if mode == "CRUISE":
        # -------------------------------------------------
        # target-speed controller (smoothed + coast band)
        # -------------------------------------------------
        # PI controller gains (tune these if needed)
        # KP: how strongly we react to the current speed error
        # KI: how strongly we react to accumulated error over time (helps on hills / drag)
        KP = speed_state["kp"]  #  read KP from state so you can tune in one place
        KI = speed_state["ki"]  #  read KI from state so you can tune in one place

        # -------------------------------------------------
        # PI speed control (normal case)
        # -------------------------------------------------
        speed_state["i_term"] += speed_error * fixed_dt

        throttle_raw = KP * speed_error + KI * speed_state["i_term"]
        throttle_raw = clamp(throttle_raw, 0.0, 1.0)
        
        brake_raw = 0.0
        
        # -------------------------------------------------
        # Near-target throttle taper (prevents overshoot)
        # -------------------------------------------------
        # If we're close to target speed, gradually reduce throttle authority
        # so we "ease into" the setpoint instead of blasting through it.
        if target_speed_mps > 0.1:
            approach_band_mps = max(0.3, APPROACH_FRAC * target_speed_mps)  # band size (m/s)
            if speed_error > 0.0 and speed_error < approach_band_mps:
                # scale goes from 1.0 (far) down toward APPROACH_MIN_SCALE (near target)
                scale = speed_error / approach_band_mps  # 0..1
                scale = clamp(scale, APPROACH_MIN_SCALE, 1.0)
                throttle_raw *= scale

        # -------------------------------------------------
        # Overspeed logic (coast band + brake band)
        # -------------------------------------------------
        overspeed_mps = -speed_error  # positive if we're going too fast

        if overspeed_mps > OVERSPEED_COAST_START_MPS:

            # bleed integrator so throttle doesn't snap back
            speed_state["i_term"] *= ITERM_DECAY_WHEN_OVERSPEED

            # cut throttle first
            throttle_raw = 0.0

            if overspeed_mps > OVERSPEED_BRAKE_START_MPS:
                # gentle proportional brake beyond brake threshold
                brake_raw = clamp(
                    (overspeed_mps - OVERSPEED_BRAKE_START_MPS) * 0.35,
                    0.0,
                    1.0
                )
            else:
                # coast only
                brake_raw = 0.0

        # -------------------------------------------------
        # Guard: never cruise-brake if we're close enough to target speed
        # (prevents "mystery braking" while hovering near/under target)
        # -------------------------------------------------
        if abs(speed_error) <= CRUISE_BRAKE_DISABLE_BAND_MPS:
            brake_raw = 0.0

        # -------------------------------------------------
        # Slew-rate limiting (prevents snap transitions)
        # -------------------------------------------------

        # --- throttle smoothing ---
        t_prev = speed_state.get("throttle_prev", 0.0)
        max_t_step = THROTTLE_SLEW_PER_S * fixed_dt
        throttle_cmd = clamp(
            throttle_raw,
            t_prev - max_t_step,
            t_prev + max_t_step
        )
        throttle_cmd = clamp(throttle_cmd, 0.0, 1.0)

        # --- brake smoothing (cruise only) ---
        b_prev_cruise = speed_state.get("brake_prev_cruise", 0.0)

        if brake_raw <= 1e-6:
            # If we don't want cruise braking this tick, clear it aggressively
            # so it doesn't linger and cause "mystery braking" near target.
            brake_cmd = 0.0
            speed_state["brake_prev_cruise"] = 0.0
        else:
            max_b_step = CRUISE_BRAKE_SLEW_PER_S * fixed_dt
            brake_cmd = clamp(
                brake_raw,
                b_prev_cruise - max_b_step,
                b_prev_cruise + max_b_step
            )
            brake_cmd = clamp(brake_cmd, 0.0, 1.0)
            speed_state["brake_prev_cruise"] = brake_cmd

        # store previous values for next tick
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
        speed_state["i_term"] *= 0.80

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


    # FINAL HARD GUARANTEE: if braking, no throttle (prevents fighting)
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
    }