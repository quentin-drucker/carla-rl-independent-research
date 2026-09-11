"""
test---lane_follow_waypoints_Town03_Opt.py
Quentin

Goal:
- baseline lane following (no autopilot)
- use CARLA map waypoints (lane centerline) as "ground truth lane"
- hold forward motion w/ a simple throttle (fixed-ish)
- compute steering each tick so car stays on lane center

Why:
- this will serve to be my stable baseline before adding sensors + rare hazards (for my research direction)

Script overview/description:
It spawns a car in a chosen CARLA map, then every tick it picks a lane-center waypoint ahead 
of the car and steers toward it, while a simple PI speed controller holds a target mph-logging
 basic "how well did it drive?" stats.
"""

import carla
# pyright: reportMissingImports=false
import math
import time    #(not strictly needed since synchronous mode is on--but I am trying it anyway to try and better match simulation time to real time)

# ----------------------------
# small helper math functions
# ----------------------------

def clamp(value_to_clamp, lo, hi):
    # Keeps values inside a range -> Used for steering [-1,1], throttle [0, MAX], brake [0, MAX]
    # (keeps value_to_clamp in [lo, hi])
    return max(lo, min(hi, value_to_clamp))

def wrap_to_pi(angle_rad):
    """
    Angles wrap around in a circle, so values like +181° and -179° mean
    almost the same direction but look very different numerically.
    This function forces any angle into the range [-pi, +pi] ([-180°, +180°])
    so the steering error always represents the SMALLEST turn needed.
    This prevents the car from suddenly trying to steer the long way around.
    """
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad

def yaw_deg_to_rad(yaw_deg):
    # CARLA gives yaw in degrees, math wants radians
    return math.radians(yaw_deg)

def get_speed_mps(vehicle):
    """
    CARLA gives velocity as a 3D vector (v.x, v.y, v.z), not a single speed.
    Each component is how fast the car is moving in that direction (m/s).

    We square each component and then take the square root NOT to cancel it out,
    but to compute the LENGTH (magnitude) of the velocity vector.
    This is standard geometry: sqrt(x^2 + y^2 + z^2).

    The result is ONE scalar value:
    "how fast the car is moving overall, regardless of direction".
    """
    v = vehicle.get_velocity()
    return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

def mps_to_mph(speed_mps):
    # Convenience converter for printing human-friendly speed numbers
    # (from meters/second to miles/hour)
    return speed_mps / 0.44704

# ----------------------------
# lane-follow control step
# ----------------------------

def lane_follow_step(world, vehicle, lookahead_m, steer_gain, target_speed_mps, fixed_dt, speed_state):
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

    # ----------------------------
    # METRIC: cross-track error (CTE)
    # ----------------------------
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
        color=carla.Color(0,255,255),
        life_time=10.0
    )
    world.debug.draw_point(
        target_loc + carla.Location(z=DEBUG_POINT_Z_OFFSET),
        size=0.15,
        color=carla.Color(255,0,255),
        life_time=0.05
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

    # ----------------------------
    # target-speed controller
    # ----------------------------
    speed_mps = get_speed_mps(vehicle)  #  measure current speed (m/s) so we can compare to target
    speed_error = target_speed_mps - speed_mps  #  positive means "too slow", negative means "too fast"

    #  PI controller gains (tune these if needed)
    # KP: how strongly we react to the current speed error
    # KI: how strongly we react to accumulated error over time (helps on hills / drag)
    KP = speed_state["kp"]  #  read KP from state so you can tune in one place
    KI = speed_state["ki"]  #  read KI from state so you can tune in one place

    speed_state["i_term"] += speed_error * fixed_dt  #  integrate speed error over time (dt seconds per tick)

    throttle_cmd = KP * speed_error + KI * speed_state["i_term"]  #  convert error into a throttle "effort"
    throttle_cmd = clamp(throttle_cmd, 0.0, 1.0)  #  CARLA throttle must be within [0, 1]

    brake_cmd = 0.0  #  default brake off

    #  if we are going too fast beyond a small buffer, brake gently instead of negative throttle
    if speed_error < -0.5:  #  only brake if we're >0.5 m/s above target (~1.1 mph above target)
        brake_cmd = clamp((-speed_error) * 0.2, 0.0, 1.0)  #  gentle brake proportional to how much we're overspeeding
        throttle_cmd = 0.0  #  don't throttle and brake at the same time (keeps behavior simple)

    # Apply the control to the vehicle.
    control = carla.VehicleControl(
        throttle=throttle_cmd,  #  throttle comes from speed controller now
        steer=steer_cmd,
        brake=brake_cmd  #  brake comes from speed controller when overspeeding
    )
    vehicle.apply_control(control)

    # Return "telemetry" so main() can log metrics without recomputing everything.
    return {
        "cte_m": cte_m,                     # lane centering error (meters)
        "heading_error_rad": heading_error, # heading misalignment (radians)
        "steer_cmd": steer_cmd,             # applied steering command [-1,1]
        "speed_mps": speed_mps,             # measured speed (m/s)
        "speed_error_mps": speed_error      # target - current (m/s)
    }


# ----------------------------
# main
# ----------------------------
def main():
    # -------- config vars/"knobs" (change these) --------

    TARGET_MAP = "Town03_Opt"    # we force target town so runs are repeatable
    SPAWN_INDEX = 165              # (if spawn fails, pick another index) 
    # NOTE: 
    # - Town03_Opt, index 165 is a nice roundabout start.
    # - Town03_Opt, index 130 is also fun.
    # - Town04_Opt, index 158 or 118 are long, curved highway off-ramps.      

    SIM_SECONDS = 30            # how long to run (time in simulation seconds)
    FIXED_DT = 0.02              # e.g. 0.05 sec per tick = 20 Hz control loop
    next_time = time.perf_counter() # cosmetic change to make sim feel more like real-time pacing

    LOOKAHEAD_M = 6              # bigger = smoother (but cuts corners to get this), 
                                 # smaller = twitchy but tighter -> (I started with 12 but that felt too far from testing)
    STEER_GAIN = 1.5             # steering aggressiveness
    
    TARGET_MPH = 60                          #  desired speed in mph (easy to conceptualize) ---> ~35 mph is the fastest it can go without losing control with my current set up.
    TARGET_SPEED_MPS = TARGET_MPH * 0.44704  #  convert mph -> m/s (CARLA uses m/s)


    # -------- connect --------

    print("Connecting to CARLA...")
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    # IMPORTANT: explicitly load map (so I don't rely on whatever CARLA last opened)
    print("Loading map:", TARGET_MAP)
    world = client.load_world(TARGET_MAP)

    print("Connected to:", world.get_map().name)

    # -------- set synchronous stepping (repeatable) --------
    # synchronous mode = I control when sim advances (world.tick())
    # fixed_delta_seconds = every tick is exactly FIXED_DT seconds
    #
    # this makes control stable and also helps later for RL episodes
    settings = world.get_settings()
    # old_settings = settings # NOTE: not storing old settings object; we manually restore to async at the end.

    settings.synchronous_mode = True # again, this means I call world.tick() to advance sim manually, so this isn't a real-time run
    settings.fixed_delta_seconds = FIXED_DT
    world.apply_settings(settings)

    # -------- spawn ego vehicle --------

    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    print("Spawn points on this map:", len(spawn_points))
    if SPAWN_INDEX < 0 or SPAWN_INDEX >= len(spawn_points):
        raise ValueError(f"SPAWN_INDEX {SPAWN_INDEX} out of range (0..{len(spawn_points)-1})")

    spawn_tf = spawn_points[SPAWN_INDEX]

    bp_lib = world.get_blueprint_library()

    # pick a simple vehicle blueprint (model3 is stable)
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]

    vehicle = world.try_spawn_actor(vehicle_bp, spawn_tf)
    if vehicle is None:
        # spawn can fail if something is occupying that spawnpoint
        raise RuntimeError("Failed to spawn ego vehicle at this spawn point. Try a different SPAWN_INDEX.")

    print("Spawned ego vehicle id:", vehicle.id)

    # -------- spectator follow camera (to track car movement) --------
    spectator = world.get_spectator()

    def update_spectator_follow():
        # grab vehicle transform again each tick
        vtf = vehicle.get_transform()

        # offset behind and above car so I can watch lane-follow
        # negative x = behind the car (in vehicle frame)
        offset = carla.Location(x=-8, z=4.5) # x is forward, y is right, z is up in vehicle frame

        # transform(offset) converts local offset -> world position
        cam_loc = vtf.transform(offset)

        # copy vehicle yaw but tilt camera downward (negative pitch looks down)
        cam_rot = carla.Rotation(
            pitch=vtf.rotation.pitch - 15.0,
            yaw=vtf.rotation.yaw,
            roll=0.0
        )

        new_tf = carla.Transform(cam_loc, cam_rot)
        spectator.set_transform(new_tf) # update spectator position + rotation accordingly

    # -------- run control loop --------

    total_ticks = int(SIM_SECONDS / FIXED_DT)
    print(f"Running for {SIM_SECONDS} simulation seconds => {total_ticks} ticks @ dt={FIXED_DT}")

    speed_state = {
        "i_term": 0.0,   #  integral term storage for PI controller
        "kp": 0.5,       #  proportional gain (tune if needed)
        "ki": 0.05       #  integral gain (tune if needed)
    }

    # ----------------------------
    # Run statistics (metrics)
    # ----------------------------
    # We keep these as simple running totals so we can print an "average stats page" at the end.
    # The goal is not fancy logging yet—just solid, interpretable summary numbers.
    stats = {
        "ticks": 0,

        # 1) Cross-track error (lane centering)
        "cte_sum": 0.0,       # sum of CTE over all ticks (meters)
        "cte_max": 0.0,       # max CTE seen (meters)

        # 2) Heading error (car facing vs desired facing)
        "heading_abs_sum": 0.0,  # sum of |heading_error| (radians)
        "heading_abs_max": 0.0,  # max |heading_error| (radians)

        # 3) Distance traveled
        "distance_m": 0.0,     # accumulated traveled distance (meters)
        "prev_loc": None,      # previous tick location for distance integration

        # 4) Speed tracking error (target speed vs actual speed)
        "speed_err_abs_sum_mps": 0.0,  # sum of |speed_error| (m/s)

        # 5) Steering smoothness (steer "rate" proxy)
        "steer_prev": None,        # last tick's steer command
        "steer_rate_abs_sum": 0.0  # sum of |Δsteer|/dt (1/sec)
    }

    # Control loop:
    try:
        for t in range(total_ticks):
            # (Optional) print speed sometimes so to know how fast it's going (in mph now)
            if t % 40 == 0:
                speed_mps = get_speed_mps(vehicle)
                speed_mph = mps_to_mph(speed_mps)  #  convert m/s -> mph
                print(f"tick {t}/{total_ticks} | speed ~ {speed_mph:.2f} mph")  #  show mph

            # Step 1: perform one lane-follow control step
            telemetry = lane_follow_step( 
                world,
                vehicle,
                lookahead_m=LOOKAHEAD_M,
                steer_gain=STEER_GAIN,
                target_speed_mps=TARGET_SPEED_MPS,
                fixed_dt=FIXED_DT,
                speed_state=speed_state
            )

            # ----------------------------
            # Update run metrics (stats)
            # ----------------------------
            if telemetry is not None:
                stats["ticks"] += 1

                # 1) Cross-track error (meters)
                cte = telemetry["cte_m"]
                stats["cte_sum"] += cte
                stats["cte_max"] = max(stats["cte_max"], cte)

                # 2) Heading error (radians)
                heading_abs = abs(telemetry["heading_error_rad"])
                stats["heading_abs_sum"] += heading_abs
                stats["heading_abs_max"] = max(stats["heading_abs_max"], heading_abs)

                # 3) Distance traveled (meters)
                # We integrate distance by summing frame-to-frame position changes:
                # distance += ||pos_t - pos_{t-1}||
                loc_now = vehicle.get_transform().location
                if stats["prev_loc"] is not None:
                    stats["distance_m"] += loc_now.distance(stats["prev_loc"])
                stats["prev_loc"] = loc_now

                # 4) Speed tracking error (m/s) -> we will convert to mph in the final report
                stats["speed_err_abs_sum_mps"] += abs(telemetry["speed_error_mps"])

                # 5) Steering smoothness proxy: average steering rate
                # steer_rate ≈ |steer[t] - steer[t-1]| / dt
                steer = telemetry["steer_cmd"]
                if stats["steer_prev"] is not None:
                    stats["steer_rate_abs_sum"] += abs(steer - stats["steer_prev"]) / FIXED_DT
                stats["steer_prev"] = steer


            # Step 2: advance the sim exactly one step
            world.tick()

            # Step 3: move spectator camera to follow vehicle
            update_spectator_follow()   

            # 4) real-time pacing (this is the key)
            next_time += FIXED_DT # when the next tick SHOULD be (in real time)
            sleep_for = next_time - time.perf_counter()
            if sleep_for > 0: # if we're ahead of real-time; sleep to pace to real-time
                time.sleep(sleep_for)
            else:
                # we're running behind real-time; skip sleeping
                next_time = time.perf_counter()


    finally:
        # ----------------------------
        # End-of-run "stats page" (+ Cleanup after)
        # ----------------------------
        if stats["ticks"] > 0:
            mean_cte = stats["cte_sum"] / stats["ticks"]
            mean_heading_rad = stats["heading_abs_sum"] / stats["ticks"]

            mean_speed_err_mps = stats["speed_err_abs_sum_mps"] / stats["ticks"]

            # Convert some things to human-friendly units for the report
            mean_heading_deg = math.degrees(mean_heading_rad)
            max_heading_deg = math.degrees(stats["heading_abs_max"])

            mean_speed_err_mph = mps_to_mph(mean_speed_err_mps)

            # Steering rate: we only have (ticks-1) differences, but dividing by ticks is fine as an average proxy.
            # If you want exact, divide by max(1, ticks-1).
            denom = max(1, stats["ticks"] - 1)
            mean_steer_rate = stats["steer_rate_abs_sum"] / denom

            print("\n==================== RUN STATS (Summary) ====================")
            print(f"Map: {TARGET_MAP} | Spawn index: {SPAWN_INDEX}")
            print(f"Sim seconds: {SIM_SECONDS} | dt: {FIXED_DT:.3f} | ticks logged: {stats['ticks']}")
            print(f"Target speed: {TARGET_MPH:.1f} mph")
            print("-------------------------------------------------------------")

            # 1) Lane centering
            print("1) Cross-Track Error (CTE) — lane centering error")
            print("   Meaning: distance from car to lane center waypoint (meters). Smaller = stays centered.")
            print(f"   Mean CTE: {mean_cte:.3f} m | Max CTE: {stats['cte_max']:.3f} m")
            print("-------------------------------------------------------------")

            # 2) Heading alignment
            print("2) Heading Error — direction alignment error")
            print("   Meaning: |desired heading to lookahead point - car yaw|. Smaller = faces where it should.")
            print(f"   Mean heading error: {mean_heading_rad:.4f} rad ({mean_heading_deg:.2f} deg)")
            print(f"   Max  heading error: {stats['heading_abs_max']:.4f} rad ({max_heading_deg:.2f} deg)")
            print("-------------------------------------------------------------")

            # 3) Progress
            print("3) Distance Traveled — how far the car moved")
            print("   Meaning: total path length integrated from frame-to-frame position changes (meters).")
            print(f"   Distance traveled: {stats['distance_m']:.1f} m")
            print("-------------------------------------------------------------")

            # 4) Speed tracking
            print("4) Speed Error — speed hold quality")
            print("   Meaning: average |target speed - actual speed|. Smaller = better PI speed hold.")
            print(f"   Mean speed error: {mean_speed_err_mph:.2f} mph")
            print("-------------------------------------------------------------")

            # 5) Smoothness
            print("5) Steering Rate — twitchiness / smoothness proxy")
            print("   Meaning: average |Δsteer| / dt. Smaller = smoother steering changes over time.")
            print(f"   Mean steering rate: {mean_steer_rate:.3f} (steer units per second)")
            print("=============================================================\n")
        else:
            print("\n(No stats collected: telemetry was None every tick.)\n")
        
        
        # -------- cleanup (IMPORTANT) --------

        # destroy actors so CARLA doesn't accumulate junk between runs
        if vehicle is not None:
            vehicle.destroy()

        # restore world settings back to normal async mode
        # (CARLA gets weird if you leave sync on and then run other scripts)
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        print("Clean exit: destroyed ego + restored settings to default.")

if __name__ == "__main__":
    main()
