"""
test2___braking_via_lidar.py
Quentin | Junior Spring Research, 2026.

Goal:
- baseline lane following (no autopilot)
- use CARLA lidar sensor as means to detect whether the car should put on the breaks 
  (if an obstacle is detected within a certain distance ahead, brake; otherwise, keep going)
- ensure comfort for the ego's passengers by maintaining a safe following distance and 
  avoiding sudden braking maneuvers.

Why:
- Gives me a sense of how CARLA's sensors may influence the actual control loop/dictate car's actions.

Script overview/description:
Uses the same framework that was made from the lane-following test1, but this time 
adding on elements to test lidar.
"""

import carla
# pyright: reportMissingImports=false
import math
import time    # (not strictly needed since synchronous mode is on--but I am trying it anyway to try and better match simulation time to real time)
import queue   # for lidar data handling

from test2___math_utils import clamp, wrap_to_pi, yaw_deg_to_rad, get_speed_mps, mps_to_mph
from test2___terminal_dashboard import print_live_status
from test2___lidar_utils import lidar_min_distance_ahead, smooth_brake_from_distance, draw_lidar_points, draw_hazard_cone, draw_hazard_capped_cone
from test2___spectator import SpectatorController
from test2___spawning import prepare_spawn_context, spawn_static_obstacle_by_index, spawn_ego_vehicle
from test2___lidar_sensor import attach_lidar_sensor
from test2___loop_utils import get_latest_lidar_frame
from test2___lane_follow import lane_follow_step
from test2___run_stats import init_run_stats, update_run_stats, print_run_summary
from test2___carla_session import connect_and_load_world, enable_sync_mode, restore_async_mode
from test2___telemetry_plotting import TelemetryBuffer, plot_telemetry

# ----------------------------------------------------------
# helper functions (geometry, control, lidar processing, etc.)
# ----------------------------------------------------------
# NOTE: basic math helpers were moved into test2___math_utils.py to keep this file more modular and clean.
# NOTE: ANSI + print_live_status moved into test2___terminal_dashboard.py
# NOTE: LiDAR helper functions moved into test2___lidar_utils.py
# NOTE: lane-following control step function moved into test2___lane_follow.py

def main():
    # =====================================================================
    # ========================  MAIN CONFIG KNOBS  ========================
    # =====================================================================
    # Adjust values here to change behavior without digging into logic.

    # ---------------------------------------------------------------------
    # WORLD / SCENARIO SETUP
    # ---------------------------------------------------------------------
    TARGET_MAP = "Town03_Opt"      # force target town so runs are repeatable
    SPAWN_INDEX = 164              # if spawn fails, pick another index
    # NOTE:
    # - Town03_Opt, index 130 is a nice starting point.
    # - Town03_Opt, index 165 is a nice roundabout start. OR 164 spawn and 215 obstacle location.
    # - Town04_Opt, index 158 or 118 are long, curved highway off-ramps.
    
    # - For pedestrian rare scenario I am trying "Town04_Opt" @ spawn index (start: 354/353, end: 348)/(start: 347, end a decent amount after 129) (these options go opposite ways down the road)
    # (its a nice 2-lane road thats straight (no turns) for a fair bit.)

    SPAWN_OBSTACLE_BY_INDEX = True
    OBSTACLE_SPAWN_INDEX = 215     # 164@17sim&~20mph secs & 215@27sim&~20mph secs for Town03_Opt
                                   # (adjust until it’s in ego's lane path)
    OBSTACLE_HOLD_BRAKE = True

    # ---------------------------------------------------------------------
    # TIMING / SIMULATION RATE
    # ---------------------------------------------------------------------
    SIM_SECONDS = 25               # total simulation duration (seconds)
    FIXED_DT = 0.02                # 0.02 sec per tick = 50 Hz control loop
    # NOTE: if changing FIXED_DT, also consider adjusting:
    #       - brake ramp tuning
    #       - LiDAR refresh assumptions
    #       - live dashboard pacing

    next_time = time.perf_counter()  # cosmetic: makes sim pacing feel real-time

    # ---------------------------------------------------------------------
    # TARGET CRUISE SPEED
    # ---------------------------------------------------------------------
    TARGET_MPH = 25                            # desired speed (easy to reason in mph)
    TARGET_SPEED_MPS = TARGET_MPH * 0.44704    # convert mph -> m/s (CARLA uses m/s)

    # ---------------------------------------------------------------------
    # LANE-FOLLOW STEERING
    # ---------------------------------------------------------------------
    LOOKAHEAD_M = 6               # larger = smoother but cuts corners more
                                  # smaller = tighter but twitchier as it reacts to more immediate points
    STEER_GAIN = 1.5              # steering aggressiveness multiplier

    # ---------------------------------------------------------------------
    # STEERING-AWARE HAZARD CONE
    # ---------------------------------------------------------------------
    CONE_HALF_ANGLE_DEG = 13.0    # width of cone "vision"
    MAX_CONE_YAW_DEG    = 45.0    # max rotation into turn direction
    CONE_DRAW_LENGTH_M  = 80.0    # how far to draw cone lines in-world
    # ^ match with lidar_min_distance_ahead(max_dist=...) in lane_follow.py
    #   so visualization matches detection distance.

    # --- "lane corridor" cap so far-away cone doesn't see environmental distractions that raise cause for braking (hazard) ---
    LANE_WIDTH_M = 3.3              # a lane is ~3.3m I think
    LANE_SIDE_MARGIN_M = 0.55       # small buffer so we don't miss near-lane hazards
    CONE_MAX_LATERAL_M = (LANE_WIDTH_M * 0.5) + LANE_SIDE_MARGIN_M

    # ---------------------------------------------------------------------
    # LiDAR BRAKING (DISTANCE + RAMP)
    # ---------------------------------------------------------------------
    BRAKE_BASE_DISTANCE_M = 8.0   # minimum clearance even at low speed
    BRAKE_HEADWAY_S       = 1.5   # clearance grows by speed_mps * headway
    PANIC_DISTANCE_M      = 6.0   # inside this -> full brake (0 disables)
    RAMP_UP_PER_S         = 2.0   # brake increase rate (per second)
    RAMP_DOWN_PER_S       = 3.0   # brake decrease rate (per second)

    # ---------------------------------------------------------------------
    # DRIVE MODES (ANTI-THROTTLE/BRAKE FIGHTING)
    # ---------------------------------------------------------------------
    STOP_SPEED_EPS_MPS = 0.20     # below this while hazard active -> STOP_HOLD
    HOLD_BRAKE         = 0.25     # steady brake used to prevent creeping
    USE_HANDBRAKE_IN_HOLD = True  # optional: makes stop rock-solid

    HAZARD_CLEAR_MARGIN_M  = 1.0  # must exceed trigger distance by this much
    HAZARD_CLEAR_TICKS_REQ = 10   # hazard must be clear this many ticks to recover
    
    # ---------------------------------------------------------------------
    # LIVE TERMINAL DASHBOARD
    # ---------------------------------------------------------------------
    LIVE_EVERY_S = 0.5  # how often to print status line (seconds)
                        # set to None to disable live updates
                        # adjust if FIXED_DT changes to keep pacing consistent
    LIVE_EVERY_TICKS = max(1, int(LIVE_EVERY_S / FIXED_DT))

    # -------------------------------------------------
    # connect
    # -------------------------------------------------
    client, world = connect_and_load_world(
        host="localhost",
        port=2000,
        timeout_s=10.0,
        target_map=TARGET_MAP
    )

    # -------------------------------------------------
    # Set synchronous stepping (repeatable) 
    # -------------------------------------------------
    # synchronous mode = I control when sim advances (world.tick())
    # fixed_delta_seconds = every tick is exactly FIXED_DT seconds
    #
    # this makes control stable and also helps later for RL episodes
    enable_sync_mode(world, fixed_dt=FIXED_DT)

    # Note: I am keeping these here as None for now so I can safely destroy them in the finally block at the end without worrying about whether they were created or not. This is a bit of a lazy approach but it keeps the cleanup code simple and robust.
    vehicle = None
    lidar = None
    obstacle = None

    # -------------------------------------------------
    # Spawn ego vehicle + optional obstacle 
    # NOTE: moved into test2_spawning.py helper
    # -------------------------------------------------
    spawn_points, spawn_tf, bp_lib = prepare_spawn_context(world, SPAWN_INDEX=SPAWN_INDEX)

    # for lidar brake testing--spawn static obstacle ahead
    obstacle = None
    if SPAWN_OBSTACLE_BY_INDEX:
        obstacle = spawn_static_obstacle_by_index(
            world,
            spawn_points=spawn_points,
            bp_lib=bp_lib,
            OBSTACLE_SPAWN_INDEX=OBSTACLE_SPAWN_INDEX,
            OBSTACLE_HOLD_BRAKE=OBSTACLE_HOLD_BRAKE
        )

    vehicle = spawn_ego_vehicle(world, bp_lib=bp_lib, spawn_tf=spawn_tf)

    # -------------------------------------------------
    # Attach and configure LiDAR sensor 
    # NOTE: moved into test2___lidar_sensor.py helper
    # -------------------------------------------------
    lidar, lidar_queue = attach_lidar_sensor(world, bp_lib, vehicle)

    # NOTE: Spectator camera modes function moved to test2___spectator.py
    # MODES: Follow (default) / Freecam ('f' key toggle) / Glide ('g' key toggle)
    spectator = world.get_spectator()
    spec_controller = SpectatorController(world, vehicle)

    # -------------------------------------------------
    # Run control loop
    # -------------------------------------------------

    total_ticks = int(SIM_SECONDS / FIXED_DT)
    print(f"Running for {SIM_SECONDS} simulation seconds => {total_ticks} ticks @ dt={FIXED_DT}")

    speed_state = {
        "i_term": 0.0,      #  integral term storage for PI controller
        "kp": 0.5,          #  proportional gain (tune if needed)
        "ki": 0.05,         #  integral gain (tune if needed)
        "brake_prev": 0.0,  #  for smooth braking ramp from LiDAR hazard detection
        
        # --- Cone params stored here so lane_follow_step can access them (for steering-aware hazard detection) ---
        "max_cone_yaw_deg": MAX_CONE_YAW_DEG,       # how far the hazard detection cone can look into the turn (steering-aware hazard detection)
        "cone_half_angle_deg": CONE_HALF_ANGLE_DEG, # width of the hazard detection cone (steering-aware hazard detection)
        "cone_draw_length_m": CONE_DRAW_LENGTH_M,   # how far to draw the hazard detection cone in-world (steering-aware hazard detection)

        # --- Drive mode state (to prevent throttle/brake fighting near stop) ---
        "drive_mode": "CRUISE",      # CRUISE | HAZARD_BRAKE | STOP_HOLD | RECOVER
        "hazard_clear_ticks": 0,     # counts consecutive ticks hazard looks clear

        # --- lateral cap (meters) for LiDAR hazard filter (lane corridor) ---
        "cone_max_lateral_m": CONE_MAX_LATERAL_M,

        "throttle_prev": 0.0,   # these two were added to help with smoother control transitions when braking (see lane follow script's cruise mode)
        "brake_prev_cruise": 0.0,
    }

    # -------------------------------------------------
    # Run statistics (metrics)
    # -------------------------------------------------
    # We keep these as simple running totals so we can print an "average stats page" at the end.
    # The goal is not fancy logging yet—just solid, interpretable summary numbers.
    stats = init_run_stats() # NOTE: dict moved to test2___run_stats.py helper

    # The following is for recording telemetry each tick and plotting at the end; NOTE: class moved to test2___telemetry_plotting.py helper
    telemetry_buffer = TelemetryBuffer() 

    # -------------------------------------------------
    # Control loop:
    # -------------------------------------------------
    try:
        last_lidar_frame = None
        last_lidar_tick = -1
        for t in range(total_ticks):

            # Step 0: advance simulation ONE tick (this drives physics + sensors)
            world.tick()

            # Step 1: grab newest lidar frame; if none arrived this tick, reuse last (NOTE: see test2___loop_utils.py for details on this function)
            lidar_frame, last_lidar_frame, last_lidar_tick = get_latest_lidar_frame(
                lidar_queue,
                last_lidar_frame=last_lidar_frame,
                last_lidar_tick=last_lidar_tick,
                t=t
            )

            # Step 2: perform one lane-follow + speed + hazard control step (using freshest LiDAR)
            telemetry = lane_follow_step(
                world,
                vehicle,
                lookahead_m=LOOKAHEAD_M,
                steer_gain=STEER_GAIN,
                target_speed_mps=TARGET_SPEED_MPS,
                fixed_dt=FIXED_DT,
                speed_state=speed_state,
                lidar_frame=lidar_frame,
                FIXED_DT=FIXED_DT, # for timing drawing points to match the control loop's pacing
            
                # --- LiDAR braking knobs (defined in main) ---
                base_distance_m=BRAKE_BASE_DISTANCE_M,
                headway_seconds=BRAKE_HEADWAY_S,
                panic_distance_m=PANIC_DISTANCE_M,
                ramp_up_per_s=RAMP_UP_PER_S,
                ramp_down_per_s=RAMP_DOWN_PER_S,
            )
            # Record telemetry for plotting later (NOTE: see test2___telemetry_plotting.py for details on this function)
            # (Step 2.5): record telemetry EVERY tick (for plotting later)
            telemetry_buffer.append(tick=t, fixed_dt=FIXED_DT, telemetry=telemetry)


            # Step 3: LiDAR visualization in-world (no disk saving)
            draw_lidar_points(
                world,
                lidar_frame,
                lidar,
                sample_step=60,
                life_time=0.1
            )
            # Step 3.5: draw the steering-aware cone (uses the SAME yaw used by the filter)
            if telemetry is not None:
                max_lat_m = speed_state.get("cone_max_lateral_m", None)

                if max_lat_m is None:
                    draw_hazard_cone(
                        world,
                        lidar,
                        cone_yaw_deg=telemetry.get("cone_yaw_deg", 0.0),
                        cone_half_angle_deg=telemetry.get("cone_half_angle_deg", 15.0),
                        length_m=speed_state.get("cone_draw_length_m", 30.0),
                        life_time= FIXED_DT * 1.05
                    )
                else:
                    draw_hazard_capped_cone(
                        world,
                        lidar,
                        cone_yaw_deg=telemetry.get("cone_yaw_deg", 0.0),
                        cone_half_angle_deg=telemetry.get("cone_half_angle_deg", 15.0),
                        max_lateral_m=max_lat_m,
                        length_m=speed_state.get("cone_draw_length_m", 30.0),
                        life_time= FIXED_DT * 1.05
                    )

            # Step 4: live terminal dashboard (color-coded, updates in-place)
            # RECOMMENT LATER:
            # if t % LIVE_EVERY_TICKS == 0:
            #     print_live_status(
            #         t=t,
            #         total_ticks=total_ticks,
            #         telemetry=telemetry,
            #         target_mph=TARGET_MPH
            #     )
            # (toggle via commenting) alert hazard logging:
            # > if LiDAR detects something close ahead, print a warning with the distance and brake command.
            # 
            # if telemetry is not None:
            #     haz = telemetry.get("hazard_brake_cmd", 0.0)
            #     brk = telemetry.get("brake_cmd", 0.0)
            #     dmin = telemetry.get("d_min_ahead_m", None)
            #     bp = telemetry.get("brake_prev_state", 0.0)
            
            #     # Only alert if something real is happening
            #     if haz > 0.05 or brk > 0.05:
            #         d_str = "None" if dmin is None else f"{dmin:.1f}m"
            #         print(f"\n⚠ Hazard ⚠ distance={d_str} hazard brake cmd={haz:.2f} actual brake cmd={brk:.2f} prev brake cmd={bp:.2f}") 
            #         #^^ (d = distance, h = hazard brake cmd, b = actual brake cmd applied)

            # Step 5: update run metrics (stats)
            update_run_stats(stats, telemetry=telemetry, vehicle=vehicle, FIXED_DT=FIXED_DT)

            # Step 6: move spectator camera to follow vehicle
            spec_controller.tick(FIXED_DT)

            # Step 7: real-time pacing (unchanged)
            next_time += FIXED_DT
            sleep_for = next_time - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_time = time.perf_counter()


    finally:
        spec_controller.close()
        print()  # ends the live status line cleanly; so the final stats will print nicely on its own.
        # -------------------------------------------------
        # End-of-run "stats page"
        # -------------------------------------------------
        print_run_summary(
            stats,
            TARGET_MAP=TARGET_MAP,
            SPAWN_INDEX=SPAWN_INDEX,
            SIM_SECONDS=SIM_SECONDS,
            FIXED_DT=FIXED_DT,
            TARGET_MPH=TARGET_MPH
        )
        
        # -------------------------------------------------
        # Cleanup/Destroying at end of run (IMPORTANT)
        # -------------------------------------------------

        # destroy actors so CARLA doesn't accumulate junk between runs
        if vehicle is not None:     # destroy ego vehicle
            vehicle.destroy()
        if lidar is not None:       # destroy lidar sensor
            lidar.destroy()
        if obstacle is not None:    # destroy obstacle spawned for lidar brake testing
            obstacle.destroy()

        # restore world settings back to normal async mode
        # (CARLA gets weird if you leave sync on and then run other scripts)
        restore_async_mode(world)

        print("Clean exit: destroyed ego + restored settings to default.")

        # -------------------------------------------------
        # Plot telemetry after run (close plot windows to exit)
        # -------------------------------------------------
        plot_telemetry(
            telemetry_buffer,
            target_mph=TARGET_MPH,

            # -------------------------------------------------
            # FIGURE 1 (if enabled): Speed Tracking
            # - speed (mph): actual measured vehicle speed
            # - target (mph): desired cruise speed
            # Useful for tuning PI gains (kp, ki) and seeing overshoot/oscillation.
            # -------------------------------------------------
            show_speed=True,

            # -------------------------------------------------
            # FIGURE 2 (if enabled): LiDAR + Braking (Tuning View)
            #
            # Top subplot:
            # - d_min_ahead (m): closest LiDAR obstacle in forward cone
            # - trigger_distance (m): safety threshold for braking
            #   (base_distance + speed * headway)
            # - shaded regions: hazard_active (LiDAR says "too close")
            #
            # Bottom subplot:
            # - brake_target: brake value computed from distance (raw goal)
            # - brake_prev_state: smoothed/ramped brake state
            # - brake_cmd: actual brake applied to vehicle
            #
            # This is your primary tuning figure for safety + comfort.
            # -------------------------------------------------
            show_braking_view=True,

            # -------------------------------------------------
            # FIGURE 3 (if enabled): Controls Overview
            # - steer_cmd: steering input [-1, 1]
            # - throttle_cmd: throttle [0, 1]
            # - brake_cmd: brake [0, 1]
            # Useful for checking throttle/brake conflicts or steering instability.
            # -------------------------------------------------
            show_controls=True,

            # -------------------------------------------------
            # FIGURE 4 (if enabled): Lane Metrics
            # - cte_m: cross-track error (distance from lane center)
            # - heading_error_rad: angular misalignment to lookahead point
            # Useful for tuning LOOKAHEAD_M and STEER_GAIN.
            # -------------------------------------------------
            show_lane_metrics=True,

            # -------------------------------------------------
            # FIGURE 5 (if enabled): Hazard Cone Parameters
            # - cone_yaw_deg: where the forward hazard cone is pointed
            # - cone_half_angle_deg: width of the cone
            # Useful when debugging steering-aware hazard detection.
            # -------------------------------------------------
            show_cone_params=True,
        )
        print("Plotted telemetry. Close plot windows or press 'q' in any window to exit.")

if __name__ == "__main__":
    main()


