"""
test3___ped_intrusion_scenario.py
Quentin | Junior Spring Research, 2026.

Goal:
- Repeatable "rare scenario" on Town04_Opt.
- Ego drives waypoint-based lane following + LiDAR braking.
- A scripted pedestrian intrudes into the ego's lane at a
  parameterized encounter distance along the route.

Parameterization:
- All scenario-level knobs live in ScenarioConfig (scenario_config.py).
- Outcome metrics are returned as RunResult (run_result.py).
- sweep.py calls run_scenario() in a loop to generate datasets.

Single-run interactive use:
    python test3___ped_intrusion_scenario.py
    (uses ScenarioConfig defaults; adjust them at the bottom of this file)
"""

import carla
# pyright: reportMissingImports=false
import math
import time
import os
import sys

# -------------------------------------------------
# CARLA PythonAPI / agents path setup
# -------------------------------------------------
CARLA_ROOT = r"C:\Users\qdruc\OneDrive\Desktop\CARLA_0.9.16"

carla_pythonapi_root = os.path.join(CARLA_ROOT, "PythonAPI")
carla_pythonapi_carla = os.path.join(CARLA_ROOT, "PythonAPI", "carla")

if carla_pythonapi_root not in sys.path:
    sys.path.append(carla_pythonapi_root)

if carla_pythonapi_carla not in sys.path:
    sys.path.append(carla_pythonapi_carla)
from agents.navigation.global_route_planner import GlobalRoutePlanner

from math_utils import get_speed_mps, mps_to_mph
from lidar_utils import draw_lidar_points, draw_lane_noodle_corridor
from spawning import prepare_spawn_context, spawn_ego_vehicle
from lidar_sensor import attach_lidar_sensor
from loop_utils import get_latest_lidar_frame
from lane_follow import lane_follow_step
from run_stats import init_run_stats, update_run_stats, print_run_summary
from carla_session import connect_and_load_world, enable_sync_mode, restore_async_mode
from spectator import SpectatorController
from telemetry_plotting import TelemetryBuffer, plot_telemetry
from scenario_config import ScenarioConfig
from run_result import RunResult, print_run_result

from walker_utils import (
    spawn_scripted_walker,
    init_scripted_crossing_state,
    trigger_scripted_crossing,
    update_scripted_crossing,
    destroy_scripted_walker,
)


# =====================================================================
# FIXED SIMULATION CONSTANTS
# (These are tuned infrastructure params, not scenario variables.
#  Scenario variables live in ScenarioConfig.)
# =====================================================================

TARGET_MAP           = "Town04_Opt"
EGO_SPAWN_INDEX      = 242    # highway straight stretch, center lane
END_MARKER_SPAWN_INDEX = 168

FIXED_DT             = 0.02   # 50 Hz simulation rate

# Lane following
LOOKAHEAD_M          = 6
STEER_GAIN           = 1.5

# LiDAR cone / corridor
CONE_HALF_ANGLE_DEG  = 13.0
MAX_CONE_YAW_DEG     = 45.0
CONE_DRAW_LENGTH_M   = 80.0
LANE_WIDTH_M         = 3.0
LANE_SIDE_MARGIN_M   = 0.4
CONE_MAX_LATERAL_M   = (LANE_WIDTH_M * 0.5) + LANE_SIDE_MARGIN_M

# LiDAR braking
BRAKE_BASE_DISTANCE_M = 5.0
BRAKE_HEADWAY_S       = 2.5
PANIC_DISTANCE_M      = 5.0
RAMP_UP_PER_S         = 8.0
RAMP_DOWN_PER_S       = 1.0

# Lane-noodle corridor
NOODLE_ENABLE        = True
NOODLE_STEP_M        = 1.25
# Narrowed from CONE_MAX_LATERAL_M (1.9m) to 1.4m.
# 1.4m = car half-width (~0.93m) + ~0.47m margin.
# Old 1.9m corridor was as wide as a full lane, catching pedestrians
# standing beside the car rather than directly in its path.
NOODLE_HALF_WIDTH_M  = 1.4
NOODLE_MAX_DIST_M    = CONE_DRAW_LENGTH_M
# Increased from 1.0m to 2.5m.
# Sensor sits on the roof ~2.3m behind the front bumper.
# Old 1.0m cutoff left the area beside the front tires inside the corridor.
# 2.5m pushes the active zone to just past the front bumper.
NOODLE_X_MIN_M       = 2.5

# Walker debug markers
WALKER_DEBUG_LIFE_S  = 12.0


# =====================================================================
# Scenario geometry helpers
# =====================================================================

def _build_route_waypoints(world, start_loc, end_loc, sampling_resolution=2.0):
    """Build a deterministic route using CARLA's GlobalRoutePlanner."""
    carla_map = world.get_map()
    grp = GlobalRoutePlanner(carla_map, sampling_resolution)
    route = grp.trace_route(start_loc, end_loc)
    if not route:
        raise RuntimeError("GlobalRoutePlanner returned an empty route.")
    return [wp for (wp, _) in route]


def _get_route_waypoint_at_distance(route_wps, distance_m):
    """Return the waypoint approximately distance_m along the route polyline."""
    if not route_wps:
        raise RuntimeError("route_wps is empty.")
    if distance_m <= 0.0:
        return route_wps[0]

    traveled = 0.0
    prev_loc = route_wps[0].transform.location
    for i in range(1, len(route_wps)):
        curr_loc = route_wps[i].transform.location
        seg_len = prev_loc.distance(curr_loc)
        if traveled + seg_len >= distance_m:
            return route_wps[i]
        traveled += seg_len
        prev_loc = curr_loc
    return route_wps[-1]


def _draw_route_debug(world, route_wps, life_time=20.0):
    """Draw the planned route as a dark-blue line for visual sanity checking."""
    for i in range(len(route_wps) - 1):
        a = route_wps[i].transform.location + carla.Location(z=0.4)
        b = route_wps[i + 1].transform.location + carla.Location(z=0.4)
        world.debug.draw_line(
            a, b,
            thickness=0.08,
            color=carla.Color(30, 30, 90),
            life_time=life_time
        )


def _compute_crossing_endpoints_from_waypoint(encounter_wp, *, side, cross, lane_width_m):
    """Build walker start/end points relative to a waypoint on the ego's route."""
    lane_center = encounter_wp.transform.location
    right = encounter_wp.transform.get_right_vector()

    start_lat = lane_width_m * 0.85
    if side.lower() == "left":
        start_lat = -start_lat
    elif side.lower() == "right":
        start_lat = +start_lat
    else:
        raise ValueError("side must be 'left' or 'right'")

    if cross.lower() == "near":
        end_lat = 0.0
    elif cross.lower() == "far":
        end_lat = -start_lat
    else:
        raise ValueError("cross must be 'near' or 'far'")

    start_loc = carla.Location(
        x=lane_center.x + right.x * start_lat,
        y=lane_center.y + right.y * start_lat,
        z=lane_center.z + 0.40
    )
    end_loc = carla.Location(
        x=lane_center.x + right.x * end_lat,
        y=lane_center.y + right.y * end_lat,
        z=lane_center.z + 0.40
    )
    return start_loc, end_loc


def _apply_weather(world, cfg: ScenarioConfig):
    """Apply weather preset + optional per-field overrides from cfg."""
    import carla as _carla
    preset = getattr(_carla.WeatherParameters, cfg.weather_preset, None)
    if preset is None:
        print(f"[weather] Unknown preset '{cfg.weather_preset}', falling back to ClearNoon.")
        preset = getattr(_carla.WeatherParameters, "ClearNoon")
    if cfg.sun_altitude_deg is not None:
        preset.sun_altitude_angle = cfg.sun_altitude_deg
    if cfg.cloudiness is not None:
        preset.cloudiness = cfg.cloudiness
    world.set_weather(preset)


def _attach_collision_sensor(world, bp_lib, vehicle):
    """
    Attach a collision sensor to the ego vehicle.
    Returns (sensor_actor, collision_flag_dict).
    collision_flag_dict["hit"] becomes True if any collision is detected.
    """
    collision_bp = bp_lib.find("sensor.other.collision")
    sensor = world.spawn_actor(
        collision_bp,
        carla.Transform(),
        attach_to=vehicle
    )
    flag = {"hit": False, "other_actor": None}

    def _on_collision(event):
        other = event.other_actor
        other_id = other.type_id if other else "unknown"
        flag["hit"] = True
        flag["other_actor"] = other_id
        print(f"[collision] ego hit: {other_id}")

    sensor.listen(_on_collision)
    return sensor, flag


# =====================================================================
# Main scenario runner
# =====================================================================

def run_scenario(
    cfg: ScenarioConfig,
    *,
    plot_after: bool = False,
    lateral_offset_fn=None,
    monitor_lateral_corridors: bool = False,
    tick_observer=None,
    post_crossing_settle_s: float = 3.0,
) -> RunResult:
    """
    Run one full scenario with the given config. Returns a RunResult.

    Args:
        cfg:        ScenarioConfig instance specifying all scenario parameters.
        plot_after: Show telemetry plots after the run. Set True for interactive
                    single runs; leave False for automated sweeps.
        lateral_offset_fn: Optional manual-test callback receiving
                    (sim_time_s, triggered, trigger_time_s) and returning a
                    signed route-relative lateral offset in meters.
        monitor_lateral_corridors: Measure original, commanded, left-candidate,
                    and right-candidate LiDAR corridors. Monitoring does not
                    change which corridor owns braking.
        tick_observer: Optional manual-test callback receiving
                    (sim_time_s, triggered, telemetry).
        post_crossing_settle_s: Tail duration after the walker reaches its
                    destination. Defaults to the historical fixed-run value.
    """
    TARGET_SPEED_MPS = cfg.target_mph * 0.44704
    total_ticks = int(cfg.sim_seconds / FIXED_DT)

    print(f"\n{'='*60}")
    print(f"TEST3: {TARGET_MAP} rare scenario")
    if cfg.run_id:
        print(f"Run ID: {cfg.run_id}")
    print(f"  start={EGO_SPAWN_INDEX}  end≈{END_MARKER_SPAWN_INDEX}")
    print(f"  sim={cfg.sim_seconds}s => {total_ticks} ticks @ dt={FIXED_DT}")
    print(f"  target={cfg.target_mph:.1f} mph | encounter={cfg.encounter_distance_m:.1f} m")
    print(f"  walker: {cfg.walker_side}/{cfg.walker_cross} @ {cfg.walker_speed_mps:.1f} m/s | startup={cfg.walker_startup_s}s")
    print(f"  weather: {cfg.weather_preset}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Outcome tracking (initialized before try so finally can read them)
    # ------------------------------------------------------------------
    result_hazard_triggered       = False
    result_collision_detected     = False
    result_stopped_before_ped     = False  # internal flag; converted to outcome string at end
    result_min_ped_dist_m         = float("inf")
    result_ego_speed_at_trigger_mps = 0.0
    result_ego_speed_at_trigger_mph = 0.0
    result_ego_dist_at_trigger_m  = 0.0
    result_trigger_time_s         = None
    result_time_to_stop_s         = None

    # Internal helpers for time_to_stop_s computation
    _first_hazard_sim_time        = None   # sim time when HAZARD_BRAKE first engaged
    _ego_fully_stopped            = False
    _ped_done_time_s              = None   # sim time when pedestrian reached destination

    # Safety / comfort metric tracking
    result_min_ttc_s = float("inf")
    result_max_jerk  = 0.0
    _jerk_abs_sum    = 0.0
    _jerk_count      = 0
    _prev_speed_mps  = 0.0
    _prev_accel_mps2 = 0.0
    # Low-pass filtered speed for jerk computation.
    # Raw double-differencing amplifies tick noise by 1/dt^2 = 2500x; EMA tames that
    # while still tracking real braking events (which unfold over 10-20 ticks).
    # alpha=0.25 gives ~3-tick smoothing window (~60ms) — fast enough to catch the
    # braking ramp, slow enough to reject PhysX stiction/quantisation spikes.
    _JERK_LP_ALPHA   = 0.25
    _speed_lp        = 0.0  # initialised to 0; will converge before trigger fires

    # ------------------------------------------------------------------
    # Connect + sync
    # ------------------------------------------------------------------
    _, world = connect_and_load_world(
        host="localhost",
        port=2000,
        timeout_s=10.0,
        target_map=TARGET_MAP
    )
    enable_sync_mode(world, fixed_dt=FIXED_DT)
    _apply_weather(world, cfg)

    vehicle         = None
    lidar           = None
    collision_sensor = None
    walker          = None

    try:
        # ------------------------------------------------------------------
        # Spawn ego + sensors
        # ------------------------------------------------------------------
        spawn_points, ego_tf, bp_lib = prepare_spawn_context(world, SPAWN_INDEX=EGO_SPAWN_INDEX)
        vehicle = spawn_ego_vehicle(world, bp_lib=bp_lib, spawn_tf=ego_tf)

        # Apply road friction override if specified in config
        if cfg.road_friction is not None:
            phys = vehicle.get_physics_control()  # type: ignore[attr-defined]
            wheels = phys.wheels
            for w in wheels:
                w.tire_friction = cfg.road_friction
            phys.wheels = wheels  # must reassign back — CARLA doesn't see in-place mutations
            vehicle.apply_physics_control(phys)  # type: ignore[attr-defined]
            print(f"[physics] tire_friction overridden -> {cfg.road_friction:.2f}")

        lidar, lidar_queue = attach_lidar_sensor(world, bp_lib, vehicle)
        collision_sensor, collision_flag = _attach_collision_sensor(world, bp_lib, vehicle)

        spec_controller = SpectatorController(world, vehicle)

        # End-region marker (visual anchor only)
        if 0 <= END_MARKER_SPAWN_INDEX < len(spawn_points):
            end_loc = spawn_points[END_MARKER_SPAWN_INDEX].location
            world.debug.draw_point(
                end_loc + carla.Location(z=1.5),
                size=0.35, color=carla.Color(255, 140, 0), life_time=30.0
            )
            world.debug.draw_string(
                end_loc + carla.Location(z=2.1),
                f"END REGION (spawn {END_MARKER_SPAWN_INDEX})",
                color=carla.Color(255, 140, 0), life_time=30.0
            )

        # ------------------------------------------------------------------
        # Build route + place pedestrian
        # ------------------------------------------------------------------
        if not (0 <= END_MARKER_SPAWN_INDEX < len(spawn_points)):
            raise RuntimeError("END_MARKER_SPAWN_INDEX is out of range.")

        end_route_loc = spawn_points[END_MARKER_SPAWN_INDEX].location
        route_wps = _build_route_waypoints(
            world, ego_tf.location, end_route_loc, sampling_resolution=2.0
        )
        route_points_world = [wp.transform.location for wp in route_wps]
        _draw_route_debug(world, route_wps, life_time=30.0)

        encounter_wp = _get_route_waypoint_at_distance(route_wps, cfg.encounter_distance_m)
        enc_loc = encounter_wp.transform.location

        world.debug.draw_point(enc_loc + carla.Location(z=1.0),
                               size=0.20, color=carla.Color(0, 255, 255), life_time=30.0)
        world.debug.draw_string(enc_loc + carla.Location(z=1.6), "ENCOUNTER_WP",
                                color=carla.Color(0, 255, 255), life_time=30.0)

        walker_start_loc, walker_end_loc = _compute_crossing_endpoints_from_waypoint(
            encounter_wp,
            side=cfg.walker_side,
            cross=cfg.walker_cross,
            lane_width_m=LANE_WIDTH_M
        )

        spawn_yaw_deg = math.degrees(math.atan2(
            walker_end_loc.y - walker_start_loc.y,
            walker_end_loc.x - walker_start_loc.x
        ))
        walker_tf = carla.Transform(walker_start_loc, carla.Rotation(yaw=spawn_yaw_deg))

        walker = spawn_scripted_walker(world, start_tf=walker_tf)
        if walker is None:
            raise RuntimeError("Failed to spawn scripted walker.")

        world.debug.draw_point(walker_start_loc + carla.Location(z=0.9),
                               size=0.18, color=carla.Color(255, 0, 0), life_time=30.0)
        world.debug.draw_point(walker_end_loc + carla.Location(z=0.9),
                               size=0.18, color=carla.Color(0, 255, 0), life_time=30.0)
        world.debug.draw_line(
            walker_start_loc + carla.Location(z=0.9),
            walker_end_loc + carla.Location(z=0.9),
            thickness=0.08, color=carla.Color(255, 255, 0), life_time=30.0
        )

        crossing_state = init_scripted_crossing_state(
            start_loc=walker_start_loc,
            end_loc=walker_end_loc,
            speed_mps=cfg.walker_speed_mps
        )

        # ------------------------------------------------------------------
        # Controller state
        # ------------------------------------------------------------------
        speed_state = {
            "i_term": 0.0,
            "kp": 0.50,
            "ki": 0.10,
            "brake_prev": 0.0,
            "max_cone_yaw_deg": MAX_CONE_YAW_DEG,
            "cone_half_angle_deg": CONE_HALF_ANGLE_DEG,
            "cone_draw_length_m": CONE_DRAW_LENGTH_M,
            "drive_mode": "CRUISE",
            "hazard_clear_ticks": 0,
            "cone_max_lateral_m": CONE_MAX_LATERAL_M,
            "throttle_prev": 0.0,
            "brake_prev_cruise": 0.0,
            "lidar_actor": lidar,
            "route_points_world": route_points_world,
            "noodle_enable": NOODLE_ENABLE,
            "noodle_step_m": NOODLE_STEP_M,
            "noodle_half_width_m": NOODLE_HALF_WIDTH_M,
            "noodle_max_dist_m": NOODLE_MAX_DIST_M,
            "noodle_x_min_m": NOODLE_X_MIN_M,
            "monitor_lateral_corridors": monitor_lateral_corridors,
            "candidate_lateral_offset_m": LANE_WIDTH_M * 0.5,
        }

        # ------------------------------------------------------------------
        # Metrics / telemetry
        # ------------------------------------------------------------------
        stats = init_run_stats()
        telemetry_buffer = TelemetryBuffer()

        dist_m   = 0.0
        prev_loc = None

        triggered              = False
        trigger_time_s         = None
        crossing_motion_started = False

        next_time = time.perf_counter()

        # ------------------------------------------------------------------
        # Run loop
        # ------------------------------------------------------------------
        last_lidar_frame = None
        last_lidar_tick  = -1

        for t in range(total_ticks):
            # --- Advance simulation ---
            world.tick()

            # --- Distance tracking ---
            loc_now = vehicle.get_transform().location
            if prev_loc is not None:
                dist_m += loc_now.distance(prev_loc)
            prev_loc = loc_now

            sim_time_s   = t * FIXED_DT
            ego_speed_mps = get_speed_mps(vehicle)

            # --- Jerk (comfort metric) ---
            # Always update velocity/accel history for a smooth derivative,
            # but only accumulate into result metrics after the trigger fires.
            # Pre-trigger jerk (spawn settling, acceleration ramp) is irrelevant.
            # Low-pass filter speed first: raw double-differencing at 50 Hz amplifies
            # tick noise by 1/dt^2 = 2500x, causing phantom 1000+ m/s^3 spikes.
            _speed_lp    = _JERK_LP_ALPHA * ego_speed_mps + (1.0 - _JERK_LP_ALPHA) * _speed_lp
            _accel_mps2  = (_speed_lp - _prev_speed_mps) / FIXED_DT
            _jerk        = abs((_accel_mps2 - _prev_accel_mps2) / FIXED_DT)
            _prev_speed_mps  = _speed_lp
            _prev_accel_mps2 = _accel_mps2
            if triggered:
                if _jerk > result_max_jerk:
                    result_max_jerk = _jerk
                _jerk_abs_sum += _jerk
                _jerk_count   += 1

            # --- TTC (risk metric, only meaningful after trigger fires) ---
            # Only computed when the pedestrian is AHEAD of the ego (positive dot
            # product with ego forward vector). A ped walking alongside or behind
            # the car is not a collision threat and would produce misleadingly low TTC.
            _ttc_this_tick = float("inf")
            if triggered and ego_speed_mps > 0.5 and walker is not None:
                _ego_tf   = vehicle.get_transform()
                _ego_fwd  = _ego_tf.get_forward_vector()
                _ped_loc  = walker.get_location()
                _to_ped_x = _ped_loc.x - _ego_tf.location.x
                _to_ped_y = _ped_loc.y - _ego_tf.location.y
                _ped_is_ahead = (_to_ped_x * _ego_fwd.x + _to_ped_y * _ego_fwd.y) > 0
                if _ped_is_ahead:
                    _ped_dist_now = _ego_tf.location.distance(_ped_loc)
                    _ttc_this_tick = _ped_dist_now / ego_speed_mps
                    if _ttc_this_tick < result_min_ttc_s:
                        result_min_ttc_s = _ttc_this_tick

            # --- Pedestrian trigger ---
            if cfg.trigger_ttc_s is not None:
                # Direct "seconds-before-crossing" mode: trigger fires when ego
                # is exactly trigger_ttc_s seconds from the encounter point.
                lead_time_s = cfg.trigger_ttc_s
            else:
                # Legacy formula: startup hesitation + full crossing time.
                cross_dist_m = walker_start_loc.distance(walker_end_loc)
                lead_time_s  = cfg.walker_startup_s + (cross_dist_m / max(cfg.walker_speed_mps, 0.1))
            lead_dist_m   = ego_speed_mps * lead_time_s
            trigger_dist_m = max(0.0, cfg.encounter_distance_m - lead_dist_m)

            if (not triggered) and (dist_m >= trigger_dist_m) and (sim_time_s >= cfg.trigger_delay_s):
                triggered      = True
                trigger_time_s = sim_time_s

                # Capture ego state at trigger moment
                result_ego_speed_at_trigger_mps = ego_speed_mps
                result_ego_speed_at_trigger_mph  = mps_to_mph(ego_speed_mps)
                result_ego_dist_at_trigger_m     = dist_m
                result_trigger_time_s            = sim_time_s

                trigger_scripted_crossing(
                    world, crossing_state, debug_life_s=WALKER_DEBUG_LIFE_S
                )
                print(
                    f"[scenario] TRIGGERED at t={sim_time_s:.2f}s | dist={dist_m:.1f}m | "
                    f"trigger_dist={trigger_dist_m:.1f}m | encounter={cfg.encounter_distance_m:.1f}m | "
                    f"ego={mps_to_mph(ego_speed_mps):.1f} mph | TTC_at_trigger={lead_time_s:.2f}s"
                )

            if triggered and (not crossing_motion_started):
                if trigger_time_s is not None and (sim_time_s - trigger_time_s) >= cfg.walker_post_trigger_delay_s:
                    crossing_motion_started = True

            if crossing_motion_started:
                update_scripted_crossing(walker, crossing_state, fixed_dt=FIXED_DT)

            # Walker debug label
            if walker is not None:
                walker_loc_dbg = walker.get_transform().location
                world.debug.draw_point(
                    walker_loc_dbg + carla.Location(z=1.0),
                    size=0.14, color=carla.Color(255, 0, 255), life_time=FIXED_DT * 1.2
                )
                world.debug.draw_string(
                    walker_loc_dbg + carla.Location(z=1.6),
                    "WALKER", color=carla.Color(255, 0, 255), life_time=FIXED_DT * 1.2
                )

            # --- LiDAR ---
            lidar_frame, last_lidar_frame, last_lidar_tick = get_latest_lidar_frame(
                lidar_queue,
                last_lidar_frame=last_lidar_frame,
                last_lidar_tick=last_lidar_tick,
                t=t
            )

            requested_lateral_offset_m = 0.0
            if lateral_offset_fn is not None:
                requested_lateral_offset_m = float(
                    lateral_offset_fn(sim_time_s, triggered, trigger_time_s)
                )

            # --- Controller step ---
            telemetry = lane_follow_step(
                world, vehicle,
                lookahead_m=LOOKAHEAD_M,
                steer_gain=STEER_GAIN,
                target_speed_mps=TARGET_SPEED_MPS,
                fixed_dt=FIXED_DT,
                speed_state=speed_state,
                lidar_frame=lidar_frame,
                FIXED_DT=FIXED_DT,
                base_distance_m=BRAKE_BASE_DISTANCE_M,
                headway_seconds=cfg.brake_headway_s,
                panic_distance_m=PANIC_DISTANCE_M,
                ramp_up_per_s=cfg.braking_ramp_up_per_s,
                ramp_down_per_s=RAMP_DOWN_PER_S,
                brake_profile=cfg.brake_profile,
                lateral_offset_m=requested_lateral_offset_m,
            )

            # Belt-and-suspenders: once the ego has made a full emergency stop,
            # keep drive_mode pinned to STOP_HOLD for the rest of the scenario.
            # This is a scenario-level override that complements the hazard_clear
            # fix in lane_follow.py — it ensures the ego cannot resume even if
            # LiDAR briefly loses the stopped pedestrian at close range.
            if _ego_fully_stopped:
                speed_state["drive_mode"] = "STOP_HOLD"
                speed_state["hazard_clear_ticks"] = 0

            if telemetry is not None:
                telemetry["jerk_mps3"]  = _jerk if triggered else float("nan")
                telemetry["ttc_s"]      = _ttc_this_tick if _ttc_this_tick != float("inf") else float("nan")

            if tick_observer is not None:
                tick_observer(sim_time_s, triggered, telemetry)

            telemetry_buffer.append(tick=t, fixed_dt=FIXED_DT, telemetry=telemetry)
            update_run_stats(stats, telemetry=telemetry, vehicle=vehicle, FIXED_DT=FIXED_DT)

            # --- Outcome tracking ---
            # Min pedestrian distance.
            # Only tracked after the crossing trigger fires — before that the
            # pedestrian is standing stationary off to the side of the lane and
            # is not yet a hazard, so any distance recorded would be meaningless.
            # Also only tracked when the pedestrian is AHEAD of the ego (positive
            # dot product with forward vector) — this excludes safe lateral passes
            # in far-cross mode where the ped walks alongside the car after crossing.
            if triggered and walker is not None:
                _ego_tf2   = vehicle.get_transform()
                _ego_fwd2  = _ego_tf2.get_forward_vector()
                _ped_loc2  = walker.get_location()
                _to_ped2_x = _ped_loc2.x - _ego_tf2.location.x
                _to_ped2_y = _ped_loc2.y - _ego_tf2.location.y
                if (_to_ped2_x * _ego_fwd2.x + _to_ped2_y * _ego_fwd2.y) > 0:
                    ped_dist = _ego_tf2.location.distance(_ped_loc2)
                    if ped_dist < result_min_ped_dist_m:
                        result_min_ped_dist_m = ped_dist
                    # Proximity-based collision fallback.
                    # CARLA's collision sensor requires a minimum impulse and will
                    # NOT fire if the ego contacts the pedestrian at low speed after
                    # heavy braking. Center-to-center contact distance for a typical
                    # CARLA sedan is ~2.6 m (vehicle front half ~2.3 m + ped radius
                    # ~0.3 m). If the ego is still moving at that range, it is a hit.
                    #
                    # Lateral corridor guard: in far-cross mode the ped may end up
                    # alongside the car after crossing — still "ahead" by dot product
                    # but laterally off to the side, no actual contact possible.
                    # We compute the perpendicular (lateral) distance from the ego's
                    # forward axis and only flag a collision if the ped is within
                    # the vehicle's direct forward corridor (~half-width + ped radius).
                    _CONTACT_DIST_M  = 2.7   # frontal center-to-center contact range
                    _MAX_LATERAL_M   = 1.2   # vehicle half-width ~1.0m + ped radius ~0.2m
                    # lateral offset = magnitude of cross-product of fwd x to_ped (2D)
                    _lateral_m = abs(
                        _ego_fwd2.x * _to_ped2_y - _ego_fwd2.y * _to_ped2_x
                    )
                    # Only apply in near-cross mode: in far-cross the ped walks
                    # ALL the way through the corridor, so proximity will fire
                    # mid-crossing even when the ped safely clears. The CARLA
                    # sensor is the only reliable detector for far-cross contacts.
                    if (ped_dist < _CONTACT_DIST_M
                            and ego_speed_mps > 0.5
                            and _lateral_m < _MAX_LATERAL_M
                            and cfg.walker_cross == "near"):
                        if not result_collision_detected:
                            print(f"[collision] proximity contact: "
                                  f"dist={ped_dist:.2f}m  lateral={_lateral_m:.2f}m  "
                                  f"speed={ego_speed_mps:.2f}m/s")
                        result_collision_detected = True

            # Hazard brake engagement
            drive_mode = speed_state.get("drive_mode", "CRUISE")
            if drive_mode in ("HAZARD_BRAKE", "STOP_HOLD") and not result_hazard_triggered:
                result_hazard_triggered = True
                _first_hazard_sim_time  = sim_time_s

            # Time-to-stop after hazard
            if result_hazard_triggered and not _ego_fully_stopped:
                if ego_speed_mps < 0.3:
                    _ego_fully_stopped = True
                    if _first_hazard_sim_time is not None:
                        result_time_to_stop_s = sim_time_s - _first_hazard_sim_time

            # Full stop: ego reached near-zero speed at any point after hazard engaged.
            # Pedestrian crossing state is intentionally NOT checked here — the ego
            # stopping completely is the meaningful safety outcome regardless of whether
            # the ped had already reached their destination by that tick.
            if result_hazard_triggered and ego_speed_mps < 0.3:
                result_stopped_before_ped = True

            # Collision sensor
            if collision_flag["hit"]:
                result_collision_detected = True

            # --- Dynamic termination ---
            # End the run 3 seconds after the pedestrian reaches its destination.
            # This is far cleaner than a fixed sim_seconds: every run ends at the
            # earliest moment the scenario is fully resolved, regardless of speed
            # or encounter distance. The 3s tail gives the ego time to complete its
            # stop or resume, and gives collision/proximity checks time to fire.
            POST_CROSSING_SETTLE_S = post_crossing_settle_s
            if crossing_state.get("done", False):
                if _ped_done_time_s is None:
                    _ped_done_time_s = sim_time_s
                    print(f"[run] Pedestrian reached destination at t={sim_time_s:.2f}s"
                          f" — settling {POST_CROSSING_SETTLE_S:.0f}s then ending run.")
                elif sim_time_s >= _ped_done_time_s + POST_CROSSING_SETTLE_S:
                    break

            # --- Visualization ---
            draw_lidar_points(world, lidar_frame, lidar, sample_step=60, life_time=0.10)
            if telemetry is not None:
                noodle_pts = speed_state.get("noodle_points_world", None)
                _noodle_hazard = speed_state.get("drive_mode", "CRUISE") in ("HAZARD_BRAKE", "STOP_HOLD")
                draw_lane_noodle_corridor(
                    world, noodle_pts,
                    half_width_m=speed_state.get("noodle_half_width_m", NOODLE_HALF_WIDTH_M),
                    life_time=FIXED_DT * 1.05,
                    z_offset=0.10,
                    hazard_active=_noodle_hazard,
                )
                if monitor_lateral_corridors and abs(requested_lateral_offset_m) > 0.05:
                    commanded_pts = speed_state.get("commanded_noodle_points_world", None)
                    commanded_distance = telemetry.get("d_min_commanded_path_m")
                    draw_lane_noodle_corridor(
                        world,
                        commanded_pts,
                        half_width_m=speed_state.get(
                            "noodle_half_width_m", NOODLE_HALF_WIDTH_M
                        ),
                        life_time=FIXED_DT * 1.05,
                        z_offset=0.18,
                        hazard_active=commanded_distance is not None,
                        center_color=carla.Color(20, 180, 70),
                        normal_edge_color=carla.Color(20, 220, 100),
                        hazard_edge_color=carla.Color(220, 20, 180),
                        normal_tick_color=carla.Color(20, 160, 80),
                        hazard_tick_color=carla.Color(180, 20, 150),
                    )

            # --- Spectator ---
            spec_controller.tick(FIXED_DT)

            # --- Pacing ---
            next_time += FIXED_DT
            sleep_for = next_time - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_time = time.perf_counter()

    finally:
        spec_controller.close()

        print_run_summary(stats)

        # Destroy actors
        if collision_sensor is not None:
            collision_sensor.stop()     # pyright: ignore
            collision_sensor.destroy()
        destroy_scripted_walker(walker)
        if lidar is not None:
            lidar.destroy()
        if vehicle is not None:
            vehicle.destroy()

        restore_async_mode(world)
        print("Clean exit: destroyed actors + restored world settings.")

        # Build result
        if result_collision_detected:
            _outcome = "collision"
        elif result_stopped_before_ped:
            _outcome = "full_stop"
        else:
            _outcome = "slowed_avoided"

        _mean_jerk = (_jerk_abs_sum / _jerk_count) if _jerk_count > 0 else 0.0
        result = RunResult(
            hazard_triggered=result_hazard_triggered,
            collision_detected=result_collision_detected,
            outcome=_outcome,
            min_ped_distance_m=result_min_ped_dist_m if result_min_ped_dist_m != float("inf") else -1.0,
            min_ttc_s=result_min_ttc_s if result_min_ttc_s != float("inf") else -1.0,
            max_jerk_mps3=result_max_jerk,
            mean_jerk_mps3=_mean_jerk,
            ego_speed_at_trigger_mps=result_ego_speed_at_trigger_mps,
            ego_speed_at_trigger_mph=result_ego_speed_at_trigger_mph,
            ego_dist_at_trigger_m=result_ego_dist_at_trigger_m,
            trigger_time_s=result_trigger_time_s,
            time_to_stop_s=result_time_to_stop_s,
            config=cfg,
        )
        print_run_result(result)

        if plot_after:
            plot_telemetry(
                telemetry_buffer,
                target_mph=cfg.target_mph,
                show_speed=True,
                show_braking_view=True,
                show_safety_metrics=True,
                show_controls=False,
                show_lane_metrics=False,
                show_cone_params=False,
            )

    return result


# =====================================================================
# Single-run interactive entrypoint
# =====================================================================

def main():
    """
    Run one scenario interactively with default (or hand-edited) config.
    Edit the ScenarioConfig fields below to change behavior for a single run.
    For automated multi-run sweeps, use sweep.py instead.
    """
    cfg = ScenarioConfig(
        # --- Pedestrian ---
        walker_speed_mps=1.8,
        walker_side="left",
        walker_cross="near",
        trigger_ttc_s=2.8,   # seconds before crossing when trigger fires (None = use legacy formula)
        trigger_delay_s=0.0,
        walker_post_trigger_delay_s=0.0,

        # --- Ego ---
        target_mph=35.0,
        encounter_distance_m=120.0,

        # --- Braking profile ---
        # Options (change this line to compare them):
        #   "proportional_ramp"  -- linear 0->1 as obstacle closes in (baseline)
        #   "step_constant"      -- full brake immediately on detection (max jerk)
        #   "cautious_ramp"      -- same as proportional but capped at 0.5 force
        #   "exponential"        -- penetration^2 curve (gentle->urgent)
        brake_profile="exponential",

        # --- Environment ---
        weather_preset="ClearSunset",
        sun_altitude_deg=0,     # 0 = horizon level && None = use preset default
        cloudiness=None,        # None = use preset default

        # --- Simulation ---
        sim_seconds=13.0,
    )

    run_scenario(cfg, plot_after=True)


if __name__ == "__main__":
    main()
