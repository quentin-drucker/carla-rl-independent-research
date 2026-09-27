"""Isolate ego-rooted swept-path LiDAR from the original corridor.

No pedestrian is spawned. The ego voluntarily shifts far route-right and
holds that offset toward a parked Model 3. The parked car is too far from the
original route corridor to trigger it, so any hazard braking must come from
``d_min_ego_swept_path_m``.

Run CARLA windowed, then from ``src``:
    python -X utf8 test23___isolated_swept_path_parked_car.py
"""

import math
import os
import sys

import carla

CARLA_ROOT = r"C:\Users\qdruc\OneDrive\Desktop\CARLA_0.9.16"
for path in (
    os.path.join(CARLA_ROOT, "PythonAPI"),
    os.path.join(CARLA_ROOT, "PythonAPI", "carla"),
):
    if path not in sys.path:
        sys.path.append(path)

from agents.navigation.global_route_planner import GlobalRoutePlanner

from carla_session import connect_and_load_world, enable_sync_mode, restore_async_mode
from lane_follow import lane_follow_step
from lidar_sensor import attach_lidar_sensor
from loop_utils import get_latest_lidar_frame
from math_utils import get_speed_mps, mps_to_mph
from route_lateral_control import offset_point_xy
from spawning import prepare_spawn_context, spawn_ego_vehicle
from spectator import SpectatorController


TARGET_MAP = "Town04_Opt"
EGO_SPAWN_INDEX = 242
END_MARKER_SPAWN_INDEX = 168
FIXED_DT = 0.02
TARGET_MPH = 25.0
PARKED_CAR_ROUTE_DISTANCE_M = 96.0
LATERAL_OFFSET_M = 4.882
SHIFT_START_S = 2.0
SHIFT_DURATION_S = 0.5
SIM_SECONDS = 20.0


def _smoothstep(value):
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def _requested_offset_m(sim_time_s):
    if sim_time_s < SHIFT_START_S:
        return 0.0
    if sim_time_s < SHIFT_START_S + SHIFT_DURATION_S:
        fraction = (sim_time_s - SHIFT_START_S) / SHIFT_DURATION_S
        return LATERAL_OFFSET_M * _smoothstep(fraction)
    return LATERAL_OFFSET_M


def _route_location_at_distance(route_points, distance_m):
    traveled_m = 0.0
    for previous, current in zip(route_points, route_points[1:]):
        segment_m = previous.distance(current)
        if traveled_m + segment_m >= distance_m:
            fraction = (distance_m - traveled_m) / max(segment_m, 1e-9)
            return carla.Location(
                x=previous.x + fraction * (current.x - previous.x),
                y=previous.y + fraction * (current.y - previous.y),
                z=previous.z + fraction * (current.z - previous.z),
            )
        traveled_m += segment_m
    return route_points[-1]


def main():
    world = None
    ego = None
    parked_car = None
    lidar = None
    spectator = None

    original_hazard_ticks = 0
    swept_detection_ticks = 0
    swept_hazard_ticks = 0
    swept_governing_ticks = 0
    active_swept_path_ticks = 0
    positive_brake_ticks = 0
    max_brake_cmd = 0.0
    min_ego_car_distance_m = float("inf")
    first_swept_hazard = None
    max_actual_offset_m = 0.0

    try:
        _, world = connect_and_load_world(
            host="localhost", port=2000, timeout_s=10.0, target_map=TARGET_MAP
        )
        enable_sync_mode(world, fixed_dt=FIXED_DT)
        spawn_points, ego_transform, blueprint_library = prepare_spawn_context(
            world, SPAWN_INDEX=EGO_SPAWN_INDEX
        )
        ego = spawn_ego_vehicle(world, bp_lib=blueprint_library, spawn_tf=ego_transform)

        planner = GlobalRoutePlanner(world.get_map(), 2.0)
        route = planner.trace_route(
            ego_transform.location,
            spawn_points[END_MARKER_SPAWN_INDEX].location,
        )
        route_points = [waypoint.transform.location for waypoint, _ in route]
        parked_anchor = _route_location_at_distance(
            route_points, PARKED_CAR_ROUTE_DISTANCE_M
        )
        parked_x, parked_y = offset_point_xy(
            route_points, parked_anchor, LATERAL_OFFSET_M
        )
        parked_waypoint = world.get_map().get_waypoint(
            parked_anchor, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        parked_transform = carla.Transform(
            carla.Location(x=parked_x, y=parked_y, z=parked_anchor.z + 0.3),
            parked_waypoint.transform.rotation,
        )
        parked_car = world.try_spawn_actor(
            blueprint_library.find("vehicle.tesla.model3"), parked_transform
        )
        if parked_car is None:
            raise RuntimeError("Failed to spawn isolated parked car.")
        parked_car.apply_control(
            carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
        )

        lidar, lidar_queue = attach_lidar_sensor(world, blueprint_library, ego)
        spectator = SpectatorController(world, ego)
        speed_state = {
            "i_term": 0.0,
            "kp": 0.50,
            "ki": 0.10,
            "brake_prev": 0.0,
            "drive_mode": "CRUISE",
            "hazard_clear_ticks": 0,
            "throttle_prev": 0.0,
            "brake_prev_cruise": 0.0,
            "lidar_actor": lidar,
            "route_points_world": route_points,
            "noodle_enable": True,
            "noodle_step_m": 1.25,
            "noodle_half_width_m": 1.4,
            "noodle_max_dist_m": 80.0,
            "noodle_x_min_m": 2.5,
            "monitor_lateral_corridors": True,
            "candidate_lateral_offset_m": 1.5,
            "transition_blend_distance_m": 20.0,
        }

        last_lidar_frame = None
        last_lidar_tick = -1
        for tick in range(int(SIM_SECONDS / FIXED_DT)):
            world.tick()
            sim_time_s = tick * FIXED_DT
            lidar_frame, last_lidar_frame, last_lidar_tick = get_latest_lidar_frame(
                lidar_queue,
                last_lidar_frame=last_lidar_frame,
                last_lidar_tick=last_lidar_tick,
                t=tick,
            )
            requested_offset_m = _requested_offset_m(sim_time_s)
            telemetry = lane_follow_step(
                world,
                ego,
                lookahead_m=6.0,
                steer_gain=1.5,
                target_speed_mps=TARGET_MPH * 0.44704,
                fixed_dt=FIXED_DT,
                speed_state=speed_state,
                lidar_frame=lidar_frame,
                FIXED_DT=FIXED_DT,
                base_distance_m=5.0,
                headway_seconds=2.5,
                panic_distance_m=5.0,
                ramp_up_per_s=8.0,
                ramp_down_per_s=1.0,
                brake_profile="exponential",
                lateral_offset_m=requested_offset_m,
                use_swept_path_clearance_override=True,
            )
            if telemetry is None:
                raise RuntimeError("lane_follow_step returned no telemetry")

            actual_offset_m = telemetry["signed_route_lateral_offset_m"]
            max_actual_offset_m = max(max_actual_offset_m, abs(actual_offset_m))
            trigger_distance_m = telemetry["trigger_distance_m"]
            original_distance_m = telemetry["d_min_original_path_m"]
            swept_distance_m = telemetry["d_min_ego_swept_path_m"]
            ego_car_distance_m = ego.get_location().distance(parked_car.get_location())
            min_ego_car_distance_m = min(min_ego_car_distance_m, ego_car_distance_m)

            original_hazard = (
                original_distance_m is not None
                and original_distance_m < trigger_distance_m
            )
            swept_hazard = (
                swept_distance_m is not None
                and swept_distance_m < trigger_distance_m
            )
            original_hazard_ticks += int(original_hazard)
            swept_detection_ticks += int(
                telemetry["ego_swept_path_lidar_status"] != "no_return"
            )
            swept_hazard_ticks += int(swept_hazard)
            swept_governing_ticks += int(
                telemetry["hazard_governing_source"].startswith("ego_swept")
            )
            active_swept_path_ticks += int(
                telemetry["active_braking_path"] == "ego_swept"
            )
            brake_cmd = telemetry["brake_cmd"]
            positive_brake_ticks += int(brake_cmd > 0.001)
            max_brake_cmd = max(max_brake_cmd, brake_cmd)
            if swept_hazard and first_swept_hazard is None:
                first_swept_hazard = {
                    "time_s": sim_time_s,
                    "ego_car_distance_m": ego_car_distance_m,
                    "swept_distance_m": swept_distance_m,
                    "trigger_distance_m": trigger_distance_m,
                    "speed_mph": mps_to_mph(get_speed_mps(ego)),
                }

            if tick % 50 == 0:
                print(
                    f"t={sim_time_s:5.2f}s speed={mps_to_mph(get_speed_mps(ego)):5.1f}mph "
                    f"request={requested_offset_m:+5.2f} actual={actual_offset_m:+5.2f} "
                    f"car={ego_car_distance_m:5.1f}m orig={original_distance_m} "
                    f"swept={swept_distance_m} mode={telemetry['drive_mode']}"
                )
            spectator.tick(FIXED_DT)

        print("\n" + "=" * 70)
        print("test23 isolated swept-path parked-car result")
        print(f"Maximum actual lateral offset: {max_actual_offset_m:.2f}m")
        print(f"Original-corridor hazard ticks: {original_hazard_ticks}")
        print(f"Swept-path detection ticks: {swept_detection_ticks}")
        print(f"Swept-path hazard ticks: {swept_hazard_ticks}")
        print(f"Swept path governed ticks: {swept_governing_ticks}")
        print(f"Swept path active-authority ticks: {active_swept_path_ticks}")
        print(f"Positive brake-command ticks: {positive_brake_ticks}")
        print(f"Maximum brake command: {max_brake_cmd:.4f}")
        print(f"Minimum ego<->parked-car center distance: {min_ego_car_distance_m:.2f}m")
        print(f"First swept hazard: {first_swept_hazard}")
        checks = {
            "original corridor stayed clear": original_hazard_ticks == 0,
            "swept path detected returns": swept_detection_ticks > 0,
            "swept path crossed brake threshold": swept_hazard_ticks > 0,
            "swept path governed braking": swept_governing_ticks > 0,
            "swept path owned committed maneuver": active_swept_path_ticks > 0,
            "braking was commanded": positive_brake_ticks > 0,
        }
        print(f"Validation checks: {checks}")
        print(f"VALIDATION: {'PASS' if all(checks.values()) else 'FAIL'}")
        print("=" * 70)
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise AssertionError(f"Isolated swept-path validation failed: {failed}")
    finally:
        if spectator is not None:
            spectator.close()
        if lidar is not None:
            lidar.stop()
            lidar.destroy()
        if parked_car is not None:
            parked_car.destroy()
        if ego is not None:
            ego.destroy()
        if world is not None:
            restore_async_mode(world)
        print("Clean exit: destroyed actors and restored asynchronous mode.")


if __name__ == "__main__":
    main()
