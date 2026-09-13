"""Manual CARLA smoke test for route-relative lateral control.

This is intentionally not an RL or pedestrian test. It verifies the mechanical
control sequence required before either is introduced:

    follow route -> ramp lateral offset -> hold -> return to route

Run CARLA in windowed mode first, then from ``src`` run:

    python test4___scripted_lateral_offset.py
    python test4___scripted_lateral_offset.py --lateral-offset-m -0.75

Positive offset means route-right; negative means route-left.
"""

import argparse
import math
import os
import sys

import carla
# pyright: reportMissingImports=false

# Match the established project entry points: CARLA's navigation agents are
# shipped with the simulator rather than the pip-installed ``carla`` module.
CARLA_ROOT = r"C:\Users\qdruc\OneDrive\Desktop\CARLA_0.9.16"
carla_pythonapi_root = os.path.join(CARLA_ROOT, "PythonAPI")
carla_pythonapi_carla = os.path.join(CARLA_ROOT, "PythonAPI", "carla")
if carla_pythonapi_root not in sys.path:
    sys.path.append(carla_pythonapi_root)
if carla_pythonapi_carla not in sys.path:
    sys.path.append(carla_pythonapi_carla)

from agents.navigation.global_route_planner import GlobalRoutePlanner

from carla_session import connect_and_load_world, enable_sync_mode, restore_async_mode
from lane_follow import lane_follow_step
from math_utils import mps_to_mph
from spawning import prepare_spawn_context, spawn_ego_vehicle
from spectator import SpectatorController


TARGET_MAP = "Town04_Opt"
EGO_SPAWN_INDEX = 242
END_MARKER_SPAWN_INDEX = 168
FIXED_DT = 0.02
LOOKAHEAD_M = 6.0
STEER_GAIN = 1.5


def _build_route_points(world, start_location, end_location):
    planner = GlobalRoutePlanner(world.get_map(), 2.0)
    route = planner.trace_route(start_location, end_location)
    if not route:
        raise RuntimeError("GlobalRoutePlanner returned an empty route.")
    return [waypoint.transform.location for waypoint, _ in route]


def _draw_route(world, route_points, life_time_s):
    for start, end in zip(route_points, route_points[1:]):
        world.debug.draw_line(
            start + carla.Location(z=0.4),
            end + carla.Location(z=0.4),
            thickness=0.08,
            color=carla.Color(30, 30, 90),
            life_time=life_time_s,
        )


def _smoothstep(value):
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def scripted_offset_m(sim_time_s, peak_offset_m):
    """Smooth test schedule: center, shift, hold, recover, center."""
    if sim_time_s < 4.0:
        return 0.0
    if sim_time_s < 6.0:
        return peak_offset_m * _smoothstep((sim_time_s - 4.0) / 2.0)
    if sim_time_s < 8.0:
        return peak_offset_m
    if sim_time_s < 10.0:
        return peak_offset_m * (1.0 - _smoothstep((sim_time_s - 8.0) / 2.0))
    return 0.0


def run_test(*, target_mph, lateral_offset_m, sim_seconds):
    world = None
    vehicle = None
    spectator = None

    max_abs_actual_offset_m = 0.0
    max_abs_steer = 0.0
    final_actual_offset_m = float("nan")

    try:
        _, world = connect_and_load_world(
            host="localhost", port=2000, timeout_s=10.0, target_map=TARGET_MAP
        )
        enable_sync_mode(world, fixed_dt=FIXED_DT)

        spawn_points, ego_transform, blueprint_library = prepare_spawn_context(
            world, SPAWN_INDEX=EGO_SPAWN_INDEX
        )
        if not 0 <= END_MARKER_SPAWN_INDEX < len(spawn_points):
            raise RuntimeError("END_MARKER_SPAWN_INDEX is out of range.")

        vehicle = spawn_ego_vehicle(
            world, bp_lib=blueprint_library, spawn_tf=ego_transform
        )
        route_points = _build_route_points(
            world,
            ego_transform.location,
            spawn_points[END_MARKER_SPAWN_INDEX].location,
        )
        _draw_route(world, route_points, life_time_s=sim_seconds + 5.0)
        spectator = SpectatorController(world, vehicle)

        speed_state = {
            "i_term": 0.0,
            "kp": 0.50,
            "ki": 0.10,
            "brake_prev": 0.0,
            "drive_mode": "CRUISE",
            "hazard_clear_ticks": 0,
            "throttle_prev": 0.0,
            "brake_prev_cruise": 0.0,
            "route_points_world": route_points,
            "noodle_enable": False,
        }

        total_ticks = int(sim_seconds / FIXED_DT)
        target_speed_mps = target_mph * 0.44704

        print("\nSCRIPTED LATERAL-OFFSET SMOKE TEST")
        print(f"target speed: {target_mph:.1f} mph")
        print(f"peak offset: {lateral_offset_m:+.2f} m (+right / -left)")
        print("cyan point: original route target; pink point: offset target")

        for tick in range(total_ticks):
            world.tick()
            sim_time_s = tick * FIXED_DT
            requested_offset_m = scripted_offset_m(sim_time_s, lateral_offset_m)

            telemetry = lane_follow_step(
                world,
                vehicle,
                lookahead_m=LOOKAHEAD_M,
                steer_gain=STEER_GAIN,
                target_speed_mps=target_speed_mps,
                fixed_dt=FIXED_DT,
                speed_state=speed_state,
                lidar_frame=None,
                FIXED_DT=FIXED_DT,
                base_distance_m=5.0,
                headway_seconds=2.5,
                panic_distance_m=5.0,
                ramp_up_per_s=4.0,
                ramp_down_per_s=1.0,
                lateral_offset_m=requested_offset_m,
            )
            if telemetry is None:
                raise RuntimeError("lane_follow_step returned no telemetry.")

            actual_offset_m = telemetry["signed_route_lateral_offset_m"]
            steer_cmd = telemetry["steer_cmd"]
            final_actual_offset_m = actual_offset_m
            max_abs_actual_offset_m = max(
                max_abs_actual_offset_m, abs(actual_offset_m)
            )
            max_abs_steer = max(max_abs_steer, abs(steer_cmd))

            if tick % int(0.5 / FIXED_DT) == 0:
                print(
                    f"t={sim_time_s:4.1f}s  "
                    f"speed={mps_to_mph(telemetry['speed_mps']):5.1f} mph  "
                    f"requested={requested_offset_m:+5.2f} m  "
                    f"actual={actual_offset_m:+5.2f} m  "
                    f"steer={steer_cmd:+5.2f}"
                )

            spectator.tick(FIXED_DT)

        print("\nTEST SUMMARY")
        print(f"maximum absolute lateral offset: {max_abs_actual_offset_m:.3f} m")
        print(f"maximum absolute steer command: {max_abs_steer:.3f}")
        print(f"final signed lateral offset: {final_actual_offset_m:+.3f} m")
        if math.isfinite(final_actual_offset_m):
            print(
                "route recovery check: "
                + ("PASS" if abs(final_actual_offset_m) <= 0.25 else "REVIEW")
            )

    finally:
        if spectator is not None:
            spectator.close()
        if vehicle is not None:
            vehicle.destroy()
        if world is not None:
            restore_async_mode(world)
        print("Clean exit: destroyed ego and restored asynchronous mode.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-mph", type=float, default=15.0)
    parser.add_argument("--lateral-offset-m", type=float, default=0.75)
    parser.add_argument("--sim-seconds", type=float, default=13.0)
    args = parser.parse_args()

    if abs(args.lateral_offset_m) > 1.5:
        parser.error("Keep this preliminary test within +/-1.5 m.")
    if args.target_mph <= 0.0 or args.target_mph > 25.0:
        parser.error("Keep this preliminary test in the range (0, 25] mph.")
    if args.sim_seconds < 11.0:
        parser.error("sim-seconds must be at least 11 to include route recovery.")

    run_test(
        target_mph=args.target_mph,
        lateral_offset_m=args.lateral_offset_m,
        sim_seconds=args.sim_seconds,
    )


if __name__ == "__main__":
    main()
