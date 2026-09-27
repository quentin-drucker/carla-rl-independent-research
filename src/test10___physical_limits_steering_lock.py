"""test10___physical_limits_steering_lock.py

Week 3, Workstream 2.2: the steering-lock and stability substudy of the
connected vehicle physical-limits suite, built on the shared harness
(physics_harness.py) and trace schema (trace_schema.py) validated by the
braking substudy (test9).

Matrix (2.6): 3 entry speeds x left/right x step/ramp = 12 runs.

Procedure per run, all via direct low-level VehicleControl (no route
following, no hazard logic, no SAC):
    1. Accelerate to a matched ACTUAL entry speed under full throttle.
    2. Release throttle to 0.0 (coast) -- isolates the steering response
       from any simultaneous accel/steer confound. A throttle-held variant
       is explicit future work, not attempted here.
    3. Apply the steering command (step = jumps directly to full lock;
       ramp = linearly ramps to full lock over RAMP_DURATION_S) and hold
       full lock for the remainder of MANEUVER_DURATION_S.
    4. Brake to a stop and destroy the actor.

CARLA's own Tesla Model 3 physics control has a speed-dependent
steering_curve (confirmed via get_physics_control() on this build: 100% of
max_steer_angle at 0 km/h, 90% at 20, 80% at 60, 70% at 120) -- logged into
every run's manifest so any speed-dependent achieved wheel angle is
correctly attributed to that documented curve rather than misread as
evidence of something else.

Evidence-status discipline (per the Week 3 plan): this script reports only
measured quantities -- achieved front-wheel steer angle, yaw rate, turn
radius, lateral acceleration, speed loss, path deviation, and whole-body
slip angle (velocity heading vs. yaw -- a standard, purely
kinematic/measurable quantity, NOT a wheel-slip/tire-skid signal). It never
labels a run "understeer", "oversteer", or "skid" -- CARLA 0.9.16's Python
API exposes no validated per-wheel slip/lock signal to support those
claims (wheel_slip_signal="not_measurable" throughout, same as test9).

Usage (CARLA must already be running):
    python -X utf8 test10___physical_limits_steering_lock.py

Writes one timestamped, non-overwritable run directory per case under
runs/physics/<timestamp>_steering_lock/.
"""

import datetime
import itertools
import os

import carla

from carla_session import connect_and_load_world, enable_sync_mode, restore_async_mode
from math_utils import get_speed_mps
from spectator import update_spectator_follow
from physics_harness import (
    MPH_TO_MPS,
    accelerate_to_matched_entry_speed,
    capture_tick_from_actor,
    compute_lateral_accel_from_yaw_rate,
    compute_turn_radius_m,
    detect_sustained_near_stop_onset_s,
    inner_wheel_steer_angle_deg,
    read_physics_settings,
)
from trace_schema import PhysicsRunManifest, write_trace_csv

HOST = "localhost"
PORT = 2000
FIXED_DT = 0.02
SPAWN_INDEX = 242
TARGET_MAP = "Town04_Opt"
MAX_ACCEL_TICKS = 2000
STOPPED_MPS = 0.15
MAX_STOP_TICKS = 1000

TARGET_SPEEDS_MPH = [15.0, 30.0, 45.0]
DIRECTIONS = {"left": -1.0, "right": +1.0}
COMMAND_TYPES = ["step", "ramp"]
RAMP_DURATION_S = 1.0
MANEUVER_DURATION_S = 5.0
# Average over the final window of the maneuver to report a steady-state
# achieved wheel angle/turn radius, rather than a single noisy last tick.
STEADY_STATE_WINDOW_S = 0.5

RUN_FAMILY = "steering_lock"


def _steer_command_for_tick(*, command_type: str, elapsed_s: float, direction_sign: float) -> float:
    if command_type == "step":
        return direction_sign * 1.0
    if command_type == "ramp":
        fraction = min(elapsed_s / RAMP_DURATION_S, 1.0)
        return direction_sign * fraction
    raise ValueError(f"unknown command_type: {command_type}")


def _run_one_case(
    *, world, bp_lib, spawn_tf, target_mph, direction_name, direction_sign, command_type, run_dir,
    spectator=None,
):
    target_mps = target_mph * MPH_TO_MPS
    case_id = f"{target_mph:.0f}mph_{direction_name}_{command_type}"

    bp = bp_lib.find("vehicle.tesla.model3")
    vehicle = world.spawn_actor(bp, spawn_tf)
    world.tick()

    steering_curve = [(p.x, p.y) for p in vehicle.get_physics_control().steering_curve]

    actual_entry_speed_mps = accelerate_to_matched_entry_speed(
        world=world, vehicle=vehicle, target_mps=target_mps, fixed_dt=FIXED_DT, max_ticks=MAX_ACCEL_TICKS
    )

    start_transform = vehicle.get_transform()
    start_pose = (start_transform.location.x, start_transform.location.y, start_transform.rotation.yaw)

    ticks = []
    prev_speed_mps = None
    prev_yaw_deg = None
    sim_time_s = 0.0
    maneuver_ticks = int(MANEUVER_DURATION_S / FIXED_DT)
    for tick_index in range(maneuver_ticks):
        elapsed_s = tick_index * FIXED_DT
        steer = _steer_command_for_tick(command_type=command_type, elapsed_s=elapsed_s, direction_sign=direction_sign)
        control = carla.VehicleControl(throttle=0.0, brake=0.0, steer=steer)
        vehicle.apply_control(control)
        world.tick()
        update_spectator_follow(spectator, vehicle)
        sim_time_s += FIXED_DT
        tick = capture_tick_from_actor(
            vehicle=vehicle,
            tick_index=tick_index,
            sim_time_s=sim_time_s,
            requested_control=control,
            prev_speed_mps=prev_speed_mps,
            prev_yaw_deg=prev_yaw_deg,
            dt_s=FIXED_DT,
            start_pose=start_pose,
        )
        ticks.append(tick)
        prev_speed_mps = tick.speed_mps
        prev_yaw_deg = tick.yaw_deg

    # Bring the vehicle to a stop before destroying it (housekeeping, not
    # part of the measured maneuver).
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
    for _ in range(MAX_STOP_TICKS):
        world.tick()
        update_spectator_follow(spectator, vehicle)
        if get_speed_mps(vehicle) < STOPPED_MPS:
            break

    vehicle.destroy()
    world.tick()

    # Steady-state summary over the final STEADY_STATE_WINDOW_S of the
    # maneuver -- averages out single-tick noise rather than reading one
    # last sample.
    steady_state_ticks = [t for t in ticks if t.sim_time_s >= MANEUVER_DURATION_S - STEADY_STATE_WINDOW_S]
    # Use the geometrically inner front wheel (larger magnitude of FL/FR),
    # not FL unconditionally -- Ackermann steering means FL is the outer
    # (smaller-angle) wheel in a right turn and the inner (larger-angle)
    # wheel in a left turn, so FL alone manufactures a false left/right
    # asymmetry. See physics_harness.inner_wheel_steer_angle_deg.
    wheel_angles = [
        inner_wheel_steer_angle_deg(front_left_deg=t.front_wheel_steer_angle_deg, front_right_deg=t.front_right_wheel_steer_angle_deg)
        for t in steady_state_ticks
    ]
    wheel_angles = [a for a in wheel_angles if a is not None]
    yaw_rates = [t.yaw_rate_dps for t in steady_state_ticks if t.yaw_rate_dps is not None]
    slip_angles = [t.body_slip_angle_deg for t in ticks if t.body_slip_angle_deg is not None]
    speeds = [t.speed_mps for t in ticks]

    achieved_wheel_angle_deg = (sum(wheel_angles) / len(wheel_angles)) if wheel_angles else None
    steady_yaw_rate_dps = (sum(yaw_rates) / len(yaw_rates)) if yaw_rates else None
    steady_speed_mps = (sum(t.speed_mps for t in steady_state_ticks) / len(steady_state_ticks)) if steady_state_ticks else None

    turn_radius_m = (
        compute_turn_radius_m(speed_mps=steady_speed_mps, yaw_rate_dps=steady_yaw_rate_dps)
        if steady_speed_mps is not None else None
    )
    lateral_accel_mps2 = (
        compute_lateral_accel_from_yaw_rate(speed_mps=steady_speed_mps, yaw_rate_dps=steady_yaw_rate_dps)
        if steady_speed_mps is not None else None
    )
    max_abs_slip_angle_deg = max((abs(a) for a in slip_angles), default=None)
    max_abs_lateral_displacement_m = max((abs(t.lateral_displacement_m or 0.0) for t in ticks), default=0.0)
    speed_loss_mps = actual_entry_speed_mps - speeds[-1] if speeds else None
    # Detects a maneuver that scrubbed off essentially all speed (via
    # full-lock cornering drag alone, no throttle) well before the nominal
    # MANEUVER_DURATION_S ended -- found live 2026-09-26 at higher entry
    # speeds. When this fires, the "steady state" window above is measuring
    # a near-stationary vehicle, not a sustained turn, and turn_radius_m /
    # lateral_accel_mps2 correctly read as "not turning" (near-zero yaw
    # rate) rather than as a loss-of-control event.
    near_stop_onset_s = detect_sustained_near_stop_onset_s(
        speed_time_pairs=[(t.speed_mps, t.sim_time_s) for t in ticks], threshold_mps=1.0
    )

    manifest = PhysicsRunManifest(
        run_id=case_id,
        test_family=RUN_FAMILY,
        created_at_iso=datetime.datetime.now().isoformat(),
        carla_version=carla.__file__,
        map_name=world.get_map().name,
        vehicle_blueprint="vehicle.tesla.model3",
        weather_preset="ClearNoon",
        tire_friction=None,
        random_seed=None,
        parameters={
            "target_mph": target_mph,
            "direction": direction_name,
            "command_type": command_type,
            "actual_entry_speed_mps": actual_entry_speed_mps,
            "actual_entry_speed_mph": actual_entry_speed_mps / MPH_TO_MPS,
            "maneuver_duration_s": MANEUVER_DURATION_S,
            "ramp_duration_s": RAMP_DURATION_S if command_type == "ramp" else None,
            "steering_curve_speed_kmh_to_max_fraction": steering_curve,
        },
        **read_physics_settings(world),
    )

    case_dir = os.path.join(run_dir, case_id)
    manifest.to_json(os.path.join(case_dir, "manifest.json"))
    write_trace_csv(os.path.join(case_dir, "trace.csv"), ticks)

    return {
        "case_id": case_id,
        "target_mph": target_mph,
        "direction": direction_name,
        "command_type": command_type,
        "actual_entry_speed_mph": actual_entry_speed_mps / MPH_TO_MPS,
        "achieved_wheel_angle_deg": achieved_wheel_angle_deg,
        "steady_yaw_rate_dps": steady_yaw_rate_dps,
        "turn_radius_m": turn_radius_m,
        "lateral_accel_mps2": lateral_accel_mps2,
        "max_abs_slip_angle_deg": max_abs_slip_angle_deg,
        "max_abs_lateral_displacement_m": max_abs_lateral_displacement_m,
        "speed_loss_mps": speed_loss_mps,
        "near_stop_onset_s": near_stop_onset_s,
        "wheel_slip_signal": "not_measurable",
    }


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", "physics", f"{timestamp}_{RUN_FAMILY}")

    client, world = connect_and_load_world(host=HOST, port=PORT, timeout_s=10.0, target_map=TARGET_MAP)
    enable_sync_mode(world, fixed_dt=FIXED_DT)

    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    spawn_tf = spawn_points[SPAWN_INDEX]
    bp_lib = world.get_blueprint_library()
    spectator = world.get_spectator()

    results = []
    try:
        for target_mph, (direction_name, direction_sign), command_type in itertools.product(
            TARGET_SPEEDS_MPH, DIRECTIONS.items(), COMMAND_TYPES
        ):
            print(f"[test10] running {target_mph:.0f}mph {direction_name} {command_type} ...", end=" ", flush=True)
            row = _run_one_case(
                world=world, bp_lib=bp_lib, spawn_tf=spawn_tf,
                target_mph=target_mph, direction_name=direction_name, direction_sign=direction_sign,
                command_type=command_type, run_dir=run_dir, spectator=spectator,
            )
            results.append(row)
            near_stop_str = f"{row['near_stop_onset_s']:.2f}s" if row['near_stop_onset_s'] is not None else "no"
            print(
                f"entry={row['actual_entry_speed_mph']:.1f}mph wheel_angle={row['achieved_wheel_angle_deg']:.1f}deg "
                f"yaw_rate={row['steady_yaw_rate_dps']:.1f}dps turn_radius={row['turn_radius_m']}m "
                f"lat_accel={row['lateral_accel_mps2']:.2f}m/s^2 max_slip_angle={row['max_abs_slip_angle_deg']}deg "
                f"near_stop_onset={near_stop_str}"
            )
    finally:
        restore_async_mode(world)

    print("\n" + "=" * 145)
    print(
        f"{'case':26} {'entry_mph':10} {'wheel_deg':10} {'yaw_dps':9} {'radius_m':9} "
        f"{'lat_accel':10} {'max_slip':9} {'max_lat_disp':12} {'speed_loss':10} {'near_stop':9}"
    )
    print("-" * 145)
    for row in results:
        near_stop_str = f"{row['near_stop_onset_s']:.2f}" if row['near_stop_onset_s'] is not None else "no"
        print(
            f"{row['case_id']:26} {row['actual_entry_speed_mph']:10.1f} "
            f"{row['achieved_wheel_angle_deg']:10.2f} {row['steady_yaw_rate_dps']:9.2f} "
            f"{(row['turn_radius_m'] if row['turn_radius_m'] is not None else float('nan')):9.2f} "
            f"{row['lateral_accel_mps2']:10.2f} "
            f"{(row['max_abs_slip_angle_deg'] if row['max_abs_slip_angle_deg'] is not None else float('nan')):9.2f} "
            f"{row['max_abs_lateral_displacement_m']:12.2f} {row['speed_loss_mps']:10.2f} {near_stop_str:9}"
        )
    print("=" * 145)
    print(
        "\nStudy limitation (stated, not re-derived): no validated per-wheel slip/lock signal is exposed by "
        "CARLA 0.9.16's Python API -- wheel_slip_signal=not_measurable for every case above. max_slip (body "
        "sideslip angle, velocity heading vs. yaw) is a measurable whole-body kinematic quantity, NOT a "
        "wheel-slip/tire-skid signal -- do not read any of these numbers as evidence of understeer, oversteer, "
        "or wheel lock. CARLA's own steering_curve (logged per-manifest) reduces max wheel angle at higher "
        "speed by design -- a lower achieved wheel angle at high entry speed reflects that curve, not a novel "
        "physical limit. near_stop_onset marks cases where the vehicle scrubbed off essentially all speed via "
        "full-lock cornering drag alone (no throttle) well before the 5s maneuver ended -- for those cases, "
        "turn_radius/lat_accel/max_slip reflect a near-stationary vehicle, not a sustained turn."
    )
    print(f"\nRun evidence written under: {run_dir}")


if __name__ == "__main__":
    main()
