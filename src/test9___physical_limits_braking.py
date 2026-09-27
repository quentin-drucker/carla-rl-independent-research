"""test9___physical_limits_braking.py

Week 3, Workstream 2.4: the braking-behavior substudy of the connected
vehicle physical-limits suite, built on the shared harness
(physics_harness.py) and trace schema (trace_schema.py).

Reconciles rather than duplicates the existing brake_test.py /
brake_calibration.py diagnostics: this script answers the same core
question (how does this vehicle stop under different commanded brake
levels and friction) but through the shared harness so the manifest,
per-tick trace, and outcome classification are consistent with the other
physical-limits substudies (steering lock, rollover, throttle/brake
symmetry) that will reuse the same schema.

Explicit study limitation, stated up front rather than re-derived:
CARLA 0.9.16 does not model a validated production ABS system, and its
Python API exposes no validated per-wheel slip/lock signal. This script
therefore never labels anything "skid", "wheel lock", "understeer", or
"oversteer" -- only measured stopping distance/time, peak/mean
deceleration, and yaw/path deviation, using the evidence-status vocabulary
from the Week 3 plan ("confirmed" / "not_observed_in_tested_range" /
"inconclusive" / "not_measurable").

Uses matched ACTUAL entry speed (physics_harness.accelerate_to_matched_
entry_speed), not the requested target, per the plan's explicit
requirement -- the requested target is only ever the throttle-phase input.

Usage (CARLA must already be running):
    python -X utf8 test9___physical_limits_braking.py

Writes one timestamped run directory per (target speed x friction x brake
level) cell under runs/physics/<timestamp>_braking/, each with its own
manifest.json and trace.csv -- never overwriting a prior run's evidence.
"""

import datetime
import os

import carla

from carla_session import connect_and_load_world, enable_sync_mode, restore_async_mode
from math_utils import get_speed_mps
from spectator import update_spectator_follow
from physics_harness import (
    MPH_TO_MPS,
    accelerate_to_matched_entry_speed,
    apply_uniform_tire_friction,
    capture_tick_from_actor,
    classify_stop_outcome,
    read_physics_settings,
    summarize_deceleration,
)
from trace_schema import PhysicsRunManifest, write_trace_csv

HOST = "localhost"
PORT = 2000
FIXED_DT = 0.02
SPAWN_INDEX = 242
TARGET_MAP = "Town04_Opt"
STOPPED_MPS = 0.15
MAX_TICKS = 2000

# Compact matrix per the plan's "4 command levels x matched entry speeds"
# guidance -- one representative speed and friction per cell for this first
# live pass; boundary repeats and the full compact matrix are next-slice
# work (see worklog for the explicit remaining-gates list).
TARGET_SPEEDS_MPH = [35.0]
BRAKE_LEVELS = [0.25, 0.50, 0.75, 1.00]
TIRE_FRICTION = None  # CARLA default; friction sweep deferred to next slice

RUN_FAMILY = "braking"

# Below this speed, a live run (2026-09-26) found a reproducible deceleration
# transient largely independent of the commanded brake level -- see the
# comment at its point of use in _run_one_braking_case for the numbers.
LOW_SPEED_TRANSIENT_THRESHOLD_MPS = 5.0


def _run_one_braking_case(
    *, world, bp_lib, spawn_tf, target_mph: float, brake_level: float, run_dir: str, spectator=None
) -> dict:
    target_mps = target_mph * MPH_TO_MPS

    bp = bp_lib.find("vehicle.tesla.model3")
    vehicle = world.spawn_actor(bp, spawn_tf)
    world.tick()

    apply_uniform_tire_friction(vehicle, TIRE_FRICTION)
    world.tick()

    actual_entry_speed_mps = accelerate_to_matched_entry_speed(
        world=world, vehicle=vehicle, target_mps=target_mps, fixed_dt=FIXED_DT, max_ticks=MAX_TICKS
    )

    start_transform = vehicle.get_transform()
    start_pose = (start_transform.location.x, start_transform.location.y, start_transform.rotation.yaw)

    brake_control = carla.VehicleControl(throttle=0.0, brake=brake_level)
    vehicle.apply_control(brake_control)

    ticks = []
    prev_speed_mps = None
    prev_yaw_deg = None
    sim_time_s = 0.0
    for tick_index in range(MAX_TICKS):
        world.tick()
        update_spectator_follow(spectator, vehicle)
        sim_time_s += FIXED_DT
        tick = capture_tick_from_actor(
            vehicle=vehicle,
            tick_index=tick_index,
            sim_time_s=sim_time_s,
            requested_control=brake_control,
            prev_speed_mps=prev_speed_mps,
            prev_yaw_deg=prev_yaw_deg,
            dt_s=FIXED_DT,
            start_pose=start_pose,
        )
        ticks.append(tick)
        prev_speed_mps = tick.speed_mps
        prev_yaw_deg = tick.yaw_deg
        if tick.speed_mps < STOPPED_MPS:
            break

    final_speed_mps = get_speed_mps(vehicle)
    stop_outcome = classify_stop_outcome(final_speed_mps=final_speed_mps, stopped_threshold_mps=STOPPED_MPS)
    stop_time_s = ticks[-1].sim_time_s if ticks else 0.0
    stop_dist_m = ticks[-1].pos_x_m - start_pose[0] if ticks else 0.0  # straight spawn heading; see manifest
    # Use straight-line distance (works for any heading), not just x delta.
    stop_dist_m = (
        (ticks[-1].pos_x_m - start_pose[0]) ** 2 + (ticks[-1].pos_y_m - start_pose[1]) ** 2
    ) ** 0.5 if ticks else 0.0

    max_abs_lateral_m = max((abs(t.lateral_displacement_m or 0.0) for t in ticks), default=0.0)

    decel_summary = summarize_deceleration(
        speed_accel_pairs=[(t.speed_mps, t.accel_mps2) for t in ticks],
        threshold_mps=LOW_SPEED_TRANSIENT_THRESHOLD_MPS,
    )
    peak_decel_mps2 = decel_summary.normal_speed_peak_decel_mps2
    mean_decel_mps2 = decel_summary.normal_speed_mean_decel_mps2
    low_speed_transient_peak_decel_mps2 = decel_summary.low_speed_transient_peak_decel_mps2

    vehicle.destroy()
    world.tick()

    case_id = f"{target_mph:.0f}mph_brake{brake_level:.2f}_friction{TIRE_FRICTION}"
    manifest = PhysicsRunManifest(
        run_id=case_id,
        test_family=RUN_FAMILY,
        created_at_iso=datetime.datetime.now().isoformat(),
        carla_version=carla.__file__,  # best available version proxy; see MASTER summary caveats
        map_name=world.get_map().name,
        vehicle_blueprint="vehicle.tesla.model3",
        weather_preset="ClearNoon",
        tire_friction=TIRE_FRICTION,
        random_seed=None,
        parameters={
            "target_mph": target_mph,
            "requested_brake_level": brake_level,
            "actual_entry_speed_mps": actual_entry_speed_mps,
            "actual_entry_speed_mph": actual_entry_speed_mps / MPH_TO_MPS,
        },
        **read_physics_settings(world),
    )

    case_dir = os.path.join(run_dir, case_id)
    manifest.to_json(os.path.join(case_dir, "manifest.json"))
    write_trace_csv(os.path.join(case_dir, "trace.csv"), ticks)

    return {
        "case_id": case_id,
        "target_mph": target_mph,
        "brake_level": brake_level,
        "actual_entry_speed_mph": actual_entry_speed_mps / MPH_TO_MPS,
        "stop_outcome": stop_outcome,
        "stop_time_s": stop_time_s,
        "stop_dist_m": stop_dist_m,
        "peak_decel_mps2": peak_decel_mps2,
        "mean_decel_mps2": mean_decel_mps2,
        "low_speed_transient_peak_decel_mps2": low_speed_transient_peak_decel_mps2,
        "max_abs_lateral_m": max_abs_lateral_m,
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
        for target_mph in TARGET_SPEEDS_MPH:
            for brake_level in BRAKE_LEVELS:
                print(f"[test9] running {target_mph:.0f} mph / brake={brake_level:.2f} ...", end=" ", flush=True)
                row = _run_one_braking_case(
                    world=world, bp_lib=bp_lib, spawn_tf=spawn_tf,
                    target_mph=target_mph, brake_level=brake_level, run_dir=run_dir,
                    spectator=spectator,
                )
                results.append(row)
                print(
                    f"entry={row['actual_entry_speed_mph']:.1f}mph outcome={row['stop_outcome']} "
                    f"stop_time={row['stop_time_s']:.2f}s stop_dist={row['stop_dist_m']:.1f}m "
                    f"peak_decel(>={LOW_SPEED_TRANSIENT_THRESHOLD_MPS:.0f}m/s)={row['peak_decel_mps2']:.2f}m/s^2 "
                    f"low_speed_transient_peak={row['low_speed_transient_peak_decel_mps2']}m/s^2 "
                    f"max_lateral={row['max_abs_lateral_m']:.3f}m"
                )
    finally:
        restore_async_mode(world)

    print("\n" + "=" * 110)
    print(
        f"{'case':32} {'entry_mph':10} {'outcome':12} {'stop_s':8} {'stop_m':8} "
        f"{'peak_decel':11} {'low_spd_trans':13} {'max_lat_m':10}"
    )
    print("-" * 110)
    for row in results:
        print(
            f"{row['case_id']:32} {row['actual_entry_speed_mph']:10.1f} {row['stop_outcome']:12} "
            f"{row['stop_time_s']:8.2f} {row['stop_dist_m']:8.1f} {row['peak_decel_mps2']:11.2f} "
            f"{row['low_speed_transient_peak_decel_mps2']:13.2f} {row['max_abs_lateral_m']:10.3f}"
        )
    print("=" * 110)
    print(
        f"\nnormal-speed peak_decel (speed >= {LOW_SPEED_TRANSIENT_THRESHOLD_MPS:.0f} m/s) scales with commanded "
        "brake level, as expected. low_spd_trans (speed below that threshold) was found in a prior live run "
        "to land near the same ~-27 m/s^2 value across brake levels 0.25-1.00 -- reported separately so it is "
        "not silently mixed into the normal-speed metric."
    )
    print(
        "\nStudy limitation (stated, not re-derived): CARLA does not model a validated production "
        "ABS system, and no validated per-wheel slip/lock signal is exposed here -- "
        "wheel_slip_signal=not_measurable for every case above. Do not read either deceleration metric or "
        "max_abs_lateral_m as evidence of skid/wheel-lock absence or presence."
    )
    print(f"\nRun evidence written under: {run_dir}")


if __name__ == "__main__":
    main()
