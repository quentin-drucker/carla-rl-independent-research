"""test12___physical_limits_throttle_brake_symmetry.py

Week 3, Workstream 2.5: the throttle/brake symmetry substudy of the
connected vehicle physical-limits suite -- the last of the four
substudies. Answers the advisor's Sep 22 question directly: does an equal
[0,1] throttle and brake command produce equal physical magnitude?

Design:
    - 4 matched command magnitudes: 0.25, 0.50, 0.75, 1.00.
    - Throttle phase: from a standing start (0 mph -- a defined initial
      condition), apply constant throttle=X for THROTTLE_DURATION_S,
      recording acceleration response.
    - Brake phase: from a matched actual entry speed of DECEL_TARGET_MPH
      (the same speed test9's braking substudy used, so this substudy's
      brake-side numbers are directly comparable to that already-published
      finding), apply constant brake=X, recording deceleration response.
    - Both phases reuse physics_harness.summarize_acceleration/
      summarize_deceleration, which split results into a "normal-speed"
      and "low-speed" (<5 m/s) regime -- a standing start begins INSIDE
      that low-speed artifact zone (already found to produce a spurious
      deceleration transient in test9 and a spurious velocity-direction
      discontinuity in test10), so the throttle side is checked for a
      similar artifact before trusting any acceleration number from it.
    - physics_harness.classify_throttle_brake_symmetry() compares the two
      normal-speed peak magnitudes at each matched command level.
    - physics_harness.compute_rise_time_s() gives a coarse "how fast does
      the response build up" descriptive metric for each phase/level.

Explicitly not claiming: which direction (throttle vs. brake) is
"better" -- only whether equal numeric commands produce comparable
physical magnitudes, per the plan's RL-design framing.

Usage (CARLA must already be running):
    python -X utf8 test12___physical_limits_throttle_brake_symmetry.py

Writes one timestamped, non-overwritable run directory per case under
runs/physics/<timestamp>_throttle_brake_symmetry/.
"""

import datetime
import os

import carla

from carla_session import connect_and_load_world, enable_sync_mode, restore_async_mode
from math_utils import get_speed_mps
from physics_harness import (
    LOW_SPEED_ARTIFACT_THRESHOLD_MPS,
    MPH_TO_MPS,
    accelerate_to_matched_entry_speed,
    apply_uniform_tire_friction,
    capture_tick_from_actor,
    classify_throttle_brake_symmetry,
    compute_rise_time_s,
    read_physics_settings,
    summarize_acceleration,
    summarize_deceleration,
)
from trace_schema import PhysicsRunManifest, write_trace_csv

HOST = "localhost"
PORT = 2000
FIXED_DT = 0.02
SPAWN_INDEX = 242
TARGET_MAP = "Town04_Opt"
STOPPED_MPS = 0.15
MAX_ACCEL_TICKS = 2000
MAX_STOP_TICKS = 2000

COMMAND_LEVELS = [0.25, 0.50, 0.75, 1.00]
THROTTLE_DURATION_S = 5.0
DECEL_TARGET_MPH = 35.0  # matches test9's braking substudy for direct comparability

RUN_FAMILY = "throttle_brake_symmetry"


def _run_throttle_case(*, world, bp_lib, spawn_tf, command_level, run_dir):
    case_id = f"throttle_{command_level:.2f}"

    bp = bp_lib.find("vehicle.tesla.model3")
    vehicle = world.spawn_actor(bp, spawn_tf)
    world.tick()
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
    for _ in range(50):  # settle fully at rest before starting the timed phase
        world.tick()

    start_transform = vehicle.get_transform()
    start_pose = (start_transform.location.x, start_transform.location.y, start_transform.rotation.yaw)

    control = carla.VehicleControl(throttle=command_level, brake=0.0)
    ticks = []
    prev_speed_mps = None
    prev_yaw_deg = None
    sim_time_s = 0.0
    duration_ticks = int(THROTTLE_DURATION_S / FIXED_DT)
    for tick_index in range(duration_ticks):
        vehicle.apply_control(control)
        world.tick()
        sim_time_s += FIXED_DT
        tick = capture_tick_from_actor(
            vehicle=vehicle, tick_index=tick_index, sim_time_s=sim_time_s, requested_control=control,
            prev_speed_mps=prev_speed_mps, prev_yaw_deg=prev_yaw_deg, dt_s=FIXED_DT, start_pose=start_pose,
        )
        ticks.append(tick)
        prev_speed_mps = tick.speed_mps
        prev_yaw_deg = tick.yaw_deg

    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
    for _ in range(MAX_STOP_TICKS):
        world.tick()
        if get_speed_mps(vehicle) < STOPPED_MPS:
            break
    vehicle.destroy()
    world.tick()

    accel_summary = summarize_acceleration(speed_accel_pairs=[(t.speed_mps, t.accel_mps2) for t in ticks])
    # Time to 90% of the NORMAL-SPEED peak acceleration (truncated at that
    # peak tick) -- restricted to speed >= the low-speed-artifact threshold
    # so this matches the same regime as normal_speed_peak_accel_mps2
    # above. Without this restriction, the launch-transient spike found
    # live (2026-09-26, ~18-20 m/s^2 in the first ~0.5s from rest, LARGER
    # than the normal-speed peak) would silently become "the peak" the
    # rise time is measured against, making it inconsistent with the peak
    # value reported in the same row.
    normal_speed_accel_series = [
        (t.sim_time_s, t.accel_mps2) for t in ticks
        if t.accel_mps2 is not None and t.speed_mps >= LOW_SPEED_ARTIFACT_THRESHOLD_MPS
    ]
    if normal_speed_accel_series:
        peak_index = max(range(len(normal_speed_accel_series)), key=lambda i: normal_speed_accel_series[i][1])
        rise_time_s = compute_rise_time_s(
            time_speed_value_pairs=normal_speed_accel_series[: peak_index + 1], target_fraction=0.9
        )
    else:
        rise_time_s = None

    manifest = PhysicsRunManifest(
        run_id=case_id, test_family=RUN_FAMILY, created_at_iso=datetime.datetime.now().isoformat(),
        carla_version=carla.__file__, map_name=world.get_map().name, vehicle_blueprint="vehicle.tesla.model3",
        weather_preset="ClearNoon", tire_friction=None, random_seed=None,
        parameters={
            "phase": "throttle", "command_level": command_level, "duration_s": THROTTLE_DURATION_S,
            "final_speed_mps": ticks[-1].speed_mps if ticks else None,
        },
        **read_physics_settings(world),
    )
    case_dir = os.path.join(run_dir, case_id)
    manifest.to_json(os.path.join(case_dir, "manifest.json"))
    write_trace_csv(os.path.join(case_dir, "trace.csv"), ticks)

    return {
        "phase": "throttle", "command_level": command_level,
        "peak_response_mps2": accel_summary.normal_speed_peak_accel_mps2,
        "mean_response_mps2": accel_summary.normal_speed_mean_accel_mps2,
        "low_speed_transient_mps2": accel_summary.low_speed_transient_peak_accel_mps2,
        "rise_time_s": rise_time_s,
        "final_speed_mph": (ticks[-1].speed_mps / MPH_TO_MPS) if ticks else None,
    }


def _run_brake_case(*, world, bp_lib, spawn_tf, command_level, run_dir):
    case_id = f"brake_{command_level:.2f}"
    target_mps = DECEL_TARGET_MPH * MPH_TO_MPS

    bp = bp_lib.find("vehicle.tesla.model3")
    vehicle = world.spawn_actor(bp, spawn_tf)
    world.tick()
    apply_uniform_tire_friction(vehicle, None)
    actual_entry_speed_mps = accelerate_to_matched_entry_speed(
        world=world, vehicle=vehicle, target_mps=target_mps, fixed_dt=FIXED_DT, max_ticks=MAX_ACCEL_TICKS
    )

    start_transform = vehicle.get_transform()
    start_pose = (start_transform.location.x, start_transform.location.y, start_transform.rotation.yaw)

    control = carla.VehicleControl(throttle=0.0, brake=command_level)
    ticks = []
    prev_speed_mps = None
    prev_yaw_deg = None
    sim_time_s = 0.0
    for tick_index in range(MAX_STOP_TICKS):
        vehicle.apply_control(control)
        world.tick()
        sim_time_s += FIXED_DT
        tick = capture_tick_from_actor(
            vehicle=vehicle, tick_index=tick_index, sim_time_s=sim_time_s, requested_control=control,
            prev_speed_mps=prev_speed_mps, prev_yaw_deg=prev_yaw_deg, dt_s=FIXED_DT, start_pose=start_pose,
        )
        ticks.append(tick)
        prev_speed_mps = tick.speed_mps
        prev_yaw_deg = tick.yaw_deg
        if tick.speed_mps < STOPPED_MPS:
            break
    vehicle.destroy()
    world.tick()

    decel_summary = summarize_deceleration(speed_accel_pairs=[(t.speed_mps, t.accel_mps2) for t in ticks])
    # "Time to 90% of FINAL speed" is meaningless for a braking run -- final
    # speed is ~0 by construction, so that target is trivially met on the
    # first tick. Instead measure time-to-90%-of-PEAK deceleration, using
    # only the NORMAL-SPEED regime (matching normal_speed_peak_decel_mps2
    # above) -- restricting to speed >= the low-speed-artifact threshold so
    # the low-speed transient (found live 2026-09-26 to land near the same
    # ~-27 m/s^2 regardless of commanded brake level) doesn't silently
    # become "the peak" this rise time is measured against.
    normal_speed_accel_series = [
        (t.sim_time_s, t.accel_mps2) for t in ticks
        if t.accel_mps2 is not None and t.speed_mps >= LOW_SPEED_ARTIFACT_THRESHOLD_MPS
    ]
    if normal_speed_accel_series:
        peak_index = min(range(len(normal_speed_accel_series)), key=lambda i: normal_speed_accel_series[i][1])
        rise_time_s = compute_rise_time_s(
            time_speed_value_pairs=normal_speed_accel_series[: peak_index + 1], target_fraction=0.9
        )
    else:
        rise_time_s = None

    manifest = PhysicsRunManifest(
        run_id=case_id, test_family=RUN_FAMILY, created_at_iso=datetime.datetime.now().isoformat(),
        carla_version=carla.__file__, map_name=world.get_map().name, vehicle_blueprint="vehicle.tesla.model3",
        weather_preset="ClearNoon", tire_friction=None, random_seed=None,
        parameters={
            "phase": "brake", "command_level": command_level,
            "actual_entry_speed_mps": actual_entry_speed_mps,
            "actual_entry_speed_mph": actual_entry_speed_mps / MPH_TO_MPS,
        },
        **read_physics_settings(world),
    )
    case_dir = os.path.join(run_dir, case_id)
    manifest.to_json(os.path.join(case_dir, "manifest.json"))
    write_trace_csv(os.path.join(case_dir, "trace.csv"), ticks)

    return {
        "phase": "brake", "command_level": command_level,
        "peak_response_mps2": decel_summary.normal_speed_peak_decel_mps2,
        "mean_response_mps2": decel_summary.normal_speed_mean_decel_mps2,
        "low_speed_transient_mps2": decel_summary.low_speed_transient_peak_decel_mps2,
        "rise_time_s": rise_time_s,
        "entry_speed_mph": actual_entry_speed_mps / MPH_TO_MPS,
    }


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", "physics", f"{timestamp}_{RUN_FAMILY}")

    client, world = connect_and_load_world(host=HOST, port=PORT, timeout_s=10.0, target_map=TARGET_MAP)
    enable_sync_mode(world, fixed_dt=FIXED_DT)

    spawn_tf = world.get_map().get_spawn_points()[SPAWN_INDEX]
    bp_lib = world.get_blueprint_library()

    throttle_results = {}
    brake_results = {}
    try:
        for level in COMMAND_LEVELS:
            print(f"[test12] running throttle={level:.2f} from rest ...", end=" ", flush=True)
            row = _run_throttle_case(world=world, bp_lib=bp_lib, spawn_tf=spawn_tf, command_level=level, run_dir=run_dir)
            throttle_results[level] = row
            print(
                f"peak={row['peak_response_mps2']} mean={row['mean_response_mps2']} "
                f"low_speed_transient={row['low_speed_transient_mps2']} rise_time={row['rise_time_s']} "
                f"final={row['final_speed_mph']:.1f}mph"
            )
        for level in COMMAND_LEVELS:
            print(f"[test12] running brake={level:.2f} from {DECEL_TARGET_MPH:.0f}mph ...", end=" ", flush=True)
            row = _run_brake_case(world=world, bp_lib=bp_lib, spawn_tf=spawn_tf, command_level=level, run_dir=run_dir)
            brake_results[level] = row
            print(
                f"peak={row['peak_response_mps2']:.2f} mean={row['mean_response_mps2']:.2f} "
                f"low_speed_transient={row['low_speed_transient_mps2']} rise_time={row['rise_time_s']} "
                f"entry={row['entry_speed_mph']:.1f}mph"
            )
    finally:
        restore_async_mode(world)

    print("\n" + "=" * 130)
    print(f"{'level':7} {'accel_peak':11} {'decel_peak':11} {'symmetry':30} {'accel_rise_s':13} {'decel_rise_s':13}")
    print("-" * 130)
    for level in COMMAND_LEVELS:
        t_row = throttle_results[level]
        b_row = brake_results[level]
        t_peak = t_row["peak_response_mps2"]
        b_peak = b_row["peak_response_mps2"]
        # A None peak means the vehicle never reached the normal-speed
        # (>=5 m/s) regime within the test duration -- "no data" is not the
        # same claim as "measured zero acceleration", and silently
        # coercing None to 0.0 would misreport an inconclusive case as a
        # real (a)symmetry finding.
        if t_peak is None or b_peak is None:
            symmetry = "inconclusive_insufficient_data"
        else:
            symmetry = classify_throttle_brake_symmetry(accel_response_mps2=t_peak, decel_response_mps2=b_peak)
        t_peak_str = f"{t_peak:.2f}" if t_peak is not None else "n/a"
        b_peak_str = f"{b_peak:.2f}" if b_peak is not None else "n/a"
        print(
            f"{level:7.2f} {t_peak_str:11} {b_peak_str:11} "
            f"{symmetry:30} {str(t_row['rise_time_s']):13} {str(b_row['rise_time_s']):13}"
        )
    print("=" * 130)
    if any(throttle_results[level]["peak_response_mps2"] is None for level in COMMAND_LEVELS):
        print(
            f"\nNote: at least one throttle level never reached the {5.0:.0f} m/s normal-speed threshold within "
            f"{THROTTLE_DURATION_S:.0f}s (e.g. throttle=0.25 reached only "
            f"{throttle_results[0.25]['final_speed_mph']:.1f} mph) -- its symmetry comparison is "
            "'inconclusive_insufficient_data', not a measured (a)symmetry result. A longer throttle duration "
            "would be needed to test that command level's normal-speed response."
        )
    print(
        "\nStudy limitation (stated, not re-derived): no validated per-wheel slip/lock signal is exposed by "
        "CARLA 0.9.16's Python API. Throttle response is measured from a standing start, which begins inside "
        "the same low-speed artifact zone (<5 m/s) already found to produce spurious readings in the braking "
        "(test9) and steering-lock (test10) substudies -- low_speed_transient_mps2 is reported separately per "
        "level and must be inspected before trusting it as real vehicle response rather than an artifact. This "
        "substudy makes no claim about which of throttle/brake is 'better' -- only whether equal [0,1] commands "
        "produce comparable physical magnitudes."
    )
    print(f"\nRun evidence written under: {run_dir}")


if __name__ == "__main__":
    main()
