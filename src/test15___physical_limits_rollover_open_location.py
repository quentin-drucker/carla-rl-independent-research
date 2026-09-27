"""test15___physical_limits_rollover_open_location.py

Week 3, Workstream 2.3 follow-up: repeats the rollover/flip boundary
substudy (test11) at an open, unobstructed location, per the explicit
remaining gate test11 itself recorded -- at the standard highway spawn
(SPAWN_INDEX=242, Town04_Opt), every case at 60mph+ ran off the paved
shoulder into a guardrail before intrinsic rollover dynamics could develop,
so the real question ("does this vehicle roll over at high speed") was
never actually answered above 45mph.

Location search (live, 2026-09-27): raycast-based openness scoring proved
an unreliable predictor by itself (a location that scored well on an
isotropic 35-90m radius scan still hit a guardrail within 0.6-1.6s once
actually driven) -- the only reliable check was live-driving the actual
maneuver. Systematically probed Town04_Opt (guardrail-lined on essentially
every road tried -- consistent with it being a mountain/highway loop map)
and Town05_Opt (dense city grid, similar clearance ceiling to Town04 in a
raycast scan) before finding a clean location in Town03_Opt:

  - spawn 90 (Town03_Opt): clean for RIGHT turns at 60/75/90 mph, 8s each,
    zero collisions, max roll 4.29 deg (at 90 mph).
  - spawn 92 (Town03_Opt, ~20m away, facing ~157 deg from spawn 90 --
    roughly the opposite direction): clean for LEFT turns at 60/75/90 mph,
    8s each, zero collisions, max roll under 1 deg.

Neither single spawn point is open in both directions -- this substudy
therefore uses TWO spawn points, one per steering direction, both in
Town03_Opt. This is a genuine methodology deviation from test11 (single
map/spawn) and is recorded explicitly in each case's manifest
(map_name/spawn_index), not silently normalized away.

Also uses physics_harness.set_instant_entry_velocity() instead of
accelerate_to_matched_entry_speed() -- found live that the throttle-ramp
method can cover 100+ meters reaching 90 mph, which would carry a coasting
maneuver's start point far past the specific spot verified open here.
set_instant_entry_velocity() settles to target speed within ~3 ticks with
negligible displacement (verified live and covered by an offline test
in tests/test_physics_harness.py). This is a legitimate substitution only
because this substudy is coasting-only (throttle=0 throughout the
maneuver) -- the acceleration phase itself is not part of what's measured
here, unlike test9/test12.

Usage (CARLA must already be running):
    python -X utf8 test15___physical_limits_rollover_open_location.py

Writes one timestamped, non-overwritable run directory per case under
runs/physics/<timestamp>_rollover_open_location/.
"""

import datetime
import os

import carla

from carla_session import connect_and_load_world, enable_sync_mode, restore_async_mode
from map_drivability import classify_point_drivability
from spectator import update_spectator_follow
from physics_harness import (
    MPH_TO_MPS,
    attach_collision_sensor,
    capture_tick_from_actor,
    classify_rollover,
    classify_upright_recovery,
    read_physics_settings,
    set_instant_entry_velocity,
)
from trace_schema import PhysicsRunManifest, write_trace_csv

HOST = "localhost"
PORT = 2000
FIXED_DT = 0.02
TARGET_MAP = "Town03_Opt"
STOPPED_MPS = 0.15

# Per-direction location: neither spawn 90 nor 92 is open in BOTH
# directions (see module docstring) -- each direction uses whichever
# spawn point was live-verified clean for it.
LOCATION_BY_DIRECTION = {
    "right": {"spawn_index": 90, "direction_sign": +1.0},
    "left": {"spawn_index": 92, "direction_sign": -1.0},
}

TARGET_SPEEDS_MPH = [60.0, 75.0, 90.0]
MANEUVER_DURATION_S = 8.0  # longer than test11's 5.0s -- live-verified clean for this long at this location
SETTLE_DURATION_S = 3.0
SETTLE_TICKS_AFTER_VELOCITY_SET = 15

EARLY_STOP_ROLL_DEG = 45.0
ROLLOVER_CONFIRM_DEG = 60.0
UPRIGHT_ROLL_DEG = 15.0

RUN_FAMILY = "rollover_open_location"


def _run_one_case(*, world, bp_lib, spawn_tf, target_mph, direction_name, direction_sign, run_dir, spectator=None):
    target_mps = target_mph * MPH_TO_MPS
    case_id = f"{target_mph:.0f}mph_{direction_name}"

    bp = bp_lib.find("vehicle.tesla.model3")
    vehicle = world.spawn_actor(bp, spawn_tf)
    world.tick()
    collision_sensor, collision_flag = attach_collision_sensor(world, bp_lib, vehicle)

    actual_entry_speed_mps = set_instant_entry_velocity(
        world=world, vehicle=vehicle, yaw_deg=spawn_tf.rotation.yaw, target_mps=target_mps,
        settle_ticks=SETTLE_TICKS_AFTER_VELOCITY_SET,
    )

    start_transform = vehicle.get_transform()
    start_pose = (start_transform.location.x, start_transform.location.y, start_transform.rotation.yaw)

    ticks = []
    prev_speed_mps = None
    prev_yaw_deg = None
    sim_time_s = 0.0
    maneuver_ticks = int(MANEUVER_DURATION_S / FIXED_DT)
    early_stopped = False
    stopped_due_to = "none"
    left_drivable_surface = False
    first_non_drivable_time_s = None

    for tick_index in range(maneuver_ticks):
        if early_stopped:
            control = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
        else:
            control = carla.VehicleControl(throttle=0.0, brake=0.0, steer=direction_sign * 1.0)
        vehicle.apply_control(control)
        world.tick()
        update_spectator_follow(spectator, vehicle)
        sim_time_s += FIXED_DT
        tick = capture_tick_from_actor(
            vehicle=vehicle, tick_index=tick_index, sim_time_s=sim_time_s, requested_control=control,
            prev_speed_mps=prev_speed_mps, prev_yaw_deg=prev_yaw_deg, dt_s=FIXED_DT, start_pose=start_pose,
        )
        ticks.append(tick)
        prev_speed_mps = tick.speed_mps
        prev_yaw_deg = tick.yaw_deg

        drivability_status, _ = classify_point_drivability(
            world.get_map(), carla.Location(x=tick.pos_x_m, y=tick.pos_y_m, z=tick.pos_z_m)
        )
        if drivability_status == "non_drivable" and not left_drivable_surface:
            left_drivable_surface = True
            first_non_drivable_time_s = sim_time_s

        if not early_stopped and collision_flag["hit"]:
            early_stopped = True
            stopped_due_to = "collision"
            print(
                f"    [!] early stop: collision with {collision_flag['other_actor']} at "
                f"t={sim_time_s:.2f}s -- releasing steer, braking (rollover reading confounded)"
            )
        elif not early_stopped and abs(tick.roll_deg) >= EARLY_STOP_ROLL_DEG:
            early_stopped = True
            stopped_due_to = "roll_threshold"
            print(f"    [!] early stop: roll={tick.roll_deg:.1f}deg at t={sim_time_s:.2f}s -- releasing steer, braking")

    settle_ticks = int(SETTLE_DURATION_S / FIXED_DT)
    for i in range(settle_ticks):
        tick_index = maneuver_ticks + i
        control = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
        vehicle.apply_control(control)
        world.tick()
        update_spectator_follow(spectator, vehicle)
        sim_time_s += FIXED_DT
        tick = capture_tick_from_actor(
            vehicle=vehicle, tick_index=tick_index, sim_time_s=sim_time_s, requested_control=control,
            prev_speed_mps=prev_speed_mps, prev_yaw_deg=prev_yaw_deg, dt_s=FIXED_DT, start_pose=start_pose,
        )
        ticks.append(tick)
        prev_speed_mps = tick.speed_mps
        prev_yaw_deg = tick.yaw_deg
        if tick.speed_mps < STOPPED_MPS and i > int(0.5 / FIXED_DT):
            break

    collision_sensor.stop()
    collision_sensor.destroy()
    vehicle.destroy()
    world.tick()

    roll_values = [t.roll_deg for t in ticks]
    z_values = [t.pos_z_m for t in ticks]
    max_abs_roll_deg = max(abs(r) for r in roll_values)
    final_roll_deg = roll_values[-1]
    final_pitch_deg = ticks[-1].pitch_deg
    z_range_m = max(z_values) - min(z_values)

    rollover = classify_rollover(max_abs_roll_deg=max_abs_roll_deg, threshold_deg=ROLLOVER_CONFIRM_DEG)
    upright = classify_upright_recovery(final_abs_roll_deg=abs(final_roll_deg), upright_threshold_deg=UPRIGHT_ROLL_DEG)
    rollover_evidence_status = (
        "confounded_by_map_collision" if stopped_due_to == "collision" else rollover.evidence_status
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
            "spawn_index": LOCATION_BY_DIRECTION[direction_name]["spawn_index"],
            "actual_entry_speed_mps": actual_entry_speed_mps,
            "actual_entry_speed_mph": actual_entry_speed_mps / MPH_TO_MPS,
            "entry_speed_method": "set_instant_entry_velocity",
            "early_stop_roll_deg": EARLY_STOP_ROLL_DEG,
            "rollover_confirm_deg": ROLLOVER_CONFIRM_DEG,
            "early_stopped": early_stopped,
            "stopped_due_to": stopped_due_to,
            "left_drivable_surface": left_drivable_surface,
            "first_non_drivable_time_s": first_non_drivable_time_s,
            "collision_other_actor": collision_flag["other_actor"],
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
        "actual_entry_speed_mph": actual_entry_speed_mps / MPH_TO_MPS,
        "early_stopped": early_stopped,
        "stopped_due_to": stopped_due_to,
        "max_abs_roll_deg": max_abs_roll_deg,
        "final_roll_deg": final_roll_deg,
        "final_pitch_deg": final_pitch_deg,
        "z_range_m": z_range_m,
        "rollover_detected": rollover.rollover_detected,
        "rollover_evidence_status": rollover_evidence_status,
        "upright_recovery": upright,
        "collision_detected": collision_flag["hit"],
        "collision_other_actor": collision_flag["other_actor"],
        "left_drivable_surface": left_drivable_surface,
        "first_non_drivable_time_s": first_non_drivable_time_s,
        "wheel_slip_signal": "not_measurable",
    }


def _print_row(row):
    print(
        f"    entry={row['actual_entry_speed_mph']:.1f}mph stopped_due_to={row['stopped_due_to']} "
        f"max_roll={row['max_abs_roll_deg']:.1f}deg final_roll={row['final_roll_deg']:.1f}deg "
        f"z_range={row['z_range_m']:.3f}m rollover={row['rollover_evidence_status']} "
        f"upright={row['upright_recovery']} left_drivable_surface={row['left_drivable_surface']} "
        f"collision={row['collision_detected']}({row['collision_other_actor']})"
    )


def _run_matrix(world, bp_lib, spawn_points, run_dir, repeat_suffix="", spectator=None):
    results = []
    first_boundary_case = None
    for target_mph in TARGET_SPEEDS_MPH:
        for direction_name, loc_cfg in LOCATION_BY_DIRECTION.items():
            spawn_tf = spawn_points[loc_cfg["spawn_index"]]
            print(f"[test15] running {target_mph:.0f}mph {direction_name} (spawn={loc_cfg['spawn_index']}){repeat_suffix} ...")
            row = _run_one_case(
                world=world, bp_lib=bp_lib, spawn_tf=spawn_tf,
                target_mph=target_mph, direction_name=direction_name,
                direction_sign=loc_cfg["direction_sign"], run_dir=run_dir, spectator=spectator,
            )
            row["case_id"] += repeat_suffix
            results.append(row)
            _print_row(row)
            if row["stopped_due_to"] == "roll_threshold" and first_boundary_case is None:
                first_boundary_case = (target_mph, direction_name)
    return results, first_boundary_case


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", "physics", f"{timestamp}_{RUN_FAMILY}")

    client, world = connect_and_load_world(host=HOST, port=PORT, timeout_s=10.0, target_map=TARGET_MAP)
    enable_sync_mode(world, fixed_dt=FIXED_DT)

    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    bp_lib = world.get_blueprint_library()
    spectator = world.get_spectator()

    all_results = []
    try:
        results, first_boundary_case = _run_matrix(world, bp_lib, spawn_points, run_dir, spectator=spectator)
        all_results.extend(results)

        if first_boundary_case is not None:
            target_mph, direction_name = first_boundary_case
            print(
                f"\n[test15] repeating first genuine roll-threshold boundary case "
                f"({target_mph:.0f}mph {direction_name}) to check reproducibility ..."
            )
            loc_cfg = LOCATION_BY_DIRECTION[direction_name]
            repeat_row = _run_one_case(
                world=world, bp_lib=bp_lib, spawn_tf=spawn_points[loc_cfg["spawn_index"]],
                target_mph=target_mph, direction_name=direction_name,
                direction_sign=loc_cfg["direction_sign"], run_dir=run_dir, spectator=spectator,
            )
            repeat_row["case_id"] += "_repeat"
            all_results.append(repeat_row)
            _print_row(repeat_row)
    finally:
        restore_async_mode(world)

    print("\n" + "=" * 150)
    print(
        f"{'case':22} {'entry_mph':10} {'stopped_due_to':15} {'max_roll':9} {'final_roll':11} "
        f"{'z_range_m':10} {'rollover':24} {'upright':11} {'left_road':10} {'collision':20}"
    )
    print("-" * 150)
    for row in all_results:
        collision_str = f"{row['collision_detected']}({row['collision_other_actor']})" if row["collision_detected"] else "False"
        print(
            f"{row['case_id']:22} {row['actual_entry_speed_mph']:10.1f} {row['stopped_due_to']:15} "
            f"{row['max_abs_roll_deg']:9.2f} {row['final_roll_deg']:11.2f} {row['z_range_m']:10.3f} "
            f"{row['rollover_evidence_status']:24} {row['upright_recovery']:11} "
            f"{str(row['left_drivable_surface']):10} {collision_str:20}"
        )
    print("=" * 150)
    n_collision_confounded = sum(1 for r in all_results if r["stopped_due_to"] == "collision")
    print(
        f"\n{n_collision_confounded}/{len(all_results)} case(s) were confounded by a map collision at this "
        f"location. Map: {TARGET_MAP} (spawn 90 for right turns, spawn 92 for left turns -- see module "
        "docstring). This substudy does NOT escalate speed further and does NOT search for a location that "
        "forces a positive rollover result -- a clean 'not_observed_in_tested_range' result at 90 mph is a "
        "valid, reportable answer in its own right."
    )
    print(
        "\nStudy limitation (stated, not re-derived): no validated per-wheel slip/lock signal is exposed by "
        "CARLA 0.9.16's Python API -- wheel_slip_signal=not_measurable for every case above. This substudy "
        "runs on Town03_Opt, a DIFFERENT map from the rest of the physical-limits suite (Town04_Opt) -- results "
        "here characterize this vehicle's rollover dynamics, not this specific highway's driving surface, and "
        "should not be silently pooled with Town04_Opt findings without noting the map change."
    )
    print(f"\nRun evidence written under: {run_dir}")


if __name__ == "__main__":
    main()
