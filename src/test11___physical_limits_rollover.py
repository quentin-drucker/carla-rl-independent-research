"""test11___physical_limits_rollover.py

Week 3, Workstream 2.3: the rollover/flip boundary substudy of the
connected vehicle physical-limits suite. Reuses the test10 steering-lock
setup (coasting full-lock turns, direct low-level control) and increases
speed only through a bounded, predeclared sequence -- this is NOT an
open-ended search for the most extreme crash.

Matrix: 5 bounded entry speeds (30/45/60/75/90 mph) x left/right, step
steering only (the more aggressive of the two commands already
characterized in test10 -- no need to also test ramp here).

Safety design per the plan:
    - Every tick, roll angle is monitored live. If it crosses
      EARLY_STOP_ROLL_DEG, steering is immediately released and the brake
      is applied -- the run does not continue driving deeper into an
      unstable state just to see what happens next.
    - After the maneuver (or an early stop), the vehicle is braked and
      given a settle window, then terminal roll is checked to classify
      whether it recovered upright.
    - The predeclared speed sequence is a hard, bounded ceiling (90 mph) --
      if no case in it reaches the rollover-confirm threshold, the
      substudy reports "not_observed_in_tested_range" and stops; it does
      not escalate further seeking a positive result.
    - The FIRST case that reaches EARLY_STOP_ROLL_DEG is automatically
      repeated once more at the end of the run, per the plan's "repeat the
      first apparent rollover boundary case to distinguish reproducible
      behavior from a one-off physics artifact."

Reported quantities: max/final roll angle, vertical position (pos_z_m)
range, terminal orientation (final roll/pitch), collision (via a collision
sensor -- CARLA's own contact detection, not a slip/lock claim), and
upright-recovery classification. No "rollover" is claimed unless roll
crosses the predeclared, measured threshold (physics_harness.classify_
rollover, default 60 deg) -- appearance alone never drives a claim here.

Usage (CARLA must already be running):
    python -X utf8 test11___physical_limits_rollover.py

Writes one timestamped, non-overwritable run directory per case (plus any
automatic repeat) under runs/physics/<timestamp>_rollover/.
"""

import datetime
import itertools
import os

import carla

from carla_session import connect_and_load_world, enable_sync_mode, restore_async_mode
from map_drivability import classify_point_drivability
from physics_harness import (
    MPH_TO_MPS,
    accelerate_to_matched_entry_speed,
    attach_collision_sensor,
    capture_tick_from_actor,
    classify_rollover,
    classify_upright_recovery,
    read_physics_settings,
)
from trace_schema import PhysicsRunManifest, write_trace_csv

HOST = "localhost"
PORT = 2000
FIXED_DT = 0.02
SPAWN_INDEX = 242
TARGET_MAP = "Town04_Opt"
MAX_ACCEL_TICKS = 3000
STOPPED_MPS = 0.15
MAX_STOP_TICKS = 1500

# Bounded, predeclared -- not an open-ended escalation. 90 mph is the hard
# ceiling for this substudy; if nothing crosses the rollover-confirm
# threshold by then, the answer is "not observed in this tested range".
TARGET_SPEEDS_MPH = [30.0, 45.0, 60.0, 75.0, 90.0]
DIRECTIONS = {"left": -1.0, "right": +1.0}
MANEUVER_DURATION_S = 5.0
SETTLE_DURATION_S = 3.0

EARLY_STOP_ROLL_DEG = 45.0  # conservative -- stop driving deeper before the confirm threshold
ROLLOVER_CONFIRM_DEG = 60.0  # matches physics_harness.classify_rollover default
UPRIGHT_ROLL_DEG = 15.0

RUN_FAMILY = "rollover"


def _run_one_case(*, world, bp_lib, spawn_tf, target_mph, direction_name, direction_sign, run_dir):
    target_mps = target_mph * MPH_TO_MPS
    case_id = f"{target_mph:.0f}mph_{direction_name}"

    bp = bp_lib.find("vehicle.tesla.model3")
    vehicle = world.spawn_actor(bp, spawn_tf)
    world.tick()
    collision_sensor, collision_flag = attach_collision_sensor(world, bp_lib, vehicle)

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
    early_stopped = False
    stopped_due_to = "none"  # "none" | "roll_threshold" | "collision"
    left_drivable_surface = False
    first_non_drivable_time_s = None

    for tick_index in range(maneuver_ticks):
        if early_stopped:
            control = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
        else:
            control = carla.VehicleControl(throttle=0.0, brake=0.0, steer=direction_sign * 1.0)
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

        drivability_status, _ = classify_point_drivability(
            world.get_map(), carla.Location(x=tick.pos_x_m, y=tick.pos_y_m, z=tick.pos_z_m)
        )
        if drivability_status == "non_drivable" and not left_drivable_surface:
            left_drivable_surface = True
            first_non_drivable_time_s = sim_time_s

        # A collision with fixed map geometry (e.g. a guardrail) produces
        # chaotic post-impact physics unrelated to the vehicle's own
        # cornering dynamics -- stop immediately rather than let bounce
        # physics corrupt the roll/orientation reading, per the same
        # "stop on a conservative threshold, don't drive deeper" rule as
        # the roll-based early stop.
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

    # Settle window: brake to a stop (or just hold brake if already at low
    # speed) and keep recording, so terminal orientation reflects the
    # vehicle after it has actually stopped moving, not mid-maneuver.
    settle_ticks = int(SETTLE_DURATION_S / FIXED_DT)
    for i in range(settle_ticks):
        tick_index = maneuver_ticks + i
        control = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
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
    # A run cut short by a map-geometry collision did not get a clean read
    # on the vehicle's own rollover dynamics at this speed -- "no rollover
    # observed" and "we couldn't actually test it here" are different
    # claims, and conflating them would overstate what this run shows.
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
            "actual_entry_speed_mps": actual_entry_speed_mps,
            "actual_entry_speed_mph": actual_entry_speed_mps / MPH_TO_MPS,
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


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", "physics", f"{timestamp}_{RUN_FAMILY}")

    client, world = connect_and_load_world(host=HOST, port=PORT, timeout_s=10.0, target_map=TARGET_MAP)
    enable_sync_mode(world, fixed_dt=FIXED_DT)

    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    spawn_tf = spawn_points[SPAWN_INDEX]
    bp_lib = world.get_blueprint_library()

    results = []
    first_boundary_case = None
    try:
        for target_mph, (direction_name, direction_sign) in itertools.product(TARGET_SPEEDS_MPH, DIRECTIONS.items()):
            print(f"[test11] running {target_mph:.0f}mph {direction_name} ...")
            row = _run_one_case(
                world=world, bp_lib=bp_lib, spawn_tf=spawn_tf,
                target_mph=target_mph, direction_name=direction_name, direction_sign=direction_sign,
                run_dir=run_dir,
            )
            results.append(row)
            _print_row(row)
            # Only a genuine roll-threshold event is a "rollover boundary
            # case" worth repeating -- a collision-triggered stop is a
            # map-geometry confound (see stopped_due_to), not a vehicle-
            # dynamics finding, and repeating it would just re-measure the
            # guardrail.
            if row["stopped_due_to"] == "roll_threshold" and first_boundary_case is None:
                first_boundary_case = (target_mph, direction_name, direction_sign)

        if first_boundary_case is not None:
            target_mph, direction_name, direction_sign = first_boundary_case
            print(
                f"\n[test11] repeating first genuine roll-threshold boundary case "
                f"({target_mph:.0f}mph {direction_name}) to check reproducibility, "
                "not a one-off physics artifact ..."
            )
            repeat_row = _run_one_case(
                world=world, bp_lib=bp_lib, spawn_tf=spawn_tf,
                target_mph=target_mph, direction_name=direction_name, direction_sign=direction_sign,
                run_dir=run_dir,
            )
            repeat_row["case_id"] += "_repeat"
            results.append(repeat_row)
            _print_row(repeat_row)
    finally:
        restore_async_mode(world)

    print("\n" + "=" * 150)
    print(
        f"{'case':22} {'entry_mph':10} {'stopped_due_to':15} {'max_roll':9} {'final_roll':11} "
        f"{'z_range_m':10} {'rollover':24} {'upright':11} {'left_road':10} {'collision':20}"
    )
    print("-" * 150)
    for row in results:
        collision_str = f"{row['collision_detected']}({row['collision_other_actor']})" if row["collision_detected"] else "False"
        print(
            f"{row['case_id']:22} {row['actual_entry_speed_mph']:10.1f} {row['stopped_due_to']:15} "
            f"{row['max_abs_roll_deg']:9.2f} {row['final_roll_deg']:11.2f} {row['z_range_m']:10.3f} "
            f"{row['rollover_evidence_status']:24} {row['upright_recovery']:11} "
            f"{str(row['left_drivable_surface']):10} {collision_str:20}"
        )
    print("=" * 150)
    n_collision_confounded = sum(1 for r in results if r["stopped_due_to"] == "collision")
    if first_boundary_case is None:
        print(
            f"\nNo case in the bounded {min(TARGET_SPEEDS_MPH):.0f}-{max(TARGET_SPEEDS_MPH):.0f} mph sequence reached "
            f"the {EARLY_STOP_ROLL_DEG:.0f} deg early-stop roll threshold via the vehicle's own dynamics. "
            f"{n_collision_confounded}/{len(results)} case(s) instead ran off the drivable surface and collided "
            "with fixed roadside infrastructure (e.g. a guardrail) before any intrinsic rollover dynamics could "
            "develop -- those are reported as rollover='confounded_by_map_collision', NOT as "
            "'not_observed_in_tested_range', since the maneuver's own dynamics were never actually tested at "
            "that speed/location. This substudy does NOT escalate speed further, and does NOT relocate to a more "
            "open area to force a positive rollover result -- per the Week 3 plan, a bounded sequence with these "
            "outcomes is itself a valid, reportable result (see MASTER_CARLA_RESEARCH_SUMMARY.md for the "
            "recommended next step: repeat this substudy at an open, unobstructed spawn location before drawing "
            "any conclusion about intrinsic rollover behavior at these speeds)."
        )
    print(
        "\nStudy limitation (stated, not re-derived): no validated per-wheel slip/lock signal is exposed by "
        "CARLA 0.9.16's Python API -- wheel_slip_signal=not_measurable for every case above. Roll/orientation "
        "and collision are CARLA's own reported physics/contact state, not a claim about real-world rollover "
        "risk for this vehicle. left_drivable_surface uses the same conservative project_to_road=False map "
        "query already validated in Week 2's map_drivability.py."
    )
    print(f"\nRun evidence written under: {run_dir}")


if __name__ == "__main__":
    main()
