"""test24___chrono_backend_smoke.py

Week 4, Work Sequence 3 (plans/Week-4_2026-10-01_0930_chrono-feasibility-plan.md):
the smallest non-interactive, backend-selectable smoke test for CARLA
0.9.16's built-in Chrono vehicle physics.

One run = one backend:
    python -X utf8 test24___chrono_backend_smoke.py --backend default
    python -X utf8 test24___chrono_backend_smoke.py --backend chrono

Both backends run the identical predeclared timeline on Town04_Opt spawn 242
(the Week 3 braking/throttle location), ClearNoon weather (set explicitly),
synchronous 0.02 s stepping:

    settle_before_enable (1 s, brake held)
    -> enable Chrono [chrono only; the default run skips the call]
    -> settle_after_enable (1 s, brake held) -> initial-state tolerance check
    -> physics_backend.SMOKE_SCHEDULE (v2): accelerate, modest right/left
       steer doublet with throttle held, coast, brake to a stop.

Comparison caveat (recorded in every manifest): the default run is the
Tesla Model 3 blueprint's default CARLA vehicle model; the Chrono run is the
same blueprint/mesh driven by the supplied Chrono SEDAN vehicle, powertrain,
and tire templates. This compares backend/configurations, not the same
physical car under two solvers.

Provenance rules:
  - Chrono paths (including every nested template reference) are validated
    and the server's --chrono launch flag is checked BEFORE connecting a
    vehicle; any failure exits without spawning.
  - The manifest's physics_backend is resolved from evidence
    (physics_backend.resolve_recorded_backend), never copied from --backend:
    a failed enable is "chrono_enable_failed"; a collision (which per CARLA's
    docs reverts Chrono to default physics) is "chrono_invalidated_by_
    collision". CARLA 0.9.16 has no server-side backend getter, so "chrono"
    means: server launched with --chrono + template validation passed +
    enable_chrono_physics() returned without error + no collision.
  - Each run gets a new, exclusive directory under src/runs/physics/
    (git-ignored); nothing is ever overwritten.
  - Actors are destroyed and asynchronous mode restored in `finally`, then
    the manifest/trace (possibly partial) are written so failures leave
    evidence too.

Does not touch the Chrono template files, the pedestrian scenario, or the
RL environment.
"""

import argparse
import datetime
import os
import platform
import sys
import time
import traceback

import carla

from carla_session import cleanup_session, enable_sync_mode, restore_async_mode
from map_drivability import classify_point_drivability
from physics_backend import (
    BACKEND_CHRONO,
    BACKEND_DEFAULT,
    CHRONO_STATUS_API_CALL_COMPLETED,
    CHRONO_STATUS_FAILED,
    CHRONO_STATUS_NOT_ATTEMPTED,
    CHRONO_STATUS_NOT_REQUESTED,
    DEFAULT_CARLA_ROOT,
    RECORDED_CHRONO,
    RECORDED_DEFAULT,
    SETTLE_AFTER_ENABLE,
    SETTLE_BEFORE_ENABLE,
    SMOKE_SCHEDULE,
    SMOKE_SCHEDULE_VERSION,
    VALID_BACKENDS,
    BackendConfigError,
    check_initial_state,
    create_run_dir,
    phase_tick_count,
    resolve_recorded_backend,
    run_preflight,
    schedule_to_dicts,
)
from physics_harness import (
    attach_collision_sensor,
    capture_tick_from_actor,
    read_physics_settings,
    summarize_acceleration,
    summarize_deceleration,
)
from spectator import update_spectator_follow
from trace_schema import PhysicsRunManifest, write_trace_csv

FIXED_DT = 0.02
TARGET_MAP = "Town04_Opt"
SPAWN_INDEX = 242
VEHICLE_BLUEPRINT = "vehicle.tesla.model3"
WEATHER_PRESET = "ClearNoon"
STOPPED_MPS = 0.15
STOP_HOLD_S = 0.5  # keep logging this long after the stop to see it settle
RUN_FAMILY = "chrono_smoke"
RUNS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "physics")

TERMINATION_STOPPED = "sequence_complete_stopped"
TERMINATION_BRAKE_TIMEOUT = "brake_phase_timeout_not_stopped"
TERMINATION_COLLISION = "collision"
TERMINATION_LEFT_ROAD = "left_drivable_surface"
TERMINATION_INITIAL_STATE = "initial_state_out_of_tolerance"
TERMINATION_CHRONO_ENABLE_FAILED = "chrono_enable_failed"
TERMINATION_SPAWN_FAILED = "spawn_failed"


class _RunAborted(Exception):
    """Ends the tick loop with a termination reason (not an error)."""

    def __init__(self, reason, detail=None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", required=True, choices=VALID_BACKENDS,
                        help="vehicle physics backend for this run (no default, on purpose)")
    parser.add_argument("--carla-root", default=DEFAULT_CARLA_ROOT)
    parser.add_argument("--chrono-base-dir", default=None,
                        help="defaults to <carla-root>/Co-Simulation/Chrono/Vehicles")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    return parser.parse_args(argv)


def _tick(world, vehicle, control, state, *, phase_name, first_tick_of_phase, spectator):
    """Apply control, advance one synchronous tick, capture a TraceTick."""
    vehicle.apply_control(control)
    t0 = time.perf_counter()
    frame = world.tick()
    wall_tick_s = time.perf_counter() - t0
    update_spectator_follow(spectator, vehicle)

    snapshot = world.get_snapshot()
    if state["t0_elapsed"] is None:
        state["t0_elapsed"] = snapshot.timestamp.elapsed_seconds - FIXED_DT
    tick = capture_tick_from_actor(
        vehicle=vehicle,
        tick_index=len(state["ticks"]),
        sim_time_s=snapshot.timestamp.elapsed_seconds - state["t0_elapsed"],
        requested_control=control,
        prev_speed_mps=state["ticks"][-1].speed_mps if state["ticks"] else None,
        prev_yaw_deg=state["ticks"][-1].yaw_deg if state["ticks"] else None,
        dt_s=FIXED_DT,
        start_pose=state["start_pose"],
        sim_frame=frame,
        wall_tick_s=wall_tick_s,
        event_marker=f"phase:{phase_name}" if first_tick_of_phase else None,
    )
    state["ticks"].append(tick)
    state["phase_by_tick"].append(phase_name)

    if state["collision_flag"]["hit"]:
        raise _RunAborted(TERMINATION_COLLISION, state["collision_flag"]["other_actor"])
    status, _ = classify_point_drivability(
        state["carla_map"], carla.Location(x=tick.pos_x_m, y=tick.pos_y_m, z=tick.pos_z_m)
    )
    if status == "non_drivable":
        raise _RunAborted(TERMINATION_LEFT_ROAD, f"x={tick.pos_x_m:.2f} y={tick.pos_y_m:.2f}")
    return tick


def _run_phase(world, vehicle, phase, state, spectator):
    """Returns True if the phase ended early because the vehicle stopped."""
    control = carla.VehicleControl(throttle=phase.throttle, brake=phase.brake, steer=phase.steer)
    hold_ticks = int(round(STOP_HOLD_S / FIXED_DT))
    stopped_at = None
    for i in range(phase_tick_count(phase, FIXED_DT)):
        tick = _tick(world, vehicle, control, state, phase_name=phase.name,
                     first_tick_of_phase=(i == 0), spectator=spectator)
        if phase.end_when_stopped and i > 0:
            if stopped_at is None and tick.speed_mps < STOPPED_MPS:
                stopped_at = i
                state["stop_time_s"] = tick.sim_time_s
            if stopped_at is not None and i - stopped_at >= hold_ticks:
                return True
    return stopped_at is not None


def _summarize(ticks, phase_by_tick):
    """Compact per-phase numbers for the console and manifest. Descriptive
    only -- the full trace is the evidence."""
    phases = {}
    for tick, name in zip(ticks, phase_by_tick):
        phases.setdefault(name, []).append(tick)
    summary = {}
    for name, rows in phases.items():
        yaw_rates = [abs(r.yaw_rate_dps) for r in rows if r.yaw_rate_dps is not None]
        walls = [r.wall_tick_s for r in rows if r.wall_tick_s is not None]
        summary[name] = {
            "ticks": len(rows),
            "start_speed_mps": rows[0].speed_mps,
            "end_speed_mps": rows[-1].speed_mps,
            "distance_m": sum(r.speed_mps * FIXED_DT for r in rows),
            "max_abs_yaw_rate_dps": max(yaw_rates) if yaw_rates else None,
            "max_abs_roll_deg": max(abs(r.roll_deg) for r in rows),
            "max_abs_pitch_deg": max(abs(r.pitch_deg) for r in rows),
            "end_lateral_displacement_m": rows[-1].lateral_displacement_m,
            "mean_wall_tick_s": sum(walls) / len(walls) if walls else None,
            "max_wall_tick_s": max(walls) if walls else None,
        }
    if "accelerate" in phases:
        acc = summarize_acceleration(
            speed_accel_pairs=[(r.speed_mps, r.accel_mps2) for r in phases["accelerate"]])
        summary["accelerate"]["normal_speed_peak_accel_mps2"] = acc.normal_speed_peak_accel_mps2
        summary["accelerate"]["normal_speed_mean_accel_mps2"] = acc.normal_speed_mean_accel_mps2
        summary["accelerate"]["low_speed_peak_accel_mps2"] = acc.low_speed_transient_peak_accel_mps2
    if "brake" in phases:
        dec = summarize_deceleration(
            speed_accel_pairs=[(r.speed_mps, r.accel_mps2) for r in phases["brake"]])
        summary["brake"]["normal_speed_peak_decel_mps2"] = dec.normal_speed_peak_decel_mps2
        summary["brake"]["normal_speed_mean_decel_mps2"] = dec.normal_speed_mean_decel_mps2
        summary["brake"]["low_speed_transient_peak_decel_mps2"] = dec.low_speed_transient_peak_decel_mps2
    return summary


def main(argv=None):
    args = _parse_args(argv)
    print(f"[test24] requested backend: {args.backend}")

    try:
        chrono_cfg, chrono_report, server_summary = run_preflight(
            backend=args.backend, carla_root=args.carla_root,
            chrono_base_dir=args.chrono_base_dir, host=args.host)
    except BackendConfigError as exc:
        print(f"\n[test24] PREFLIGHT FAILED -- nothing was spawned:\n  {exc}", file=sys.stderr)
        return 2
    if chrono_report is not None:
        print(f"[test24] Chrono templates validated ({len(chrono_report['referenced_files'])} nested "
              f"references); server launched with --chrono: {server_summary['any_has_chrono_flag']}")

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    client_version = client.get_client_version()
    server_version = client.get_server_version()
    if client_version != server_version:
        print(f"[test24] client {client_version} != server {server_version}; refusing to run",
              file=sys.stderr)
        return 2

    # Never start from a world left in synchronous mode by a crashed script
    # (MASTER summary, Week 3 occupancy finding): load_world could stall.
    current = client.get_world()
    if current.get_settings().synchronous_mode:
        print("[test24] world was left in synchronous mode; restoring async before loading map")
        restore_async_mode(current)
    world = client.load_world(TARGET_MAP)
    carla_map = world.get_map()
    print(f"[test24] connected: client={client_version} server={server_version} map={carla_map.name}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = create_run_dir(runs_root=RUNS_ROOT, timestamp=timestamp,
                             run_family=RUN_FAMILY, backend=args.backend)

    state = {
        "ticks": [], "phase_by_tick": [], "start_pose": None, "t0_elapsed": None,
        "collision_flag": {"hit": False, "other_actor": None}, "carla_map": carla_map,
        "stop_time_s": None,
    }
    actors = []
    chrono_status = CHRONO_STATUS_NOT_REQUESTED if args.backend == BACKEND_DEFAULT else CHRONO_STATUS_NOT_ATTEMPTED
    chrono_error = None
    chrono_enabled_after_tick = None
    termination_reason = None
    termination_detail = None
    error_traceback = None
    initial_state_problems = None
    spawn_record = None
    physics_settings = None
    run_wall_start = time.perf_counter()

    try:
        world.set_weather(getattr(carla.WeatherParameters, WEATHER_PRESET))
        enable_sync_mode(world, fixed_dt=FIXED_DT)
        world.tick()
        physics_settings = read_physics_settings(world)

        spawn_tf = carla_map.get_spawn_points()[SPAWN_INDEX]
        spawn_record = {"x": spawn_tf.location.x, "y": spawn_tf.location.y, "z": spawn_tf.location.z,
                        "yaw_deg": spawn_tf.rotation.yaw}
        bp_lib = world.get_blueprint_library()
        vehicle = world.try_spawn_actor(bp_lib.find(VEHICLE_BLUEPRINT), spawn_tf)
        if vehicle is None:
            raise _RunAborted(TERMINATION_SPAWN_FAILED, f"spawn point {SPAWN_INDEX} occupied")
        actors.append(vehicle)
        sensor, state["collision_flag"] = attach_collision_sensor(world, bp_lib, vehicle)
        actors.append(sensor)
        spectator = world.get_spectator()

        _run_phase(world, vehicle, SETTLE_BEFORE_ENABLE, state, spectator)

        if args.backend == BACKEND_CHRONO:
            print("[test24] calling enable_chrono_physics() ...")
            try:
                vehicle.enable_chrono_physics(
                    chrono_cfg.max_substeps, chrono_cfg.max_substep_delta_time_s,
                    chrono_cfg.vehicle_json, chrono_cfg.powertrain_json, chrono_cfg.tire_json,
                    chrono_cfg.api_base_path,
                )
            except Exception as exc:
                chrono_status = CHRONO_STATUS_FAILED
                chrono_error = repr(exc)
                raise _RunAborted(TERMINATION_CHRONO_ENABLE_FAILED, chrono_error) from exc
            chrono_status = CHRONO_STATUS_API_CALL_COMPLETED
            chrono_enabled_after_tick = len(state["ticks"]) - 1
            print("[test24] enable_chrono_physics() returned without error")

        _run_phase(world, vehicle, SETTLE_AFTER_ENABLE, state, spectator)

        last = state["ticks"][-1]
        initial_state_problems = check_initial_state(
            speed_mps=last.speed_mps, roll_deg=last.roll_deg, pitch_deg=last.pitch_deg)
        if initial_state_problems:
            raise _RunAborted(TERMINATION_INITIAL_STATE, "; ".join(initial_state_problems))

        state["start_pose"] = (last.pos_x_m, last.pos_y_m, last.yaw_deg)
        stopped = False
        for phase in SMOKE_SCHEDULE:
            stopped = _run_phase(world, vehicle, phase, state, spectator)
        termination_reason = TERMINATION_STOPPED if stopped else TERMINATION_BRAKE_TIMEOUT
    except _RunAborted as aborted:
        termination_reason = aborted.reason
        termination_detail = aborted.detail
    except BaseException as exc:  # includes KeyboardInterrupt -- still clean up and record
        termination_reason = f"exception:{type(exc).__name__}"
        termination_detail = repr(exc)
        error_traceback = traceback.format_exc()
    finally:
        cleanup_report = cleanup_session(world, actors)

    run_wall_s = time.perf_counter() - run_wall_start
    collision = state["collision_flag"]["hit"]
    recorded_backend = resolve_recorded_backend(
        requested_backend=args.backend, chrono_enable_status=chrono_status, collision_detected=collision)

    ticks = state["ticks"]
    if ticks and termination_reason is not None:
        last = ticks[-1]
        marker = f"termination:{termination_reason}"
        last.event_marker = f"{last.event_marker}+{marker}" if last.event_marker else marker
    summary = _summarize(ticks, state["phase_by_tick"]) if ticks else {}

    manifest = PhysicsRunManifest(
        run_id=os.path.basename(run_dir),
        test_family=RUN_FAMILY,
        created_at_iso=datetime.datetime.now().isoformat(),
        carla_version=server_version,
        map_name=carla_map.name,
        vehicle_blueprint=VEHICLE_BLUEPRINT,
        fixed_delta_seconds=(physics_settings or {}).get("fixed_delta_seconds", FIXED_DT),
        substepping_enabled=(physics_settings or {}).get("substepping_enabled"),
        max_substep_delta_time=(physics_settings or {}).get("max_substep_delta_time"),
        max_substeps=(physics_settings or {}).get("max_substeps"),
        weather_preset=WEATHER_PRESET,
        tire_friction=None,
        random_seed=None,
        physics_backend=recorded_backend,
        parameters={
            "requested_backend": args.backend,
            "recorded_backend": recorded_backend,
            "backend_evidence_note": (
                "CARLA 0.9.16 exposes no server-side physics-backend getter. 'chrono' means: server "
                "launched with --chrono, template validation passed, enable_chrono_physics() returned "
                "without error, and no collision (which reverts Chrono to default physics) occurred."
            ),
            "comparison_caveat": (
                "default = Tesla Model 3 blueprint under CARLA default vehicle physics; chrono = same "
                "blueprint/mesh driven by the supplied Chrono sedan vehicle/powertrain/tire templates."
            ),
            "client_version": client_version,
            "server_version": server_version,
            "carla_module_file": carla.__file__,
            "python_version": sys.version,
            "platform": platform.platform(),
            "host": args.host,
            "port": args.port,
            "server_processes": server_summary,
            "chrono": {
                "enable_status": chrono_status,
                "enable_error": chrono_error,
                "enabled_after_tick_index": chrono_enabled_after_tick,
                "config": chrono_report,
            },
            "spawn_index": SPAWN_INDEX,
            "spawn_transform": spawn_record,
            "settle_phases": schedule_to_dicts((SETTLE_BEFORE_ENABLE, SETTLE_AFTER_ENABLE)),
            "schedule_version": SMOKE_SCHEDULE_VERSION,
            "schedule": schedule_to_dicts(SMOKE_SCHEDULE),
            "stopped_threshold_mps": STOPPED_MPS,
            "stop_hold_s": STOP_HOLD_S,
            "initial_state_problems": initial_state_problems,
            "termination_reason": termination_reason,
            "termination_detail": termination_detail,
            "error_traceback": error_traceback,
            "collision_detected": collision,
            "collision_other_actor": state["collision_flag"]["other_actor"],
            "stop_time_s": state["stop_time_s"],
            "tick_count": len(ticks),
            "run_wall_s": run_wall_s,
            "cleanup": cleanup_report,
            "wheel_slip_signal": "not_measurable",
            "phase_summary": summary,
        },
    )
    manifest.to_json(os.path.join(run_dir, "manifest.json"), overwrite=False)
    write_trace_csv(os.path.join(run_dir, "trace.csv"), ticks)

    print(f"\n[test24] termination: {termination_reason}" + (f" ({termination_detail})" if termination_detail else ""))
    print(f"[test24] recorded backend: {recorded_backend}   ticks: {len(ticks)}   wall: {run_wall_s:.1f}s")
    print(f"[test24] cleanup: {cleanup_report}")
    for name, s in summary.items():
        print(f"    {name:22} v {s['start_speed_mps']:6.2f}->{s['end_speed_mps']:6.2f} m/s  "
              f"dist {s['distance_m']:6.2f} m  max|yaw rate| {s['max_abs_yaw_rate_dps'] or 0:6.2f} deg/s  "
              f"max|roll| {s['max_abs_roll_deg']:5.2f}  wall/tick {(s['mean_wall_tick_s'] or 0) * 1000:6.2f} ms")
    print(f"[test24] evidence: {run_dir}")

    if args.backend == BACKEND_CHRONO and recorded_backend != RECORDED_CHRONO:
        print(f"\n[test24] !!! CHRONO RUN NOT VALID: recorded backend is '{recorded_backend}'. "
              "Do not use this run as Chrono evidence. !!!", file=sys.stderr)
        return 1
    if args.backend == BACKEND_DEFAULT and recorded_backend != RECORDED_DEFAULT:
        return 1
    if termination_reason != TERMINATION_STOPPED or cleanup_report["errors"]:
        print("[test24] run did not complete cleanly; inspect the manifest before using it.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
