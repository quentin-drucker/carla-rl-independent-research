"""test25___chrono_hold_diagnostic.py

Week 4 follow-up to test24: WHY doesn't a Chrono vehicle stay still with the
brake held? test24's Chrono runs (2026-10-01) never reached the measured
sequence: right after enable_chrono_physics() the parked car jolted, then
crept at ~0.16 m/s while turning ~4.5 deg/s, with brake=1.0 commanded.
Bit-identical on repeat. (test24's 1 s window made the creep look sideways;
this script's 10 s holds showed it is mainly backward, i.e. downhill.)

Findings (2026-10-01; two full runs, bit-identical): default physics never
moved, even unbraked on the slope. Chrono with brake 1.0 rolled ~2.0 m down
the 0.62 deg slope and ~12 cm on the flat road; unbraked it rolled ~3.4 m.
Diagnosed cause: Chrono 6.0.0's ChBrakeSimple cannot model static sticking,
and CARLA 0.9.16 never enables Chrono's brake locking. The hand-brake case
is NOT a parking-brake test: CARLA passes brake + hand_brake to Chrono
unclamped, so that case commands an out-of-range 2.0. See the branch worklog
and MASTER_CARLA_RESEARCH_SUMMARY.md (Week 4) for the backend decision.

This script holds a parked car still under a small set of controlled
variations, one variable at a time, so you can both MEASURE and WATCH which
one changes the creep:

  case                         backend  location     control while holding     isolates
  default_brake_slope          default  spawn 242    brake 1.0                 control: does default hold?
  default_no_brake_slope       default  spawn 242    no brake                  control: does the slope roll it?
  chrono_brake_slope           chrono   spawn 242    brake 1.0                 reproduces test24
  chrono_handbrake_slope       chrono   spawn 242    brake 1.0 + hand brake    is the hand brake passed through?
  chrono_no_brake_slope        chrono   spawn 242    no brake                  do Chrono's brakes do anything at rest?
  default_brake_flat           default  spawn 11     brake 1.0                 control on flat road
  chrono_brake_flat            chrono   spawn 11     brake 1.0                 is it the 0.6 deg slope?
  chrono_brake_enable_at_spawn chrono   spawn 242    brake 1.0                 is it WHEN Chrono is enabled?

Spawn 242 (Town04_Opt) is the test24 location: road slope ~0.62 deg.
Spawn 11 is ~280 m away, same heading (-179 deg), road slope 0.00 deg.

Every case: spawn -> 1 s default-physics settle with brake 1.0 (except the
enable_at_spawn case, which enables Chrono on the very first tick, while the
car is still dropping onto the road) -> enable Chrono [chrono cases] -> hold
for HOLD_S seconds with the case's control.

WHAT YOU WILL SEE (watch the CARLA server window):
  - The camera is FIXED beside the car (it does not follow), so any creep or
    rotation is visible against the road.
  - A red dot marks where the car was when the hold started, a green arrow
    shows its starting heading, and a yellow trail draws its path.
  - Text above the car shows the case, speed, distance moved, and heading change.
  - By default the simulation is paced to real time so you can watch.

Usage (CARLA must be running with --chrono; from src/ in the venv):
    python -X utf8 test25___chrono_hold_diagnostic.py                     # all cases
    python -X utf8 test25___chrono_hold_diagnostic.py --pause             # wait for Enter between cases
    python -X utf8 test25___chrono_hold_diagnostic.py --case chrono_brake_slope --case default_brake_slope
    python -X utf8 test25___chrono_hold_diagnostic.py --list              # list cases
    python -X utf8 test25___chrono_hold_diagnostic.py --fast              # no real-time pacing

Evidence: one manifest.json + trace.csv per case under
src/runs/physics/<timestamp>_chrono_hold_diag/<case>/ (git-ignored, never
overwritten). Chrono templates are validated and the --chrono server flag is
checked before anything spawns; templates are never modified.
"""

import argparse
import datetime
import math
import os
import platform
import sys
import time
import traceback
from dataclasses import asdict, dataclass

import carla

from carla_session import cleanup_session, enable_sync_mode, restore_async_mode
from physics_backend import (
    BACKEND_CHRONO,
    BACKEND_DEFAULT,
    CHRONO_STATUS_API_CALL_COMPLETED,
    CHRONO_STATUS_FAILED,
    CHRONO_STATUS_NOT_ATTEMPTED,
    CHRONO_STATUS_NOT_REQUESTED,
    DEFAULT_CARLA_ROOT,
    BackendConfigError,
    resolve_recorded_backend,
    run_preflight,
)
from physics_harness import attach_collision_sensor, capture_tick_from_actor, read_physics_settings, summarize_stationary_hold
from trace_schema import PhysicsRunManifest, write_trace_csv

FIXED_DT = 0.02
TARGET_MAP = "Town04_Opt"
VEHICLE_BLUEPRINT = "vehicle.tesla.model3"
WEATHER_PRESET = "ClearNoon"
SETTLE_S = 1.0
HOLD_S = 10.0
SLOPE_SPAWN = 242
FLAT_SPAWN = 11
RUN_FAMILY = "chrono_hold_diag"
RUNS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "physics")


@dataclass(frozen=True)
class HoldCase:
    name: str
    backend: str
    spawn_index: int
    brake: float
    hand_brake: bool = False
    enable_at_spawn: bool = False
    isolates: str = ""


CASES = (
    HoldCase("default_brake_slope", BACKEND_DEFAULT, SLOPE_SPAWN, 1.0,
             isolates="control: does default physics hold the car?"),
    HoldCase("default_no_brake_slope", BACKEND_DEFAULT, SLOPE_SPAWN, 0.0,
             isolates="control: does the 0.6 deg slope roll an unbraked car?"),
    HoldCase("chrono_brake_slope", BACKEND_CHRONO, SLOPE_SPAWN, 1.0,
             isolates="reproduces the test24 creep"),
    HoldCase("chrono_handbrake_slope", BACKEND_CHRONO, SLOPE_SPAWN, 1.0, hand_brake=True,
             isolates="is CARLA's hand brake passed through to Chrono?"),
    HoldCase("chrono_no_brake_slope", BACKEND_CHRONO, SLOPE_SPAWN, 0.0,
             isolates="do Chrono's brakes do anything at standstill?"),
    HoldCase("default_brake_flat", BACKEND_DEFAULT, FLAT_SPAWN, 1.0,
             isolates="control on a flat road"),
    HoldCase("chrono_brake_flat", BACKEND_CHRONO, FLAT_SPAWN, 1.0,
             isolates="is the creep caused by the slope?"),
    HoldCase("chrono_brake_enable_at_spawn", BACKEND_CHRONO, SLOPE_SPAWN, 1.0, enable_at_spawn=True,
             isolates="does enabling Chrono before landing (no default settle) change it?"),
)
CASES_BY_NAME = {c.name: c for c in CASES}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Chrono held-brake creep diagnostic (see module docstring)")
    parser.add_argument("--case", action="append", choices=list(CASES_BY_NAME),
                        help="run only this case (repeatable); default: all cases in table order")
    parser.add_argument("--list", action="store_true", help="list cases and exit")
    parser.add_argument("--pause", action="store_true", help="wait for Enter before each case")
    parser.add_argument("--fast", action="store_true", help="do not pace the simulation to real time")
    parser.add_argument("--carla-root", default=DEFAULT_CARLA_ROOT)
    parser.add_argument("--chrono-base-dir", default=None)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    return parser.parse_args(argv)


# ----------------------------------------------------------------------
# Visual aids (CARLA debug drawing + fixed spectator) -- display only.
# ----------------------------------------------------------------------

def _look_at(cam_loc, target_loc):
    dx, dy, dz = target_loc.x - cam_loc.x, target_loc.y - cam_loc.y, target_loc.z - cam_loc.z
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    return carla.Transform(cam_loc, carla.Rotation(pitch=pitch, yaw=yaw))


def _place_fixed_spectator(world, spawn_tf):
    """3/4 view from the car's right-rear, elevated, not following."""
    f = spawn_tf.get_forward_vector()
    r = spawn_tf.get_right_vector()
    target = spawn_tf.location
    cam = carla.Location(x=target.x - 4.0 * f.x + 7.0 * r.x,
                         y=target.y - 4.0 * f.y + 7.0 * r.y,
                         z=target.z + 4.5)
    world.get_spectator().set_transform(_look_at(cam, target))


def _draw_start_markers(world, tick, life_s):
    debug = world.debug
    start = carla.Location(x=tick.pos_x_m, y=tick.pos_y_m, z=tick.pos_z_m + 0.05)
    debug.draw_point(start, size=0.12, color=carla.Color(255, 0, 0), life_time=life_s)
    yaw = math.radians(tick.yaw_deg)
    tip = carla.Location(x=start.x + 3.0 * math.cos(yaw), y=start.y + 3.0 * math.sin(yaw), z=start.z)
    debug.draw_arrow(start, tip, thickness=0.05, arrow_size=0.2, color=carla.Color(0, 255, 0), life_time=life_s)


def _draw_live(world, case, tick, start_tick, prev_tick, life_s):
    debug = world.debug
    if prev_tick is not None and tick.tick_index % 5 == 0:
        debug.draw_line(
            carla.Location(x=prev_tick.pos_x_m, y=prev_tick.pos_y_m, z=prev_tick.pos_z_m + 0.05),
            carla.Location(x=tick.pos_x_m, y=tick.pos_y_m, z=tick.pos_z_m + 0.05),
            thickness=0.04, color=carla.Color(255, 220, 0), life_time=life_s)
    moved_cm = 100.0 * math.hypot(tick.pos_x_m - start_tick.pos_x_m, tick.pos_y_m - start_tick.pos_y_m) if start_tick else 0.0
    turned = (tick.yaw_deg - start_tick.yaw_deg + 180.0) % 360.0 - 180.0 if start_tick else 0.0
    text = f"{case.name}  v={tick.speed_mps:.2f} m/s  moved={moved_cm:.0f} cm  turned={turned:+.1f} deg"
    debug.draw_string(carla.Location(x=tick.pos_x_m, y=tick.pos_y_m, z=tick.pos_z_m + 2.2), text,
                      draw_shadow=True, color=carla.Color(255, 255, 255), life_time=FIXED_DT * 1.5)


# ----------------------------------------------------------------------
# One case
# ----------------------------------------------------------------------

def _run_case(*, world, carla_map, bp_lib, case, chrono_cfg, run_dir, realtime, server_version):
    spawn_tf = carla_map.get_spawn_points()[case.spawn_index]
    _place_fixed_spectator(world, spawn_tf)
    actors, ticks = [], []
    chrono_status = CHRONO_STATUS_NOT_REQUESTED if case.backend == BACKEND_DEFAULT else CHRONO_STATUS_NOT_ATTEMPTED
    chrono_error = None
    hold_start_index = None
    termination, detail, error_tb = None, None, None
    hand_brake_readback = None
    collision_flag = {"hit": False, "other_actor": None}
    life_s = SETTLE_S + HOLD_S + 3.0
    state = {"t0": None, "next_wall": time.perf_counter()}

    def tick_once(control):
        vehicle.apply_control(control)
        t_wall = time.perf_counter()
        frame = world.tick()
        wall_tick_s = time.perf_counter() - t_wall
        snap = world.get_snapshot()
        if state["t0"] is None:
            state["t0"] = snap.timestamp.elapsed_seconds - FIXED_DT
        tick = capture_tick_from_actor(
            vehicle=vehicle, tick_index=len(ticks), sim_time_s=snap.timestamp.elapsed_seconds - state["t0"],
            requested_control=control,
            prev_speed_mps=ticks[-1].speed_mps if ticks else None,
            prev_yaw_deg=ticks[-1].yaw_deg if ticks else None,
            dt_s=FIXED_DT, sim_frame=frame, wall_tick_s=wall_tick_s,
        )
        ticks.append(tick)
        start_tick = ticks[hold_start_index] if hold_start_index is not None else None
        _draw_live(world, case, tick, start_tick, ticks[-6] if len(ticks) > 5 else None, life_s)
        if realtime:
            state["next_wall"] += FIXED_DT
            delay = state["next_wall"] - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                state["next_wall"] = time.perf_counter()
        if collision_flag["hit"]:
            raise RuntimeError(f"collision with {collision_flag['other_actor']}")
        return tick

    def enable_chrono():
        nonlocal chrono_status, chrono_error
        try:
            vehicle.enable_chrono_physics(
                chrono_cfg.max_substeps, chrono_cfg.max_substep_delta_time_s,
                chrono_cfg.vehicle_json, chrono_cfg.powertrain_json, chrono_cfg.tire_json,
                chrono_cfg.api_base_path)
            chrono_status = CHRONO_STATUS_API_CALL_COMPLETED
        except Exception as exc:
            chrono_status = CHRONO_STATUS_FAILED
            chrono_error = repr(exc)
            raise

    vehicle = None
    try:
        vehicle = world.try_spawn_actor(bp_lib.find(VEHICLE_BLUEPRINT), spawn_tf)
        if vehicle is None:
            raise RuntimeError(f"spawn point {case.spawn_index} occupied")
        actors.append(vehicle)
        sensor, collision_flag = attach_collision_sensor(world, bp_lib, vehicle)
        actors.append(sensor)

        settle = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
        hold = carla.VehicleControl(throttle=0.0, brake=case.brake, steer=0.0, hand_brake=case.hand_brake)

        if case.enable_at_spawn:
            enable_chrono()
            hold_start_index = 0
            _draw_start_markers(world, tick_once(hold), life_s)
            for _ in range(int(round((SETTLE_S + HOLD_S) / FIXED_DT)) - 1):
                tick_once(hold)
        else:
            for _ in range(int(round(SETTLE_S / FIXED_DT))):
                tick_once(settle)
            if case.backend == BACKEND_CHRONO:
                enable_chrono()
            hold_start_index = len(ticks)
            first = tick_once(hold)
            _draw_start_markers(world, first, life_s)
            for _ in range(int(round(HOLD_S / FIXED_DT)) - 1):
                tick_once(hold)
        hand_brake_readback = vehicle.get_control().hand_brake
        termination = "hold_complete"
    except BaseException as exc:  # still clean up and record (incl. Ctrl+C)
        termination = f"exception:{type(exc).__name__}"
        detail = repr(exc)
        error_tb = traceback.format_exc()
    finally:
        cleanup = cleanup_session(world, actors, restore_async=False)
        try:
            world.tick()
        except Exception:  # noqa: BLE001
            pass

    recorded = resolve_recorded_backend(
        requested_backend=case.backend, chrono_enable_status=chrono_status,
        collision_detected=collision_flag["hit"])
    hold_ticks = ticks[hold_start_index:] if hold_start_index is not None else []
    summary = summarize_stationary_hold(hold_ticks)
    # Skip the first 0.3 s (the enable jolt) for a settled-creep view.
    settled = [t for t in hold_ticks if t.sim_time_s - hold_ticks[0].sim_time_s >= 0.3] if hold_ticks else []
    summary_after_jolt = summarize_stationary_hold(settled)
    walls = [t.wall_tick_s for t in hold_ticks if t.wall_tick_s is not None]

    case_dir = os.path.join(run_dir, case.name)
    os.makedirs(case_dir, exist_ok=False)
    manifest = PhysicsRunManifest(
        run_id=f"{os.path.basename(run_dir)}/{case.name}",
        test_family=RUN_FAMILY,
        created_at_iso=datetime.datetime.now().isoformat(),
        carla_version=server_version,
        map_name=carla_map.name,
        vehicle_blueprint=VEHICLE_BLUEPRINT,
        weather_preset=WEATHER_PRESET,
        tire_friction=None,
        random_seed=None,
        physics_backend=recorded,
        parameters={
            "case": asdict(case),
            "settle_s": 0.0 if case.enable_at_spawn else SETTLE_S,
            "hold_s": SETTLE_S + HOLD_S if case.enable_at_spawn else HOLD_S,
            "hold_start_tick_index": hold_start_index,
            "spawn_transform": {"x": spawn_tf.location.x, "y": spawn_tf.location.y,
                                "z": spawn_tf.location.z, "yaw_deg": spawn_tf.rotation.yaw},
            "chrono": {"enable_status": chrono_status, "enable_error": chrono_error,
                       "max_substeps": chrono_cfg.max_substeps if chrono_cfg else None,
                       "max_substep_delta_time_s": chrono_cfg.max_substep_delta_time_s if chrono_cfg else None},
            "hand_brake_requested": case.hand_brake,
            "hand_brake_readback": hand_brake_readback,
            "termination_reason": termination,
            "termination_detail": detail,
            "error_traceback": error_tb,
            "collision_detected": collision_flag["hit"],
            "hold_summary": summary,
            "hold_summary_after_first_0p3s": summary_after_jolt,
            "mean_wall_tick_s": sum(walls) / len(walls) if walls else None,
            "realtime_paced": realtime,
            "cleanup": cleanup,
        },
        **read_physics_settings(world),
    )
    return manifest, ticks, case_dir


def _fmt(v, spec):
    return "  n/a " if v is None else format(v, spec)


def main(argv=None):
    args = _parse_args(argv)
    if args.list:
        for c in CASES:
            print(f"{c.name:30} backend={c.backend:7} spawn={c.spawn_index:3} brake={c.brake} "
                  f"hand_brake={c.hand_brake} enable_at_spawn={c.enable_at_spawn}  -- {c.isolates}")
        return 0
    cases = [CASES_BY_NAME[n] for n in args.case] if args.case else list(CASES)

    chrono_cfg, chrono_report, server_summary = None, None, None
    try:
        backend = BACKEND_CHRONO if any(c.backend == BACKEND_CHRONO for c in cases) else BACKEND_DEFAULT
        chrono_cfg, chrono_report, server_summary = run_preflight(
            backend=backend, carla_root=args.carla_root,
            chrono_base_dir=args.chrono_base_dir, host=args.host)
    except BackendConfigError as exc:
        print(f"[test25] PREFLIGHT FAILED -- nothing was spawned:\n  {exc}", file=sys.stderr)
        return 2

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    client_version, server_version = client.get_client_version(), client.get_server_version()
    if client_version != server_version:
        print(f"[test25] client {client_version} != server {server_version}; refusing", file=sys.stderr)
        return 2
    current = client.get_world()
    if current.get_settings().synchronous_mode:
        restore_async_mode(current)
    world = client.load_world(TARGET_MAP)
    carla_map = world.get_map()
    bp_lib = world.get_blueprint_library()

    run_dir = os.path.join(RUNS_ROOT, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{RUN_FAMILY}")
    os.makedirs(run_dir, exist_ok=False)
    print(f"[test25] {len(cases)} case(s); evidence -> {run_dir}")
    print("[test25] Watch the CARLA window: red dot = start, green arrow = start heading, yellow = path.\n")

    results = []
    exit_code = 0
    try:
        world.set_weather(getattr(carla.WeatherParameters, WEATHER_PRESET))
        enable_sync_mode(world, fixed_dt=FIXED_DT)
        world.tick()
        for case in cases:
            print(f"--- {case.name}: {case.isolates}")
            if args.pause:
                input("    press Enter to start this case ... ")
            manifest, ticks, case_dir = _run_case(
                world=world, carla_map=carla_map, bp_lib=bp_lib, case=case,
                chrono_cfg=chrono_cfg, run_dir=run_dir, realtime=not args.fast,
                server_version=server_version)
            manifest.parameters.update({
                "client_version": client_version, "server_version": server_version,
                "carla_module_file": carla.__file__, "python_version": sys.version,
                "platform": platform.platform(), "server_processes": server_summary,
                "chrono_config": chrono_report if case.backend == BACKEND_CHRONO else None,
            })
            manifest.to_json(os.path.join(case_dir, "manifest.json"), overwrite=False)
            write_trace_csv(os.path.join(case_dir, "trace.csv"), ticks)
            p = manifest.parameters
            s = p["hold_summary_after_first_0p3s"] or {}
            print(f"    recorded backend={manifest.physics_backend}  termination={p['termination_reason']}"
                  f"{'  ' + p['termination_detail'] if p['termination_detail'] else ''}")
            if s:
                print(f"    after jolt: moved {100 * s['net_displacement_m']:.1f} cm net "
                      f"(direction {_fmt(s['motion_direction_rel_heading_deg'], '+.0f')} deg vs heading: "
                      f"0=fwd, 180=back, +90=right), turned {s['heading_change_deg']:+.2f} deg, "
                      f"speed early {_fmt(s['early_mean_speed_mps'], '.3f')} -> late {_fmt(s['late_mean_speed_mps'], '.3f')} m/s, "
                      f"hand_brake readback={p['hand_brake_readback']}")
            results.append(manifest)
            if p["termination_reason"] != "hold_complete":
                exit_code = 1
            if p["termination_reason"] == "exception:KeyboardInterrupt":
                print("[test25] interrupted -- evidence for this case saved; stopping.")
                break
            time.sleep(0 if args.fast or args.pause else 1.5)
    finally:
        report = cleanup_session(world, [], restore_async=True)
        print(f"\n[test25] async restored: {report.get('async_restored')}  errors: {report['errors']}")

    print(f"\n{'case':30} {'backend':32} {'net cm':>7} {'dir deg':>8} {'turn deg':>9} "
          f"{'early v':>8} {'late v':>8} {'pitch jolt':>10} {'ms/tick':>8}")
    for m in results:
        p = m.parameters
        s, full = p["hold_summary_after_first_0p3s"] or {}, p["hold_summary"] or {}
        print(f"{p['case']['name']:30} {m.physics_backend:32} "
              f"{_fmt(s.get('net_displacement_m') and 100 * s['net_displacement_m'], '7.1f')} "
              f"{_fmt(s.get('motion_direction_rel_heading_deg'), '+8.0f')} "
              f"{_fmt(s.get('heading_change_deg'), '+9.2f')} "
              f"{_fmt(s.get('early_mean_speed_mps'), '8.3f')} {_fmt(s.get('late_mean_speed_mps'), '8.3f')} "
              f"{_fmt(full.get('max_abs_pitch_change_deg'), '10.2f')} "
              f"{_fmt(p['mean_wall_tick_s'] and 1000 * p['mean_wall_tick_s'], '8.2f')}")
    print(f"\n[test25] evidence: {run_dir}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
