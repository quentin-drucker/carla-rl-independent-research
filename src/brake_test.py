"""
brake_test.py

Standalone braking diagnostic: measures stopping distance and time when the
ego vehicle applies 100% brake from a given cruise speed.

Tests the absolute physical braking limits of the vehicle under different:
  - Initial speeds (mph)
  - Tire friction coefficients (None = CARLA default dry asphalt ~3.5)

Note on ABS: CARLA's default vehicle physics does NOT model ABS.
At low friction coefficients, wheels may lock and produce longer stopping
distances than an ABS-equipped vehicle would achieve.

Usage:
    python brake_test.py

CARLA must be running before you run this script.
"""

import time
import carla

# ---------------------------------------------------------------
# Test matrix — edit these to change what gets tested
# ---------------------------------------------------------------
TEST_SPEEDS_MPH   = [25.0, 35.0, 45.0]
TEST_FRICTIONS    = [None, 3.5, 2.0, 1.0, 0.5]
# None = don't touch physics (CARLA's own default, typically ~3.5 dry)

FRICTION_LABELS = {
    None: "default (CARLA)",
    3.5:  "dry asphalt  (~3.5)",
    2.0:  "wet road     (~2.0)",
    1.0:  "very wet/ice (~1.0)",
    0.5:  "icy road     (~0.5)",
}

# ---------------------------------------------------------------
# Sim constants
# ---------------------------------------------------------------
HOST         = "localhost"
PORT         = 2000
FIXED_DT     = 0.02
SPAWN_INDEX  = 242
TARGET_MAP   = "Town04_Opt"
MPH_TO_MPS   = 0.44704
STOPPED_MPS  = 0.15   # speed threshold to declare "stopped"
MAX_TICKS    = 2000   # safety cap (~40 seconds)


def _apply_friction(vehicle: carla.Actor, friction: float):
    phys = vehicle.get_physics_control()  # type: ignore[attr-defined]
    wheels = phys.wheels          # get the list
    for w in wheels:
        w.tire_friction = friction
    phys.wheels = wheels          # must reassign back — CARLA doesn't see in-place mutations
    vehicle.apply_physics_control(phys)  # type: ignore[attr-defined]


def _get_speed_mps(vehicle: carla.Actor) -> float:
    v = vehicle.get_velocity()  # type: ignore[attr-defined]
    return (v.x**2 + v.y**2 + v.z**2) ** 0.5


def run_brake_test(
    world: carla.World,
    spawn_tf: carla.Transform,
    bp_lib,
    target_mph: float,
    friction,
) -> dict:
    """
    Spawns vehicle, accelerates to target_mph, then full-brakes.
    Returns dict with stop_time_s and stop_distance_m.
    """
    target_mps = target_mph * MPH_TO_MPS

    # Spawn
    bp = bp_lib.find("vehicle.tesla.model3")
    vehicle = world.spawn_actor(bp, spawn_tf)
    world.tick()

    # Apply friction override
    if friction is not None:
        _apply_friction(vehicle, friction)
        world.tick()

    # --- Phase 1: accelerate to target speed ---
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=0.0)) # pyright: ignore
    for _ in range(MAX_TICKS):
        world.tick()
        if _get_speed_mps(vehicle) >= target_mps * 0.97:
            break

    # --- Phase 2: full brake ---
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0)) # pyright: ignore

    start_loc   = vehicle.get_location()
    stop_ticks  = 0
    dist_m      = 0.0
    prev_loc    = start_loc

    for tick in range(MAX_TICKS):
        world.tick()
        loc_now = vehicle.get_location()
        dist_m += loc_now.distance(prev_loc)
        prev_loc = loc_now
        stop_ticks += 1
        if _get_speed_mps(vehicle) < STOPPED_MPS:
            break

    stop_time_s = stop_ticks * FIXED_DT

    vehicle.destroy()
    world.tick()

    return {
        "target_mph":    target_mph,
        "friction":      friction,
        "stop_time_s":   stop_time_s,
        "stop_dist_m":   dist_m,
        "stop_dist_ft":  dist_m * 3.28084,
    }


def main():
    # Connect
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)
    world = client.load_world(TARGET_MAP)

    # Sync mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DT
    world.apply_settings(settings)

    # Spawn context
    bp_lib       = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    spawn_tf     = spawn_points[SPAWN_INDEX]

    results = []
    total = len(TEST_SPEEDS_MPH) * len(TEST_FRICTIONS)
    i = 0
    for speed in TEST_SPEEDS_MPH:
        for friction in TEST_FRICTIONS:
            i += 1
            label = FRICTION_LABELS.get(friction, str(friction))
            print(f"  [{i}/{total}] {speed:.0f} mph | friction={label} ...", end=" ", flush=True)
            r = run_brake_test(world, spawn_tf, bp_lib, speed, friction)
            results.append(r)
            print(f"stop in {r['stop_time_s']:.2f}s over {r['stop_dist_m']:.1f}m ({r['stop_dist_ft']:.0f}ft)")
            time.sleep(0.3)  # brief pause between tests

    # Restore async
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    # Print table
    print("\n" + "=" * 72)
    print(f"{'Speed':>8}  {'Friction':>24}  {'Stop time':>10}  {'Dist (m)':>10}  {'Dist (ft)':>10}")
    print("-" * 72)
    for r in results:
        label = FRICTION_LABELS.get(r["friction"], str(r["friction"]))
        print(
            f"{r['target_mph']:>7.0f}mph  {label:>24}  "
            f"{r['stop_time_s']:>9.2f}s  {r['stop_dist_m']:>9.1f}m  {r['stop_dist_ft']:>9.0f}ft"
        )
    print("=" * 72)
    print("\nNote: No ABS modeled. Low friction → wheel lock → longer stopping distance.")


if __name__ == "__main__":
    main()
