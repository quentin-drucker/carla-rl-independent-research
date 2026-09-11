"""
spawning.py

Goal:
- Hold spawning helpers for test2 scripts (ego vehicle + optional static obstacle).
- Keep logic and prints identical to the original test2___braking_via_lidar.py spawning block.
"""

import carla
# pyright: reportMissingImports=false


def prepare_spawn_context(world, *, SPAWN_INDEX):
    # -------------------------------------------------
    # spawn ego vehicle 
    # -------------------------------------------------
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    if SPAWN_INDEX < 0 or SPAWN_INDEX >= len(spawn_points):
        raise ValueError(f"SPAWN_INDEX {SPAWN_INDEX} out of range (0..{len(spawn_points)-1})")

    spawn_tf = spawn_points[SPAWN_INDEX]

    bp_lib = world.get_blueprint_library()

    return spawn_points, spawn_tf, bp_lib


def spawn_static_obstacle_by_index(world, *, spawn_points, bp_lib, OBSTACLE_SPAWN_INDEX, OBSTACLE_HOLD_BRAKE):
    # for lidar brake testing--spawn static obstacle ahead
    obstacle = None

    obs_tf = spawn_points[OBSTACLE_SPAWN_INDEX]

    obs_bp = bp_lib.filter("vehicle.*")[0]   # simple: any vehicle
    obstacle = world.try_spawn_actor(obs_bp, obs_tf)

    if obstacle is None:
        print(f"[obstacle] FAILED to spawn at index {OBSTACLE_SPAWN_INDEX} (occupied). Try another index.")
    else:
        if OBSTACLE_HOLD_BRAKE:
            obstacle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        # big visible marker
        world.debug.draw_point(
            obstacle.get_transform().location + carla.Location(z=1.0),
            size=0.3,
            color=carla.Color(255, 0, 0),
            life_time=30.0
        )
        print(f"[obstacle] Spawned id={obstacle.id} at spawn index {OBSTACLE_SPAWN_INDEX}")

    return obstacle


def spawn_ego_vehicle(world, *, bp_lib, spawn_tf):
    # pick a simple vehicle blueprint (model3 is stable)
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]

    vehicle = world.try_spawn_actor(vehicle_bp, spawn_tf)
    if vehicle is None:
        # spawn can fail if something is occupying that spawnpoint
        raise RuntimeError("Failed to spawn ego vehicle at this spawn point. Try a different SPAWN_INDEX.")

    return vehicle