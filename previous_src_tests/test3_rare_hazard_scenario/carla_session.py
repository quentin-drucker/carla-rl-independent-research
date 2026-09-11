"""
carla_session.py

Goal:
- Hold CARLA session setup/teardown helpers (connect, load world, apply sync, restore async).
- Keep behavior identical to test2___braking_via_lidar.py (no logic changes, just modularization).
"""

import carla
# pyright: reportMissingImports=false

def connect_and_load_world(*, host, port, timeout_s, target_map):
    client = carla.Client(host, port)
    client.set_timeout(timeout_s)
    world = client.load_world(target_map)
    print(f"[carla] Connected | map={world.get_map().name}")
    return client, world


def enable_sync_mode(world, *, fixed_dt):
    # -------------------------------------------------
    # Set synchronous stepping (repeatable) 
    # -------------------------------------------------
    # synchronous mode = I control when sim advances (world.tick())
    # fixed_delta_seconds = every tick is exactly FIXED_DT seconds
    #
    # this makes control stable and also helps later for RL episodes
    settings = world.get_settings()
    settings.synchronous_mode = True  # again, this means I call world.tick() to advance sim manually, so this isn't a real-time run
    settings.fixed_delta_seconds = fixed_dt
    world.apply_settings(settings)


def restore_async_mode(world):
    # restore world settings back to normal async mode
    # (CARLA gets weird if you leave sync on and then run other scripts)
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)