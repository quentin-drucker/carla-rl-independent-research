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


def cleanup_session(world, actors, *, restore_async=True):
    """Destroy actors (sensors stopped first, reverse spawn order), then
    optionally restore async mode and verify it by reading settings back.
    Never raises -- cleanup must always run to completion. Returns a report
    dict for the run manifest: {"destroyed", "errors", "async_restored"}.
    """
    report = {"destroyed": [], "errors": []}
    for actor in reversed(actors):
        try:
            type_id = actor.type_id
            if type_id.startswith("sensor."):
                actor.stop()
            if actor.destroy():
                report["destroyed"].append(type_id)
            else:
                report["errors"].append(f"destroy returned False for {type_id}")
        except Exception as exc:  # noqa: BLE001 -- cleanup must continue
            report["errors"].append(f"destroy failed: {exc!r}")
    if restore_async and world is not None:  # harmless if already async
        try:
            restore_async_mode(world)
            report["async_restored"] = not world.get_settings().synchronous_mode
        except Exception as exc:  # noqa: BLE001
            report["async_restored"] = False
            report["errors"].append(f"restore_async_mode failed: {exc!r}")
    return report
