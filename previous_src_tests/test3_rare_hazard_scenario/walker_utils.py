"""
walker_utils.py
Quentin | Junior Spring Research, 2026.

Goal:
- Helpers for spawning and commanding a pedestrian (walker) for a repeatable "rare scenario".
- This version does NOT rely on CARLA walker AI navigation / NavMesh.
- Instead, it uses deterministic scripted movement along a straight crossing path.

Why:
- Town04_Opt / some route areas may not support walker NavMesh well.
- For research, deterministic movement is actually better anyway:
  same geometry, same timing, same repeatability.
"""

import carla
# pyright: reportMissingImports=false
import math
from typing import Optional


# -------------------------------------------------
# small helpers
# -------------------------------------------------

def _try_spawn_with_small_offsets(world, bp, tf, *, nudge_step_m=0.50, max_tries=12):
    """
    Try to spawn an actor at tf, but if occupied, nudge it forward/back
    along its yaw direction to find a nearby free spot.
    """
    actor = world.try_spawn_actor(bp, tf)
    if actor is not None:
        return actor

    yaw_rad = math.radians(tf.rotation.yaw)
    fwd = carla.Vector3D(x=math.cos(yaw_rad), y=math.sin(yaw_rad), z=0.0)

    for i in range(1, max_tries + 1):
        sign = 1.0 if (i % 2 == 1) else -1.0
        k = (i + 1) // 2
        delta = sign * k * nudge_step_m

        nudged = carla.Transform(
            carla.Location(
                x=tf.location.x + fwd.x * delta,
                y=tf.location.y + fwd.y * delta,
                z=tf.location.z
            ),
            tf.rotation
        )
        actor = world.try_spawn_actor(bp, nudged)
        if actor is not None:
            return actor

    return None


def _yaw_deg_from_locations(a: carla.Location, b: carla.Location) -> float:
    """
    Yaw angle in degrees pointing from a -> b in XY plane.
    """
    dx = b.x - a.x
    dy = b.y - a.y
    return math.degrees(math.atan2(dy, dx))


def _lerp_location(a: carla.Location, b: carla.Location, alpha: float) -> carla.Location:
    """
    Linear interpolation between two CARLA locations.
    alpha=0 -> a
    alpha=1 -> b
    """
    alpha = max(0.0, min(1.0, alpha))
    return carla.Location(
        x=a.x + (b.x - a.x) * alpha,
        y=a.y + (b.y - a.y) * alpha,
        z=a.z + (b.z - a.z) * alpha
    )


# -------------------------------------------------
# scripted walker spawn + crossing state
# -------------------------------------------------

def spawn_scripted_walker(world, *, start_tf: carla.Transform) -> Optional[carla.Actor]:
    """
    Spawn only the pedestrian actor. No AI controller.
    """
    bp_lib = world.get_blueprint_library()
    walker_bp = bp_lib.filter("walker.pedestrian.*")[0]

    walker = _try_spawn_with_small_offsets(world, walker_bp, start_tf)
    if walker is None:
        print("[walker] FAILED to spawn scripted walker.")
        return None

    return walker


def init_scripted_crossing_state(*, start_loc: carla.Location, end_loc: carla.Location, speed_mps: float):
    """
    Small dict holding scripted crossing state.
    """
    dx = end_loc.x - start_loc.x
    dy = end_loc.y - start_loc.y
    dz = end_loc.z - start_loc.z

    dist_xy = math.sqrt(dx * dx + dy * dy)
    if dist_xy < 1e-6:
        dir_x, dir_y = 1.0, 0.0
    else:
        dir_x = dx / dist_xy
        dir_y = dy / dist_xy

    return {
        "triggered": False,
        "done": False,
        "start_loc": start_loc,
        "end_loc": end_loc,
        "speed_mps": float(speed_mps),
        "dir_x": dir_x,
        "dir_y": dir_y,
    }


def trigger_scripted_crossing(world, crossing_state, *, debug_life_s: float = 12.0):
    """
    Mark the crossing as active and draw debug markers.
    """
    if crossing_state is None:
        return

    crossing_state["triggered"] = True
    crossing_state["elapsed_s"] = 0.0

    start_loc = crossing_state["start_loc"]
    end_loc = crossing_state["end_loc"]

    world.debug.draw_point(
        start_loc + carla.Location(z=0.7),
        size=0.18,
        color=carla.Color(255, 0, 0),
        life_time=debug_life_s
    )
    world.debug.draw_point(
        end_loc + carla.Location(z=0.7),
        size=0.18,
        color=carla.Color(0, 255, 0),
        life_time=debug_life_s
    )
    world.debug.draw_line(
        start_loc + carla.Location(z=0.7),
        end_loc + carla.Location(z=0.7),
        thickness=0.06,
        color=carla.Color(255, 255, 0),
        life_time=debug_life_s
    )


def update_scripted_crossing(walker: Optional[carla.Actor], crossing_state, *, fixed_dt: float):
    """
    Advance scripted crossing using WalkerControl instead of teleporting.
    This keeps the walker on the surface much better than set_transform().
    """
    if walker is None or crossing_state is None:
        return

    if not crossing_state["triggered"] or crossing_state["done"]:
        return

    end_loc = crossing_state["end_loc"]
    speed_mps = crossing_state["speed_mps"]

    walker_loc = walker.get_transform().location

    # stop once close enough to destination (XY only — end_loc has a z offset
    # for spawning that differs from the walker's settled ground-level z, so
    # 3D distance would never drop below threshold and the ped would overshoot)
    dist_to_goal_xy = math.sqrt(
        (walker_loc.x - end_loc.x) ** 2 + (walker_loc.y - end_loc.y) ** 2
    )
    if dist_to_goal_xy <= 0.35:
        walker.apply_control(carla.WalkerControl(           # type: ignore[attr-defined]
            direction=carla.Vector3D(x=0.0, y=0.0, z=0.0),
            speed=0.0,
            jump=False
        ))
        crossing_state["done"] = True
        return

    direction = carla.Vector3D(
        x=crossing_state["dir_x"],
        y=crossing_state["dir_y"],
        z=0.0
    )

    walker.apply_control(carla.WalkerControl(               # type: ignore[attr-defined]
        direction=direction,
        speed=speed_mps,
        jump=False
    ))


def destroy_scripted_walker(walker: Optional[carla.Actor]):
    """
    Clean destroy for scripted walker.
    """
    if walker is not None:
        walker.destroy()