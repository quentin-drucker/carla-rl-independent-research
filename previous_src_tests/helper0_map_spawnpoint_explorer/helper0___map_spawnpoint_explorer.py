"""
helper-map_spawnpoint_explorer.py
Quentin

Purpose of this helper script:
- Print all maps available in this CARLA build.
- Visualize the spawn points IN the world by drawing their indices in 3D space.
  (So I can fly around in spectator mode and pick a good curvy road spawn.)
- Draw a large fixed ground grid for scale reference (each square = 1 meter).
- Draw world axes (+X, +Y, +Z) at the grid origin for orientation.
"""

import carla
# pyright: reportMissingImports=false
import time


def draw_world_axes(world, origin, length=10.0, life_time=60.0):
    """
    Draw CARLA world axes at a given origin.
    Red: +X, Green: +Y, Blue: +Z
    """
    x_end = origin + carla.Location(x=length, y=0, z=0)
    y_end = origin + carla.Location(x=0, y=length, z=0)
    z_end = origin + carla.Location(x=0, y=0, z=length)

    world.debug.draw_line(origin, x_end, thickness=0.15, color=carla.Color(255, 0, 0), life_time=life_time)
    world.debug.draw_line(origin, y_end, thickness=0.15, color=carla.Color(0, 255, 0), life_time=life_time)
    world.debug.draw_line(origin, z_end, thickness=0.15, color=carla.Color(0, 0, 255), life_time=life_time)

    world.debug.draw_string(x_end, "+X", color=carla.Color(255, 0, 0), life_time=life_time)
    world.debug.draw_string(y_end, "+Y", color=carla.Color(0, 255, 0), life_time=life_time)
    world.debug.draw_string(z_end, "+Z", color=carla.Color(0, 0, 255), life_time=life_time)


def draw_big_ground_grid(world, *,
                         origin=carla.Location(0.0, 0.0, 0.0),
                         half_size=200.0,     # meters (grid spans ~400m x 400m)
                         spacing=1.0,         # meters per cell (spacing=1.0 => each square is 1m)
                         life_time=60.0):
    """
    Draw a large, fixed ground grid in world space.

    NOTE ON SCALE:
    - spacing=1.0 means each grid cell is 1 meter x 1 meter (CARLA locations are in meters).
    - every 10th line is thicker/brighter so you can count 10m chunks quickly.

    This grid is fixed in world coordinates (it does NOT follow the spectator).
    If you fly far away from (0,0), you may leave the grid region.
    """
    # Lift slightly to reduce z-fighting with the road
    z = origin.z + 0.05

    n = int(half_size / spacing)
    for i in range(-n, n + 1):
        offset = i * spacing

        # Lines parallel to Y (vary X)
        a = carla.Location(origin.x + offset, origin.y - half_size, z)
        b = carla.Location(origin.x + offset, origin.y + half_size, z)

        # Lines parallel to X (vary Y)
        c = carla.Location(origin.x - half_size, origin.y + offset, z)
        d = carla.Location(origin.x + half_size, origin.y + offset, z)

        major = (i % 10 == 0)  # every 10m if spacing=1m
        col = carla.Color(200, 200, 200) if major else carla.Color(100, 100, 100)
        thick = 0.07 if major else 0.03

        world.debug.draw_line(a, b, thickness=thick, color=col, life_time=life_time)
        world.debug.draw_line(c, d, thickness=thick, color=col, life_time=life_time)

    world.debug.draw_string(
        carla.Location(origin.x, origin.y, z + 0.8),
        f"GRID: spacing={spacing}m (each square is {spacing}m)",
        color=carla.Color(255, 255, 255),
        life_time=life_time
    )


def main():
    # ---- CONNECT TO CARLA ----
    print("Connecting to CARLA...")

    simulation_time = 60  # seconds

    # If you want to force a specific map instead of whatever CARLA currently has loaded:
    # NOTE: Copy/paste a string EXACTLY from client.get_available_maps().
    # Example:
    # target_map = "/Game/Carla/Maps/Town03"
    # target_map = "/Game/Carla/Maps/Town10HD_Opt"
    target_map = "Town04_Opt"  # Set to None to use whatever map is currently loaded.

    # Map NOTE:
    # - "Town03_Opt" spawnpoint 165 leads to a roundabout which could be an option for lane following.
    # - "Town04_Opt" has spawnpoints 158 and 118 which are curved highway off ramps -- could be an option.

    # This 'client' object is the Python handle to the CARLA server (CarlaUE4.exe).
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    # ---- MAP DISCOVERY ----
    print("\n=== AVAILABLE MAPS (copy-pasteable paths) ===")
    maps = client.get_available_maps()

    full_paths = []
    for i, m in enumerate(maps):
        full_path = f"Carla/Maps/{m}"
        full_paths.append(full_path)
        print(f"[{i:2d}] {m:<20} -> {full_path}")

    # ---- LOAD TARGET MAP (optional) ----
    if target_map is not None:
        print("\nLoading target map:", target_map)
        world = client.load_world(target_map)
    else:
        world = client.get_world()

    # Connection proof
    print("\nConnected to:", world.get_map().name)

    # ---- UNIVERSAL SCALE VISUALIZATION (fixed in world space) ----
    # Each square = 1 meter (spacing=1.0)
    grid_origin = carla.Location(0.0, 0.0, 0.0)
    draw_big_ground_grid(
        world,
        origin=grid_origin,
        half_size=200.0,
        spacing=1.0,                 # ✅ each grid cell is 1 meter
        life_time=simulation_time
    )
    draw_world_axes(world, grid_origin, length=15.0, life_time=simulation_time)
    print("✅ Drew fixed world grid (1m squares) + world axes at (0,0,0).")

    # ---- SPAWNPOINT LIST + VISUALIZATION ----
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    print(f"\nSpawn points found: {len(spawn_points)}")

    for i, sp in enumerate(spawn_points):
        world.debug.draw_string(
            sp.location,
            str(i),
            draw_shadow=False,
            color=carla.Color(255, 255, 0),
            life_time=simulation_time,
            persistent_lines=False
        )

    print(f"\n✅ Spawn point indices drawn for {simulation_time} seconds.")
    print("Fly around in spectator mode and note good spawn point indices.")
    print("Grid note: each small square is 1 meter, major lines are every 10 meters.\n")

    time.sleep(simulation_time)


if __name__ == "__main__":
    main()