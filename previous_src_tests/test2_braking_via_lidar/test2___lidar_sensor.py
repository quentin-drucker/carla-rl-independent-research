"""
test2___lidar_sensor.py

Goal:
- Attach and configure a LiDAR sensor on the ego vehicle.
- Keep logic identical to the original test2___braking_via_lidar.py LiDAR setup block.
"""

import carla
# pyright: reportMissingImports=false
import queue


def attach_lidar_sensor(world, bp_lib, vehicle):
    # -------------------------------------------------
    # Attach and configure LiDAR sensor
    # -------------------------------------------------
    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")

    # Starter LiDAR configuration (tune later)
    lidar_bp.set_attribute("range", "50.0")
    lidar_bp.set_attribute("rotation_frequency", "50.0")   # NOTE: Important--should match world tick rate (currently dt=0.02, so 50.0 Hz)
    lidar_bp.set_attribute("channels", "32")
    lidar_bp.set_attribute("points_per_second", "200000")
    lidar_bp.set_attribute("upper_fov", "10.0")
    lidar_bp.set_attribute("lower_fov", "-30.0")

    lidar_tf = carla.Transform(carla.Location(x=0.0, z=2.0))  # roof-ish
    lidar = world.spawn_actor(lidar_bp, lidar_tf, attach_to=vehicle)

    # using imported queue library to handle lidar data:
    lidar_queue = queue.Queue()

    def on_lidar(data): # callback function to store lidar data in the queue
        lidar_queue.put(data) # put the latest lidar data into the queue

    lidar.listen(on_lidar) # start listening to lidar data with the callback

    return lidar, lidar_queue