# -- Note on how to run -- 
# Make sure CARLA is running and map (like "Town10HD") is loaded. From the venv terminal, run: python <filename>.py      (so in this case: python test-first_ego_camera.py)


import carla # import carla loads CARLA's Python API module (the library that contains Client, World, Transform, sensors, etc.)
# pyright: reportMissingImports=false
import time # import time loads Python's standard time module, used for time.sleep(seconds)
from pathlib import Path
import random # to choose a random spawn point

def main():
    # 1) Connect to the running CARLA server
    client = carla.Client("localhost", 2000)    # creates connection object to CARLA server (locally) on port 2000.
    client.set_timeout(10.0)                    # how long the client will wait before giving up on a request (in seconds)
    world = client.get_world()                  # returns a World object representing the currently loaded map and everything in it (actors, sensors, settings, etc.).

    bp_lib = world.get_blueprint_library()      # returns a BlueprintLibrary object--is basically a template you can spawn (vehicles, pedestrians, sensors, traffic lights, etc.)
    actors = []  # keep track of actors so we can destroy everything cleanly at the end -- otherwise there'll be "ghost actors" accumulating in the world

    out_dir = Path("out") / "rgb"
    out_dir.mkdir(parents=True, exist_ok=True)

    simulation_duration = 20  # seconds
    max_saves = 20  # maximum number of images to save
    save_target_interval = simulation_duration / max_saves  # save an image every N seconds (for 20 sim time & 20 max saves, this is ~1.00s)
    saved = {"count": 0} # to track number of saved images
    last_saved_time = {"t": None}  # to track last save time

    try:
        # 2) Spawn an ego vehicle at a random spawn point
        vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0] 
            # returns a list of blueprints whose ID matches that pattern -- Result is a list; we take [0] to grab first match 
            # ("vehicle.tesla.model3" is a specific blueprint ID.)
        
        # Option 1: For hardcoded spawn point, uncomment below: 
        # spawn_point = world.get_map().get_spawn_points()[0] # selects first spawnpoint in the list

        # Option 2: For random spawn point, use below:
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)    
            # world.get_map() returns a Map object (roads, lanes, spawn points, waypoints).
            # map.get_spawn_points() returns a list of recommended car spawn transforms (locations + rotations). -- Each spawnpoint is a carla.Transform
                # --> in this case we randomly select a spawn point
        
        ego = world.spawn_actor(vehicle_bp, spawn_point) 
            # world.spawn_actor(blueprint, transform) spawns the actor in the world according to transform orientation (location/rotation).      
            # returns an Actor object. for vehicles it'll be a carla.Vehicle object (a subclass of Actor).

        actors.append(ego) # Adds it to the list for later cleanup.

        # Teleport spectator to hover above ego:
        spectator = world.get_spectator()
        spectator.set_transform(carla.Transform(
            ego.get_transform().location + carla.Location(z=30),
            carla.Rotation(pitch=-90)
        ))
            # Moves the "spectator" camera to hover above the ego vehicle, looking straight down.

        # Option 1: set autopilot on:
        ego.set_autopilot(True)
        print("✓ Autopilot enabled.")

        # Option 2: keep it still for now (no autopilot):
        # ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            # vehicle.apply_control(control) --> Sends a control command to the vehicle for the next simulation step.
            # carla.VehicleControl(...) parameters (common ones):
                # throttle (0.0 to 1.0) - how much gas
                # brake (0.0 to 1.0) - how much braking
                # steer (-1.0 to 1.0) - left/right
                # hand_brake (bool)
                # reverse (bool)


        # 3) Attach an RGB camera to the ego
        cam_bp = bp_lib.find("sensor.camera.rgb")
            # Finds exactly one blueprint by exact ID. -- returns a Blueprint
        
        cam_bp.set_attribute("image_size_x", "800")
        cam_bp.set_attribute("image_size_y", "600")
        cam_bp.set_attribute("fov", "90")
            # Sets blueprint properties/attributes before spawning the Actor.
                # For RGB camera: 
                    # image_size_x (width in pixels)
                    # image_size_y (height in pixels)
                    # fov (field of view in degrees)



        cam_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4),  # hood/roof-ish mount
            carla.Rotation(pitch=-10) 
)   
            # Orienting camera in world via Transform
                # carla.Location(x, y, z)
                    # (Position in meters)  
                    # for an attached sensor, this is relative to vehicle's origin (0,0,0)
                    # x=1.5 means forward 1.5m (toward the hood)
                    # z=2.4 means up 2.4m (up toward roof-ish level)
                    # y defaults to 0 if not provided (centered line)
                # carla.Rotation(pitch, yaw, roll)
                    # (Orientation in degrees)
                    # pitch=-10 means camera is tilted slightly downward (10 degrees)
                    # yaw and roll default to 0 if not provided
                # carla.Transform(location, rotation)
                    # bundle location and rotation into one object


        camera = world.spawn_actor(cam_bp, cam_transform, attach_to=ego) # world.spawn_actor(blueprint, transform, attach_to=actor)
        actors.append(camera)
            # Same spawn method as vehicle, but with attach_to=ego so it follows the car.

        # 4) Save frames to disk to see it's working
        def on_image(image):
            # If we have already saved the maximum number of images, do nothing and return immediately.
            if saved["count"] >= max_saves:
                print(f"Reached max saves of {max_saves}")
                return

            now = time.time() # Get the current real-world time (in seconds)
            # If this is the very first image received, initialize the "last saved" timestamp
            if last_saved_time["t"] is None:
                last_saved_time["t"] = now  # start the timer on the first frame

            # If not enough time has passed since the last saved image, skip this frame (regardless of FPS or frame count)
            if (now - last_saved_time["t"]) < save_target_interval:
                return

            # Construct the output filename using image.frame keeps filenames unique across runs
            filename = out_dir / f"{saved['count']:06d}.png"
            image.save_to_disk(str(filename)) # save image to disk
            saved["count"] += 1               # increment saved count
            last_saved_time["t"] = now        # update last saved time


        camera.listen(on_image)

        # <OLD LINE; COMMENTED OUT FOR NOW> camera.listen(lambda image: image.save_to_disk("out/rgb/%06d.png" % image.frame))
            # camera.listen(callback) 
                # Starts the sensor stream --> CARLA will call callback(image) every time the camera produces a new frame.
            # lambda image: ...
                # just a little anonymous function meaning "when an image arrives, do this".
                # note: `image` is a carla.Image object.
            # image.save_to_disk(path_pattern)
                # Saves the image as a PNG (or other supported formats depending on sensor).
                # "%06d" means zero-pad the frame number to 6 digits.
                # image.frame is the simulation frame count.
                    # (filenames will be like: 000001.png, 000002.png, etc.)
        
        print(f"✓ Ego + RGB camera spawned. Capturing for {simulation_duration} seconds...")
        time.sleep(simulation_duration) # This "sleep" just gives the sensor time to produce frames
        print("✓ Done. Check: out/rgb/")

    finally:
        # 5) Cleanup: stop sensors first, then destroy everything
        for a in actors:
            if a.type_id.startswith("sensor."):
                a.stop()
        for a in actors:
            a.destroy()
        # a.type_id
            # A string ID like "sensor.camera.rgb" or "vehicle.tesla.model3".
        # sensor.stop()
            # Stops the sensor from sending data (important to call before destroying).
        # actor.destroy()
            # Removes the actor from the CARLA simulation world (and frees resources).
        print("Cleaned up actors.")

if __name__ == "__main__":
    main()