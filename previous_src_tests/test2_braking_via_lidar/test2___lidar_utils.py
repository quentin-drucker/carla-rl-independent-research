"""
test2___lidar_utils.py

Goal:
- Hold LiDAR helper functions used by test2 scripts.
- Keep logic identical to the original test2___braking_via_lidar.py.
"""

import carla
# pyright: reportMissingImports=false
import math

from test2___math_utils import clamp


# ---          ---          ---
# Lidar helper functions 
# ---          ---          ---

def lidar_min_distance_ahead(lidar_data,
                             cone_half_angle_deg, # only consider points within a forward cone (e.g. 10° to left and right) since we only care about obstacles in our lane/path ahead, not adjacent lanes or behind us
                             cone_yaw_deg,   # rotate cone centerline left/right (used when steering to look for hazards in the direction we're turning toward, not just straight ahead)
                             z_min,        # ignore ground-ish points (sensor-local)
                             z_max,        # ignore super-high points
                             max_dist,
                             x_min,       # ignore extremely near points (self/roof/noise)
                             max_lateral_m=None, # optional: ignore points that are too far left/right from the centerline (e.g. if we want to focus on points in our lane and ignore adjacent lanes, environmental distractions around the road)
                             ):
    """
    Returns min distance of LiDAR points that are:
    - inside a (possibly rotated) forward cone around angle = cone_yaw_deg
    - within vertical band z_min..z_max (sensor coords)
    - within max_dist
    - not too close (x_along > x_min) to ignore self/noise

    Sensor frame conventions (CARLA LiDAR):
    - +x forward, +y right, +z up (sensor-local)
    """
    if lidar_data is None:
        return None

    half_rad = math.radians(cone_half_angle_deg)
    yaw_rad  = math.radians(cone_yaw_deg)
    tan_theta = math.tan(half_rad)

    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)

    min_d = None

    for p in lidar_data:
        x = p.point.x
        y = p.point.y
        z = p.point.z

        # Z-band filter
        if z < z_min or z > z_max:
            continue

        # Rotate point into "cone frame" whose +x axis is the cone centerline.
        # x_along = forward component along the rotated cone axis
        # y_lat   = lateral offset from cone axis
        x_along = x * cos_y + y * sin_y
        y_lat   = -x * sin_y + y * cos_y

        if x_along <= x_min:
            continue

        # lane/corridor cap (applies after rotation into cone frame)
        if (max_lateral_m is not None) and (abs(y_lat) > max_lateral_m):
            continue


        # Cone check around rotated axis
        if abs(y_lat) > x_along * tan_theta:
            continue

        d = math.sqrt(x*x + y*y + z*z)
        if d > max_dist:
            continue

        if (min_d is None) or (d < min_d):
            min_d = d

    return min_d

def smooth_brake_from_distance(d_min, speed_mps, *,
                              buffer_m=6.0,
                              headway_s=1.2,
                              d_stop_hard=6.0):
    """
    Convert distance-to-obstacle into a brake command [0,1].

    - buffer_m + speed*headway_s defines when we START braking smoothly.
    - if d_min <= d_stop_hard we jump to emergency braking.
    """
    if d_min is None:
        return 0.0

    # emergency region
    if d_min <= d_stop_hard:
        return 1.0

    # start braking when below this "desired clearance"
    d_start = buffer_m + speed_mps * headway_s

    if d_min >= d_start:
        return 0.0

    # Map distance in (d_stop_hard .. d_start) -> brake in (1 .. 0) smoothly
    # normalized closeness: 0 far, 1 very close
    closeness = (d_start - d_min) / max(1e-3, (d_start - d_stop_hard))
    closeness = clamp(closeness, 0.0, 1.0)

    # Smooth curve so it feathers in (square is a nice simple easing)
    brake_cmd = closeness * closeness
    return clamp(brake_cmd, 0.0, 1.0)

def draw_lidar_points(world, lidar_data, lidar_actor, sample_step=60, life_time=0.06):
    """
    Visualize LiDAR by drawing a subset of points in the CARLA world.

    - sample_step: draw 1 out of every N points (higher = faster, fewer points)
    - life_time: how long each dot stays visible (seconds). Keep small for a "live" look.
    """
    if lidar_data is None:
        return

    lidar_tf = lidar_actor.get_transform()

    i = 0
    for p in lidar_data: # iterate over LiDAR points; each point has .point (x,y,z) in sensor coords
        if i % sample_step == 0: # NOTE: only draw some points to reduce clutter and improve performance
            pt = p.point  # sensor-local coordinates (x forward, y right, z up)

            # Convert sensor-local point into world coordinates
            world_loc = lidar_tf.transform(carla.Location(x=pt.x, y=pt.y, z=pt.z))

            world.debug.draw_point(
                world_loc,
                size=0.075,
                color=carla.Color(0, 0, 255),  # blue points for LiDAR
                life_time=life_time
            )
        i += 1

def draw_hazard_cone(world, lidar_actor, *, cone_yaw_deg, cone_half_angle_deg, length_m, life_time):
    """
    Draws 2 lines representing the cone boundaries in world space.
    The cone is defined in the LiDAR sensor frame (+x forward, +y right).
    """
    if lidar_actor is None:
        return

    tf = lidar_actor.get_transform()
    origin = carla.Location(
        x=tf.location.x,
        y=tf.location.y,
        z=tf.location.z - 1.5   # lidar initially set to 2 (meters) so lower the origin to draw cone vision closer to ground.
    )


    fwd = tf.get_forward_vector()
    right = tf.get_right_vector()

    # boundary angles relative to sensor forward
    a0 = math.radians(cone_yaw_deg - cone_half_angle_deg)
    a1 = math.radians(cone_yaw_deg + cone_half_angle_deg)

    # boundary direction vectors in world space: dir = fwd*cos(a) + right*sin(a)
    dir0 = carla.Vector3D(
        x=fwd.x * math.cos(a0) + right.x * math.sin(a0),
        y=fwd.y * math.cos(a0) + right.y * math.sin(a0),
        z=fwd.z * math.cos(a0) + right.z * math.sin(a0),
    )
    dir1 = carla.Vector3D(
        x=fwd.x * math.cos(a1) + right.x * math.sin(a1),
        y=fwd.y * math.cos(a1) + right.y * math.sin(a1),
        z=fwd.z * math.cos(a1) + right.z * math.sin(a1),
    )

    end0 = carla.Location(
        x=origin.x + dir0.x * length_m,
        y=origin.y + dir0.y * length_m,
        z=origin.z + dir0.z * length_m,
    )
    end1 = carla.Location(
        x=origin.x + dir1.x * length_m,
        y=origin.y + dir1.y * length_m,
        z=origin.z + dir1.z * length_m,
    )

    # draw the two boundary rays
    world.debug.draw_line(origin, end0, thickness=0.04, color=carla.Color(80, 120, 160), life_time=life_time) #muted blue-gray
    world.debug.draw_line(origin, end1, thickness=0.04, color=carla.Color(80, 120, 160), life_time=life_time) #muted blue-gray

    # optional: draw the centerline too (makes it easier to read)
    ac = math.radians(cone_yaw_deg)
    dirc = carla.Vector3D(
        x=fwd.x * math.cos(ac) + right.x * math.sin(ac),
        y=fwd.y * math.cos(ac) + right.y * math.sin(ac),
        z=fwd.z * math.cos(ac) + right.z * math.sin(ac),
    )
    endc = carla.Location(
        x=origin.x + dirc.x * length_m,
        y=origin.y + dirc.y * length_m,
        z=origin.z + dirc.z * length_m,
    )
    world.debug.draw_line(origin, endc, thickness=0.03, color=carla.Color(90, 90, 90), life_time=life_time) # subtle gray


def draw_hazard_capped_cone(world, lidar_actor, *, cone_yaw_deg, cone_half_angle_deg, max_lateral_m, length_m, life_time):
    """
    Draw a "capped cone":
    - near field: true cone
    - far field: hard lateral cap at +/- max_lateral_m (meters), forming a corridor
    This matches the max_lateral_m filter used in lidar_min_distance_ahead.
    """
    if lidar_actor is None:
        return

    tf = lidar_actor.get_transform()
    origin = carla.Location(
        x=tf.location.x,
        y=tf.location.y,
        z=tf.location.z - 1.5
    )

    fwd = tf.get_forward_vector()
    right = tf.get_right_vector()

    half_rad = math.radians(cone_half_angle_deg)
    tan_theta = math.tan(half_rad)

    # Where does the cone boundary reach the lateral cap?
    # abs(y) = x * tan(theta)  =>  x = max_lateral / tan(theta)
    # If tan(theta) ~ 0, just fall back to normal cone drawing behavior.
    if abs(tan_theta) < 1e-6:
        draw_hazard_cone(world, lidar_actor,
                        cone_yaw_deg=cone_yaw_deg,
                        cone_half_angle_deg=cone_half_angle_deg,
                        length_m=length_m,
                        life_time=life_time)
        return

    x_cap = max_lateral_m / tan_theta
    x_cap = max(0.0, min(x_cap, length_m))

    # Helper: convert (x_local, y_local) in "cone frame" into world location.
    # Cone frame is rotated by cone_yaw_deg relative to sensor forward.
    yaw_rad = math.radians(cone_yaw_deg)
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)

    def to_world(xf, yf):
        # rotate from cone-frame -> sensor frame
        x_s = xf * cos_y - yf * sin_y
        y_s = xf * sin_y + yf * cos_y

        return carla.Location(
            x=origin.x + fwd.x * x_s + right.x * y_s,
            y=origin.y + fwd.y * x_s + right.y * y_s,
            z=origin.z + fwd.z * x_s + right.z * y_s,
        )

    # Points:
    # Start at origin, go out along cone edges until x_cap
    left_cap  = to_world(x_cap, -max_lateral_m)
    right_cap = to_world(x_cap, +max_lateral_m)

    # Then extend corridor edges out to length_m at fixed +/- max_lateral_m
    left_end  = to_world(length_m, -max_lateral_m)
    right_end = to_world(length_m, +max_lateral_m)

    # Draw:
    col = carla.Color(80, 120, 160)  # same muted blue-gray as your cone
    world.debug.draw_line(origin, left_cap,  thickness=0.04, color=col, life_time=life_time)
    world.debug.draw_line(origin, right_cap, thickness=0.04, color=col, life_time=life_time)
    world.debug.draw_line(left_cap, left_end,  thickness=0.04, color=col, life_time=life_time)
    world.debug.draw_line(right_cap, right_end, thickness=0.04, color=col, life_time=life_time)

    # Centerline (optional, but matches your existing cone draw style)
    center_end = to_world(length_m, 0.0)
    world.debug.draw_line(origin, center_end, thickness=0.02, color=col, life_time=life_time)