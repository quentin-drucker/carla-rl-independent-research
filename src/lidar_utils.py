"""
lidar_utils.py

Goal:
- Hold LiDAR helper functions used by test2 scripts.
- Keep logic identical to the original test2___braking_via_lidar.py.
"""

import carla
# pyright: reportMissingImports=false
import math

from math_utils import clamp
from route_lateral_control import route_unit_right_xy


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


# -------------------------------------------------
# NEW: Lane-following "noodle corridor" LiDAR hazard filter
# -------------------------------------------------

def _project_point_to_segment_2d(px, py, ax, ay, bx, by):
    """
    Project point P onto segment AB in XY plane.
    Returns: (t_clamped, qx, qy, dist2)
      - t_clamped in [0,1] is the segment interpolation
      - Q = A + t*(B-A) is closest point on segment
      - dist2 is squared lateral distance from P to segment
    """
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    denom = abx * abx + aby * aby
    if denom < 1e-9:
        # degenerate segment
        qx, qy = ax, ay
        dx = px - qx
        dy = py - qy
        return 0.0, qx, qy, dx * dx + dy * dy

    t = (apx * abx + apy * aby) / denom
    t = max(0.0, min(1.0, t))
    qx = ax + t * abx
    qy = ay + t * aby

    dx = px - qx
    dy = py - qy
    return t, qx, qy, dx * dx + dy * dy


def lidar_min_distance_in_lane_noodle(
    world,
    vehicle,
    lidar_actor,
    lidar_data,
    *,
    lookahead_m: float,
    step_m: float,
    half_width_m: float,
    z_min: float,
    z_max: float,
    max_dist_m: float,
    x_min_m: float,
):
    """
    Lane-following corridor ("noodle") hazard check.

    Idea:
    - Build a polyline along the *driving lane centerline* ahead of the ego using CARLA waypoints.
    - Transform each LiDAR point into world space.
    - Keep points that:
        (A) are within a lateral half-width of the polyline in XY, AND
        (B) are within [x_min_m, max_dist_m] along the polyline distance from the ego waypoint, AND
        (C) within z_min..z_max in sensor-local coordinates (same as your old filter).
    - Return the minimum Euclidean distance from the LiDAR origin to any kept point (world space).

    This fixes curve false-positives because "ahead" is defined by the lane centerline,
    not by a straight cone from the car.
    """
    if lidar_data is None or lidar_actor is None or vehicle is None:
        return None, None  # (d_min, noodle_points)

    carla_map = world.get_map()
    ego_loc = vehicle.get_transform().location
    ego_wp = carla_map.get_waypoint(
        ego_loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    if ego_wp is None:
        return None, None

    # --- build waypoint polyline ahead ---
    # We include the ego_wp point as s=0 anchor.
    pts = []  # list of (x, y)
    s_vals = []  # cumulative distance along noodle for each point
    pts.append((ego_wp.transform.location.x, ego_wp.transform.location.y))
    s_vals.append(0.0)

    traveled = 0.0
    wp = ego_wp
    while traveled < max_dist_m:
        nxt = wp.next(step_m)
        if not nxt:
            break
        wp = nxt[0]
        wloc = wp.transform.location
        traveled += step_m
        pts.append((wloc.x, wloc.y))
        s_vals.append(traveled)

    # Need at least 2 points to form a segment
    if len(pts) < 2:
        return None, None

    lidar_tf = lidar_actor.get_transform()
    lidar_origin = lidar_tf.location

    min_d = None

    # For debug drawing, return the noodle polyline points (world locations)
    noodle_world_points = [carla.Location(x=p[0], y=p[1], z=lidar_origin.z - 1.5) for p in pts]

    # --- evaluate LiDAR points ---
    for p in lidar_data:
        # Keep your SAME vertical filtering in sensor frame (cheap + avoids overhead)
        z = p.point.z
        if z < z_min or z > z_max:
            continue

        # Ignore extremely near points in sensor forward direction (same intent as x_min)
        # NOTE: p.point.x is sensor-local forward (+x).
        if p.point.x <= x_min_m:
            continue

        # Convert this LiDAR return into world coordinates
        world_loc = lidar_tf.transform(carla.Location(x=p.point.x, y=p.point.y, z=p.point.z))
        px, py = world_loc.x, world_loc.y

        # Find closest segment of noodle polyline in XY
        best_s = None
        best_dist2 = None

        for i in range(len(pts) - 1):
            ax, ay = pts[i]
            bx, by = pts[i + 1]

            t, qx, qy, dist2 = _project_point_to_segment_2d(px, py, ax, ay, bx, by)

            if (best_dist2 is None) or (dist2 < best_dist2):
                # approximate along-noodle distance at projection:
                s_here = s_vals[i] + t * (s_vals[i + 1] - s_vals[i])
                best_dist2 = dist2
                best_s = s_here

        if best_s is None or best_dist2 is None:
            continue

        # Lateral gate (lane corridor width)
        if best_dist2 > (half_width_m * half_width_m):
            continue

        # Along-noodle "ahead" gate
        if best_s <= x_min_m:
            continue
        if best_s > max_dist_m:
            continue

        # Distance metric: use LiDAR origin -> point (world)
        dx = world_loc.x - lidar_origin.x
        dy = world_loc.y - lidar_origin.y
        dz = world_loc.z - lidar_origin.z
        d = math.sqrt(dx * dx + dy * dy + dz * dz)

        if (min_d is None) or (d < min_d):
            min_d = d

    return min_d, noodle_world_points

def lidar_min_distance_along_route_noodle(
    lidar_actor,
    lidar_data,
    route_points_world,
    ego_loc,
    *,
    half_width_m: float,
    z_min: float,
    z_max: float,
    max_dist_m: float,
    x_min_m: float,
):
    """
    Hazard filter using an externally supplied planned route polyline
    instead of rebuilding a noodle from local CARLA wp.next() calls.

    route_points_world:
        list of carla.Location points describing the ego planned route.
    ego_loc:
        current ego location used to determine where "ahead" starts.
    """
    if lidar_data is None or lidar_actor is None or not route_points_world or len(route_points_world) < 2:
        return None, None

    # Find closest route index to ego
    closest_i = None
    closest_d2 = None
    for i, p in enumerate(route_points_world):
        dx = p.x - ego_loc.x
        dy = p.y - ego_loc.y
        dz = p.z - ego_loc.z
        d2 = dx * dx + dy * dy + dz * dz
        if (closest_d2 is None) or (d2 < closest_d2):
            closest_d2 = d2
            closest_i = i

    if closest_i is None:
        return None, None

    # Build local route segment ahead of ego out to max_dist_m
    pts = []
    s_vals = []

    pts.append((route_points_world[closest_i].x, route_points_world[closest_i].y))
    s_vals.append(0.0)

    traveled = 0.0
    prev = route_points_world[closest_i]

    for i in range(closest_i + 1, len(route_points_world)):
        curr = route_points_world[i]
        seg = prev.distance(curr)

        if traveled + seg > max_dist_m:
            break

        traveled += seg
        pts.append((curr.x, curr.y))
        s_vals.append(traveled)
        prev = curr

    if len(pts) < 2:
        return None, None

    lidar_tf = lidar_actor.get_transform()
    lidar_origin = lidar_tf.location

    min_d = None
    noodle_world_points = [carla.Location(x=p[0], y=p[1], z=lidar_origin.z - 1.5) for p in pts]

    for p in lidar_data:
        z = p.point.z
        if z < z_min or z > z_max:
            continue

        if p.point.x <= x_min_m:
            continue

        world_loc = lidar_tf.transform(carla.Location(x=p.point.x, y=p.point.y, z=p.point.z))
        px, py = world_loc.x, world_loc.y

        best_s = None
        best_dist2 = None

        for i in range(len(pts) - 1):
            ax, ay = pts[i]
            bx, by = pts[i + 1]

            t, qx, qy, dist2 = _project_point_to_segment_2d(px, py, ax, ay, bx, by)

            if (best_dist2 is None) or (dist2 < best_dist2):
                s_here = s_vals[i] + t * (s_vals[i + 1] - s_vals[i])
                best_dist2 = dist2
                best_s = s_here

        if best_s is None or best_dist2 is None:
            continue

        if best_dist2 > (half_width_m * half_width_m):
            continue

        if best_s <= x_min_m:
            continue
        if best_s > max_dist_m:
            continue

        dx = world_loc.x - lidar_origin.x
        dy = world_loc.y - lidar_origin.y
        dz = world_loc.z - lidar_origin.z
        d = math.sqrt(dx * dx + dy * dy + dz * dz)

        if (min_d is None) or (d < min_d):
            min_d = d

    return min_d, noodle_world_points


def lidar_min_distances_along_route_corridors(
    lidar_actor,
    lidar_data,
    route_points_world,
    ego_loc,
    *,
    lateral_offsets_m,
    half_width_m: float,
    z_min: float,
    z_max: float,
    max_dist_m: float,
    x_min_m: float,
):
    """Measure several parallel route corridors in one LiDAR pass.

    ``lateral_offsets_m`` uses the project's route-relative convention:
    positive is route-right and negative is route-left. Each LiDAR return is
    projected onto the original route once; its signed lateral displacement is
    then compared with every requested corridor center.

    This is monitoring only. The caller decides which corridor, if any, owns
    braking. The function returns ``(minimum_distances, corridor_polylines)``;
    both dictionaries are keyed by the requested float offsets.
    """
    offsets = tuple(dict.fromkeys(float(value) for value in lateral_offsets_m))
    minimum_distances = {offset: None for offset in offsets}
    corridor_polylines = {offset: None for offset in offsets}

    if (
        not offsets
        or lidar_data is None
        or lidar_actor is None
        or not route_points_world
        or len(route_points_world) < 2
    ):
        return minimum_distances, corridor_polylines

    closest_i = None
    closest_d2 = None
    for index, point in enumerate(route_points_world):
        dx = point.x - ego_loc.x
        dy = point.y - ego_loc.y
        dz = point.z - ego_loc.z
        distance_sq = dx * dx + dy * dy + dz * dz
        if closest_d2 is None or distance_sq < closest_d2:
            closest_i = index
            closest_d2 = distance_sq

    if closest_i is None:
        return minimum_distances, corridor_polylines

    route_xy = []
    route_s = []
    route_indices = []
    traveled = 0.0
    previous = route_points_world[closest_i]
    route_xy.append((previous.x, previous.y))
    route_s.append(0.0)
    route_indices.append(closest_i)

    for index in range(closest_i + 1, len(route_points_world)):
        current = route_points_world[index]
        segment_length = previous.distance(current)
        if traveled + segment_length > max_dist_m:
            break
        traveled += segment_length
        route_xy.append((current.x, current.y))
        route_s.append(traveled)
        route_indices.append(index)
        previous = current

    if len(route_xy) < 2:
        return minimum_distances, corridor_polylines

    lidar_transform = lidar_actor.get_transform()
    lidar_origin = lidar_transform.location

    for offset in offsets:
        shifted_points = []
        for (x, y), route_index in zip(route_xy, route_indices):
            right_x, right_y = route_unit_right_xy(route_points_world, route_index)
            shifted_points.append(
                carla.Location(
                    x=x + right_x * offset,
                    y=y + right_y * offset,
                    z=lidar_origin.z - 1.5,
                )
            )
        corridor_polylines[offset] = shifted_points

    half_width_sq = half_width_m * half_width_m

    for lidar_point in lidar_data:
        local_z = lidar_point.point.z
        if local_z < z_min or local_z > z_max:
            continue
        if lidar_point.point.x <= x_min_m:
            continue

        world_location = lidar_transform.transform(
            carla.Location(
                x=lidar_point.point.x,
                y=lidar_point.point.y,
                z=lidar_point.point.z,
            )
        )
        point_x = world_location.x
        point_y = world_location.y

        best_distance_sq = None
        best_along_route_m = None
        best_signed_lateral_m = None

        for index in range(len(route_xy) - 1):
            start_x, start_y = route_xy[index]
            end_x, end_y = route_xy[index + 1]
            projection, projected_x, projected_y, distance_sq = _project_point_to_segment_2d(
                point_x, point_y, start_x, start_y, end_x, end_y
            )
            if best_distance_sq is None or distance_sq < best_distance_sq:
                segment_dx = end_x - start_x
                segment_dy = end_y - start_y
                segment_length = math.hypot(segment_dx, segment_dy)
                if segment_length <= 1e-9:
                    continue

                right_x = -segment_dy / segment_length
                right_y = segment_dx / segment_length
                best_distance_sq = distance_sq
                best_along_route_m = route_s[index] + projection * (
                    route_s[index + 1] - route_s[index]
                )
                best_signed_lateral_m = (
                    (point_x - projected_x) * right_x
                    + (point_y - projected_y) * right_y
                )

        if best_along_route_m is None or best_signed_lateral_m is None:
            continue
        if best_along_route_m <= x_min_m or best_along_route_m > max_dist_m:
            continue

        dx = world_location.x - lidar_origin.x
        dy = world_location.y - lidar_origin.y
        dz = world_location.z - lidar_origin.z
        physical_distance_m = math.sqrt(dx * dx + dy * dy + dz * dz)

        for offset in offsets:
            corridor_distance_sq = (best_signed_lateral_m - offset) ** 2
            if corridor_distance_sq > half_width_sq:
                continue
            current_minimum = minimum_distances[offset]
            if current_minimum is None or physical_distance_m < current_minimum:
                minimum_distances[offset] = physical_distance_m

    return minimum_distances, corridor_polylines

def draw_lane_noodle_corridor(
    world,
    noodle_points_world,
    *,
    half_width_m: float,
    life_time: float,
    z_offset: float = 0.0,
    hazard_active: bool = False,
    tick_interval_m: float = 5.0,
    center_color=None,
    normal_edge_color=None,
    hazard_edge_color=None,
    normal_tick_color=None,
    hazard_tick_color=None,
):
    """
    Debug draw:
    - centerline polyline (gray)
    - left/right boundary edges (blue-gray normally, orange when hazard_active)
    - cross-tick marks every tick_interval_m meters (makes tunnel shape clearer)

    noodle_points_world is the polyline returned by lidar_min_distance_*_noodle().
    hazard_active: when True, edges turn orange to show the filter is firing.
    """
    if noodle_points_world is None or len(noodle_points_world) < 2:
        return

    col_center = center_color or carla.Color(90, 90, 90)
    col_edge = (
        (hazard_edge_color or carla.Color(220, 100, 0))
        if hazard_active
        else (normal_edge_color or carla.Color(80, 120, 160))
    )
    col_tick = (
        (hazard_tick_color or carla.Color(180, 70, 0))
        if hazard_active
        else (normal_tick_color or carla.Color(60, 100, 140))
    )

    cumulative_s = 0.0
    next_tick_s  = tick_interval_m  # first tick drawn at this distance

    for i in range(len(noodle_points_world) - 1):
        a = noodle_points_world[i]
        b = noodle_points_world[i + 1]

        az = a + carla.Location(z=z_offset)
        bz = b + carla.Location(z=z_offset)

        # centerline segment
        world.debug.draw_line(az, bz, thickness=0.03, color=col_center, life_time=life_time)

        dx = b.x - a.x
        dy = b.y - a.y
        seg_len = math.sqrt(dx * dx + dy * dy)
        if seg_len < 1e-6:
            continue

        # perpendicular (2D) unit vector for left/right offsets
        nx = -dy / seg_len
        ny =  dx / seg_len

        aL = carla.Location(x=a.x + nx * half_width_m, y=a.y + ny * half_width_m, z=a.z + z_offset)
        bL = carla.Location(x=b.x + nx * half_width_m, y=b.y + ny * half_width_m, z=b.z + z_offset)
        aR = carla.Location(x=a.x - nx * half_width_m, y=a.y - ny * half_width_m, z=a.z + z_offset)
        bR = carla.Location(x=b.x - nx * half_width_m, y=b.y - ny * half_width_m, z=b.z + z_offset)

        world.debug.draw_line(aL, bL, thickness=0.04, color=col_edge, life_time=life_time)
        world.debug.draw_line(aR, bR, thickness=0.04, color=col_edge, life_time=life_time)

        # cross-tick marks at regular intervals along the corridor
        seg_start_s = cumulative_s
        seg_end_s   = cumulative_s + seg_len
        while next_tick_s <= seg_end_s:
            t = (next_tick_s - seg_start_s) / seg_len
            tx = a.x + t * dx
            ty = a.y + t * dy
            tz = a.z + z_offset
            tick_L = carla.Location(x=tx + nx * half_width_m, y=ty + ny * half_width_m, z=tz)
            tick_R = carla.Location(x=tx - nx * half_width_m, y=ty - ny * half_width_m, z=tz)
            world.debug.draw_line(tick_L, tick_R, thickness=0.03, color=col_tick, life_time=life_time)
            next_tick_s += tick_interval_m

        cumulative_s += seg_len
