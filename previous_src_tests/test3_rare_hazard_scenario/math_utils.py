"""
test2___math_utils.py

Goal:
- Hold pure math / small conversion utilities used by test2 scripts.
- Keep logic identical to the original test2___braking_via_lidar.py.

NOTE:
- These functions were moved AS-IS (comments/docstrings preserved).
"""

import math


def clamp(value_to_clamp, lo, hi):
    # Keeps values inside a range -> Used for steering [-1,1], throttle [0, MAX], brake [0, MAX]
    # (keeps value_to_clamp in [lo, hi])
    return max(lo, min(hi, value_to_clamp))


def wrap_to_pi(angle_rad):
    """
    Angles wrap around in a circle, so values like +181° and -179° mean
    almost the same direction but look very different numerically.
    This function forces any angle into the range [-pi, +pi] ([-180°, +180°])
    so the steering error always represents the SMALLEST turn needed.
    This prevents the car from suddenly trying to steer the long way around.
    """
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


def yaw_deg_to_rad(yaw_deg):
    # CARLA gives yaw in degrees, math wants radians
    return math.radians(yaw_deg)


def get_speed_mps(vehicle):
    """
    CARLA gives velocity as a 3D vector (v.x, v.y, v.z), not a single speed.
    Each component is how fast the car is moving in that direction (m/s).

    We square each component and then take the square root NOT to cancel it out,
    but to compute the LENGTH (magnitude) of the velocity vector.
    This is standard geometry: sqrt(x^2 + y^2 + z^2).

    The result is ONE scalar value:
    "how fast the car is moving overall, regardless of direction".
    """
    v = vehicle.get_velocity()
    return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)


def mps_to_mph(speed_mps):
    # Convenience converter for printing human-friendly speed numbers
    # (from meters/second to miles/hour)
    return speed_mps / 0.44704