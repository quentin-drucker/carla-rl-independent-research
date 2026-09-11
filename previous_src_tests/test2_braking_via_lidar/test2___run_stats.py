"""
test2___run_stats.py

Goal:
- Hold run-stat tracking helpers for test2 scripts.
- Keep logic + printed summary identical to the original test2___braking_via_lidar.py.
"""

import math

from test2___math_utils import mps_to_mph


def init_run_stats():
    # Step 5: update run metrics (stats)
    # (kept as the same fields your main script uses)
    return {
        "ticks": 0,

        "cte_sum": 0.0,
        "cte_max": 0.0,

        "heading_abs_sum": 0.0,
        "heading_abs_max": 0.0,

        "distance_m": 0.0,
        "prev_loc": None,

        "speed_err_abs_sum_mps": 0.0,

        "steer_prev": None,
        "steer_rate_abs_sum": 0.0,
    }


def update_run_stats(stats, *, telemetry, vehicle, FIXED_DT):
    # Step 5: update run metrics (stats)
    if telemetry is None:
        return

    stats["ticks"] += 1

    cte = telemetry["cte_m"]
    stats["cte_sum"] += cte
    stats["cte_max"] = max(stats["cte_max"], cte)

    heading_abs = abs(telemetry["heading_error_rad"])
    stats["heading_abs_sum"] += heading_abs
    stats["heading_abs_max"] = max(stats["heading_abs_max"], heading_abs)

    loc_now = vehicle.get_transform().location
    if stats["prev_loc"] is not None:
        stats["distance_m"] += loc_now.distance(stats["prev_loc"])
    stats["prev_loc"] = loc_now

    stats["speed_err_abs_sum_mps"] += abs(telemetry["speed_error_mps"])

    steer = telemetry["steer_cmd"]
    if stats["steer_prev"] is not None:
        stats["steer_rate_abs_sum"] += abs(steer - stats["steer_prev"]) / FIXED_DT
    stats["steer_prev"] = steer


def print_run_summary(stats, *, TARGET_MAP, SPAWN_INDEX, SIM_SECONDS, FIXED_DT, TARGET_MPH):
    # -------------------------------------------------
    # End-of-run "stats page"
    # -------------------------------------------------
    if stats["ticks"] > 0:
        mean_cte = stats["cte_sum"] / stats["ticks"]
        mean_heading_rad = stats["heading_abs_sum"] / stats["ticks"]

        mean_speed_err_mps = stats["speed_err_abs_sum_mps"] / stats["ticks"]

        # Convert some things to human-friendly units for the report
        mean_heading_deg = math.degrees(mean_heading_rad)
        max_heading_deg = math.degrees(stats["heading_abs_max"])

        mean_speed_err_mph = mps_to_mph(mean_speed_err_mps)

        # Steering rate: we only have (ticks-1) differences, but dividing by ticks is fine as an average proxy.
        # If you want exact, divide by max(1, ticks-1).
        denom = max(1, stats["ticks"] - 1)
        mean_steer_rate = stats["steer_rate_abs_sum"] / denom

        print("\n==================== RUN STATS (Summary) ====================")
        print(f"Map: {TARGET_MAP} | Spawn index: {SPAWN_INDEX}")
        print(f"Sim seconds: {SIM_SECONDS} | dt: {FIXED_DT:.3f} | ticks logged: {stats['ticks']}")
        print(f"Target speed: {TARGET_MPH:.1f} mph")
        print("-------------------------------------------------------------")

        # 1) Lane centering
        print("1) Cross-Track Error (CTE) — lane centering error")
        print("   Meaning: distance from car to lane center waypoint (meters). Smaller = stays centered.")
        print(f"   Mean CTE: {mean_cte:.3f} m | Max CTE: {stats['cte_max']:.3f} m")
        print("-------------------------------------------------------------")

        # 2) Heading alignment
        print("2) Heading Error — direction alignment error")
        print("   Meaning: |desired heading to lookahead point - car yaw|. Smaller = faces where it should.")
        print(f"   Mean heading error: {mean_heading_rad:.4f} rad ({mean_heading_deg:.2f} deg)")
        print(f"   Max  heading error: {stats['heading_abs_max']:.4f} rad ({max_heading_deg:.2f} deg)")
        print("-------------------------------------------------------------")

        # 3) Progress
        print("3) Distance Traveled — how far the car moved")
        print("   Meaning: total path length integrated from frame-to-frame position changes (meters).")
        print(f"   Distance traveled: {stats['distance_m']:.1f} m")
        print("-------------------------------------------------------------")

        # 4) Speed tracking
        print("4) Speed Error — speed hold quality")
        print("   Meaning: average |target speed - actual speed|. Smaller = better PI speed hold.")
        print(f"   Mean speed error: {mean_speed_err_mph:.2f} mph")
        print("-------------------------------------------------------------")

        # 5) Smoothness
        print("5) Steering Rate — twitchiness / smoothness proxy")
        print("   Meaning: average |Δsteer| / dt. Smaller = smoother steering changes over time.")
        print(f"   Mean steering rate: {mean_steer_rate:.3f} (steer units per second)")
        print("=============================================================\n")
    else:
        print("\n(No stats collected: telemetry was None every tick.)\n")