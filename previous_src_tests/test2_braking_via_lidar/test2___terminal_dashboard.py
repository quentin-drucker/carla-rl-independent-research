"""
test2___terminal_dashboard.py

Goal:
- Hold terminal "live dashboard" helpers (colored, single-line status output)
- Keep logic identical to the original test2___braking_via_lidar.py.
"""

from test2___math_utils import mps_to_mph


# ---          ---          ---          ---          ---
# Terminal "live dashboard" helpers (single-line + colored terminal output)
# ---          ---          ---          ---          ---
ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",

    "cyan": "\033[36m",
    "magenta": "\033[35m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "bright_red": "\033[91m",
}

def print_live_status(*, t, total_ticks, telemetry, target_mph):
    """
    Print metric updates in console.
    # If changing to one-line prints, use carriage return '\\r' + flush=True. (and make sure to print() nl at end of run so summary doesn't get messed up on same line)
    """
    if telemetry is None:
        line = (
            f"{ANSI['dim']}tick {t:04d}/{total_ticks} | waiting for telemetry...{ANSI['reset']}   "
        )
        print(line, flush=True)
        return

    v_mph = mps_to_mph(telemetry["speed_mps"])
    steer = telemetry["steer_cmd"]
    thr = telemetry.get("throttle_cmd", 0.0)
    brk = telemetry.get("brake_cmd", 0.0)
    d_min = telemetry.get("d_min_ahead_m", None)
    haz = telemetry.get("hazard_brake_cmd", 0.0)

    # --- Semantic Coloring (for controls and hazard info) ---

    # steering: highlight only when turning
    steer_col = ANSI["magenta"] if abs(steer) > 0.05 else ANSI["dim"]

    # throttle: green when applied
    thr_col = ANSI["green"] if thr > 0.05 else ANSI["dim"]

    # brake: bright red when applied
    brk_col = ANSI["bright_red"] if brk > 0.05 else ANSI["dim"]

    # hazard: dim -> yellow -> bright red
    if haz >= 0.7:
        haz_col = ANSI["bright_red"]
    elif haz >= 0.3:
        haz_col = ANSI["yellow"]
    else:
        haz_col = ANSI["dim"]

    # distance: warn in "yellow" when below a "likely braking soon" threshold
    speed_mps = telemetry["speed_mps"]
    d_warn = 6.0 + speed_mps * 1.2  # buffer_m + speed*headway_s (match your controller)
    dist_col = ANSI["yellow"] if (d_min is not None and d_min < d_warn) else ANSI["dim"]

    d_str = " None " if d_min is None else f"{d_min:5.1f}m"

    line = (
        f"{ANSI['dim']}tick {t:04d}/{total_ticks}{ANSI['reset']} | "
        f"{ANSI['cyan']}{ANSI['bold']}v={v_mph:6.2f} mph{ANSI['reset']} "
        f"{ANSI['dim']}(target: {target_mph:4.0f}){ANSI['reset']} | "
        f"{steer_col}steer={steer:+.2f}{ANSI['reset']} | "              #steer = steering command (-1 full left, +1 full right)
        f"{thr_col}thr={thr:.2f}{ANSI['reset']} | "                     #thr = throttle command from speed controller (0 = no throttle, 1 = full throttle)
        f"{brk_col}brk={brk:.2f}{ANSI['reset']} | "                     #brk = actual brake command applied this tick (from speed controller or LiDAR hazard override)
        f"{dist_col}dmin={d_str}{ANSI['reset']} | "                     #dmin = distance to closest LiDAR point ahead in the "danger zone"
        f"{haz_col}{ANSI['bold']}haz={haz:.2f}{ANSI['reset']}   "       #haz = hazard brake command from LiDAR-based hazard detection (0 = no hazard, 1 = full emergency brake
    )
    print(line, flush=True)