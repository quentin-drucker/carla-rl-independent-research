"""Visual test of scripted steering around one pedestrian encounter.

The original route corridor continues to own emergency braking. A second green
corridor shows the commanded lateral path. If that path contains a LiDAR return,
its edges turn magenta. The terminal prints original, commanded, left-candidate,
and right-candidate minimum distances every half second after the trigger.

Start CARLA in windowed mode, then run from ``src``:

    python -X utf8 test5___scripted_pedestrian_steering.py

Use ``--lateral-offset-m -1.5`` to mirror the maneuver route-left.
"""

import argparse
import math

from scenario_config import ScenarioConfig
from test3___ped_intrusion_scenario import run_scenario


def _smoothstep(value):
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def build_evasive_offset_fn(peak_offset_m):
    """Return a scripted shift/hold/recover callback keyed to hazard trigger."""

    def evasive_offset(sim_time_s, triggered, trigger_time_s):
        if not triggered or trigger_time_s is None:
            return 0.0

        elapsed_s = sim_time_s - trigger_time_s
        if elapsed_s < 1.0:
            return peak_offset_m * _smoothstep(elapsed_s / 1.0)
        if elapsed_s < 4.5:
            return peak_offset_m
        if elapsed_s < 6.5:
            return peak_offset_m * (
                1.0 - _smoothstep((elapsed_s - 4.5) / 2.0)
            )
        return 0.0

    return evasive_offset


def _format_distance(value):
    return "  -- " if value is None else f"{value:5.1f}"


class CorridorConsoleObserver:
    def __init__(self, fixed_dt=0.02):
        self._print_every_ticks = max(1, int(0.5 / fixed_dt))
        self._tick = 0
        self.maximum_actual_offset_m = 0.0
        self.final_actual_offset_m = float("nan")

    def __call__(self, sim_time_s, triggered, telemetry):
        if telemetry is None:
            return

        actual_offset_m = telemetry["signed_route_lateral_offset_m"]
        self.maximum_actual_offset_m = max(
            self.maximum_actual_offset_m, abs(actual_offset_m)
        )
        self.final_actual_offset_m = actual_offset_m

        if triggered and self._tick % self._print_every_ticks == 0:
            print(
                f"[corridors] t={sim_time_s:5.2f}s "
                f"request={telemetry['lateral_offset_requested_m']:+4.2f}m "
                f"actual={actual_offset_m:+4.2f}m | "
                f"original={_format_distance(telemetry['d_min_original_path_m'])}m "
                f"commanded={_format_distance(telemetry['d_min_commanded_path_m'])}m "
                f"left={_format_distance(telemetry['d_min_left_candidate_m'])}m "
                f"right={_format_distance(telemetry['d_min_right_candidate_m'])}m"
            )
        self._tick += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lateral-offset-m",
        type=float,
        default=1.5,
        help="Peak route-relative offset: positive right, negative left.",
    )
    parser.add_argument("--target-mph", type=float, default=15.0)
    parser.add_argument("--trigger-ttc-s", type=float, default=4.0)
    args = parser.parse_args()

    if abs(args.lateral_offset_m) > 1.5 or abs(args.lateral_offset_m) < 0.25:
        parser.error("Use a preliminary lateral offset between 0.25 m and 1.5 m.")
    if args.target_mph <= 0.0 or args.target_mph > 25.0:
        parser.error("Keep this preliminary test in the range (0, 25] mph.")

    config = ScenarioConfig(
        walker_speed_mps=1.8,
        walker_side="left",
        walker_cross="near",
        trigger_ttc_s=args.trigger_ttc_s,
        target_mph=args.target_mph,
        encounter_distance_m=60.0,
        brake_profile="exponential",
        weather_preset="ClearSunset",
        sun_altitude_deg=0,
        sim_seconds=16.0,
    )

    observer = CorridorConsoleObserver()
    result = run_scenario(
        config,
        plot_after=False,
        lateral_offset_fn=build_evasive_offset_fn(args.lateral_offset_m),
        monitor_lateral_corridors=True,
        tick_observer=observer,
        post_crossing_settle_s=6.0,
    )

    print("\nSCRIPTED STEERING INTERACTION SUMMARY")
    print(f"collision: {result.collision_detected}")
    print(f"scenario outcome: {result.outcome}")
    print(f"minimum pedestrian distance: {result.min_ped_distance_m:.3f} m")
    print(f"maximum absolute route offset: {observer.maximum_actual_offset_m:.3f} m")
    if math.isfinite(observer.final_actual_offset_m):
        print(f"final signed route offset: {observer.final_actual_offset_m:+.3f} m")
        print(
            "route recovery check: "
            + ("PASS" if abs(observer.final_actual_offset_m) <= 0.25 else "REVIEW")
        )
    print(
        "Reminder: original-route LiDAR still owns braking; the other corridor "
        "readings are observational in this test."
    )


if __name__ == "__main__":
    main()
