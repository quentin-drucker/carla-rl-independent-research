"""test8___offset_magnitude_sweep.py

Follow-up to this week's braking-authority attempt (tried and reverted --
see worklog 2026-09-18): since letting steering suppress the original
corridor's braking isn't safe yet, this instead asks a narrower, safe
question: how much does the swerve's PEAK OFFSET itself change real
clearance from the pedestrian, using tools already validated this week
(map-drivability check) to confirm a wider offset still stays on the road?

Braking authority is completely UNCHANGED here -- the original-route
corridor still owns braking exactly as before this whole investigation.
This sweep only varies the scripted lateral_offset_fn's peak_offset_m,
with everything else held fixed at the "intermediate"
(trigger_ttc_s=4.0, steering_only, brake_headway_s=0.1) configuration --
the case originally flagged as brake-dominated -- to see whether a wider
swerve gives more real pedestrian clearance and whether it's still
map-drivable.

test5's CLI caps peak offset at 1.5m as a leftover "preliminary" bound
from before the drivability check existed; this script intentionally
goes past that (up to 2.5m) since the drivability check can now confirm
whether that's actually safe road geometry, run by run.

Run CARLA (windowed), then from ``src``:

    python -X utf8 test8___offset_magnitude_sweep.py
"""

from scenario_config import ScenarioConfig
from test3___ped_intrusion_scenario import run_scenario
from test5___scripted_pedestrian_steering import (
    CorridorConsoleObserver,
    build_evasive_offset_fn,
)

OFFSETS_M = (1.0, 1.5, 2.0, 2.5)
TRIGGER_TTC_S = 4.0        # "intermediate" timing from the Week 2 matrix
BRAKE_HEADWAY_S = 0.1      # "steering_only" -- isolates real swerve clearance


def _run_one(offset_m):
    print(f"\n{'='*70}")
    print(f"OFFSET SWEEP RUN: offset={offset_m:+.2f}m")
    print(f"{'='*70}")

    config = ScenarioConfig(
        walker_speed_mps=1.8,
        walker_side="left",
        walker_cross="near",
        trigger_ttc_s=TRIGGER_TTC_S,
        target_mph=15.0,
        encounter_distance_m=60.0,
        brake_profile="exponential",
        brake_headway_s=BRAKE_HEADWAY_S,
        weather_preset="ClearSunset",
        sun_altitude_deg=0,
        sim_seconds=18.0,
    )

    observer = CorridorConsoleObserver()
    result = run_scenario(
        config,
        plot_after=False,
        lateral_offset_fn=build_evasive_offset_fn(offset_m),
        monitor_lateral_corridors=True,
        tick_observer=observer,
        post_crossing_settle_s=6.0,
    )

    row = {
        "offset_m": offset_m,
        "collision": result.collision_detected,
        "outcome": result.outcome,
        "min_ped_distance_m": result.min_ped_distance_m,
        "max_abs_offset_m": observer.maximum_actual_offset_m,
        "commanded_ever_non_drivable": observer.commanded_ever_non_drivable,
        "transition_ever_non_drivable": observer.transition_ever_non_drivable,
    }
    print(
        f"-> offset={offset_m:+.2f}m collision={row['collision']} "
        f"outcome={row['outcome']} min_ped_dist={row['min_ped_distance_m']:.2f}m "
        f"commanded_non_drivable={row['commanded_ever_non_drivable']} "
        f"transition_non_drivable={row['transition_ever_non_drivable']}"
    )
    return row


def main():
    print(
        "OFFSET MAGNITUDE SWEEP: braking authority UNCHANGED (original corridor "
        "still owns braking). Only varying peak swerve offset to see how much "
        "real pedestrian clearance it buys, and whether wider offsets are still "
        "map-drivable on this route."
    )
    rows = [_run_one(offset_m) for offset_m in OFFSETS_M]

    print(f"\n\n{'#'*90}")
    print("OFFSET MAGNITUDE SWEEP SUMMARY")
    print(f"{'#'*90}")
    header = (
        f"{'offset_m':9} {'collision':9} {'outcome':15} {'min_ped_m':9} "
        f"{'max_off_m':9} {'commanded_ND':13} {'transition_ND':13}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['offset_m']:9.2f} {str(row['collision']):9} {row['outcome']:15} "
            f"{row['min_ped_distance_m']:9.2f} {row['max_abs_offset_m']:9.2f} "
            f"{str(row['commanded_ever_non_drivable']):13} "
            f"{str(row['transition_ever_non_drivable']):13}"
        )


if __name__ == "__main__":
    main()
