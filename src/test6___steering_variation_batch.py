"""Batch visual runner: several randomized steering iterations back to back.

Runs four scripted-steering episodes in one CARLA session, in this order:

    1. Pedestrian-free lateral shift, route-RIGHT   (test4 mechanics)
    2. Pedestrian-free lateral shift, route-LEFT    (test4 mechanics)
    3. Pedestrian encounter with a route-RIGHT evasive maneuver (test5 mechanics)
    4. Pedestrian encounter with a route-LEFT evasive maneuver  (test5 mechanics)

Each run randomizes target speed (and, for the pedestrian runs, trigger TTC and
offset magnitude) within bounds already validated in Tests 1-3 of the
preliminary steering plan, so back-to-back runs show variance rather than
repeating one fixed configuration. A ``--seed`` makes a given batch
reproducible; omit it (or change it) to see a different draw.

This does not change scenario mechanics -- it only calls the same
``run_test`` (test4) and ``run_scenario`` (test3, driven the way test5 drives
it) entry points already used by the individual manual tests, so results stay
directly comparable to the single-run smoke tests.

Run CARLA in windowed mode first, then from ``src``:

    python -X utf8 test6___steering_variation_batch.py
    python -X utf8 test6___steering_variation_batch.py --seed 7

Visualization while it runs (see also test4/test5 docstrings):
  - Dark blue line: planned route.
  - Cyan point: unshifted route lookahead target; pink point: offset target.
  - Runs 3-4 additionally draw LiDAR corridors: blue/gray (turning orange
    when braking) = original route corridor; green/magenta = commanded
    parallel-offset corridor; gold/orange = swept transition corridor that
    blends from the ego's actual offset to the commanded target. Only the
    original (blue/gray -> orange) corridor ever owns braking.
"""

import argparse
import random

from scenario_config import ScenarioConfig
from test3___ped_intrusion_scenario import run_scenario
from test4___scripted_lateral_offset import run_test as run_pedestrian_free_test
from test5___scripted_pedestrian_steering import (
    CorridorConsoleObserver,
    build_evasive_offset_fn,
)


def _run_pedestrian_free_iteration(label, *, direction_sign, rng):
    target_mph = rng.uniform(12.0, 18.0)
    offset_m = direction_sign * rng.uniform(0.6, 1.2)

    print(f"\n{'#' * 60}")
    print(f"# {label}: pedestrian-free shift, "
          f"{'RIGHT' if direction_sign > 0 else 'LEFT'}")
    print(f"# target_mph={target_mph:.1f}  peak_offset_m={offset_m:+.2f}")
    print(f"{'#' * 60}")

    run_pedestrian_free_test(
        target_mph=target_mph,
        lateral_offset_m=offset_m,
        sim_seconds=13.0,
    )


def _run_pedestrian_iteration(label, *, direction_sign, rng):
    target_mph = rng.uniform(12.0, 18.0)
    trigger_ttc_s = rng.uniform(3.5, 4.5)
    offset_m = direction_sign * rng.uniform(0.75, 1.5)

    print(f"\n{'#' * 60}")
    print(f"# {label}: pedestrian encounter, evasive "
          f"{'RIGHT' if direction_sign > 0 else 'LEFT'}")
    print(f"# target_mph={target_mph:.1f}  trigger_ttc_s={trigger_ttc_s:.2f}  "
          f"peak_offset_m={offset_m:+.2f}")
    print(f"{'#' * 60}")

    config = ScenarioConfig(
        walker_speed_mps=1.8,
        walker_side="left",
        walker_cross="near",
        trigger_ttc_s=trigger_ttc_s,
        target_mph=target_mph,
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
        lateral_offset_fn=build_evasive_offset_fn(offset_m),
        monitor_lateral_corridors=True,
        tick_observer=observer,
        post_crossing_settle_s=6.0,
    )

    print(
        f"-> collision={result.collision_detected}  outcome={result.outcome}  "
        f"min_ped_dist={result.min_ped_distance_m:.3f} m  "
        f"max_abs_offset={observer.maximum_actual_offset_m:.3f} m  "
        f"final_offset={observer.final_actual_offset_m:+.3f} m"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible variance across the four runs. "
             "Omit for a different draw each invocation.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("STEERING VARIATION BATCH: 4 runs "
          "(pedestrian-free right/left, then pedestrian right/left)")
    if args.seed is not None:
        print(f"seed={args.seed}")

    _run_pedestrian_free_iteration("Run 1/4", direction_sign=+1.0, rng=rng)
    _run_pedestrian_free_iteration("Run 2/4", direction_sign=-1.0, rng=rng)
    _run_pedestrian_iteration("Run 3/4", direction_sign=+1.0, rng=rng)
    _run_pedestrian_iteration("Run 4/4", direction_sign=-1.0, rng=rng)

    print("\nBatch complete. Reminder: only the original-route LiDAR corridor "
          "owns braking in runs 3-4; the commanded/transition/candidate "
          "corridors are observational only.")


if __name__ == "__main__":
    main()
