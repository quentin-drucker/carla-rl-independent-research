"""test7___steering_validation_matrix.py

Week 2 plan step 4: a small, fixed, deterministic validation matrix for the
scripted pedestrian-avoidance maneuver -- NOT a random sweep and NOT an RL
policy comparison. This is a feasibility study of the maneuver + hazard-based
recovery + map-drivability check built earlier this week.

Matrix (2 x 3 x 2 = 12 runs):
    direction : right (+1.5 m), left (-1.5 m)
    timing    : early (trigger_ttc_s=5.0), intermediate (4.0), late (3.0)
    maneuver  : steering_only (brake_headway_s=0.1 -- braking effectively
                    reactive/last-resort only), steering_plus_braking
                    (brake_headway_s=2.5 -- the scenario's normal AEB
                    trigger, active alongside the scripted steering)

"steering_only" does NOT disable emergency braking outright (that would
risk an uninstrumented hard collision for no research value); it shrinks
the LiDAR braking trigger distance to near the panic/base distance so the
scripted steering gets first crack at clearing the pedestrian before
braking would normally engage. This isolates whether the swerve alone,
given room, avoids the pedestrian -- versus the default scenario where the
original-route corridor's braking authority may dominate first (the
brake-vs-steering interaction observed manually before this run).

Uses a fixed, recorded random_seed for provenance even though this
scenario has no actual randomness (scripted walker, no RNG draws) --
recorded per the plan's "Seed: Fixed and recorded" matrix column.

Run CARLA (windowed, so you can watch), then from ``src``:

    python -X utf8 test7___steering_validation_matrix.py

Prints a per-run summary line as each of the 12 runs completes, then a
final table. This does not gate anything or change scenario mechanics --
it only calls run_scenario()/build_evasive_offset_fn() the same way
test5/test6 already do.
"""

import itertools

from scenario_config import ScenarioConfig
from test3___ped_intrusion_scenario import run_scenario
from test5___scripted_pedestrian_steering import (
    CorridorConsoleObserver,
    build_evasive_offset_fn,
)

MATRIX_SEED = 2026  # fixed, recorded provenance value; scenario itself is deterministic

DIRECTIONS = {
    "right": +1.5,
    "left": -1.5,
}
TIMINGS_TRIGGER_TTC_S = {
    "early": 5.0,
    "intermediate": 4.0,
    "late": 3.0,
}
MANEUVERS_BRAKE_HEADWAY_S = {
    "steering_only": 0.1,
    "steering_plus_braking": 2.5,
}


class ValidationObserver(CorridorConsoleObserver):
    """Adds a coarse steering-aggressiveness signal on top of the existing
    corridor/offset tracking. Full skid/ABS/tire-dynamics analysis is out of
    scope this week (explicitly deferred in the Week 2 plan) -- this is only
    a proxy so obviously extreme steering commands stand out in the table.
    """

    def __init__(self, fixed_dt=0.02):
        super().__init__(fixed_dt=fixed_dt)
        self.max_abs_steer_cmd = 0.0

    def __call__(self, sim_time_s, triggered, telemetry):
        super().__call__(sim_time_s, triggered, telemetry)
        if telemetry is not None:
            self.max_abs_steer_cmd = max(
                self.max_abs_steer_cmd, abs(telemetry.get("steer_cmd", 0.0))
            )


def _run_one(*, direction_name, offset_m, timing_name, trigger_ttc_s,
             maneuver_name, brake_headway_s):
    label = f"{direction_name}/{timing_name}/{maneuver_name}"
    print(f"\n{'='*70}")
    print(f"MATRIX RUN: {label}")
    print(
        f"  offset={offset_m:+.2f}m  trigger_ttc_s={trigger_ttc_s:.1f}  "
        f"brake_headway_s={brake_headway_s:.1f}  seed={MATRIX_SEED}"
    )
    print(f"{'='*70}")

    config = ScenarioConfig(
        walker_speed_mps=1.8,
        walker_side="left",
        walker_cross="near",
        trigger_ttc_s=trigger_ttc_s,
        target_mph=15.0,
        encounter_distance_m=60.0,
        brake_profile="exponential",
        brake_headway_s=brake_headway_s,
        weather_preset="ClearSunset",
        sun_altitude_deg=0,
        sim_seconds=18.0,
        random_seed=MATRIX_SEED,
    )

    observer = ValidationObserver()
    recovery_controller = build_evasive_offset_fn(offset_m)
    result = run_scenario(
        config,
        plot_after=False,
        lateral_offset_fn=recovery_controller,
        monitor_lateral_corridors=True,
        tick_observer=observer,
        post_crossing_settle_s=6.0,
    )

    row = {
        "label": label,
        "direction": direction_name,
        "timing": timing_name,
        "maneuver": maneuver_name,
        "collision": result.collision_detected,
        "outcome": result.outcome,
        "brake_dominated": result.outcome == "full_stop",
        "min_ped_distance_m": result.min_ped_distance_m,
        "max_abs_offset_m": observer.maximum_actual_offset_m,
        "final_offset_m": observer.final_actual_offset_m,
        "commanded_ever_non_drivable": observer.commanded_ever_non_drivable,
        "transition_ever_non_drivable": observer.transition_ever_non_drivable,
        "recovered": recovery_controller.recovered,
        "recovery_time_s": recovery_controller.recovery_time_s,
        "used_fallback_timeout": recovery_controller.used_fallback_timeout,
        "hazard_reappeared_during_recovery": recovery_controller.hazard_reappeared_during_recovery,
        "max_abs_steer_cmd": observer.max_abs_steer_cmd,
    }

    print(
        f"-> RESULT [{label}] collision={row['collision']} outcome={row['outcome']} "
        f"brake_dominated={row['brake_dominated']} "
        f"min_ped_dist={row['min_ped_distance_m']:.2f}m "
        f"max_offset={row['max_abs_offset_m']:.2f}m "
        f"recovered={row['recovered']} fallback_timeout={row['used_fallback_timeout']}"
    )
    return row


def _print_summary_table(rows):
    print(f"\n\n{'#'*100}")
    print("VALIDATION MATRIX SUMMARY (12 runs: 2 directions x 3 timings x 2 maneuvers)")
    print(f"{'#'*100}")
    header = (
        f"{'label':32} {'collision':9} {'outcome':15} {'brake_dom':9} "
        f"{'min_ped_m':9} {'max_off_m':9} {'recovered':9} {'fallback':8} {'reapp':6}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['label']:32} {str(row['collision']):9} {row['outcome']:15} "
            f"{str(row['brake_dominated']):9} {row['min_ped_distance_m']:9.2f} "
            f"{row['max_abs_offset_m']:9.2f} {str(row['recovered']):9} "
            f"{str(row['used_fallback_timeout']):8} {str(row['hazard_reappeared_during_recovery']):6}"
        )

    any_non_drivable = [
        row["label"] for row in rows
        if row["commanded_ever_non_drivable"] or row["transition_ever_non_drivable"]
    ]
    print(
        "\nRuns where a candidate path was ever map-flagged non_drivable: "
        + (", ".join(any_non_drivable) if any_non_drivable else "none")
    )

    n_collisions = sum(1 for row in rows if row["collision"])
    n_brake_dominated = sum(1 for row in rows if row["brake_dominated"])
    n_recovered = sum(1 for row in rows if row["recovered"])
    n_fallback = sum(1 for row in rows if row["used_fallback_timeout"])
    print(
        f"\nTotals: collisions={n_collisions}/12  brake_dominated={n_brake_dominated}/12  "
        f"recovered={n_recovered}/12  used_fallback_timeout={n_fallback}/12"
    )
    print(
        "\nReminder: this is a feasibility study of the scripted maneuver, hazard-based "
        "recovery, and map-drivability check -- not an RL policy comparison, and not a "
        "claim about tire/skid dynamics (out of scope this week)."
    )


def main():
    rows = []
    for direction_name, timing_name, maneuver_name in itertools.product(
        DIRECTIONS, TIMINGS_TRIGGER_TTC_S, MANEUVERS_BRAKE_HEADWAY_S
    ):
        row = _run_one(
            direction_name=direction_name,
            offset_m=DIRECTIONS[direction_name],
            timing_name=timing_name,
            trigger_ttc_s=TIMINGS_TRIGGER_TTC_S[timing_name],
            maneuver_name=maneuver_name,
            brake_headway_s=MANEUVERS_BRAKE_HEADWAY_S[maneuver_name],
        )
        rows.append(row)

    _print_summary_table(rows)


if __name__ == "__main__":
    main()
