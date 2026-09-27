"""test16___far_cross_swerve_and_merge_back.py

WARNING (2026-09-27): this script's ORIGINAL config (peak_offset_m=1.5)
produces a REAL COLLISION -- confirmed live by Quentin (visually, the
pedestrian's geometry glitches/gets shoved) and independently by
pedestrian_contact.py's ground-truth geometric check (a sustained 0.5s
contact window). CARLA's own collision sensor did NOT report it
(result.collision_detected read False in reproducible automated runs) --
see MASTER_CARLA_RESEARCH_SUMMARY.md's "a real coverage gap found via a
far-crossing pedestrian test" entry for the full root-cause trace. This
script currently demonstrates a discovered SAFETY GAP, not a validated
swerve-and-merge-back success. Do not treat a clean printed outcome from
this script as proof of no collision -- verify with pedestrian_contact.py
against the recorded trace, exactly as this investigation had to.

Original intent (still not achieved as of 2026-09-27): demonstrate a full
swerve-avoid-and-merge-back cycle using ONLY already-validated, unmodified
code -- no changes to braking authority, hazard governance, or any
safety-relevant control logic. That non-goal (see the Week 3 plan's "Two
attempts... reverted" history) is still respected -- nothing here touches
braking authority. The actual problem found instead: the original-lane
hazard corridor only watches a band +/-1.4m from route centerline, but
this walker_cross="far" pedestrian ends up at +2.55m -- well outside it.
Once they exit that band, NOTHING currently watches that space for a
pedestrian hazard (the commanded/transition corridors are observational
telemetry only, and only track vehicle actors regardless). Widening the
swerve offset alone (tested up to 3.2m) does not fix this, since the
corridor's blind spot doesn't move with the ego's offset.

Usage (CARLA must already be running, windowed so you can watch):
    python -X utf8 test16___far_cross_swerve_and_merge_back.py
"""

from scenario_config import ScenarioConfig
from test3___ped_intrusion_scenario import run_scenario
from test5___scripted_pedestrian_steering import build_evasive_offset_fn


def _tick_printer():
    state = {"last_print_s": -1.0, "last_recovery_state": None}

    def observer(sim_time_s, triggered, telemetry):
        if telemetry is None:
            return
        if sim_time_s - state["last_print_s"] >= 0.5:
            state["last_print_s"] = sim_time_s
            print(
                f"    t={sim_time_s:5.2f}s triggered={triggered} "
                f"speed={telemetry.get('speed_mps', 0.0):5.2f}m/s "
                f"lateral_offset_actual={telemetry.get('signed_route_lateral_offset_m', 0.0):+.2f}m "
                f"lateral_offset_requested={telemetry.get('lateral_offset_requested_m', 0.0):+.2f}m "
                f"drive_mode={telemetry.get('drive_mode')} "
                f"brake_governed_by={telemetry.get('hazard_governing_source')}"
            )

    return observer


def main():
    cfg = ScenarioConfig(
        walker_speed_mps=1.8,
        walker_side="left",
        walker_cross="far",  # the untested case -- pedestrian actually exits the lane
        trigger_ttc_s=5.0,   # generous lead time so the ego isn't forced to zero before the ped clears
        target_mph=25.0,     # moderate speed, easier to watch and to avoid a forced full stop
        encounter_distance_m=90.0,
        brake_profile="exponential",
        weather_preset="ClearSunset",
        sun_altitude_deg=0,
        sim_seconds=20.0,
    )

    recovery_controller = build_evasive_offset_fn(peak_offset_m=1.5)

    print("=" * 70)
    print("test16: far-crossing pedestrian scenario")
    print("KNOWN ISSUE (2026-09-27): this config produces a REAL COLLISION")
    print("around 3s after the pedestrian finishes crossing, which CARLA's")
    print("own collision sensor does NOT reliably report. Do not trust the")
    print("printed 'Collision: False' below at face value -- verify against")
    print("the recorded trace with pedestrian_contact.py. See this script's")
    print("module docstring and MASTER_CARLA_RESEARCH_SUMMARY.md.")
    print("=" * 70)

    result = run_scenario(
        cfg,
        plot_after=False,
        lateral_offset_fn=recovery_controller,
        monitor_lateral_corridors=True,
        tick_observer=_tick_printer(),
        post_crossing_settle_s=5.0,
    )

    print("\n" + "=" * 70)
    print(f"Outcome: {result.outcome}  (collision={result.collision_detected})")
    print(f"Recovered (swerve returned to lane): {recovery_controller.recovered}")
    print(f"Used fallback timeout instead of a real hazard-clear: {recovery_controller.used_fallback_timeout}")
    print(f"Hazard reappeared mid-recovery: {recovery_controller.hazard_reappeared_during_recovery}")
    print(f"Recovery (return-to-center) duration: {recovery_controller.recovery_time_s}")
    print("=" * 70)
    if result.outcome == "full_stop":
        print(
            "\nNote: outcome='full_stop' means the ego's speed dropped below the "
            "stopped threshold at some point -- this can still happen even in a "
            "successful swerve+merge-back run (original-lane braking is anchored "
            "to the pedestrian, who is still crossing for the first couple "
            "seconds). Check 'Recovered' above, not just 'Outcome', to see "
            "whether the merge-back actually completed after that."
        )


if __name__ == "__main__":
    main()
