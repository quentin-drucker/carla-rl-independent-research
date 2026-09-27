"""test16___far_cross_swerve_and_merge_back.py

Demonstrates a full swerve-avoid-and-merge-back cycle using ONLY already-
validated, unmodified code -- no changes to braking authority, hazard
governance, or any safety-relevant control logic.

Context (2026-09-27): Quentin watched test13 and noticed the ego always
just stops for the pedestrian rather than visibly swerving around it and
resuming. The instinct was that the LiDAR hazard corridor governing
braking needs to "move with" the swerve. That specific idea -- letting
steering suppress/override original-lane braking -- is exactly the
approach already tried TWICE (2026-09-18) and reverted both times: the
first attempt produced a dangerous false clearance; the hardened retry
fixed that but introduced governing-source chatter, reduced recovery
success, worsened one outcome, and nearly doubled jerk. The Week 3 plan
explicitly lists a third such attempt as a Non-goal pending real design
review. This script does NOT do that.

The actual, safer explanation: every existing test (test5/test6/test7/
test8/test13/test14) uses ScenarioConfig(walker_cross="near") -- the
pedestrian walks to lane center and STAYS there. Original-lane braking is
deliberately anchored to the pedestrian's own position, not the ego's, so
a stationary in-lane pedestrian correctly forces the ego to a full stop
regardless of how well it swerves -- that's the safety net working as
designed, not a bug. The already-built, already-validated
HazardClearRecoveryController (test5) DOES support a full swerve-out ->
hold -> hazard-clear -> return-to-lane cycle; it has just never been
exercised against a pedestrian who actually finishes crossing and exits
the lane. That requires walker_cross="far" -- and no existing script uses
it. This script does, with generous timing so the pedestrian very likely
clears the lane before the ego's original-lane braking would force it all
the way to zero.

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
    print("test16: far-crossing pedestrian -- full swerve/avoid/merge-back demo")
    print("Watch the console for drive_mode and lateral offset over time.")
    print("Expected: swerve out (~1.5m) while the pedestrian crosses, brief")
    print("slow-down (may or may not reach a full stop depending on timing),")
    print("then lateral offset ramps back toward 0.0m once the pedestrian")
    print("clears the original lane -- that ramp-back IS the merge-back.")
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
