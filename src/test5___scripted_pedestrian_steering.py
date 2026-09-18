"""Visual test of scripted steering around one pedestrian encounter.

The original route corridor continues to own emergency braking. A second green
corridor shows the commanded lateral path (a constant parallel offset). If that
path contains a LiDAR return, its edges turn magenta. A third, gold/orange
corridor shows the swept transition path -- a corridor that starts at the
ego's actual measured lateral offset and blends toward the commanded target
over a bounded forward distance, approximating the vehicle's real swept path
during the maneuver rather than a lane the ego is not yet on. All corridors
remain observational; only the original corridor owns braking. The terminal
prints original, commanded, transition, left-candidate, and right-candidate
minimum distances every half second after the trigger.

Route recovery is hazard/state-based, not timer-based: the scripted offset
holds at its peak until the pedestrian is measured behind the ego or the
original-route LiDAR corridor reads clear for a short, stable interval, then
ramps back to zero. A maximum hold duration remains as a fail-safe fallback
only -- see HazardClearRecoveryController below.

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


class HazardClearRecoveryController:
    """Stateful signed lateral-offset schedule driven by hazard-clear state.

    State machine: SHIFT_OUT -> HOLD_AVOIDANCE -> HAZARD_CLEAR -> RETURN ->
    RECOVERED. Route recovery (leaving HOLD_AVOIDANCE) begins only once the
    pedestrian is measured behind the ego, or the original-route LiDAR
    corridor has read clear for ``clear_confirm_s`` continuous seconds --
    never merely because a timer expired. ``max_hold_s`` is a fail-safe
    timeout in case the hazard signal never clears; it is recorded via
    ``used_fallback_timeout`` when it fires so it can be distinguished from a
    normal hazard-clear recovery.
    """

    def __init__(
        self,
        peak_offset_m,
        *,
        shift_duration_s=1.0,
        return_duration_s=2.0,
        clear_confirm_s=0.5,
        max_hold_s=8.0,
    ):
        self.peak_offset_m = peak_offset_m
        self.shift_duration_s = shift_duration_s
        self.return_duration_s = return_duration_s
        self.clear_confirm_s = clear_confirm_s
        self.max_hold_s = max_hold_s

        self.state = "IDLE"
        self._state_entered_s = None
        self._hold_started_s = None
        self._clear_since_s = None
        self._return_started_s = None
        self._return_via_fallback = False

        # Diagnostics, read after the run.
        self.recovered = False
        self.used_fallback_timeout = False
        self.hazard_reappeared_during_recovery = False
        self.recovery_time_s = None  # RETURN duration once recovery completes
        self.time_hazard_clear_to_recover_start_s = None

    def __call__(self, sim_time_s, triggered, trigger_time_s, hazard_clear_info):
        if not triggered or trigger_time_s is None:
            return 0.0

        hazard_clear_now = (
            hazard_clear_info.get("pedestrian_behind_ego", False)
            or hazard_clear_info.get("original_corridor_clear", False)
        )

        if self.state == "IDLE":
            self.state = "SHIFT_OUT"
            self._state_entered_s = sim_time_s

        if self.state == "SHIFT_OUT":
            elapsed_s = sim_time_s - self._state_entered_s
            if elapsed_s >= self.shift_duration_s:
                self.state = "HOLD_AVOIDANCE"
                self._hold_started_s = sim_time_s
                self._clear_since_s = None
                return self.peak_offset_m
            return self.peak_offset_m * _smoothstep(elapsed_s / self.shift_duration_s)

        if self.state == "HOLD_AVOIDANCE":
            if hazard_clear_now:
                if self._clear_since_s is None:
                    self._clear_since_s = sim_time_s
                elif (sim_time_s - self._clear_since_s) >= self.clear_confirm_s:
                    self.time_hazard_clear_to_recover_start_s = (
                        sim_time_s - self._clear_since_s
                    )
                    self.state = "HAZARD_CLEAR"
                    self._return_via_fallback = False
                    return self.peak_offset_m
            else:
                self._clear_since_s = None

            if (sim_time_s - self._hold_started_s) >= self.max_hold_s:
                self.used_fallback_timeout = True
                self.state = "HAZARD_CLEAR"
                self._return_via_fallback = True

            return self.peak_offset_m

        if self.state == "HAZARD_CLEAR":
            self.state = "RETURN"
            self._return_started_s = sim_time_s
            return self.peak_offset_m

        if self.state == "RETURN":
            # A fallback-timeout recovery is a deliberate override precisely
            # because the hazard signal never confirmed clear -- re-arming on
            # "not clear" here would make the failsafe re-trigger forever and
            # never actually recover. Only a recovery that started from a
            # genuine confirmed hazard-clear re-arms if the hazard reappears.
            if not hazard_clear_now and not self._return_via_fallback:
                self.hazard_reappeared_during_recovery = True
                self.state = "HOLD_AVOIDANCE"
                self._hold_started_s = sim_time_s
                self._clear_since_s = None
                return self.peak_offset_m

            elapsed_s = sim_time_s - self._return_started_s
            if elapsed_s >= self.return_duration_s:
                self.state = "RECOVERED"
                self.recovered = True
                self.recovery_time_s = elapsed_s
                return 0.0
            return self.peak_offset_m * (1.0 - _smoothstep(elapsed_s / self.return_duration_s))

        # RECOVERED
        return 0.0


def build_evasive_offset_fn(peak_offset_m, **kwargs):
    """Return a HazardClearRecoveryController for the given peak offset.

    Kept as a thin factory (rather than exposing the class directly at every
    call site) so existing callers -- test6's batch runner, this module's own
    main() -- pick up hazard-based recovery without changing their call
    shape. Extra keyword arguments are forwarded to the controller.
    """
    return HazardClearRecoveryController(peak_offset_m, **kwargs)


def _format_distance(value):
    return "  -- " if value is None else f"{value:5.1f}"


class CorridorConsoleObserver:
    def __init__(self, fixed_dt=0.02):
        self._print_every_ticks = max(1, int(0.5 / fixed_dt))
        self._tick = 0
        self.maximum_actual_offset_m = 0.0
        self.final_actual_offset_m = float("nan")
        self.commanded_ever_non_drivable = False
        self.transition_ever_non_drivable = False
        self.ever_governed_by_transition = False

    def __call__(self, sim_time_s, triggered, telemetry):
        if telemetry is None:
            return

        actual_offset_m = telemetry["signed_route_lateral_offset_m"]
        self.maximum_actual_offset_m = max(
            self.maximum_actual_offset_m, abs(actual_offset_m)
        )
        self.final_actual_offset_m = actual_offset_m

        if telemetry.get("hazard_governing_source") == "transition":
            self.ever_governed_by_transition = True

        commanded_drivability = telemetry.get("commanded_path_drivability")
        transition_drivability = telemetry.get("transition_path_drivability")
        commanded_status = (
            commanded_drivability["status"] if commanded_drivability else "--"
        )
        transition_status = (
            transition_drivability["status"] if transition_drivability else "--"
        )
        if commanded_status == "non_drivable":
            self.commanded_ever_non_drivable = True
        if transition_status == "non_drivable":
            self.transition_ever_non_drivable = True

        if triggered and self._tick % self._print_every_ticks == 0:
            print(
                f"[corridors] t={sim_time_s:5.2f}s "
                f"request={telemetry['lateral_offset_requested_m']:+4.2f}m "
                f"actual={actual_offset_m:+4.2f}m | "
                f"original={_format_distance(telemetry['d_min_original_path_m'])}m "
                f"commanded={_format_distance(telemetry['d_min_commanded_path_m'])}m "
                f"transition={_format_distance(telemetry['d_min_transition_path_m'])}m "
                f"left={_format_distance(telemetry['d_min_left_candidate_m'])}m "
                f"right={_format_distance(telemetry['d_min_right_candidate_m'])}m | "
                f"drivable(commanded={commanded_status}, transition={transition_status}) | "
                f"brake_governed_by={telemetry.get('hazard_governing_source')} "
                f"mode={telemetry.get('drive_mode')}"
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
    recovery_controller = build_evasive_offset_fn(args.lateral_offset_m)
    result = run_scenario(
        config,
        plot_after=False,
        lateral_offset_fn=recovery_controller,
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
        "commanded path ever map-flagged non_drivable: "
        f"{observer.commanded_ever_non_drivable}"
    )
    print(
        "transition path ever map-flagged non_drivable: "
        f"{observer.transition_ever_non_drivable}"
    )
    print(
        "braking hazard ever governed by the swept-transition path "
        f"(not just the original lane): {observer.ever_governed_by_transition}"
    )
    print(f"recovery controller final state: {recovery_controller.state}")
    print(f"recovery completed (hazard/state-based): {recovery_controller.recovered}")
    if recovery_controller.time_hazard_clear_to_recover_start_s is not None:
        print(
            "time from hazard-clear confirmation to recovery start: "
            f"{recovery_controller.time_hazard_clear_to_recover_start_s:.2f} s"
        )
    if recovery_controller.recovery_time_s is not None:
        print(f"recovery ramp-down duration: {recovery_controller.recovery_time_s:.2f} s")
    print(f"fallback hold-timeout used: {recovery_controller.used_fallback_timeout}")
    print(
        "hazard reappeared during recovery: "
        f"{recovery_controller.hazard_reappeared_during_recovery}"
    )
    print(
        "Reminder: original-route LiDAR still owns braking; the commanded, "
        "transition, and candidate corridor readings are observational only."
    )


if __name__ == "__main__":
    main()
