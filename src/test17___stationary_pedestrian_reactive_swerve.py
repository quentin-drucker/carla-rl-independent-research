"""test17___stationary_pedestrian_reactive_swerve.py

Tests swerving around a STATIONARY pedestrian standing in the ego's
original lane, then merging back -- the simpler case Quentin asked for
after test16's far-crossing pedestrian produced an undetected collision
(root cause: a fixed, guessed peak_offset_m that happened to undershoot
where the pedestrian actually was).

Unlike test16, this pedestrian's position is fixed and known in advance
(ScenarioConfig(walker_cross="stationary"), lane center), so the fix here
needs no new reactive/stateful controller -- just computing the RIGHT
offset (reactive_avoidance.compute_required_clearance_offset_m) instead
of guessing one, then handing it to the existing, unmodified
build_evasive_offset_fn / HazardClearRecoveryController. No changes to
braking authority or hazard governance.

Two cases:
    A. correctly-computed offset -- should clear the pedestrian and merge
       back with no contact.
    B. a deliberately UNDERSIZED offset -- confirms original-lane braking
       still engages and stops the ego short, since a stationary
       lane-center pedestrian never leaves the corridor's watched band
       (unlike test16's far-crossing case). This is the "obviously it
       should still brake" backstop check.

Ground truth for contact uses pedestrian_contact.py (independent of
CARLA's own collision sensor, which test16 showed can miss real contact).

Usage (CARLA must already be running, windowed so you can watch):
    python -X utf8 test17___stationary_pedestrian_reactive_swerve.py
"""

from scenario_config import ScenarioConfig
from test3___ped_intrusion_scenario import run_scenario
from test5___scripted_pedestrian_steering import build_evasive_offset_fn
from reactive_avoidance import compute_required_clearance_offset_m
from pedestrian_contact import compute_ego_pedestrian_contact_radius_m, detect_contact_ticks

CONTACT_RADIUS_M = compute_ego_pedestrian_contact_radius_m()


def _make_observer():
    positions = {"ego": [], "pedestrian": []}

    def observer(sim_time_s, triggered, telemetry):
        if telemetry is None:
            positions["ego"].append(None)
            positions["pedestrian"].append(None)
            return
        px, py = telemetry.get("pos_x_m"), telemetry.get("pos_y_m")
        pdx, pdy = telemetry.get("pedestrian_x_m"), telemetry.get("pedestrian_y_m")
        positions["ego"].append((px, py) if px is not None else None)
        positions["pedestrian"].append((pdx, pdy) if pdx is not None else None)

    return observer, positions


def _run_case(*, label, peak_offset_m):
    print(f"\n{'=' * 70}\nCASE: {label} (peak_offset_m={peak_offset_m:.2f})\n{'=' * 70}")
    cfg = ScenarioConfig(
        walker_speed_mps=1.8,
        walker_side="left",
        walker_cross="stationary",
        trigger_ttc_s=5.0,
        target_mph=25.0,
        encounter_distance_m=90.0,
        brake_profile="exponential",
        weather_preset="ClearSunset",
        sun_altitude_deg=0,
        sim_seconds=25.0,
    )
    recovery_controller = build_evasive_offset_fn(peak_offset_m=peak_offset_m)
    observer, positions = _make_observer()

    # A stationary pedestrian has zero distance to "cross", so
    # crossing_state["done"] fires immediately at trigger time (not once
    # the ego has actually reached/passed them) -- the normal "settle N
    # seconds after done" termination pattern would end the run long before
    # the ego (still tens of meters away at trigger) ever gets close.
    # post_crossing_settle_s is set generously past sim_seconds so it never
    # cuts the run short; sim_seconds is the real bound here.
    result = run_scenario(
        cfg,
        plot_after=False,
        lateral_offset_fn=recovery_controller,
        monitor_lateral_corridors=True,
        tick_observer=observer,
        post_crossing_settle_s=30.0,
    )

    contact_ticks = detect_contact_ticks(
        positions["ego"], positions["pedestrian"], contact_radius_m=CONTACT_RADIUS_M
    )

    print(f"  CARLA collision_detected: {result.collision_detected}")
    print(f"  Ground-truth contact ticks (independent check): {len(contact_ticks)}")
    print(f"  Outcome: {result.outcome}  min_ped_distance={result.min_ped_distance_m:.2f}m")
    print(f"  Recovered (swerved back to lane): {recovery_controller.recovered}")
    return {"label": label, "contact_ticks": len(contact_ticks), "result": result,
            "recovered": recovery_controller.recovered}


def main():
    # Case A: the offset actually needed to clear a pedestrian standing at
    # lane center (lateral_m=0.0), computed rather than guessed.
    required_offset_m = compute_required_clearance_offset_m(
        pedestrian_lateral_m=0.0, side_sign=+1,
    )
    print(f"Computed required clearance offset: {required_offset_m:.3f}m")
    case_a = _run_case(label="A: correctly-computed offset (should clear, no contact)",
                        peak_offset_m=required_offset_m)

    # Case B: deliberately undersized -- should NOT clear on steering alone,
    # but original-lane braking must still catch it (pedestrian is at lane
    # center, always inside the watched band, unlike test16's far-cross case).
    case_b = _run_case(label="B: undersized offset (braking backstop should still catch it)",
                        peak_offset_m=0.5)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for case in (case_a, case_b):
        verdict = "CONTACT" if case["contact_ticks"] > 0 else "clear"
        print(f"  {case['label']}: {verdict} ({case['contact_ticks']} contact ticks), "
              f"outcome={case['result'].outcome}, recovered={case['recovered']}")


if __name__ == "__main__":
    main()
