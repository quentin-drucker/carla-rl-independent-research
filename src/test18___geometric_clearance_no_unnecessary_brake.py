"""test18___geometric_clearance_no_unnecessary_brake.py

Validates the geometric ego-clearance override (ego_clearance_override.py,
wired opt-in into lane_follow_step() / run_scenario()) against a STATIONARY
lane-center pedestrian -- same scenario as test17, but with
use_geometric_clearance_override=True.

This is the follow-up to test17's second finding: original-lane braking
never looked at the ego's own actual position, so a correctly-swerved-
around pedestrian still forced a full stop. Quentin asked for the LiDAR
corridor itself to "move with" the ego -- that specific mechanism is the
already-flagged, twice-reverted Non-goal. This tests a DIFFERENT mechanism
for the same category of change (explicit sign-off obtained 2026-09-27):
a live, ground-truth ego-position-vs-pedestrian-position clearance check,
not a second LiDAR corridor reading.

STATUS (2026-09-27): this has NOT yet demonstrated the behavior it was built
to test. Three live-tested iterations of the override are recorded in
ego_clearance_override.py's module docstring -- read that before trusting
anything below. The current (third) version is SAFE (0 ground-truth contact
in both cases) but FUNCTIONALLY INERT here: both cases still come back
full_stop, identical to test17, because the override's "pedestrian must
already be behind the ego" requirement can never be satisfied before the
untouched original-lane corridor has already braked the ego to a stop.

Two cases (current observed behavior, not the originally intended one):
    A. correctly-computed offset -- full_stop, 0 contact ticks (same as
       test17 case A; the override did not change this outcome).
    B. deliberately UNDERSIZED offset -- full_stop, 0 contact ticks (the
       original-lane braking backstop, unaffected by any of this).

Ground truth for contact uses pedestrian_contact.py, independent of CARLA's
own collision sensor (proven unreliable by test16).

Usage (CARLA must already be running, windowed so you can watch):
    python -X utf8 test18___geometric_clearance_no_unnecessary_brake.py
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

    result = run_scenario(
        cfg,
        plot_after=False,
        lateral_offset_fn=recovery_controller,
        monitor_lateral_corridors=True,
        tick_observer=observer,
        post_crossing_settle_s=30.0,
        use_geometric_clearance_override=True,
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
    required_offset_m = compute_required_clearance_offset_m(
        pedestrian_lateral_m=0.0, side_sign=+1,
    )
    print(f"Computed required clearance offset: {required_offset_m:.3f}m")
    case_a = _run_case(label="A: correctly-computed offset (should clear, NOT full-stop)",
                        peak_offset_m=required_offset_m)

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
