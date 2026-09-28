"""test18___geometric_clearance_no_unnecessary_brake.py

Validates the ego-rooted swept-path clearance experiment against a STATIONARY
lane-center pedestrian -- the same scenario as test17, but with
use_swept_path_clearance_override=True.

This is the follow-up to test17's second finding: original-lane braking
never looked at the ego's own actual position, so a correctly-swerved-
around pedestrian still forced a full stop. Quentin asked for the LiDAR
corridor itself to "move with" the ego -- that specific mechanism is the
already-flagged, twice-reverted Non-goal. This tests a DIFFERENT mechanism
for the same category of change (explicit sign-off obtained 2026-09-27):
a single ego-rooted swept path with additive LiDAR hazard detection plus a
ground-truth pedestrian-clearance gate for this controlled scenario, not a
tick-to-tick swap between original and transition corridor owners.

STATUS (2026-09-27): prior circular-distance/behind-ego variants did not
demonstrate the intended behavior. This version evaluates the pedestrian
against one ego-rooted intended swept path, requires the path to be drivable
and unoccupied, and uses its own LiDAR tube as a veto. It remains opt-in and
is not yet a general object tracker.

Two acceptance cases (live-validated 2026-09-27):
    A. correctly-computed offset -- should avoid oriented contact and should
       not full-stop once the actual ego tracks the confirmed-clear path.
    B. deliberately UNDERSIZED offset -- should retain braking/full-stop and
       avoid contact.

Ground truth for contact uses pedestrian_contact.py, independent of CARLA's
own collision sensor (proven unreliable by test16).

Usage (CARLA must already be running, windowed so you can watch):
    python -X utf8 test18___geometric_clearance_no_unnecessary_brake.py
"""

from scenario_config import ScenarioConfig
from test3___ped_intrusion_scenario import run_scenario
from test5___scripted_pedestrian_steering import build_evasive_offset_fn
from reactive_avoidance import compute_required_clearance_offset_m
from pedestrian_contact import detect_oriented_contact_ticks


# The pure geometric minimum is 1.682m with a 0.30m margin, but the live
# controller only reached ~1.75m when asked for a buffered 1.982m path. Using
# the theoretical minimum as the command left effectively no allowance for
# tracking dynamics. This remains a declared test parameter, not a hidden
# controller constant.
PLANNED_CLEARANCE_MARGIN_M = 0.6


def _make_observer():
    positions = {"ego": [], "pedestrian": [], "swept_path": []}

    def observer(sim_time_s, triggered, telemetry):
        if telemetry is None:
            positions["ego"].append(None)
            positions["pedestrian"].append(None)
            positions["swept_path"].append(None)
            return
        px, py = telemetry.get("pos_x_m"), telemetry.get("pos_y_m")
        yaw_deg = telemetry.get("yaw_deg")
        pdx, pdy = telemetry.get("pedestrian_x_m"), telemetry.get("pedestrian_y_m")
        positions["ego"].append(
            (px, py, yaw_deg)
            if px is not None and py is not None and yaw_deg is not None
            else None
        )
        positions["pedestrian"].append((pdx, pdy) if pdx is not None else None)
        drivability = telemetry.get("transition_path_drivability")
        occupancy = telemetry.get("transition_path_occupancy")
        positions["swept_path"].append({
            "sim_time_s": sim_time_s,
            "requested_m": telemetry.get("lateral_offset_requested_m"),
            "actual_m": telemetry.get("signed_route_lateral_offset_m"),
            "clearance_m": telemetry.get("swept_path_clearance_m"),
            "lidar_distance_m": telemetry.get("d_min_ego_swept_path_m"),
            "clear_now": telemetry.get("swept_path_clear_now", False),
            "confirmed_clear": telemetry.get("swept_path_confirmed_clear", False),
            "drivability": drivability.get("status") if drivability else None,
            "occupancy": occupancy.get("status") if occupancy else None,
        })

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
    # The original 1.0s scripted shift did not finish establishing a clear
    # path until ~0.9s after trigger; by then the exponential brake profile
    # had already made a full stop physically unavoidable. This focused test
    # uses a still-smooth but earlier 0.5s commitment. Safety gates and the
    # undersized-offset backstop remain unchanged.
    recovery_controller = build_evasive_offset_fn(
        peak_offset_m=peak_offset_m,
        shift_duration_s=0.5,
    )
    observer, positions = _make_observer()

    result = run_scenario(
        cfg,
        plot_after=False,
        lateral_offset_fn=recovery_controller,
        monitor_lateral_corridors=True,
        tick_observer=observer,
        post_crossing_settle_s=30.0,
        use_swept_path_clearance_override=True,
    )

    contact_ticks = detect_oriented_contact_ticks(
        positions["ego"], positions["pedestrian"]
    )

    print(f"  CARLA collision_detected: {result.collision_detected}")
    print(f"  Ground-truth contact ticks (independent check): {len(contact_ticks)}")
    print(f"  Outcome: {result.outcome}  min_ped_distance={result.min_ped_distance_m:.2f}m")
    print(f"  Recovery command completed: {recovery_controller.recovered}")
    print(f"  Ego physically returned to route: {result.physically_returned_to_route}")
    swept_samples = [sample for sample in positions["swept_path"] if sample is not None]
    clear_samples = [sample for sample in swept_samples if sample["clear_now"]]
    confirmed_samples = [sample for sample in swept_samples if sample["confirmed_clear"]]
    measured_clearances = [
        sample["clearance_m"] for sample in swept_samples
        if sample["clearance_m"] is not None
    ]
    print(
        "  Swept-path gate: "
        f"clear_ticks_observed={len(clear_samples)} "
        f"confirmed_ticks={len(confirmed_samples)} "
        f"clearance_range="
        f"{(min(measured_clearances), max(measured_clearances)) if measured_clearances else None}"
    )
    if clear_samples:
        print(f"  First swept-path clear tick: t={clear_samples[0]['sim_time_s']:.2f}s")
    if confirmed_samples:
        print(
            "  First swept-path confirmed-clear tick: "
            f"t={confirmed_samples[0]['sim_time_s']:.2f}s"
        )
    if swept_samples:
        blocker_counts = {}
        for sample in swept_samples:
            key = (
                sample["drivability"],
                sample["occupancy"],
                sample["lidar_distance_m"] is not None,
            )
            blocker_counts[key] = blocker_counts.get(key, 0) + 1
        print(f"  Swept-path status counts (drivable, occupancy, lidar_hit): {blocker_counts}")
    return {"label": label, "contact_ticks": len(contact_ticks), "result": result,
            "command_recovered": recovery_controller.recovered,
            "physically_recovered": result.physically_returned_to_route}


def main():
    required_offset_m = compute_required_clearance_offset_m(
        pedestrian_lateral_m=0.0,
        side_sign=+1,
        safety_margin_m=PLANNED_CLEARANCE_MARGIN_M,
    )
    print(
        "Computed buffered clearance offset: "
        f"{required_offset_m:.3f}m "
        f"(planned margin={PLANNED_CLEARANCE_MARGIN_M:.2f}m)"
    )
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
              f"outcome={case['result'].outcome}, "
              f"command_recovered={case['command_recovered']}, "
              f"physically_recovered={case['physically_recovered']}")


if __name__ == "__main__":
    main()
