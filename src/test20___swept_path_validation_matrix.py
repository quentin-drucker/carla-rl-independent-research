"""test20___swept_path_validation_matrix.py

Week 3: a small validation matrix for the ego-rooted swept-path clearance
experiment (swept_path_clearance.py / oriented_clearance.py, wired opt-in
via use_swept_path_clearance_override), per the handoff report's
"RECOMMENDED NEXT WORK" item 4. test18 and test19 each demonstrated ONE
configuration; this exercises several axes at once so a working case isn't
mistaken for the general case:

    1. Both swerve directions (right and left).
    2. Two speeds (25 mph, 35 mph).
    3. Two TTCs (5.0s generous, 3.0s tight).
    4. Both stationary and moving (far-crossing) pedestrians.
    5. A deliberately BLOCKED escape path (a stationary "other vehicle"
       parked exactly where the swerve would go) -- the swept-path
       occupancy gate should refuse to confirm clear, not silently ignore
       the obstruction.
    6. A repeated run of the baseline case, to sanity-check determinism.

This is still a small sample (11 runs), not an exhaustive sweep -- treat it
as "characterized a bit," not "fully validated." Every run is checked
against pedestrian_contact.py's oriented ground-truth contact detector
(independent of CARLA's own collision sensor, proven unreliable by test16),
not just the printed outcome string.

Usage (CARLA must already be running, windowed so you can watch):
    python -X utf8 test20___swept_path_validation_matrix.py
"""

from scenario_config import ScenarioConfig
from test3___ped_intrusion_scenario import run_scenario
from test5___scripted_pedestrian_steering import build_evasive_offset_fn
from reactive_avoidance import compute_required_clearance_offset_m
from pedestrian_contact import detect_oriented_contact_ticks

# Matches test18's finding: the pure geometric minimum left almost no
# allowance for real path-tracking error, so a buffered margin is used as
# the declared test parameter (not a hidden controller constant).
PLANNED_CLEARANCE_MARGIN_M = 0.6
SHIFT_DURATION_S = 0.5


def _make_observer():
    positions = {"ego": [], "pedestrian": []}
    swept_hazard_ticks = {"count": 0}

    def observer(sim_time_s, triggered, telemetry):
        if telemetry is None:
            positions["ego"].append(None)
            positions["pedestrian"].append(None)
            return
        x_m, y_m, yaw_deg = (
            telemetry.get("pos_x_m"), telemetry.get("pos_y_m"), telemetry.get("yaw_deg")
        )
        pdx, pdy = telemetry.get("pedestrian_x_m"), telemetry.get("pedestrian_y_m")
        positions["ego"].append(
            (x_m, y_m, yaw_deg) if None not in (x_m, y_m, yaw_deg) else None
        )
        positions["pedestrian"].append((pdx, pdy) if pdx is not None else None)

        swept_distance_m = telemetry.get("d_min_ego_swept_path_m")
        trigger_distance_m = telemetry.get("trigger_distance_m")
        if (
            swept_distance_m is not None
            and trigger_distance_m is not None
            and swept_distance_m < trigger_distance_m
        ):
            swept_hazard_ticks["count"] += 1

    return observer, positions, swept_hazard_ticks


def _run_case(*, label, cfg, peak_offset_m, other_vehicle_offset_m=None, post_crossing_settle_s):
    print(f"\n{'=' * 70}\nCASE: {label}\n{'=' * 70}")
    recovery_controller = build_evasive_offset_fn(
        peak_offset_m=peak_offset_m, shift_duration_s=SHIFT_DURATION_S
    )
    observer, positions, swept_hazard_ticks = _make_observer()

    result = run_scenario(
        cfg,
        plot_after=False,
        lateral_offset_fn=recovery_controller,
        monitor_lateral_corridors=True,
        tick_observer=observer,
        post_crossing_settle_s=post_crossing_settle_s,
        use_swept_path_clearance_override=True,
        other_vehicle_offset_m=other_vehicle_offset_m,
    )

    contact_ticks = detect_oriented_contact_ticks(positions["ego"], positions["pedestrian"])
    summary = {
        "label": label,
        "outcome": result.outcome,
        "collision_detected": result.collision_detected,
        "contact_ticks": len(contact_ticks),
        "min_ped_distance_m": result.min_ped_distance_m,
        "recovered": recovery_controller.recovered,
        "swept_hazard_ticks": swept_hazard_ticks["count"],
    }
    print(
        f"  outcome={summary['outcome']} collision={summary['collision_detected']} "
        f"contact_ticks={summary['contact_ticks']} "
        f"min_ped_dist={summary['min_ped_distance_m']:.2f}m "
        f"recovered={summary['recovered']} "
        f"swept_hazard_ticks={summary['swept_hazard_ticks']}"
    )
    return summary


def _stationary_cfg(*, target_mph, trigger_ttc_s):
    return ScenarioConfig(
        walker_speed_mps=1.8,
        walker_side="left",
        walker_cross="stationary",
        trigger_ttc_s=trigger_ttc_s,
        target_mph=target_mph,
        encounter_distance_m=90.0,
        brake_profile="exponential",
        weather_preset="ClearSunset",
        sun_altitude_deg=0,
        sim_seconds=25.0,
    )


def _far_cross_cfg(*, walker_side, target_mph=25.0, trigger_ttc_s=5.0):
    return ScenarioConfig(
        walker_speed_mps=1.8,
        walker_side=walker_side,
        walker_cross="far",
        trigger_ttc_s=trigger_ttc_s,
        target_mph=target_mph,
        encounter_distance_m=90.0,
        brake_profile="exponential",
        weather_preset="ClearSunset",
        sun_altitude_deg=0,
        sim_seconds=20.0,
    )


def main():
    results = []

    # 1. Baseline: stationary pedestrian, swerve RIGHT, correct offset.
    offset_right = compute_required_clearance_offset_m(
        pedestrian_lateral_m=0.0, side_sign=+1, safety_margin_m=PLANNED_CLEARANCE_MARGIN_M
    )
    results.append(_run_case(
        label="1. Stationary, swerve RIGHT, 25mph, TTC=5.0s (baseline)",
        cfg=_stationary_cfg(target_mph=25.0, trigger_ttc_s=5.0),
        peak_offset_m=offset_right,
        post_crossing_settle_s=30.0,
    ))

    # 2. Mirror direction: swerve LEFT.
    offset_left = compute_required_clearance_offset_m(
        pedestrian_lateral_m=0.0, side_sign=-1, safety_margin_m=PLANNED_CLEARANCE_MARGIN_M
    )
    results.append(_run_case(
        label="2. Stationary, swerve LEFT, 25mph, TTC=5.0s (mirror direction)",
        cfg=_stationary_cfg(target_mph=25.0, trigger_ttc_s=5.0),
        peak_offset_m=offset_left,
        post_crossing_settle_s=30.0,
    ))

    # 3. Higher speed.
    results.append(_run_case(
        label="3. Stationary, swerve RIGHT, 35mph, TTC=5.0s (higher speed)",
        cfg=_stationary_cfg(target_mph=35.0, trigger_ttc_s=5.0),
        peak_offset_m=offset_right,
        post_crossing_settle_s=30.0,
    ))

    # 4. Tighter TTC.
    results.append(_run_case(
        label="4. Stationary, swerve RIGHT, 25mph, TTC=3.0s (tighter timing)",
        cfg=_stationary_cfg(target_mph=25.0, trigger_ttc_s=3.0),
        peak_offset_m=offset_right,
        post_crossing_settle_s=30.0,
    ))

    # 5. Moving/far-crossing pedestrian, swerve RIGHT with a computed
    #    (not guessed) offset -- test19 used the historically-unsafe 1.5m
    #    on purpose as a backstop case; this checks the "should actually
    #    clear efficiently" side of the same scenario.
    offset_far_right = compute_required_clearance_offset_m(
        pedestrian_lateral_m=2.55, side_sign=+1, safety_margin_m=PLANNED_CLEARANCE_MARGIN_M
    )
    results.append(_run_case(
        label="5. Far-crossing pedestrian, swerve RIGHT, computed offset",
        cfg=_far_cross_cfg(walker_side="left"),
        peak_offset_m=offset_far_right,
        post_crossing_settle_s=5.0,
    ))

    # 6. Mirror direction, moving pedestrian.
    offset_far_left = compute_required_clearance_offset_m(
        pedestrian_lateral_m=-2.55, side_sign=-1, safety_margin_m=PLANNED_CLEARANCE_MARGIN_M
    )
    results.append(_run_case(
        label="6. Far-crossing pedestrian, swerve LEFT, computed offset (mirror)",
        cfg=_far_cross_cfg(walker_side="right"),
        peak_offset_m=offset_far_left,
        post_crossing_settle_s=5.0,
    ))

    # 7. test16/test19's original unsafe-offset backstop case, repeated here
    #    for one-table comparison against the computed-offset case above.
    results.append(_run_case(
        label="7. Far-crossing pedestrian, swerve RIGHT, UNDERSIZED 1.5m (test16/19 backstop)",
        cfg=_far_cross_cfg(walker_side="left"),
        peak_offset_m=1.5,
        post_crossing_settle_s=5.0,
    ))

    # 8. Blocked escape path: a stationary "other vehicle" parked exactly at
    #    the swerve's target offset. The swept-path occupancy gate must
    #    refuse to confirm clear -- braking should NOT release just because
    #    ground-truth pedestrian clearance and LiDAR happen to look fine.
    results.append(_run_case(
        label="8. Stationary pedestrian, swerve RIGHT BLOCKED by another vehicle",
        cfg=_stationary_cfg(target_mph=25.0, trigger_ttc_s=5.0),
        peak_offset_m=offset_right,
        other_vehicle_offset_m=offset_right,
        post_crossing_settle_s=30.0,
    ))

    # 9. Repeatability check: rerun case 1 verbatim.
    results.append(_run_case(
        label="9. Repeat of case 1 (determinism check)",
        cfg=_stationary_cfg(target_mph=25.0, trigger_ttc_s=5.0),
        peak_offset_m=offset_right,
        post_crossing_settle_s=30.0,
    ))

    print("\n" + "=" * 70)
    print("MATRIX SUMMARY")
    print("=" * 70)
    for r in results:
        verdict = "CONTACT" if r["contact_ticks"] > 0 else "clear"
        print(
            f"  {r['label']}\n"
            f"      -> {verdict} ({r['contact_ticks']} contact ticks) | "
            f"outcome={r['outcome']} | recovered={r['recovered']} | "
            f"min_ped_dist={r['min_ped_distance_m']:.2f}m | "
            f"swept_hazard_ticks={r['swept_hazard_ticks']}"
        )
    any_contact = any(r["contact_ticks"] > 0 for r in results)
    print("\n" + ("AT LEAST ONE CASE HAD REAL CONTACT" if any_contact else "No contact in any case (0/9)."))


if __name__ == "__main__":
    main()
