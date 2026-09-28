"""test21___swerve_into_occupied_lane_validation.py

test13 (2026-09-27, Week 3 Workstream 1.2) validated candidate-path vehicle
occupancy as OBSERVATIONAL TELEMETRY ONLY -- it explicitly did not gate
braking or steering. Now that use_swept_path_clearance_override actually
reads transition_path_occupancy/commanded_path_occupancy to decide whether
to release original-lane braking, and the swept-path LiDAR tube can
ADDITIVELY trigger braking on its own, this question has real behavioral
stakes: if the ego swerves toward a lane with a stationary obstacle (e.g. a
parked car) sitting in it, does it actually brake for that obstacle, or
does it just fail to release the pedestrian-braking gate while never really
"seeing" the car itself?

This reuses test18 case A's exact scenario (a stationary lane-center
pedestrian, correctly-computed swerve offset that normally clears cleanly
with no full stop -- see test18/test20 case 1) but adds a stationary
"other vehicle" parked exactly at that swerve's target offset, 6m past the
pedestrian encounter point (test3's existing other_vehicle_offset_m
mechanism, previously only exercised observationally by test13).

Reports, separately:
  - transition/commanded path occupancy statuses actually observed
  - how many ticks the swept-path LiDAR tube itself registered the car as
    a hazard (d_min_ego_swept_path_m < trigger_distance_m) -- this is the
    direct answer to "does the swept-path sensor actually see it"
  - whether the ego ever braked meaningfully, and its final outcome
  - ground-truth minimum ego-to-parked-car distance (from the newly-added
    other_vehicle_x_m/y_m telemetry), not just a printed outcome string
  - ground-truth ego-to-pedestrian contact (pedestrian_contact.py, oriented)

This is intentionally NOT isolated from the pedestrian hazard (removing the
pedestrian would also remove the trigger that starts the scripted swerve in
the first place) -- so a full stop here could be caused by the pedestrian
gate never releasing, the car's own LiDAR hazard, or both. The per-tick
breakdown below is what actually distinguishes those cases; don't rely on
the outcome string alone.

Usage (CARLA must already be running, windowed so you can watch):
    python -X utf8 test21___swerve_into_occupied_lane_validation.py
"""

import math

from scenario_config import ScenarioConfig
from test3___ped_intrusion_scenario import run_scenario
from test5___scripted_pedestrian_steering import build_evasive_offset_fn
from reactive_avoidance import compute_required_clearance_offset_m
from pedestrian_contact import detect_oriented_contact_ticks

PLANNED_CLEARANCE_MARGIN_M = 0.6
SHIFT_DURATION_S = 0.5


def main():
    samples = {
        "ego": [], "pedestrian": [], "other_vehicle_xy": None,
        "swept_hazard_ticks": 0, "occupancy_statuses": set(),
        "swept_detection_ticks": 0,
        "drivability_statuses": set(), "min_ego_to_car_m": float("inf"),
    }

    def observer(sim_time_s, triggered, telemetry):
        if telemetry is None:
            samples["ego"].append(None)
            samples["pedestrian"].append(None)
            return
        x_m, y_m, yaw_deg = (
            telemetry.get("pos_x_m"), telemetry.get("pos_y_m"), telemetry.get("yaw_deg")
        )
        pdx, pdy = telemetry.get("pedestrian_x_m"), telemetry.get("pedestrian_y_m")
        samples["ego"].append(
            (x_m, y_m, yaw_deg) if None not in (x_m, y_m, yaw_deg) else None
        )
        samples["pedestrian"].append((pdx, pdy) if pdx is not None else None)

        ovx, ovy = telemetry.get("other_vehicle_x_m"), telemetry.get("other_vehicle_y_m")
        if ovx is not None and ovy is not None:
            samples["other_vehicle_xy"] = (ovx, ovy)
            if x_m is not None and y_m is not None:
                d = math.hypot(ovx - x_m, ovy - y_m)
                samples["min_ego_to_car_m"] = min(samples["min_ego_to_car_m"], d)

        swept_distance_m = telemetry.get("d_min_ego_swept_path_m")
        if telemetry.get("ego_swept_path_lidar_status") != "no_return":
            samples["swept_detection_ticks"] += 1
        trigger_distance_m = telemetry.get("trigger_distance_m")
        if (
            swept_distance_m is not None
            and trigger_distance_m is not None
            and swept_distance_m < trigger_distance_m
        ):
            samples["swept_hazard_ticks"] += 1

        occupancy = telemetry.get("transition_path_occupancy")
        if occupancy is not None:
            samples["occupancy_statuses"].add(occupancy.get("status"))
        drivability = telemetry.get("transition_path_drivability")
        if drivability is not None:
            samples["drivability_statuses"].add(drivability.get("status"))

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
    offset_m = compute_required_clearance_offset_m(
        pedestrian_lateral_m=0.0, side_sign=+1, safety_margin_m=PLANNED_CLEARANCE_MARGIN_M
    )
    print(f"Swerve target offset: {offset_m:.3f}m -- parking a car exactly there.")

    recovery_controller = build_evasive_offset_fn(
        peak_offset_m=offset_m, shift_duration_s=SHIFT_DURATION_S
    )
    result = run_scenario(
        cfg,
        plot_after=False,
        lateral_offset_fn=recovery_controller,
        monitor_lateral_corridors=True,
        tick_observer=observer,
        post_crossing_settle_s=30.0,
        use_swept_path_clearance_override=True,
        other_vehicle_offset_m=offset_m,
    )

    ped_contact_ticks = detect_oriented_contact_ticks(samples["ego"], samples["pedestrian"])

    print("\n" + "=" * 70)
    print("test21 swerve-into-occupied-lane result")
    print("=" * 70)
    print(f"Outcome: {result.outcome}")
    print(f"CARLA collision event: {result.collision_detected}")
    print(f"Pedestrian oriented-contact ticks (ground truth): {len(ped_contact_ticks)}")
    print(f"Min ego<->pedestrian distance: {result.min_ped_distance_m:.2f}m")
    if samples["min_ego_to_car_m"] != float("inf"):
        print(f"Min ego<->parked-car distance: {samples['min_ego_to_car_m']:.2f}m")
    else:
        print("Min ego<->parked-car distance: never recorded (car position telemetry missing)")
    print(f"Swept-path LiDAR hazard ticks (car should trip this): {samples['swept_hazard_ticks']}")
    print(f"Swept-path LiDAR detection ticks (any in-tube return): {samples['swept_detection_ticks']}")
    print(f"transition_path_occupancy statuses observed: {samples['occupancy_statuses']}")
    print(f"transition_path_drivability statuses observed: {samples['drivability_statuses']}")
    print(f"Recovery command completed: {recovery_controller.recovered}")
    print(f"Ego physically returned to route: {result.physically_returned_to_route}")
    print(f"Final measured route offset: {result.final_route_lateral_offset_m:+.2f}m")
    print("=" * 70)


if __name__ == "__main__":
    main()
