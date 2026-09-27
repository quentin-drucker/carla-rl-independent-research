"""test22___swerve_into_occupied_lane_wide_separation.py

Follow-up to test21: Quentin raised a real concern -- test21 used a swerve
offset (1.98m) not much larger than the swept-path tube's own half-width
(~1.38m: ego_half_width_m 1.082 + pedestrian_radius_m 0.3), so a full stop
there could plausibly be explained by the swept-path tube itself brushing
the pedestrian (still standing at lane center, 0.0m) during the swerve's
ramp-up, rather than by the parked car specifically. test21's own numbers
argued against that (only 1 swept-path hazard tick total, and the ORIGINAL
corridor is unambiguously centered on the pedestrian at 0.0m regardless of
any swerve) -- but "argued against" is not the same as "ruled out", and this
test makes the separation impossible to argue about.

Uses a much larger swerve offset (~4.88m, computed with a big declared
margin) and parks the car at that SAME large offset -- the pedestrian at
0.0m is now more than 3m from the near edge of the swept-path tube for
nearly the entire encounter, not just after the ramp completes. If a full
stop happens now, per-tick telemetry (not just a summary outcome) shows
DIRECTLY, tick by tick, whether the hazard came from the (always-centered-
on-the-pedestrian) original corridor, the swept-path tube, or both, and
what the ego's actual lateral offset was at that exact moment -- so
"still near 0.0m, must be about the pedestrian" vs "already near 4.88m,
must be about the car" is a fact read off the trace, not an inference.

Usage (CARLA must already be running, windowed so you can watch):
    python -X utf8 test22___swerve_into_occupied_lane_wide_separation.py
"""

import math

from scenario_config import ScenarioConfig
from test3___ped_intrusion_scenario import run_scenario
from test5___scripted_pedestrian_steering import build_evasive_offset_fn
from reactive_avoidance import compute_required_clearance_offset_m
from pedestrian_contact import detect_oriented_contact_ticks

# Deliberately large: pedestrian sits at 0.0m, so this pushes the swerve
# target (and the parked car) to ~4.88m -- more than 3m clear of the swept-
# path tube's ~1.38m half-width even at the pedestrian's position, for the
# entire encounter, not just once the ramp completes.
LARGE_CLEARANCE_MARGIN_M = 3.5
SHIFT_DURATION_S = 0.5


def main():
    trace = []  # per-tick control/sensor evidence after trigger
    samples = {
        "ego": [], "pedestrian": [], "min_ego_to_car_m": float("inf"),
        "swept_hazard_ticks": 0, "original_hazard_ticks": 0,
        "swept_detection_ticks": 0,
        "active_swept_path_ticks": 0,
        "first_active_path_handoff": None,
        "first_swept_hazard": None,
        "hazard_source_counts": {},
        "occupancy_statuses": set(),
        "max_target_lidar_height_counts": {"above": 0, "middle": 0, "below": 0},
        "closest_target_lidar_stats": None,
        "closest_target_projection_stats": None,
        "closest_target_center_to_path_m": None,
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
        if ovx is not None and ovy is not None and x_m is not None and y_m is not None:
            ego_to_car_m = math.hypot(ovx - x_m, ovy - y_m)
            if ego_to_car_m < samples["min_ego_to_car_m"]:
                samples["min_ego_to_car_m"] = ego_to_car_m
                samples["closest_target_lidar_stats"] = telemetry.get(
                    "other_vehicle_lidar_height_stats"
                )
                samples["closest_target_projection_stats"] = telemetry.get(
                    "other_vehicle_lidar_path_projection_stats"
                )
                samples["closest_target_center_to_path_m"] = telemetry.get(
                    "other_vehicle_center_to_swept_path_m"
                )

        target_stats = telemetry.get("other_vehicle_lidar_height_stats")
        if target_stats:
            counts = samples["max_target_lidar_height_counts"]
            counts["above"] = max(
                counts["above"], target_stats["count_above_minus_1_0"]
            )
            counts["middle"] = max(
                counts["middle"],
                target_stats["count_between_minus_1_8_and_minus_1_0"],
            )
            counts["below"] = max(
                counts["below"], target_stats["count_below_minus_1_8"]
            )

        trigger_distance_m = telemetry.get("trigger_distance_m")
        d_min_original_m = telemetry.get("d_min_original_path_m")
        d_min_swept_m = telemetry.get("d_min_ego_swept_path_m")
        hazard_brake_cmd = telemetry.get("hazard_brake_cmd")
        signed_offset_m = telemetry.get("signed_route_lateral_offset_m")
        requested_offset_m = telemetry.get("lateral_offset_requested_m")
        drive_mode = telemetry.get("drive_mode")
        active_braking_path = telemetry.get("active_braking_path")
        hazard_source = telemetry.get("hazard_governing_source")

        if active_braking_path == "ego_swept":
            samples["active_swept_path_ticks"] += 1
            if samples["first_active_path_handoff"] is None:
                samples["first_active_path_handoff"] = {
                    "time_s": sim_time_s,
                    "actual_offset_m": signed_offset_m,
                    "original_distance_m": d_min_original_m,
                    "swept_distance_m": d_min_swept_m,
                    "drive_mode": drive_mode,
                }

        if triggered:
            trace.append((
                sim_time_s, signed_offset_m, requested_offset_m,
                d_min_original_m, d_min_swept_m, hazard_brake_cmd, drive_mode,
                active_braking_path,
            ))

        if (
            d_min_swept_m is not None and trigger_distance_m is not None
            and d_min_swept_m < trigger_distance_m
        ):
            samples["swept_hazard_ticks"] += 1
            if samples["first_swept_hazard"] is None:
                samples["first_swept_hazard"] = {
                    "time_s": sim_time_s,
                    "actual_offset_m": signed_offset_m,
                    "distance_m": d_min_swept_m,
                    "trigger_distance_m": trigger_distance_m,
                }
        if hazard_brake_cmd == 1.0:
            samples["hazard_source_counts"][hazard_source] = (
                samples["hazard_source_counts"].get(hazard_source, 0) + 1
            )
        if telemetry.get("ego_swept_path_lidar_status") != "no_return":
            samples["swept_detection_ticks"] += 1
        if (
            d_min_original_m is not None and trigger_distance_m is not None
            and d_min_original_m < trigger_distance_m
        ):
            samples["original_hazard_ticks"] += 1

        occupancy = telemetry.get("transition_path_occupancy")
        if occupancy is not None:
            samples["occupancy_statuses"].add(occupancy.get("status"))

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
        pedestrian_lateral_m=0.0, side_sign=+1, safety_margin_m=LARGE_CLEARANCE_MARGIN_M
    )
    print(f"Swerve target / parked-car offset: {offset_m:.3f}m (pedestrian stays at 0.0m)")

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
    print("test22 wide-separation swerve-into-occupied-lane result")
    print("=" * 70)
    print(f"Outcome: {result.outcome}")
    print(f"CARLA collision event: {result.collision_detected}")
    print(f"Pedestrian oriented-contact ticks (ground truth): {len(ped_contact_ticks)}")
    print(f"Min ego<->pedestrian distance: {result.min_ped_distance_m:.2f}m")
    if samples["min_ego_to_car_m"] != float("inf"):
        print(f"Min ego<->parked-car distance: {samples['min_ego_to_car_m']:.2f}m")
    print(f"ORIGINAL-corridor hazard ticks: {samples['original_hazard_ticks']}")
    print(f"SWEPT-PATH hazard ticks: {samples['swept_hazard_ticks']}")
    print(f"SWEPT-PATH detection ticks (any in-tube return): {samples['swept_detection_ticks']}")
    print(f"SWEPT-PATH active-authority ticks: {samples['active_swept_path_ticks']}")
    print(f"First active-path handoff: {samples['first_active_path_handoff']}")
    print(f"First swept-path hazard: {samples['first_swept_hazard']}")
    print(f"Active hazard-source counts: {samples['hazard_source_counts']}")
    print(f"transition_path_occupancy statuses observed: {samples['occupancy_statuses']}")
    print(
        "Max raw LiDAR returns within 3m of parked car by local-Z band "
        f"(>=-1.0, -1.8..-1.0, <-1.8): {samples['max_target_lidar_height_counts']}"
    )
    print(
        "Raw LiDAR target stats at closest ego/car approach: "
        f"{samples['closest_target_lidar_stats']}"
    )
    print(
        "Target-return projection stats at closest approach: "
        f"{samples['closest_target_projection_stats']}"
    )
    print(
        "Parked-car CENTER distance to swept path at closest approach: "
        f"{samples['closest_target_center_to_path_m']}"
    )
    print(f"Recovered (returned to lane): {recovery_controller.recovered}")

    print("\n--- Per-tick trace while hazard braking was commanded (hazard_brake_cmd=1) ---")
    print(
        f"{'t_s':>6} {'actual_m':>9} {'request_m':>9} "
        f"{'d_orig_m':>9} {'d_swept_m':>10} {'mode':>12}"
    )
    printed = 0
    for (t, offset, requested, d_orig, d_swept, brake_cmd, mode, active_path) in trace:
        if brake_cmd == 1.0:
            d_orig_str = f"{d_orig:.2f}" if d_orig is not None else "None"
            d_swept_str = f"{d_swept:.2f}" if d_swept is not None else "None"
            offset_str = f"{offset:.2f}" if offset is not None else "None"
            requested_str = f"{requested:.2f}" if requested is not None else "None"
            print(
                f"{t:6.2f} {offset_str:>9} {requested_str:>9} "
                f"{d_orig_str:>9} {d_swept_str:>10} {mode:>12} {active_path}"
            )
            printed += 1
            if printed >= 40:
                print("  ... (truncated after 40 rows)")
                break
    if printed == 0:
        print("  (no ticks with hazard_brake_cmd=1 -- hazard braking never engaged)")
    print("=" * 70)


if __name__ == "__main__":
    main()
