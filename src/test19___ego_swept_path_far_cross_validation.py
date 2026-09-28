"""Focused regression for test16's outside-original-corridor collision.

This keeps test16's historically unsafe 1.5m swerve and far-crossing walker,
but opts into the ego-rooted swept-path safety experiment. The intended path
intersects the pedestrian's +2.55m endpoint, so the swept LiDAR tube must add
braking even after the original +/-1.4m route corridor loses the pedestrian.

Success here means no oriented-footprint contact; a full stop is acceptable
because this is the unsafe-path backstop case, not the clear-path efficiency
case covered by test18 A.
"""

from pedestrian_contact import detect_oriented_contact_ticks
from scenario_config import ScenarioConfig
from test3___ped_intrusion_scenario import run_scenario
from test5___scripted_pedestrian_steering import build_evasive_offset_fn


def main():
    samples = {"ego": [], "pedestrian": [], "swept_hazard_ticks": 0}

    def observer(sim_time_s, triggered, telemetry):
        if telemetry is None:
            samples["ego"].append(None)
            samples["pedestrian"].append(None)
            return
        x_m = telemetry.get("pos_x_m")
        y_m = telemetry.get("pos_y_m")
        yaw_deg = telemetry.get("yaw_deg")
        pedestrian_x_m = telemetry.get("pedestrian_x_m")
        pedestrian_y_m = telemetry.get("pedestrian_y_m")
        samples["ego"].append(
            (x_m, y_m, yaw_deg)
            if x_m is not None and y_m is not None and yaw_deg is not None
            else None
        )
        samples["pedestrian"].append(
            (pedestrian_x_m, pedestrian_y_m)
            if pedestrian_x_m is not None and pedestrian_y_m is not None
            else None
        )
        swept_distance_m = telemetry.get("d_min_ego_swept_path_m")
        trigger_distance_m = telemetry.get("trigger_distance_m")
        if (
            swept_distance_m is not None
            and trigger_distance_m is not None
            and swept_distance_m < trigger_distance_m
        ):
            samples["swept_hazard_ticks"] += 1

    cfg = ScenarioConfig(
        walker_speed_mps=1.8,
        walker_side="left",
        walker_cross="far",
        trigger_ttc_s=5.0,
        target_mph=25.0,
        encounter_distance_m=90.0,
        brake_profile="exponential",
        weather_preset="ClearSunset",
        sun_altitude_deg=0,
        sim_seconds=20.0,
    )
    recovery_controller = build_evasive_offset_fn(peak_offset_m=1.5)
    result = run_scenario(
        cfg,
        plot_after=False,
        lateral_offset_fn=recovery_controller,
        monitor_lateral_corridors=True,
        tick_observer=observer,
        post_crossing_settle_s=5.0,
        use_swept_path_clearance_override=True,
    )
    contact_ticks = detect_oriented_contact_ticks(
        samples["ego"], samples["pedestrian"]
    )

    print("\n" + "=" * 70)
    print("test19 swept-path far-cross regression")
    print(f"Outcome: {result.outcome}")
    print(f"CARLA collision event: {result.collision_detected}")
    print(f"Oriented contact ticks: {len(contact_ticks)}")
    print(f"Swept-path LiDAR hazard ticks: {samples['swept_hazard_ticks']}")
    print(f"Minimum pedestrian center distance: {result.min_ped_distance_m:.2f}m")
    print(f"Recovery command completed: {recovery_controller.recovered}")
    print(f"Ego physically returned to route: {result.physically_returned_to_route}")
    print("=" * 70)


if __name__ == "__main__":
    main()
