"""test13___candidate_path_occupancy_validation.py

Week 3, Workstream 1.2: live validation of the candidate-path
vehicle-occupancy check (vehicle_occupancy.py), wired into lane_follow_step()
behind commanded_path_occupancy / transition_path_occupancy telemetry.

Per the Week 3 plan, this validates the four required fixed cases:
    1. No adjacent vehicle -- both corridors should read "clear".
    2. An actor clearly inside the candidate (commanded) path -- should
       read "occupied".
    3. An actor just outside the candidate path -- should read "clear".
    4. A deliberately unavailable side -- should read "unknown", never
       silently promoted to "clear".

Cases 1-3 are run for BOTH left and right scripted evasive directions
(matching the plan's "fixed left/right cases" wording); case 4 is
demonstrated once, live, using the same real actor-gathering path
(vehicle_occupancy.gather_occupancy_actors against the live CARLA world)
combined with a deliberately withheld corridor, so the live integration
(not just the already offline-tested pure geometry in
test_vehicle_occupancy.py) is exercised end to end.

This is observational-signal validation only -- occupancy does not gate
braking or steering this week, matching the map-drivability precedent.

Run CARLA (windowed, so you can watch), then from ``src``:

    python -X utf8 test13___candidate_path_occupancy_validation.py
"""

from scenario_config import ScenarioConfig
from test3___ped_intrusion_scenario import run_scenario
from test5___scripted_pedestrian_steering import build_evasive_offset_fn
from vehicle_occupancy import check_corridor_occupancy, gather_occupancy_actors

CANDIDATE_OFFSET_M = 1.5  # matches test5/test7's validated scripted offset
CLEARLY_INSIDE_EXTRA_M = 0.0  # spawn AT the candidate offset -- clearly inside

# The occupancy threshold is corridor_half_width_m + actor_occupancy_radius_m.
# Corridor half-width is NOODLE_HALF_WIDTH_M (1.4m, see lane_follow.py). The
# Tesla Model 3's occupancy radius (confirmed live via
# vehicle.bounding_box.extent: x=2.396m, y=1.082m) is
# max(2.396, 1.082) + 0.3m safety margin = ~2.70m -- an initial choice of
# 3.5m extra here (total 5.0m from route center) was NOT far enough past
# the ~4.10m threshold and was live-caught as a false "occupied" before
# this constant was corrected.
#
# A second live probe found the LEFT side of this route stretch has a
# physical obstruction (consistent with the guardrail found in the Week 3
# rollover substudy, test11): total offset -7.5m failed to spawn at all
# (try_spawn_actor returned None), while -7.0m spawned cleanly. 5.5m extra
# (total 7.0m from route center) clears the ~4.10m occupancy threshold
# with a full meter of margin while staying spawnable on both sides.
JUST_OUTSIDE_EXTRA_M = 5.5


class OccupancyObserver:
    """Tracks commanded/transition occupancy status across a run."""

    def __init__(self):
        self.commanded_statuses = set()
        self.transition_statuses = set()
        self.ticks_with_commanded_occupied = 0
        self.ticks_observed = 0

    def __call__(self, sim_time_s, triggered, telemetry):
        if telemetry is None:
            return
        commanded = telemetry.get("commanded_path_occupancy")
        transition = telemetry.get("transition_path_occupancy")
        if commanded is not None:
            self.commanded_statuses.add(commanded["status"])
            self.ticks_observed += 1
            if commanded["status"] == "occupied":
                self.ticks_with_commanded_occupied += 1
        if transition is not None:
            self.transition_statuses.add(transition["status"])


def _run_case(*, label, direction_offset_m, other_vehicle_offset_m):
    print(f"\n{'=' * 70}\nCASE: {label}\n{'=' * 70}")
    config = ScenarioConfig(
        walker_speed_mps=1.8,
        walker_side="left",
        walker_cross="near",
        trigger_ttc_s=4.0,
        target_mph=15.0,
        encounter_distance_m=60.0,
        brake_profile="exponential",
        weather_preset="ClearSunset",
        sun_altitude_deg=0,
        sim_seconds=16.0,
    )
    observer = OccupancyObserver()
    recovery_controller = build_evasive_offset_fn(direction_offset_m)
    result = run_scenario(
        config,
        plot_after=False,
        lateral_offset_fn=recovery_controller,
        monitor_lateral_corridors=True,
        tick_observer=observer,
        post_crossing_settle_s=4.0,
        other_vehicle_offset_m=other_vehicle_offset_m,
    )
    print(
        f"-> commanded_statuses_seen={observer.commanded_statuses} "
        f"transition_statuses_seen={observer.transition_statuses} "
        f"ticks_commanded_occupied={observer.ticks_with_commanded_occupied}/{observer.ticks_observed} "
        f"collision={result.collision_detected} outcome={result.outcome}"
    )
    return observer


def _case_4_unavailable_side_live():
    """Demonstrates the "unknown" branch live, end-to-end, using the real
    actor-gathering path against the live CARLA world -- combined with a
    deliberately withheld (None) corridor, since that combination is what
    a genuinely unavailable side looks like at the call site in
    lane_follow.py (see check_corridor_occupancy's docstring for why an
    empty/None corridor must never be promoted to "clear").
    """
    print(f"\n{'=' * 70}\nCASE: deliberately unavailable side (live)\n{'=' * 70}")
    import carla
    from carla_session import connect_and_load_world, enable_sync_mode, restore_async_mode

    client, world = connect_and_load_world(host="localhost", port=2000, timeout_s=10.0, target_map="Town04_Opt")
    enable_sync_mode(world, fixed_dt=0.02)
    spawn_tf = world.get_map().get_spawn_points()[242]
    bp_lib = world.get_blueprint_library()
    vehicle = world.spawn_actor(bp_lib.find("vehicle.tesla.model3"), spawn_tf)
    world.tick()
    blocker_tf = carla.Transform(
        carla.Location(x=spawn_tf.location.x + 10.0, y=spawn_tf.location.y, z=spawn_tf.location.z),
        spawn_tf.rotation,
    )
    blocker = world.try_spawn_actor(bp_lib.find("vehicle.tesla.model3"), blocker_tf)
    world.tick()
    try:
        actors = gather_occupancy_actors(world, vehicle)
        assert len(actors) == 1, f"expected exactly one other actor, found {len(actors)}"
        result = check_corridor_occupancy(None, actors, lateral_half_width_m=1.4)
        print(f"-> real live actors gathered={len(actors)} corridor=None -> status={result['status']}")
        assert result["status"] == "unknown", "an unavailable corridor must never read as clear or occupied"
        print("PASS: unavailable side correctly reports 'unknown', not 'clear'.")
    finally:
        if blocker is not None:
            blocker.destroy()
        vehicle.destroy()
        world.tick()
        restore_async_mode(world)


def main():
    for direction_name, direction_offset_m in (("right", CANDIDATE_OFFSET_M), ("left", -CANDIDATE_OFFSET_M)):
        sign = 1.0 if direction_offset_m > 0 else -1.0

        obs = _run_case(
            label=f"{direction_name}: no adjacent vehicle",
            direction_offset_m=direction_offset_m,
            other_vehicle_offset_m=None,
        )
        assert "occupied" not in obs.commanded_statuses, (
            f"{direction_name}/no-vehicle: commanded path falsely reported occupied"
        )

        obs = _run_case(
            label=f"{direction_name}: actor clearly inside the candidate path",
            direction_offset_m=direction_offset_m,
            other_vehicle_offset_m=direction_offset_m + CLEARLY_INSIDE_EXTRA_M,
        )
        assert obs.ticks_with_commanded_occupied > 0, (
            f"{direction_name}/inside: commanded path never reported occupied despite a blocking actor on it"
        )

        obs = _run_case(
            label=f"{direction_name}: actor just outside the candidate path",
            direction_offset_m=direction_offset_m,
            other_vehicle_offset_m=sign * (abs(direction_offset_m) + JUST_OUTSIDE_EXTRA_M),
        )
        assert obs.ticks_with_commanded_occupied == 0, (
            f"{direction_name}/outside: commanded path falsely reported occupied for a clearly-clear actor"
        )

    _case_4_unavailable_side_live()

    print("\n" + "#" * 70)
    print("ALL CANDIDATE-PATH OCCUPANCY VALIDATION CASES PASSED")
    print("#" * 70)


if __name__ == "__main__":
    main()
