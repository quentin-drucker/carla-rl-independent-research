"""hazard_governance.py

Decides which LiDAR corridor reading governs the braking-hazard decision
for one tick: the ORIGINAL (unswerved) route corridor, or the swept-
transition corridor the ego is actually driving during an evasive
maneuver.

Pulled out of lane_follow_step() as a small pure function so this
safety-relevant decision can be unit-tested directly, without needing a
live CARLA vehicle/world.
"""

OFFSET_COMMIT_THRESHOLD_M = 0.2  # meters; below this, not really steering yet


def select_hazard_governing_distance(
    *,
    d_min_original_path_m,
    d_min_transition_path_m,
    transition_path_drivability,
    requested_lateral_offset_m,
    trigger_distance_m,
    monitor_lateral_corridors,
    offset_commit_threshold_m=OFFSET_COMMIT_THRESHOLD_M,
):
    """Return (distance_m, source) where source is "original" or "transition".

    Default: the original corridor always governs -- the safety-conservative
    baseline that keeps the ego from resuming into a pedestrian who moves
    back into the original lane.

    Only shifts to the swept-transition corridor when ALL of:
      - corridor monitoring is actually enabled this tick (monitor_lateral_
        corridors), since the transition corridor isn't computed otherwise;
      - the ego is actively committed to a nontrivial lateral offset
        (abs(requested_lateral_offset_m) > offset_commit_threshold_m) --
        a near-zero offset means there's no real evasive path to trust yet;
      - the transition corridor itself reads LiDAR-clear past the same
        trigger_distance_m margin used for the original corridor (None --
        no return -- is treated as clear, the same convention the original
        corridor already uses);
      - the transition corridor is confirmed "drivable" by the map check.
        "non_drivable" or "unknown" both fall back to the original corridor
        -- LiDAR clearance alone is never treated as proof an escape path
        is safe.

    If any condition fails, this returns the original corridor's own
    distance unchanged, exactly matching pre-existing behavior.
    """
    if (
        monitor_lateral_corridors
        and abs(requested_lateral_offset_m) > offset_commit_threshold_m
    ):
        transition_lidar_clear = (
            d_min_transition_path_m is None
            or d_min_transition_path_m > trigger_distance_m
        )
        transition_is_drivable = (
            transition_path_drivability is not None
            and transition_path_drivability.get("status") == "drivable"
        )
        if transition_lidar_clear and transition_is_drivable:
            return d_min_transition_path_m, "transition"

    return d_min_original_path_m, "original"
