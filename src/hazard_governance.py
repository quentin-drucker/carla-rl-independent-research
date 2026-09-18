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

# How much wider than the normal corridor half-width the CORROBORATING
# governance check uses. See select_hazard_governing_distance() docstring
# for why this exists: a single normal-width corridor reading was found
# live (2026-09-18) to read a persistent false "clear" -- not a one-tick
# blip -- while the ego actually passed a pedestrian at ~2m/0.37s TTC,
# because the pedestrian sat just outside the normal corridor's lateral
# band the whole time. A wider corroborating band catches that specific
# failure (the miss distance was ~1.45-1.5m from centerline; this margin
# pushes the corroborating band to 2.4m, comfortably past it) without
# being so wide it can never confirm clearance in a real near-cross case.
GOVERNANCE_WIDTH_MARGIN_M = 1.0


def select_hazard_governing_distance(
    *,
    d_min_original_path_m,
    d_min_transition_path_m,
    d_min_transition_path_wide_m,
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
      - the NORMAL-width transition corridor reads LiDAR-clear past
        trigger_distance_m (None -- no return -- treated as clear, same
        convention the original corridor uses);
      - a SEPARATE, WIDER corroborating corridor
        (d_min_transition_path_wide_m, queried at normal half-width +
        GOVERNANCE_WIDTH_MARGIN_M) ALSO reads clear past trigger_distance_m.
        This is the fix for the 2026-09-18 regression: requiring two
        independently-computed corridors of different widths to agree
        means a pedestrian sitting just outside the normal corridor's
        narrow band -- the exact failure found live -- gets caught by the
        wider one instead of producing a silent false "clear";
      - the (normal) transition corridor is confirmed "drivable" by the map
        check. "non_drivable" or "unknown" both fall back to the original
        corridor -- LiDAR clearance alone is never treated as proof an
        escape path is safe.

    The distance RETURNED when shifting is still the normal-width corridor's
    own reading (not the wider one) -- the wider corridor is only a gate on
    whether to trust the shift at all, not the graduated braking response
    once it's trusted.

    If any condition fails, this returns the original corridor's own
    distance unchanged, exactly matching pre-existing (and pre-2026-09-18)
    behavior.
    """
    if (
        monitor_lateral_corridors
        and abs(requested_lateral_offset_m) > offset_commit_threshold_m
    ):
        transition_lidar_clear = (
            d_min_transition_path_m is None
            or d_min_transition_path_m > trigger_distance_m
        )
        transition_wide_lidar_clear = (
            d_min_transition_path_wide_m is None
            or d_min_transition_path_wide_m > trigger_distance_m
        )
        transition_is_drivable = (
            transition_path_drivability is not None
            and transition_path_drivability.get("status") == "drivable"
        )
        if transition_lidar_clear and transition_wide_lidar_clear and transition_is_drivable:
            return d_min_transition_path_m, "transition"

    return d_min_original_path_m, "original"


def is_original_corridor_confirmed_clear(
    d_min_original_path_m, trigger_distance_m, clear_margin_m
):
    """True only if the original corridor POSITIVELY confirms clearance.

    Unlike lane_follow_step()'s own hazard_active/hazard_clear checks (which
    treat a None LiDAR reading as "nothing detected = clear" in every mode
    except STOP_HOLD), this always requires an actual measured distance
    above trigger_distance_m + clear_margin_m. A None reading (no LiDAR
    return at all) is treated as unknown, not clear.

    Why stricter here: this feeds a scripted evasive maneuver's decision to
    start returning toward a hazard it just steered away from -- the same
    "silence near a close, possibly-stationary pedestrian is more likely a
    sensor gap than genuine clearance" reasoning lane_follow_step() already
    applies in STOP_HOLD, generalized here since this check only ever runs
    in an analogous close-range, recently-triggered context. Found via a
    live oscillation bug on 2026-09-18: treating None as clear let a
    momentary LiDAR gap next to a stationary pedestrian satisfy the
    recovery controller's clear-confirmation window, causing it to start
    returning and then immediately re-arm when the hazard reappeared.
    """
    return (
        d_min_original_path_m is not None
        and d_min_original_path_m > (trigger_distance_m + clear_margin_m)
    )
