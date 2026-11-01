"""
circuit.py

Pure pieces of the city circuit (run_circuit.py, Week 8, 2026-10-25): the
ego drives laps of a city route; pedestrians appear at random points ahead of
it; each one is an encounter handled by the trained policy plus merge-back,
after which the car keeps driving to the next one.

- straight_flags: route points far enough from curves and junctions, because
  the passage frame assumes a straight road near the hazard.
- place_hazards: random, spaced hazard positions along the route.
- sample_hazard: which pedestrian and how much warning (oracle onset TTC).
- encounter_done: when one encounter is over and the next can be set up.
- lane_change_offsets, without_backtracking: turn the route planner's lane changes (sideways jumps,
  points behind the previous one) into a smooth path.

Offline-tested in tests/test_circuit.py.
"""

import math
import random
from typing import List, Optional, Sequence, Tuple

WALKERS = ("right_edge", "left_edge", "stationary")
TTC_RANGE_S = (0.6, 1.5)  # inside the trained range (0.5-1.6), away from its edges


def _turn_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def straight_flags(yaws_deg: Sequence[float], junction: Sequence[bool], *, window: int = 15,
                   max_turn_deg: float = 10.0) -> List[bool]:
    """Per route point: True if every point within `window` points either side
    is outside a junction and heads within max_turn_deg of this one."""
    n = len(yaws_deg)
    out = []
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        out.append(not any(junction[lo:hi]) and all(_turn_deg(yaws_deg[j], yaws_deg[i]) <= max_turn_deg
                                                     for j in range(lo, hi)))
    return out


def lane_change_offsets(xy: Sequence[Sequence[float]], lane_yaws_deg: Sequence[float], junction: Sequence[bool], *,
                        fixed: Optional[Sequence[bool]] = None, ramp_m: float = 60.0,
                        min_step_m: float = 0.25) -> List[float]:
    """Per route point: lateral offset (m, + = right, along the point's lane right vector) that turns the
    route planner's lane changes into smooth ones. The planner steps 1.5-7 m sideways within a few metres
    (the car swerved at those spots every lap, Quentin 2026-11-01). Each sideways step outside junctions
    (a segment's component across its lane heading > min_step_m) is spread over a smoothstep ramp of up to
    ramp_m centred on it: points before the step move toward the new lane, points after it back toward the
    old. fixed: points whose offset must stay 0 (hazard zones). Neither a ramp nor its offsets enter those or
    junctions (finish the turn, then change lanes): a ramp blocked on one side slides to the other, and is
    shortened only when there is not ramp_m of room between them."""
    n = len(xy)
    fixed = fixed or [False] * n
    s = [0.0]  # distance ALONG the lanes: a jump segment's sideways length would squeeze the ramp
    for i, (a, b) in enumerate(zip(xy, xy[1:])):
        y = math.radians(lane_yaws_deg[i])
        s.append(s[-1] + max(0.0, math.cos(y) * (b[0] - a[0]) + math.sin(y) * (b[1] - a[1])))  # backward: 0
    out = [0.0] * n
    for i in range(n - 1):
        if junction[i] or junction[i + 1]:
            continue
        y = math.radians(lane_yaws_deg[i])
        dx, dy = xy[i + 1][0] - xy[i][0], xy[i + 1][1] - xy[i][1]
        side = math.cos(y) * dy - math.sin(y) * dx  # CARLA is left-handed: + = right
        if abs(side) <= min_step_m:
            continue
        mid = (s[i] + s[i + 1]) / 2.0
        lo = next((s[j] for j in range(i, -1, -1) if fixed[j] or junction[j]), -math.inf)
        hi = next((s[j] for j in range(i + 1, n) if fixed[j] or junction[j]), math.inf)
        start = max(lo, min(mid - ramp_m / 2.0, hi - ramp_m))  # blocked on one side: slide to the other
        end = min(hi, start + ramp_m)
        if end - start < 1e-6:
            continue
        for k in range(n):
            if start < s[k] < end:
                x = (s[k] - start) / (end - start)
                out[k] += side * (x * x * (3.0 - 2.0 * x) - (1.0 if k > i else 0.0))
    return out


def without_backtracking(xy: Sequence[Sequence[float]], lane_yaws_deg: Sequence[float], *,
                         min_ahead_m: float = 0.1) -> List[Tuple[float, float]]:
    """Same points, except those not at least min_ahead_m ahead (along the lane) of the last point that was:
    they are replaced by interpolation between their neighbours that are. The planner's lane-change waypoint
    can sit ~2-4 m BEHIND the previous one, and route legs repeat their joint point, so the path doubled back."""
    out = [tuple(p) for p in xy]
    last = 0
    for i in range(1, len(xy)):
        y = math.radians(lane_yaws_deg[last])
        if math.cos(y) * (xy[i][0] - xy[last][0]) + math.sin(y) * (xy[i][1] - xy[last][1]) >= min_ahead_m:
            for k in range(last + 1, i):
                t = (k - last) / (i - last)
                out[k] = (xy[last][0] + t * (xy[i][0] - xy[last][0]), xy[last][1] + t * (xy[i][1] - xy[last][1]))
            last = i
    return out


def place_hazards(route_s: Sequence[float], ok: Sequence[bool], *, n: int, min_gap_m: float, start_m: float,
                  end_margin_m: float, rng: random.Random) -> List[float]:
    """Up to n hazard positions (distance along the route, ascending) on valid
    points, at least min_gap_m apart, between start_m and end - end_margin_m."""
    end = route_s[-1] - end_margin_m
    candidates = [s for s, good in zip(route_s, ok) if good and start_m <= s <= end]
    rng.shuffle(candidates)
    chosen: List[float] = []
    for s in candidates:
        if all(abs(s - c) >= min_gap_m for c in chosen):
            chosen.append(s)
            if len(chosen) == n:
                break
    return sorted(chosen)


def sample_hazard(rng: random.Random) -> dict:
    return {"walker": rng.choice(WALKERS), "ttc_s": rng.uniform(*TTC_RANGE_S)}


def curve_speed_limit_mps(route_s: Sequence[float], route_yaws_deg: Sequence[float], i_now: int, *,
                          v_max_mps: float, lookahead_m: float = 60.0, a_lat_mps2: float = 3.0,
                          a_dec_mps2: float = 2.5, span_m: float = 10.0) -> float:
    """Cruise speed allowed now at route point i_now: the classical
    lane-follower does not slow for curves (Town04's highway never needed it;
    the first city lap ran wide at a corner at 35 mph and wedged). For every
    curve within lookahead_m: v_curve = sqrt(a_lat / curvature), reachable by
    braking at a_dec over the distance to it."""
    s0, v = route_s[i_now], v_max_mps
    n = len(route_s)
    for j in range(i_now, n):
        if route_s[j] - s0 > lookahead_m:
            break
        k = j
        while k + 1 < n and route_s[k] - route_s[j] < span_m:
            k += 1
        ds = route_s[k] - route_s[j]
        if ds <= 0.0:
            continue
        kappa = math.radians(_turn_deg(route_yaws_deg[k], route_yaws_deg[j])) / ds
        if kappa > 1e-4:
            v = min(v, math.sqrt(a_lat_mps2 / kappa + 2.0 * a_dec_mps2 * (route_s[j] - s0)))
    return v


def encounter_done(t_since_onset_s: float, merge_back_state: Optional[str], *, window_s: float = 8.0,
                   max_s: float = 20.0) -> bool:
    """After the scoring window, once the car is back in its lane (or held
    after a contact); never later than max_s (e.g. waiting beside a person)."""
    if t_since_onset_s >= max_s:
        return True
    return t_since_onset_s >= window_s and merge_back_state in (None, "returned", "holding_after_contact")


def training_spots(route_s: Sequence[float], ok: Sequence[bool], *, approach_m: float = 160.0, after_m: float = 30.0,
                   exclude_s: Sequence[float] = (), exclude_m: float = 60.0) -> List[int]:
    """Route indices usable as a TRAINING hazard: every point from approach_m
    (+20 m to reach speed) before it to after_m beyond it is straight and
    outside junctions (the training env has no curve speed limit), and it is
    more than exclude_m from every held-out evaluation hazard."""
    n = len(route_s)
    bad_prefix = [0]
    for good in ok:
        bad_prefix.append(bad_prefix[-1] + (not good))
    spots, lo, hi = [], 0, 0
    for i in range(n):
        while lo < n and route_s[lo] < route_s[i] - approach_m - 20.0:
            lo += 1
        while hi < n and route_s[hi] <= route_s[i] + after_m:
            hi += 1
        if route_s[i] - route_s[0] < approach_m + 20.0 or hi >= n:
            continue
        if bad_prefix[hi] - bad_prefix[lo] == 0 and all(abs(route_s[i] - e) >= exclude_m for e in exclude_s):
            spots.append(i)
    return spots
