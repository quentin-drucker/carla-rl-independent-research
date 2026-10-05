"""
passage.py

The "passage": the lateral range across the road that the ego's CENTER can
occupy at a given point ahead while its whole body stays on drivable surface
and clear of obstacles (Phase 2, 2026-10-05; Prof. Izmirli's idea, recorded
in MASTER "Design artifact: tightening lateral constraints" -> Resolution and
"Phase 2 design decisions").

Coordinates are route-relative lateral offsets in meters, +right / -left
(same convention as lane_follow_step(lateral_offset_m=...)); 0 is the route
center line.

The steering action is a passage coordinate u in [-1, +1]:
    u = 0   -> lateral target 0 (keep the normal path)
    u = +1  -> the rightmost admissible center position (the passage's right edge)
    u = -1  -> the leftmost admissible center position (the passage's left edge)
    between -> proportional (u * edge on that side)
Extremes are Izmirli's "aim at the extreme edge" rule. Intermediate values
may point into an obstacle's exclusion zone; learning when not to do that is
the policy's job.

Obstacles are circles at a lateral offset (the pedestrian now; another car
later), so a second obstacle just narrows the passage. For a crossing
pedestrian the caller should pass its PREDICTED lateral position when the ego
reaches its line (Phase 4); a stationary pedestrian needs no prediction.

Pure: no CARLA imports. Drivability enters as a probe callable so the CARLA
side (map_drivability.classify_point_drivability_seam_tolerant at points
offset from the route) stays in the caller. Offline-tested in
tests/test_passage.py.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from pedestrian_contact import DEFAULT_PEDESTRIAN_RADIUS_M, EGO_HALF_WIDTH_M

DEFAULT_SIDE_MARGIN_M = 0.3       # clearance kept from road edges and obstacles
DEFAULT_SCAN_MAX_M = 8.0          # how far either side of the route to look for road
DEFAULT_SCAN_STEP_M = 0.05
DEFAULT_MAX_GAP_M = 0.10          # non-drivable runs this short are treated as lane seams


@dataclass(frozen=True)
class Obstacle:
    lateral_m: float
    radius_m: float = DEFAULT_PEDESTRIAN_RADIUS_M


@dataclass(frozen=True)
class Passage:
    road_left_m: Optional[float]     # drivable surface edge (body may reach it), None = unknown
    road_right_m: Optional[float]
    intervals: Tuple[Tuple[float, float], ...]  # admissible ego-CENTER intervals, left to right
    left_edge_m: Optional[float]     # leftmost admissible center position (u = -1)
    right_edge_m: Optional[float]    # rightmost admissible center position (u = +1)

    @property
    def empty(self) -> bool:
        return not self.intervals

    def admits(self, lateral_m: float) -> bool:
        return any(lo <= lateral_m <= hi for lo, hi in self.intervals)


def drivable_extent_m(
    is_drivable: Callable[[float], Optional[bool]],
    *,
    scan_max_m: float = DEFAULT_SCAN_MAX_M,
    step_m: float = DEFAULT_SCAN_STEP_M,
    max_gap_m: float = DEFAULT_MAX_GAP_M,
) -> Tuple[Optional[float], Optional[float]]:
    """(left_m, right_m): the contiguous drivable surface around lateral 0.

    Scans outward from the route center in steps. A non-drivable run no
    longer than max_gap_m that is followed by drivable surface is bridged
    (Town04_Opt has ~2 cm gaps between adjacent driving lanes). A None probe
    result ("unknown") ends the scan on that side conservatively. Returns
    (None, None) if the route center itself is not drivable.
    """
    if is_drivable(0.0) is not True:
        return None, None

    def scan(direction: int) -> float:
        last_good = 0.0
        gap_start = None
        n = int(round(scan_max_m / step_m))
        for i in range(1, n + 1):
            lat = direction * i * step_m
            ok = is_drivable(lat)
            if ok is True:
                last_good, gap_start = lat, None
                continue
            if ok is None:
                break
            if gap_start is None:
                gap_start = lat
            if abs(lat - gap_start) + step_m > max_gap_m:
                break
        return last_good

    return scan(-1), scan(+1)


def admissible_center_intervals(
    road_left_m: float,
    road_right_m: float,
    obstacles: Sequence[Obstacle] = (),
    *,
    ego_half_width_m: float = EGO_HALF_WIDTH_M,
    margin_m: float = DEFAULT_SIDE_MARGIN_M,
) -> List[Tuple[float, float]]:
    """Ego-center intervals inside the road (shrunk by half-width + margin)
    minus each obstacle's exclusion zone (radius + half-width + margin)."""
    lo, hi = road_left_m + ego_half_width_m + margin_m, road_right_m - ego_half_width_m - margin_m
    if lo > hi:
        return []
    intervals = [(lo, hi)]
    for ob in obstacles:
        reach = ob.radius_m + ego_half_width_m + margin_m
        cut_lo, cut_hi = ob.lateral_m - reach, ob.lateral_m + reach
        nxt = []
        for a, b in intervals:
            if cut_hi <= a or cut_lo >= b:
                nxt.append((a, b))
                continue
            if a < cut_lo:
                nxt.append((a, cut_lo))
            if cut_hi < b:
                nxt.append((cut_hi, b))
        intervals = nxt
    return sorted(intervals)


def compute_passage(
    is_drivable: Callable[[float], Optional[bool]],
    obstacles: Sequence[Obstacle] = (),
    *,
    ego_half_width_m: float = EGO_HALF_WIDTH_M,
    margin_m: float = DEFAULT_SIDE_MARGIN_M,
    scan_max_m: float = DEFAULT_SCAN_MAX_M,
    step_m: float = DEFAULT_SCAN_STEP_M,
    max_gap_m: float = DEFAULT_MAX_GAP_M,
) -> Passage:
    road_left, road_right = drivable_extent_m(is_drivable, scan_max_m=scan_max_m, step_m=step_m, max_gap_m=max_gap_m)
    if road_left is None:
        return Passage(None, None, (), None, None)
    intervals = admissible_center_intervals(road_left, road_right, obstacles,
                                            ego_half_width_m=ego_half_width_m, margin_m=margin_m)
    return Passage(
        road_left_m=road_left, road_right_m=road_right, intervals=tuple(intervals),
        left_edge_m=intervals[0][0] if intervals else None,
        right_edge_m=intervals[-1][1] if intervals else None,
    )


def passage_coordinate_to_offset_m(u: float, passage: Passage, *, u_min: float = -1.0, u_max: float = 1.0) -> float:
    """Map the action u to a route-relative lateral target in meters.

    u is clipped to [u_min, u_max] (right-swerve-only = u_min 0.0). Positive
    u scales toward the right edge, negative toward the left edge; a side
    with no admissible room (edge on the other side of 0, or empty passage)
    maps to 0, i.e. no swerve that way.
    """
    if not -1.0 <= u_min <= u_max <= 1.0:
        raise ValueError("need -1 <= u_min <= u_max <= 1")
    u = min(max(float(u), u_min), u_max)
    if passage.empty:
        return 0.0
    if u >= 0.0:
        return u * max(passage.right_edge_m, 0.0)
    return -u * min(passage.left_edge_m, 0.0)
