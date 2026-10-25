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

Offline-tested in tests/test_circuit.py.
"""

import random
from typing import List, Optional, Sequence

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


def encounter_done(t_since_onset_s: float, merge_back_state: Optional[str], *, window_s: float = 8.0,
                   max_s: float = 20.0) -> bool:
    """After the scoring window, once the car is back in its lane (or held
    after a contact); never later than max_s (e.g. waiting beside a person)."""
    if t_since_onset_s >= max_s:
        return True
    return t_since_onset_s >= window_s and merge_back_state in (None, "returned", "holding_after_contact")
