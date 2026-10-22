"""
walker_script.py

Scripted pedestrian behavior that can CHANGE after the hazard starts (Week 7,
2026-10-21): the question is whether a policy that re-decides every tick can
beat one that commits once, which needs pedestrians whose best response
changes after onset (turn back, start late, pause, change speed).

A script is a JSON-friendly list of phases, run in order from the moment the
walker starts moving (the hazard onset):

    {"kind": "walk", "to_frac": f, "speed_mps": v, "max_s": t (optional)}
        walk along the crossing line toward lateral f * start_lateral_m
        (1 = where they started, 0 = the ego lane center, -1 = the mirror
        position on the far side). Ends when that lateral position is reached
        or passed ("reached"), or after max_s seconds ("max_s"), whichever
        comes first: a turn-back can be a lateral threshold or a delay.
    {"kind": "wait", "s": t}
        stand still for t seconds.

After the last phase the walker stands still. [] = stands still throughout.
Every phase change is recorded in `transitions` (time after onset, lateral
position, reason) so each run's manifest shows exactly when the behavior
changed. Lateral is +right of the ego lane center, in meters.

Pure (no CARLA): CarlaAEBEnv turns (direction, speed) into WalkerControl.
Offline-tested in tests/test_walker_script.py.
"""

from typing import List, Optional, Sequence, Tuple

_KEYS = {"walk": {"kind", "to_frac", "speed_mps", "max_s"}, "wait": {"kind", "s"}}


def validate_walker_script(phases: Sequence[dict]) -> List[dict]:
    """A checked copy of the phases; raises ValueError on anything malformed."""
    out = []
    for i, ph in enumerate(phases):
        kind = ph.get("kind")
        if kind not in _KEYS or not set(ph) <= _KEYS[kind]:
            raise ValueError(f"phase {i}: bad phase {ph!r}")
        if kind == "walk":
            if "to_frac" not in ph or float(ph.get("speed_mps", 0.0)) <= 0.0:
                raise ValueError(f"phase {i}: walk needs to_frac and a positive speed_mps")
            if ph.get("max_s") is not None and float(ph["max_s"]) <= 0.0:
                raise ValueError(f"phase {i}: max_s must be positive")
        elif float(ph.get("s", -1.0)) < 0.0:
            raise ValueError(f"phase {i}: wait needs s >= 0")
        out.append(dict(ph))
    return out


class WalkerScript:
    """Runs one script for one episode. step() is called every tick after
    onset with the walker's measured lateral position."""

    def __init__(self, phases: Sequence[dict], *, start_lateral_m: float):
        self.phases = validate_walker_script(phases)
        if self.phases and abs(start_lateral_m) < 1e-6:
            raise ValueError("to_frac targets need a walker that starts off the lane center")
        self.start_lateral_m = float(start_lateral_m)
        self.index = -1          # -1 = not started; len(phases) = done
        self.transitions: List[dict] = []
        self._phase_start_s = 0.0
        self._direction = 0

    @property
    def done(self) -> bool:
        return self.index >= len(self.phases)

    @property
    def change_time_s(self) -> Optional[float]:
        """Seconds after onset of the first change away from the initial phase."""
        return self.transitions[1]["t_s"] if len(self.transitions) > 1 else None

    def _enter(self, index: int, t_s: float, lateral_m: float, reason: str) -> None:
        self.index, self._phase_start_s = index, t_s
        kind = "done" if self.done else self.phases[index]["kind"]
        if kind == "walk":
            target = self.phases[index]["to_frac"] * self.start_lateral_m
            self._direction = 1 if target > lateral_m else -1
        self.transitions.append({"phase": index, "kind": kind, "t_s": round(t_s, 4),
                                 "lateral_m": round(lateral_m, 4), "reason": reason})

    def _finished(self, t_s: float, lateral_m: float) -> Optional[str]:
        ph, elapsed = self.phases[self.index], t_s - self._phase_start_s + 1e-9
        if ph["kind"] == "wait":
            return "elapsed" if elapsed >= ph["s"] else None
        target = ph["to_frac"] * self.start_lateral_m
        if (target - lateral_m) * self._direction <= 0.0:
            return "reached"
        return "max_s" if ph.get("max_s") is not None and elapsed >= ph["max_s"] else None

    def step(self, t_s: float, lateral_m: float) -> Tuple[int, float]:
        """(direction, speed_mps) to command now: direction +1 = toward +lateral
        (right), -1 = left; speed 0 = stand still."""
        if self.index < 0:
            self._enter(0, t_s, lateral_m, "onset")
        while not self.done:
            reason = self._finished(t_s, lateral_m)
            if reason is None:
                break
            self._enter(self.index + 1, t_s, lateral_m, reason)
        if self.done or self.phases[self.index]["kind"] == "wait":
            return 0, 0.0
        return self._direction, float(self.phases[self.index]["speed_mps"])
