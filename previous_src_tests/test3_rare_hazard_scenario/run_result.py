"""
run_result.py

Defines RunResult -- outcome metrics recorded at the end of a single
scenario run. Saved as result.json alongside config.json.

Usage:
    result = run_scenario(cfg)          # returned by test3's run_scenario()
    result.to_json("runs/.../result.json")
    result = RunResult.from_json("runs/.../result.json")
    print_run_result(result)
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional

from scenario_config import ScenarioConfig
from math_utils import mps_to_mph


@dataclass
class RunResult:

    # ------------------------------------------------------------------
    # Scenario outcome flags
    # ------------------------------------------------------------------
    hazard_triggered: bool
    # True if the ego's HAZARD_BRAKE mode engaged at any point.
    # If False, the ego never detected the pedestrian as a hazard.

    collision_detected: bool
    # True if CARLA's collision sensor fired (ego physically hit something).

    outcome: str
    # Three-way outcome classification:
    #   "full_stop"     - ego reached near-zero speed (< 0.3 m/s) while ped
    #                     was still mid-crossing. Ideal braking response.
    #   "slowed_avoided"- ego did NOT fully stop, but no collision occurred.
    #                     Partial braking was sufficient to avoid the ped.
    #   "collision"     - ego physically hit the pedestrian.

    min_ped_distance_m: float
    # Closest 3D distance between ego center and pedestrian center
    # during the entire run (meters). Lower = closer call.

    min_ttc_s: float
    # Minimum Time-to-Collision observed after the pedestrian trigger fires.
    # Computed as: ped_distance / ego_speed each tick (simplified; assumes ped
    # is the primary closing threat). Lower = more dangerous encounter.
    # -1.0 if the trigger never fired or ego was stopped the whole time.

    max_jerk_mps3: float
    # Maximum instantaneous jerk magnitude (|d²v/dt²|) over the run (m/s³).
    # Jerk is the rate of change of acceleration -- the metric most directly
    # felt by passengers as a lurch or snap. Higher = more uncomfortable.

    mean_jerk_mps3: float
    # Mean jerk magnitude over the run (m/s³). Complements max_jerk_mps3 --
    # max captures the worst moment, mean captures overall ride roughness.

    # ------------------------------------------------------------------
    # Ego state at the moment the pedestrian trigger fired
    # ------------------------------------------------------------------
    ego_speed_at_trigger_mps: float
    # Actual ego speed (m/s) when the pedestrian crossing was triggered.
    # This is the speed the ego was actually traveling at -- NOT target_mph.
    # Use this to understand what conditions the ego was in at the moment
    # of the encounter (e.g., still accelerating vs. at cruise speed).

    ego_speed_at_trigger_mph: float
    # Same as above in mph (convenience field).

    ego_dist_at_trigger_m: float
    # Total distance the ego had traveled (meters) when trigger fired.

    trigger_time_s: Optional[float]
    # Simulation time (seconds) when the pedestrian trigger fired.
    # None if the trigger never fired (e.g., run too short).

    time_to_stop_s: Optional[float]
    # Seconds elapsed from the first HAZARD_BRAKE engagement to when
    # ego speed dropped below 0.3 m/s.
    # None if the ego never fully stopped after the hazard.

    # ------------------------------------------------------------------
    # Config that produced this result (for self-contained result files)
    # ------------------------------------------------------------------
    config: ScenarioConfig

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        # asdict() recurses into ScenarioConfig automatically.
        return asdict(self)

    def to_json(self, path: str) -> None:
        """Write result to a JSON file. Creates parent directories if needed."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[result] saved -> {path}")

    @classmethod
    def from_dict(cls, d: dict) -> "RunResult":
        d = dict(d)
        d["config"] = ScenarioConfig.from_dict(d["config"])
        return cls(**d)

    @classmethod
    def from_json(cls, path: str) -> "RunResult":
        """Load result from a JSON file saved by to_json()."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


def print_run_result(result: RunResult) -> None:
    """Print a human-readable summary of a RunResult."""
    cfg = result.config
    print("\n==================== SCENARIO RESULT ====================")
    if cfg.run_id:
        print(f"Run ID : {cfg.run_id}")
    print(f"Target speed     : {cfg.target_mph:.1f} mph")
    print(f"Encounter dist   : {cfg.encounter_distance_m:.1f} m")
    print(f"Walker           : {cfg.walker_side}/{cfg.walker_cross} @ {cfg.walker_speed_mps:.1f} m/s")
    print(f"Weather          : {cfg.weather_preset}")
    print("---------------------------------------------------------")
    print(f"Hazard triggered : {result.hazard_triggered}")
    print(f"Collision        : {result.collision_detected}")
    print(f"Outcome          : {result.outcome}")
    print(f"Min ped distance : {result.min_ped_distance_m:.2f} m")
    ttc_str = f"{result.min_ttc_s:.2f} s" if result.min_ttc_s >= 0 else "n/a"
    print(f"Min TTC          : {ttc_str}  (risk: lower = more dangerous)")
    print(f"Max jerk         : {result.max_jerk_mps3:.2f} m/s³  (comfort: lower = smoother)")
    print(f"Mean jerk        : {result.mean_jerk_mps3:.2f} m/s³")
    print("---------------------------------------------------------")
    if result.trigger_time_s is not None:
        print(f"Trigger fired at : t={result.trigger_time_s:.2f}s")
        print(f"Ego speed at trigger: {result.ego_speed_at_trigger_mph:.1f} mph"
              f"  (target: {cfg.target_mph:.1f} mph)")
        print(f"Ego dist at trigger : {result.ego_dist_at_trigger_m:.1f} m")
    else:
        print("Trigger fired at : (never triggered)")
    if result.time_to_stop_s is not None:
        print(f"Time to stop     : {result.time_to_stop_s:.2f}s after hazard brake")
    else:
        print("Time to stop     : (did not fully stop)")
    print("=========================================================\n")
