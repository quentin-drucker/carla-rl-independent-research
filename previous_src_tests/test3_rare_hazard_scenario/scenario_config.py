"""
scenario_config.py

Defines ScenarioConfig -- a single dataclass capturing every parameter
that can vary between scenario runs.

Usage (single run, default values):
    cfg = ScenarioConfig()

Usage (custom single run):
    cfg = ScenarioConfig(target_mph=45, walker_side="right", weather_preset="HardRainNoon")

Usage (re-run from a saved config file):
    cfg = ScenarioConfig.from_json("runs/20260314_120000/run_0001/config.json")

Usage (save config to file):
    cfg.to_json("runs/20260314_120000/run_0001/config.json")
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional


# All preset names accepted by CARLA's WeatherParameters class.
VALID_WEATHER_PRESETS = [
    "Default",
    "ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon",
    "SoftRainNoon", "MidRainyNoon", "HardRainNoon",
    "ClearSunset", "CloudySunset", "WetSunset", "WetCloudySunset",
    "SoftRainSunset", "MidRainSunset", "HardRainSunset",
]


@dataclass
class ScenarioConfig:

    # ------------------------------------------------------------------
    # Pedestrian parameters
    # ------------------------------------------------------------------
    walker_speed_mps: float = 1.8
    # Walking speed of the scripted pedestrian (m/s).
    # Typical range: 1.0 (slow) to 2.5 (jogging).

    walker_side: str = "left"
    # Which side of the ego lane the pedestrian starts from.
    # "left" or "right" (relative to ego's forward direction).

    walker_cross: str = "far"
    # How far across the lane the pedestrian walks.
    # "near" = stops roughly at lane center (partial intrusion).
    # "far"  = crosses all the way to the opposite lane edge.

    walker_startup_s: float = 0.5
    # Seconds the pedestrian waits (standing still) after the trigger
    # fires before actually starting to walk. Models hesitation.
    # Only used when trigger_ttc_s is None (see below).

    trigger_ttc_s: Optional[float] = None
    # "Time-to-crossing" at the moment the trigger fires (seconds).
    # Intuitive meaning: if the ego maintained cruise speed without braking,
    # it would reach the crossing point exactly this many seconds after the
    # trigger fires (and the pedestrian starts walking).
    #
    # When set, this OVERRIDES the walker_startup_s lead-time formula and
    # directly controls how tight the encounter is:
    #   higher value = more lead time = easier for ego to stop
    #   lower value  = less lead time = tighter / more dangerous
    #
    # Typical useful range (at 35 mph): 2.0s (very tight) to 4.0s (generous).
    # With defaults (speed=1.8 m/s, far cross): lead formula ≈ 3.3s equivalent.
    #
    # None = use the original formula: walker_startup_s + (cross_dist / walker_speed_mps)

    trigger_delay_s: float = 0.0
    # Extra time-based guard: trigger won't fire until sim_time >= this,
    # even if the distance condition is already met. Set to 0 to disable.

    walker_post_trigger_delay_s: float = 0.0
    # Additional pause between the trigger event and the first walker
    # motion step. Separate from walker_startup_s. Usually 0.

    # ------------------------------------------------------------------
    # Ego vehicle parameters
    # ------------------------------------------------------------------
    target_mph: float = 35.0
    # Cruise control target speed for the ego vehicle (mph).

    encounter_distance_m: float = 120.0
    # Distance along the planned route (from ego start) where the
    # pedestrian crossing is placed (meters).
    #
    # NOTE: if this is short, the ego may not have reached target_mph
    # yet when the encounter happens. That is intentional and informative
    # -- the actual ego speed at trigger is logged in RunResult. Short
    # distances test "ego still accelerating" scenarios; long distances
    # test "ego at cruising speed" scenarios.

    braking_ramp_up_per_s: float = 4.0
    # How quickly the hazard brake ramps up to full force (brake units per second).
    # Higher = more aggressive / abrupt stop (better safety margin, worse comfort).
    # Lower  = gentler ramp (smoother ride, longer stopping distance).
    # 4.0 reaches full brake in ~0.25s (comfortable AEB onset).
    # Suggested range: 2.0 (gentle) to 20.0 (slam).

    brake_headway_s: float = 2.5
    # (Can think of this as road "awareness") 
    # How far ahead (in time-gap seconds) the LiDAR braking trigger activates.
    # trigger_distance_m = BRAKE_BASE_DISTANCE_M + speed_mps * brake_headway_s
    # Higher = braking starts earlier / more conservative (better safety margin).
    # Lower  = braking starts later / more reactive (shorter trip time, tighter margin).
    # Suggested range: 1.0 (reactive) to 4.0 (very conservative).

    brake_profile: str = "proportional_ramp"
    # Which braking profile to use when a hazard is detected.
    # Controls how brake_target is computed from the obstacle distance.
    #
    # "proportional_ramp" (default / baseline):
    #     Linearly scales brake force from 0 at the trigger edge to 1 at
    #     panic distance. Smooth onset, progressive response.
    #
    # "step_constant":
    #     Applies full brake (1.0) immediately when any hazard is detected,
    #     regardless of distance. Fastest reaction; highest jerk. Models a
    #     worst-case "slam the brakes" strategy.
    #
    # "cautious_ramp":
    #     Same linear scaling as proportional_ramp but capped at 0.5 max
    #     brake force. Prioritizes comfort; may not stop in time at high speed
    #     or close encounters.
    #
    # "exponential":
    #     brake_target = penetration². Stays gentle near the trigger edge,
    #     then ramps steeply as the obstacle closes in. Compromise between
    #     comfort and urgency.
    #
    # All profiles still go through the same ramp-rate limiter
    # (braking_ramp_up_per_s), so they are fairly comparable.

    road_friction: Optional[float] = None
    # Tire friction coefficient applied to all 4 wheels via CARLA physics.
    # None = use CARLA's default (dry asphalt, typically ~3.5 in Bullet units).
    # Lower values simulate degraded grip:
    #   ~2.0  wet road    ~1.0  very wet / light ice    ~0.5  icy road
    # Note: CARLA does not model ABS natively. Use use_abs=True for software ABS.

    use_abs: bool = False
    # Software ABS (Anti-lock Braking System) toggle.
    # When True, the braking controller monitors actual deceleration each tick
    # and pulses the brake down briefly if wheel lock is detected (i.e. brake
    # is high but deceleration is below the expected threshold). This prevents
    # the locked-wheel / kinetic-friction penalty that reduces stopping force
    # when full brake is applied on CARLA's PhysX tires.
    #
    # Effect: most significant for step_constant (which applies brake=1.0 instantly).
    # With ABS off, step_constant locks wheels and stopping distance increases.
    # With ABS on, brake pulses maintain rolling friction -> shorter stopping distance.
    #
    # False = no ABS (CARLA default behavior, wheels can lock)
    # True  = software ABS enabled

    # ------------------------------------------------------------------
    # Environment parameters
    # ------------------------------------------------------------------
    weather_preset: str = "ClearSunset"
    # CARLA WeatherParameters preset name. Must be one of VALID_WEATHER_PRESETS.
    # Examples: "ClearNoon", "HardRainNoon", "WetSunset", "CloudyNoon"

    sun_altitude_deg: Optional[float] = None
    # Override the preset's sun altitude angle (degrees).
    # Negative = below horizon (night). None = use the preset's default.
    # Example: -10.0 for near-dark, 45.0 for afternoon sun.

    cloudiness: Optional[float] = None
    # Override the preset's cloudiness (0 = clear, 100 = overcast).
    # None = use the preset's default.

    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------
    sim_seconds: float = 18
    # Maximum run duration in seconds. Run ends when this elapses
    # (regardless of whether the scenario event has completed).

    # ------------------------------------------------------------------
    # Sweep provenance (set automatically by sweep.py -- leave as None
    # for hand-run single scenarios)
    # ------------------------------------------------------------------
    random_seed: Optional[int] = None
    # Seed used if this config was generated by a random sweep.
    # None = manually specified config (not randomly generated).

    run_id: Optional[str] = None
    # Identifier assigned by sweep.py, e.g. "run_0042".
    # None for single interactive runs.

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        """Write config to a JSON file. Creates parent directories if needed."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[config] saved -> {path}")

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioConfig":
        """Construct from a dict, ignoring any unknown keys (forward-compatible)."""
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, path: str) -> "ScenarioConfig":
        """Load config from a JSON file saved by to_json()."""
        with open(path) as f:
            return cls.from_dict(json.load(f))
