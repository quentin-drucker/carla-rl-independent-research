"""
rl_collision_signal.py

Per-tick ego-pedestrian collision signal for CarlaAEBEnv (Phase 1, 2026-10-04).

Why this exists -- the environment's reward and termination counted a hit only
when CARLA's collision sensor fired, or when a proximity fallback fired, and
that fallback runs only for walker_cross == "near". The Week 4 baseline found
the sensor reported NO collision in any of 22 pedestrian contacts (test26), so
stationary and far-cross hits -- and any future hit beside the car while
steering -- went unpunished and never ended the episode.

Two signal modes (CarlaAEBEnv(collision_signal=...)):

    "legacy"     sensor OR near-cross proximity fallback. Exactly last
                 semester's semantics; v3 was trained and evaluated with it.
    "geometric"  sensor OR proximity fallback OR oriented-footprint contact
                 (the ego's rotated rectangle vs. a pedestrian circle, any
                 direction -- the same check encounter_metrics uses). A
                 superset of "legacy": it can only add collisions, never
                 remove one legacy counted.

All three component signals are computed and reported every tick in both
modes, so a run's info dict always shows which detector fired.

Pure: no CARLA imports. Offline-tested in tests/test_rl_collision_signal.py.
"""

from dataclasses import asdict, dataclass
from typing import Optional

from pedestrian_contact import (
    DEFAULT_PEDESTRIAN_RADIUS_M,
    compute_oriented_ego_pedestrian_clearance_m,
)

COLLISION_SIGNAL_LEGACY = "legacy"
COLLISION_SIGNAL_GEOMETRIC = "geometric"
COLLISION_SIGNAL_MODES = (COLLISION_SIGNAL_LEGACY, COLLISION_SIGNAL_GEOMETRIC)

# Legacy proximity fallback thresholds, unchanged from carla_aeb_env.py / test3.
LEGACY_CONTACT_DIST_M = 2.7     # frontal center-to-center contact range
LEGACY_MAX_LATERAL_M = 1.2      # vehicle half-width ~1.0 m + pedestrian radius ~0.2 m
LEGACY_MIN_SPEED_MPS = 0.5

# Oriented-footprint contact: clearance at or below this counts as contact.
# 0.0 matches encounter_metrics.EncounterProtocol.contact_tolerance_m.
GEOMETRIC_CONTACT_TOLERANCE_M = 0.0


def validate_collision_signal_mode(mode: str) -> str:
    if mode not in COLLISION_SIGNAL_MODES:
        raise ValueError(f"collision_signal must be one of {COLLISION_SIGNAL_MODES}, got {mode!r}")
    return mode


def legacy_proximity_contact(
    *,
    triggered: bool,
    walker_cross: str,
    to_ped_x: float,
    to_ped_y: float,
    ego_forward_x: float,
    ego_forward_y: float,
    ped_dist_m: float,
    ego_speed_mps: float,
) -> bool:
    """The environment's original near-cross proximity fallback, unchanged.

    Same arithmetic, in the same order, as the inline code it replaces, so
    legacy mode is bit-identical. ped_dist_m is whatever distance the caller
    used before (the env passes CARLA's 3D Location.distance).
    """
    if not triggered or walker_cross != "near":
        return False
    fwd_dot = to_ped_x * ego_forward_x + to_ped_y * ego_forward_y
    if fwd_dot <= 0:
        return False
    lateral_m = abs(ego_forward_x * to_ped_y - ego_forward_y * to_ped_x)
    return ped_dist_m < LEGACY_CONTACT_DIST_M and ego_speed_mps > LEGACY_MIN_SPEED_MPS and lateral_m < LEGACY_MAX_LATERAL_M


def geometric_clearance_m(
    *,
    ego_x_m: float,
    ego_y_m: float,
    ego_yaw_deg: float,
    pedestrian_x_m: float,
    pedestrian_y_m: float,
    pedestrian_radius_m: float = DEFAULT_PEDESTRIAN_RADIUS_M,
) -> float:
    """Signed oriented-footprint clearance (negative = overlap)."""
    return compute_oriented_ego_pedestrian_clearance_m(
        ego_x_m=ego_x_m, ego_y_m=ego_y_m, ego_yaw_deg=ego_yaw_deg,
        pedestrian_x_m=pedestrian_x_m, pedestrian_y_m=pedestrian_y_m,
        pedestrian_radius_m=pedestrian_radius_m,
    )


def is_geometric_contact(clearance_m: Optional[float]) -> bool:
    return clearance_m is not None and clearance_m <= GEOMETRIC_CONTACT_TOLERANCE_M


@dataclass(frozen=True)
class CollisionSignals:
    mode: str
    sensor: bool
    legacy_proximity: bool
    geometric_contact: bool
    clearance_m: Optional[float]  # None = no pedestrian this tick
    collision: bool               # the value that drives reward and termination

    def to_info(self) -> dict:
        return {f"collision_{k}" if k != "collision" else "collision": v for k, v in asdict(self).items()}


def combine_collision_signals(
    mode: str,
    *,
    sensor: bool,
    legacy_proximity: bool,
    clearance_m: Optional[float],
) -> CollisionSignals:
    validate_collision_signal_mode(mode)
    geometric = is_geometric_contact(clearance_m)
    collision = bool(sensor or legacy_proximity)
    if mode == COLLISION_SIGNAL_GEOMETRIC:
        collision = collision or geometric
    return CollisionSignals(
        mode=mode, sensor=bool(sensor), legacy_proximity=bool(legacy_proximity),
        geometric_contact=geometric, clearance_m=clearance_m, collision=collision,
    )
