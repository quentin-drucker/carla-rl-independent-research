"""
trace_schema.py

A controller-independent, per-tick trace record shared by:
  - the Week 3 vehicle physical-limits test suite (steering lock, rollover,
    braking, throttle/brake response), and
  - the planned trajectory-visualization tooling (Workstream 3), so both
    consume the same tick format instead of two divergent schemas.

Design intent (see plans/Week-3_...-physical-limits-trajectory-plan.md):
  - Physical-limits tests use direct low-level VehicleControl and mostly
    leave `route_*`/`pedestrian_*`/`event_marker` at their default None --
    there is no route or pedestrian in those tests.
  - Scenario/trajectory runs (test3-family) can populate the same TraceTick
    with route-relative fields and event markers without changing the core
    pose/speed/control columns the physical-limits tools already rely on.

This module holds only pure data structures and serialization -- no CARLA
imports, no live-actor access, so it is fully offline-testable.
"""

import csv
import json
import os
from dataclasses import dataclass, asdict, fields
from typing import Optional, List


@dataclass
class TraceTick:
    """One simulation tick of vehicle state, shared across test families."""

    # ------------------------------------------------------------------
    # Identity / time
    # ------------------------------------------------------------------
    tick_index: int
    sim_time_s: float

    # ------------------------------------------------------------------
    # World-frame pose
    # ------------------------------------------------------------------
    pos_x_m: float
    pos_y_m: float
    pos_z_m: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------
    speed_mps: float
    # World-frame velocity components (m/s). Used to derive body slip angle
    # (velocity heading vs. yaw) -- kept as raw components, not just the
    # derived angle, so a future consumer can recompute differently.
    vel_x_mps: Optional[float] = None
    vel_y_mps: Optional[float] = None
    # Longitudinal acceleration, finite-difference of speed_mps over dt.
    # None on the first tick of a run (no previous sample to differentiate).
    accel_mps2: Optional[float] = None
    # Signed yaw rate, finite-difference of yaw_deg over dt (deg/s).
    # None on the first tick of a run.
    yaw_rate_dps: Optional[float] = None
    # Whole-body sideslip angle (velocity heading minus yaw), degrees.
    # None below a minimum speed where velocity heading is unstable. See
    # physics_harness.compute_body_slip_angle_deg -- NOT a wheel-slip signal.
    body_slip_angle_deg: Optional[float] = None

    # ------------------------------------------------------------------
    # Requested vs. applied control
    # ------------------------------------------------------------------
    # "Requested" is what the harness/controller commanded this tick.
    # "Applied" is read back from the actor's own control state after the
    # tick advances -- CARLA can clamp/reject a requested value (e.g. a
    # reverse-gear conflict), so the two are not guaranteed equal and both
    # must be logged rather than inferring one from the other.
    requested_throttle: float = 0.0
    requested_brake: float = 0.0
    requested_steer: float = 0.0
    applied_throttle: Optional[float] = None
    applied_brake: Optional[float] = None
    applied_steer: Optional[float] = None
    # Signed route-relative lateral offset commanded by the controller this
    # tick (route-right-positive), distinct from applied_steer -- the plan's
    # Workstream 3.1 lists these as two separate quantities: a controller's
    # high-level lateral TARGET vs. the low-level steer angle used to chase
    # it. None outside a scenario run with an active lateral_offset_fn.
    requested_lateral_offset_m: Optional[float] = None

    # ------------------------------------------------------------------
    # Lateral quantities
    # ------------------------------------------------------------------
    # Perpendicular signed distance from a caller-defined reference line
    # (e.g. the initial heading line for a straight physical-limits test,
    # or the planned route for a scenario run). None if not computed.
    lateral_displacement_m: Optional[float] = None
    lateral_accel_mps2: Optional[float] = None

    # ------------------------------------------------------------------
    # Wheel / contact signals CARLA exposes reliably.
    # ------------------------------------------------------------------
    # Front-left/front-right steer angles from vehicle.get_wheel_steer_angle().
    # Both are logged, not just FL -- Ackermann geometry means the inner and
    # outer front wheels turn different amounts, so FL alone is direction-
    # biased (in a right turn FL is the outer/smaller-angle wheel; in a left
    # turn it is the inner/larger-angle wheel). A single "achieved wheel
    # angle" summary should use the inner wheel (max magnitude of the two),
    # not FL unconditionally -- see physics_harness.get_front_wheel_steer_
    # angles_deg. CARLA 0.9.16's Python API does not expose a validated
    # per-wheel slip-ratio or lock-detection signal, so no slip/lock field
    # exists here -- do not add one without first confirming a specific
    # reliable API source.
    front_wheel_steer_angle_deg: Optional[float] = None
    front_right_wheel_steer_angle_deg: Optional[float] = None

    # ------------------------------------------------------------------
    # Scenario-context fields (None outside a pedestrian scenario run).
    # ------------------------------------------------------------------
    route_longitudinal_m: Optional[float] = None
    route_lateral_m: Optional[float] = None
    pedestrian_x_m: Optional[float] = None
    pedestrian_y_m: Optional[float] = None
    event_marker: Optional[str] = None


@dataclass
class PhysicsRunManifest:
    """Compact provenance record for one physical-limits test run.

    Written once per run alongside its trace file. Captures enough of the
    simulator/vehicle/physics configuration that a result can be
    interpreted or reproduced without re-reading the harness source.
    """

    run_id: str
    test_family: str
    # e.g. "steering_lock", "rollover", "braking", "throttle_brake_symmetry"
    created_at_iso: str

    carla_version: str
    map_name: str
    vehicle_blueprint: str

    fixed_delta_seconds: float
    substepping_enabled: bool
    max_substep_delta_time: Optional[float]
    max_substeps: Optional[int]

    weather_preset: str
    tire_friction: Optional[float]
    # None = CARLA default; otherwise the uniform friction applied to all
    # four wheels for this run (see apply_uniform_tire_friction()).

    random_seed: Optional[int]
    parameters: dict
    # Test-family-specific inputs (entry speed, steer command, etc.) --
    # kept as a free-form dict so this manifest doesn't need a new dataclass
    # field for every test family.

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "PhysicsRunManifest":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, path: str) -> "PhysicsRunManifest":
        with open(path) as f:
            return cls.from_dict(json.load(f))


_TRACE_FIELD_NAMES = [f.name for f in fields(TraceTick)]


def write_trace_csv(path: str, ticks: List[TraceTick]) -> None:
    """Write a list of TraceTick rows to CSV, one row per tick.

    Raises ValueError rather than silently overwriting an existing trace --
    physical-limits evidence must not be clobbered by a re-run using the
    same path (see plan: "use timestamped outputs and refuse accidental
    overwrite of retained evidence").
    """
    if os.path.exists(path):
        raise ValueError(
            f"refusing to overwrite existing trace file: {path} "
            "(use a new timestamped run directory instead)"
        )
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TRACE_FIELD_NAMES)
        writer.writeheader()
        for tick in ticks:
            writer.writerow(asdict(tick))


def read_trace_csv(path: str) -> List[TraceTick]:
    """Read a trace CSV back into TraceTick rows.

    Empty-string CSV cells (from None fields) are converted back to None;
    every other field is cast to the type of TraceTick's declared default
    where that default indicates int/float, otherwise left as parsed float
    for numeric-looking optional fields.
    """
    field_types = {f.name: f.type for f in fields(TraceTick)}
    int_fields = {"tick_index"}
    str_fields = {"event_marker"}

    ticks = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for name in _TRACE_FIELD_NAMES:
                raw = row.get(name, "")
                if raw == "" or raw is None:
                    parsed[name] = None
                elif name in int_fields:
                    parsed[name] = int(raw)
                elif name in str_fields:
                    parsed[name] = raw
                else:
                    parsed[name] = float(raw)
            ticks.append(TraceTick(**parsed))
    return ticks
