"""
rl_eval_protocol.py

Pure helpers for scoring learned policies with the common encounter protocol
(Phase 1, 2026-10-04). Used by eval_policy_encounters.py.

Why not eval_sac.py / eval_sac_on_sweep.py: they infer "full_stop" from any
non-collision termination (so a far-cross pedestrian clearing the road counts
as a stop), measure center-to-center distance, use per-controller time
windows, rely on CARLA's collision sensor (which missed every Week 4 contact),
save no traces or provenance, and overwrite their CSVs. Their outcome labels
are not trustworthy.

The protocol here is the one test26 uses for the scripted baselines:
  - onset = the scenario's scripted pedestrian trigger (controller-independent);
  - score the recorded TraceTick trace with encounter_metrics over
    [onset, onset + horizon_s] (8 s), ignoring the env's own early
    termination, so every controller gets the same observation window;
  - write timestamped, never-overwritten evidence with full provenance.

The legacy label is still computed (legacy_outcome_label) so a re-audit can
show where last semester's labels and the protocol disagree. It is never the
reported outcome.

Pure: no CARLA, SB3, or torch imports. Offline-tested in
tests/test_rl_eval_protocol.py.
"""

import csv
import hashlib
import os
from dataclasses import dataclass
from typing import List, Optional

from scenario_config import ScenarioConfig

EVAL_PROTOCOL_VERSION = 1
RUN_FAMILY = "rl_encounter_eval"

LEGACY_COLLISION = "collision"
LEGACY_FULL_STOP = "full_stop"
LEGACY_SLOWED_AVOIDED = "slowed_avoided"


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_model_path(model_arg: str, *, search_dir: Optional[str] = None) -> str:
    """Absolute path of an existing model ZIP. Accepts SB3's extensionless
    form ("sac_v3_1600k") and paths relative to search_dir. Raises
    FileNotFoundError rather than guessing a default model."""
    candidates = [model_arg] if model_arg.endswith(".zip") else [model_arg + ".zip", model_arg]
    bases = [os.getcwd()] + ([search_dir] if search_dir else [])
    for base in bases:
        for c in candidates:
            p = c if os.path.isabs(c) else os.path.join(base, c)
            if os.path.isfile(p):
                return os.path.abspath(p)
    raise FileNotFoundError(f"model not found: {model_arg!r} (tried {candidates} under {bases})")


@dataclass(frozen=True)
class WindowDecision:
    stop: bool
    reason: Optional[str]  # "window_complete" | "env_truncated" | None


def window_decision(
    *,
    sim_time_s: float,
    trigger_time_s: Optional[float],
    horizon_s: float,
    truncated: bool,
    fixed_dt_s: float,
    legacy_ended: bool = True,
) -> WindowDecision:
    """Whether the evaluation episode should stop after this tick.

    The env's own `terminated` is deliberately ignored: a collision, the
    first full stop, or a far-cross pedestrian clearing the road must not cut
    the observation window short (MASTER known problems #1 and #2). The
    episode runs until onset + horizon (plus one tick, so the window's end is
    sampled) AND until the legacy episode has ended (so the audit label is
    exact; ticks past the window never affect the metrics), or until the env
    truncates, in which case encounter_metrics reports incomplete_window.
    """
    window_done = trigger_time_s is not None and sim_time_s >= trigger_time_s + horizon_s + fixed_dt_s - 1e-9
    if window_done and legacy_ended:
        return WindowDecision(True, "window_complete")
    if truncated:
        return WindowDecision(True, "env_truncated")
    return WindowDecision(False, None)


def extended_sim_seconds(original_sim_seconds: float, horizon_s: float, legacy_timeout_s: float) -> float:
    """The env truncates at cfg.sim_seconds. Extend it so a late trigger still
    gets a full window and the legacy episode can reach its own end (the env's
    post-trigger timeout); this only moves the cap, never earlier behaviour."""
    return original_sim_seconds + max(horizon_s, legacy_timeout_s) + 1.0


def legacy_truncated(*, tick: int, original_sim_seconds: float, fixed_dt_s: float, post_trigger_timeout: bool) -> bool:
    """The env's truncation rule evaluated against the ORIGINAL time cap
    (tick >= int(sim_seconds / dt), or the post-trigger timeout)."""
    return tick >= int(original_sim_seconds / fixed_dt_s) or bool(post_trigger_timeout)


class LegacyLabelTracker:
    """Reproduces the outcome eval_sac_on_sweep.py would have recorded for
    this episode under the env's legacy semantics:
        legacy collision = CARLA sensor OR near-cross proximity fallback;
        legacy episode end = first tick with a legacy collision, a full stop,
                             a far-cross pedestrian clearing, or truncation;
        outcome = collision if any legacy collision up to that end,
                  else full_stop if the end was a termination,
                  else slowed_avoided.
    Fed from the env's per-tick info, so it is independent of the
    collision_signal mode the env actually ran with."""

    def __init__(self):
        self.ended = False
        self.end_tick: Optional[int] = None
        self.end_reason: Optional[str] = None
        self._collision = False
        self._terminated = False

    def update(self, *, tick: int, info: dict, truncated: bool) -> None:
        if self.ended:
            return
        legacy_hit = bool(info.get("collision_sensor")) or bool(info.get("collision_legacy_proximity"))
        self._collision = self._collision or legacy_hit
        for reason, flag in (("collision", legacy_hit), ("full_stop", info.get("full_stop_achieved")),
                             ("ped_crossed", info.get("ped_crossed"))):
            if flag:
                self.ended, self.end_tick, self.end_reason, self._terminated = True, tick, reason, True
                return
        if truncated:
            self.ended, self.end_tick, self.end_reason = True, tick, "truncated"

    @property
    def label(self) -> Optional[str]:
        if not self.ended:
            return None
        return legacy_outcome_label(collision_seen=self._collision, terminated=self._terminated)


def legacy_outcome_label(*, collision_seen: bool, terminated: bool) -> str:
    """eval_sac_on_sweep.py's original rule, verbatim."""
    if collision_seen:
        return LEGACY_COLLISION
    return LEGACY_FULL_STOP if terminated else LEGACY_SLOWED_AVOIDED


def trace_telemetry(telemetry: Optional[dict], info: dict) -> Optional[dict]:
    """Copy of lane_follow_step's telemetry plus the pedestrian's ground-truth
    position from the env info, in the keys TrajectoryRecorder reads. The
    position is never passed into lane_follow_step itself (that switches on
    control features)."""
    if telemetry is None:
        return None
    out = dict(telemetry)
    out["pedestrian_x_m"] = info.get("pedestrian_x_m")
    out["pedestrian_y_m"] = info.get("pedestrian_y_m")
    out.setdefault("speed_mps", info.get("ego_speed_mps", 0.0))
    return out


def load_sweep_configs(csv_path: str, canonical_profile: str = "proportional_ramp") -> List[ScenarioConfig]:
    """The matched-grid configs, parsed exactly as eval_sac_on_sweep.py does
    (one row per scenario: the canonical profile's rows)."""
    configs = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("brake_profile") != canonical_profile:
                continue
            cfg = ScenarioConfig(
                target_mph=float(r["target_mph"]),
                encounter_distance_m=float(r["encounter_distance_m"]),
                walker_speed_mps=float(r["walker_speed_mps"]),
                walker_side=r["walker_side"],
                walker_cross=r["walker_cross"],
                walker_startup_s=float(r.get("walker_startup_s", 0.5)),
                trigger_ttc_s=float(r["trigger_ttc_s"]) if r.get("trigger_ttc_s") else None,
                trigger_delay_s=float(r.get("trigger_delay_s", 0.0)),
                braking_ramp_up_per_s=float(r.get("braking_ramp_up_per_s", 4.0)),
                brake_headway_s=float(r.get("brake_headway_s", 2.5)),
                weather_preset=r.get("weather_preset", "ClearSunset"),
                sim_seconds=float(r.get("sim_seconds", 18)),
                brake_profile=canonical_profile,
            )
            cfg.run_id = r.get("run_id", "")
            configs.append(cfg)
    return configs


SUMMARY_COLUMNS = [
    "episode", "run_id", "target_mph", "trigger_ttc_s", "walker_cross", "walker_side", "walker_speed_mps",
    "outcome", "safe_success", "window_complete", "stop_reason", "speed_at_onset_mps", "min_clearance_m",
    "contact_speed_mps", "time_to_stop_s", "stopping_distance_m", "speed_when_passing_mps",
    "max_abs_route_lateral_m", "drivable_violation_ticks", "peak_decel_normal_speed_mps2",
    "max_abs_jerk_normal_speed_mps3", "onset_time_s", "legacy_label", "legacy_end_reason",
    "sensor_collision_ticks", "legacy_proximity_ticks", "geometric_contact_ticks", "physics_label",
]


def summary_row(*, episode: int, cfg: ScenarioConfig, metrics: dict, stop_reason: str, legacy_label,
                legacy_end_reason, signal_counts: dict, physics_label: Optional[str]) -> dict:
    row = {c: metrics.get(c) for c in SUMMARY_COLUMNS if c in metrics}
    row.update(
        episode=episode, run_id=cfg.run_id or "", target_mph=cfg.target_mph, trigger_ttc_s=cfg.trigger_ttc_s,
        walker_cross=cfg.walker_cross, walker_side=cfg.walker_side, walker_speed_mps=cfg.walker_speed_mps,
        stop_reason=stop_reason, legacy_label=legacy_label, legacy_end_reason=legacy_end_reason,
        sensor_collision_ticks=signal_counts.get("sensor", 0),
        legacy_proximity_ticks=signal_counts.get("legacy_proximity", 0),
        geometric_contact_ticks=signal_counts.get("geometric_contact", 0),
        physics_label=physics_label,
    )
    return {c: row.get(c) for c in SUMMARY_COLUMNS}


REQUIRED_MANIFEST_KEYS = (
    "created_at_iso", "git_commit", "git_dirty", "model_path", "model_sha256", "physics_backend",
    "collision_signal", "protocol", "suite", "seed", "deterministic_policy",
)


def check_manifest(manifest: dict) -> dict:
    missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in manifest]
    if missing:
        raise ValueError(f"eval manifest missing provenance keys: {missing}")
    return manifest
