"""
physics_backend.py

Week 4 Work Sequence 3 (plans/Week-4_2026-10-01_0930_chrono-feasibility-plan.md):
explicit physics-backend selection and provenance for the CARLA-Chrono
feasibility smoke test (test24___chrono_backend_smoke.py).

Everything here is pure / filesystem-only -- no CARLA imports -- so path
validation, the predeclared control schedule, backend labelling, and
output-directory safety are all offline-tested (tests/test_physics_backend.py)
before any live run.

Why validate so carefully: CARLA's 0.9.16 Chrono guide warns that bad
template paths can crash Unreal, and Sedan_Vehicle.json itself references
nested sub-templates (chassis, suspension, wheel, brake, steering,
driveline) plus visualization meshes. Every reference is checked before
enable_chrono_physics() is ever called.

Why the backend label is resolved rather than just copied from the CLI:
CARLA 0.9.16 exposes no server-side getter that reports which physics
backend a vehicle is using, and the docs state that a collision silently
reverts a Chrono vehicle to default physics. A requested backend is
therefore NOT evidence of the backend in effect; resolve_recorded_backend()
encodes the only evidence actually available, so a default-physics run can
never be written out labelled "chrono".
"""

import hashlib
import json
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

BACKEND_DEFAULT = "default"
BACKEND_CHRONO = "chrono"
VALID_BACKENDS = (BACKEND_DEFAULT, BACKEND_CHRONO)

# Recorded-backend labels (what ends up in PhysicsRunManifest.physics_backend).
RECORDED_DEFAULT = "default"
RECORDED_CHRONO = "chrono"
RECORDED_CHRONO_ENABLE_FAILED = "chrono_enable_failed"
RECORDED_CHRONO_INVALIDATED_BY_COLLISION = "chrono_invalidated_by_collision"

CHRONO_STATUS_NOT_REQUESTED = "not_requested"
CHRONO_STATUS_NOT_ATTEMPTED = "not_attempted"
CHRONO_STATUS_API_CALL_COMPLETED = "api_call_completed"
CHRONO_STATUS_FAILED = "failed"

DEFAULT_CARLA_ROOT = r"C:\Users\qdruc\OneDrive\Desktop\CARLA_0.9.16"
CHRONO_BASE_RELATIVE_TO_ROOT = os.path.join("Co-Simulation", "Chrono", "Vehicles")

# The supplied sedan templates, used unmodified this week (plan: "Editing
# Chrono vehicle/tire/powertrain templates" is explicitly deferred). Paths
# are relative to the Chrono base directory, exactly as CARLA's own
# PythonAPI/examples/manual_control_chrono.py passes them.
SEDAN_VEHICLE_JSON = "sedan/vehicle/Sedan_Vehicle.json"
SEDAN_POWERTRAIN_JSON = "sedan/powertrain/Sedan_SimpleMapPowertrain.json"
SEDAN_TIRE_JSON = "sedan/tire/Sedan_TMeasyTire.json"

# Same substep settings the official manual_control_chrono.py example uses
# (5000, 0.002) -- the configuration already manually verified on this
# machine. At the project's 0.02 s fixed step that is 10 Chrono substeps
# per CARLA tick.
CHRONO_MAX_SUBSTEPS = 5000
CHRONO_MAX_SUBSTEP_DELTA_TIME_S = 0.002

# File extensions treated as template file references when walking a
# template's JSON values.
_REFERENCE_EXTENSIONS = (".json", ".obj")


class BackendConfigError(ValueError):
    """Raised for any invalid backend selection or Chrono configuration.
    The message lists every problem found, not just the first."""


def validate_backend_name(name: str) -> str:
    if name not in VALID_BACKENDS:
        raise BackendConfigError(
            f"unknown physics backend {name!r}; expected one of {VALID_BACKENDS}"
        )
    return name


@dataclass(frozen=True)
class ChronoConfig:
    carla_root: str
    base_dir: str
    vehicle_json: str = SEDAN_VEHICLE_JSON
    powertrain_json: str = SEDAN_POWERTRAIN_JSON
    tire_json: str = SEDAN_TIRE_JSON
    max_substeps: int = CHRONO_MAX_SUBSTEPS
    max_substep_delta_time_s: float = CHRONO_MAX_SUBSTEP_DELTA_TIME_S

    @property
    def api_base_path(self) -> str:
        """Base path in the form enable_chrono_physics() expects: absolute,
        with a trailing separator (the official example appends os.sep, and
        CARLA concatenates it directly with the relative template paths)."""
        return os.path.join(os.path.abspath(self.base_dir), "")

    def top_level_templates(self) -> List[Tuple[str, str]]:
        return [
            ("vehicle_json", self.vehicle_json),
            ("powertrain_json", self.powertrain_json),
            ("tire_json", self.tire_json),
        ]


def build_chrono_config(*, carla_root: str, base_dir: Optional[str] = None) -> ChronoConfig:
    """base_dir defaults to <carla_root>/Co-Simulation/Chrono/Vehicles."""
    if base_dir is None:
        base_dir = os.path.join(carla_root, CHRONO_BASE_RELATIVE_TO_ROOT)
    return ChronoConfig(carla_root=carla_root, base_dir=base_dir)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_json_comments(text: str) -> str:
    """Removes // line and /* */ block comments outside string literals.

    Chrono parses templates with RapidJSON, which accepts comments; the
    supplied Sedan_TMeasyTire.json uses // comments (found 2026-10-01), so
    Python's strict json module rejects a template Chrono loads fine. Used
    for read-only validation only -- the template files are never rewritten.
    """
    out = []
    i, n = 0, len(text)
    in_string = False
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
        elif ch == '"':
            in_string = True
            out.append(ch)
            i += 1
        elif text.startswith("//", i):
            newline = text.find("\n", i)
            i = n if newline == -1 else newline
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def collect_template_references(obj) -> List[str]:
    """Every string value anywhere in a parsed template JSON that looks like
    a template file reference (ends in .json/.obj), in first-seen order,
    de-duplicated."""
    found: List[str] = []

    def _walk(node):
        if isinstance(node, dict):
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for value in node:
                _walk(value)
        elif isinstance(node, str) and node.lower().endswith(_REFERENCE_EXTENSIONS):
            if node not in found:
                found.append(node)

    _walk(obj)
    return found


def validate_chrono_config(cfg: ChronoConfig) -> dict:
    """Read-only validation of a Chrono configuration before any live call.

    Checks: CARLA root and base directory exist; the three top-level
    templates exist and parse as JSON; every file they reference (and,
    recursively, every file referenced by a referenced JSON) exists under
    the base directory. Collects ALL problems and raises one
    BackendConfigError listing them, so a fix-and-retry loop never has to
    rediscover them one at a time.

    Returns a JSON-serializable provenance report (absolute paths, sizes,
    SHA-256 hashes) for the run manifest -- the hashes let a later reader
    confirm the templates were the unmodified supplied ones.
    """
    problems: List[str] = []
    if not os.path.isdir(cfg.carla_root):
        problems.append(f"CARLA root directory not found: {cfg.carla_root}")
    if not os.path.isdir(cfg.base_dir):
        problems.append(f"Chrono base directory not found: {cfg.base_dir}")
        raise BackendConfigError("; ".join(problems))

    top_level = {}
    nested = {}
    pending: List[str] = []
    for role, rel in cfg.top_level_templates():
        path = os.path.join(cfg.base_dir, rel)
        if not os.path.isfile(path):
            problems.append(f"{role} template not found: {path}")
            continue
        top_level[role] = {
            "relative_path": rel,
            "absolute_path": os.path.abspath(path),
            "bytes": os.path.getsize(path),
            "sha256": _sha256(path),
        }
        pending.append(rel)

    visited = set()
    while pending:
        rel = pending.pop(0)
        if rel in visited:
            continue
        visited.add(rel)
        path = os.path.join(cfg.base_dir, rel)
        if not rel.lower().endswith(".json"):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                parsed = json.loads(strip_json_comments(f.read()))
        except (OSError, ValueError) as exc:
            problems.append(f"template is not readable JSON: {path} ({exc})")
            continue
        for ref in collect_template_references(parsed):
            ref_path = os.path.join(cfg.base_dir, ref)
            if not os.path.isfile(ref_path):
                problems.append(f"{rel} references a missing file: {ref_path}")
                continue
            if ref not in nested and ref not in {v["relative_path"] for v in top_level.values()}:
                nested[ref] = {"bytes": os.path.getsize(ref_path), "sha256": _sha256(ref_path)}
            pending.append(ref)

    if problems:
        raise BackendConfigError("; ".join(problems))

    return {
        "carla_root": os.path.abspath(cfg.carla_root),
        "base_dir": os.path.abspath(cfg.base_dir),
        "api_base_path": cfg.api_base_path,
        "max_substeps": cfg.max_substeps,
        "max_substep_delta_time_s": cfg.max_substep_delta_time_s,
        "templates": top_level,
        "referenced_files": dict(sorted(nested.items())),
    }


# ----------------------------------------------------------------------
# Server launch-flag check
# ----------------------------------------------------------------------

def cmdline_has_chrono_flag(cmdline: str) -> bool:
    """True if a process command line contains the exact token --chrono."""
    try:
        tokens = shlex.split(cmdline, posix=False)
    except ValueError:
        tokens = cmdline.split()
    return any(token.strip('"').lower() == "--chrono" for token in tokens)


def summarize_server_processes(processes: Optional[List[dict]]) -> dict:
    """processes: list of {"pid", "name", "cmdline"} dicts for running CARLA
    server processes, or None if the process table could not be queried.

    The --chrono launch flag is necessary for enable_chrono_physics() to do
    anything; the API call itself returns nothing to confirm it worked, so
    this is checked independently before a Chrono run.
    """
    if processes is None:
        return {"queried": False, "processes": [], "any_has_chrono_flag": None}
    return {
        "queried": True,
        "processes": processes,
        "any_has_chrono_flag": any(cmdline_has_chrono_flag(p.get("cmdline") or "") for p in processes),
    }


def query_carla_server_processes() -> Optional[List[dict]]:
    """Live, Windows-only: list running CarlaUE4 processes with their
    command lines via PowerShell/CIM. Returns None if the query fails (the
    caller must then refuse a Chrono run rather than assume the flag)."""
    if os.name != "nt":
        return None
    command = (
        "Get-CimInstance Win32_Process -Filter \"Name like 'CarlaUE4%'\" | "
        "Select-Object ProcessId, Name, CommandLine | ConvertTo-Json -Compress"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return []
    try:
        parsed = json.loads(out)
    except ValueError:
        return None
    if isinstance(parsed, dict):
        parsed = [parsed]
    return [
        {"pid": p.get("ProcessId"), "name": p.get("Name"), "cmdline": p.get("CommandLine")}
        for p in parsed
    ]


LOCAL_HOSTS = ("localhost", "127.0.0.1")


def run_preflight(
    *, backend: str, carla_root: str, chrono_base_dir: Optional[str], host: str,
    process_query=query_carla_server_processes,
):
    """Everything that must pass before a vehicle exists. Returns
    (chrono_cfg, chrono_report, server_summary); cfg/report are None for the
    default backend. Raises BackendConfigError on any problem.

    process_query is injectable so the refusal paths are offline-testable.
    """
    validate_backend_name(backend)
    server_summary = summarize_server_processes(process_query() if host in LOCAL_HOSTS else None)
    if backend == BACKEND_DEFAULT:
        return None, None, server_summary

    cfg = build_chrono_config(carla_root=carla_root, base_dir=chrono_base_dir)
    report = validate_chrono_config(cfg)
    if host not in LOCAL_HOSTS:
        raise BackendConfigError(f"cannot verify the --chrono server flag on non-local host {host!r}")
    if not server_summary["queried"]:
        raise BackendConfigError("could not query running CarlaUE4 processes to verify --chrono")
    if not server_summary["any_has_chrono_flag"]:
        raise BackendConfigError(
            "no running CarlaUE4 process was launched with --chrono; enable_chrono_physics() "
            "would not take effect. Restart CARLA with: CarlaUE4.exe --chrono"
        )
    return cfg, report, server_summary


# ----------------------------------------------------------------------
# Backend labelling
# ----------------------------------------------------------------------

def resolve_recorded_backend(
    *, requested_backend: str, chrono_enable_status: str, collision_detected: bool
) -> str:
    """The backend label written into the run manifest.

    - default requested -> "default" (Chrono is never enabled on this path)
    - chrono requested, enable call completed, no collision -> "chrono"
    - chrono requested, enable call completed, collision -> the docs say the
      vehicle reverts to default physics after a collision, so the run is
      "chrono_invalidated_by_collision", never plain "chrono"
    - chrono requested, enable call not completed -> "chrono_enable_failed"
    """
    validate_backend_name(requested_backend)
    if requested_backend == BACKEND_DEFAULT:
        if chrono_enable_status != CHRONO_STATUS_NOT_REQUESTED:
            raise BackendConfigError(
                f"default-backend run has inconsistent chrono status {chrono_enable_status!r}"
            )
        return RECORDED_DEFAULT
    if chrono_enable_status != CHRONO_STATUS_API_CALL_COMPLETED:
        return RECORDED_CHRONO_ENABLE_FAILED
    if collision_detected:
        return RECORDED_CHRONO_INVALIDATED_BY_COLLISION
    return RECORDED_CHRONO


# ----------------------------------------------------------------------
# Predeclared control schedule
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ControlPhase:
    name: str
    duration_s: float
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    # Brake phase only: finish early once the vehicle has stopped.
    end_when_stopped: bool = False


# Pre-sequence settling (identical tick counts under both backends so the
# measured sequence starts at the same sim time). Chrono is enabled -- or,
# for the default backend, deliberately not enabled -- between the two.
SETTLE_BEFORE_ENABLE = ControlPhase("settle_before_enable", 1.0, brake=1.0)
SETTLE_AFTER_ENABLE = ControlPhase("settle_after_enable", 1.0, brake=1.0)

# The measured sequence: acceleration, a modest steer doublet (right then
# left, so net heading change is ~0 and the car stays in its lane group --
# right first because the Town04_Opt spawn-242 location has more room to the
# right; see MASTER summary Week 3 occupancy finding), coast, and braking
# with the wheel centred so braking is not confounded by steering.
#
# Version 2 (2026-10-01). Version 1 coasted through the steer doublet
# (accelerate 4 s, coast 1 s, steer +/-0.05 for 1 s each while coasting,
# coast 0.5 s, brake). Its single default-backend run showed ~-3 m/s^2
# coasting drag, so braking began at only 2.4 m/s -- entirely inside the
# documented < 5 m/s low-speed artifact zone -- and barely exercised
# braking. v2 holds the throttle through the doublet instead. Changed
# before any Chrono run, so no Chrono result influenced it.
SMOKE_SCHEDULE_VERSION = 2
SMOKE_SCHEDULE: Tuple[ControlPhase, ...] = (
    ControlPhase("accelerate", 4.0, throttle=0.6),
    ControlPhase("accelerate_steer_right", 1.0, throttle=0.6, steer=0.05),
    ControlPhase("accelerate_steer_left", 1.0, throttle=0.6, steer=-0.05),
    ControlPhase("coast", 1.0),
    ControlPhase("brake", 5.0, brake=0.6, end_when_stopped=True),
)


def phase_tick_count(phase: ControlPhase, fixed_dt: float) -> int:
    return int(round(phase.duration_s / fixed_dt))


def schedule_to_dicts(schedule) -> List[dict]:
    return [asdict(p) for p in schedule]


# ----------------------------------------------------------------------
# Initial-state tolerance
# ----------------------------------------------------------------------

INITIAL_MAX_SPEED_MPS = 0.1
INITIAL_MAX_ABS_ROLL_DEG = 3.0
INITIAL_MAX_ABS_PITCH_DEG = 3.0


def check_initial_state(*, speed_mps: float, roll_deg: float, pitch_deg: float) -> List[str]:
    """Problems with the vehicle's state just before the measured sequence
    (empty list = within tolerance). A car still moving or tilted after
    settling -- e.g. bouncing after a physics-backend switch -- would make
    the two backends start from different states."""
    problems = []
    if speed_mps > INITIAL_MAX_SPEED_MPS:
        problems.append(f"speed {speed_mps:.3f} m/s > {INITIAL_MAX_SPEED_MPS}")
    if abs(roll_deg) > INITIAL_MAX_ABS_ROLL_DEG:
        problems.append(f"|roll| {abs(roll_deg):.2f} deg > {INITIAL_MAX_ABS_ROLL_DEG}")
    if abs(pitch_deg) > INITIAL_MAX_ABS_PITCH_DEG:
        problems.append(f"|pitch| {abs(pitch_deg):.2f} deg > {INITIAL_MAX_ABS_PITCH_DEG}")
    return problems


# ----------------------------------------------------------------------
# Output safety
# ----------------------------------------------------------------------

def create_run_dir(*, runs_root: str, timestamp: str, run_family: str, backend: str) -> str:
    """Creates and returns <runs_root>/<timestamp>_<run_family>_<backend>.
    Raises FileExistsError instead of reusing an existing directory, so a
    re-run can never mix with or overwrite earlier evidence."""
    validate_backend_name(backend)
    path = os.path.join(runs_root, f"{timestamp}_{run_family}_{backend}")
    os.makedirs(runs_root, exist_ok=True)
    os.makedirs(path, exist_ok=False)
    return path
