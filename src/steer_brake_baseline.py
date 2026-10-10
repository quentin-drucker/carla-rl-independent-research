"""
steer_brake_baseline.py

Pure (no CARLA) pieces of the deterministic steering-plus-braking baseline
(test26___steer_brake_baseline.py), offline-tested in
tests/test_steer_brake_baseline.py.

Design ("oracle onset", chosen 2026-10-02): every controller starts reacting
at the SAME predeclared moment -- the scenario trigger, which fires when the
ego is `onset_ttc_s` seconds (at its current speed) from the pedestrian --
using ground truth instead of LiDAR detection. This isolates the physical
question that must be answered before steering enters the RL action space:
is a braking-plus-steering maneuver achievable, and in which conditions does
it succeed where braking alone fails? Perception timing is a separate,
later question (LiDAR-in-the-loop follow-up).

Four matched controller modes:
  no_intervention -- hazard braking disabled for the whole run; the ego keeps
                     cruising (establishes that the encounter is dangerous).
  brake_only      -- from onset: scripted brake target, wheel on the route.
  brake_steer     -- from onset: the same scripted brake target PLUS the
                     existing scripted swerve (test5's
                     HazardClearRecoveryController).
  brake_passage_edge -- (Phase 2, 2026-10-05) from onset: the same brake PLUS
                     Prof. Izmirli's rule: aim the steering at the extreme
                     right edge of the "passage" (passage.py), i.e. passage
                     coordinate u = +1. A scripted reference for the learned
                     steering policy, which outputs u itself.
LiDAR hazard detection is disabled in all three (the scripted decision is
False before onset), so pre-onset driving is identical across modes.
"""

import math
from typing import List, Tuple

MODE_NO_INTERVENTION = "no_intervention"
MODE_BRAKE_ONLY = "brake_only"
MODE_BRAKE_STEER = "brake_steer"
MODE_BRAKE_PASSAGE_EDGE = "brake_passage_edge"
MODES = (MODE_NO_INTERVENTION, MODE_BRAKE_ONLY, MODE_BRAKE_STEER, MODE_BRAKE_PASSAGE_EDGE)

# Tesla Model 3 footprint, same constants as pedestrian_contact.py.
from pedestrian_contact import EGO_HALF_LENGTH_M, EGO_HALF_WIDTH_M  # noqa: E402


def make_hazard_command_fn(mode: str, *, brake_target: float = 1.0):
    """Returns a run_scenario hazard_command_fn for a mode:
    (sim_time_s, triggered, trigger_time_s) -> (hazard_active, brake_target)."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if not 0.0 <= brake_target <= 1.0:
        raise ValueError("brake_target must be in [0, 1]")

    def hazard_command(sim_time_s, triggered, trigger_time_s):
        if mode == MODE_NO_INTERVENTION:
            return False, 0.0
        return bool(triggered), brake_target

    return hazard_command


def mode_uses_steering(mode: str) -> bool:
    return mode in (MODE_BRAKE_STEER, MODE_BRAKE_PASSAGE_EDGE)


class PassageEdgeOffset:
    """run_scenario lateral_offset_fn for brake_passage_edge: 0 before onset,
    then the lateral target for passage coordinate `u` (default +1, the
    passage's right edge), re-evaluated every tick from the latest passage.

    The caller supplies the passage via set_passage() (computed from the CARLA
    map and the pedestrian's position; for a stationary pedestrian it is the
    same every tick, so computing it once before onset is exact). Until a
    passage is set, the target is 0 and `missing_passage_ticks` counts the
    post-onset ticks affected, so a run can never silently swerve on nothing.
    """

    def __init__(self, *, u: float = 1.0, u_min: float = 0.0, u_max: float = 1.0):
        self.u, self.u_min, self.u_max = u, u_min, u_max
        self.passage = None
        self.missing_passage_ticks = 0
        self.last_target_m = 0.0

    def set_passage(self, passage) -> None:
        self.passage = passage

    def __call__(self, sim_time_s, triggered, trigger_time_s, hazard_clear_info=None):
        from passage import passage_coordinate_to_offset_m

        if not triggered:
            self.last_target_m = 0.0
        elif self.passage is None:
            self.missing_passage_ticks += 1
            self.last_target_m = 0.0
        else:
            self.last_target_m = passage_coordinate_to_offset_m(self.u, self.passage, u_min=self.u_min,
                                                                u_max=self.u_max)
        return self.last_target_m


def expand_runs(modes, passage_u=(1.0,)) -> List[Tuple[str, object, str]]:
    """[(mode, u, label)]: brake_passage_edge once per aim position u, every
    other mode once (u = None). The label names the run's folder."""
    runs = []
    for mode in modes:
        if mode == MODE_BRAKE_PASSAGE_EDGE:
            runs += [(mode, u, f"{mode}_u{u:g}") for u in passage_u]
        else:
            runs.append((mode, None, mode))
    return runs


def max_abs_yaw_change_deg(yaws_deg):
    """Largest heading change from the first sample, wrapped to [-180, 180]
    (spin check). None for no samples."""
    if not yaws_deg:
        return None
    return max(abs((y - yaws_deg[0] + 180.0) % 360.0 - 180.0) for y in yaws_deg)


def best_option(rows):
    """(kind, row) for one scenario's runs. "safe": the safe run with the most
    clearance. Else "no_contact_unsafe": a run that missed the pedestrian but
    is not safe (left the road, or unresolved), most clearance. Else
    "all_contact": the run with the lowest impact speed. Ties keep the first."""
    safe = [r for r in rows if r["safe_success"]]
    if safe:
        return "safe", max(safe, key=lambda r: r["min_clearance_m"])
    missed = [r for r in rows if r["outcome"] != "contact"]
    if missed:
        return "no_contact_unsafe", max(missed, key=lambda r: r["min_clearance_m"])
    return "all_contact", min(rows, key=lambda r: r["contact_speed_mps"])


def _option_name(r) -> str:
    return r["mode"] if r.get("passage_u") is None else f"u={r['passage_u']:g}"


def _result_cell(r) -> str:
    if r["outcome"] == "contact":
        return f"hit {r['contact_speed_mps']:.1f} m/s"
    what = {"passed_clear": "passed", "stopped_clear": "stopped"}.get(r["outcome"], r["outcome"])
    off_road = " OFF-ROAD" if r.get("drivable_violation_ticks") else ""
    return f"{what} {r['min_clearance_m']:.2f} m{off_road}"


def _best_cell(rows) -> str:
    kind, r = best_option(rows)
    if kind == "safe":
        return f"{_option_name(r)} ({r['min_clearance_m']:.2f} m)"
    if kind == "no_contact_unsafe":
        return f"none safe; {_option_name(r)} misses ({_result_cell(r)})"
    return f"all hit; least bad {_option_name(r)} at {r['contact_speed_mps']:.1f} m/s"


def format_best_route_table(rows) -> str:
    """Markdown report from test26 summary rows: per speed, every option's
    result per onset TTC with the best option and the best aim position u;
    then the gate facts (live passage edges, missing-passage ticks, and per u
    whether the car left the road or swung its heading)."""
    out = []
    options = list(dict.fromkeys(_option_name(r) for r in rows))
    for mph in sorted({r["mph"] for r in rows}):
        out += [f"### {mph:g} mph", "",
                "| Onset TTC (s) | " + " | ".join(options) + " | Best option | Best u |",
                "|---" * (len(options) + 3) + "|"]
        for ttc in sorted({r["onset_ttc_s"] for r in rows if r["mph"] == mph}):
            case = [r for r in rows if r["mph"] == mph and r["onset_ttc_s"] == ttc]
            by_name = {_option_name(r): r for r in case}
            aimed = [r for r in case if r.get("passage_u") is not None]
            out.append(f"| {ttc:g} | " + " | ".join(_result_cell(by_name[o]) if o in by_name else "-" for o in options)
                       + f" | {_best_cell(case)} | {_best_cell(aimed) if aimed else '-'} |")
        out.append("")

    aimed = [r for r in rows if r.get("passage_u") is not None]
    if aimed:
        edges = sorted({f"({r['passage_left_edge_m']:.2f}, {r['passage_right_edge_m']:.2f})" for r in aimed})
        out += ["### Passage gate", "",
                "Live passage edges (left, right) in m: " + "; ".join(edges),
                f"Missing-passage ticks, all runs: {sum(r['missing_passage_ticks'] for r in aimed)}", "",
                "| u | aim offset (m) | max lateral reached (m) | max heading change (deg) | runs off-road |",
                "|---|---|---|---|---|"]
        for u in sorted({r["passage_u"] for r in aimed}):
            at_u = [r for r in aimed if r["passage_u"] == u]
            out.append(f"| {u:g} | {max(r['target_offset_m'] for r in at_u):.2f} "
                       f"| {max(r['max_abs_route_lateral_m'] for r in at_u):.2f} "
                       f"| {max(r['max_abs_yaw_change_deg'] for r in at_u):.1f} "
                       f"| {sum(bool(r['drivable_violation_ticks']) for r in at_u)} of {len(at_u)} |")
        out.append("")
    out.append(f"Contacts: {sum(r['outcome'] == 'contact' for r in rows)} of {len(rows)} runs at the 0.3 m "
               f"pedestrian radius; {sum(bool(r.get('contact_r0188')) for r in rows)} at 0.188 m.")
    return "\n".join(out)


def footprint_corners_xy(
    x_m: float, y_m: float, yaw_deg: float,
    *, half_length_m: float = EGO_HALF_LENGTH_M, half_width_m: float = EGO_HALF_WIDTH_M,
) -> List[Tuple[float, float]]:
    """World-frame (x, y) of the ego footprint's four corners: front-left,
    front-right, rear-right, rear-left. CARLA is left-handed (y to the
    right of +x when viewed from above), so 'right' = forward rotated +90 deg."""
    yaw = math.radians(yaw_deg)
    fx, fy = math.cos(yaw), math.sin(yaw)
    rx, ry = -fy, fx  # right vector in CARLA's left-handed frame
    corners = []
    for lon, lat in ((1, -1), (1, 1), (-1, 1), (-1, -1)):
        corners.append((
            x_m + lon * half_length_m * fx + lat * half_width_m * rx,
            y_m + lon * half_length_m * fy + lat * half_width_m * ry,
        ))
    return corners


def onset_gap_m(*, speed_mps: float, onset_ttc_s: float,
                half_length_m: float = EGO_HALF_LENGTH_M, pedestrian_radius_m: float = 0.3) -> float:
    """Free distance between the ego's front bumper and the pedestrian's
    near edge at onset. run_scenario's trigger measures ego-CENTER distance
    to the encounter point (speed x TTC), so the bumper gap is shorter by
    the half-length plus the pedestrian radius."""
    return speed_mps * onset_ttc_s - half_length_m - pedestrian_radius_m
