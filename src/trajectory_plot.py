"""
trajectory_plot.py

Week 3 Workstream 3.2/3.3: fixed-condition overlay MVP for the trajectory
records built by trajectory_recording.py / written via trace_schema.py.

Split deliberately into two layers, same reasoning as physics_harness.py's
pure-function/live-wrapper split:
  - prepare_overlay_series() / prepare_route_reference() are pure functions
    over TraceTick lists -- no matplotlib, fully offline-testable. This is
    where coordinate-transform correctness actually lives (3.3's first
    bullet: "test coordinate transforms ... offline using synthetic
    trajectories").
  - render_overlay_figure() / render_time_series_panel() do the matplotlib
    drawing from already-prepared data. Not unit-tested for pixel content
    (not meaningful), only for "does it run and produce a file."

Uses the Agg backend and always saves to a file rather than plt.show() --
this tool is for comparing MULTIPLE persisted runs after the fact (e.g.
three repeats + one positive control), not live single-run tuning (that
role is already filled by telemetry_plotting.py, which is unaffected).
Refuses to overwrite an existing image, same non-overwrite discipline as
write_trace_csv, so a comparison plot is never silently clobbered.
"""

import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from trace_schema import TraceTick  # noqa: E402


def prepare_overlay_series(ticks: List[TraceTick]) -> dict:
    """Pure extraction of plot-ready arrays from one run's TraceTick list.

    route_lateral_m falls back to lateral_displacement_m when a run has no
    route (e.g. a physical-limits trace reused through this same tool) --
    both are "signed perpendicular offset from a reference line," just
    against different reference lines, so a caller plotting either family
    gets a consistent field name.
    """
    sim_time_s = [t.sim_time_s for t in ticks]
    pos_x_m = [t.pos_x_m for t in ticks]
    pos_y_m = [t.pos_y_m for t in ticks]
    route_lateral_m = [
        t.route_lateral_m if t.route_lateral_m is not None else t.lateral_displacement_m
        for t in ticks
    ]
    speed_mps = [t.speed_mps for t in ticks]
    requested_lateral_offset_m = [t.requested_lateral_offset_m for t in ticks]
    applied_steer = [t.applied_steer for t in ticks]
    applied_brake = [t.applied_brake for t in ticks]

    event_ticks = {}
    for i, t in enumerate(ticks):
        if not t.event_marker:
            continue
        for name in t.event_marker.split("+"):
            event_ticks.setdefault(name, []).append(i)

    return {
        "sim_time_s": sim_time_s,
        "pos_x_m": pos_x_m,
        "pos_y_m": pos_y_m,
        "route_lateral_m": route_lateral_m,
        "speed_mps": speed_mps,
        "requested_lateral_offset_m": requested_lateral_offset_m,
        "applied_steer": applied_steer,
        "applied_brake": applied_brake,
        "event_ticks": event_ticks,
    }


def prepare_route_reference(route_points_world) -> dict:
    """Pure extraction of a route centerline's world X/Y for spatial overlay.

    route_points_world: list of points with .x/.y attributes (carla.Location
    or any duck-typed equivalent -- kept duck-typed so offline tests can
    pass plain namedtuples/objects with no CARLA import).
    """
    return {
        "x_m": [p.x for p in route_points_world],
        "y_m": [p.y for p in route_points_world],
    }


def render_overlay_figure(
    prepared_series_list: List[dict],
    labels: List[str],
    *,
    save_path: str,
    route_reference: Optional[dict] = None,
    title: str = "Fixed-condition trajectory overlay",
) -> None:
    """World-frame spatial overlay of one or more prepared runs.

    Raises ValueError if save_path already exists (see module docstring).
    """
    if os.path.exists(save_path):
        raise ValueError(f"refusing to overwrite existing plot: {save_path}")
    if len(prepared_series_list) != len(labels):
        raise ValueError("prepared_series_list and labels must be the same length")

    fig, ax = plt.subplots(figsize=(9, 7))

    if route_reference is not None:
        ax.plot(
            route_reference["x_m"], route_reference["y_m"],
            color="black", linestyle="--", linewidth=1.2, label="route centerline", zorder=1,
        )

    # Fixed vertical stagger per event name so labels for events that land
    # close together in space (common -- trigger/hazard_activation/closest_
    # approach/termination all cluster near the encounter point on a short
    # straight scenario) don't render on top of each other illegibly.
    _EVENT_LABEL_ROW = {
        "trigger": 0, "hazard_activation": 1, "closest_approach": 2, "termination": 3,
    }

    for series, label in zip(prepared_series_list, labels):
        ax.plot(series["pos_x_m"], series["pos_y_m"], linewidth=1.8, label=label, zorder=3)

        for event_name, indices in series["event_ticks"].items():
            xs = [series["pos_x_m"][i] for i in indices]
            ys = [series["pos_y_m"][i] for i in indices]
            marker = "x" if event_name == "trigger" else "o"
            ax.scatter(xs, ys, marker=marker, s=60, zorder=4)
            row = _EVENT_LABEL_ROW.get(event_name, len(_EVENT_LABEL_ROW))
            for x, y in zip(xs, ys):
                ax.annotate(
                    event_name, (x, y), fontsize=7,
                    xytext=(6, 6 + row * 11), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", linewidth=0.5, alpha=0.6),
                )

    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)

    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def render_time_series_panel(
    prepared_series_list: List[dict],
    labels: List[str],
    *,
    save_path: str,
    title: str = "Fixed-condition time-series comparison",
) -> None:
    """Companion time-series panel: lateral offset, speed, brake, steer vs.
    time, one subplot each, all runs overlaid per subplot so a spatial
    curve in render_overlay_figure() can be tied back to control behavior.
    """
    if os.path.exists(save_path):
        raise ValueError(f"refusing to overwrite existing plot: {save_path}")
    if len(prepared_series_list) != len(labels):
        raise ValueError("prepared_series_list and labels must be the same length")

    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    fig.suptitle(title)

    panels = [
        ("route_lateral_m", "lateral offset (m)\n+right / -left"),
        ("speed_mps", "speed (m/s)"),
        ("applied_brake", "brake (0..1)"),
        ("applied_steer", "steer (-1..+1)"),
    ]

    for ax, (key, ylabel) in zip(axes, panels):
        for series, label in zip(prepared_series_list, labels):
            ax.plot(series["sim_time_s"], series[key], linewidth=1.5, label=label)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("time (s)")

    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
