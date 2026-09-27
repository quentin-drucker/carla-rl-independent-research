"""
test14___trajectory_visualization_mvp.py
Quentin | Week 3 Workstream 3.2/3.3 (trajectory-visualization tooling).

Runs the EXISTING scripted/classical controller (test3's run_scenario(),
completely unmodified in its control logic) under identical fixed
conditions multiple times, recording each run to a persisted TraceTick CSV
via trajectory_recording.TrajectoryRecorder, then renders:
  - a world-frame spatial overlay of all repeats + route centerline +
    event markers (trigger, closest_approach), and
  - a companion time-series panel (lateral offset, speed, brake, steer).

Positive control: one run with a materially different walker_side, so the
plot should visibly separate it from the fixed-condition repeats -- this is
the check the plan's 3.3 explicitly asks for ("a known changed
configuration produces a correctly distinguishable trajectory"), not a
second real experiment.

Usage:
    python test14___trajectory_visualization_mvp.py
Requires a running CARLA server (localhost:2000) -- this is a live/manual
test, not part of the offline suite (see tests/test_trajectory_recording.py
and tests/test_trajectory_plot.py for the offline-testable pieces this
script assembles).
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from scenario_config import ScenarioConfig
from test3___ped_intrusion_scenario import run_scenario
from test5___scripted_pedestrian_steering import build_evasive_offset_fn
from trajectory_recording import TrajectoryRecorder
from trace_schema import read_trace_csv
from trajectory_plot import (
    prepare_overlay_series,
    render_overlay_figure,
    render_time_series_panel,
)

FIXED_DT = 0.02
RUN_DIR = os.path.join(os.path.dirname(__file__), "runs", "trajectory_viz")


def _base_config(run_id: str) -> ScenarioConfig:
    return ScenarioConfig(
        run_id=run_id,
        walker_speed_mps=1.8,
        walker_side="left",
        walker_cross="near",
        trigger_ttc_s=2.8,
        trigger_delay_s=0.0,
        walker_post_trigger_delay_s=0.0,
        target_mph=35.0,
        encounter_distance_m=120.0,
        brake_profile="exponential",
        weather_preset="ClearSunset",
        sun_altitude_deg=0,
        cloudiness=None,
        sim_seconds=13.0,
    )


def _record_one_run(cfg: ScenarioConfig, csv_path: str, *, lateral_offset_fn=None) -> None:
    """Runs one scenario, recording every tick's telemetry into a
    TrajectoryRecorder via run_scenario()'s existing tick_observer hook --
    no changes to run_scenario()'s control logic."""
    recorder = TrajectoryRecorder(dt_s=FIXED_DT)
    state = {"tick": 0, "prev_triggered": False, "prev_drive_mode": "CRUISE"}

    def observer(sim_time_s, triggered, telemetry):
        marker = None
        if triggered and not state["prev_triggered"]:
            marker = "trigger"
        elif telemetry is not None:
            mode = telemetry.get("drive_mode", "CRUISE")
            if mode in ("HAZARD_BRAKE",) and state["prev_drive_mode"] not in ("HAZARD_BRAKE", "STOP_HOLD"):
                marker = "hazard_activation"
            state["prev_drive_mode"] = mode
        state["prev_triggered"] = triggered

        recorder.record(
            tick_index=state["tick"], sim_time_s=sim_time_s,
            telemetry=telemetry, event_marker=marker,
        )
        state["tick"] += 1

    run_scenario(cfg, tick_observer=observer, lateral_offset_fn=lateral_offset_fn,
                 monitor_lateral_corridors=lateral_offset_fn is not None)

    recorder.tag_closest_approach()
    if recorder.ticks:
        last = recorder.ticks[-1]
        from dataclasses import replace
        new_marker = "termination" if not last.event_marker else f"{last.event_marker}+termination"
        recorder.ticks[-1] = replace(last, event_marker=new_marker)

    recorder.write(csv_path)
    print(f"[trajectory_viz] wrote {len(recorder.ticks)} ticks -> {csv_path}")


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(RUN_DIR, stamp)
    os.makedirs(out_dir, exist_ok=True)

    run_specs = [
        ("baseline_repeat_1", _base_config("traj_baseline_1"), None),
        ("baseline_repeat_2", _base_config("traj_baseline_2"), None),
        ("baseline_repeat_3", _base_config("traj_baseline_3"), None),
    ]
    control_cfg = _base_config("traj_positive_control")
    control_cfg.walker_side = "right"  # deliberately different -- weak positive control
    run_specs.append(("positive_control_walker_right", control_cfg, None))

    # Second, stronger positive control: an actual evasive swerve. Changing
    # only walker_side leaves the EGO's own spatial path essentially
    # unchanged (this controller has no lateral reaction to which side the
    # pedestrian starts from) -- found live while inspecting the first
    # overlay plot. A commanded lateral offset is the config change that
    # should visibly separate the EGO's own trajectory, which is what the
    # plan's 3.3 success criterion is actually about.
    swerve_cfg = _base_config("traj_positive_control_swerve")
    run_specs.append(("positive_control_swerve", swerve_cfg, build_evasive_offset_fn(peak_offset_m=2.5)))

    csv_paths = []
    for label, cfg, offset_fn in run_specs:
        csv_path = os.path.join(out_dir, f"{label}.csv")
        _record_one_run(cfg, csv_path, lateral_offset_fn=offset_fn)
        csv_paths.append((label, csv_path))

    prepared = []
    labels = []
    for label, csv_path in csv_paths:
        ticks = read_trace_csv(csv_path)
        prepared.append(prepare_overlay_series(ticks))
        labels.append(label)

    overlay_path = os.path.join(out_dir, "overlay.png")
    timeseries_path = os.path.join(out_dir, "timeseries.png")
    render_overlay_figure(prepared, labels, save_path=overlay_path,
                           title="Week 3 Workstream 3.3: fixed-condition repeats + positive control")
    render_time_series_panel(prepared, labels, save_path=timeseries_path,
                              title="Week 3 Workstream 3.3: fixed-condition repeats + positive control")

    print(f"[trajectory_viz] overlay plot -> {overlay_path}")
    print(f"[trajectory_viz] time-series plot -> {timeseries_path}")


if __name__ == "__main__":
    main()
