"""
trajectory_recording.py

Week 3 Workstream 3.1: a controller-independent trajectory record.

Converts the telemetry dict already returned by lane_follow_step (plus a
couple of scenario-loop-only quantities: triggered/drive_mode transitions,
which lane_follow_step itself does not know about) into the shared
TraceTick format from trace_schema.py -- the same schema the physical-
limits test suite (Workstream 2) already writes.

This module is deliberately pure/offline-testable: build_trace_tick_from_
telemetry() takes a plain dict and floats, no CARLA actor objects. The only
CARLA-touching code in the trajectory-visualization path lives in the
calling script (e.g. test14), which just passes the same telemetry dict
run_scenario()'s tick_observer already receives.

Why not modify run_scenario() itself to write traces: run_scenario() is
used by sweep.py and every existing test1/test5/test8/test13 script.
Recording is opt-in, additive instrumentation -- it belongs in the
tick_observer callback contract that already exists for exactly this kind
of manual-test observation, not baked into the scenario runner.
"""

from dataclasses import replace
from typing import List, Optional

from trace_schema import TraceTick
from physics_harness import compute_derived_kinematics, compute_body_slip_angle_deg


def build_trace_tick_from_telemetry(
    *,
    tick_index: int,
    sim_time_s: float,
    telemetry: dict,
    prev_tick: Optional[TraceTick],
    dt_s: float,
    event_marker: Optional[str] = None,
) -> TraceTick:
    """Build one TraceTick from a lane_follow_step telemetry dict.

    prev_tick supplies the previous sample for finite-difference accel/yaw
    rate/body-slip-angle -- None on the first tick of a run, same contract
    as physics_harness.compute_derived_kinematics.

    Any telemetry key this function needs but that is missing (e.g. an
    older caller that hasn't added pos_x_m/pedestrian_x_m yet) reads as
    None rather than raising -- a partially-populated TraceTick is
    reportable (route/physical-limits fields degrade to
    not_measurable/None), a crash mid-run is not.
    """
    pos_x_m = telemetry.get("pos_x_m")
    pos_y_m = telemetry.get("pos_y_m")
    speed_mps = telemetry.get("speed_mps", 0.0)
    yaw_deg = telemetry.get("yaw_deg")

    prev_speed_mps = prev_tick.speed_mps if prev_tick is not None else None
    prev_yaw_deg = prev_tick.yaw_deg if prev_tick is not None else None
    accel_mps2, yaw_rate_dps = compute_derived_kinematics(
        prev_speed_mps=prev_speed_mps,
        curr_speed_mps=speed_mps,
        prev_yaw_deg=prev_yaw_deg,
        curr_yaw_deg=yaw_deg if yaw_deg is not None else 0.0,
        dt_s=dt_s,
    )

    vel_x_mps = None
    vel_y_mps = None
    if pos_x_m is not None and pos_y_m is not None and prev_tick is not None and dt_s > 0:
        vel_x_mps = (pos_x_m - prev_tick.pos_x_m) / dt_s
        vel_y_mps = (pos_y_m - prev_tick.pos_y_m) / dt_s

    body_slip_angle_deg = None
    if vel_x_mps is not None and vel_y_mps is not None and yaw_deg is not None:
        body_slip_angle_deg = compute_body_slip_angle_deg(
            vel_x_mps=vel_x_mps, vel_y_mps=vel_y_mps, yaw_deg=yaw_deg
        )

    return TraceTick(
        tick_index=tick_index,
        sim_time_s=sim_time_s,
        pos_x_m=pos_x_m if pos_x_m is not None else 0.0,
        pos_y_m=pos_y_m if pos_y_m is not None else 0.0,
        pos_z_m=telemetry.get("pos_z_m") or 0.0,
        yaw_deg=yaw_deg if yaw_deg is not None else 0.0,
        pitch_deg=telemetry.get("pitch_deg") or 0.0,
        roll_deg=telemetry.get("roll_deg") or 0.0,
        speed_mps=speed_mps,
        vel_x_mps=vel_x_mps,
        vel_y_mps=vel_y_mps,
        accel_mps2=accel_mps2,
        yaw_rate_dps=yaw_rate_dps,
        body_slip_angle_deg=body_slip_angle_deg,
        requested_throttle=telemetry.get("throttle_cmd", 0.0),
        requested_brake=telemetry.get("brake_cmd", 0.0),
        requested_steer=telemetry.get("steer_cmd", 0.0),
        applied_throttle=telemetry.get("throttle_cmd"),
        applied_brake=telemetry.get("brake_cmd"),
        applied_steer=telemetry.get("steer_cmd"),
        requested_lateral_offset_m=telemetry.get("lateral_offset_requested_m"),
        route_longitudinal_m=None,  # not computed by lane_follow_step today
        route_lateral_m=telemetry.get("signed_route_lateral_offset_m"),
        pedestrian_x_m=telemetry.get("pedestrian_x_m"),
        pedestrian_y_m=telemetry.get("pedestrian_y_m"),
        event_marker=event_marker,
    )


class TrajectoryRecorder:
    """Accumulates TraceTick rows for one scenario run and applies the
    post-hoc "closest_approach" event marker (the closest point to the
    pedestrian can only be known once the whole run is seen).

    Usage (from a tick_observer callback):
        recorder = TrajectoryRecorder()
        ...
        def observer(sim_time_s, triggered, telemetry):
            recorder.record(tick_index=t, sim_time_s=sim_time_s,
                             telemetry=telemetry, event_marker=marker_or_None)
        ...
        recorder.tag_closest_approach()
        recorder.write(path, manifest)
    """

    def __init__(self, dt_s: float):
        self.dt_s = dt_s
        self.ticks: List[TraceTick] = []

    def record(
        self, *, tick_index: int, sim_time_s: float, telemetry: Optional[dict],
        event_marker: Optional[str] = None,
    ) -> Optional[TraceTick]:
        if telemetry is None:
            return None
        prev_tick = self.ticks[-1] if self.ticks else None
        tick = build_trace_tick_from_telemetry(
            tick_index=tick_index,
            sim_time_s=sim_time_s,
            telemetry=telemetry,
            prev_tick=prev_tick,
            dt_s=self.dt_s,
            event_marker=event_marker,
        )
        self.ticks.append(tick)
        return tick

    def tag_closest_approach(self) -> None:
        """Find the tick with the smallest ego-to-pedestrian distance and
        set its event_marker to "closest_approach" (preserving any existing
        marker by appending, since a tick could coincidentally also be e.g.
        the trigger tick on a very short encounter).

        No-op if no tick has both pedestrian_x_m/y_m populated (e.g. a run
        with no pedestrian at all).
        """
        best_idx = None
        best_dist = float("inf")
        for i, tick in enumerate(self.ticks):
            if tick.pedestrian_x_m is None or tick.pedestrian_y_m is None:
                continue
            dx = tick.pos_x_m - tick.pedestrian_x_m
            dy = tick.pos_y_m - tick.pedestrian_y_m
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is None:
            return

        existing = self.ticks[best_idx]
        new_marker = (
            "closest_approach"
            if not existing.event_marker
            else f"{existing.event_marker}+closest_approach"
        )
        self.ticks[best_idx] = replace(existing, event_marker=new_marker)

    def write(self, path: str) -> None:
        from trace_schema import write_trace_csv
        write_trace_csv(path, self.ticks)
