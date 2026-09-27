import os
import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajectory_recording import build_trace_tick_from_telemetry, TrajectoryRecorder  # noqa: E402
from trace_schema import read_trace_csv  # noqa: E402


def _telemetry(**overrides):
    base = dict(
        pos_x_m=0.0, pos_y_m=0.0, pos_z_m=0.0,
        yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
        speed_mps=10.0,
        throttle_cmd=0.5, brake_cmd=0.0, steer_cmd=0.0,
        lateral_offset_requested_m=1.5, signed_route_lateral_offset_m=0.0,
        pedestrian_x_m=None, pedestrian_y_m=None,
    )
    base.update(overrides)
    return base


class BuildTraceTickTests(unittest.TestCase):
    def test_first_tick_has_no_derived_kinematics(self):
        tick = build_trace_tick_from_telemetry(
            tick_index=0, sim_time_s=0.0, telemetry=_telemetry(),
            prev_tick=None, dt_s=0.02,
        )
        self.assertIsNone(tick.accel_mps2)
        self.assertIsNone(tick.yaw_rate_dps)
        self.assertIsNone(tick.vel_x_mps)
        self.assertIsNone(tick.body_slip_angle_deg)

    def test_pose_and_control_fields_pass_through(self):
        tick = build_trace_tick_from_telemetry(
            tick_index=5, sim_time_s=0.1,
            telemetry=_telemetry(pos_x_m=12.0, pos_y_m=3.0, speed_mps=8.0, steer_cmd=0.25),
            prev_tick=None, dt_s=0.02,
        )
        self.assertEqual(tick.pos_x_m, 12.0)
        self.assertEqual(tick.pos_y_m, 3.0)
        self.assertEqual(tick.speed_mps, 8.0)
        self.assertEqual(tick.applied_steer, 0.25)
        self.assertEqual(tick.requested_lateral_offset_m, 1.5)

    def test_second_tick_derives_velocity_from_position_delta(self):
        tick0 = build_trace_tick_from_telemetry(
            tick_index=0, sim_time_s=0.0,
            telemetry=_telemetry(pos_x_m=0.0, pos_y_m=0.0, speed_mps=10.0),
            prev_tick=None, dt_s=0.02,
        )
        tick1 = build_trace_tick_from_telemetry(
            tick_index=1, sim_time_s=0.02,
            telemetry=_telemetry(pos_x_m=0.2, pos_y_m=0.0, speed_mps=10.0),
            prev_tick=tick0, dt_s=0.02,
        )
        self.assertAlmostEqual(tick1.vel_x_mps, 10.0, places=6)
        self.assertAlmostEqual(tick1.vel_y_mps, 0.0, places=6)
        self.assertAlmostEqual(tick1.accel_mps2, 0.0, places=6)

    def test_missing_telemetry_keys_degrade_to_none_not_crash(self):
        tick = build_trace_tick_from_telemetry(
            tick_index=0, sim_time_s=0.0, telemetry={"speed_mps": 5.0},
            prev_tick=None, dt_s=0.02,
        )
        self.assertEqual(tick.pos_x_m, 0.0)
        self.assertIsNone(tick.requested_lateral_offset_m)
        self.assertIsNone(tick.pedestrian_x_m)


class TrajectoryRecorderTests(unittest.TestCase):
    def test_none_telemetry_tick_is_skipped(self):
        rec = TrajectoryRecorder(dt_s=0.02)
        result = rec.record(tick_index=0, sim_time_s=0.0, telemetry=None)
        self.assertIsNone(result)
        self.assertEqual(len(rec.ticks), 0)

    def test_records_accumulate_in_order(self):
        rec = TrajectoryRecorder(dt_s=0.02)
        for i in range(3):
            rec.record(tick_index=i, sim_time_s=i * 0.02, telemetry=_telemetry(pos_x_m=float(i)))
        self.assertEqual([t.tick_index for t in rec.ticks], [0, 1, 2])
        self.assertEqual([t.pos_x_m for t in rec.ticks], [0.0, 1.0, 2.0])

    def test_tag_closest_approach_marks_min_distance_tick(self):
        rec = TrajectoryRecorder(dt_s=0.02)
        # Ego moves along x=0..4; pedestrian fixed at (2, 0.1) -> closest at tick 2.
        for i in range(5):
            rec.record(
                tick_index=i, sim_time_s=i * 0.02,
                telemetry=_telemetry(pos_x_m=float(i), pedestrian_x_m=2.0, pedestrian_y_m=0.1),
            )
        rec.tag_closest_approach()
        markers = [t.event_marker for t in rec.ticks]
        self.assertEqual(markers, [None, None, "closest_approach", None, None])

    def test_tag_closest_approach_appends_to_existing_marker(self):
        rec = TrajectoryRecorder(dt_s=0.02)
        for i in range(3):
            marker = "trigger" if i == 1 else None
            rec.record(
                tick_index=i, sim_time_s=i * 0.02,
                telemetry=_telemetry(pos_x_m=float(i), pedestrian_x_m=1.0, pedestrian_y_m=0.0),
                event_marker=marker,
            )
        rec.tag_closest_approach()
        self.assertEqual(rec.ticks[1].event_marker, "trigger+closest_approach")

    def test_tag_closest_approach_is_noop_without_pedestrian_data(self):
        rec = TrajectoryRecorder(dt_s=0.02)
        for i in range(3):
            rec.record(tick_index=i, sim_time_s=i * 0.02, telemetry=_telemetry(pos_x_m=float(i)))
        rec.tag_closest_approach()
        self.assertTrue(all(t.event_marker is None for t in rec.ticks))

    def test_write_round_trips_through_shared_trace_schema(self):
        rec = TrajectoryRecorder(dt_s=0.02)
        for i in range(3):
            rec.record(tick_index=i, sim_time_s=i * 0.02, telemetry=_telemetry(pos_x_m=float(i)))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.csv")
            rec.write(path)
            read_back = read_trace_csv(path)
        self.assertEqual(len(read_back), 3)
        self.assertEqual([t.pos_x_m for t in read_back], [0.0, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
