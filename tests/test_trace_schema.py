import os
import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trace_schema import (  # noqa: E402
    TraceTick,
    PhysicsRunManifest,
    write_trace_csv,
    read_trace_csv,
)


def _make_tick(tick_index=0, **overrides):
    base = dict(
        tick_index=tick_index,
        sim_time_s=tick_index * 0.02,
        pos_x_m=1.0,
        pos_y_m=2.0,
        pos_z_m=0.0,
        yaw_deg=10.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        speed_mps=5.0,
    )
    base.update(overrides)
    return TraceTick(**base)


class TraceTickSerializationTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def test_round_trip_preserves_required_fields(self):
        ticks = [_make_tick(0), _make_tick(1)]
        path = os.path.join(self._tmpdir.name, "trace.csv")
        write_trace_csv(path, ticks)
        loaded = read_trace_csv(path)

        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].tick_index, 0)
        self.assertEqual(loaded[1].tick_index, 1)
        self.assertAlmostEqual(loaded[0].pos_x_m, 1.0)
        self.assertAlmostEqual(loaded[1].speed_mps, 5.0)

    def test_round_trip_preserves_none_as_none_not_zero(self):
        # accel_mps2 is None on a first tick (no previous sample) -- must
        # not silently become 0.0 after a CSV round trip, which would look
        # like "measured zero acceleration" instead of "not computed".
        tick = _make_tick(
            0, accel_mps2=None, yaw_rate_dps=None, applied_throttle=None,
            body_slip_angle_deg=None,
        )
        path = os.path.join(self._tmpdir.name, "trace.csv")
        write_trace_csv(path, [tick])
        loaded = read_trace_csv(path)

        self.assertIsNone(loaded[0].accel_mps2)
        self.assertIsNone(loaded[0].yaw_rate_dps)
        self.assertIsNone(loaded[0].applied_throttle)
        self.assertIsNone(loaded[0].body_slip_angle_deg)

    def test_round_trip_preserves_velocity_and_slip_angle(self):
        tick = _make_tick(0, vel_x_mps=12.3, vel_y_mps=-1.5, body_slip_angle_deg=7.25)
        path = os.path.join(self._tmpdir.name, "trace.csv")
        write_trace_csv(path, [tick])
        loaded = read_trace_csv(path)

        self.assertAlmostEqual(loaded[0].vel_x_mps, 12.3)
        self.assertAlmostEqual(loaded[0].vel_y_mps, -1.5)
        self.assertAlmostEqual(loaded[0].body_slip_angle_deg, 7.25)

    def test_round_trip_preserves_event_marker_string(self):
        tick = _make_tick(0, event_marker="hazard_activation")
        path = os.path.join(self._tmpdir.name, "trace.csv")
        write_trace_csv(path, [tick])
        loaded = read_trace_csv(path)

        self.assertEqual(loaded[0].event_marker, "hazard_activation")

    def test_refuses_to_overwrite_existing_trace_file(self):
        path = os.path.join(self._tmpdir.name, "trace.csv")
        write_trace_csv(path, [_make_tick(0)])
        with self.assertRaises(ValueError):
            write_trace_csv(path, [_make_tick(1)])

    def test_empty_tick_list_writes_header_only(self):
        path = os.path.join(self._tmpdir.name, "trace.csv")
        write_trace_csv(path, [])
        loaded = read_trace_csv(path)
        self.assertEqual(loaded, [])


class PhysicsRunManifestSerializationTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def _make_manifest(self, **overrides):
        base = dict(
            run_id="20260926_120000_braking",
            test_family="braking",
            created_at_iso="2026-09-26T12:00:00",
            carla_version="0.9.16",
            map_name="Carla/Maps/Town04_Opt",
            vehicle_blueprint="vehicle.tesla.model3",
            fixed_delta_seconds=0.02,
            substepping_enabled=True,
            max_substep_delta_time=0.01,
            max_substeps=10,
            weather_preset="ClearNoon",
            tire_friction=None,
            random_seed=2026,
            parameters={"target_mph": 35.0},
        )
        base.update(overrides)
        return PhysicsRunManifest(**base)

    def test_round_trip_preserves_fields(self):
        manifest = self._make_manifest()
        path = os.path.join(self._tmpdir.name, "manifest.json")
        manifest.to_json(path)
        loaded = PhysicsRunManifest.from_json(path)

        self.assertEqual(loaded.run_id, manifest.run_id)
        self.assertEqual(loaded.parameters, manifest.parameters)
        self.assertIsNone(loaded.tire_friction)

    def test_from_dict_ignores_unknown_keys(self):
        d = self._make_manifest().to_dict()
        d["some_future_field"] = "ignored"
        loaded = PhysicsRunManifest.from_dict(d)
        self.assertEqual(loaded.test_family, "braking")


if __name__ == "__main__":
    unittest.main()
