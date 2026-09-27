import math
import os
import sys
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trace_schema import TraceTick  # noqa: E402
from trajectory_plot import (  # noqa: E402
    prepare_overlay_series,
    prepare_route_reference,
    render_overlay_figure,
    render_time_series_panel,
)

_Point = namedtuple("_Point", ["x", "y"])


def _straight_trajectory(n=10, event_at=None, event_name="trigger"):
    """Synthetic straight-line trajectory along world X, y=0, constant speed."""
    ticks = []
    for i in range(n):
        ticks.append(TraceTick(
            tick_index=i, sim_time_s=i * 0.1,
            pos_x_m=float(i), pos_y_m=0.0, pos_z_m=0.0,
            yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
            speed_mps=10.0,
            applied_steer=0.0, applied_brake=0.0,
            route_lateral_m=0.0,
            event_marker=event_name if i == event_at else None,
        ))
    return ticks


def _curved_trajectory(n=10, radius_m=20.0):
    """Synthetic quarter-circle arc, used to check the spatial plot handles
    non-straight paths without distortion (e.g. no accidental transpose)."""
    ticks = []
    for i in range(n):
        theta = (i / (n - 1)) * (math.pi / 2)
        x = radius_m * math.sin(theta)
        y = radius_m * (1 - math.cos(theta))
        ticks.append(TraceTick(
            tick_index=i, sim_time_s=i * 0.1,
            pos_x_m=x, pos_y_m=y, pos_z_m=0.0,
            yaw_deg=math.degrees(theta), pitch_deg=0.0, roll_deg=0.0,
            speed_mps=8.0,
            applied_steer=0.3, applied_brake=0.0,
            route_lateral_m=1.5,
        ))
    return ticks


class PrepareOverlaySeriesTests(unittest.TestCase):
    def test_straight_trajectory_coordinates_pass_through_unchanged(self):
        ticks = _straight_trajectory(n=5)
        series = prepare_overlay_series(ticks)
        self.assertEqual(series["pos_x_m"], [0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(series["pos_y_m"], [0.0, 0.0, 0.0, 0.0, 0.0])
        for actual, expected in zip(series["sim_time_s"], [0.0, 0.1, 0.2, 0.3, 0.4]):
            self.assertAlmostEqual(actual, expected, places=9)

    def test_curved_trajectory_preserves_x_y_pairing(self):
        ticks = _curved_trajectory(n=4, radius_m=20.0)
        series = prepare_overlay_series(ticks)
        # Last point of a quarter circle of radius 20 should be near (20, 20),
        # not swapped or mirrored.
        self.assertAlmostEqual(series["pos_x_m"][-1], 20.0, places=6)
        self.assertAlmostEqual(series["pos_y_m"][-1], 20.0, places=6)
        self.assertAlmostEqual(series["pos_x_m"][0], 0.0, places=6)
        self.assertAlmostEqual(series["pos_y_m"][0], 0.0, places=6)

    def test_route_lateral_falls_back_to_lateral_displacement(self):
        tick = TraceTick(
            tick_index=0, sim_time_s=0.0, pos_x_m=0.0, pos_y_m=0.0, pos_z_m=0.0,
            yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0, speed_mps=5.0,
            route_lateral_m=None, lateral_displacement_m=2.5,
        )
        series = prepare_overlay_series([tick])
        self.assertEqual(series["route_lateral_m"], [2.5])

    def test_event_marker_indices_extracted_correctly(self):
        ticks = _straight_trajectory(n=6, event_at=3, event_name="trigger")
        series = prepare_overlay_series(ticks)
        self.assertEqual(series["event_ticks"], {"trigger": [3]})

    def test_combined_event_marker_splits_into_both_names(self):
        ticks = _straight_trajectory(n=4)
        ticks[2] = ticks[2].__class__(**{**ticks[2].__dict__, "event_marker": "trigger+closest_approach"})
        series = prepare_overlay_series(ticks)
        self.assertEqual(series["event_ticks"], {"trigger": [2], "closest_approach": [2]})

    def test_two_runs_do_not_leak_into_each_others_arrays(self):
        run_a = _straight_trajectory(n=5)
        run_b = _curved_trajectory(n=5)
        series_a = prepare_overlay_series(run_a)
        series_b = prepare_overlay_series(run_b)
        self.assertNotEqual(series_a["pos_y_m"], series_b["pos_y_m"])
        self.assertEqual(series_a["pos_y_m"], [0.0] * 5)


class PrepareRouteReferenceTests(unittest.TestCase):
    def test_extracts_x_y_from_duck_typed_points(self):
        points = [_Point(0.0, 0.0), _Point(5.0, 0.0), _Point(10.0, 1.0)]
        ref = prepare_route_reference(points)
        self.assertEqual(ref["x_m"], [0.0, 5.0, 10.0])
        self.assertEqual(ref["y_m"], [0.0, 0.0, 1.0])


class RenderFiguresSmokeTests(unittest.TestCase):
    """Rendering correctness (pixel content) is not meaningful to unit-test;
    these just confirm the functions run against real prepared data and
    produce a non-empty file, and that the non-overwrite guard holds --
    same discipline as write_trace_csv."""

    def test_render_overlay_figure_produces_file(self):
        series = [prepare_overlay_series(_straight_trajectory(n=5, event_at=2))]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "overlay.png")
            render_overlay_figure(series, ["run_a"], save_path=path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_render_overlay_figure_with_route_reference(self):
        series = [prepare_overlay_series(_curved_trajectory(n=5))]
        route_ref = prepare_route_reference([_Point(x, 0.0) for x in range(20)])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "overlay.png")
            render_overlay_figure(series, ["run_a"], save_path=path, route_reference=route_ref)
            self.assertTrue(os.path.exists(path))

    def test_render_overlay_figure_refuses_overwrite(self):
        series = [prepare_overlay_series(_straight_trajectory(n=3))]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "overlay.png")
            render_overlay_figure(series, ["run_a"], save_path=path)
            with self.assertRaises(ValueError):
                render_overlay_figure(series, ["run_a"], save_path=path)

    def test_render_overlay_figure_multi_run_overlay(self):
        series = [
            prepare_overlay_series(_straight_trajectory(n=5)),
            prepare_overlay_series(_curved_trajectory(n=5)),
        ]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "overlay.png")
            render_overlay_figure(series, ["baseline", "positive_control"], save_path=path)
            self.assertTrue(os.path.exists(path))

    def test_render_overlay_figure_rejects_mismatched_labels(self):
        series = [prepare_overlay_series(_straight_trajectory(n=3))]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "overlay.png")
            with self.assertRaises(ValueError):
                render_overlay_figure(series, ["a", "b"], save_path=path)

    def test_render_time_series_panel_produces_file(self):
        series = [prepare_overlay_series(_straight_trajectory(n=5))]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "timeseries.png")
            render_time_series_panel(series, ["run_a"], save_path=path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_render_time_series_panel_refuses_overwrite(self):
        series = [prepare_overlay_series(_straight_trajectory(n=3))]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "timeseries.png")
            render_time_series_panel(series, ["run_a"], save_path=path)
            with self.assertRaises(ValueError):
                render_time_series_panel(series, ["run_a"], save_path=path)


if __name__ == "__main__":
    unittest.main()
