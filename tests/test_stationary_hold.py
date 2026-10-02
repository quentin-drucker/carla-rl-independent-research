import math
import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from physics_harness import summarize_stationary_hold  # noqa: E402
from trace_schema import TraceTick  # noqa: E402


def _ticks(positions_yaws_speeds, dt=0.02, pitch=0.0):
    return [
        TraceTick(tick_index=i, sim_time_s=i * dt, pos_x_m=x, pos_y_m=y, pos_z_m=0.0,
                  yaw_deg=yaw, pitch_deg=pitch, roll_deg=0.0, speed_mps=v)
        for i, (x, y, yaw, v) in enumerate(positions_yaws_speeds)
    ]


class SummarizeStationaryHoldTests(unittest.TestCase):
    def test_perfectly_still_vehicle(self):
        s = summarize_stationary_hold(_ticks([(5.0, 5.0, 90.0, 0.0)] * 200))
        self.assertEqual(s["net_displacement_m"], 0.0)
        self.assertEqual(s["path_length_m"], 0.0)
        self.assertEqual(s["heading_change_deg"], 0.0)
        self.assertIsNone(s["motion_direction_rel_heading_deg"])
        self.assertEqual(s["late_mean_speed_mps"], 0.0)

    def test_backward_roll_is_about_180(self):
        # Facing +x (yaw 0), moving toward -x.
        s = summarize_stationary_hold(_ticks([(-0.01 * i, 0.0, 0.0, 0.5) for i in range(100)]))
        self.assertAlmostEqual(abs(s["motion_direction_rel_heading_deg"]), 180.0, places=6)
        self.assertAlmostEqual(s["net_displacement_m"], 0.99, places=6)

    def test_sideways_slide_to_the_right_is_plus_90(self):
        # Facing +x; CARLA's right-hand side is +y (left-handed, z-up world).
        s = summarize_stationary_hold(_ticks([(0.0, 0.01 * i, 0.0, 0.5) for i in range(100)]))
        self.assertAlmostEqual(s["motion_direction_rel_heading_deg"], 90.0, places=6)

    def test_heading_change_unwraps_across_180(self):
        yaws = [179.0 + 0.1 * i for i in range(30)]  # 179 -> 181.9 == -178.1
        yaws = [y - 360.0 if y > 180.0 else y for y in yaws]
        s = summarize_stationary_hold(_ticks([(0.0, 0.0, y, 0.0) for y in yaws]))
        self.assertAlmostEqual(s["heading_change_deg"], 2.9, places=6)

    def test_decaying_vs_steady_creep_windows(self):
        decaying = [(0, 0, 0, max(0.0, 0.3 - 0.05 * (i * 0.02))) for i in range(500)]
        steady = [(0, 0, 0, 0.16) for _ in range(500)]
        d = summarize_stationary_hold(_ticks(decaying))
        s = summarize_stationary_hold(_ticks(steady))
        self.assertLess(d["late_mean_speed_mps"], d["early_mean_speed_mps"])
        self.assertAlmostEqual(s["late_mean_speed_mps"], s["early_mean_speed_mps"])

    def test_empty_input(self):
        self.assertEqual(summarize_stationary_hold([]), {})

    def test_pitch_change_is_relative_to_first_tick(self):
        rows = _ticks([(0, 0, 0, 0)] * 3, pitch=0.8)
        rows[1].pitch_deg = 3.1
        self.assertAlmostEqual(summarize_stationary_hold(rows)["max_abs_pitch_change_deg"], 2.3)


if __name__ == "__main__":
    unittest.main()
