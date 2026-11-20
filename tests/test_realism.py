"""Week 11 realism options: steering rate limit, brake cap, their plumbing; all off by default."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from carla_aeb_env import cap_brake  # noqa: E402
from lane_follow import limit_steer_rate  # noqa: E402


class SteerRateLimitTests(unittest.TestCase):
    def test_default_is_unlimited(self):
        state = {"steer_prev": -1.0}
        self.assertEqual(limit_steer_rate(1.0, state, 0.02), 1.0)
        state = {"steer_rate_limit_per_s": None, "steer_prev": -1.0}
        self.assertEqual(limit_steer_rate(1.0, state, 0.02), 1.0)

    def test_clamps_change_per_tick_both_ways(self):
        state = {"steer_rate_limit_per_s": 2.0}
        self.assertEqual(limit_steer_rate(0.0, state, 0.02), 0.0)  # first tick: nothing to limit against
        out = [limit_steer_rate(1.0, state, 0.02) for _ in range(30)]
        self.assertAlmostEqual(out[0], 0.04)
        self.assertAlmostEqual(out[24], 1.0)  # full lock after 0.5 s
        self.assertAlmostEqual(limit_steer_rate(-1.0, state, 0.02), 0.96)
        self.assertAlmostEqual(state["steer_prev"], 0.96)
        self.assertAlmostEqual(limit_steer_rate(0.97, state, 0.02), 0.97)  # small changes pass unchanged


class BrakeCapTests(unittest.TestCase):
    def test_cap(self):
        self.assertEqual(cap_brake(1.0, 0.75), 0.75)
        self.assertEqual(cap_brake(0.3, 0.75), 0.3)
        self.assertEqual(cap_brake(1.0, None), 1.0)
        self.assertIsNone(cap_brake(None, 0.75))  # no hazard command: cruise control decides


class PlumbingTests(unittest.TestCase):
    def test_run_circuit_follows_manifest_unless_flag_given(self):
        from run_circuit import _parse_args, realism_setting
        with tempfile.TemporaryDirectory() as d:
            model = os.path.join(d, "final_model.zip")
            with open(os.path.join(d, "manifest.json"), "w") as f:
                json.dump({"steer_rate_limit_per_s": 2.0, "max_brake": 0.75}, f)
            self.assertEqual(realism_setting(None, model, "steer_rate_limit_per_s"), 2.0)
            self.assertEqual(realism_setting(1.0, model, "steer_rate_limit_per_s"), 1.0)
            self.assertEqual(realism_setting(None, model, "max_brake"), 0.75)
        self.assertIsNone(realism_setting(None, None, "max_brake"))
        a = _parse_args(["--scripted-aim", "wider_side"])
        self.assertIsNone(a.steer_rate_limit)
        self.assertIsNone(a.max_brake)
        self.assertFalse(a.keep_markers)
        with self.assertRaises(SystemExit):
            _parse_args(["--scripted-aim", "wider_side", "--max-brake", "1.5"])

    def test_train_steering_flags(self):
        from train_steering import _parse_args
        a = _parse_args(["--commit", "per_tick"])
        self.assertIsNone(a.steer_rate_limit)
        self.assertIsNone(a.max_brake)
        a = _parse_args(["--commit", "per_tick", "--steer-rate-limit", "2", "--max-brake", "0.75"])
        self.assertEqual((a.steer_rate_limit, a.max_brake), (2.0, 0.75))
        with self.assertRaises(SystemExit):
            _parse_args(["--commit", "per_tick", "--steer-rate-limit", "0"])


if __name__ == "__main__":
    unittest.main()
