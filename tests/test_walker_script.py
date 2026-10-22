import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from walker_script import WalkerScript, validate_walker_script  # noqa: E402

DT = 0.02
START = 2.55  # right-side start, m (+right)


def run(script, *, speed_lag=None, seconds=6.0):
    """Drive the script with an ideal walker (moves at the commanded velocity),
    returning (lateral path, script)."""
    lat, path = START, []
    for k in range(int(seconds / DT)):
        direction, speed = script.step(k * DT, lat)
        lat += direction * speed * DT
        path.append(lat)
    return path, script


class ValidateTests(unittest.TestCase):
    def test_rejects_unknown_kind_and_bad_values(self):
        for bad in ([{"kind": "run", "to_frac": 0}], [{"kind": "walk", "speed_mps": 1.0}],
                    [{"kind": "walk", "to_frac": 0, "speed_mps": 0.0}], [{"kind": "wait"}],
                    [{"kind": "wait", "s": -1}], [{"kind": "walk", "to_frac": 0, "speed_mps": 1, "max_s": 0}],
                    [{"kind": "walk", "to_frac": 0, "speed_mps": 1, "extra": 1}]):
            with self.assertRaises(ValueError, msg=bad):
                validate_walker_script(bad)

    def test_needs_an_off_center_start(self):
        with self.assertRaises(ValueError):
            WalkerScript([{"kind": "walk", "to_frac": -1, "speed_mps": 1.8}], start_lateral_m=0.0)


class BehaviorTests(unittest.TestCase):
    def test_empty_script_stands_still(self):
        path, s = run(WalkerScript([], start_lateral_m=START))
        self.assertTrue(all(abs(p - START) < 1e-9 for p in path))
        self.assertEqual(s.transitions[0]["kind"], "done")

    def test_cross_walks_to_the_far_side_and_stops(self):
        path, s = run(WalkerScript([{"kind": "walk", "to_frac": -1, "speed_mps": 1.8}], start_lateral_m=START))
        self.assertAlmostEqual(path[-1], -START, delta=0.05)
        self.assertEqual([t["kind"] for t in s.transitions], ["walk", "done"])

    def test_turn_back_after_a_delay(self):
        script = WalkerScript([{"kind": "walk", "to_frac": -1, "speed_mps": 1.8, "max_s": 0.6},
                               {"kind": "walk", "to_frac": 1, "speed_mps": 1.8}], start_lateral_m=START)
        path, s = run(script)
        turn = s.transitions[1]
        self.assertEqual((turn["phase"], turn["kind"], turn["reason"]), (1, "walk", "max_s"))
        self.assertAlmostEqual(turn["t_s"], 0.6, delta=DT + 1e-9)
        self.assertAlmostEqual(turn["lateral_m"], START - 1.8 * 0.6, delta=0.05)
        self.assertAlmostEqual(min(path), START - 1.8 * 0.6, delta=0.05)  # never went further
        self.assertAlmostEqual(path[-1], START, delta=0.05)               # back where they started

    def test_turn_back_at_a_lateral_threshold(self):
        script = WalkerScript([{"kind": "walk", "to_frac": 0.4, "speed_mps": 1.8},
                               {"kind": "walk", "to_frac": 1, "speed_mps": 1.8}], start_lateral_m=START)
        path, s = run(script)
        self.assertEqual(s.transitions[1]["reason"], "reached")
        self.assertAlmostEqual(s.transitions[1]["lateral_m"], 0.4 * START, delta=0.05)
        self.assertGreater(min(path), 0.4 * START - 0.05)

    def test_late_start_waits_then_walks(self):
        script = WalkerScript([{"kind": "wait", "s": 0.8}, {"kind": "walk", "to_frac": -1, "speed_mps": 1.2}],
                              start_lateral_m=START)
        path, s = run(script)
        self.assertTrue(all(abs(p - START) < 1e-9 for p in path[:39]))   # still at 0.76 s
        self.assertLess(path[45], START)                                  # moving by 0.9 s
        self.assertAlmostEqual(s.transitions[1]["t_s"], 0.8, delta=DT + 1e-9)

    def test_pause_then_restart(self):
        script = WalkerScript([{"kind": "walk", "to_frac": 0.5, "speed_mps": 1.8}, {"kind": "wait", "s": 1.0},
                               {"kind": "walk", "to_frac": -1, "speed_mps": 1.8}], start_lateral_m=START)
        path, s = run(script)
        self.assertEqual([t["kind"] for t in s.transitions], ["walk", "wait", "walk", "done"])
        self.assertAlmostEqual(s.transitions[2]["t_s"] - s.transitions[1]["t_s"], 1.0, delta=DT + 1e-9)
        self.assertAlmostEqual(path[-1], -START, delta=0.05)

    def test_left_start_mirrors(self):
        script = WalkerScript([{"kind": "walk", "to_frac": -1, "speed_mps": 1.8}], start_lateral_m=-START)
        lat = -START
        for k in range(300):
            d, v = script.step(k * DT, lat)
            lat += d * v * DT
        self.assertAlmostEqual(lat, START, delta=0.05)

    def test_phase_and_change_time(self):
        script = WalkerScript([{"kind": "walk", "to_frac": -1, "speed_mps": 1.8, "max_s": 0.6},
                               {"kind": "walk", "to_frac": 1, "speed_mps": 1.8}], start_lateral_m=START)
        self.assertIsNone(script.change_time_s)
        run(script)
        self.assertAlmostEqual(script.change_time_s, 0.6, delta=DT + 1e-9)


if __name__ == "__main__":
    unittest.main()
