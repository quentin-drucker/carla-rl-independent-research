import random
import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_steering import MPH_RANGE, TTC_RANGE_S, sample_scenario  # noqa: E402


class SampleScenarioTests(unittest.TestCase):
    def test_defaults_draw_exactly_what_week5_drew(self):
        a, b = random.Random(7), random.Random(7)
        cfg = sample_scenario(a)
        mph, ttc = b.uniform(*MPH_RANGE), b.uniform(*TTC_RANGE_S)
        self.assertEqual((cfg.target_mph, cfg.trigger_ttc_s, cfg.walker_cross, cfg.walker_script),
                         (mph, ttc, "stationary", None))
        self.assertEqual(cfg.run_id, f"train_{mph:.1f}mph_ttc{ttc:.2f}")

    def test_dynamic_family_samples_both_walkers_in_range(self):
        rng = random.Random(1)
        cfgs = [sample_scenario(rng, walkers=["right_cross", "right_turn_back"], ttc_range=(0.5, 1.6))
                for _ in range(200)]
        self.assertEqual({c.run_id.split("_", 3)[3] for c in cfgs}, {"right_cross", "right_turn_back"})
        self.assertTrue(all(0.5 <= c.trigger_ttc_s <= 1.6 and c.walker_script is not None for c in cfgs))


class ArgsTests(unittest.TestCase):
    def test_right_only_by_default_and_both_sides_on_request(self):
        from train_steering import _parse_args
        self.assertEqual(_parse_args(["--commit", "per_tick"]).u_min, 0.0)
        self.assertEqual(_parse_args(["--commit", "per_tick", "--u-min", "-1"]).u_min, -1.0)
        import eval_policy_encounters as e
        a = e._parse_args(["--suite", "test26", "--steer-commit", "per_tick", "--fixed-u", "-1", "-0.5", "--u-min", "-1"])
        self.assertEqual((a.u_min, a.fixed_u), (-1.0, [-1.0, -0.5]))


if __name__ == "__main__":
    unittest.main()
