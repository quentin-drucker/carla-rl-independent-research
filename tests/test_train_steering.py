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


class RotateMapsTests(unittest.TestCase):
    def test_loads_the_next_town_every_block_of_episodes(self):
        import gymnasium
        from train_steering import RotateMaps

        class _Env(gymnasium.Env):  # records the town of every episode, like CarlaAEBEnv's _map_name
            observation_space = action_space = gymnasium.spaces.Discrete(1)

            def __init__(self):
                self._map_name, self._route_builder, self.loads, self.episodes = "Town05_Opt", "b05", 0, []

            def _cleanup_actors(self):
                pass

            def _connect(self):
                self.loads += 1

            def reset(self, **kwargs):
                self.episodes.append((self._map_name, self._route_builder))
                return 0, {}

        env = _Env()
        rotating = RotateMaps(env, {"Town05_Opt": "b05", "Town01_Opt": "b01"}, every=2)
        for _ in range(6):
            rotating.reset()
        self.assertEqual([m for m, _ in env.episodes], ["Town05_Opt"] * 2 + ["Town01_Opt"] * 2 + ["Town05_Opt"] * 2)
        self.assertEqual(env.episodes[2], ("Town01_Opt", "b01"))  # each town keeps its own spot builder
        self.assertEqual(env.loads, 2)  # no reload when the first town is already loaded

    def test_several_towns_on_the_command_line(self):
        from train_steering import _parse_args
        a = _parse_args(["--commit", "per_tick", "--circuit-map", "Town05_Opt", "Town01_Opt"])
        self.assertEqual(a.circuit_map, ["Town05_Opt", "Town01_Opt"])


if __name__ == "__main__":
    unittest.main()
