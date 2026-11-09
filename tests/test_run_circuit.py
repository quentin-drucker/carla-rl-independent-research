import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval_policy_encounters import _FixedAction  # noqa: E402
from run_circuit import POLICY_TICK_COLUMNS, critic_values, next_spawnable_hazard  # noqa: E402


class _Env:
    """next_encounter fails like CarlaAEBEnv for the hazards in `blocked`."""

    def __init__(self, blocked, error="[CarlaAEBEnv] Failed to spawn walker."):
        self.blocked, self.error, self.set_up = blocked, error, []

    def next_encounter(self, cfg):
        if cfg.run_id in self.blocked:
            raise RuntimeError(self.error)
        self.set_up.append(cfg.run_id)


PLAN = [(100.0 * i, SimpleNamespace(run_id=f"hz{i}")) for i in range(4)]


class NextSpawnableHazardTests(unittest.TestCase):
    def test_blocked_spawns_are_skipped_and_logged(self):
        env, skipped = _Env({"hz1", "hz2"}), []
        self.assertEqual(next_spawnable_hazard(env, PLAN, 1, skipped), 3)
        self.assertEqual(env.set_up, ["hz3"])
        self.assertEqual([s["hazard"] for s in skipped], [1, 2])

    def test_no_spawnable_hazard_left_ends_the_plan(self):
        skipped = []
        self.assertEqual(next_spawnable_hazard(_Env({"hz3"}), PLAN, 3, skipped), 4)
        self.assertEqual(len(skipped), 1)

    def test_other_errors_still_stop_the_run(self):
        with self.assertRaises(RuntimeError):
            next_spawnable_hazard(_Env({"hz1"}, error="CARLA timed out"), PLAN, 1, [])


class CriticValuesTests(unittest.TestCase):
    def test_scripted_controller_has_no_critic(self):
        self.assertEqual(critic_values(_FixedAction(1.0, 1.0), np.zeros(14, np.float32), None), ["", ""])

    def test_matches_the_critics_on_the_scaled_action(self):
        import gymnasium as gym
        import torch
        from stable_baselines3 import SAC

        class _Box(gym.Env):  # the circuit's spaces: 14 inputs, action [brake 0..1, u -1..1]
            observation_space = gym.spaces.Box(-1.0, 1.0, (14,), np.float32)
            action_space = gym.spaces.Box(np.array([0.0, -1.0], np.float32), np.array([1.0, 1.0], np.float32))

        model = SAC("MlpPolicy", _Box(), policy_kwargs={"net_arch": [8]}, device="cpu", seed=0)
        obs, action = np.full(14, 0.3, np.float32), np.array([1.0, -0.5], np.float32)
        q = critic_values(model, obs, action)
        with torch.no_grad():  # brake 1.0 is the top of its range: the critics see +1
            expected = model.critic(torch.as_tensor(obs[None]), torch.tensor([[1.0, -0.5]]))
        self.assertEqual(len(q), 2)
        np.testing.assert_allclose(q, [float(e) for e in expected], rtol=1e-6)
        self.assertEqual(len(POLICY_TICK_COLUMNS), 3 + 14 + 2 + 2)


class BrakeFirstTests(unittest.TestCase):
    def test_full_brake_while_braking_stops_else_the_wrapped_controller_and_its_attributes(self):
        from eval_policy_encounters import _BrakeFirst
        from rl_steering import STEERING_STATE_DIM, STATE_DIM, STEERING_STATE_NAMES

        class Inner:
            critic = "the inner critic"
            def predict(self, obs, deterministic=True):
                return np.array([0.3, -1.0], dtype=np.float32), None

        obs = np.zeros(STEERING_STATE_DIM, dtype=np.float32)
        obs[0] = 15.0 / 26.82
        line = STATE_DIM + STEERING_STATE_NAMES.index("distance_to_pedestrian_line_norm")
        bf = _BrakeFirst(Inner())
        obs[line] = 30.0 / 80.0
        self.assertEqual(bf.predict(obs)[0].tolist(), [1.0, 0.0])
        obs[line] = 12.0 / 80.0
        self.assertEqual(bf.predict(obs)[0].tolist(), [0.30000001192092896, -1.0])
        self.assertEqual(bf.critic, "the inner critic")

    def test_once_fired_after_onset_it_keeps_braking_until_disarmed(self):
        from eval_policy_encounters import _BrakeFirst
        from rl_steering import STEERING_STATE_DIM, STATE_DIM, STEERING_STATE_NAMES

        class Inner:
            def predict(self, obs, deterministic=True):
                return np.array([0.3, -1.0], dtype=np.float32), None

        obs = np.zeros(STEERING_STATE_DIM, dtype=np.float32)
        obs[0] = 15.0 / 26.82
        line = STATE_DIM + STEERING_STATE_NAMES.index("distance_to_pedestrian_line_norm")
        bf = _BrakeFirst(Inner())
        obs[line] = 30.0 / 80.0
        bf.predict(obs)  # before onset (not armed): no latch
        obs[line] = 12.0 / 80.0
        self.assertEqual(bf.predict(obs)[0].tolist()[1], -1.0)
        bf.armed = True
        obs[line] = 30.0 / 80.0
        bf.predict(obs)
        obs[line] = 12.0 / 80.0  # the margin ran out while the brakes took hold: keep braking
        self.assertEqual(bf.predict(obs)[0].tolist(), [1.0, 0.0])
        bf.armed = False  # encounter over
        self.assertEqual(bf.predict(obs)[0].tolist()[1], -1.0)


if __name__ == "__main__":
    unittest.main()
