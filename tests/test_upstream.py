"""Tests execute the pinned native update, without launching an RL campaign."""

import copy
import itertools
import unittest

import numpy as np
import torch

from solvercell.config import Config
from solvercell.environment import InterventionEnv
from solvercell.upstream import NativeLSFlow


class NativeLSFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_learner(self):
        cfg = Config(method="lsflow_upstream", candidates=2, bootstrap_candidates=2,
                     sampling_steps=2, batch_size=2, device="cpu")
        env = InterventionEnv()
        return NativeLSFlow(cfg, env), env

    def make_batch(self, learner, env):
        rows = []
        obs = env.reset(seed=17)
        for _ in range(2):
            action, cost = learner.act(obs, stochastic=True)
            nxt, reward, done, _ = env.step(action)
            rows.append({"obs": obs, "action": action, "cost": cost, "reward": reward,
                         "next_obs": nxt, "done": float(done)})
            obs = nxt
        return {key: np.asarray([row[key] for row in rows], dtype=np.float32) for key in rows[0]}

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, atol=0., rtol=0.)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_native_updates_pretrain_checkpoint_and_rng_isolation(self):
        learner, env = self.make_learner()
        batch = self.make_batch(learner, env)
        self.assertEqual(batch["cost"].shape, (2, 40))
        self.assertEqual(batch["action"].shape, (2, 20))
        self.assertEqual(learner.native.cfg.reward_scale, 1.)
        pretrain = learner.pretrain(batch)
        self.assertEqual(pretrain["pretrain_updates"], 8)
        for _ in range(2):
            info = learner.update(batch)
            self.assertTrue(all(np.isfinite(value) for value in info.values()))
        self.assertEqual((learner.updates, learner.actor_updates, learner.native.step), (2, 2, 2))
        learner.native.policy_version = 7

        rng_before = torch.random.get_rng_state().clone()
        action1, cost1 = learner.act(batch["obs"][0], generator=torch.Generator().manual_seed(11))
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        action2, cost2 = learner.act(batch["obs"][0], generator=torch.Generator().manual_seed(11))
        np.testing.assert_array_equal(action1, action2)
        np.testing.assert_array_equal(cost1, cost2)
        self.assertEqual(action1.shape, (20,))
        self.assertEqual(cost1.shape, (40,))
        self.assertLessEqual(action1.sum(), 10)
        self.assertAlmostEqual(float(np.linalg.norm(cost1)), 1., places=5)

        snapshot = copy.deepcopy(learner.state_dict())
        restored, _ = self.make_learner()
        restored.load_state_dict(snapshot)
        self.assert_nested_equal(snapshot, restored.state_dict())
        # Resume the actual stochastic native update twice from the same state.
        expected_info = learner.update(batch)
        expected_state = copy.deepcopy(learner.state_dict())
        restored.load_state_dict(snapshot)
        actual_info = restored.update(batch)
        actual_state = restored.state_dict()
        self.assert_nested_equal(expected_info, actual_info)
        self.assert_nested_equal(expected_state, actual_state)

    def test_native_linear_decoder_matches_exhaustive_objective(self):
        learner, _ = self.make_learner()
        # Keep a tiny subset of primitive columns to exhaustively verify the
        # native objective map, including tied duplicate columns and zeros.
        weights = learner.weights_p[:, :6].clone()
        weights[:, 1] = weights[:, 0]
        learner.weights_p = weights
        learner.action_dim = 6
        learner.cfg = type("DecodeConfig", (), {"budget": 2})()
        rng = np.random.default_rng(29)
        costs = np.concatenate([rng.normal(size=(12, 40)), np.zeros((1, 40))]).astype(np.float32)
        actions = learner.decode_cost(costs).numpy()
        feasible = np.asarray([a for a in itertools.product((0., 1.), repeat=6) if sum(a) <= 2], dtype=np.float32)
        scores = costs @ weights.numpy()
        selected = np.sum(scores * actions, axis=1)
        optimum = np.max(scores @ feasible.T, axis=1)
        np.testing.assert_allclose(selected, optimum, atol=2e-6, rtol=1e-6)
        self.assertEqual(learner.decoder_calls, len(costs))
        np.testing.assert_array_equal(actions[-1], np.zeros(6))


if __name__ == "__main__":
    unittest.main()
