"""Exact native parity tests for tensor-only candidate/smoothing hot paths."""

from copy import deepcopy
from dataclasses import replace
import unittest

import numpy as np
import torch

from solvercell.config import Config
from solvercell.environment import InterventionEnv
from solvercell.fast_native import TensorNativeLSFlow
from solvercell.trainer import evaluate
from solvercell.upstream import NativeLSFlow


class TensorNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def pair(self):
        cfg = Config(method="lsflow_upstream", batch_size=8, candidates=3,
                     bootstrap_candidates=5, sampling_steps=3, horizon=3,
                     device="cpu", threads=1)
        env = InterventionEnv(horizon=cfg.horizon)
        original = NativeLSFlow(cfg, env)
        optimized = TensorNativeLSFlow(replace(cfg, method="lsflow_tensor"), env)
        rng = np.random.default_rng(619)
        costs = rng.normal(size=(cfg.batch_size, cfg.n_arms)).astype(np.float32)
        costs /= np.linalg.norm(costs, axis=-1, keepdims=True)
        batch = {
            "obs": rng.integers(0, 4, (cfg.batch_size, cfg.n_arms)).astype(np.float32),
            "cost": costs,
            "action": original.decode_cost(costs).numpy(),
            "reward": rng.uniform(1, 20, cfg.batch_size).astype(np.float32),
            "next_obs": rng.integers(0, 4, (cfg.batch_size, cfg.n_arms)).astype(np.float32),
            "done": (np.arange(cfg.batch_size) % 3 == 0).astype(np.float32),
        }
        return original, optimized, env, batch

    def assert_tree_equal(self, left, right):
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right, atol=0.0, rtol=0.0)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_two_real_updates_match_all_state_bitwise(self):
        original, optimized, _, batch = self.pair()
        # Populate optimizer, Q-normalization and learned-temperature states
        # before cloning; comparing only fresh networks would miss resume bugs.
        original.pretrain(batch)
        original.update(batch)
        snapshot = deepcopy(original.state_dict())
        expected_info = [original.update(batch) for _ in range(2)]
        expected_state = deepcopy(original.state_dict())
        optimized.load_state_dict(snapshot)
        actual_info = [optimized.update(batch) for _ in range(2)]
        self.assert_tree_equal(expected_info, actual_info)
        self.assert_tree_equal(expected_state, optimized.state_dict())

    def test_candidate_and_trimmed_value_match_with_multiple_perturbations(self):
        original, optimized, _, batch = self.pair()
        snapshot = deepcopy(original.state_dict())
        obs = torch.as_tensor(batch["obs"])
        for count, perturbations in ((2, 1), (5, 3)):
            original.native.cfg.M_smooth = count
            original.native.cfg.J_smooth = perturbations
            optimized.native.cfg.M_smooth = count
            optimized.native.cfg.J_smooth = perturbations
            # Both candidate branches, including the K=1 actor-training case.
            for use_target, candidates in ((False, 3), (True, 1)):
                torch.manual_seed(91)
                with original._native_service():
                    expected = original.native._sample_candidates(obs, candidates, use_target=use_target)
                expected_rng = torch.get_rng_state().clone()
                torch.manual_seed(91)
                actual = optimized.native._sample_candidates(obs, candidates, use_target=use_target)
                self.assert_tree_equal(expected, actual)
                self.assert_tree_equal(expected_rng, torch.get_rng_state())
            torch.manual_seed(92)
            with original._native_service():
                expected = original.native._smoothed_value(obs)
            expected_rng = torch.get_rng_state().clone()
            torch.manual_seed(92)
            actual = optimized.native._smoothed_value(obs)
            self.assert_tree_equal(expected, actual)
            self.assert_tree_equal(expected_rng, torch.get_rng_state())

    def test_behavior_and_evaluation_match_with_same_checkpoint(self):
        original, optimized, env, batch = self.pair()
        original.pretrain(batch)
        original.update(batch)
        snapshot = deepcopy(original.state_dict())
        expected = [original.act(obs, stochastic=True) for obs in batch["obs"][:3]]
        expected_rng = torch.get_rng_state().clone()
        optimized.load_state_dict(snapshot)
        actual = [optimized.act(obs, stochastic=True) for obs in batch["obs"][:3]]
        self.assert_tree_equal(expected, actual)
        self.assert_tree_equal(expected_rng, torch.get_rng_state())
        original.load_state_dict(snapshot)
        optimized.load_state_dict(snapshot)
        expected_eval = evaluate(env, original, episodes=2, seed_start=310000)
        actual_eval = evaluate(env, optimized, episodes=2, seed_start=310000)
        expected_eval.pop("eval_seconds")
        actual_eval.pop("eval_seconds")
        self.assert_tree_equal(expected_eval, actual_eval)


if __name__ == "__main__":
    unittest.main()
