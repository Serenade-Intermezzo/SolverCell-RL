"""Task semantics and independent ring/snapshot checks for preference data."""
from copy import deepcopy
import io
import unittest

import numpy as np
import torch

from solvercell.environment import InterventionEnv
from solvercell.geometry import decode
from solvercell.preference_data import PreferenceReplay, PreferenceTask


class PreferenceTaskTest(unittest.TestCase):
    def setUp(self):
        self.env = InterventionEnv(instance_seed=0)
        self.task = PreferenceTask(self.env)

    def test_time_encoding_and_no_mutation(self):
        state = np.arange(40, dtype=np.float32) % 4
        original = state.copy()
        obs = self.task.encode(state, 7)
        self.assertEqual((self.task.obs_dim, self.task.action_dim, self.task.cost_dim), (41, 20, 8))
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_allclose(obs[:-1], original/3)
        self.assertAlmostEqual(float(obs[-1]), 7/20)
        self.assertEqual(self.task.encode(state, 0)[-1], 0)
        self.assertEqual(self.task.encode(state, 20)[-1], 1)
        np.testing.assert_array_equal(state, original)
        for remaining in (-1, 21, True, 1.2):
            with self.assertRaises(ValueError):
                self.task.encode(state, remaining)

    def test_original_reward_units_and_fixed_cohort_values(self):
        for level, expected in ((0, [6., 4.]), (1, [6., 6.]), (2, [6., 8.])):
            actual = self.task.components(np.full(40, level))
            self.assertEqual(actual.dtype, np.float32)
            np.testing.assert_array_equal(actual, expected)
        self.env.reset(seed=101)
        action_rng = np.random.default_rng(99)
        for _ in range(self.env.horizon):
            state, reward, _, _ = self.env.step(self.env.random_action(action_rng))
            components = self.task.components(state)
            self.assertAlmostEqual(float(components @ np.array([.5, .5])), reward, delta=1e-5)

    def test_exact_one_step_oracle_and_source_conflict(self):
        state = np.full(40, 2)
        action = np.zeros(20)
        action[self.env.bad_actions] = 1
        np.testing.assert_allclose(self.task.expected_components(action, state),
                                   [40*.8089126046655851, 40*.1685], rtol=1e-7)
        rng = np.random.default_rng(10)
        saved_state, saved_t = self.env.state.copy(), self.env.t
        for _ in range(30):
            state = rng.integers(4, size=40)
            action = self.env.random_action(rng)
            expected = self.task.expected_components(action, state)
            self.assertAlmostEqual(float(expected.mean()), self.env.expected_reward(action, state), delta=3e-6)
        np.testing.assert_array_equal(self.env.state, saved_state)
        self.assertEqual(self.env.t, saved_t)

    def test_native_quotient_decoding_and_ties(self):
        generator = torch.Generator().manual_seed(9)
        costs = torch.randn(128, 8, generator=generator, dtype=torch.float64)
        original = costs.clone()
        actual = self.task.decode(costs)
        expected = decode((costs @ self.task.basis.double().T) @ torch.tensor(self.env.weights_p), 10)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(costs, original, rtol=0, atol=0)
        self.assertEqual(actual.dtype, costs.dtype)
        self.assertEqual(actual.device, costs.device)
        # Equal quotient coordinates give equal distinct-group scores because
        # all eight released column norms are identical: the first ten win.
        tied = self.task.decode(torch.ones(8))
        np.testing.assert_array_equal(tied.numpy(), [1.]*10+[0.]*10)
        np.testing.assert_array_equal(self.task.decode_numpy(np.zeros(8, np.float32)), np.zeros(20))
        identity = self.task.basis.double().T @ self.task.basis.double()
        torch.testing.assert_close(identity, torch.eye(8, dtype=torch.float64), rtol=1e-6, atol=1e-7)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_decode_matches_cpu(self):
        costs = torch.arange(16, dtype=torch.float32).reshape(2, 8)-5
        actual = self.task.decode(costs.cuda())
        self.assertEqual(actual.device.type, "cuda")
        torch.testing.assert_close(actual.cpu(), self.task.decode(costs))

    def test_malformed_task_inputs_and_unequal_partition(self):
        for state in (np.zeros(39), np.full(40, .5), np.full(40, np.nan), np.full(40, 4)):
            with self.assertRaises(ValueError):
                self.task.components(state)
        for costs in (torch.zeros(7), torch.full((8,), float("nan")), torch.zeros(8, dtype=torch.int64)):
            with self.assertRaises(ValueError):
                self.task.decode(costs)
        with self.assertRaises(ValueError):
            self.task.expected_components(np.ones(20), np.zeros(40))
        instance = self.env.instance_dict()
        instance["bad_arms"], instance["good_arms"] = list(range(19)), list(range(19, 40))
        with self.assertRaisesRegex(ValueError, "equal-size"):
            PreferenceTask(InterventionEnv.from_instance(instance))


class PreferenceReplayTest(unittest.TestCase):
    def make_replay(self):
        return PreferenceReplay(4, 3, 2, 2)

    def row(self, index):
        return dict(obs=np.array([index, index+.1, index+.2], np.float32),
                    action=np.array([index % 2, (index+1) % 2], np.float32),
                    cost=np.array([index, -index], np.float32),
                    reward_components=np.array([2*index, 3*index], np.float32),
                    next_obs=np.array([index+1, index+1.1, index+1.2], np.float32),
                    done=float(index % 2), behavior_preference=np.array([.25, .75], np.float32))

    def test_wrapped_physical_ring_and_resume_samples(self):
        replay = self.make_replay()
        for index in range(7):
            replay.add(**self.row(index))
        self.assertEqual((replay.size, replay.pos), (4, 3))
        np.testing.assert_array_equal(replay.data["obs"][:, 0], [4, 5, 6, 3])
        snapshot = replay.state_dict()
        stream = io.BytesIO()
        torch.save(snapshot, stream)
        stream.seek(0)
        restored = self.make_replay()
        restored.load_state_dict(torch.load(stream, weights_only=False))
        rng1, rng2 = np.random.default_rng(45), np.random.default_rng(45)
        for _ in range(3):
            first, second = replay.sample(7, rng1), restored.sample(7, rng2)
            for key in first:
                np.testing.assert_array_equal(first[key], second[key])
        replay.add(**self.row(7))
        restored.add(**self.row(7))
        for key in replay.data:
            np.testing.assert_array_equal(replay.data[key], restored.data[key])

    def test_partial_numpy_bridge_independence_and_no_rng_use(self):
        replay = self.make_replay()
        source = self.row(1)
        replay.add(**source)
        source["obs"][:] = 100
        self.assertEqual(replay.data["obs"][0, 0], 1)
        np_before, torch_before = np.random.get_state(), torch.get_rng_state().clone()
        snapshot = replay.state_dict()
        self.assertEqual(snapshot["data"]["obs"].shape, (1, 3))
        snapshot["data"] = {key: value.numpy() for key, value in snapshot["data"].items()}
        restored = self.make_replay()
        for index in range(4):
            restored.add(**self.row(index+10))
        restored.load_state_dict(snapshot)
        self.assertEqual((restored.size, restored.pos), (1, 1))
        for value in restored.data.values():
            self.assertFalse(np.any(value[1:]))
        snapshot["data"]["obs"][:] = 55
        self.assertEqual(restored.data["obs"][0, 0], 1)
        np.testing.assert_array_equal(np.random.get_state()[1], np_before[1])
        self.assertEqual(np.random.get_state()[2:], np_before[2:])
        torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)

    def test_empty_and_full_snapshots(self):
        replay = self.make_replay()
        for compact in (True, False):
            restored = self.make_replay()
            restored.load_state_dict(replay.state_dict(compact=compact))
            self.assertEqual((restored.size, restored.pos), (0, 0))
        with self.assertRaises(ValueError):
            replay.sample(1, np.random.default_rng(0))

    def test_bad_add_rejected_atomically(self):
        replay = self.make_replay()
        replay.add(**self.row(0))
        original = deepcopy(replay.data)
        invalid = {"obs": np.zeros(2), "action": np.array([.5, 0]), "cost": np.array([np.inf, 0]),
                   "reward_components": np.ones(3), "next_obs": np.full(3, np.nan),
                   "done": .2, "behavior_preference": np.array([.2, .2])}
        for key, value in invalid.items():
            row = self.row(1)
            row[key] = value
            with self.assertRaises(ValueError):
                replay.add(**row)
            self.assertEqual((replay.size, replay.pos), (1, 1))
            for field in replay.data:
                np.testing.assert_array_equal(replay.data[field], original[field])

    def test_bad_snapshot_rejected_atomically(self):
        replay = self.make_replay()
        replay.add(**self.row(0))
        snapshot = replay.state_dict()
        variants = []
        for key, value in (("format_version", True), ("capacity", 5), ("size", 5), ("pos", 3), ("obs_dim", 4)):
            variant = deepcopy(snapshot)
            variant[key] = value
            variants.append(variant)
        for key, value in (("obs", torch.zeros(1, 3, dtype=torch.float64)),
                           ("reward_components", torch.full((1, 2), float("nan"))),
                           ("done", torch.tensor([.3])),
                           ("behavior_preference", torch.tensor([[1., 1.]]))):
            variant = deepcopy(snapshot)
            variant["data"][key] = value
            variants.append(variant)
        for variant in variants:
            with self.assertRaises(ValueError):
                replay.load_state_dict(variant)
            self.assertEqual((replay.size, replay.pos), (1, 1))
            np.testing.assert_array_equal(replay.data["reward_components"][0], [0, 0])


if __name__ == "__main__":
    unittest.main()
