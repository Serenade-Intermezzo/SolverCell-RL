"""Causal preference/continuation contracts and exact learner recovery."""
from copy import deepcopy
from dataclasses import replace
import io
import math
import unittest

import numpy as np
import torch
from torch import nn

from solvercell.preference_config import PreferenceConfig
from solvercell.preference_learner import PreferenceLearner


class TinyTask:
    obs_dim, cost_dim, action_dim = 3, 2, 2

    def decode(self, costs):
        return torch.nn.functional.one_hot(costs.argmax(-1), 2).to(costs.dtype)


def config(**kwargs):
    values = dict(hidden=16, batch_size=4, sampling_steps=4, candidates=2,
                  bootstrap_candidates=1, guide_interval=2, guide_batch_size=2,
                  guide_suffixes=2, guide_particles=3, seed=7)
    values.update(kwargs)
    return replace(PreferenceConfig(), **values)


def batch(size=4):
    rng = np.random.default_rng(812)
    actions = np.eye(2, dtype=np.float32)[np.arange(size) % 2]
    return {"obs": rng.uniform(0, 1, (size, 3)).astype(np.float32),
            "next_obs": rng.uniform(0, 1, (size, 3)).astype(np.float32),
            "action": actions, "reward_components": rng.uniform(0, 40, (size, 2)).astype(np.float32),
            "done": (np.arange(size) % 2).astype(np.float32),
            "behavior_preference": np.tile(np.array([1., 0.], np.float32), (size, 1))}


class ConstantVectorQ(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.calls = []

    def forward(self, obs, action, prefs):
        self.calls.append((obs.clone(), action.clone(), prefs.clone()))
        return obs.new_tensor(self.value).expand(len(obs), -1)


class ActionVectorQ(nn.Module):
    def forward(self, obs, action, prefs):
        return action * action.new_tensor([4., 2.])


class RecordingActor(nn.Module):
    def __init__(self, steps=4):
        super().__init__()
        self.steps = steps
        self.sample_calls = []
        self.suffix_calls = []

    def sample(self, obs, prefs, generator=None):
        self.sample_calls.append((obs.clone(), prefs.clone()))
        return obs.new_tensor([1., 0.]).expand(len(obs), -1)

    def sample_path(self, obs, prefs, generator=None):
        # Each path state has an independently recoverable observation/index
        # identifier. The learner must restart suffixes at exactly this prefix.
        return torch.stack([torch.stack((obs[:, 0] + index, obs[:, 1] - index), -1)
                            for index in range(self.steps + 1)], dim=1)

    def suffix(self, obs, prefs, prefix, index, generator=None):
        self.suffix_calls.append((obs.clone(), prefs.clone(), prefix.clone(), index))
        # Two legitimate terminal branches of this mocked continuation kernel;
        # their nonlinear log-mean-exp differs from mean utility.
        return obs.new_tensor([[2., -1.], [-1., 2.]]).repeat(len(obs) // 2, 1)


class PreferenceLearnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def equal(self, left, right):
        if isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.equal(a, b)
        elif torch.is_tensor(left):
            self.assertTrue(torch.equal(left, right), "tensor mismatch")
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            self.assertEqual(left, right)

    def test_matched_critics_preference_streams_and_initialization_rng(self):
        torch.manual_seed(412)
        rng_before = torch.get_rng_state().clone()
        diffusion = PreferenceLearner(config(), TinyTask())
        sphere = PreferenceLearner(config(method="preference_sphere"), TinyTask())
        self.equal(rng_before, torch.get_rng_state())
        for name in ("q1", "q2", "tq1", "tq2"):
            self.equal(getattr(diffusion, name).state_dict(), getattr(sphere, name).state_dict())
        for name in ("generator", "preference_generator", "guide_generator", "behavior_generator"):
            self.equal(getattr(diffusion, name).get_state(), getattr(sphere, name).get_state())
        # Method-specific sampler draw counts must not desynchronize utility
        # relabeling, including after diffusion-only guide fitting.
        for _ in range(4):
            diffusion.update(batch())
            sphere.update(batch())
            self.equal(diffusion.sample_preferences(20), sphere.sample_preferences(20))

    def test_same_preference_bootstrap_uses_whole_lower_scalarized_vector(self):
        learner = PreferenceLearner(config(gamma=.5, reward_scale=.1), TinyTask())
        actor = RecordingActor()
        learner.target_actor = actor
        learner.tq1, learner.tq2 = ConstantVectorQ([10., 0.]), ConstantVectorQ([0., 10.])
        data = batch(3)
        data["done"] = np.array([0., 0., 1.], np.float32)
        data["reward_components"][:] = [[2., 4.], [6., 8.], [10., 12.]]
        prefs = torch.tensor([[.9, .1], [.1, .9], [.5, .5]])
        targets = learner.critic_targets(data, prefs)
        expected = torch.tensor([[.2, 5.4], [5.6, .8], [1., 1.2]])
        torch.testing.assert_close(targets, expected, atol=1e-6, rtol=0)
        self.assertEqual(len(actor.sample_calls), 1)
        self.equal(actor.sample_calls[0][1], prefs[:2])
        self.equal(learner.tq1.calls[0][2], prefs[:2])
        self.equal(learner.tq2.calls[0][2], prefs[:2])
        self.equal(actor.sample_calls[0][0], torch.from_numpy(data["next_obs"][:2]))
        # Componentwise min would incorrectly give zero continuation to both.
        self.assertGreater(float(targets[:2].sum()), float(torch.from_numpy(data["reward_components"][:2]).sum() * .1))

    def test_terminal_batch_never_queries_next_actor_or_critics(self):
        learner = PreferenceLearner(config(), TinyTask())
        learner.target_actor = RecordingActor()
        learner.tq1, learner.tq2 = ConstantVectorQ([1e20, 1e20]), ConstantVectorQ([-1e20, -1e20])
        data = batch()
        data["done"][:] = 1
        before = learner.generator.get_state().clone()
        targets = learner.critic_targets(data, [.25, .75])
        self.equal(targets, torch.from_numpy(data["reward_components"]) * learner.cfg.reward_scale)
        self.equal(before, learner.generator.get_state())
        self.assertFalse(learner.target_actor.sample_calls)
        self.assertFalse(learner.tq1.calls)
        self.assertFalse(learner.tq2.calls)

    def test_behavior_preference_does_not_leak_into_relabelled_continuation(self):
        left = PreferenceLearner(config(), TinyTask())
        right = PreferenceLearner(config(), TinyTask())
        original, relabeled = batch(), batch()
        relabeled["behavior_preference"][:] = [.125, .875]
        for _ in range(3):
            self.equal(left.update(original), right.update(relabeled))
        self.equal(left.state_dict(), right.state_dict())

    def test_guide_targets_restart_actual_prefix_and_use_joint_logmeanexp(self):
        learner = PreferenceLearner(config(guide_temperature=.5), TinyTask())
        actor = RecordingActor()
        learner.target_actor = actor
        learner.tq1, learner.tq2 = ActionVectorQ(), ActionVectorQ()
        obs = torch.tensor([[.1, .2, 1.], [.3, .4, .5], [.8, .9, .25]])
        prefs = torch.tensor([[1., 0.], [.5, .5], [0., 1.]])
        global_before = torch.get_rng_state().clone()
        training_before = learner.generator.get_state().clone()
        prefix, indices, labels, diagnostics = learner.guide_training_targets(obs, prefs)
        expected_prefix = torch.stack((obs[:, 0] + indices, obs[:, 1] - indices), -1)
        self.equal(prefix, expected_prefix)
        expected = torch.logsumexp(torch.stack((4 * prefs[:, 0], 2 * prefs[:, 1]), -1) / .5, dim=1) - math.log(2)
        torch.testing.assert_close(labels, expected)
        self.assertFalse(torch.allclose(labels, (2 * prefs[:, 0] + prefs[:, 1]) / .5))
        self.assertFalse(labels.requires_grad)
        for repeated_obs, repeated_prefs, repeated_prefix, index in actor.suffix_calls:
            selected = torch.nonzero(indices == index, as_tuple=False).flatten()
            self.equal(repeated_obs, obs[selected].repeat_interleave(2, 0))
            self.equal(repeated_prefs, prefs[selected].repeat_interleave(2, 0))
            self.equal(repeated_prefix, prefix[selected].repeat_interleave(2, 0))
        self.assertEqual(diagnostics["guide_baseline_mean"], 0.)
        self.assertEqual(diagnostics["guide_label_suffix_nfe"], int(((4 - indices) * 2).sum()))
        self.equal(training_before, learner.generator.get_state())
        self.equal(global_before, torch.get_rng_state())

    def test_guide_fitting_does_not_change_direct_base_learning(self):
        enabled = PreferenceLearner(config(guide_interval=1), TinyTask())
        delayed = PreferenceLearner(config(guide_interval=1000), TinyTask())
        for _ in range(3):
            enabled.update(batch())
            delayed.update(batch())
        for name in ("actor", "target_actor", "q1", "q2", "tq1", "tq2", "actor_optim", "q_optim"):
            self.equal(getattr(enabled, name).state_dict(), getattr(delayed, name).state_dict())
        self.equal(enabled.generator.get_state(), delayed.generator.get_state())
        self.equal(enabled.preference_generator.get_state(), delayed.preference_generator.get_state())
        self.assertEqual(enabled.guide_updates, 3)
        self.assertEqual(delayed.guide_updates, 0)

    def test_external_evaluation_rng_and_actual_forward_row_counts(self):
        learner = PreferenceLearner(config(), TinyTask())
        before = {name: getattr(learner, name).get_state().clone() for name in
                  ("generator", "preference_generator", "guide_generator", "behavior_generator")}
        global_before = torch.get_rng_state().clone()
        obs, prefs = np.array([.2, .3, 1.], np.float32), [.25, .75]
        for mode in ("direct", "terminal_weighting", "smc"):
            actor_rows, q_rows, guide_rows = [], [], []
            handles = [learner.target_actor.register_forward_hook(lambda module, args, out: actor_rows.append(len(args[0]))),
                       learner.tq1.register_forward_hook(lambda module, args, out: q_rows.append(len(args[0]))),
                       learner.tq2.register_forward_hook(lambda module, args, out: q_rows.append(len(args[0]))),
                       learner.guide.register_forward_hook(lambda module, args, out: guide_rows.append(len(args[0])))]
            try:
                action, cost, stats = learner.act(obs, prefs, mode, torch.Generator().manual_seed(313))
            finally:
                for handle in handles:
                    handle.remove()
            self.assertEqual(stats["nfe"], sum(actor_rows))
            self.assertEqual(stats["q_calls"], sum(q_rows))
            self.assertEqual(stats["guide_calls"], sum(guide_rows))
            self.assertEqual(stats["decoder_calls"], 1 if mode == "direct" else 4)
            self.assertAlmostEqual(float(np.linalg.norm(cost)), 1., places=6)
            self.equal(action, TinyTask().decode(torch.from_numpy(cost[None])).numpy()[0])
        for name, state in before.items():
            self.equal(state, getattr(learner, name).get_state())
        self.equal(global_before, torch.get_rng_state())
        sphere = PreferenceLearner(config(method="preference_sphere"), TinyTask())
        _, _, stats = sphere.act(obs, prefs, "direct", torch.Generator().manual_seed(11))
        self.assertEqual(stats["nfe"], 4)
        with self.assertRaisesRegex(ValueError, "diffusion"):
            sphere.act(obs, prefs, "smc")

    def test_serialized_exact_checkpoint_continuation_and_inference(self):
        for method in ("preference_diffusion", "preference_sphere"):
            with self.subTest(method=method):
                original = PreferenceLearner(config(method=method), TinyTask())
                for _ in range(3):
                    original.update(batch())
                stream = io.BytesIO()
                torch.save(original.state_dict(), stream)
                stream.seek(0)
                restored = PreferenceLearner(config(method=method), TinyTask())
                restored.load_state_dict(torch.load(stream, map_location="cpu", weights_only=False))
                self.equal(original.state_dict(), restored.state_dict())
                for _ in range(3):
                    self.equal(original.update(batch()), restored.update(batch()))
                modes = ("direct", "terminal_weighting", "smc") if method == "preference_diffusion" else ("direct", "terminal_weighting")
                for mode in modes:
                    self.equal(original.act([.2, .3, 1.], [.5, .5], mode),
                               restored.act([.2, .3, 1.], [.5, .5], mode))
                self.equal(original.state_dict(), restored.state_dict())


if __name__ == "__main__":
    unittest.main()
