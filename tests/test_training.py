"""Small deterministic tests for the matched learner and evaluation boundary."""

from copy import deepcopy
from dataclasses import replace
import io
import random
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from solvercell.config import Config, debug_config
from solvercell.environment import InterventionEnv
from solvercell.geometry import decode, project_cell
from solvercell.learner import Learner, Replay
from solvercell.trainer import evaluate


def tiny_config(**kwargs):
    return replace(debug_config(Config()), hidden=16, horizon=3, **kwargs)


def fixed_batch(cfg):
    rng = np.random.default_rng(812)
    costs = rng.standard_normal((cfg.batch_size, cfg.n_actions)).astype(np.float32)
    return {
        "obs": rng.integers(0, 4, (cfg.batch_size, cfg.n_arms)).astype(np.float32),
        "next_obs": rng.integers(0, 4, (cfg.batch_size, cfg.n_arms)).astype(np.float32),
        "action": decode(torch.from_numpy(costs), cfg.budget).numpy(),
        "cost": costs,
        "reward": np.arange(1, cfg.batch_size + 1, dtype=np.float32),
        "done": (np.arange(cfg.batch_size) % 2).astype(np.float32),
    }


class ConstantQ(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, obs, action):
        return obs.new_full((len(obs),), self.value)


class TrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_nested_equal(self, left, right, *, ignore=()):
        if isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                if key not in ignore:
                    self.assert_nested_equal(left[key], right[key], ignore=ignore)
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b, ignore=ignore)
        elif torch.is_tensor(left):
            self.assertTrue(torch.equal(left, right), "Tensor values differ")
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            self.assertEqual(left, right)

    def test_matched_critic_initialization_across_methods(self):
        reference = None
        for method in ("cell_diffusion", "point_diffusion", "canonical_diffusion",
                       "cell_flow", "sphere_actioncritic"):
            cfg = tiny_config(method=method, diffusion_scale=0.0 if method == "cell_flow" else 1.0)
            learner = Learner(cfg, InterventionEnv())
            critics = {name: deepcopy(getattr(learner, name).state_dict())
                       for name in ("q1", "q2", "tq1", "tq2")}
            if reference is None:
                reference = critics
            else:
                self.assert_nested_equal(reference, critics)

    def test_evaluation_preserves_every_training_rng(self):
        cfg = tiny_config()
        env = InterventionEnv(horizon=cfg.horizon, seed=71)
        learner = Learner(cfg, env)
        env.reset()
        env.step(env.random_action())
        before = {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().clone(),
            "learner_generator": learner.generator.get_state().clone(),
            "env_rng": deepcopy(env.rng.bit_generator.state),
            "env_state": env.observation(), "env_t": env.t,
            "actor": deepcopy(learner.actor.state_dict()),
        }
        result = evaluate(env, learner, episodes=2, seed_start=100000)
        after = {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().clone(),
            "learner_generator": learner.generator.get_state().clone(),
            "env_rng": deepcopy(env.rng.bit_generator.state),
            "env_state": env.observation(), "env_t": env.t,
            "actor": deepcopy(learner.actor.state_dict()),
        }
        self.assert_nested_equal(before, after)
        repeated = evaluate(env, learner, episodes=2, seed_start=100000)
        self.assert_nested_equal(result, repeated, ignore={"eval_seconds"})

    def test_serialized_learner_roundtrip_continues_exactly(self):
        cfg = tiny_config()
        original = Learner(cfg, InterventionEnv())
        batch = fixed_batch(cfg)
        for _ in range(3):
            original.update(batch)
        stream = io.BytesIO()
        torch.save(original.state_dict(), stream)
        stream.seek(0)
        loaded = torch.load(stream, map_location="cpu", weights_only=False)
        restored = Learner(cfg, InterventionEnv())
        restored.load_state_dict(loaded)
        self.assert_nested_equal(original.state_dict(), restored.state_dict())
        for _ in range(3):
            left, right = original.update(batch), restored.update(batch)
            self.assert_nested_equal(left, right, ignore={"projection_seconds"})
        self.assert_nested_equal(original.state_dict(), restored.state_dict(), ignore={"projection_seconds"})
        obs = np.zeros(cfg.n_arms, dtype=np.float32)
        self.assert_nested_equal(original.act(obs), restored.act(obs))

    def test_two_identical_seed_tiny_loops(self):
        def loop():
            cfg = tiny_config(seed=5)
            env = InterventionEnv(horizon=cfg.horizon, instance_seed=cfg.instance_seed, seed=105)
            learner = Learner(cfg, env)
            replay = Replay(64, cfg.n_arms, cfg.n_actions)
            rng = np.random.default_rng(106)
            obs = env.reset()
            trace = []
            for step in range(8):
                if step < 4:
                    cost = rng.standard_normal(cfg.n_actions).astype(np.float32)
                    action = decode(torch.from_numpy(cost), cfg.budget).numpy()
                else:
                    action, cost = learner.act(obs)
                nxt, reward, done, info = env.step(action)
                replay.add(obs, action, cost, reward, nxt, float(done))
                trace.append((obs.copy(), action.copy(), nxt.copy(), reward, done))
                obs = env.reset() if done else nxt
                if step >= 4:
                    trace.append(learner.update(replay.sample(cfg.batch_size, rng)))
            return trace, learner.state_dict()

        self.assert_nested_equal(loop(), loop(), ignore={"projection_seconds"})

    def test_terminal_targets_do_not_bootstrap(self):
        cfg = tiny_config()
        batch = fixed_batch(cfg)
        batch["done"][:] = 1
        left = Learner(cfg, InterventionEnv())
        right = Learner(cfg, InterventionEnv())
        for learner, value in ((left, 1e6), (right, -1e6)):
            learner.tq1, learner.tq2 = ConstantQ(value), ConstantQ(value)
            # This step updates only critics so mock target modules need not
            # participate in actor optimization or expose trainable weights.
            learner.updates = 1
        left_info, right_info = left.update(batch), right.update(batch)
        self.assertAlmostEqual(left_info["target_mean"],
                               float((batch["reward"] * cfg.reward_scale).mean()), places=7)
        self.assert_nested_equal(left_info, right_info)
        self.assert_nested_equal(left.q1.state_dict(), right.q1.state_dict())
        self.assert_nested_equal(left.q2.state_dict(), right.q2.state_dict())

    def test_independent_cell_preserves_endpoint_marginal_but_changes_pairing(self):
        # All rows have the same state and action label. Replace the independent
        # Gaussian draw by a row permutation of the first draw: this gives an
        # exactly matched finite endpoint marginal and tests the learner's RNG
        # wiring without a noisy statistical independence test. Production uses
        # a fresh iid draw at this same call site.
        cfg = tiny_config(batch_size=8, endpoint_jitter=0.)
        batch = fixed_batch(cfg)
        batch["obs"][:] = 1.
        source = torch.randn((8, cfg.n_actions), generator=torch.Generator().manual_seed(730))
        permutation = torch.arange(8).roll(1)
        independent_source = source[permutation].clone()
        fixed_cost = torch.full((cfg.n_actions,), -1.)
        fixed_cost[:cfg.budget] = 1.
        action = decode(fixed_cost, cfg.budget).expand(8, -1)
        captured = {}
        original_randn = torch.randn

        for method in ("cell_diffusion", "cell_independent_diffusion"):
            learner = Learner(replace(cfg, method=method), InterventionEnv())
            draws = [source] if method == "cell_diffusion" else [source, independent_source]
            draw_index = 0
            original_loss = learner.actor.loss

            def controlled_randn(*args, **kwargs):
                nonlocal draw_index
                if draw_index < len(draws):
                    result = draws[draw_index].clone()
                    draw_index += 1
                    return result
                return original_randn(*args, **kwargs)

            def fixed_candidates(obs, count, **kwargs):
                costs = fixed_cost.expand(len(obs), count, -1).clone()
                return costs, decode(costs, cfg.budget), obs.repeat_interleave(count, 0)

            def capture_loss(obs, z0, targets, **kwargs):
                captured[method] = (z0.detach().clone(), targets.detach().clone())
                return original_loss(obs, z0, targets, **kwargs)

            with mock.patch.object(learner, "candidates", side_effect=fixed_candidates), \
                 mock.patch.object(learner.actor, "loss", side_effect=capture_loss), \
                 mock.patch("solvercell.learner.torch.randn", side_effect=controlled_randn):
                info = learner.update(batch)
            self.assertTrue(all(np.isfinite(value) for value in info.values()))
            self.assertEqual(draw_index, len(draws))

        coupled_z, coupled_endpoint = captured["cell_diffusion"]
        independent_z, independent_endpoint = captured["cell_independent_diffusion"]
        torch.testing.assert_close(coupled_z, source, atol=0., rtol=0.)
        torch.testing.assert_close(independent_z, source, atol=0., rtol=0.)
        torch.testing.assert_close(coupled_endpoint, project_cell(source, action, cfg.budget, cfg.margin))
        torch.testing.assert_close(independent_endpoint, coupled_endpoint[permutation])
        self.assertFalse(torch.equal(independent_endpoint, coupled_endpoint))
        for endpoint in (coupled_endpoint, independent_endpoint):
            torch.testing.assert_close(decode(endpoint, cfg.budget), action, atol=0., rtol=0.)
        coupled_distance = (coupled_endpoint - source).square().sum(-1)
        independent_distance = (independent_endpoint - source).square().sum(-1)
        self.assertTrue(torch.all(coupled_distance <= independent_distance + 1e-5))
        self.assertGreater((independent_distance - coupled_distance).mean().item(), 1e-3)


if __name__ == "__main__":
    unittest.main()
