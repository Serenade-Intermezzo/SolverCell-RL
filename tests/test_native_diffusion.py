"""Integration checks for native critics and lifted diffusion/spherical actors."""

from copy import deepcopy
import io
import math
import random
import unittest

import numpy as np
import torch

from solvercell.config import Config
from solvercell.environment import InterventionEnv
from solvercell.native_diffusion import NativeDiffusion
from solvercell.policies import _unit
from solvercell.trainer import evaluate
from solvercell.upstream import NativeLSFlow


BRIDGE_METHODS = (
    "native_full_point_diffusion", "native_quotient_sphere",
    "native_quotient_point_diffusion", "native_quotient_cell_diffusion",
    "native_quotient_independent_diffusion", "native_quotient_cell_flow",
)


class NativeDiffusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, method):
        return Config(method=method, horizon=3, candidates=2, bootstrap_candidates=2,
                      sampling_steps=2, batch_size=2, device="cpu",
                      diffusion_scale=0. if method.endswith("_flow") else 1.).validate()

    def make(self, method):
        env = InterventionEnv(horizon=3, seed=83)
        return NativeDiffusion(self.config(method), env), env

    def assert_nested_equal(self, left, right):
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right, atol=0., rtol=0.)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                if not isinstance(key, str) or "seconds" not in key:
                    self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def batch(self, learner, env):
        rows = []
        obs = env.reset(seed=84)
        for _ in range(2):
            action, cost = learner.act(obs, stochastic=True)
            next_obs, reward, done, _ = env.step(action)
            rows.append({"obs": obs, "action": action, "cost": cost,
                         "next_obs": next_obs, "reward": reward, "done": float(done)})
            obs = next_obs
        return {key: np.asarray([row[key] for row in rows], dtype=np.float32) for key in rows[0]}

    def test_every_actor_replacement_preserves_original_critic_initialization(self):
        reference = NativeLSFlow(self.config("lsflow_upstream"), InterventionEnv(horizon=3))
        critics = {name: deepcopy(getattr(reference.native, name).state_dict())
                   for name in ("q1", "q2", "tq1", "tq2")}
        torch_rng = torch.get_rng_state().clone()
        numpy_rng = deepcopy(np.random.get_state())
        for method in BRIDGE_METHODS:
            with self.subTest(method=method):
                learner, _ = self.make(method)
                for name, expected in critics.items():
                    self.assert_nested_equal(expected, getattr(learner.native, name).state_dict())
                self.assert_nested_equal(torch_rng, torch.get_rng_state())
                self.assert_nested_equal(numpy_rng, np.random.get_state())
                self.assertEqual(learner.native.cfg.reward_scale, 1.)

    def test_quotient_samples_and_projected_endpoints_stay_in_the_native_cost_domain(self):
        for method in BRIDGE_METHODS:
            with self.subTest(method=method):
                learner, _ = self.make(method)
                obs = torch.ones(5, 40)
                costs = learner.service.model.sample(obs, K=3, steps=2)
                self.assertEqual(costs.shape, (5, 3, 40))
                torch.testing.assert_close(costs.norm(dim=-1), torch.ones(5, 3), atol=1e-6, rtol=0.)
                if "quotient" in method:
                    self.assertEqual(learner.basis.shape, (40, 8))
                    recovered = (costs @ learner.basis) @ learner.basis.T
                    torch.testing.assert_close(costs, recovered, atol=2e-7, rtol=1e-6)
                if not learner.is_spherical:
                    old = _unit(costs[:, 0] @ learner.basis)
                    source = _unit(torch.randn_like(old))
                    endpoint = learner._endpoint(obs, source, old)
                    lifted = endpoint @ learner.basis.T
                    torch.testing.assert_close(lifted.norm(dim=-1), torch.ones(5), atol=1e-6, rtol=0.)
                    self.assertTrue(torch.equal(learner.projector.decode(lifted),
                                                learner.projector.decode(old @ learner.basis.T)))

    def test_real_updates_train_inner_actor_and_checkpoint_resumes_exactly(self):
        for method in BRIDGE_METHODS:
            with self.subTest(method=method):
                learner, env = self.make(method)
                batch = self.batch(learner, env)
                initial = [p.detach().clone() for p in learner.service.model.parameters()]
                for _ in range(2):
                    info = learner.update(batch)
                    self.assertTrue(all(math.isfinite(value) for value in info.values()))
                self.assertTrue(any(not torch.equal(before, after) for before, after in
                                    zip(initial, learner.service.model.parameters())))
                self.assertEqual((learner.updates, learner.actor_updates, learner.native.step), (2, 2, 2))
                stream = io.BytesIO()
                torch.save(learner.state_dict(), stream)
                stream.seek(0)
                snapshot = torch.load(stream, map_location="cpu", weights_only=False)
                expected_info = learner.update(batch)
                expected_state = deepcopy(learner.state_dict())
                restored, _ = self.make(method)
                restored.load_state_dict(snapshot)
                self.assert_nested_equal(snapshot, restored.state_dict())
                actual_info = restored.update(batch)
                self.assert_nested_equal(expected_info, actual_info)
                self.assert_nested_equal(expected_state, restored.state_dict())

    def test_evaluation_does_not_perturb_training_random_streams(self):
        for method in BRIDGE_METHODS:
            with self.subTest(method=method):
                learner, env = self.make(method)
                env.reset(seed=86)
                before = {"torch": torch.get_rng_state().clone(), "numpy": deepcopy(np.random.get_state()),
                          "python": random.getstate(), "env_rng": deepcopy(env.rng.bit_generator.state),
                          "env_state": env.observation(), "env_t": env.t}
                first = evaluate(env, learner, episodes=2, seed_start=100000)
                after = {"torch": torch.get_rng_state().clone(), "numpy": deepcopy(np.random.get_state()),
                         "python": random.getstate(), "env_rng": deepcopy(env.rng.bit_generator.state),
                         "env_state": env.observation(), "env_t": env.t}
                self.assert_nested_equal(before, after)
                self.assert_nested_equal(first, evaluate(env, learner, episodes=2, seed_start=100000))


if __name__ == "__main__":
    unittest.main()
