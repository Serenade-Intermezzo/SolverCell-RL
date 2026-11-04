"""Numerical mechanism checks; these are not RL performance evidence."""

import math
import unittest

import torch

from solvercell.policies import Policy, _sphere_path


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(123)
        self.obs = torch.randn(16, 40)
        self.z0 = torch.randn(16, 20)
        self.target = torch.randn(16, 20)

    def test_finite_gradients_and_reproducible_sampling(self):
        for kind in ("diffusion", "spherical_flow"):
            with self.subTest(kind=kind):
                policy = Policy(40, 20, hidden=32, steps=5, kind=kind)
                loss, diagnostics = policy.loss(self.obs, self.z0, self.target, torch.Generator().manual_seed(5))
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(math.isfinite(value) for value in diagnostics.values()))
                loss.backward()
                for param in policy.parameters():
                    self.assertIsNotNone(param.grad)
                    self.assertTrue(torch.isfinite(param.grad).all())
                a = policy.sample(self.obs, torch.Generator().manual_seed(9))
                b = policy.sample(self.obs, torch.Generator().manual_seed(9))
                self.assertEqual(a.shape, (16, 20))
                self.assertTrue(torch.isfinite(a).all())
                self.assertTrue(torch.equal(a, b))
                if kind == "spherical_flow":
                    torch.testing.assert_close(a.norm(dim=-1), torch.ones(16), atol=1e-6, rtol=1e-6)
                self.assertEqual(policy.sampler_nfe, 10 if kind == "spherical_flow" else 5)

    def test_geodesic_endpoints_tangency_and_antipodes(self):
        start = torch.tensor([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
        target = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]])
        for t_value in (0., 0.25, 0.5, 1.):
            t = torch.full((3, 1), t_value)
            point, velocity = _sphere_path(start, target, t)
            self.assertTrue(torch.isfinite(velocity).all())
            torch.testing.assert_close(point.norm(dim=-1), torch.ones(3))
            torch.testing.assert_close((point * velocity).sum(-1), torch.zeros(3), atol=1e-6, rtol=0.)
            if t_value == 0.:
                torch.testing.assert_close(point, start)
            if t_value == 1.:
                torch.testing.assert_close(point, target, atol=2e-6, rtol=0.)
        torch.testing.assert_close(velocity[0], torch.zeros(3))

    def test_sde_noise_ablation_preserves_training(self):
        stochastic = Policy(40, 20, hidden=32, steps=5, diffusion_scale=1.)
        deterministic = Policy(40, 20, hidden=32, steps=5, diffusion_scale=0.)
        deterministic.load_state_dict(stochastic.state_dict())
        a, _ = stochastic.loss(self.obs, self.z0, self.target, torch.Generator().manual_seed(17))
        b, _ = deterministic.loss(self.obs, self.z0, self.target, torch.Generator().manual_seed(17))
        torch.testing.assert_close(a, b, atol=0., rtol=0.)
        # With zero heads only the injected Brownian increments distinguish the
        # samplers. This catches accidentally implementing the diffusion as ODE.
        for policy in (stochastic, deterministic):
            for parameter in policy.parameters():
                parameter.data.zero_()
        a = stochastic.sample(self.obs, torch.Generator().manual_seed(19))
        b = deterministic.sample(self.obs, torch.Generator().manual_seed(19))
        reference = torch.randn((16, 20), generator=torch.Generator().manual_seed(19))
        torch.testing.assert_close(b, reference, atol=0., rtol=0.)
        self.assertGreater((a - b).square().mean().item(), 1e-3)

    def test_tiny_fixed_interpolants_can_be_learned(self):
        # Repeating the generator seed fixes sampled times/noise. This only
        # tests the optimization path on a tiny supervised batch, not general-
        # ization, endpoint-distribution recovery, or policy-improvement gains.
        for kind in ("diffusion", "spherical_flow"):
            with self.subTest(kind=kind):
                policy = Policy(40, 20, hidden=64, steps=5, kind=kind)
                optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
                losses = []
                for _ in range(160):
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = policy.loss(self.obs, self.z0, self.target, torch.Generator().manual_seed(31))
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.detach().item())
                self.assertLess(losses[-1], 0.05 * losses[0])


if __name__ == "__main__":
    unittest.main()
