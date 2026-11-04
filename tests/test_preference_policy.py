"""Finite-policy and particle-target checks, not RL performance evidence."""

import copy
import math
import unittest

import torch
from torch import nn

from solvercell.preference_policy import (
    JointPreferencePotential, PreferenceActor, preference_smc, terminal_weight_sample,
)


class _Potential(nn.Module):
    def __init__(self, gauge=False):
        super().__init__()
        self.gauge = gauge
        self.rows = 0

    def forward(self, obs, prefs, x, index):
        self.rows += len(x)
        value = 4.0 * x[:, 0] - 0.6 * x[:, 1] + prefs[:, 0]
        # A time-only constant cannot change the final tilted distribution.
        return value.double() + (2500.0 * (index + 1) if self.gauge else 0.0)


class _IdentityChain:
    """A four-point prior and identity kernels give an exact SMC oracle.

    This isolates generic particle bookkeeping from neural/Gaussian kernels,
    which are checked separately against torch.distributions below.
    """
    obs_dim, act_dim, steps, sampler_nfe, kind = 1, 2, 2, 2, "diffusion"
    points = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])

    def _initial(self, obs, generator):
        return self.points.to(obs).repeat(len(obs) // 4, 1)

    def transition(self, obs, prefs, x, index, generator=None):
        return x.clone()

    def sample_raw(self, obs, prefs, generator=None):
        return self._initial(obs, generator)


class PreferencePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(70)
        self.obs = torch.randn(12, 5)
        self.prefs = torch.softmax(torch.randn(12, 2), -1)
        self.targets = torch.nn.functional.normalize(torch.randn(12, 4), dim=-1)

    def test_raw_gaussian_kernel_path_and_checkpoint_rng(self):
        actor = PreferenceActor(5, 4, hidden=24, steps=4)
        x = torch.randn(12, 4)
        for index in (0, actor.steps - 1):
            with self.subTest(index=index):
                mean, std = actor.transition_parameters(self.obs, self.prefs, x, index)
                level = actor.steps - 1 - index
                self.assertAlmostEqual(float(std.square()), float(actor.betas[level]), places=6)
                self.assertGreater(float(std), 0.)
                noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(711))
                actual = actor.transition(self.obs, self.prefs, x, index, torch.Generator().manual_seed(711))
                torch.testing.assert_close(actual, mean + std * noise, atol=0., rtol=0.)
                expected_logp = torch.distributions.Normal(mean, std).log_prob(actual).sum(-1)
                logp = actor.transition_log_prob(self.obs, self.prefs, x, actual, index)
                torch.testing.assert_close(logp, expected_logp)
                self.assertTrue(torch.isfinite(logp).all())

        generator = torch.Generator().manual_seed(85)
        global_rng = torch.get_rng_state().clone()
        initial = torch.randn((12, 4), generator=generator)
        at_one = actor.transition(self.obs, self.prefs, initial, 0, generator)
        saved_rng = generator.get_state().clone()
        saved_actor = copy.deepcopy(actor.state_dict())
        expected = actor.suffix(self.obs, self.prefs, at_one, 1, generator)
        restored = PreferenceActor(5, 4, hidden=24, steps=4)
        restored.load_state_dict(saved_actor)
        replay_rng = torch.Generator().set_state(saved_rng)
        actual = restored.suffix(self.obs, self.prefs, at_one, 1, replay_rng)
        torch.testing.assert_close(actual, expected, atol=0., rtol=0.)
        torch.testing.assert_close(restored.suffix(self.obs, self.prefs, actual, 4, replay_rng), actual, atol=0., rtol=0.)

        # Construction above consumes global RNG; actual calls with an injected
        # generator must not. Capture anew after constructing the second model.
        global_rng = torch.get_rng_state().clone()
        path = actor.sample_path(self.obs, self.prefs, torch.Generator().manual_seed(85))
        sample = actor.sample(self.obs, self.prefs, torch.Generator().manual_seed(85))
        self.assertTrue(torch.equal(torch.get_rng_state(), global_rng))
        self.assertEqual(path.shape, (12, 5, 4))
        torch.testing.assert_close(path[:, 0], initial, atol=0., rtol=0.)
        torch.testing.assert_close(path[:, 1], at_one, atol=0., rtol=0.)
        torch.testing.assert_close(path[:, -1], expected, atol=0., rtol=0.)
        self.assertFalse(torch.allclose(path[:, -1].norm(dim=-1), torch.ones(12)))
        torch.testing.assert_close(sample, torch.nn.functional.normalize(path[:, -1], dim=-1))
        torch.testing.assert_close(sample.norm(dim=-1), torch.ones(12))

    def test_vp_training_corruption_and_normalized_weight_reduction(self):
        actor = PreferenceActor(5, 4, hidden=24, steps=4)
        for parameter in actor.parameters():
            parameter.data.zero_()
        # Non-unit supplied endpoints are retained in the VP corruption.
        target = self.targets * 2.5
        weights = torch.arange(1, 13, dtype=torch.float32)
        captured = []
        hook = actor.register_forward_pre_hook(lambda module, args: captured.append((args[2].clone(), args[3].clone())))
        rng = torch.Generator().manual_seed(15)
        level = torch.randint(4, (12,), generator=rng)
        epsilon = torch.randn((12, 4), generator=rng)
        abar = actor.alpha_bars[level, None]
        expected_input = abar.sqrt() * target + (1 - abar).sqrt() * epsilon
        expected_loss = (epsilon.square().mean(-1) * weights / weights.sum()).sum()
        loss, diagnostics = actor.loss(self.obs, self.prefs, target, weights, torch.Generator().manual_seed(15))
        hook.remove()
        torch.testing.assert_close(captured[0][0], expected_input, atol=0., rtol=0.)
        torch.testing.assert_close(captured[0][1], (level + 1.) / 4, atol=0., rtol=0.)
        torch.testing.assert_close(loss, expected_loss)
        scaled, _ = actor.loss(self.obs, self.prefs, target, weights * 17, torch.Generator().manual_seed(15))
        torch.testing.assert_close(loss, scaled)
        self.assertTrue(all(math.isfinite(v) for v in diagnostics.values()))
        with self.assertRaises(ValueError):
            actor.loss(self.obs, self.prefs, target, torch.zeros(12))
        with self.assertRaises(ValueError):
            actor.loss(self.obs, self.prefs, target, -torch.ones(12))

    def test_equal_nfe_parameter_counts_and_both_losses_train(self):
        counts = []
        for kind in ("diffusion", "sphere"):
            actor = PreferenceActor(5, 4, hidden=24, steps=8, kind=kind)
            counts.append(sum(p.numel() for p in actor.parameters()))
            loss, _ = actor.loss(self.obs, self.prefs, self.targets, generator=torch.Generator().manual_seed(33))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters()))
            rows = []
            hook = actor.register_forward_hook(lambda module, args, output: rows.append(len(output)))
            a = actor.sample(self.obs, self.prefs, torch.Generator().manual_seed(34))
            hook.remove()
            self.assertEqual(sum(rows), len(self.obs) * 8)
            self.assertEqual(actor.sampler_nfe, 8)
            torch.testing.assert_close(a.norm(dim=-1), torch.ones(12), atol=1e-6, rtol=1e-6)
            b = actor.sample(self.obs, self.prefs, torch.Generator().manual_seed(34))
            torch.testing.assert_close(a, b, atol=0., rtol=0.)
        self.assertEqual(counts[0], counts[1])

    def test_joint_potential_indices_gradient_and_state_roundtrip(self):
        potential = JointPreferencePotential(5, 4, steps=4, hidden=32)
        x = torch.randn(12, 4)
        index = torch.arange(12) % 5
        target = self.prefs[:, 0] * x[:, 0] + self.prefs[:, 1] * x[:, 1] + 2.0 * self.prefs.prod(-1) * x[:, 2] * x[:, 3]
        initial_loss = (potential(self.obs, self.prefs, x, index) - target).square().mean().item()
        optim = torch.optim.Adam(potential.parameters(), lr=5e-3)
        for _ in range(100):
            optim.zero_grad(set_to_none=True)
            loss = (potential(self.obs, self.prefs, x, index) - target).square().mean()
            loss.backward()
            optim.step()
        self.assertLess(loss.item(), initial_loss * 0.02)
        other = JointPreferencePotential(5, 4, steps=4, hidden=32)
        other.load_state_dict(copy.deepcopy(potential.state_dict()))
        torch.testing.assert_close(other(self.obs, self.prefs, x, index), potential(self.obs, self.prefs, x, index), atol=0., rtol=0.)
        torch.testing.assert_close(potential(self.obs, self.prefs, x, 2), potential(self.obs, self.prefs, x, torch.full((12,), 2)), atol=0., rtol=0.)
        torch.testing.assert_close(potential(self.obs, self.prefs, x, 2), potential(self.obs, self.prefs, x, torch.tensor([[2]])), atol=0., rtol=0.)
        with self.assertRaises(ValueError):
            potential(self.obs, self.prefs, x, torch.full((12,), 0.5))

    def test_no_resampling_telescopes_to_terminal_importance_sampler(self):
        actor = PreferenceActor(5, 4, hidden=24, steps=4)
        potential = _Potential(gauge=True)
        obs, prefs = self.obs[:3], self.prefs[:3]
        terminal_inputs = []

        def terminal(x):
            terminal_inputs.append(x.clone())
            return 1500.0 + 0.8 * x[:, 0].double() - x[:, 2].double().square()

        smc, stats = preference_smc(actor, potential, obs, prefs, terminal, 11, torch.Generator().manual_seed(12), ess_fraction=0.)
        baseline, base_stats = terminal_weight_sample(actor, obs, prefs, terminal, 11, torch.Generator().manual_seed(12))
        torch.testing.assert_close(terminal_inputs[0], terminal_inputs[1], atol=0., rtol=0.)
        torch.testing.assert_close(smc, baseline, atol=0., rtol=0.)
        self.assertAlmostEqual(stats["log_normalizer_mean"], base_stats["log_normalizer_mean"], places=9)
        self.assertAlmostEqual(stats["ess_final"], base_stats["ess_final"], places=9)
        self.assertEqual(stats["nfe"], 3 * 11 * 4)
        self.assertEqual(stats["guide_calls"], potential.rows)
        self.assertEqual(stats["terminal_evaluations"], 3 * 11)
        self.assertEqual(stats["resampling_events"], 0.)

    def test_initial_tilt_resampling_matches_explicit_ancestor_oracle(self):
        actor = _IdentityChain()
        obs, prefs = torch.zeros(1, 1), torch.tensor([[0.4, 0.6]])

        class ExactH(nn.Module):
            def forward(self, obs, prefs, x, index):
                return (1.5 * x[:, 0]).double()

        potential = ExactH()
        terminal = lambda x: (1.5 * x[:, 0]).double()
        rng = torch.Generator().manual_seed(38)
        log_weights = terminal(actor.points).view(1, 4)
        ancestors = torch.multinomial(torch.softmax(log_weights, -1), 4, replacement=True, generator=rng)
        final = torch.multinomial(torch.full((1, 4), 0.25, dtype=torch.float64), 1, replacement=True, generator=rng)
        expected = actor.points[ancestors[0, final[0, 0]]].view(1, 2)
        actual, stats = preference_smc(actor, potential, obs, prefs, terminal, 4, torch.Generator().manual_seed(38), ess_fraction=.8)
        torch.testing.assert_close(actual, expected, atol=0., rtol=0.)
        self.assertEqual(stats["resampling_events"], 1.)
        self.assertEqual(stats["resampled_rows"], 4.)
        self.assertEqual(stats["ess_final"], 4.)
        self.assertAlmostEqual(stats["log_normalizer_mean"], float(torch.logsumexp(log_weights, -1) - math.log(4)), places=12)

    def test_resampling_gauge_invariance_and_zero_support_errors(self):
        actor = PreferenceActor(5, 4, hidden=24, steps=4)
        terminal = lambda x: -x[:, 0].double().square()
        a, stats_a = preference_smc(actor, _Potential(), self.obs[:2], self.prefs[:2], terminal, 16, torch.Generator().manual_seed(24), .9)
        b, stats_b = preference_smc(actor, _Potential(gauge=True), self.obs[:2], self.prefs[:2], terminal, 16, torch.Generator().manual_seed(24), .9)
        torch.testing.assert_close(a, b, atol=0., rtol=0.)
        self.assertGreater(stats_a["resampling_events"], 0.)
        self.assertEqual(stats_a["resampling_events"], stats_b["resampling_events"])
        self.assertAlmostEqual(stats_a["log_normalizer_mean"], stats_b["log_normalizer_mean"], places=8)
        with self.assertRaisesRegex(ValueError, "zero weight"):
            terminal_weight_sample(actor, self.obs[:1], self.prefs[:1], lambda x: x.new_full((len(x),), -math.inf), 4)
        with self.assertRaisesRegex(ValueError, "invalid"):
            preference_smc(actor, _Potential(), self.obs[:1], self.prefs[:1], lambda x: x.new_full((len(x),), math.inf), 4)


if __name__ == "__main__":
    unittest.main()
