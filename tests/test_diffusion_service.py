"""Native-service integration and weighted unit-cost diffusion mechanism checks."""

from copy import deepcopy
import math
import unittest
from unittest import mock

import numpy as np
import torch

from solvercell.diffusion_service import DiffusionRFMService, UnitDiffusionPolicy, _native_service
from solvercell.policies import _unit
from solvercell.upstream import DPMD, DPMDConfig, Experience, _dpmd


class DiffusionServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def service(self, **kwargs):
        service = DiffusionRFMService(steps=2, **kwargs)
        service.device = torch.device("cpu")
        service.init(40, 40, lr=4e-4, seed=73, force=True)
        return service

    def test_raw_weighted_loss_has_correct_coordinate_reductions(self):
        model = UnitDiffusionPolicy(3, 4, bridge_sigma=0.5, steps=2)
        for parameter in model.parameters():
            parameter.data.zero_()
        obs = torch.zeros(2, 3)
        z0 = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        target = torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 1.]])
        weights = torch.tensor([2., 0.])
        torch.manual_seed(101)
        t = torch.rand(2, 1)
        epsilon = torch.randn(2, 4)
        velocity_target = target - z0 + (0.5 / math.sqrt(4)) * math.pi * torch.cos(math.pi * t) * epsilon
        expected = (weights * (velocity_target.square().sum(-1) + epsilon.square().mean(-1))).mean()
        torch.manual_seed(101)
        loss, diagnostics = model.weighted_loss(obs, z0, target, weights)
        torch.testing.assert_close(loss, expected)
        self.assertAlmostEqual(diagnostics["weighted_velocity_sqnorm"] + diagnostics["weighted_noise_mse"], float(loss), places=6)
        torch.manual_seed(101)
        scaled, _ = model.weighted_loss(obs, z0, target, 7 * weights)
        torch.testing.assert_close(scaled, 7 * loss)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(model.hidden, 256)
        self.assertEqual(model.loss_reduction["weights"], "raw_native_weights")

    def test_sampling_targets_and_native_perturbation_keep_unit_costs(self):
        service = self.service()
        obs = np.zeros((3, 40), dtype=np.float32)
        torch.manual_seed(102)
        samples = service.sample(obs, K=2, steps=2)
        torch.manual_seed(102)
        targets = service.sample_target(obs, K=2, steps=2)
        np.testing.assert_array_equal(samples, targets)
        self.assertEqual(samples.shape, (3, 2, 40))
        np.testing.assert_allclose(np.linalg.norm(samples, axis=-1), 1., atol=1e-6)
        noisy = service.sample(obs, K=2, steps=2, kappa=28., J_noise=3)
        self.assertEqual(noisy.shape, (3, 6, 40))
        np.testing.assert_allclose(np.linalg.norm(noisy, axis=-1), 1., atol=1e-6)
        original = _native_service._RFMService()
        original.device = service.device
        self.assertIs(type(service).perturb, _native_service._RFMService.perturb)
        torch.manual_seed(103)
        perturbation = service.perturb(samples[:, 0], kappa=28., J=2)
        torch.manual_seed(103)
        np.testing.assert_array_equal(perturbation, original.perturb(samples[:, 0], kappa=28., J=2))
        before = deepcopy(service.model_target.state_dict())
        loss = service.update(obs, samples[:, 0], np.array([1., 2., 3.], dtype=np.float32))
        self.assertTrue(math.isfinite(loss))
        for key, value in service.model_target.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        service.soft_update_target(1.)
        for current, target in zip(service.model.parameters(), service.model_target.parameters()):
            torch.testing.assert_close(current, target, atol=0., rtol=0.)

    def test_projection_modes_choose_old_actual_or_independent_unit_source(self):
        obs = np.ones((4, 40), dtype=np.float32)
        targets = _unit(torch.arange(1., 161.).reshape(4, 40)).numpy()
        for mode in ("point", "cell", "independent"):
            captured = {}

            def projector(observations, source, old_target):
                captured["projection_source"] = source.clone()
                captured["observations"] = observations.clone()
                # A finite, nonzero homogeneous orthant-cell projection. The
                # integration test here checks callback/normalization wiring;
                # benchmark-specific action equivalence belongs to its projector.
                return old_target.sign() * source.abs().clamp_min(0.1)

            service = self.service(mode=mode, endpoint_projector=projector)
            original_loss = service.model.weighted_loss

            def capture_loss(observations, source, endpoints, weights):
                captured["training_source"], captured["endpoint"] = source.clone(), endpoints.clone()
                return original_loss(observations, source, endpoints, weights)

            with mock.patch.object(service.model, "weighted_loss", side_effect=capture_loss):
                loss = service.update(obs, targets, np.ones(4, dtype=np.float32))
            self.assertTrue(math.isfinite(loss))
            self.assertTrue(torch.equal(captured["observations"], torch.from_numpy(obs)))
            self.assertEqual(torch.equal(captured["projection_source"], captured["training_source"]), mode == "cell")
            if mode == "point":
                torch.testing.assert_close(captured["projection_source"], torch.from_numpy(targets))
            for key in ("projection_source", "training_source", "endpoint"):
                torch.testing.assert_close(captured[key].norm(dim=-1), torch.ones(4), atol=1e-6, rtol=0.)
            self.assertTrue(torch.all(captured["endpoint"] > 0))
            self.assertTrue(all(math.isfinite(value) for value in service.last_diagnostics.values()))

    def test_real_native_dpmd_updates_with_only_service_replaced(self):
        service = self.service()
        cfg = DPMDConfig(flow_steps=2, num_particles=2, M_smooth=2)
        rng = np.random.default_rng(104)
        obs = rng.integers(0, 4, size=(4, 40)).astype(np.float32)
        costs = service.sample(obs, K=1, steps=2)[:, 0]
        batch = Experience(obs, costs, np.arange(1., 5., dtype=np.float32), obs[::-1].copy(),
                           np.array([0., 1., 0., 1.], dtype=np.float32), costs, np.zeros(4, dtype=np.int64))
        with mock.patch.object(_dpmd, "rfm_service", service):
            native = DPMD(40, 40, device=torch.device("cpu"), cfg=cfg)
            for _ in range(2):
                info = native.update(batch)
                self.assertTrue(all(math.isfinite(value) for value in info.values()))
        self.assertEqual(native.step, 2)
        self.assertEqual(service.update_count, 2)
        self.assertEqual(native.cfg.reward_scale, 1.)
        self.assertLess(service.last_diagnostics["source_unit_norm_error"], 1e-6)
        self.assertLess(service.last_diagnostics["endpoint_unit_norm_error"], 1e-6)


if __name__ == "__main__":
    unittest.main()
