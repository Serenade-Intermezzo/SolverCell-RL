"""Tensor-only hot paths for the pinned native LSFlow learner.

This optimization changes data movement, not the learning rule. The actor
sampler, tangent-Gaussian perturbation, trimmed Bellman reduction, loss,
diagnostics and random draw order remain upstream. In particular, diagnostics
are not skipped: doing so would change the native global RNG stream.

The adapter's service is read dynamically, allowing a compatible replacement
service whose model.sample(obs, K=..., steps=...) returns a tensor. Vendored
classes and the original NativeLSFlow adapter are never modified globally.
"""

from __future__ import annotations

from types import MethodType

import torch

from .upstream import NativeLSFlow, _dpmd, _service_module


class TensorNativeLSFlow(NativeLSFlow):
    """Native LSFlow with tensor-only candidate and smoothing intermediates.

    Every model and optimizer state key is unchanged, so a checkpoint from
    NativeLSFlow can be loaded directly. CPU parity is verified bit-for-bit;
    any speed claim must be based on a separate measured benchmark.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.install_tensor_paths()

    def install_tensor_paths(self):
        """Bind local overrides without touching DPMD's class or vendor files."""
        adapter = self

        @torch.no_grad()
        def sample_candidates(native, obs, K, *, use_target=False):
            obs = _dpmd._fit_width(obs, native.obs_dim)
            model = adapter.service.model_target if use_target else adapter.service.model
            return model.sample(obs, K=K, steps=native.cfg.flow_steps)

        @torch.no_grad()
        def smoothed_value(native, next_obs):
            next_obs = _dpmd._fit_width(next_obs, native.obs_dim)
            batch = next_obs.shape[0]
            candidates = max(1, int(native.cfg.M_smooth))
            perturbations = max(1, int(native.cfg.J_smooth))
            concentration = float(native.cfg.kappa_smooth)

            costs = native._sample_candidates(next_obs, candidates, use_target=True)
            centers = costs.reshape(batch * candidates, native.act_dim)
            # This is exactly the function called by upstream service.perturb;
            # omitting its CPU NumPy wrapper does not omit or replace any draw.
            perturbed = _service_module._vmf_perturb_torch(
                centers, kappa=concentration, J=perturbations
            ).reshape(batch, candidates, perturbations, native.act_dim)
            repeated_obs = next_obs.view(batch, 1, 1, native.obs_dim).expand(
                batch, candidates, perturbations, native.obs_dim
            ).reshape(batch * candidates * perturbations, native.obs_dim)
            flat_cost = perturbed.reshape(batch * candidates * perturbations, native.act_dim)
            q1 = native.tq1(repeated_obs, flat_cost).view(batch, candidates, perturbations)
            q2 = native.tq2(repeated_obs, flat_cost).view(batch, candidates, perturbations)
            q_min = torch.minimum(q1, q2)
            flat_values = q_min.view(batch, -1)
            trim = int(0.2 * flat_values.shape[1])
            values, _ = torch.sort(flat_values, dim=1, descending=True)
            return values[:, :-trim].mean(dim=1) if trim > 0 else values.mean(dim=1)

        self.native._sample_candidates = MethodType(sample_candidates, self.native)
        self.native._smoothed_value = MethodType(smoothed_value, self.native)

