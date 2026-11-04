"""Native-policy improvement with diffusion on the decoder's effective space.

The original latent critic, raw observations/rewards, Bellman smoothing,
pretraining, alpha and update schedule remain unchanged. Quotient variants
remove only the decoder-invisible kernel from actor costs. Their samples are
lifted through an orthonormal basis before every original critic/decoder call.
"""
from copy import deepcopy
import time

import numpy as np
import torch
from torch import nn

from .diffusion_service import DiffusionRFMService
from .fast_native import TensorNativeLSFlow
from .native_geometry import NativeCellProjector
from .policies import _unit
from .upstream import _service_module


class LiftedPolicy(nn.Module):
    """Expose native-dimensional samples while keeping an inner quotient model."""
    def __init__(self, inner, basis):
        super().__init__()
        self.inner = inner
        self.register_buffer("basis", basis.detach().clone())
        self.act_dim = basis.shape[0]

    @torch.no_grad()
    def sample(self, obs, K=1, steps=30):
        return self.inner.sample(obs, K=K, steps=steps) @ self.basis.T

    def rfm_loss(self, obs, targets, weights):
        return self.inner.rfm_loss(obs, _unit(targets @ self.basis), weights)

    def weighted_loss(self, obs, z0, targets, weights):
        # Diffusion service maps old targets before creating its inner sources;
        # these three arguments are already in quotient coordinates.
        return self.inner.weighted_loss(obs, z0, targets, weights)


class QuotientDiffusionService(DiffusionRFMService):
    """Keep original service model sampling compatible with tensor fast paths."""
    def __init__(self, basis, **kwargs):
        super().__init__(**kwargs)
        self.basis = basis

    def init(self, obs_dim, act_dim, lr, seed=0, force=False):
        rebuild = force or self.model is None
        super().init(obs_dim, self.basis.shape[1], lr, seed, force)
        if rebuild:
            self.model = LiftedPolicy(self.model, self.basis)
            self.model_target = LiftedPolicy(self.model_target, self.basis)

    def update(self, obs_np, z1_np, w_np):
        old = torch.as_tensor(z1_np, device=self.device, dtype=torch.float32)
        inner_target = _unit(old @ self.basis)
        return super().update(obs_np, inner_target, w_np)


class NativeDiffusion(TensorNativeLSFlow):
    def __init__(self, cfg, env):
        # Construct exactly the original actor and critics first: different
        # replacement actor sizes must never change critic initialization.
        super().__init__(cfg, env)
        self.projector = NativeCellProjector(self.weights_p, cfg.budget)
        quotient = "quotient" in cfg.method
        self.basis = self.projector.basis if quotient else torch.eye(self.cost_dim, device=self.device)
        self.projected_rows = self.fallback_rows = 0
        self.last_projection = {}
        self.projection_seconds = 0.0
        self.is_spherical = cfg.method == "native_quotient_sphere"
        mode = "independent" if "independent" in cfg.method else ("cell" if "cell" in cfg.method else "point")
        numpy_state = np.random.get_state()
        devices = [] if self.device.type == "cpu" else [self.device.index or torch.cuda.current_device()]
        try:
            with torch.random.fork_rng(devices=devices):
                if self.is_spherical:
                    service = type(_service_module.rfm_service)()
                    service.device = self.device
                    service.init(self.obs_dim, self.basis.shape[1], lr=cfg.lr, seed=cfg.seed, force=True)
                    service.model = LiftedPolicy(service.model, self.basis)
                    service.model_target = LiftedPolicy(service.model_target, self.basis)
                else:
                    options = {"mode": mode, "endpoint_projector": self._endpoint,
                               "bridge_sigma": cfg.bridge_sigma, "diffusion_scale": cfg.diffusion_scale,
                               "steps": cfg.sampling_steps}
                    service = (QuotientDiffusionService(self.basis, **options) if quotient
                               else DiffusionRFMService(**options))
                    service.device = self.device
                    service.init(self.obs_dim, self.cost_dim, lr=cfg.lr, seed=cfg.seed, force=True)
                self.service = service
        finally:
            np.random.set_state(numpy_state)
        self.bridge_spec = {
            "actor_space": "orthonormal_decoder_quotient" if quotient else "full_native_cost",
            "actor_dimension": int(self.basis.shape[1]), "critic_cost_dimension": self.cost_dim,
            "endpoint_mode": "original_spherical_point" if self.is_spherical else mode,
            "decoder_ties": "unique_column_scores_broadcast_then_stable_topk",
            "margin": cfg.margin, "bridge_vector_sigma": cfg.bridge_sigma,
            "diffusion_scale": cfg.diffusion_scale,
            "loss_reduction": None if self.is_spherical else self.service.loss_reduction,
            "tensor_paths": True, "critic_initialization": "unchanged_native_before_actor_replacement",
        }
        self.screen_overrides.update({"actor_replacement": self.bridge_spec})

    @torch.no_grad()
    def decode_cost(self, cost):
        costs = torch.as_tensor(cost, device=self.device, dtype=torch.float32)
        action = self.projector.decode(costs)
        self.decoder_calls += int(action.numel() // self.action_dim)
        return action

    @torch.no_grad()
    def _endpoint(self, obs, source, old_target):
        started = time.perf_counter()
        full_old = old_target @ self.basis.T
        full_source = source @ self.basis.T
        actions = self.projector.decode(full_old)
        endpoint, stats = self.projector.project(full_source, actions, self.cfg.margin, full_old)
        normalized = _unit(endpoint)
        if not torch.equal(self.projector.decode(normalized), actions):
            raise RuntimeError("Normalized native endpoint changed its target action")
        self.projected_rows += stats["rows"]
        self.fallback_rows += stats["fallback_rows"]
        group_counts = actions @ self.projector.membership.to(actions.dtype)
        selected = group_counts > 0
        unselected = group_counts < self.projector.group_sizes
        distinct = ~torch.eye(self.projector.n_groups, device=self.device, dtype=torch.bool)
        mask = selected.unsqueeze(-1) & unselected.unsqueeze(-2) & distinct
        group_scores = normalized @ self.projector.unique_columns
        gaps = (group_scores.unsqueeze(-1) - group_scores.unsqueeze(-2)).masked_fill(~mask, float("inf"))
        min_gap = gaps.amin(dim=(-1, -2))
        min_gap = torch.where(torch.isfinite(min_gap), min_gap, 0.)
        self.last_projection = {"projection_fallback_fraction": stats["fallback_fraction"],
                                "projection_distance_before_normalize": stats["projection_distance_sq_mean"],
                                "endpoint_normalized_score_gap_mean": float(min_gap.mean()),
                                "endpoint_action_count_mean": float(actions.sum(-1).mean())}
        self.projection_seconds += time.perf_counter() - started
        inner_endpoint = _unit(normalized @ self.basis)
        if not torch.equal(self.projector.decode(inner_endpoint @ self.basis.T), actions):
            raise RuntimeError("Quotient roundtrip changed the projected target action")
        return inner_endpoint

    def update(self, batch):
        info = super().update(batch)
        if not self.is_spherical:
            info.update(self.service.last_diagnostics)
            info.update(self.last_projection)
            info["projection_fallback_fraction_total"] = self.fallback_rows / max(1, self.projected_rows)
        return self._finite(info)

    def state_dict(self):
        state = super().state_dict()
        state["bridge_spec"] = deepcopy(self.bridge_spec)
        state["bridge_diagnostics"] = {"projected_rows": self.projected_rows, "fallback_rows": self.fallback_rows,
                                       "last_projection": self.last_projection,
                                       "service_last_diagnostics": getattr(self.service, "last_diagnostics", {}),
                                       "service_projection_seconds": getattr(self.service, "projection_seconds", 0.0),
                                       "service_update_count": getattr(self.service, "update_count", 0)}
        return state

    def load_state_dict(self, state):
        if state.get("bridge_spec") != self.bridge_spec:
            raise ValueError("Native actor replacement specification changed")
        super().load_state_dict(state)
        d = state["bridge_diagnostics"]
        self.projected_rows, self.fallback_rows = d["projected_rows"], d["fallback_rows"]
        self.last_projection = d["last_projection"]
        if not self.is_spherical:
            self.service.last_diagnostics = d["service_last_diagnostics"]
            self.service.projection_seconds = d["service_projection_seconds"]
            self.service.update_count = d["service_update_count"]
