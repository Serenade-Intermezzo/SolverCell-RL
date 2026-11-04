"""Replace only LSFlow's actor service with a unit-cost diffusion generator.

The native service's sampling wrapper, execution/smoothing perturbation,
target synchronization, and target EMA are inherited unchanged. DPMD remains
responsible for its original latent critics, Bellman targets, and raw mirror-
descent weights. These are actor replacements, not upstream reproductions.

The interpolant starts from a uniform sphere direction and targets a unit cost.
Its path is Euclidean and its interior noise is Gaussian; it is not a spherical
Brownian diffusion. Final sample normalization restores the native cost domain.
Normalizing a projected endpoint preserves a positively homogeneous decoder's
action but does not establish a nearest-point coupling on the sphere, preserve
the unnormalized numerical margin, or guarantee exact finite-step sampling.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from .policies import Policy, _unit


_VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "lsflow"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))
_native_service = importlib.import_module("models.rfm.service")
if not Path(_native_service.__file__).resolve().is_relative_to(_VENDOR):
    raise ImportError(f"Native RFM service resolved outside pinned vendor: {_native_service.__file__}")


class UnitDiffusionPolicy(Policy):
    """Native-width two-head network with a unit prior and unit output costs.

    bridge_sigma controls the bridge noise's RMS vector norm; the coordinate
    standard deviation is bridge_sigma/sqrt(act_dim). The velocity objective
    sums coordinates as native RFM does, while the standard-normal noise head
    uses coordinate mean-square error. Native raw weights multiply both terms.
    """

    loss_reduction = {
        "velocity": "sum_over_coordinates",
        "noise": "mean_over_coordinates",
        "weights": "raw_native_weights",
        "batch": "mean",
    }

    def __init__(self, obs_dim, act_dim, *, bridge_sigma=0.25, diffusion_scale=1.0, steps=36):
        super().__init__(obs_dim, act_dim, hidden=256, steps=steps, kind="diffusion",
                         bridge_sigma=bridge_sigma / math.sqrt(act_dim), diffusion_scale=diffusion_scale)
        self.bridge_norm_scale = float(bridge_sigma)

    def weighted_loss(self, obs: Tensor, z0: Tensor, targets: Tensor, weights: Tensor):
        """Actual weighted stochastic-interpolant regression, without resampling."""
        expected = (len(obs), self.act_dim)
        if z0.shape != expected or targets.shape != expected or weights.shape != (len(obs),):
            raise ValueError("Expected [B,D] source/endpoint and [B] native weights")
        if not torch.isfinite(weights).all() or torch.any(weights < 0):
            raise ValueError("Native actor weights must be finite and nonnegative")
        if not bool(weights.sum() > 0):
            raise ValueError("At least one native actor weight must be positive")
        weights = weights.detach()
        t = torch.rand((len(obs), 1), device=obs.device, dtype=obs.dtype)
        epsilon = torch.randn(z0.shape, device=z0.device, dtype=z0.dtype)
        sigma = self.bridge_sigma * torch.sin(math.pi * t)
        sigma_prime = self.bridge_sigma * math.pi * torch.cos(math.pi * t)
        point = (1.0 - t) * z0 + t * targets + sigma * epsilon
        velocity_target = targets - z0 + sigma_prime * epsilon
        velocity, noise = self(obs, point, t)
        velocity_errors = (velocity - velocity_target).square().sum(dim=-1)
        noise_errors = (noise - epsilon).square().mean(dim=-1)
        weighted_velocity = (weights * velocity_errors).mean()
        weighted_noise = (weights * noise_errors).mean()
        loss = weighted_velocity + weighted_noise
        return loss, {
            "velocity_sqnorm": float(velocity_errors.detach().mean()),
            "noise_mse": float(noise_errors.detach().mean()),
            "weighted_velocity_sqnorm": float(weighted_velocity.detach()),
            "weighted_noise_mse": float(weighted_noise.detach()),
            "weight_mean": float(weights.mean()),
            "weight_ess": float(weights.sum().square() / weights.square().sum()),
        }

    @torch.no_grad()
    def sample(self, obs: Tensor, K: int = 1, steps: int | None = None) -> Tensor:
        """Native RFM model interface: return [B,K,D], normalized to the sphere."""
        obs = self._obs_batch(obs)
        steps = self.steps if steps is None else steps
        if not isinstance(K, int) or K < 1 or not isinstance(steps, int) or steps < 1:
            raise ValueError("K and steps must be positive integers")
        batch = len(obs)
        expanded = obs.repeat_interleave(K, dim=0)
        x = _unit(torch.randn((batch * K, self.act_dim), device=obs.device, dtype=obs.dtype))
        dt = 1.0 / steps
        # Midpoint time labels with left-step states, as in the initial policy.
        # This remains a finite-step Euler-Maruyama approximation.
        for step in range(steps):
            t = x.new_full((batch * K, 1), (step + 0.5) * dt)
            velocity, predicted_noise = self(expanded, x, t)
            sigma = self.bridge_sigma * torch.sin(math.pi * t)
            g = self.diffusion_scale * sigma
            drift = velocity - 0.5 * self.diffusion_scale**2 * sigma * predicted_noise
            x = x + dt * drift
            if self.diffusion_scale > 0:
                x = x + math.sqrt(dt) * g * torch.randn_like(x)
        if not torch.isfinite(x).all():
            raise FloatingPointError("Nonfinite native-structure diffusion sample")
        return _unit(x).reshape(batch, K, self.act_dim)


class DiffusionRFMService(_native_service._RFMService):
    """Drop-in actor service for unchanged native DPMD training code.

    ``mode='point'`` retains the old sampled unit coefficient as endpoint, or
    sends that same old coefficient to an optional projector as its source
    (e.g. to match the other modes' safety margin without coupling to z0).
    ``mode='cell'`` calls ``endpoint_projector(obs, z0, old_target)`` with the
    actual unit training source. ``mode='independent'`` passes a separately
    sampled unit direction, preserving its conditional endpoint marginal while
    breaking the source-conditioned coupling. The callback must return a finite
    nonzero [B,D] tensor and enforce decoder-action equivalence; this generic
    service cannot verify benchmark-specific cells. Its result is normalized.

    The caller controls ``device`` before ``init``, just as in native RFM. All
    sampling uses global Torch RNG, so the adapter's existing fork_rng evaluation
    boundary and native RNG checkpointing continue to apply. Callback and scalar
    sampler settings belong in the outer resolved config/checkpoint provenance.
    """

    def __init__(
        self, *, mode: str = "point",
        endpoint_projector: Callable[[Tensor, Tensor, Tensor], Tensor] | None = None,
        bridge_sigma: float = 0.25, diffusion_scale: float = 1.0, steps: int = 36,
    ):
        super().__init__()
        if mode not in {"point", "cell", "independent"}:
            raise ValueError(f"Unknown native diffusion endpoint mode: {mode}")
        if mode != "point" and endpoint_projector is None:
            raise ValueError(f"{mode} mode requires a benchmark endpoint projector")
        self.mode, self.endpoint_projector = mode, endpoint_projector
        self.bridge_sigma, self.diffusion_scale, self.steps = bridge_sigma, diffusion_scale, steps
        self.loss_reduction = UnitDiffusionPolicy.loss_reduction.copy()
        self.last_diagnostics = {}
        self.projection_seconds = 0.0
        self.update_count = 0

    def init(self, obs_dim: int, act_dim: int, lr: float, seed: int = 0, force: bool = False):
        self.lr, self.seed = float(lr), int(seed)
        if force or self.model is None:
            # Same init API/seeding behavior as the native service. The outer
            # adapter can fork RNG around replacement to preserve critic init.
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            self.obs_dim, self.act_dim = int(obs_dim), int(act_dim)
            options = {"bridge_sigma": self.bridge_sigma, "diffusion_scale": self.diffusion_scale, "steps": self.steps}
            self.model = UnitDiffusionPolicy(self.obs_dim, self.act_dim, **options).to(self.device)
            self.model_target = UnitDiffusionPolicy(self.obs_dim, self.act_dim, **options).to(self.device)
            self.model_target.load_state_dict(self.model.state_dict())
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            self.last_diagnostics = {}
            self.projection_seconds = 0.0
            self.update_count = 0

    def update(self, obs_np: np.ndarray, z1_np: np.ndarray, w_np: np.ndarray) -> float:
        assert self.model is not None and self.optim is not None
        obs = _native_service._flatten_obs(obs_np).to(self.device)
        if obs.shape[1] != self.obs_dim:
            obs = obs[:, :self.obs_dim] if obs.shape[1] > self.obs_dim else torch.nn.functional.pad(
                obs, (0, self.obs_dim - obs.shape[1]))
        old_target = torch.as_tensor(z1_np, device=self.device, dtype=torch.float32)
        weights = torch.as_tensor(w_np, device=self.device, dtype=torch.float32)
        if old_target.ndim == 1:
            old_target = old_target.unsqueeze(0)
        if weights.ndim == 0:
            weights = weights.unsqueeze(0)
        if old_target.shape != (len(obs), self.act_dim):
            raise ValueError("Native diffusion target shape does not match observations and latent dimension")
        if not torch.isfinite(old_target).all() or torch.any(old_target.norm(dim=-1) <= 1e-8):
            raise ValueError("Native diffusion endpoints must be finite, nonzero costs")
        with torch.no_grad():
            old_target = _unit(old_target)
            z0 = _unit(torch.randn_like(old_target))
            elapsed = 0.0
            endpoint = old_target
            if self.endpoint_projector is not None:
                if self.mode == "point":
                    projection_source = old_target
                elif self.mode == "cell":
                    projection_source = z0
                else:
                    projection_source = _unit(torch.randn_like(z0))
                started = time.perf_counter()
                endpoint = self.endpoint_projector(obs, projection_source, old_target)
                elapsed = time.perf_counter() - started
                if not isinstance(endpoint, Tensor) or endpoint.shape != old_target.shape:
                    raise ValueError("Endpoint projector must return a [B,D] torch tensor")
                endpoint = endpoint.to(device=self.device, dtype=old_target.dtype)
                if not torch.isfinite(endpoint).all() or torch.any(endpoint.norm(dim=-1) <= 1e-8):
                    raise ValueError("Endpoint projector returned nonfinite or zero costs")
                endpoint = _unit(endpoint)
            endpoint = endpoint.detach()
            source_norm_error = float((z0.norm(dim=-1) - 1).abs().max())
            endpoint_norm_error = float((endpoint.norm(dim=-1) - 1).abs().max())
            endpoint_transport = float((endpoint - z0).square().sum(-1).mean())
            old_transport = float((old_target - z0).square().sum(-1).mean())
        self.optim.zero_grad(set_to_none=True)
        loss, diagnostics = self.model.weighted_loss(obs, z0, endpoint, weights)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite native-structure diffusion objective")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0, error_if_nonfinite=True)
        self.optim.step()
        self.update_count += 1
        self.projection_seconds += elapsed
        self.last_diagnostics = {
            **diagnostics, "actor_grad_norm": float(grad_norm),
            "endpoint_transport": endpoint_transport, "old_point_transport": old_transport,
            "projection_seconds": elapsed, "source_unit_norm_error": source_norm_error,
            "endpoint_unit_norm_error": endpoint_norm_error,
        }
        return float(loss.detach().cpu())


__all__ = ["DiffusionRFMService", "UnitDiffusionPolicy"]
