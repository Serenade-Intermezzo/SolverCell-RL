"""Continuous cost policies used by the solver-cell mechanism experiments.

Diffusion and its zero-SDE-noise ablation have exactly the same two-head
architecture and loss. ``spherical_flow`` is an adapted reference: its Fourier
time features, geodesic flow objective, and exponential-map Heun sampler follow
``vendor/lsflow/models/rfm/policy.py``. It is not an upstream LSFlow reproduction:
the trainer supplies the endpoint coupling, the loss is unweighted coordinate
MSE, and time is uniform instead of antithetic/clamped to [0.1, 0.9]. We also
handle exact antipodes and explicitly project exponential-map inputs tangent.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _unit(x: Tensor) -> Tensor:
    """Normalize, with a deterministic first-axis convention for zero vectors."""
    norm = x.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(x)
    fallback[..., 0] = 1.0
    return torch.where(norm > 1e-8, x / norm.clamp_min(1e-8), fallback)


def _tangent(x: Tensor, v: Tensor) -> Tensor:
    return v - (x * v).sum(dim=-1, keepdim=True) * x


def _sphere_expmap(x: Tensor, v: Tensor) -> Tensor:
    v = _tangent(x, v)
    norm = v.norm(dim=-1, keepdim=True)
    # torch.sinc(u) = sin(pi*u)/(pi*u), including the exact zero limit.
    return _unit(torch.cos(norm) * x + torch.sinc(norm / math.pi) * v)


def _sphere_path(z0: Tensor, target: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """Constant-speed shortest geodesic and its analytic tangent velocity.

    An antipodal endpoint has no unique shortest geodesic. Choose a stable
    orthogonal coordinate direction there (and within roundoff of the pole).
    atan2 avoids acos' endpoint singularities and preserves identical endpoints.
    """
    p, q = _unit(z0), _unit(target)
    dot = (p * q).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    direction = _tangent(p, q)
    length = direction.norm(dim=-1, keepdim=True)
    axis = F.one_hot(p.abs().argmin(dim=-1), p.shape[-1]).to(p)
    fallback = _unit(_tangent(p, axis))
    direction = torch.where(length > 1e-6, direction / length.clamp_min(1e-8), fallback)
    theta = torch.atan2(length, dot)
    phase = t * theta
    point = _unit(torch.cos(phase) * p + torch.sin(phase) * direction)
    velocity = theta * (-torch.sin(phase) * p + torch.cos(phase) * direction)
    return point, _tangent(point, velocity)


class TimeEmbed(nn.Module):
    """The upstream reference's 32-dimensional, 16-harmonic time embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(16) * math.pi, persistent=False)
        self.proj = nn.Sequential(nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 32))

    def forward(self, t: Tensor) -> Tensor:
        phase = t * self.freqs.unsqueeze(0)
        return self.proj(torch.cat((phase.sin(), phase.cos()), dim=-1))


class Policy(nn.Module):
    """Observation-conditioned cost generator with reproducible RNG injection.

    ``steps`` denotes integration steps, not equal network-evaluation budgets:
    diffusion uses ``steps`` evaluations; spherical Heun uses ``2 * steps``.
    ``diffusion_scale=0`` removes SDE sampling noise but retains both learned
    heads and the full stochastic-interpolant training loss. ``bridge_sigma``
    stays positive so this ablation retains exactly the same training targets.
    All inputs and supplied generators must be on the policy's device.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int = 128,
        steps: int = 12,
        kind: str = "diffusion",
        bridge_sigma: float = 0.5,
        diffusion_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if kind not in {"diffusion", "spherical_flow"}:
            raise ValueError(f"Unknown policy kind: {kind}")
        if min(obs_dim, act_dim, hidden, steps) < 1:
            raise ValueError("Dimensions, hidden width, and integration steps must be positive")
        if kind == "spherical_flow" and act_dim < 2:
            raise ValueError("Spherical geodesic flow requires act_dim >= 2")
        if not math.isfinite(bridge_sigma) or bridge_sigma <= 0:
            raise ValueError("bridge_sigma must be finite and positive")
        if not math.isfinite(diffusion_scale) or diffusion_scale < 0:
            raise ValueError("diffusion_scale must be finite and nonnegative")
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.hidden, self.steps, self.kind = hidden, steps, kind
        self.bridge_sigma, self.diffusion_scale = bridge_sigma, diffusion_scale
        self.sampler_nfe = steps * (2 if kind == "spherical_flow" else 1)
        self.time_embed = TimeEmbed()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim + 32, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.velocity_head = nn.Linear(hidden, act_dim)
        self.noise_head = nn.Linear(hidden, act_dim) if kind == "diffusion" else None

    def _obs_batch(self, obs: Tensor) -> Tensor:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if obs.ndim != 2 or obs.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected observations with shape [B, {self.obs_dim}]")
        return obs

    def forward(self, obs: Tensor, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor | None]:
        """Return velocity and conditional bridge-noise estimate (if present)."""
        obs = self._obs_batch(obs)
        if t.ndim == 0:
            t = t.expand(x.shape[0], 1)
        elif t.ndim == 1:
            t = t[:, None]
        h = self.net(torch.cat((obs, x, self.time_embed(t)), dim=-1))
        velocity = self.velocity_head(h)
        if self.kind == "spherical_flow":
            return _tangent(x, velocity), None
        return velocity, self.noise_head(h)

    def loss(
        self, obs: Tensor, z0: Tensor, targets: Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        obs = self._obs_batch(obs)
        expected = (obs.shape[0], self.act_dim)
        if tuple(z0.shape) != expected or tuple(targets.shape) != expected:
            raise ValueError(f"Expected z0 and targets with shape {expected}")
        t = torch.rand((obs.shape[0], 1), dtype=obs.dtype, device=obs.device, generator=generator)
        if self.kind == "spherical_flow":
            point, target_velocity = _sphere_path(z0, targets, t)
            velocity, _ = self(obs, point, t)
            velocity_loss = F.mse_loss(velocity, target_velocity)
            return velocity_loss, {"velocity_mse": velocity_loss.detach().item(), "noise_mse": 0.0}

        epsilon = torch.randn(z0.shape, dtype=z0.dtype, device=z0.device, generator=generator)
        sigma = self.bridge_sigma * torch.sin(math.pi * t)
        sigma_prime = self.bridge_sigma * math.pi * torch.cos(math.pi * t)
        point = (1.0 - t) * z0 + t * targets + sigma * epsilon
        target_velocity = targets - z0 + sigma_prime * epsilon
        velocity, predicted_noise = self(obs, point, t)
        velocity_loss = F.mse_loss(velocity, target_velocity)
        noise_loss = F.mse_loss(predicted_noise, epsilon)
        loss = velocity_loss + noise_loss
        return loss, {"velocity_mse": velocity_loss.detach().item(), "noise_mse": noise_loss.detach().item()}

    @torch.no_grad()
    def sample(self, obs: Tensor, generator: torch.Generator | None = None) -> Tensor:
        obs = self._obs_batch(obs)
        x = torch.randn((obs.shape[0], self.act_dim), dtype=obs.dtype, device=obs.device, generator=generator)
        dt = 1.0 / self.steps
        if self.kind == "spherical_flow":
            x = _unit(x)
            for step in range(self.steps):
                t = torch.full((obs.shape[0], 1), step * dt, dtype=obs.dtype, device=obs.device)
                v0, _ = self(obs, x, t)
                provisional = _sphere_expmap(x, dt * v0)
                v1, _ = self(obs, provisional, t + dt)
                x = _sphere_expmap(x, 0.5 * dt * (v0 + v1))
            return x

        # Euler-Maruyama with midpoint time labels avoids score evaluation at
        # either singular bridge endpoint. x is still the left-step state; this
        # is not a stochastic midpoint state integrator or an exact sampler.
        for step in range(self.steps):
            t = torch.full((obs.shape[0], 1), (step + 0.5) * dt, dtype=obs.dtype, device=obs.device)
            velocity, predicted_noise = self(obs, x, t)
            sigma = self.bridge_sigma * torch.sin(math.pi * t)
            g = self.diffusion_scale * sigma
            # score = -E[epsilon | X_t, obs] / sigma. Cancel sigma analytically
            # in (g^2/2)*score to avoid a numerical division near endpoints.
            drift = velocity - 0.5 * self.diffusion_scale**2 * sigma * predicted_noise
            x = x + dt * drift
            if self.diffusion_scale > 0:
                noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
                x = x + math.sqrt(dt) * g * noise
        return x
