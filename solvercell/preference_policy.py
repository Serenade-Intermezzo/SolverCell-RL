"""Preference-conditioned finite policies and inference-time importance samplers.

The diffusion policy uses a known VP corruption and an epsilon predictor, but
its deployed finite reverse kernels deliberately have variance beta_t even at
the final step.  The Gaussian initial prior and this variance choice are not an
exact reverse-posterior claim.  Likelihoods and potentials refer to RAW chain
states; only the cost returned to a caller is normalized.

The sphere comparator is geodesic FM with exponential-map Heun integration,
adapted from the local reference.  ``steps`` is its NFE budget, so it performs
``steps // 2`` integration intervals.  Both actors have the same architecture
and number of parameters.  Neither actor is an upstream LSFlow reproduction.

SMC uses the actual finite base kernels, includes the initial potential, and
uses the exact terminal callback as its last potential.  Finite particle
outputs approximate the terminal tilt; they are not exact tilted samples.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn

from .policies import TimeEmbed, _sphere_expmap, _sphere_path, _tangent, _unit


def _condition(obs: Tensor, prefs: Tensor, obs_dim: int) -> tuple[Tensor, Tensor]:
    if obs.ndim == 1:
        obs = obs.unsqueeze(0)
    if prefs.ndim == 1:
        prefs = prefs.unsqueeze(0)
    if obs.ndim != 2 or obs.shape[1] != obs_dim or obs.shape[0] < 1:
        raise ValueError(f"obs must have shape [B, {obs_dim}] with B > 0")
    if prefs.shape != (obs.shape[0], 2):
        raise ValueError("prefs must have shape [B, 2]")
    return obs, prefs


def _state(x: Tensor, batch: int, act_dim: int) -> Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.shape != (batch, act_dim):
        raise ValueError(f"cost state must have shape [{batch}, {act_dim}]")
    return x


def _index(index: int, maximum: int) -> int:
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index <= maximum:
        raise ValueError(f"index must be an integer in [0, {maximum}]")
    return index


def _weighted_mean(errors: Tensor, weights: Tensor | None) -> tuple[Tensor, Tensor]:
    if weights is None:
        return errors.mean(), errors.new_tensor(float(errors.numel()))
    if weights.numel() != errors.numel():
        raise ValueError("weights must contain one nonnegative weight per example")
    weights = weights.to(device=errors.device, dtype=errors.dtype).reshape_as(errors).detach()
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("weights must be finite and nonnegative")
    total = weights.sum()
    if not bool(torch.isfinite(total)) or not bool(total > 0):
        raise ValueError("weights must have a finite positive sum")
    return (errors * (weights / total)).sum(), total


class PreferenceActor(nn.Module):
    """Two-objective actor. Inputs are already normalized by the learner.

    ``loss`` preserves the supplied diffusion endpoints; callers should use
    the same unit endpoint costs for both actors. Sphere geodesics necessarily
    normalize their endpoints. Loss weights are normalized by their sum, so
    candidate count does not silently alter the optimization step size.
    """

    def __init__(
        self, obs_dim: int, act_dim: int, hidden: int = 128,
        steps: int = 8, kind: str = "diffusion", beta_end: float = .20,
    ) -> None:
        super().__init__()
        if min(obs_dim, act_dim, hidden) < 1:
            raise ValueError("dimensions and hidden width must be positive")
        if not isinstance(steps, int) or steps < 2 or steps % 2:
            raise ValueError("steps must be positive and even (at least 2), to match actor NFE")
        if kind not in {"diffusion", "sphere"}:
            raise ValueError("kind must be 'diffusion' or 'sphere'")
        if not math.isfinite(beta_end) or not .02 <= beta_end < 1:
            raise ValueError('beta_end must lie in [.02,1)')
        if kind == "sphere" and act_dim < 2:
            raise ValueError("sphere requires act_dim >= 2")
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.hidden, self.steps, self.kind = hidden, steps, kind
        self.sampler_nfe = steps
        self.time_embed = TimeEmbed()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 2 + act_dim + 32, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, act_dim),
        )
        # Registered in both comparators for a stable architecture/state schema.
        betas = torch.linspace(0.02, beta_end, steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alpha_bars", (1.0 - betas).cumprod(0))

    def forward(self, obs: Tensor, prefs: Tensor, x: Tensor, t: Tensor) -> Tensor:
        obs, prefs = _condition(obs, prefs, self.obs_dim)
        x = _state(x, obs.shape[0], self.act_dim)
        if t.ndim == 0:
            t = t.expand(obs.shape[0], 1)
        elif t.ndim == 1:
            t = t[:, None]
        if t.shape != (obs.shape[0], 1):
            raise ValueError("time must be scalar or one value per example")
        prediction = self.net(torch.cat((obs, prefs, x, self.time_embed(t)), dim=-1))
        return _tangent(x, prediction) if self.kind == "sphere" else prediction

    def loss(
        self, obs: Tensor, prefs: Tensor, targets: Tensor,
        weights: Tensor | None = None, generator: torch.Generator | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        obs, prefs = _condition(obs, prefs, self.obs_dim)
        targets = _state(targets, obs.shape[0], self.act_dim)
        if self.kind == "diffusion":
            level = torch.randint(self.steps, (obs.shape[0],), device=obs.device, generator=generator)
            epsilon = torch.randn(targets.shape, device=targets.device, dtype=targets.dtype, generator=generator)
            abar = self.alpha_bars[level, None]
            x = abar.sqrt() * targets + (1.0 - abar).sqrt() * epsilon
            t = (level.to(obs.dtype) + 1.0) / self.steps
            prediction = self(obs, prefs, x, t)
            errors = (prediction - epsilon).square().mean(-1)
            diagnostic_key = "noise_mse"
        else:
            start = torch.randn(targets.shape, device=targets.device, dtype=targets.dtype, generator=generator)
            t = torch.rand((obs.shape[0], 1), device=obs.device, dtype=obs.dtype, generator=generator)
            x, velocity = _sphere_path(start, targets, t)
            prediction = self(obs, prefs, x, t)
            errors = (prediction - velocity).square().mean(-1)
            diagnostic_key = "velocity_mse"
        loss, weight_sum = _weighted_mean(errors, weights)
        return loss, {
            "loss": float(loss.detach()),
            diagnostic_key: float(errors.detach().mean()),
            "weight_sum": float(weight_sum.detach()),
        }

    def _initial(self, obs: Tensor, generator: torch.Generator | None) -> Tensor:
        x = torch.randn((obs.shape[0], self.act_dim), device=obs.device, dtype=obs.dtype, generator=generator)
        return _unit(x) if self.kind == "sphere" else x

    def transition_parameters(
        self, obs: Tensor, prefs: Tensor, x: Tensor, index: int,
    ) -> tuple[Tensor, Tensor]:
        """Actual RAW Gaussian kernel mean and scalar std (differentiable).

        ``index`` increases in generation time. The VP level decreases from
        ``steps - 1`` to 0; even level 0 retains strictly positive noise.
        """
        if self.kind != "diffusion":
            raise ValueError("raw Gaussian transitions only exist for diffusion")
        index = _index(index, self.steps - 1)
        obs, prefs = _condition(obs, prefs, self.obs_dim)
        x = _state(x, obs.shape[0], self.act_dim)
        level = self.steps - 1 - index
        t = obs.new_full((obs.shape[0], 1), (level + 1.0) / self.steps)
        epsilon = self(obs, prefs, x, t)
        mean = (x - self.betas[level] / (1.0 - self.alpha_bars[level]).sqrt() * epsilon) / self.alphas[level].sqrt()
        return mean, self.betas[level].sqrt()

    def transition_log_prob(
        self, obs: Tensor, prefs: Tensor, x: Tensor, next_x: Tensor, index: int,
    ) -> Tensor:
        """Conditional log-density of raw states, never of normalized costs."""
        mean, std = self.transition_parameters(obs, prefs, x, index)
        next_x = _state(next_x, mean.shape[0], self.act_dim)
        return (-0.5 * ((next_x - mean) / std).square() - std.log() - 0.5 * math.log(2.0 * math.pi)).sum(-1)

    @torch.no_grad()
    def transition(
        self, obs: Tensor, prefs: Tensor, x: Tensor, index: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        mean, std = self.transition_parameters(obs, prefs, x, index)
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        return mean + std * noise

    @torch.no_grad()
    def suffix(
        self, obs: Tensor, prefs: Tensor, x: Tensor, index: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if self.kind != "diffusion":
            raise ValueError("suffix sampling is defined only for the diffusion chain")
        index = _index(index, self.steps)
        obs, prefs = _condition(obs, prefs, self.obs_dim)
        x = _state(x, obs.shape[0], self.act_dim)
        for step in range(index, self.steps):
            x = self.transition(obs, prefs, x, step, generator)
        return x

    @torch.no_grad()
    def sample_path(
        self, obs: Tensor, prefs: Tensor, generator: torch.Generator | None = None,
    ) -> Tensor:
        if self.kind != "diffusion":
            raise ValueError("sample_path is defined only for the diffusion chain")
        obs, prefs = _condition(obs, prefs, self.obs_dim)
        x = self._initial(obs, generator)
        path = [x]
        for step in range(self.steps):
            x = self.transition(obs, prefs, x, step, generator)
            path.append(x)
        return torch.stack(path, dim=1)

    @torch.no_grad()
    def sample_raw(
        self, obs: Tensor, prefs: Tensor, generator: torch.Generator | None = None,
    ) -> Tensor:
        obs, prefs = _condition(obs, prefs, self.obs_dim)
        x = self._initial(obs, generator)
        if self.kind == "diffusion":
            return self.suffix(obs, prefs, x, 0, generator)
        intervals = self.steps // 2
        dt = 1.0 / intervals
        for step in range(intervals):
            t = obs.new_full((obs.shape[0], 1), step * dt)
            v0 = self(obs, prefs, x, t)
            provisional = _sphere_expmap(x, dt * v0)
            v1 = self(obs, prefs, provisional, t + dt)
            x = _sphere_expmap(x, 0.5 * dt * (v0 + v1))
        return x

    @torch.no_grad()
    def sample(
        self, obs: Tensor, prefs: Tensor, generator: torch.Generator | None = None,
    ) -> Tensor:
        return _unit(self.sample_raw(obs, prefs, generator))


class JointPreferencePotential(nn.Module):
    """Joint log-h approximator; it does not add independently learned scores.

    ``index`` is the generation-forward PATH index, including 0 and ``steps``.
    Training targets and the distinction between log(E exp U) and E(U) are the
    learner's responsibility. SMC always overrides the terminal learned value
    with the supplied exact terminal log-weight callback.
    """

    def __init__(self, obs_dim: int, act_dim: int, steps: int, hidden: int = 128) -> None:
        super().__init__()
        if min(obs_dim, act_dim, steps, hidden) < 1:
            raise ValueError("dimensions, steps, and hidden width must be positive")
        self.obs_dim, self.act_dim, self.steps = obs_dim, act_dim, steps
        self.time_embed = TimeEmbed()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 2 + act_dim + 32, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1),
        )

    def forward(self, obs: Tensor, prefs: Tensor, x: Tensor, index: int | Tensor) -> Tensor:
        obs, prefs = _condition(obs, prefs, self.obs_dim)
        x = _state(x, obs.shape[0], self.act_dim)
        if isinstance(index, int):
            index = _index(index, self.steps)
            t = obs.new_full((obs.shape[0], 1), index / self.steps)
        else:
            idx = torch.as_tensor(index, device=obs.device)
            if idx.numel() == 1:
                idx = idx.reshape(1).expand(obs.shape[0])
            if idx.numel() != obs.shape[0]:
                raise ValueError("index must be scalar or have one entry per example")
            idx = idx.reshape(obs.shape[0], 1)
            if not bool(torch.isfinite(idx).all()) or bool(((idx < 0) | (idx > self.steps) | (idx != idx.long())).any()):
                raise ValueError("path indices must be integers in [0, steps]")
            t = idx.to(obs.dtype) / self.steps
        return self.net(torch.cat((obs, prefs, x, self.time_embed(t)), dim=-1)).squeeze(-1)


def _log_values(value: Tensor, rows: int, name: str, allow_zero: bool = False) -> Tensor:
    if not isinstance(value, Tensor) or value.numel() != rows:
        raise ValueError(f"{name} must return a tensor with one log-weight per cost")
    value = value.reshape(-1).to(torch.float64)
    invalid = torch.isnan(value) | torch.isposinf(value)
    if not allow_zero:
        invalid = invalid | torch.isneginf(value)
    if bool(invalid.any()):
        raise ValueError(f"{name} returned invalid log-weights")
    return value


def _probabilities(log_weights: Tensor) -> tuple[Tensor, Tensor]:
    log_normalizer = torch.logsumexp(log_weights, dim=-1)
    if not bool(torch.isfinite(log_normalizer).all()):
        raise ValueError("all particles have zero weight for at least one observation")
    probabilities = torch.exp(log_weights - log_normalizer[:, None])
    ess = probabilities.square().sum(-1).reciprocal()
    return probabilities, ess


def _particle_inputs(actor: PreferenceActor, obs: Tensor, prefs: Tensor, particles: int) -> tuple[Tensor, Tensor, int]:
    if not isinstance(particles, int) or isinstance(particles, bool) or particles < 1:
        raise ValueError("particles must be a positive integer")
    obs, prefs = _condition(obs, prefs, actor.obs_dim)
    return obs.repeat_interleave(particles, 0), prefs.repeat_interleave(particles, 0), obs.shape[0]


def _draw_cost(x: Tensor, probabilities: Tensor, generator: torch.Generator | None) -> Tensor:
    chosen = torch.multinomial(probabilities, 1, replacement=True, generator=generator)
    return _unit(x.gather(1, chosen[:, :, None].expand(-1, 1, x.shape[-1])).squeeze(1))


@torch.no_grad()
def terminal_weight_sample(
    actor: PreferenceActor, obs: Tensor, prefs: Tensor,
    terminal_logweight: Callable[[Tensor], Tensor], particles: int,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Same-particle-budget terminal importance sampling, NOT argmax reranking.

    The callback receives raw costs in B-major order: [B * particles, D].
    It can capture obs/prefs.repeat_interleave(particles, 0) in a closure.
    """
    repeated_obs, repeated_prefs, batch = _particle_inputs(actor, obs, prefs, particles)
    raw = actor.sample_raw(repeated_obs, repeated_prefs, generator)
    log_weights = _log_values(terminal_logweight(raw), batch * particles, "terminal_logweight", True).view(batch, particles)
    probabilities, ess = _probabilities(log_weights)
    costs = _draw_cost(raw.view(batch, particles, actor.act_dim), probabilities, generator)
    nfe = batch * particles * actor.sampler_nfe
    return costs, {
        "nfe": float(nfe), "actor_nfe": float(nfe), "guide_calls": 0.0,
        "terminal_evaluations": float(batch * particles),
        "ess_mean": float(ess.mean()), "ess_min": float(ess.min()),
        "ess_final": float(ess.mean()), "ess_final_fraction": float((ess / particles).mean()),
        "resampling_events": 0.0, "resampled_rows": 0.0,
        "terminal_draws": float(batch),
        "log_normalizer_mean": float((torch.logsumexp(log_weights, -1) - math.log(particles)).mean()),
    }


@torch.no_grad()
def preference_smc(
    actor: PreferenceActor, potential: JointPreferencePotential,
    obs: Tensor, prefs: Tensor, terminal_logweight: Callable[[Tensor], Tensor],
    particles: int, generator: torch.Generator | None = None, ess_fraction: float = 0.5,
) -> tuple[Tensor, dict[str, float]]:
    """Base-propagation SMC with corrected initial/intermediate potentials.

    The unnormalized initial weight is h_0, increments are h_{k+1}/h_k,
    and the final h_K is exp(terminal_logweight). Without resampling they
    telescope to the terminal weight EXACTLY, regardless of learned h.
    Potentials may be inaccurate: this affects finite-particle efficiency,
    not the specified Feynman--Kac target. There is no gradient/drift proposal
    and therefore no omitted proposal density correction.

    ``ess_fraction=0`` disables intermediate resampling; ``1`` resamples any
    strictly non-uniform population. Final weighted selection is always made.
    All counters count processed rows, not merely batched Python invocations.
    """
    if actor.kind != "diffusion":
        raise ValueError("SMC requires a stochastic diffusion actor")
    if not math.isfinite(ess_fraction) or not 0.0 <= ess_fraction <= 1.0:
        raise ValueError("ess_fraction must lie in [0, 1]")
    repeated_obs, repeated_prefs, batch = _particle_inputs(actor, obs, prefs, particles)
    rows = batch * particles
    x = actor._initial(repeated_obs, generator).view(batch, particles, actor.act_dim)
    log_h = _log_values(potential(repeated_obs, repeated_prefs, x.reshape(rows, -1), 0), rows, "potential").view(batch, particles)
    log_weights = log_h.clone()  # Omitting this silently changes the target.
    log_z = log_weights.new_zeros(batch)
    ess_history: list[Tensor] = []
    resampling_events = 0
    resampled_rows = 0

    for index in range(actor.steps):
        probabilities, ess = _probabilities(log_weights)
        ess_history.append(ess)
        resample = ess < ess_fraction * particles
        if bool(resample.any()):
            selected = resample.nonzero(as_tuple=False).squeeze(-1)
            ancestors = torch.multinomial(probabilities[selected], particles, replacement=True, generator=generator)
            log_z[selected] += torch.logsumexp(log_weights[selected], -1) - math.log(particles)
            x[selected] = x[selected].gather(1, ancestors[:, :, None].expand(-1, -1, actor.act_dim))
            log_h[selected] = log_h[selected].gather(1, ancestors)
            log_weights[selected] = 0.0
            resampling_events += selected.numel()
            resampled_rows += selected.numel() * particles

        raw = actor.transition(repeated_obs, repeated_prefs, x.reshape(rows, -1), index, generator)
        x = raw.view(batch, particles, actor.act_dim)
        if index + 1 == actor.steps:
            next_h = _log_values(terminal_logweight(raw), rows, "terminal_logweight", True)
        else:
            next_h = _log_values(potential(repeated_obs, repeated_prefs, raw, index + 1), rows, "potential")
        next_h = next_h.view(batch, particles)
        log_weights = log_weights + next_h - log_h
        log_h = next_h

    probabilities, final_ess = _probabilities(log_weights)
    ess_history.append(final_ess)
    log_z += torch.logsumexp(log_weights, -1) - math.log(particles)
    costs = _draw_cost(x, probabilities, generator)
    all_ess = torch.stack(ess_history)
    nfe = rows * actor.steps
    return costs, {
        "nfe": float(nfe), "actor_nfe": float(nfe),
        "guide_calls": float(rows * actor.steps), "terminal_evaluations": float(rows),
        "ess_mean": float(all_ess.mean()), "ess_min": float(all_ess.min()),
        "ess_final": float(final_ess.mean()), "ess_final_fraction": float((final_ess / particles).mean()),
        "resampling_events": float(resampling_events), "resampled_rows": float(resampled_rows),
        "terminal_draws": float(batch), "log_normalizer_mean": float(log_z.mean()),
    }
