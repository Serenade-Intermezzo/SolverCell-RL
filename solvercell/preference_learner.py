"""Preference-conditioned base-policy learning and inference-only guidance.

The target actor is the declared direct policy for behavior, Bellman sampling,
candidate generation, and guide suffix labels. Guidance never enters its TD
targets. Component critics condition on the relabeled preference, including at
the next state. This module is separate from all frozen historical learners.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .preference_policy import (
    JointPreferencePotential, PreferenceActor, preference_smc, terminal_weight_sample,
)


class PreferenceVectorQ(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim + action_dim + 2, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, obs, action, prefs):
        return self.net(torch.cat((obs, action, prefs), dim=-1))


def centered_guide_loss(predicted, labels, prefixes):
    """Fit relative log potentials; common within-context offsets are a gauge.

    Consecutive prefixes must share observation, preference and path index.
    Centering preserves the log-potential scale used by SMC, unlike ranking
    losses which identify an ordering but not the required density ratios.
    """
    if prefixes < 2 or predicted.shape != labels.shape or not predicted.numel() or predicted.numel() % prefixes:
        raise ValueError('Guide predictions/labels must form complete prefix groups')
    prediction = predicted.reshape(-1, prefixes)
    target = labels.detach().reshape(-1, prefixes)
    return F.mse_loss(prediction - prediction.mean(1, keepdim=True),
                      target - target.mean(1, keepdim=True))


class PreferenceLearner:
    """Fixed preference per continuation; guide fitting changes no base targets.

    ``update`` takes encoded obs/next_obs, action, original-unit two-component
    rewards, and done. Stored behavior preferences, if present, are deliberately
    ignored when relabeling utility: physical dynamics do not depend on utility.
    ``act`` samples the target base policy, a terminal tilt, or SMC. Callers use
    their own Generator for evaluation; no training RNG is then advanced.
    """

    COUNTER_KEYS = ("nfe", "actor_sampling_nfe", "actor_training_nfe", "guide_calls", "q_calls", "decoder_calls")

    def __init__(self, cfg, task):
        self.cfg, self.task = cfg, task
        self.device = torch.device(cfg.device)
        self.obs_dim, self.cost_dim = int(task.obs_dim), int(task.cost_dim)
        self.action_dim = int(task.action_dim)
        kinds = {"preference_diffusion": "diffusion", "preference_sphere": "sphere"}
        if cfg.method not in kinds:
            raise ValueError(f"Unsupported preference method: {cfg.method}")
        self.kind = kinds[cfg.method]
        if cfg.sampling_steps < 2 or (self.kind == "sphere" and cfg.sampling_steps % 2):
            raise ValueError("Sampling budget must be >=2 and even for the Heun sphere actor")
        if cfg.temperature <= 0 or cfg.guide_temperature <= 0 or cfg.reward_scale <= 0:
            raise ValueError("Temperatures and reward scale must be positive")
        if min(cfg.candidates, cfg.bootstrap_candidates, cfg.actor_interval,
               cfg.guide_interval, cfg.guide_batch_size, cfg.guide_suffixes, cfg.guide_particles) <= 0:
            raise ValueError("Candidate, update and guide budgets must be positive")
        preference_grid = tuple(float(value) for value in cfg.train_preferences)
        if not preference_grid or any(not math.isfinite(v) or not 0 <= v <= 1 for v in preference_grid):
            raise ValueError("train_preferences must be a nonempty finite grid in [0,1]")
        self.preference_grid = torch.tensor(preference_grid, device=self.device, dtype=torch.float32)

        # Construct on CPU within isolated forks before moving devices. Actor
        # architecture/draw counts cannot affect critic initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.seed + 701)
            self.actor = PreferenceActor(self.obs_dim, self.cost_dim, hidden=cfg.hidden,
                                         steps=cfg.sampling_steps, kind=self.kind,
                                         beta_end=cfg.diffusion_beta_end, final_noise=cfg.diffusion_final_noise,
                                         prediction_type=cfg.diffusion_prediction).to(self.device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.seed + 702)
            self.q1 = PreferenceVectorQ(self.obs_dim, self.action_dim, cfg.hidden).to(self.device)
            self.q2 = PreferenceVectorQ(self.obs_dim, self.action_dim, cfg.hidden).to(self.device)
        self.target_actor = deepcopy(self.actor).requires_grad_(False)
        self.tq1, self.tq2 = deepcopy(self.q1).requires_grad_(False), deepcopy(self.q2).requires_grad_(False)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.q_optim = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.lr)
        self.guide = None
        self.guide_optim = None
        if self.kind == "diffusion":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(cfg.seed + 707)
                self.guide = JointPreferencePotential(self.obs_dim, self.cost_dim,
                                                      cfg.sampling_steps, hidden=cfg.hidden).to(self.device)
            self.guide_optim = torch.optim.Adam(self.guide.parameters(), lr=cfg.lr)
        self.generator = torch.Generator(device=self.device).manual_seed(cfg.seed + 703)
        self.preference_generator = torch.Generator(device=self.device).manual_seed(cfg.seed + 704)
        self.guide_generator = torch.Generator(device=self.device).manual_seed(cfg.seed + 705)
        self.behavior_generator = torch.Generator(device=self.device).manual_seed(cfg.seed + 706)
        self.updates = self.actor_updates = self.guide_updates = 0
        self.counters = {key: 0 for key in self.COUNTER_KEYS}

    @property
    def decoder_calls(self):
        return self.counters["decoder_calls"]

    def _tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=torch.float32)

    def _preferences(self, prefs, batch):
        prefs = self._tensor(prefs)
        if prefs.ndim == 1:
            prefs = prefs[None].expand(batch, -1)
        if prefs.shape != (batch, 2) or not torch.isfinite(prefs).all():
            raise ValueError("Preferences must have shape [B,2] and finite values")
        if torch.any(prefs < 0) or not torch.allclose(prefs.sum(-1), prefs.new_ones(batch), atol=1e-6, rtol=0):
            raise ValueError("Preferences must be nonnegative and sum to one")
        return prefs

    def sample_preferences(self, batch):
        indices = torch.randint(len(self.preference_grid), (batch,), device=self.device,
                                generator=self.preference_generator)
        weight = self.preference_grid[indices]
        return torch.stack((weight, 1 - weight), dim=-1)

    def _batch(self, batch):
        result = {key: self._tensor(batch[key])
                  for key in ("obs", "action", "reward_components", "next_obs", "done")}
        size = result["obs"].shape[0]
        expected = {"obs": (size, self.obs_dim), "next_obs": (size, self.obs_dim),
                    "action": (size, self.action_dim), "reward_components": (size, 2), "done": (size,)}
        if size == 0:
            raise ValueError("Empty training batches are unsupported")
        for key, shape in expected.items():
            if result[key].shape != shape or not torch.isfinite(result[key]).all():
                raise ValueError(f"Invalid finite batch field {key}; expected shape {shape}")
        if not torch.all((result["done"] == 0) | (result["done"] == 1)):
            raise ValueError("done must be binary")
        return result

    def _record_actor(self, rows, training=False):
        amount = int(rows) if training else int(rows) * self.cfg.sampling_steps
        key = "actor_training_nfe" if training else "actor_sampling_nfe"
        self.counters[key] += amount
        self.counters["nfe"] += amount

    @torch.no_grad()
    def _decode(self, costs):
        if costs.ndim != 2 or costs.shape[-1] != self.cost_dim or not torch.isfinite(costs).all():
            raise ValueError("Decoder costs must be finite [B,cost_dim]")
        # Actual deployment normalizes the terminal raw diffusion coefficient.
        # Use precisely that terminal action in guide labels and weighting too.
        costs = F.normalize(costs, dim=-1)
        actions = self.task.decode(costs).to(device=self.device, dtype=torch.float32)
        if actions.shape != (len(costs), self.action_dim):
            raise ValueError("Task decoder returned the wrong shape")
        self.counters["decoder_calls"] += len(costs)
        return actions

    def _twin_vector(self, obs, actions, prefs, target=True):
        q1, q2 = (self.tq1, self.tq2) if target else (self.q1, self.q2)
        first, second = q1(obs, actions, prefs), q2(obs, actions, prefs)
        self.counters["q_calls"] += 2 * len(obs)
        # Select a complete critic vector by scalarized utility. Componentwise
        # minima would fabricate a vector produced by neither critic.
        select_first = (first * prefs).sum(-1) <= (second * prefs).sum(-1)
        return torch.where(select_first[:, None], first, second)

    @torch.no_grad()
    def _sample(self, obs, prefs, generator):
        costs = self.target_actor.sample(obs, prefs, generator=generator)
        self._record_actor(len(obs))
        return costs

    @torch.no_grad()
    def critic_targets(self, batch, prefs):
        batch = self._batch(batch)
        prefs = self._preferences(prefs, len(batch["obs"]))
        targets = batch["reward_components"] / self._tensor(self.cfg.component_scales) * self.cfg.reward_scale
        active = batch["done"] == 0
        if active.any():
            count = self.cfg.bootstrap_candidates
            nxt = batch["next_obs"][active].repeat_interleave(count, dim=0)
            same_prefs = prefs[active].repeat_interleave(count, dim=0)
            costs = self._sample(nxt, same_prefs, self.generator)
            values = self._twin_vector(nxt, self._decode(costs), same_prefs)
            # More than one draw is a Monte Carlo mean for the same DIRECT
            # policy, never candidate max or guide-enhanced bootstrap.
            targets[active] += self.cfg.gamma * values.reshape(-1, count, 2).mean(1)
        return targets

    @torch.no_grad()
    def guide_training_targets(self, obs, prefs, generator=None, path_indices=None):
        """Fresh actual target-actor paths and conditional suffix log-mean-exp.

        The suffix index denotes the stored path position, not a forward-noise
        level applied to a replay action. Returns raw prefixes, indices, detached
        labels, and numeric diagnostics. This is a finite-R log-mean-exp target
        with Monte Carlo/log bias, not an exact conditional log partition.
        """
        if self.kind != "diffusion":
            raise ValueError("Guide suffix labels require the diffusion actor")
        generator = self.guide_generator if generator is None else generator
        obs = self._tensor(obs)
        prefs = self._preferences(prefs, len(obs))
        steps, suffixes, size = self.cfg.sampling_steps, self.cfg.guide_suffixes, len(obs)
        path = self.target_actor.sample_path(obs, prefs, generator=generator)
        if path.shape != (size, steps + 1, self.cost_dim):
            raise ValueError("Diffusion path has incompatible dimensions")
        self._record_actor(size)
        indices = (torch.randint(steps, (size,), generator=generator, device=self.device)
                   if path_indices is None else torch.as_tensor(path_indices, device=self.device, dtype=torch.long))
        if indices.shape != (size,) or torch.any(indices < 0) or torch.any(indices >= steps):
            raise ValueError('Invalid guide path indices')
        prefix = path[torch.arange(size, device=self.device), indices].clone()
        zero_actions = obs.new_zeros((size, self.action_dim))
        baseline = (self._twin_vector(obs, zero_actions, prefs) * prefs).sum(-1)
        labels = obs.new_empty(size)
        terminal_values = []
        suffix_nfe = 0
        for index in torch.unique(indices, sorted=True).tolist():
            selected = torch.nonzero(indices == index, as_tuple=False).flatten()
            repeated_obs = obs[selected].repeat_interleave(suffixes, dim=0)
            repeated_prefs = prefs[selected].repeat_interleave(suffixes, dim=0)
            repeated_prefix = prefix[selected].repeat_interleave(suffixes, dim=0)
            terminal = self.target_actor.suffix(repeated_obs, repeated_prefs, repeated_prefix,
                                                index, generator=generator)
            rows = len(repeated_obs)
            nfe = rows * (steps - index)
            self.counters["nfe"] += nfe
            self.counters["actor_sampling_nfe"] += nfe
            suffix_nfe += nfe
            utility = (self._twin_vector(repeated_obs, self._decode(terminal), repeated_prefs)
                       * repeated_prefs).sum(-1).reshape(-1, suffixes)
            logits = (utility - baseline[selected, None]) / self.cfg.guide_temperature
            labels[selected] = torch.logsumexp(logits, dim=1) - math.log(suffixes)
            terminal_values.append(utility.flatten())
        if not torch.isfinite(labels).all():
            raise FloatingPointError("Nonfinite guide log-mean-exp labels")
        return prefix, indices, labels, {
            "guide_label_mean": float(labels.mean()),
            "guide_label_std": float(labels.std(unbiased=False)),
            "guide_baseline_mean": float(baseline.mean()),
            "guide_terminal_utility_mean": float(torch.cat(terminal_values).mean()),
            "guide_label_suffix_nfe": int(suffix_nfe),
            "guide_label_path_nfe": size * steps,
        }

    def update(self, batch):
        batch = self._batch(batch)
        cfg, size = self.cfg, len(batch["obs"])
        obs, actions = batch["obs"], batch["action"]
        prefs = self.sample_preferences(size)
        targets = self.critic_targets(batch, prefs)
        self.q_optim.zero_grad(set_to_none=True)
        first, second = self.q1(obs, actions, prefs), self.q2(obs, actions, prefs)
        self.counters["q_calls"] += 2 * size
        critic_loss = F.smooth_l1_loss(first, targets) + F.smooth_l1_loss(second, targets)
        critic_loss.backward()
        q_grad = nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 10.0)
        self.q_optim.step()
        info = {"critic_loss": float(critic_loss.detach()), "critic_grad_norm": float(q_grad),
                "target_component0": float(targets[:, 0].mean()),
                "target_component1": float(targets[:, 1].mean()),
                "training_preference_mean": float(prefs[:, 0].mean())}
        if self.updates % cfg.actor_interval == 0:
            with torch.no_grad():
                expanded_obs = obs.repeat_interleave(cfg.candidates, dim=0)
                expanded_prefs = prefs.repeat_interleave(cfg.candidates, dim=0)
                old_costs = self._sample(expanded_obs, expanded_prefs, self.generator)
                candidates = self._decode(old_costs)
                utility = (self._twin_vector(expanded_obs, candidates, expanded_prefs)
                           * expanded_prefs).sum(-1).reshape(size, cfg.candidates)
                weights = torch.softmax((utility - utility.mean(1, keepdim=True)) / cfg.temperature, dim=1)
            self.actor_optim.zero_grad(set_to_none=True)
            actor_loss, diagnostics = self.actor.loss(expanded_obs, expanded_prefs, old_costs,
                                                      weights=weights.flatten(), generator=self.generator)
            self._record_actor(size * cfg.candidates, training=True)
            actor_loss.backward()
            actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
            self.actor_optim.step()
            self.actor_updates += 1
            info.update({"actor_loss": float(actor_loss.detach()), "actor_grad_norm": float(actor_grad),
                         "actor_weight_ess": float(weights.square().sum(1).reciprocal().mean()),
                         "actor_candidate_utility_mean": float(utility.mean())})
            info.update({f"actor_{key}": float(value) for key, value in diagnostics.items()})
        with torch.no_grad():
            for target, online in ((self.tq1, self.q1), (self.tq2, self.q2), (self.target_actor, self.actor)):
                for target_parameter, parameter in zip(target.parameters(), online.parameters()):
                    target_parameter.lerp_(parameter, cfg.tau)
        self.updates += 1
        if self.guide is not None and self.updates % cfg.guide_interval == 0:
            count = min(size, cfg.guide_batch_size)
            if cfg.guide_objective != 'absolute':
                count = (count // cfg.guide_prefixes) * cfg.guide_prefixes
                if count == 0:
                    raise ValueError('No complete guide prefix group')
            indices = torch.randperm(size, device=self.device, generator=self.guide_generator)[:count]
            guide_obs, guide_prefs = obs[indices], prefs[indices]
            fixed_indices = None
            if cfg.guide_objective != 'absolute':
                groups = count // cfg.guide_prefixes
                guide_obs = guide_obs[:groups].repeat_interleave(cfg.guide_prefixes, 0)
                guide_prefs = guide_prefs[:groups].repeat_interleave(cfg.guide_prefixes, 0)
                fixed_indices = torch.randint(cfg.sampling_steps, (groups,), device=self.device,
                                              generator=self.guide_generator).repeat_interleave(cfg.guide_prefixes)
            prefix, path_index, labels, diagnostics = self.guide_training_targets(
                guide_obs, guide_prefs, path_indices=fixed_indices)
            self.guide_optim.zero_grad(set_to_none=True)
            predicted = self.guide(guide_obs, guide_prefs, prefix, path_index)
            self.counters["guide_calls"] += count
            guide_loss = (centered_guide_loss(predicted, labels, cfg.guide_prefixes)
                          if cfg.guide_objective == 'centered' else F.mse_loss(predicted, labels))
            guide_loss.backward()
            guide_grad = nn.utils.clip_grad_norm_(self.guide.parameters(), 5.0)
            self.guide_optim.step()
            self.guide_updates += 1
            info.update({"guide_loss": float(guide_loss.detach()), "guide_grad_norm": float(guide_grad)})
            info.update(diagnostics)
        if not all(math.isfinite(value) for value in info.values()):
            raise FloatingPointError(f"Nonfinite preference training diagnostics: {info}")
        return info

    @torch.no_grad()
    def act(self, obs, pref, mode="direct", generator=None):
        """Return one action, its unit cost and exact per-call forward-row counts."""
        generator = self.behavior_generator if generator is None else generator
        obs = self._tensor(obs)
        if obs.shape != (self.obs_dim,) or not torch.isfinite(obs).all():
            raise ValueError(f"act observation must be finite shape ({self.obs_dim},)")
        obs = obs[None]
        prefs = self._preferences(pref, 1)
        before = self.counters.copy()
        diagnostics = {}
        if mode == "direct":
            costs = self._sample(obs, prefs, generator)
        elif mode in {"terminal_weighting", "smc"}:
            if mode == "smc" and self.guide is None:
                raise ValueError("SMC guidance is defined only for the diffusion actor")
            baseline = (self._twin_vector(obs, obs.new_zeros((1, self.action_dim)), prefs) * prefs).sum(-1)

            def terminal_logweight(raw_costs):
                rows = len(raw_costs)
                repeated_obs, repeated_prefs = obs.expand(rows, -1), prefs.expand(rows, -1)
                utility = (self._twin_vector(repeated_obs, self._decode(raw_costs), repeated_prefs)
                           * repeated_prefs).sum(-1)
                return (utility - baseline.expand(rows)) / self.cfg.guide_temperature

            if mode == "terminal_weighting":
                costs, diagnostics = terminal_weight_sample(
                    self.target_actor, obs, prefs, terminal_logweight,
                    particles=self.cfg.guide_particles, generator=generator)
            else:
                costs, diagnostics = preference_smc(
                    self.target_actor, self.guide, obs, prefs, terminal_logweight,
                    particles=self.cfg.guide_particles, generator=generator,
                    ess_fraction=self.cfg.guide_ess_fraction)
            nfe = int(diagnostics["nfe"])
            self.counters["nfe"] += nfe
            self.counters["actor_sampling_nfe"] += nfe
            self.counters["guide_calls"] += int(diagnostics.get("guide_calls", 0))
        else:
            raise ValueError(f"Unknown inference mode: {mode}")
        actions = self._decode(costs)
        stats = {str(key): float(value) for key, value in diagnostics.items()
                 if isinstance(value, (int, float, np.number))}
        stats.update({key: self.counters[key] - before[key] for key in self.COUNTER_KEYS})
        stats["mode"] = mode
        return actions[0].cpu().numpy().copy(), costs[0].cpu().numpy().copy(), stats

    def parameter_counts(self):
        return {"actor": sum(p.numel() for p in self.actor.parameters()),
                "critic_total": sum(p.numel() for net in (self.q1, self.q2) for p in net.parameters()),
                "guide": 0 if self.guide is None else sum(p.numel() for p in self.guide.parameters())}

    def _spec(self):
        return {"method": self.cfg.method, "obs_dim": self.obs_dim, "cost_dim": self.cost_dim,
                "action_dim": self.action_dim, "sampling_steps": self.cfg.sampling_steps,
                "train_preferences": self.preference_grid.detach().cpu().tolist(),
                "diffusion_beta_end": self.cfg.diffusion_beta_end,
                "diffusion_final_noise":self.cfg.diffusion_final_noise,"diffusion_prediction":self.cfg.diffusion_prediction,
                "component_scales":list(self.cfg.component_scales),
                "guide_objective": self.cfg.guide_objective, "guide_prefixes": self.cfg.guide_prefixes}

    def state_dict(self):
        return {"format_version": 2, "spec": self._spec(),
                **{name: getattr(self, name).state_dict() for name in
                   ("actor", "target_actor", "q1", "q2", "tq1", "tq2", "actor_optim", "q_optim")},
                "guide": None if self.guide is None else self.guide.state_dict(),
                "guide_optim": None if self.guide_optim is None else self.guide_optim.state_dict(),
                **{name: getattr(self, name).get_state() for name in
                   ("generator", "preference_generator", "guide_generator", "behavior_generator")},
                "updates": self.updates, "actor_updates": self.actor_updates,
                "guide_updates": self.guide_updates, "counters": self.counters.copy()}

    def load_state_dict(self, state):
        spec = dict(state.get('spec', {}))
        if state.get('format_version') == 1:
            for key, value in {'diffusion_beta_end': .2, 'guide_objective': 'absolute', 'guide_prefixes': 4}.items():
                spec.setdefault(key, value)
        for key,value in {'diffusion_final_noise':True,'diffusion_prediction':'epsilon','component_scales':[1.,1.]}.items():
            spec.setdefault(key,value)
        if state.get("format_version") not in (1, 2) or spec != self._spec():
            raise ValueError("Incompatible preference learner checkpoint specification")
        if set(state["counters"]) != set(self.COUNTER_KEYS):
            raise ValueError("Incompatible checkpoint counters")
        for name in ("actor", "target_actor", "q1", "q2", "tq1", "tq2", "actor_optim", "q_optim"):
            getattr(self, name).load_state_dict(state[name])
        if self.guide is not None:
            self.guide.load_state_dict(state["guide"])
            self.guide_optim.load_state_dict(state["guide_optim"])
        elif state["guide"] is not None or state["guide_optim"] is not None:
            raise ValueError("Sphere checkpoint cannot contain a diffusion guide")
        for name in ("generator", "preference_generator", "guide_generator", "behavior_generator"):
            getattr(self, name).set_state(state[name].cpu())
        self.updates, self.actor_updates, self.guide_updates = (
            int(state[name]) for name in ("updates", "actor_updates", "guide_updates"))
        self.counters = {key: int(value) for key, value in state["counters"].items()}
