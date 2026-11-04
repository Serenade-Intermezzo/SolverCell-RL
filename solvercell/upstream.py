"""Adapter around the pinned, unmodified LSFlow DPMD and RFM implementation.

This is a native-code *screen reference*, not a reproduction of paper results.
The screen overrides learning rate, discount, target EMA, candidate counts,
Bellman smoothing count, integration steps, and device. Native 256-wide nets,
raw observations, reward_scale=1, tangent execution noise, weighted RFM loss,
temperature learning, and smoothed latent Bellman updates remain upstream.

For this benchmark, the upstream linear MIP maximizes c.T @ weights_p @ a,
subject to binary a and sum(a)<=budget. Sorting positive weights_p.T @ c is
an exact replacement. Ties use index order rather than the MIP solver's choice;
identical columns in this released instance have identical transition effects.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import importlib
from pathlib import Path
import sys

import numpy as np
import torch

from .geometry import decode


_VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "lsflow"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))
_dpmd = importlib.import_module("algos.dpmd_rf")
_service_module = importlib.import_module("models.rfm.service")
for _module in (_dpmd, _service_module):
    if not Path(_module.__file__).resolve().is_relative_to(_VENDOR):
        raise ImportError(f"LSFlow import resolved outside pinned vendor: {_module.__file__}")
DPMD, DPMDConfig, Experience = _dpmd.DPMD, _dpmd.DPMDConfig, _dpmd.Experience


class NativeLSFlow:
    """Native learner behind the common experiment-runner interface.

    Upstream DPMD accesses a module-global RFM service. Each adapter owns an
    instance of that original service class and temporarily binds it during
    native calls. This permits sequential checkpoint/roundtrip tests without a
    second learner overwriting the first actor. It is not thread-safe; run one
    experiment per process. No vendored file or native training method changes.

    ``act(..., stochastic=True)`` returns the clean coefficient for replay,
    while its primitive action is decoded from the perturbed coefficient. This
    matches upstream's driver, including its clean-center critic convention.
    """

    def __init__(self, cfg, env):
        self.cfg, self.device = cfg, torch.device(cfg.device)
        self.obs_dim, self.cost_dim = int(cfg.n_arms), int(cfg.n_arms)
        self.action_dim = int(cfg.n_actions)
        self.weights_p = torch.tensor(np.array(env.weights_p, copy=True), dtype=torch.float32, device=self.device)
        if self.weights_p.shape != (self.cost_dim, self.action_dim):
            raise ValueError("Native LSFlow requires weights_p with shape [n_arms, n_actions]")
        if (env.n_arms, env.n_actions, env.budget) != (cfg.n_arms, cfg.n_actions, cfg.budget):
            raise ValueError("Config and environment dimensions/budget disagree")
        self.native_config = DPMDConfig(
            lr=cfg.lr, gamma=cfg.gamma, tau=cfg.tau, reward_scale=1.0,
            flow_steps=cfg.sampling_steps, num_particles=cfg.candidates,
            M_smooth=cfg.bootstrap_candidates,
        )
        self.screen_overrides = {
            "lr": cfg.lr, "gamma": cfg.gamma, "tau": cfg.tau,
            "flow_steps": cfg.sampling_steps, "num_particles": cfg.candidates,
            "M_smooth": cfg.bootstrap_candidates, "device": str(self.device),
            "reward_scale": 1.0, "observation_normalization": "none",
            "linear_decoder": "exact_positive_topk_stable_index_ties",
        }
        self.service = type(_service_module.rfm_service)()
        self.service.device = self.device
        self.service.init(self.obs_dim, self.cost_dim, lr=cfg.lr, seed=cfg.seed, force=True)
        with self._native_service():
            self.native = DPMD(self.obs_dim, self.cost_dim, device=self.device, cfg=self.native_config)
        self.updates = self.actor_updates = self.decoder_calls = self.pretrain_updates = 0
        self.projection_seconds = 0.0

    @contextmanager
    def _native_service(self):
        previous = _dpmd.rfm_service
        _dpmd.rfm_service = self.service
        try:
            yield
        finally:
            _dpmd.rfm_service = previous

    @contextmanager
    def _generator_rng(self, generator):
        """Give native global-Torch sampling a local reproducible RNG context."""
        seed = int(torch.randint(0, 2**63 - 1, (), generator=generator, device=generator.device).item())
        devices = []
        if self.device.type == "cuda":
            devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            yield

    @torch.no_grad()
    def decode_cost(self, cost):
        """Map native N-dimensional arm coefficients to A primitive actions."""
        c = torch.as_tensor(cost, dtype=self.weights_p.dtype, device=self.device)
        if c.shape[-1] != self.cost_dim:
            raise ValueError(f"Expected native cost dimension {self.cost_dim}")
        actions = decode(c @ self.weights_p, self.cfg.budget)
        self.decoder_calls += int(actions.numel() // self.action_dim)
        return actions

    @torch.no_grad()
    def act(self, obs, generator=None, stochastic=False):
        raw_obs = np.asarray(obs, dtype=np.float32)
        if raw_obs.shape != (self.obs_dim,):
            raise ValueError(f"Expected one raw observation of shape {(self.obs_dim,)}")
        rng_context = self._generator_rng(generator) if generator is not None else nullcontext()
        with rng_context, self._native_service():
            candidates = self.native.sample_candidates(raw_obs, K=self.native_config.num_particles)
            values = self.native.score_actions(raw_obs, candidates)
            clean_cost = candidates[int(np.argmax(values))]
            executed_cost = clean_cost
            if stochastic:
                executed_cost = self.service.perturb(clean_cost, kappa=self.native_config.kappa_exec, J=1)[0, 0]
            action = self.decode_cost(executed_cost).cpu().numpy()
        return action, clean_cost.copy()

    def _experience(self, batch):
        arrays = {key: np.asarray(batch[key], dtype=np.float32) for key in (
            "obs", "action", "cost", "reward", "next_obs", "done")}
        batch_size = len(arrays["obs"])
        if batch_size < 2:
            raise ValueError("Native DPMD needs batch_size >= 2 for its unbiased Q standard deviation")
        expected = {
            "obs": (batch_size, self.obs_dim), "next_obs": (batch_size, self.obs_dim),
            "cost": (batch_size, self.cost_dim), "action": (batch_size, self.action_dim),
            "reward": (batch_size,), "done": (batch_size,),
        }
        for key, shape in expected.items():
            if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
                raise ValueError(f"Native replay {key} must be finite with shape {shape}")
        return Experience(
            arrays["obs"], arrays["cost"], arrays["reward"], arrays["next_obs"], arrays["done"],
            arrays["cost"], np.zeros(batch_size, dtype=np.int64),
        )

    @staticmethod
    def _finite(info):
        result = {key: float(value) for key, value in info.items()}
        if not all(np.isfinite(value) for value in result.values()):
            raise FloatingPointError(f"Nonfinite native LSFlow metrics: {result}")
        return result

    def pretrain(self, batch):
        """Eight native myopic steps, tracked separately from RL/actor updates.

        Supply a zero-argument replay sampler to draw a fresh batch per step,
        as in upstream. A supplied dictionary repeats that batch explicitly.
        """
        losses = []
        with self._native_service():
            for _ in range(8):
                experience = self._experience(batch() if callable(batch) else batch)
                loss = self.native.pretrain_critics_step(experience)
                self._finite({"pretrain_critic_loss": loss})
                losses.append(loss)
                self.pretrain_updates += 1
            self.service.sync_target_from_current()
        return self._finite({"pretrain_critic_loss": np.mean(losses), "pretrain_updates": self.pretrain_updates})

    def update(self, batch):
        with self._native_service():
            info = self.native.update(self._experience(batch))
        self.updates += 1
        # Native delay_update gates target EMA, not actor optimization.
        self.actor_updates += 1
        info = self._finite(info)
        info.update({"critic_loss": info["q1_loss"] + info["q2_loss"],
                     "actor_loss": info["policy_loss"], "native_log_alpha": float(self.native.log_alpha.detach())})
        return self._finite(info)

    def parameter_counts(self):
        return {"actor": sum(p.numel() for p in self.service.model.parameters()),
                "critic_total": sum(p.numel() for p in self.native.q1.parameters())
                + sum(p.numel() for p in self.native.q2.parameters())}

    def state_dict(self):
        """All trainable/EMA/optimizer/native-counter state and global RNGs."""
        return {
            "schema_version": 1, "native_config": asdict(self.native_config),
            "screen_overrides": self.screen_overrides.copy(),
            "weights_p": self.weights_p.detach().clone(),
            "actor": self.service.model.state_dict(), "target_actor": self.service.model_target.state_dict(),
            "actor_optim": self.service.optim.state_dict(),
            **{name: getattr(self.native, name).state_dict() for name in ("q1", "q2", "tq1", "tq2", "q_optim", "alpha_optim")},
            "log_alpha": self.native.log_alpha.detach().clone(),
            "q_running_mean": self.native._q_mean, "q_running_std": self.native._q_std,
            "native_step": self.native.step, "policy_version": self.native.policy_version,
            "updates": self.updates, "actor_updates": self.actor_updates,
            "pretrain_updates": self.pretrain_updates, "decoder_calls": self.decoder_calls,
            "projection_seconds": self.projection_seconds,
            "torch_rng_state": torch.random.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
            "numpy_rng_state": np.random.get_state(),
        }

    def load_state_dict(self, state):
        if state["native_config"] != asdict(self.native_config):
            raise ValueError("Cannot restore native LSFlow state with a different DPMD config")
        if not torch.equal(state["weights_p"].to(self.device), self.weights_p):
            raise ValueError("Cannot restore native LSFlow state onto a different environment decoder")
        self.service.model.load_state_dict(state["actor"])
        self.service.model_target.load_state_dict(state["target_actor"])
        self.service.optim.load_state_dict(state["actor_optim"])
        for name in ("q1", "q2", "tq1", "tq2", "q_optim", "alpha_optim"):
            getattr(self.native, name).load_state_dict(state[name])
        with torch.no_grad():
            self.native.log_alpha.copy_(state["log_alpha"].to(self.device))
        self.native._q_mean, self.native._q_std = state["q_running_mean"], state["q_running_std"]
        self.native.step, self.native.policy_version = state["native_step"], state["policy_version"]
        for name in ("updates", "actor_updates", "pretrain_updates", "decoder_calls", "projection_seconds"):
            setattr(self, name, state[name])
        torch.random.set_rng_state(state["torch_rng_state"].cpu())
        if self.device.type == "cuda" and state["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng_state"].cpu(), self.device)
        np.random.set_state(state["numpy_rng_state"])
