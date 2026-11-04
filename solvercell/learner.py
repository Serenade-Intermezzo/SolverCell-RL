from copy import deepcopy
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .geometry import decode, project_cell
from .policies import Policy


class QNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, obs, action):
        return self.net(torch.cat((obs, action), -1)).squeeze(-1)


class Replay:
    def __init__(self, capacity, obs_dim, action_dim, cost_dim=None):
        self.data = {"obs": np.zeros((capacity, obs_dim), np.float32),
                     "action": np.zeros((capacity, action_dim), np.float32),
                     "cost": np.zeros((capacity, cost_dim or action_dim), np.float32),
                     "reward": np.zeros(capacity, np.float32),
                     "next_obs": np.zeros((capacity, obs_dim), np.float32),
                     "done": np.zeros(capacity, np.float32)}
        self.capacity, self.size, self.pos = capacity, 0, 0

    def add(self, obs, action, cost, reward, next_obs, done):
        for key, value in locals().copy().items():
            if key in self.data:
                self.data[key][self.pos] = value
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, size, rng):
        indices = rng.integers(self.size, size=size)
        return {k: v[indices] for k, v in self.data.items()}

    def state_dict(self, compact=True):
        """Snapshot valid rows without changing their physical ring indices.

        Before the first wrap, only ``[:size]`` can be sampled; unused rows need
        not be serialized. Once full, all rows and the next-write position are
        required. Sampling RNG belongs to the caller and is not touched here.
        Independent CPU tensors avoid NumPy's pickle byte-encoding overhead
        while keeping this snapshot independent of later replay writes.
        """
        if not isinstance(compact, bool):
            raise ValueError("Replay compact must be a bool")
        rows = self.size if compact else self.capacity
        return {"format_version": 1, "compact": compact,
                "capacity": self.capacity, "size": self.size, "pos": self.pos,
                "data": {key: torch.from_numpy(value[:rows].copy())
                         for key, value in self.data.items()}}

    def load_state_dict(self, state):
        """Restore a versioned snapshot or the legacy ``replay.__dict__``.

        Capacity, dimensions and dtype must match this replay. Validation is
        completed before changing any state; compact unused rows are reset to
        zero exactly as in a fresh buffer. No RNG is read or advanced.
        """
        if not isinstance(state, dict):
            raise ValueError("Replay state must be a dictionary")
        if "format_version" not in state:
            if "compact" in state:
                raise ValueError("Compact replay state requires format_version")
            compact = False
        else:
            version = state["format_version"]
            if type(version) is not int or version != 1:
                raise ValueError(f"Unsupported replay format_version: {version!r}")
            compact = state.get("compact")
            if not isinstance(compact, bool):
                raise ValueError("Replay compact must be a bool")

        counters = {}
        for key in ("capacity", "size", "pos"):
            value = state.get(key)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"Replay {key} must be an integer")
            counters[key] = int(value)
        capacity, size, pos = (counters[key] for key in ("capacity", "size", "pos"))
        if capacity <= 0 or capacity != self.capacity:
            raise ValueError(f"Replay capacity mismatch: {capacity} != {self.capacity}")
        if not 0 <= size <= capacity or not 0 <= pos < capacity:
            raise ValueError("Replay size or position is out of bounds")
        if size < capacity and pos != size:
            raise ValueError("Partially filled replay must have pos == size")

        data = state.get("data")
        if not isinstance(data, dict) or data.keys() != self.data.keys():
            raise ValueError("Replay data fields do not match")
        rows = size if compact else capacity
        arrays = {}
        for key, target in self.data.items():
            value = data[key]
            expected = (rows, *target.shape[1:])
            if torch.is_tensor(value):
                if value.device.type != "cpu" or value.dtype != torch.float32 or value.layout != torch.strided:
                    raise ValueError(f"Replay {key} must be a dense float32 CPU tensor")
                value = value.detach().numpy()
            if not isinstance(value, np.ndarray) or value.shape != expected:
                raise ValueError(f"Replay {key} must have shape {expected}")
            if value.dtype != target.dtype:
                raise ValueError(f"Replay {key} dtype mismatch: {value.dtype} != {target.dtype}")
            arrays[key] = value

        restored = {key: np.zeros_like(target) for key, target in self.data.items()}
        for key, value in arrays.items():
            restored[key][:rows] = value
        self.data = restored
        self.capacity, self.size, self.pos = capacity, size, pos


class Learner:
    """Matched action-critic harness; this class is NOT the original LSFlow learner."""
    def __init__(self, cfg, env):
        self.cfg, self.device = cfg, torch.device(cfg.device)
        # Separate initialization streams keep the critics identical across actors.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.seed + 701)
            kind = "spherical_flow" if cfg.method == "sphere_actioncritic" else "diffusion"
            self.actor = Policy(cfg.n_arms, cfg.n_actions, hidden=cfg.hidden,
                                steps=cfg.sampling_steps, kind=kind,
                                bridge_sigma=cfg.bridge_sigma,
                                diffusion_scale=cfg.diffusion_scale).to(self.device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.seed + 702)
            self.q1 = QNet(cfg.n_arms, cfg.n_actions, cfg.hidden).to(self.device)
            self.q2 = QNet(cfg.n_arms, cfg.n_actions, cfg.hidden).to(self.device)
        self.target_actor = deepcopy(self.actor)
        self.tq1, self.tq2 = deepcopy(self.q1), deepcopy(self.q2)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.q_optim = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.lr)
        self.generator = torch.Generator(device=self.device).manual_seed(cfg.seed + 703)
        self.updates = self.actor_updates = self.decoder_calls = 0
        self.projection_seconds = 0.0

    def tensor_obs(self, obs):
        # Preserve upstream observations in replay; normalize only network inputs.
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device) / 3.0

    @torch.no_grad()
    def candidates(self, obs, count, target=False, generator=None):
        batch = len(obs)
        expanded = obs.repeat_interleave(count, 0)
        costs = (self.target_actor if target else self.actor).sample(expanded, generator=generator or self.generator)
        actions = decode(costs, self.cfg.budget)
        self.decoder_calls += len(costs)
        return costs.reshape(batch, count, -1), actions.reshape(batch, count, -1), expanded

    @torch.no_grad()
    def act(self, obs, generator=None, stochastic=False):
        o = self.tensor_obs(np.asarray(obs)[None])
        costs, actions, expanded = self.candidates(o, self.cfg.candidates, generator=generator)
        q = torch.minimum(self.q1(expanded, actions.flatten(0, 1)), self.q2(expanded, actions.flatten(0, 1)))
        idx = int(torch.argmax(q))
        return actions[0, idx].cpu().numpy(), costs[0, idx].cpu().numpy()

    def update(self, batch):
        cfg = self.cfg
        obs, nxt = self.tensor_obs(batch["obs"]), self.tensor_obs(batch["next_obs"])
        action = torch.as_tensor(batch["action"], device=self.device)
        rew = torch.as_tensor(batch["reward"], device=self.device) * cfg.reward_scale
        done = torch.as_tensor(batch["done"], device=self.device)
        with torch.no_grad():
            _, next_actions, expanded = self.candidates(nxt, cfg.bootstrap_candidates, target=True)
            qa = torch.minimum(self.tq1(expanded, next_actions.flatten(0, 1)), self.tq2(expanded, next_actions.flatten(0, 1)))
            # Candidate-greedy behavior; bootstrap uses the same declared selection rule.
            value = qa.reshape(len(nxt), -1).max(-1).values
            target = rew + cfg.gamma * (1 - done) * value
        self.q_optim.zero_grad(set_to_none=True)
        q1, q2 = self.q1(obs, action), self.q2(obs, action)
        critic_loss = F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)
        critic_loss.backward()
        q_grad = nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 10.0)
        self.q_optim.step()
        info = {"critic_loss": float(critic_loss.detach()), "q_mean": float(q1.detach().mean()),
                "target_mean": float(target.mean()), "critic_grad_norm": float(q_grad)}
        if self.updates % cfg.actor_interval == 0:
            with torch.no_grad():
                costs, acts, expanded = self.candidates(obs, cfg.candidates, target=True)
                values = torch.minimum(self.tq1(expanded, acts.flatten(0, 1)), self.tq2(expanded, acts.flatten(0, 1))).reshape(len(obs), -1)
                advantage = values - values.mean(-1, keepdim=True)
                weights = torch.softmax(advantage / cfg.temperature, -1)
                chosen = torch.multinomial(weights, 1, generator=self.generator).squeeze(1)
                row = torch.arange(len(obs), device=self.device)
                selected_action, old_cost = acts[row, chosen], costs[row, chosen]
                z0 = torch.randn(old_cost.shape, generator=self.generator, device=self.device)
                started = time.perf_counter()
                if cfg.method in {"cell_diffusion", "cell_independent_diffusion", "cell_flow"}:
                    projection_noise = (torch.randn(z0.shape, generator=self.generator, device=self.device)
                                        if cfg.method == "cell_independent_diffusion" else z0)
                    endpoint = project_cell(projection_noise, selected_action, cfg.budget, cfg.margin)
                    if cfg.endpoint_jitter:
                        jitter = (2 * torch.rand(endpoint.shape, generator=self.generator, device=self.device) - 1) * cfg.endpoint_jitter
                        endpoint = endpoint + jitter
                    if not torch.equal(decode(endpoint, cfg.budget), selected_action):
                        raise RuntimeError("Cell endpoint changes the target action")
                elif cfg.method == "canonical_diffusion":
                    endpoint = (selected_action * 2 - 1) * max(1.0, cfg.margin)
                elif cfg.method == "point_diffusion":
                    # Match the exact same safety margin; this target depends
                    # on the old sampled cost, not the newly sampled z0.
                    endpoint = project_cell(old_cost, selected_action, cfg.budget, cfg.margin)
                else:
                    endpoint = old_cost
                elapsed = time.perf_counter() - started
                self.projection_seconds += elapsed
                transport = (endpoint - z0).square().sum(-1).mean()
                old_transport = (old_cost - z0).square().sum(-1).mean()
                matched_point = project_cell(old_cost, selected_action, cfg.budget, cfg.margin)
                matched_transport = (matched_point - z0).square().sum(-1).mean()
                distinct = torch.stack([torch.unique(a, dim=0).shape[0] * torch.ones((), device=self.device) for a in acts]).mean()
            self.actor_optim.zero_grad(set_to_none=True)
            actor_loss, diagnostics = self.actor.loss(obs, z0, endpoint, generator=self.generator)
            actor_loss.backward()
            actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
            self.actor_optim.step()
            self.actor_updates += 1
            info.update({"actor_loss": float(actor_loss.detach()), "actor_grad_norm": float(actor_grad),
                         "endpoint_transport": float(transport), "old_point_transport": float(old_transport),
                         "margin_matched_point_transport": float(matched_transport),
                         "candidate_unique_actions": float(distinct),
                         "weight_ess": float((1 / weights.square().sum(-1)).mean()),
                         "projection_seconds": elapsed})
            info.update({str(k): float(v) for k, v in diagnostics.items()})
        with torch.no_grad():
            for target_net, online in ((self.tq1, self.q1), (self.tq2, self.q2), (self.target_actor, self.actor)):
                for target_p, p in zip(target_net.parameters(), online.parameters()):
                    target_p.lerp_(p, cfg.tau)
        self.updates += 1
        if not all(np.isfinite(v) for v in info.values()):
            raise FloatingPointError(f"Nonfinite training metrics: {info}")
        return info

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "target_actor": self.target_actor.state_dict(),
                "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
                "tq1": self.tq1.state_dict(), "tq2": self.tq2.state_dict(),
                "actor_optim": self.actor_optim.state_dict(), "q_optim": self.q_optim.state_dict(),
                "generator": self.generator.get_state(), "updates": self.updates,
                "actor_updates": self.actor_updates, "decoder_calls": self.decoder_calls,
                "projection_seconds": self.projection_seconds}

    def parameter_counts(self):
        return {"actor": sum(p.numel() for p in self.actor.parameters()),
                "critic_total": sum(p.numel() for p in self.q1.parameters()) + sum(p.numel() for p in self.q2.parameters())}

    def load_state_dict(self, state):
        for name in ("actor", "target_actor", "q1", "q2", "tq1", "tq2", "actor_optim", "q_optim"):
            getattr(self, name).load_state_dict(state[name])
        self.generator.set_state(state["generator"].cpu())
        for name in ("updates", "actor_updates", "decoder_calls", "projection_seconds"):
            setattr(self, name, state[name])


ALGORITHMS = {name: Learner for name in ("cell_diffusion", "cell_independent_diffusion", "point_diffusion", "canonical_diffusion", "cell_flow", "sphere_actioncritic")}
