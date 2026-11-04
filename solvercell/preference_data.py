"""Preference-task observations, original-unit rewards, and isolated replay.

The two objectives split the released intervention reward across its synthetic
low/high response cohorts. They do not change dynamics, feasibility, or the
native quotient decoder. The old scalar-reward Replay is deliberately separate.
"""
from __future__ import annotations

import numbers

import numpy as np
import torch
from torch import Tensor

from .environment import InterventionEnv
from .native_geometry import NativeCellProjector


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _array(value, name, shape):
    if torch.is_tensor(value):
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise ValueError(f"{name} must be a dense CPU value")
        value = value.detach().numpy()
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be numeric with shape {shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = value.astype(np.float32, copy=True)
    if not np.isfinite(converted).all():
        raise ValueError(f"{name} must be finite and representable as float32")
    return converted


class PreferenceTask:
    """Two-cohort Intervention task with remaining time in its observation.

    ``components`` returns N times each cohort's mean state reward, in the
    original reward units. Equal preferences therefore recover the original
    scalar reward for the required equal-size cohort partition. The caller
    applies any configured reward scaling only after forming these components.
    """

    def __init__(self, env: InterventionEnv):
        if not isinstance(env, InterventionEnv):
            raise TypeError("PreferenceTask requires an InterventionEnv")
        self.env = env
        self.n_arms, self.n_actions = env.n_arms, env.n_actions
        self.action_dim, self.obs_dim = env.n_actions, env.n_arms + 1
        self.horizon, self.budget = _integer(env.horizon, "horizon"), env.budget
        if env.n_states != 4:
            raise ValueError("PreferenceTask requires the released four-state task")
        self.groups = tuple(np.array(group, dtype=np.int64, copy=True)
                            for group in (env.bad_arms, env.good_arms))
        if any(group.ndim != 1 or len(group) == 0 for group in self.groups):
            raise ValueError("response cohorts must be nonempty index vectors")
        if len(self.groups[0]) != len(self.groups[1]):
            raise ValueError("equal-size response cohorts are required for equal-weight reward parity")
        joined = np.concatenate(self.groups)
        if not np.array_equal(np.sort(joined), np.arange(env.n_arms)):
            raise ValueError("response cohorts must partition all arms exactly once")
        for group in self.groups:
            group.setflags(write=False)
        self.projector = NativeCellProjector(torch.tensor(np.array(env.weights_p), dtype=torch.float32), env.budget)
        self.cost_dim = self.projector.n_groups
        self.basis = self.projector.basis
        self._projectors = {torch.device("cpu"): self.projector}

    def _state(self, state):
        result = _array(state, "state", (self.n_arms,))
        if np.any(result != np.floor(result)) or np.any((result < 0) | (result > 3)):
            raise ValueError("state entries must be integers in [0, 3]")
        return result.astype(np.int64)

    def encode(self, state, remaining_steps) -> np.ndarray:
        state = self._state(state)
        remaining = _integer(remaining_steps, "remaining_steps", minimum=0)
        if remaining > self.horizon:
            raise ValueError("remaining_steps exceeds the task horizon")
        result = np.empty(self.obs_dim, dtype=np.float32)
        result[:-1] = state / 3.0
        result[-1] = remaining / self.horizon
        return result

    def _components_from_arm_rewards(self, rewards) -> np.ndarray:
        return np.asarray([self.n_arms * np.mean(rewards[group], dtype=np.float64)
                           for group in self.groups], dtype=np.float32)

    def components(self, next_state) -> np.ndarray:
        states = self._state(next_state)
        rewards = self.env.state_r[np.arange(self.n_arms), states]
        return self._components_from_arm_rewards(rewards)

    def expected_components(self, action, state) -> np.ndarray:
        """Exact known-model *one-step* component means; no future value claim."""
        states = self._state(state)
        probabilities = self.env.transition_probabilities(action)
        ids = np.arange(self.n_arms)
        up = self.env.state_r[ids, np.minimum(states + 1, 3)]
        down = self.env.state_r[ids, np.maximum(states - 1, 0)]
        return self._components_from_arm_rewards(probabilities * up + (1-probabilities) * down)

    @torch.no_grad()
    def decode(self, costs: Tensor) -> Tensor:
        """Lift quotient coefficients and use the original canonical decoder.

        Preserve batch prefix, floating dtype and device. Input coefficients,
        environment state, and environment RNG are never modified.
        """
        if not torch.is_tensor(costs) or not costs.is_floating_point() or costs.ndim < 1:
            raise ValueError("costs must be a floating-point tensor")
        if costs.shape[-1] != self.cost_dim or not torch.isfinite(costs).all():
            raise ValueError(f"costs must be finite with final dimension {self.cost_dim}")
        projector = self._projectors.get(costs.device)
        if projector is None:
            projector = NativeCellProjector(self.projector.weights.to(costs.device), self.budget)
            self._projectors[costs.device] = projector
        lifted = costs @ projector.basis.to(dtype=costs.dtype).T
        return projector.decode(lifted)

    def decode_numpy(self, costs) -> np.ndarray:
        costs = np.asarray(costs)
        if costs.dtype.kind != "f" or costs.ndim < 1:
            raise ValueError("costs must be a floating-point array")
        # Copy avoids warnings/aliasing with read-only NumPy coefficient arrays.
        return self.decode(torch.from_numpy(costs.copy())).numpy()


class PreferenceReplay:
    """A float32 ring buffer; sampling RNG remains owned by the caller."""

    def __init__(self, capacity, obs_dim, action_dim, cost_dim):
        self.capacity = _integer(capacity, "capacity")
        self.obs_dim = _integer(obs_dim, "obs_dim")
        self.action_dim = _integer(action_dim, "action_dim")
        self.cost_dim = _integer(cost_dim, "cost_dim")
        self.size, self.pos = 0, 0
        shapes = {"obs": (self.obs_dim,), "action": (self.action_dim,), "cost": (self.cost_dim,),
                  "reward_components": (2,), "next_obs": (self.obs_dim,), "done": (),
                  "behavior_preference": (2,)}
        self.data = {key: np.zeros((self.capacity, *shape), dtype=np.float32) for key, shape in shapes.items()}

    @staticmethod
    def _validate_semantics(data):
        for key in ("action", "done"):
            if not np.all((data[key] == 0) | (data[key] == 1)):
                raise ValueError(f"{key} must be binary")
        preference = data["behavior_preference"]
        if (np.any((preference < 0) | (preference > 1))
                or not np.allclose(preference.sum(axis=-1), 1., rtol=0., atol=1e-6)):
            raise ValueError("behavior_preference must lie on the two-objective simplex")

    def add(self, obs, action, cost, reward_components, next_obs, done, behavior_preference):
        inputs = dict(obs=obs, action=action, cost=cost, reward_components=reward_components,
                      next_obs=next_obs, done=done, behavior_preference=behavior_preference)
        values = {key: _array(inputs[key], key, target.shape[1:]) for key, target in self.data.items()}
        self._validate_semantics(values)
        # Complete validation before mutating any field or counter.
        for key, value in values.items():
            self.data[key][self.pos] = value
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, size, rng) -> dict[str, np.ndarray]:
        size = _integer(size, "sample size")
        if self.size == 0:
            raise ValueError("cannot sample an empty preference replay")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("sample requires a caller-owned numpy.random.Generator")
        indices = rng.integers(self.size, size=size)
        return {key: value[indices] for key, value in self.data.items()}

    def state_dict(self, compact=True):
        if not isinstance(compact, bool):
            raise ValueError("compact must be a bool")
        rows = self.size if compact else self.capacity
        return {"format_version": 1, "compact": compact,
                "capacity": self.capacity, "size": self.size, "pos": self.pos,
                "obs_dim": self.obs_dim, "action_dim": self.action_dim, "cost_dim": self.cost_dim,
                "data": {key: torch.from_numpy(value[:rows].copy()) for key, value in self.data.items()}}

    def load_state_dict(self, state):
        """Atomically restore physical ring rows from dense CPU tensors/NumPy.

        No reordering, sampling or RNG use occurs. Unused rows of a compact
        snapshot are zeroed; snapshots never alias this buffer's storage.
        """
        if not isinstance(state, dict) or type(state.get("format_version")) is not int or state["format_version"] != 1:
            raise ValueError("unsupported preference replay format_version")
        if not isinstance(state.get("compact"), bool):
            raise ValueError("snapshot compact must be a bool")
        for key in ("capacity", "obs_dim", "action_dim", "cost_dim"):
            if _integer(state.get(key), key) != getattr(self, key):
                raise ValueError(f"snapshot {key} mismatch")
        size = _integer(state.get("size"), "size", minimum=0)
        pos = _integer(state.get("pos"), "pos", minimum=0)
        if size > self.capacity or pos >= self.capacity or (size < self.capacity and pos != size):
            raise ValueError("snapshot ring counters are inconsistent")
        data = state.get("data")
        if not isinstance(data, dict) or data.keys() != self.data.keys():
            raise ValueError("snapshot data fields do not match")
        rows = size if state["compact"] else self.capacity
        arrays = {}
        for key, target in self.data.items():
            value = data[key]
            if torch.is_tensor(value):
                if value.device.type != "cpu" or value.dtype != torch.float32 or value.layout != torch.strided:
                    raise ValueError(f"snapshot {key} must be a dense float32 CPU tensor")
                value = value.detach().numpy()
            if (not isinstance(value, np.ndarray) or value.dtype != np.float32
                    or value.shape != (rows, *target.shape[1:]) or not np.isfinite(value).all()):
                raise ValueError(f"snapshot {key} has invalid shape, dtype, or finite values")
            arrays[key] = value
        self._validate_semantics({key: value[:size] for key, value in arrays.items()})
        restored = {key: np.zeros_like(target) for key, target in self.data.items()}
        for key, value in arrays.items():
            restored[key][:rows] = value
        self.data, self.size, self.pos = restored, size, pos
