"""NumPy port of LSFlow's Dynamic Intervention benchmark.

The instance construction and transition law are copied from upstream's
``get_rmab_sigmoid`` / ``MultiActionRMAB`` at commit
7e50d03e7129974a11061c9ce500bc2dc5593246 (MIT). See docs/UPSTREAM_AUDIT.md.
Instance generation uses a private legacy RandomState to preserve upstream
draw order. Rollouts use an independent Generator, so evaluations do not
change the training random stream. The default A=20 is an explicit choice:
upstream's N=40, A=40, B=10 constructor fails while generating its weights.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import numpy as np


UPSTREAM_COMMIT = "7e50d03e7129974a11061c9ce500bc2dc5593246"
_ARRAY_FIELDS = (
    "state_r", "weights_p", "theta_p", "a_vals", "b_vals", "x0_vals",
    "good_arms", "bad_arms", "good_actions", "bad_actions",
    "base_transitions", "wrapper_init_state",
)


def make_intervention_instance(
    n_arms: int = 40,
    n_actions: int = 20,
    budget: int = 10,
    horizon: int = 20,
    instance_seed: int = 0,
) -> dict[str, Any]:
    """Generate the same instance parameters as the original NumPy factory.

    The unused base transitions and initial state draws are retained because
    removing them would change all subsequent random parameters.
    """
    if not (0 < budget < n_actions and budget < n_arms and horizon > 0):
        raise ValueError("Require 0 < budget < min(n_actions, n_arms), horizon > 0")
    rng = np.random.RandomState(instance_seed)
    # get_rmab_sigmoid, then multistate_rmab, then MultiStateRMAB.__init__.
    wrapper_init_state = rng.choice([0, 1], size=n_arms, p=[0.5, 0.5])
    rng.choice(n_arms, budget, replace=False)  # upstream's unused bad_arms
    base_transitions = np.full((n_arms, 2, 2), -1.0)
    base_transitions[:, 0, 0] = rng.rand(n_arms) * 0.2
    base_transitions[:, 0, 1] = rng.rand(n_arms) * 0.2 + 0.7
    state_r = np.tile([0.1, 0.15, 0.2, 1.0], (n_arms, 1))
    state_r[:budget] = [0.2, 0.15, 0.1, 1.0]
    state_r[:budget, -1] = rng.choice(np.arange(4, 7), budget, replace=True)
    state_r[budget:, -1] = rng.choice(np.arange(1, 4), n_arms - budget, replace=True)
    # TorchMultiStateRMAB copies these parameters and consumes no NumPy draws.
    good_arms = np.arange(int(np.ceil(n_arms * (budget / n_actions))), n_arms)
    good_actions = rng.choice(n_actions, budget, replace=False)
    bad_arms = np.asarray([j for j in range(n_arms) if j not in good_arms])
    bad_actions = np.asarray([i for i in range(n_actions) if i not in good_actions])
    a_vals, b_vals, x0_vals = (np.zeros(n_arms) for _ in range(3))
    for j in good_arms:
        a_vals[j] = (rng.rand(1) * 0.3 + 0.7).round(1).item()
        b_vals[j] = (rng.rand(1) * 0.5 + 1.4).round(1).item()
        x0_vals[j] = rng.randint(9, 11)
    for j in bad_arms:
        a_vals[j] = (rng.rand(1) * 0.3 + 0.2).round(1).item()
        x0_vals[j] = rng.randint(1, 3)
        b_vals[j] = 1
    weights_p = np.zeros((n_arms, n_actions))
    # Exact grouping in MultiActionRMAB.get_sigmoid_weights, including its
    # restriction on how many untouched arms must remain for each group.
    for actions, arms in ((good_actions, good_arms), (bad_actions, bad_arms)):
        for idx, action_i in enumerate(actions):
            if idx % 3 == 0 and idx <= len(actions) - 3:
                unused = [i for i in arms if weights_p[i].sum() == 0]
                try:
                    selected = rng.choice(unused, 3, replace=False)
                except ValueError as exc:
                    raise ValueError(
                        f"Upstream weight construction does not support N={n_arms}, "
                        f"A={n_actions}, B={budget}; fewer than three unused arms "
                        "remain. The supported initial instance is N=40,A=20,B=10."
                    ) from exc
                weights_p[selected, action_i] = 3
                for mm in range(3):
                    weights_p[selected, actions[idx + mm]] = 3
            elif idx >= len(actions) - (len(actions) % 3):
                unused = [i for i in arms if weights_p[i].sum() <= 1]
                try:
                    selected = rng.choice(unused, 3, replace=False)
                except ValueError as exc:
                    raise ValueError(
                        f"Upstream weight construction does not support N={n_arms}, "
                        f"A={n_actions}, B={budget}; fewer than three unused arms "
                        "remain. The supported initial instance is N=40,A=20,B=10."
                    ) from exc
                weights_p[selected, action_i] = 3
    result: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "lsflow_dynamic_intervention",
        "upstream_commit": UPSTREAM_COMMIT,
        "n_arms": int(n_arms), "n_actions": int(n_actions),
        "budget": int(budget), "horizon": int(horizon),
        "instance_seed": int(instance_seed),
        "construction_rng": "numpy.random.RandomState (upstream draw order)",
        "rollout_rng": "independent numpy.random.Generator",
        "state_r": state_r, "weights_p": weights_p,
        "theta_p": np.zeros(n_arms),
        "a_vals": a_vals, "b_vals": b_vals, "x0_vals": x0_vals,
        "good_arms": good_arms, "bad_arms": bad_arms,
        "good_actions": good_actions, "bad_actions": bad_actions,
        "base_transitions": base_transitions,
        "wrapper_init_state": wrapper_init_state,
    }
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in result.items()}


class InterventionEnv:
    """Finite-horizon intervention environment with binary budgeted actions.

    Observation: N integer-valued arm states represented as float32.
    Action: A binary entries, with sum <= B.
    Reward: sum of state rewards after the stochastic transition.
    ``clone`` shares instance values, not arrays or random streams.
    """

    def __init__(
        self, n_arms: int = 40, n_actions: int = 20, budget: int = 10,
        horizon: int = 20, instance_seed: int = 0, *,
        seed: int | None = 0, instance: dict[str, Any] | None = None,
    ) -> None:
        self._instance = copy.deepcopy(instance) if instance is not None else make_intervention_instance(
            n_arms, n_actions, budget, horizon, instance_seed
        )
        self.n_arms = int(self._instance["n_arms"])
        self.n_actions = self.action_dim = int(self._instance["n_actions"])
        self.budget = int(self._instance["budget"])
        self.horizon = int(self._instance["horizon"])
        self.instance_seed = int(self._instance["instance_seed"])
        for key in _ARRAY_FIELDS:
            dtype = np.int64 if key in {
                "good_arms", "bad_arms", "good_actions", "bad_actions", "wrapper_init_state"
            } else np.float64
            value = np.asarray(self._instance[key], dtype=dtype)
            value.setflags(write=False)
            setattr(self, key, value)
        self.n_states = int(self.state_r.shape[1])
        if self.weights_p.shape != (self.n_arms, self.n_actions):
            raise ValueError("Instance weights_p shape mismatch")
        if self.state_r.shape[0] != self.n_arms:
            raise ValueError("Instance state_r shape mismatch")
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(self.n_arms, dtype=np.float32)
        self.t = 0

    @classmethod
    def from_instance(cls, instance: dict[str, Any], *, seed: int | None = 0) -> "InterventionEnv":
        return cls(instance=instance, seed=seed)

    @property
    def instance_hash(self) -> str:
        encoded = json.dumps(self._instance, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def instance_dict(self) -> dict[str, Any]:
        """Return independent JSON-serializable instance parameters."""
        return copy.deepcopy(self._instance)

    def clone(self, seed: int | None = 0, *, copy_state: bool = False) -> "InterventionEnv":
        """Create the same task with a separate rollout RNG and fresh state.

        With copy_state=True only the state/time is copied; the RNG is still
        seeded independently. Use reset(seed=...) for paired evaluations.
        """
        env = self.from_instance(self.instance_dict(), seed=seed)
        if copy_state:
            env.state = self.state.copy()
            env.t = self.t
        return env

    def observation(self) -> np.ndarray:
        return self.state.copy()

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        # Upstream BaseRMAB.fresh_reset samples only states {0,1}, although
        # subsequent dynamics can reach {0,1,2,3}. Preserve that behavior.
        self.state = self.rng.binomial(1, 0.5, self.n_arms).astype(np.float32)
        return self.observation()

    def _action(self, action: np.ndarray) -> np.ndarray:
        result = np.asarray(action, dtype=np.float64)
        if result.shape != (self.n_actions,):
            raise ValueError(f"Expected action shape {(self.n_actions,)}, got {result.shape}")
        if not np.all(np.isfinite(result)) or not np.all((result == 0) | (result == 1)):
            raise ValueError("Action must be finite and binary")
        if result.sum() > self.budget:
            raise ValueError(f"Action exceeds budget {self.budget}")
        return result

    def transition_probabilities(self, action: np.ndarray) -> np.ndarray:
        action = self._action(action)
        values = self.theta_p + self.weights_p @ action
        intercept = self.a_vals / (1 + np.exp(self.x0_vals))
        probability = self.a_vals / (1 + np.exp(-(self.b_vals * values - self.x0_vals)))
        return 0.98 * (probability - intercept) + 0.01

    def expected_reward(self, action: np.ndarray, state: np.ndarray | None = None) -> float:
        """Exact model-based expected *one-step* reward, not future return."""
        states = self.state if state is None else np.asarray(state)
        if states.shape != (self.n_arms,) or np.any(states != np.floor(states)) or np.any(
            (states < 0) | (states >= self.n_states)
        ):
            raise ValueError("Invalid arm states")
        states = states.astype(np.int64)
        probability = self.transition_probabilities(action)
        ids = np.arange(self.n_arms)
        up = self.state_r[ids, np.minimum(states + 1, self.n_states - 1)]
        down = self.state_r[ids, np.maximum(states - 1, 0)]
        return float(np.sum(probability * up + (1 - probability) * down))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.t >= self.horizon:
            raise RuntimeError("Episode has terminated; call reset() before step()")
        probability = self.transition_probabilities(action)
        expected = self.expected_reward(action)
        up = self.rng.binomial(1, probability)
        self.state = np.clip(self.state + 2 * up - 1, 0, self.n_states - 1).astype(np.float32)
        self.t += 1
        # Upstream builds per-arm rewards in float32 before summing.
        rewards = self.state_r[np.arange(self.n_arms), self.state.astype(np.int64)].astype(np.float32)
        reward = float(np.sum(rewards, dtype=np.float32))
        return self.observation(), reward, self.t >= self.horizon, {
            "next_state_expected": expected,
            "action_count": int(np.sum(action)),
        }

    def random_action(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Uniformly sample a size-B subset, as upstream's Random policy does."""
        rng = self.rng if rng is None else rng
        action = np.zeros(self.n_actions, dtype=np.float32)
        action[rng.choice(self.n_actions, self.budget, replace=False)] = 1
        return action

    def greedy_action(self, state: np.ndarray | None = None) -> np.ndarray:
        """Model-informed myopic marginal greedy, stopping at nonpositive gain.

        This diagnostic baseline uses the known transition model. It differs
        from the paper's empirical warmup-then-greedy intervention baseline.
        It is not an exact optimizer for the nonlinear sigmoid reward.
        """
        action = np.zeros(self.n_actions, dtype=np.float32)
        current = self.expected_reward(action, state)
        for _ in range(self.budget):
            best_value, best_idx = current, None
            for idx in np.flatnonzero(action == 0):
                candidate = action.copy()
                candidate[idx] = 1
                value = self.expected_reward(candidate, state)
                if value > best_value + 1e-12:
                    best_value, best_idx = value, idx
            if best_idx is None:
                break
            action[best_idx] = 1
            current = best_value
        return action

    # Compatibility aliases for upstream-style callers.
    get_random_action = random_action
    calc_action_expected_value = expected_reward

