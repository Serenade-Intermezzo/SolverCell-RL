from dataclasses import asdict, dataclass, replace
import hashlib
import json


NATIVE_METHODS = {"lsflow_upstream", "lsflow_tensor", "native_full_point_diffusion",
                  "native_quotient_sphere", "native_quotient_point_diffusion",
                  "native_quotient_cell_diffusion", "native_quotient_independent_diffusion",
                  "native_quotient_cell_flow"}


def is_native_method(method):
    return method in NATIVE_METHODS


@dataclass(frozen=True)
class Config:
    method: str = "cell_diffusion"
    seed: int = 0
    instance_seed: int = 0
    n_arms: int = 40
    n_actions: int = 20
    budget: int = 10
    horizon: int = 20
    train_steps: int = 4000
    warmup_steps: int = 1000
    batch_size: int = 64
    replay_capacity: int = 100000
    hidden: int = 128
    lr: float = 0.0004
    gamma: float = 0.99
    tau: float = 0.005
    actor_interval: int = 2
    candidates: int = 8
    bootstrap_candidates: int = 4
    sampling_steps: int = 12
    temperature: float = 1.0
    margin: float = 0.1
    endpoint_jitter: float = 0.0
    bridge_sigma: float = 0.5
    diffusion_scale: float = 1.0
    reward_scale: float = 0.025
    exploration_epsilon: float = 0.1
    eval_interval: int = 500
    eval_episodes: int = 20
    eval_seed_start: int = 100000
    final_eval_episodes: int = 50
    final_seed_start: int = 200000
    log_interval: int = 100
    checkpoint_interval: int = 1000
    device: str = "cpu"
    threads: int = 1
    reference_profile: str = "controlled_short_screen"

    def validate(self):
        methods = {"cell_diffusion", "cell_independent_diffusion", "point_diffusion", "canonical_diffusion", "cell_flow", "sphere_actioncritic"} | NATIVE_METHODS
        if self.method not in methods:
            raise ValueError(f"Unknown method: {self.method}")
        if not 0 < self.budget < self.n_actions:
            raise ValueError("Require 0 < budget < n_actions")
        if self.margin <= self.endpoint_jitter or self.endpoint_jitter < 0:
            raise ValueError("Endpoint jitter must be nonnegative and smaller than margin")
        if min(self.train_steps, self.warmup_steps, self.batch_size, self.sampling_steps, self.eval_interval) <= 0:
            raise ValueError("Budgets must be positive")
        if self.eval_seed_start + self.eval_episodes > self.final_seed_start:
            raise ValueError("Validation and held-out evaluation seeds overlap")
        if self.method in {"cell_flow", "native_quotient_cell_flow"} and self.diffusion_scale != 0:
            raise ValueError("cell_flow requires diffusion_scale=0")
        return self

    def to_dict(self):
        return asdict(self)

    def run_id(self):
        digest = hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:10]
        return f"{self.method}-s{self.seed}-{digest}"


def load_config(path):
    with open(path) as f:
        raw = json.load(f)
    return Config(**raw).validate()


def debug_config(config):
    return replace(config, train_steps=4, warmup_steps=8, batch_size=2,
                   candidates=2, bootstrap_candidates=2, sampling_steps=2,
                   eval_interval=4, eval_episodes=2, final_eval_episodes=2,
                   checkpoint_interval=4, log_interval=1).validate()
