"""Separate, versioned configuration for the preference benchmark extension."""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class PreferenceConfig:
    method: str = 'preference_diffusion'
    seed: int = 0
    instance_seed: int = 0
    n_arms: int = 40
    n_actions: int = 20
    budget: int = 10
    horizon: int = 20
    train_steps: int = 1000
    warmup_steps: int = 256
    batch_size: int = 64
    replay_capacity: int = 100000
    hidden: int = 128
    lr: float = .0004
    gamma: float = .99
    tau: float = .005
    actor_interval: int = 1
    candidates: int = 8
    bootstrap_candidates: int = 1
    sampling_steps: int = 8
    diffusion_beta_end: float = .20
    diffusion_final_noise: bool = True
    diffusion_prediction: str = 'epsilon'
    component_scales: tuple = (1., 1.)
    temperature: float = .5
    reward_scale: float = .025
    guide_interval: int = 4
    guide_batch_size: int = 16
    guide_suffixes: int = 4
    guide_particles: int = 8
    guide_ess_fraction: float = .5
    guide_temperature: float = .5
    guide_objective: str = 'absolute'
    guide_prefixes: int = 4
    train_preferences: tuple = (0., .25, .5, .75, 1.)
    validation_preferences: tuple = (0., .5, 1.)
    heldout_preferences: tuple = (.125, .375, .625, .875)
    eval_interval: int = 250
    eval_episodes: int = 4
    final_eval_episodes: int = 8
    eval_seed_start: int = 100000
    final_seed_start: int = 200000
    checkpoint_interval: int = 250
    log_interval: int = 25
    device: str = 'cpu'
    threads: int = 1
    profile: str = 'preference_intervention_prototype_v1'

    def validate(self):
        if self.method not in {'preference_diffusion', 'preference_sphere'}:
            raise ValueError('Unknown preference method')
        if (self.n_arms, self.n_actions, self.budget) != (40, 20, 10):
            raise ValueError('This audited preference extension fixes N40/A20/B10')
        positive = ('horizon', 'train_steps', 'warmup_steps', 'batch_size', 'replay_capacity',
                    'hidden', 'actor_interval', 'candidates', 'bootstrap_candidates',
                    'sampling_steps', 'guide_interval', 'guide_batch_size', 'guide_suffixes',
                    'guide_particles', 'eval_interval', 'eval_episodes', 'final_eval_episodes',
                    'checkpoint_interval', 'log_interval', 'threads')
        for name in positive:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + ' must be a positive integer')
        if self.sampling_steps % 2:
            raise ValueError('Use even NFE: spherical Heun takes NFE/2 steps')
        if not math.isfinite(self.diffusion_beta_end) or not .02 <= self.diffusion_beta_end < 1:
            raise ValueError('diffusion_beta_end must lie in [.02,1)')
        if type(self.diffusion_final_noise) is not bool or self.diffusion_prediction not in {'epsilon','x0'}:
            raise ValueError('Invalid diffusion endpoint or prediction type')
        if len(self.component_scales)!=2 or any(not math.isfinite(v) or v<=0 for v in self.component_scales):
            raise ValueError('Two positive finite component scales required')
        if self.guide_objective not in {'absolute', 'grouped_absolute', 'centered'}:
            raise ValueError('Unknown guide objective')
        if type(self.guide_prefixes) is not int or self.guide_prefixes < 2:
            raise ValueError('guide_prefixes must be an integer >=2')
        if self.guide_objective != 'absolute' and min(self.batch_size, self.guide_batch_size) < self.guide_prefixes:
            raise ValueError('Centered guide requires at least one complete prefix group')
        if self.replay_capacity < self.batch_size:
            raise ValueError('Replay must hold a training batch')
        for name in ('lr', 'temperature', 'guide_temperature', 'reward_scale'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + ' must be finite and positive')
        if not 0 <= self.gamma < 1 or not 0 < self.tau <= 1:
            raise ValueError('Invalid discount or target rate')
        if not 0 <= self.guide_ess_fraction <= 1:
            raise ValueError('Invalid ESS fraction')
        for name in ('train_preferences', 'validation_preferences', 'heldout_preferences'):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values) or any(
                    not math.isfinite(x) or not 0 <= x <= 1 for x in values):
                raise ValueError(name + ' must contain distinct weights in [0,1]')
        if not set(self.validation_preferences) <= set(self.train_preferences):
            raise ValueError('Validation preferences must be training preferences')
        if set(self.heldout_preferences) & set(self.train_preferences):
            raise ValueError('Heldout preferences must not be used for fitting')
        if self.eval_seed_start + self.eval_episodes > self.final_seed_start:
            raise ValueError('Episode seed ranges overlap')
        if self.device != 'cpu' and not self.device.startswith('cuda'):
            raise ValueError('Invalid device')
        return self

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    def run_id(self):
        digest = hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:10]
        return f'{self.method}-s{self.seed}-{digest}'


def load_preference_config(path):
    raw = json.loads(Path(path).read_text())
    for name in ('train_preferences', 'validation_preferences', 'heldout_preferences','component_scales'):
        if name in raw:
            raw[name] = tuple(raw[name])
    return PreferenceConfig(**raw).validate()


def debug_preference_config(cfg):
    return replace(cfg, train_steps=4, warmup_steps=8, batch_size=2, hidden=32,
                   candidates=2, bootstrap_candidates=1, sampling_steps=2,
                   guide_interval=1, guide_batch_size=2, guide_suffixes=2,
                   guide_prefixes=2,
                   guide_particles=2, train_preferences=(0., .5, 1.),
                   validation_preferences=(0., .5, 1.), heldout_preferences=(.25, .75),
                   eval_interval=2, eval_episodes=1, final_eval_episodes=2,
                   checkpoint_interval=2, log_interval=1).validate()
