"""Headless preference benchmark with complete, resumable experiment artifacts.

The environment and native quotient decoder are preserved. This is a new
multi-objective harness, not an upstream DPMD reproduction. Guided execution
is an inference-time improvement of the conditional base policy; its critic
continues to evaluate the base policy at the same preference.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from .environment import InterventionEnv
from .preference_data import PreferenceTask, PreferenceReplay
from .preference_learner import PreferenceLearner
from .storage import atomic_json, append_json, atomic_checkpoint, source_manifest, worker_identity


def modes_for(cfg):
    return ('direct', 'terminal_weighting', 'smc') if cfg.method == 'preference_diffusion' else (
        'direct', 'terminal_weighting')


def myopic_action(task, state, preference):
    """Model-informed primitive-action marginal greedy; broader than native support."""
    env = task.env
    action = np.zeros(env.n_actions, np.float32)
    value = float(np.dot(task.expected_components(action, state), preference))
    for _ in range(env.budget):
        best, best_value = None, value
        for i in np.flatnonzero(action == 0):
            candidate = action.copy()
            candidate[i] = 1
            v = float(np.dot(task.expected_components(candidate, state), preference))
            if v > best_value + 1e-12:
                best, best_value = int(i), v
        if best is None:
            break
        action[best], value = 1, best_value
    return action


def evaluate_preference(task, learner, weight, mode, episodes, seed_start):
    """Isolated common-random-number rollouts; preferences fixed per episode."""
    if not 0 <= weight <= 1 or episodes < 1:
        raise ValueError('Invalid evaluation weight or episode count')
    env = task.env.clone()
    local = PreferenceTask(env)
    preference = np.array([weight, 1 - weight], np.float32)
    component_returns, original_returns, counts, completed_lengths = [], [], [], []
    actions_seen, totals, diagnostics = set(), {}, {}
    diagnostic_keys = {'ess_mean', 'ess_min', 'ess_final', 'ess_final_fraction',
                       'log_normalizer_mean'}
    start = time.perf_counter()
    for seed in range(seed_start, seed_start + episodes):
        state = env.reset(seed=seed)
        gen = torch.Generator(device=learner.device).manual_seed(seed + 2000000)
        rng = np.random.default_rng(seed + 3000000)
        components = np.zeros(2, np.float64)
        original, steps = 0., 0
        for _ in range(env.horizon):
            encoded = local.encode(state, env.horizon - env.t)
            if mode == 'random_prior':
                cost = rng.normal(size=local.cost_dim).astype(np.float32)
                action = local.decode(torch.from_numpy(cost[None])).cpu().numpy()[0]
                stats = {'decoder_rows': 1.}
            elif mode == 'myopic_primitive':
                action = myopic_action(local, state, preference)
                stats = {}
            else:
                action, _, stats = learner.act(encoded, preference, mode=mode, generator=gen)
            state, reward, done, info = env.step(action)
            components += local.components(state)
            original += reward
            steps += 1
            counts.append(info['action_count'])
            actions_seen.add(np.packbits(action.astype(np.uint8)).tobytes())
            for key, value in stats.items():
                if isinstance(value, (float, int, np.number)):
                    if key in diagnostic_keys:
                        diagnostics.setdefault(key, []).append(float(value))
                    else:
                        totals[key] = totals.get(key, 0.) + float(value)
            if done:
                break
        if steps != env.horizon:
            raise RuntimeError('Evaluation did not complete its trajectory')
        component_returns.append(components.tolist())
        original_returns.append(original)
        completed_lengths.append(steps)
    matrix = np.asarray(component_returns)
    returns = matrix @ preference
    return {'preference': preference.tolist(), 'low_response_weight': float(weight),
            'mode': mode, 'episodes': episodes, 'seed_start': seed_start,
            'component_returns': component_returns, 'component_return_mean': matrix.mean(0).tolist(),
            'scalarized_returns': returns.tolist(), 'return_mean': float(returns.mean()),
            'return_std': float(returns.std(ddof=1)) if episodes > 1 else 0.,
            'original_returns': original_returns, 'original_return_mean': float(np.mean(original_returns)),
            'episode_lengths': completed_lengths, 'mean_action_count': float(np.mean(counts)),
            'unique_actions': len(actions_seen), 'feasibility_rate': 1.,
            'compute_totals': totals,
            'sampler_diagnostics': {key: float(min(values) if key == 'ess_min' else np.mean(values))
                                    for key, values in diagnostics.items()},
            'eval_seconds': time.perf_counter() - start}


def train_preference(cfg, output, *, resume=False, stop_after_updates=None):
    cfg.validate()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'config.json'
    if config_path.exists() and not resume:
        raise FileExistsError('Refusing to overwrite existing preference experiment')
    if resume and (not config_path.exists() or json.loads(config_path.read_text()) != cfg.to_dict()):
        raise ValueError('Resume requires identical preference config')
    if resume and (output / 'metrics.json').exists():
        raise ValueError('A completed experiment must not be resumed')
    if stop_after_updates is not None and stop_after_updates < 1:
        raise ValueError('Stop budget must be positive')
    torch.set_num_threads(cfg.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if cfg.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = source_manifest()
    if os.environ.get('SOLVERCELL_SOURCE_SHA256', source['source_sha256']) != source['source_sha256']:
        raise RuntimeError('Frozen source fingerprint mismatch')
    checkpoint = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=False) if resume else None
    if checkpoint and checkpoint['source_sha256'] != source['source_sha256']:
        raise ValueError('Cannot resume changed scientific source')
    env = InterventionEnv(cfg.n_arms, cfg.n_actions, cfg.budget, cfg.horizon,
                          instance_seed=cfg.instance_seed, seed=cfg.seed + 100)
    task = PreferenceTask(env)
    if checkpoint and checkpoint['instance_hash'] != env.instance_hash:
        raise ValueError('Instance mismatch')
    learner = PreferenceLearner(cfg, task)
    replay = PreferenceReplay(cfg.replay_capacity, task.obs_dim, cfg.n_actions, task.cost_dim)
    replay_rng = np.random.default_rng(cfg.seed + 901)
    cost_rng = np.random.default_rng(cfg.seed + 902)
    episode_rng = np.random.default_rng(cfg.seed + 903)
    behavior_generator = torch.Generator(device=cfg.device).manual_seed(cfg.seed + 904)
    state = env.reset()
    lam = float(episode_rng.choice(cfg.train_preferences))
    preference = np.array([lam, 1 - lam], np.float32)
    env_steps = train_steps = episode_count = 0
    episode_components = np.zeros(2, np.float64)
    episode_return, previous_wall = 0., 0.
    validation, last_info = [], {}
    if checkpoint:
        learner.load_state_dict(checkpoint['learner'])
        replay.load_state_dict(checkpoint['replay'])
        env.state, env.t = checkpoint['env_state'].copy(), checkpoint['env_t']
        env.rng.bit_generator.state = checkpoint['env_rng']
        state = env.observation()
        for rng, key in ((replay_rng, 'replay_rng'), (cost_rng, 'cost_rng'), (episode_rng, 'episode_rng')):
            rng.bit_generator.state = checkpoint[key]
        behavior_generator.set_state(checkpoint['behavior_generator'].cpu())
        random.setstate(checkpoint['python_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if cfg.device.startswith('cuda'):
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        env_steps, train_steps = checkpoint['env_steps'], checkpoint['train_steps']
        episode_count = checkpoint['episode_count']
        episode_components = checkpoint['episode_components'].copy()
        episode_return, preference = checkpoint['episode_return'], checkpoint['preference'].copy()
        validation, previous_wall = checkpoint['validation'], checkpoint['wall_seconds']
        last_info = checkpoint['last_info']
        for name in ('metrics.jsonl', 'evaluations.jsonl', 'episodes.jsonl'):
            path = output / name
            if path.exists():
                rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows
                                        if row.get('env_steps', 0) <= env_steps and row.get('split') != 'heldout'))
    identity = worker_identity()
    identity.update(method=cfg.method, source_sha256=source['source_sha256'], config=cfg.to_dict())
    atomic_json(config_path, cfg.to_dict())
    atomic_json(output / 'source.json', source)
    atomic_json(output / 'worker.json', identity)
    atomic_json(output / 'instance.json', env.instance_dict())
    atomic_json(output / 'effective_algorithm.json', {
        'profile': cfg.profile, 'decoder': 'native rank quotient; stable original W scores',
        'obs': 'state/3 plus remaining-horizon fraction', 'cost_dim': task.cost_dim,
        'reward_components': 'N times per-capita utility in low/high response cohorts',
        'equal_preference_original_reward': True, 'critic_reward_scale': cfg.reward_scale,
        'behavior': 'one direct conditional target-actor sample; no final execution perturbation',
        'bellman': 'same relabeled preference and direct target actor; inner guide does not alter TD policy',
        'guide': 'joint log-exponential continuation, fresh actual base suffixes; finite sample labels',
        'deployment': 'direct/terminal importance weighting/finite-particle telescoping SMC',
        'inference_scope': 'base-policy value guided improvement, not evaluation-Q of the guided policy',
        'sampler_nfe_per_sample': cfg.sampling_steps, 'original_DPMD_reproduction': False,
        'preference_split': {'training': list(cfg.train_preferences),
                             'unseen_interpolation': list(cfg.heldout_preferences)}})
    start = time.perf_counter()
    stopped = [False]
    handlers = {}
    if __import__('threading').current_thread() is __import__('threading').main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, lambda *_: stopped.__setitem__(0, True))

    def wall():
        return previous_wall + time.perf_counter() - start

    def status(value, **extra):
        atomic_json(output / 'status.json', {'status': value, 'env_steps': env_steps,
                    'train_steps': train_steps, 'updates': learner.updates,
                    'source_sha256': source['source_sha256'], 'wall_seconds': wall(), **extra})

    def save():
        atomic_checkpoint(output / 'checkpoint.pt', {
            'schema_version': 'preference_v1', 'config': cfg.to_dict(),
            'source_sha256': source['source_sha256'], 'instance_hash': env.instance_hash,
            'learner': learner.state_dict(), 'replay': replay.state_dict(),
            'env_state': env.state.copy(), 'env_t': env.t, 'env_rng': env.rng.bit_generator.state,
            'replay_rng': replay_rng.bit_generator.state, 'cost_rng': cost_rng.bit_generator.state,
            'episode_rng': episode_rng.bit_generator.state, 'behavior_generator': behavior_generator.get_state(),
            'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if cfg.device.startswith('cuda') else None,
            'env_steps': env_steps, 'train_steps': train_steps, 'episode_count': episode_count,
            'episode_components': episode_components.copy(), 'episode_return': episode_return,
            'preference': preference.copy(), 'validation': validation, 'last_info': last_info,
            'wall_seconds': wall()})

    def validate():
        status('running', phase='validation')
        direct_values = []
        for weight in cfg.validation_preferences:
            for mode in modes_for(cfg):
                row = evaluate_preference(task, learner, weight, mode, cfg.eval_episodes, cfg.eval_seed_start)
                row.update(split='validation', preference_split='training', env_steps=env_steps, train_steps=train_steps)
                append_json(output / 'evaluations.jsonl', row)
                if mode == 'direct':
                    direct_values.append(row['return_mean'])
        point = {'train_steps': train_steps, 'return_mean': float(np.mean(direct_values))}
        validation.append(point)
        print(json.dumps({'event': 'validation', 'method': cfg.method, **point}), flush=True)

    try:
        status('running', phase='initializing')
        if not resume:
            for name in ('metrics.jsonl', 'evaluations.jsonl', 'episodes.jsonl'):
                (output / name).touch(exist_ok=False)
            baselines = []
            for weight in cfg.validation_preferences:
                for mode in ('random_prior', 'myopic_primitive'):
                    row = evaluate_preference(task, learner, weight, mode, cfg.eval_episodes, cfg.eval_seed_start)
                    row['scope'] = 'model-informed broader primitive support' if mode == 'myopic_primitive' else 'native prior'
                    baselines.append(row)
            atomic_json(output / 'behavioral_baselines.json', baselines)
            validate()
        while train_steps < cfg.train_steps and not stopped[0]:
            warmup = env_steps < cfg.warmup_steps
            encoded = task.encode(state, env.horizon - env.t)
            if warmup:
                cost = cost_rng.normal(size=task.cost_dim).astype(np.float32)
                cost /= max(float(np.linalg.norm(cost)), 1e-8)
                action = task.decode(torch.from_numpy(cost[None])).cpu().numpy()[0]
            else:
                action, cost, _ = learner.act(encoded, preference, mode='direct', generator=behavior_generator)
            nxt, reward, done, _ = env.step(action)
            components = task.components(nxt)
            if not np.isclose(.5 * components.sum(), reward, atol=1e-4):
                raise RuntimeError('Reward decomposition changed original utility')
            next_encoded = task.encode(nxt, env.horizon - env.t)
            replay.add(encoded, action, cost, components, next_encoded, float(done), preference)
            env_steps += 1
            episode_return += reward
            episode_components += components
            state = nxt
            if done:
                episode_count += 1
                append_json(output / 'episodes.jsonl', {'env_steps': env_steps, 'train_steps': train_steps,
                            'episode': episode_count, 'length': cfg.horizon, 'preference': preference.tolist(),
                            'original_return': episode_return, 'component_returns': episode_components.tolist(),
                            'phase': 'warmup' if warmup else 'train'})
                episode_return, episode_components = 0., np.zeros(2, np.float64)
                state = env.reset()
                lam = float(episode_rng.choice(cfg.train_preferences))
                preference = np.array([lam, 1 - lam], np.float32)
            if warmup:
                continue
            last_info = learner.update(replay.sample(cfg.batch_size, replay_rng))
            train_steps += 1
            if train_steps % cfg.log_interval == 0 or train_steps == cfg.train_steps:
                append_json(output / 'metrics.jsonl', {'env_steps': env_steps, 'train_steps': train_steps,
                            'wall_seconds': wall(), **last_info})
            if train_steps % cfg.eval_interval == 0 or train_steps == cfg.train_steps:
                validate()
            if train_steps % cfg.checkpoint_interval == 0 or train_steps == cfg.train_steps:
                save()
            if stop_after_updates is not None and train_steps >= stop_after_updates:
                stopped[0] = train_steps < cfg.train_steps
        if stopped[0]:
            save()
            status('stopped', phase='safe_checkpoint')
            return {'status': 'stopped', 'train_steps': train_steps}
        if learner.updates != cfg.train_steps or env_steps != cfg.train_steps + cfg.warmup_steps:
            raise RuntimeError('Incomplete training accounting')
        status('running', phase='heldout')
        heldout = []
        for weight in cfg.heldout_preferences:
            for mode in modes_for(cfg):
                row = evaluate_preference(task, learner, weight, mode,
                                          cfg.final_eval_episodes, cfg.final_seed_start)
                row.update(split='heldout', preference_split='unseen_interpolation',
                           env_steps=env_steps, train_steps=train_steps)
                append_json(output / 'evaluations.jsonl', row)
                heldout.append(row)
        save()
        for value in learner.state_dict().values():
            if isinstance(value, dict):
                for tensor in value.values():
                    if torch.is_tensor(tensor) and tensor.is_floating_point() and not torch.isfinite(tensor).all():
                        raise FloatingPointError('Nonfinite scientific state')
        y = np.array([v['return_mean'] for v in validation])
        x = np.array([v['train_steps'] for v in validation])
        last_window = max(1, int(np.ceil(.2 * cfg.train_steps / cfg.eval_interval)))
        direct = [r for r in heldout if r['mode'] == 'direct']
        result = {'status': 'complete', 'method': cfg.method, 'seed': cfg.seed,
                  'train_steps': train_steps, 'updates': learner.updates, 'actor_updates': learner.actor_updates,
                  'warmup_steps': cfg.warmup_steps, 'env_steps': env_steps, 'complete_episodes': episode_count,
                  'final_validation_return': float(y[-1]), 'last_window_return': float(y[-last_window:].mean()),
                  'validation_auc': float(np.trapz(y, x) / cfg.train_steps),
                  'heldout_return_mean': float(np.mean([r['return_mean'] for r in direct])),
                  'heldout_return_std': float(np.std([r['return_mean'] for r in direct])),
                  'heldout_scope': 'unseen preference interpolation, across-preference SD; not training-seed SD',
                  'validation_points': len(validation), 'heldout_rows': len(heldout),
                  'source_sha256': source['source_sha256'], 'instance_hash': env.instance_hash,
                  'sampler_nfe_per_sample': cfg.sampling_steps, 'wall_seconds': wall(),
                  'parameter_counts': learner.parameter_counts(), 'seed_scope': 'single_training_seed',
                  'scope': 'new preference extension and algorithm harness; not original DPMD reproduction'}
        atomic_json(output / 'metrics.json', result)
        status('complete', phase='finished')
        print(json.dumps({'event': 'complete', **result}), flush=True)
        return result
    except Exception as error:
        status('failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
