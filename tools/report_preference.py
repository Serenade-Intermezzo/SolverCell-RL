"""Audit and render the bounded preference prototype; never select on heldout."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from solvercell.preference_config import load_preference_config
from solvercell.preference_trainer import modes_for
from solvercell.storage import atomic_json


def read(path):
    return json.loads(path.read_text())


def finite(value):
    if torch.is_tensor(value) and value.is_floating_point():
        assert torch.isfinite(value).all()
    elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
        assert np.isfinite(value).all()
    elif isinstance(value, dict):
        for x in value.values(): finite(x)
    elif isinstance(value, (list, tuple)):
        for x in value: finite(x)
    elif isinstance(value, float):
        assert np.isfinite(value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = read(Path(args.manifest))
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    runs = []
    assert manifest['device'] == 'cpu'
    for index, job in enumerate(manifest['jobs']):
        output = root / job['output']
        cfg = load_preference_config(root / job['config'])
        assert read(output / 'config.json') == cfg.to_dict()
        metrics, source, worker = (read(output / name) for name in ('metrics.json', 'source.json', 'worker.json'))
        assert metrics['status'] == read(output / 'status.json')['status'] == 'complete'
        exit_record = read(output / 'exit.json')
        assert exit_record['returncode'] == 0
        assert metrics['source_sha256'] == source['source_sha256'] == worker['source_sha256'] == manifest['expected_source_sha256']
        assert exit_record['source_sha256'] == manifest['expected_source_sha256']
        assert hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest() == job['config_sha256']
        assert worker['comm'] == 'llmagent'
        assert cfg.device == 'cpu' and worker['config'] == cfg.to_dict()
        expected_cpus = sorted(set(job['cpu_set'] if 'cpu_set' in job else manifest['cpu_sets'][index]))
        assert worker['cpu_affinity'] == expected_cpus
        launcher = read(output / 'launcher.json')
        assert launcher['cpus'] == expected_cpus and launcher['pid'] == worker['pid']
        assert launcher['source_sha256'] == manifest['expected_source_sha256']
        assert launcher['config_sha256'] == job['config_sha256']
        assert metrics['train_steps'] == metrics['updates'] == cfg.train_steps
        assert metrics['env_steps'] == cfg.train_steps + cfg.warmup_steps
        checkpoint = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=False)
        assert checkpoint['source_sha256'] == manifest['expected_source_sha256']
        assert checkpoint['train_steps'] == checkpoint['learner']['updates'] == cfg.train_steps
        assert metrics['actor_updates'] == checkpoint['learner']['actor_updates'] == (cfg.train_steps + cfg.actor_interval - 1) // cfg.actor_interval
        expected_guides = cfg.train_steps // cfg.guide_interval if cfg.method == 'preference_diffusion' else 0
        assert checkpoint['learner']['guide_updates'] == expected_guides
        assert checkpoint['env_steps'] == metrics['env_steps']
        complete_episodes, episode_remainder = divmod(metrics['env_steps'], cfg.horizon)
        assert checkpoint['episode_count'] == metrics['complete_episodes'] == complete_episodes
        assert checkpoint['env_t'] == episode_remainder
        assert np.asarray(checkpoint['episode_components']).shape == (2,)
        assert np.isclose(np.mean(checkpoint['episode_components']), checkpoint['episode_return'], atol=1e-3)
        if episode_remainder == 0:
            assert checkpoint['episode_return'] == 0 and np.all(np.asarray(checkpoint['episode_components']) == 0)
        assert checkpoint['config'] == cfg.to_dict() and checkpoint['instance_hash'] == metrics['instance_hash']
        assert checkpoint['replay']['size'] == min(cfg.replay_capacity, metrics['env_steps'])
        finite(checkpoint)
        evaluations = [json.loads(s) for s in (output / 'evaluations.jsonl').read_text().splitlines()]
        finite(evaluations)
        steps = [0, *range(cfg.eval_interval, cfg.train_steps, cfg.eval_interval), cfg.train_steps]
        expected_rows = len(modes_for(cfg)) * (len(steps) * len(cfg.validation_preferences) + len(cfg.heldout_preferences))
        assert len(evaluations) == expected_rows
        for mode in modes_for(cfg):
            for weight in cfg.validation_preferences:
                curve = [r for r in evaluations if r['split'] == 'validation' and r['mode'] == mode and r['low_response_weight'] == weight]
                assert [r['train_steps'] for r in curve] == steps
                assert all(r['episodes'] == cfg.eval_episodes and r['seed_start'] == cfg.eval_seed_start for r in curve)
                assert all(r['preference_split'] == 'training' for r in curve)
            for weight in cfg.heldout_preferences:
                rows = [r for r in evaluations if r['split'] == 'heldout' and r['mode'] == mode and r['low_response_weight'] == weight]
                assert len(rows) == 1 and rows[0]['episodes'] == cfg.final_eval_episodes
                assert rows[0]['seed_start'] == cfg.final_seed_start
                assert rows[0]['train_steps'] == cfg.train_steps and rows[0]['env_steps'] == metrics['env_steps']
                assert rows[0]['preference_split'] == 'unseen_interpolation'
        for row in evaluations:
            component = np.asarray(row['component_returns'])
            original = np.asarray(row['original_returns'])
            assert component.shape == (row['episodes'], 2) and original.shape == (row['episodes'],)
            assert len(row['episode_lengths']) == row['episodes']
            assert np.allclose(row['preference'], [row['low_response_weight'], 1 - row['low_response_weight']])
            assert np.allclose(component.mean(1), original, atol=1e-3)
            assert np.allclose(component @ np.asarray(row['preference']), row['scalarized_returns'])
            assert row['feasibility_rate'] == 1 and all(n == cfg.horizon for n in row['episode_lengths'])
        episodes = [json.loads(s) for s in (output / 'episodes.jsonl').read_text().splitlines()]
        assert len(episodes) == metrics['env_steps'] // cfg.horizon == metrics['complete_episodes']
        assert all(r['length'] == cfg.horizon for r in episodes)
        runs.append({'id': job['id'], 'config': cfg.to_dict(), 'metrics': metrics,
                     'evaluations': evaluations, 'audit': 'verified_complete',
                     'training_episode_accounting': {'complete_episodes': complete_episodes,
                         'partial_episode_steps': episode_remainder, 'horizon': cfg.horizon,
                         'partial_episode_original_return': checkpoint['episode_return'],
                         'partial_episode_component_returns': np.asarray(checkpoint['episode_components']).tolist()},
                     'worker_evidence': {'pid': worker['pid'], 'cpu_affinity': worker['cpu_affinity'],
                         'device': cfg.device, 'comm': worker['comm']},
                     'guide_updates': expected_guides,
                     'learner_counters': checkpoint['learner'].get('counters', {})})
    study = {'scope': 'single-training-seed preference implementation pilot; not a superiority claim',
             'compute_accounting': {
                 'learner_counters': 'Cumulative forward-row counters for base training, guide fitting, behavior, validation and heldout; excludes warmup decoding and independent behavioral baselines. Not complete experiment FLOPs or wall cost.',
                 'evaluation_compute_totals': 'Additive per-rollout counters; NFE counts network evaluations times rows, not batch invocations. actor_nfe aliases sampling nfe during evaluation and must not be added to nfe.',
                 'sampler_diagnostics': 'Separate from additive cost counters: ess_mean, ess_final, ess_final_fraction and log_normalizer_mean are averaged over decisions; ess_min is the minimum over decisions. These are inference diagnostics, not computational work.'},
             'source_sha256': manifest['expected_source_sha256'], 'runs': runs}
    atomic_json(destination / 'study.json', study)
    lines = ['# Preference Intervention: completed implementation pilot', '',
             'All reported runs passed config/source/exit, full-curve, checkpoint, reward-vector and trajectory audits.', '',
             '**Scope:** new two-cohort preference task on the unchanged LSFlow Intervention dynamics and native quotient decoder. '
             'This is one training seed and one instance at a short budget. It does not establish diffusion superiority or a Pareto frontier.', '',
             'The actor NFE per sample is matched: spherical Heun uses half as many integration intervals as Gaussian diffusion. '
             'Joint potential fitting and SMC potential evaluations add cost and are recorded separately. '
             'Terminal weighting and SMC each use the same configured number of base particles and terminal action/value queries. '
             'The direct policy uses one sample and is a cheaper reference.', '',
             'Checkpoint learner counters cover training, guide fitting, behavior, validation and heldout, '
             'and exclude warmup decoding and independent behavioral baselines. They are forward-row counts, '
             'not complete experiment FLOPs. Per-evaluation compute totals are additive counts; ESS and '
             'log-normalizer diagnostics are reported separately and are not costs. The evaluation actor_nfe '
             'field aliases sampling nfe and must not be added to it.', '',
             'The base-policy critic uses the same relabeled preference at both ends of the Bellman target. '
             'Guidance is an inference-time base-policy-value improvement; it is not an exact critic evaluation of the guided policy. '
             'Finite suffix log-mean-exp labels and finite-particle SMC are approximations. '
             'No target reward, checkpoint or configuration was selected using heldout preferences.', '',
             '| Model | Updates | Last-window direct validation | Direct validation AUC | Wall seconds |',
             '|---|---:|---:|---:|---:|']
    for run in runs:
        m = run['metrics']
        lines.append(f"| {run['id']} | {m['updates']} | {m['last_window_return']:.3f} | {m['validation_auc']:.3f} | {m['wall_seconds']:.1f} |")
    lines += ['', 'Training uses a fixed transition/update budget. Any unfinished training episode is retained '
              'in the final checkpoint; every evaluation trajectory completes the full horizon.', '',
              '| Model | Complete training episodes (including warmup) | Partial final episode | Last-window evaluation points |',
              '|---|---:|---:|---:|']
    for run in runs:
        account, cfg = run['training_episode_accounting'], run['config']
        window_points = max(1, int(np.ceil(.2 * cfg['train_steps'] / cfg['eval_interval'])))
        lines.append(f"| {run['id']} | {account['complete_episodes']} | {account['partial_episode_steps']}/{account['horizon']} steps | {window_points} |")
    lines += ['', '## Unseen preference interpolation: final checkpoints', '',
              'Raw component returns use original reward units (N times per-capita cohort utility). '
              'At equal weights the scalarized return equals the original return. '
              'These rows use common final episode seeds and are descriptive; no cross-seed uncertainty is inferred.', '',
              '| Model | Mode | Low-response weight | Low-group return | High-group return | Scalarized return |',
              '|---|---|---:|---:|---:|---:|']
    for run in runs:
        for row in run['evaluations']:
            if row['split'] == 'heldout':
                a, b = row['component_return_mean']
                lines.append(f"| {run['id']} | {row['mode']} | {row['low_response_weight']:.3f} | {a:.3f} | {b:.3f} | {row['return_mean']:.3f} |")
    lines += ['', 'Prefer measured response curves and compute totals over a claim from one favorable weight. '
              'Failure to change cohort allocation is a controllability failure even if average utility is high. '
              'No monotonicity or guidance-efficiency claim is inferred merely from low regression loss.', '',
              '[Complete machine-readable results and compute totals](study.json) · [Curves](preference_curves.png)', '']
    (destination / 'RESULTS.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for run in runs:
        cfg = run['config']
        for mode in modes_for(load_preference_config(root / next(j['config'] for j in manifest['jobs'] if j['id'] == run['id']))):
            rows = run['evaluations']
            curve = []
            for step in sorted(set(r['train_steps'] for r in rows if r['split'] == 'validation')):
                values = [r['return_mean'] for r in rows if r['split'] == 'validation' and r['mode'] == mode and r['train_steps'] == step]
                curve.append((step, np.mean(values)))
            label = run['id'] + '/' + mode
            if mode == 'direct':
                axes[0].plot(*np.asarray(curve).T, label=label)
            heldout = sorted([r for r in rows if r['split'] == 'heldout' and r['mode'] == mode], key=lambda r: r['low_response_weight'])
            weights = [r['low_response_weight'] for r in heldout]
            axes[1].plot(weights, [r['component_return_mean'][0] for r in heldout], marker='o', label=label)
            axes[2].plot(weights, [r['component_return_mean'][1] for r in heldout], marker='o', label=label)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    axes[0].set(xlabel='Training updates', ylabel='Scalarized validation return', title='Conditional direct base policies')
    axes[1].set(xlabel='Low-response preference weight (unseen)', ylabel='Low-group return', title='Final heldout preference response')
    axes[2].set(xlabel='Low-response preference weight (unseen)', ylabel='High-group return', title='Final heldout preference response')
    fig.tight_layout()
    fig.savefig(destination / 'preference_curves.png', dpi=170)
    fig.savefig(destination / 'preference_curves.pdf')
    plt.close(fig)
    print(json.dumps({'verified_complete': [r['id'] for r in runs], 'output': str(destination.resolve())}))


if __name__ == '__main__':
    main()
