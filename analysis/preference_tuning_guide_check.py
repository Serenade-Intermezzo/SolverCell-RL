"""Validation-only guide mechanism check on a source-frozen base triplet."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.preference_guide_probe import calibration
from solvercell.environment import InterventionEnv
from solvercell.preference_config import load_preference_config
from solvercell.preference_data import PreferenceTask
from solvercell.preference_learner import PreferenceLearner
from solvercell.preference_trainer import evaluate_preference
from solvercell.storage import atomic_json, source_manifest


def same(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right): same(a, b)
    else:
        assert left == right


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest = json.loads(args.manifest.read_text())
    assert source_manifest()['source_sha256'] == manifest['expected_source_sha256']
    ids = ('diff_b060_t010', 'group_b060_t010', 'center_b060_t010')
    jobs = {job['id']: job for job in manifest['jobs']}
    checkpoints, hashes = {}, {}
    for name in ids:
        path = ROOT / jobs[name]['output']
        assert json.loads((path / 'status.json').read_text())['status'] == 'complete'
        assert json.loads((path / 'exit.json').read_text())['returncode'] == 0
        checkpoints[name] = torch.load(path / 'checkpoint.pt', map_location='cpu', weights_only=False)
        assert checkpoints[name]['source_sha256'] == manifest['expected_source_sha256']
        hashes[name] = hashlib.sha256((path / 'checkpoint.pt').read_bytes()).hexdigest()
    anchor = checkpoints[ids[0]]
    learner_fields = ('actor', 'target_actor', 'q1', 'q2', 'tq1', 'tq2', 'actor_optim', 'q_optim',
                      'generator', 'preference_generator', 'behavior_generator', 'updates', 'actor_updates')
    training_fields = ('replay', 'env_state', 'env_t', 'env_rng', 'replay_rng', 'cost_rng', 'episode_rng',
                       'behavior_generator', 'python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng', 'env_steps',
                       'train_steps', 'episode_count', 'episode_components', 'episode_return', 'preference', 'validation')
    for name in ids[1:]:
        for key in learner_fields: same(anchor['learner'][key], checkpoints[name]['learner'][key])
        for key in training_fields: same(anchor[key], checkpoints[name][key])
    atomic_json(args.output / 'base_equivalence.json', {'status': 'bitwise_equal', 'arms': ids,
        'learner_fields': learner_fields, 'training_fields': training_fields, 'checkpoint_sha256': hashes})
    atomic_json(args.output / 'design.json', {'scope': 'validation-only post-training mechanism check; no heldout selection',
        'parent_manifest': str(args.manifest), 'source_sha256': manifest['expected_source_sha256'],
        'arms': ids, 'episode_seed_start': 500000, 'episodes': 16, 'validation_preferences': [0, .5, 1],
        'smc_ess_fractions': [.5, .9], 'guide_temperature': .5,
        'calibration': 'two same replay rows;16 actual prefixes×128 suffixes;fixed context;split-half MC check',
        'decision': 'No guide promotion without calibration and utility benefit; no threshold selected from final results'})
    records, evaluations, initial_ref = [], [], None
    start = time.perf_counter()
    for name in ids:
        checkpoint = checkpoints[name]
        cfg = load_preference_config(ROOT / jobs[name]['config'])
        env = InterventionEnv(cfg.n_arms, cfg.n_actions, cfg.budget, cfg.horizon,
                              instance_seed=cfg.instance_seed, seed=cfg.seed + 100)
        task = PreferenceTask(env)
        learner = PreferenceLearner(cfg, task)
        learner.load_state_dict(checkpoint['learner'])
        sub = args.output / name
        sub.mkdir()
        calibrated = calibration(learner, checkpoint, cfg, sub)
        refs = [row['reference_logmeanexp_128'] for row in calibrated['groups']]
        if initial_ref is None: initial_ref = refs
        else: same(initial_ref, refs)
        atomic_json(sub / 'calibration.json', calibrated)
        records.append({'arm': name, 'calibration': calibrated})
        for weight in cfg.validation_preferences:
            for mode, threshold in (('terminal_weighting', .5), ('smc', .5), ('smc', .9)):
                learner.cfg = replace(cfg, guide_ess_fraction=threshold)
                row = evaluate_preference(task, learner, weight, mode, 16, 500000)
                row.update(arm=name, ess_fraction=threshold if mode == 'smc' else None)
                evaluations.append(row)
        print(json.dumps({'arm': name, 'complete': True}), flush=True)
    paired = []
    for row in evaluations:
        if row['mode'] != 'smc': continue
        terminal = next(r for r in evaluations if r['arm'] == row['arm'] and r['mode'] == 'terminal_weighting'
                        and r['low_response_weight'] == row['low_response_weight'])
        delta = np.asarray(row['scalarized_returns']) - np.asarray(terminal['scalarized_returns'])
        paired.append({'arm': row['arm'], 'weight': row['low_response_weight'], 'ess_fraction': row['ess_fraction'],
            'differences': delta.tolist(), 'mean_difference': float(delta.mean()),
            'episode_mean_se': float(delta.std(ddof=1)/4),
            'resampling_events': row['compute_totals']['resampling_events']})
    result = {'scope': 'validation-only guide diagnostic; learned-critic MC is not ground truth',
        'source_sha256': manifest['expected_source_sha256'], 'base_equivalence': 'bitwise_equal',
        'identical_calibration_references': True, 'checkpoint_sha256': hashes,
        'calibration': records, 'evaluations': evaluations, 'paired_differences': paired,
        'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'wall_seconds': time.perf_counter()-start}
    atomic_json(args.output / 'summary.json', result)
    lines = ['# Validation-only guide mechanism check', '',
        'All three base learners, replay, training environment/RNG and direct validation curves are bitwise equal. '
        'Reference suffix draws/targets are also exactly equal across all18 calibration groups. '
        'Only the guide data grouping/loss differs. No final preference or final episode is used.', '',
        '| Guide | Median centered RMSE | Median constant-relative RMSE ratio | Median within-group r |',
        '|---|---:|---:|---:|']
    for record in records:
        groups = record['calibration']['groups']
        ratios = [g['centered_rmse']/g['constant_predictor_centered_rmse'] for g in groups]
        lines.append(f"| {record['arm']} | {np.median([g['centered_rmse'] for g in groups]):.5f} | "
                     f"{np.median(ratios):.3f} | {np.median([g['within_group_pearson'] for g in groups]):.3f} |")
    lines += ['', 'Each group fixes state, preference and noise level. Only two replay states are probed. '
        'Finite128-suffix log-mean-exp has Monte Carlo noise/bias; inspect all group values and split-half consistency in summary.json.', '',
        '| Guide | Weight | ESS fraction | SMC minus terminal | Paired episode-mean SE | Resampling events |',
        '|---|---:|---:|---:|---:|---:|']
    for row in paired:
        lines.append(f"| {row['arm']} | {row['weight']:.1f} | {row['ess_fraction']:.1f} | "
                     f"{row['mean_difference']:.3f} | {row['episode_mean_se']:.3f} | {row['resampling_events']:.0f} |")
    lines += ['', 'These16 episodes per context describe a fixed seed0 checkpoint. They do not establish training-seed stability '
              'or an efficiency advantage. All SMC calls add guide cost; compare full counters and wall time. '
              'No larger guide sweep or default change follows automatically.', '']
    (args.output / 'RESULTS.md').write_text('\n'.join(lines))
    assert source_manifest()['source_sha256'] == manifest['expected_source_sha256']
    for name in ids:
        assert hashlib.sha256((ROOT / jobs[name]['output'] / 'checkpoint.pt').read_bytes()).hexdigest() == hashes[name]
    print(json.dumps({'complete': True, 'output': str(args.output)}))


if __name__ == '__main__': main()
