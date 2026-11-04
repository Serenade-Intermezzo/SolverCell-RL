#!/usr/bin/env python3
"""Read-only fixed-checkpoint guide probe; no fitting or heldout selection.

Reproduce with .venv/bin/python analysis/preference_guide_probe.py. Only this
analysis script and its reports are written; frozen scientific modules, saved
configs, and the completed pilot checkpoint must retain their exact hashes.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solvercell.environment import InterventionEnv
from solvercell.preference_config import load_preference_config
from solvercell.preference_data import PreferenceTask
from solvercell.preference_learner import PreferenceLearner
from solvercell.preference_trainer import evaluate_preference
from solvercell.storage import append_json, atomic_json, source_manifest


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def models_hash(learner):
    digest = hashlib.sha256()
    for name in ('actor', 'target_actor', 'q1', 'q2', 'tq1', 'tq2', 'guide'):
        for key, value in sorted(getattr(learner, name).state_dict().items()):
            digest.update(f'{name}.{key}:{value.dtype}:{tuple(value.shape)}'.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def correlation(first, second):
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.std() < 1e-12 or second.std() < 1e-12:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def aggregate(rows):
    scalar = np.asarray([row['scalarized_returns'][0] for row in rows])
    components = np.asarray([row['component_returns'][0] for row in rows])
    totals = {key: sum(row['compute_totals'].get(key, 0.) for row in rows)
              for key in sorted({key for row in rows for key in row['compute_totals']})}
    diagnostic = {key: float(min(row['sampler_diagnostics'][key] for row in rows)
                            if key == 'ess_min' else np.mean([
                                row['sampler_diagnostics'][key] for row in rows]))
                  for key in rows[0]['sampler_diagnostics']}
    return {'episodes': len(rows), 'scalarized_returns': scalar.tolist(),
            'return_mean': float(scalar.mean()), 'return_std': float(scalar.std(ddof=1)),
            'component_return_mean': components.mean(0).tolist(),
            'original_return_mean': float(np.mean([row['original_returns'][0] for row in rows])),
            'compute_totals': totals, 'sampler_diagnostics': diagnostic,
            'eval_seconds': sum(row['eval_seconds'] for row in rows),
            'episode_lengths': [row['episode_lengths'][0] for row in rows]}


@torch.no_grad()
def calibration(learner, checkpoint, cfg, output):
    """Within ONE obs/pref/index, compare 16 prefixes with fresh suffix means."""
    size = int(checkpoint['replay']['size'])
    replay_rows = [0, size - 1]
    prefix_count, suffix_count, chunk_size = 16, 128, 256
    indices = [0, 4, 7]
    if cfg.sampling_steps != 8:
        raise ValueError('This prespecified diagnostic expects the eight-step pilot')
    groups, all_logits = [], {}
    before = learner.counters.copy()
    start = time.perf_counter()
    path_nfe = suffix_nfe = guide_calls = 0
    for state_number, replay_row in enumerate(replay_rows):
        obs = checkpoint['replay']['data']['obs'][replay_row].to(learner.device)[None]
        for pref_number, weight in enumerate(cfg.validation_preferences):
            prefs = obs.new_tensor([[weight, 1 - weight]])
            repeated_obs, repeated_prefs = obs.expand(prefix_count, -1), prefs.expand(prefix_count, -1)
            path_seed = 800000 + state_number * 1000 + pref_number * 100
            gen = torch.Generator(device=learner.device).manual_seed(path_seed)
            path = learner.target_actor.sample_path(repeated_obs, repeated_prefs, generator=gen)
            path_nfe += prefix_count * cfg.sampling_steps
            baseline = (learner._twin_vector(obs, obs.new_zeros((1, learner.action_dim)), prefs) * prefs).sum(-1)
            for index in indices:
                group_start = time.perf_counter()
                prefix = path[:, index]
                prediction = learner.guide(repeated_obs, repeated_prefs, prefix, index)
                guide_calls += prefix_count
                suffix_seed = path_seed + index + 1
                suffix_gen = torch.Generator(device=learner.device).manual_seed(suffix_seed)
                robs = repeated_obs.repeat_interleave(suffix_count, 0)
                rprefs = repeated_prefs.repeat_interleave(suffix_count, 0)
                rx = prefix.repeat_interleave(suffix_count, 0)
                values = []
                for begin in range(0, len(rx), chunk_size):
                    end = begin + chunk_size
                    raw = learner.target_actor.suffix(robs[begin:end], rprefs[begin:end], rx[begin:end],
                                                      index, generator=suffix_gen)
                    utility = (learner._twin_vector(robs[begin:end], learner._decode(raw), rprefs[begin:end])
                               * rprefs[begin:end]).sum(-1)
                    values.append((utility - baseline) / cfg.guide_temperature)
                suffix_nfe += len(rx) * (cfg.sampling_steps - index)
                logits = torch.cat(values).reshape(prefix_count, suffix_count).double()
                reference = torch.logsumexp(logits, 1) - math.log(suffix_count)
                halves = torch.stack([torch.logsumexp(part, 1) - math.log(suffix_count // 2)
                                      for part in logits.chunk(2, 1)], 1)
                scaled = (logits - logits.max(1, keepdim=True).values).exp()
                # Delta-method conditional MC SE, not a CI for true environment Q.
                mc_se = scaled.std(1, unbiased=True) / scaled.mean(1) / math.sqrt(suffix_count)
                pred, ref = prediction.double().cpu().numpy(), reference.cpu().numpy()
                centered_pred, centered_ref = pred - pred.mean(), ref - ref.mean()
                group_id = f'row{replay_row}_weight{weight:g}_index{index}'
                all_logits[group_id] = logits.cpu().numpy()
                groups.append({
                    'group_id': group_id, 'replay_row': replay_row, 'encoded_obs': obs[0].cpu().tolist(),
                    'preference': [weight, 1 - weight], 'path_index': index,
                    'path_seed': path_seed, 'suffix_seed': suffix_seed,
                    'prefixes': prefix.cpu().tolist(), 'learned_logh': pred.tolist(),
                    'reference_logmeanexp_128': ref.tolist(),
                    'reference_split_half_64': halves.cpu().tolist(), 'reference_delta_mc_se': mc_se.cpu().tolist(),
                    'within_group_pearson': correlation(pred, ref),
                    'reference_split_half_pearson': correlation(halves[:, 0].cpu(), halves[:, 1].cpu()),
                    'learned_range': float(np.ptp(pred)), 'reference_range': float(np.ptp(ref)),
                    'learned_std': float(pred.std()), 'reference_std': float(ref.std()),
                    'reference_mean_delta_mc_se': float(mc_se.mean()),
                    'mean_logh_bias_vs_mc': float((pred - ref).mean()),
                    'centered_rmse': float(np.sqrt(np.mean((centered_pred - centered_ref) ** 2))),
                    'constant_predictor_centered_rmse': float(ref.std()),
                    'split_half_rmse': float((halves[:, 0] - halves[:, 1]).square().mean().sqrt()),
                    'suffix_nfe': len(rx) * (cfg.sampling_steps - index),
                    'terminal_evaluations': len(rx), 'guide_calls': prefix_count,
                    'wall_seconds': time.perf_counter() - group_start,
                })
    np.savez_compressed(output / 'calibration_terminal_logits.npz', **all_logits)
    return {'groups': groups, 'groups_count': len(groups), 'prefixes_per_group': prefix_count,
            'suffixes_per_prefix': suffix_count, 'suffix_chunk_size': chunk_size,
            'replay_rows': replay_rows, 'path_indices': indices,
            'path_nfe': path_nfe, 'suffix_nfe': suffix_nfe, 'actor_nfe': path_nfe + suffix_nfe,
            'guide_calls': guide_calls,
            'q_calls': learner.counters['q_calls'] - before['q_calls'],
            'decoder_calls': learner.counters['decoder_calls'] - before['decoder_calls'],
            'terminal_evaluations': len(groups) * prefix_count * suffix_count,
            'wall_seconds': time.perf_counter() - start,
            'scope': 'Reference uses the same fixed learned target critics and actual target-actor suffix kernel; '
                     'it is not a ground-truth return oracle. Correlations are within fixed obs/preference/index only.',
            'limitations': ['128 finite suffixes give noisy downward-biased log-mean-exp estimates.',
                            'Split-half consistency and delta-method MC SE diagnose sampling uncertainty, not critic error.',
                            'Only two prespecified replay rows; no population calibration claim.',
                            'Group-wise additive guide offsets do not affect particle selection; centered errors matter.']}


def write_report(result, output):
    lines = ['# Fixed-checkpoint preference-guide diagnostic', '',
             'Completed post-hoc validation-only probe. No training, checkpoint selection, heldout tuning, '
             'source/config edits, efficiency claim, or method-superiority claim.', '',
             f"Scientific source: `{result['provenance']['source_sha256']}`. "
             f"Checkpoint: `{result['provenance']['checkpoint_path']}`; SHA256 "
             f"`{result['provenance']['checkpoint_sha256']}`.", '',
             'All arms use the same trained target actor, target vector critics, guide, eight particles, '
             'eight denoising steps, and fresh paired episode seeds 400000–400015. '
             'Only validation preferences 0, 0.5, 1 are queried. ESS threshold changes are read-only '
             'inference diagnostics; none is selected as a new configuration.', '',
             '| Low-response weight | Inference | Return mean ± episode SD | Resampling events | Actor NFE rows | Guide rows | Q rows | Decoder rows | Wall s |',
             '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in result['evaluations']:
        stats = row['compute_totals']
        lines.append(f"| {row['weight']:g} | {row['arm']} | {row['return_mean']:.6f} ± {row['return_std']:.6f} "
                     f"| {stats['resampling_events']:.0f} | {stats['actor_sampling_nfe']:.0f} "
                     f"| {stats['guide_calls']:.0f} | {stats['q_calls']:.0f} "
                     f"| {stats['decoder_calls']:.0f} | {row['eval_seconds']:.3f} |")
    lines += ['', 'Actor NFE, guide, Q and decoder counts are processed rows, not Python invocations. '
              'Episode SD describes rollout variability for one fixed checkpoint, not training-seed stability. '
              'SMC adds guide calls even if it never resamples. Threshold 1 resamples almost every '
              'strictly nonuniform population; this is a diagnostic intervention, not an efficiency recommendation.', '',
              f"Threshold-0 terminal-telescoping check: **{result['sanity']['smc_zero_exact_episode_outputs']}** "
              '(all 48 paired episodes have exactly equal component returns, scalarized returns, '
              'original returns, episode lengths, action counts, and unique-action counts).', '',
              '## Within-group actual-suffix calibration', '',
              'Each row fixes one replay observation, one preference and one path index, then compares '
              '16 fresh actual prefixes. Each prefix has 128 independent fresh suffixes. The reference '
              'is log-mean-exp of the same learned terminal utility used during deployment. It can '
              'diagnose guide fit to the fixed critic, not critic correctness or actual return calibration.', '',
              '| Replay row | Weight | Index | Learned range | MC reference range | Within-group r | Split-half r | Mean MC SE | Centered RMSE / constant RMSE |',
              '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in result['calibration']['groups']:
        corr = lambda value: 'undefined' if value is None else f'{value:.3f}'
        ratio = row['centered_rmse'] / row['constant_predictor_centered_rmse'] if row['constant_predictor_centered_rmse'] else float('inf')
        lines.append(f"| {row['replay_row']} | {row['preference'][0]:g} | {row['path_index']} "
                     f"| {row['learned_range']:.5f} | {row['reference_range']:.5f} "
                     f"| {corr(row['within_group_pearson'])} | {corr(row['reference_split_half_pearson'])} "
                     f"| {row['reference_mean_delta_mc_se']:.5f} | {ratio:.3f} |")
    lines += ['', 'Log-mean-exp with 128 suffixes has finite-sample noise and downward bias. '
              'The split halves use 64 suffixes each; low split-half correlation limits interpretation '
              'of a low learned-guide correlation. Group mean offsets are removed for centered RMSE '
              'because common additive log-potential offsets do not change normalized particle weights. '
              'No cross-state pooled correlation is reported. No correlation threshold is used for tuning.', '',
              '## Artifacts and scope', '',
              '- `summary.json`: provenance, exact aggregate statistics, paired episode differences, all calibration prefixes and estimates.',
              '- `episodes.jsonl`: every original single-episode `evaluate_preference` result, including returns, sampler diagnostics and compute.',
              '- `calibration_terminal_logits.npz`: all 128 raw suffix terminal logweights for every prefix, sufficient to recompute MC reference statistics.',
              '- `design.json`: intervention and checks written before running.', '',
              f"Total diagnostic wall time: {result['runtime']['wall_seconds']:.3f} s; "
              f"process CPU time: {result['runtime']['process_cpu_seconds']:.3f} s; "
              f"maximum resident set: {result['runtime']['max_rss_kib']} KiB; CPU threads: 1.", '',
              'The scientific source, original config, original checkpoint, all model tensors, learner-owned RNGs '
              'and update counts passed before/after immutability checks. Evaluation counters alone are '
              'advanced in this separate analysis process. This probe does not update the completed pilot results.', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, default=ROOT / 'outputs/preference_pilot_20261104/diffusion')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/preference_guide_probe_20261104')
    args = parser.parse_args()
    start, cpu_start = time.perf_counter(), time.process_time()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output, run = args.output.resolve(), args.checkpoint_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    episode_path = output / 'episodes.jsonl'
    if episode_path.exists():
        raise FileExistsError('Choose a new output path; refusing to overwrite completed diagnostic rows')
    source = source_manifest()
    cfg = load_preference_config(run / 'config.json')
    checkpoint_hash, config_hash = file_hash(run / 'checkpoint.pt'), file_hash(run / 'config.json')
    checkpoint = torch.load(run / 'checkpoint.pt', map_location='cpu', weights_only=False)
    saved_source = json.loads((run / 'source.json').read_text())
    status, exit_record = (json.loads((run / name).read_text()) for name in ('status.json', 'exit.json'))
    if source['source_sha256'] != checkpoint['source_sha256'] or source['files'] != saved_source['files']:
        raise RuntimeError('Frozen checkpoint scientific source mismatch')
    if status['status'] != 'complete' or exit_record['returncode'] != 0 or checkpoint['train_steps'] != cfg.train_steps:
        raise RuntimeError('The requested checkpoint is not a completed pilot')
    if tuple(cfg.validation_preferences) != (0., .5, 1.) or cfg.device != 'cpu':
        raise ValueError('Diagnostic requires this fixed CPU pilot and declared validation preferences')
    env = InterventionEnv(cfg.n_arms, cfg.n_actions, cfg.budget, cfg.horizon,
                          instance_seed=cfg.instance_seed, seed=cfg.seed + 100)
    if env.instance_hash != checkpoint['instance_hash']:
        raise RuntimeError('Checkpoint instance mismatch')
    task = PreferenceTask(env)
    learner = PreferenceLearner(cfg, task)
    learner.load_state_dict(checkpoint['learner'])
    for name in ('actor', 'target_actor', 'q1', 'q2', 'tq1', 'tq2', 'guide'):
        getattr(learner, name).eval()
    models_before = models_hash(learner)
    rng_names = ('generator', 'preference_generator', 'guide_generator', 'behavior_generator')
    rng_before = {name: getattr(learner, name).get_state().clone() for name in rng_names}
    updates_before = (learner.updates, learner.actor_updates, learner.guide_updates)
    design = {'scope': 'post-hoc fixed-checkpoint validation-only mechanism probe; no fitting or heldout tuning',
              'hypotheses': ['With ESS threshold zero SMC telescopes to terminal weighting, exactly under paired RNG.',
                             'Nonzero resampling is necessary for this SMC implementation to change generated paths.',
                             'Within fixed obs/pref/index, a useful guide must distinguish suffix utility variation beyond finite MC noise.'],
              'arms': ['terminal_weighting', 'smc_ess_0', 'smc_ess_0.5', 'smc_ess_1'],
              'preferences': list(cfg.validation_preferences), 'episodes_per_arm_preference': 16,
              'episode_seed_start': 400000, 'calibration_prefixes': 16, 'calibration_suffixes': 128,
              'decision_rules': ['Threshold-zero mismatch is an implementation blocker.',
                                 'Zero resampling establishes equivalence, not guide utility.',
                                 'Report all threshold and calibration outcomes; do not select a new threshold or claim superiority.'],
              'frozen_source_sha256': source['source_sha256'], 'checkpoint_sha256': checkpoint_hash}
    atomic_json(output / 'design.json', design)
    print(json.dumps({'phase': 'verified', 'source_sha256': source['source_sha256'], 'checkpoint_sha256': checkpoint_hash}), flush=True)
    evaluations, raw_by_arm = [], {}
    for weight in cfg.validation_preferences:
        for label, mode, threshold in [('terminal_weighting', 'terminal_weighting', .5),
                                       ('smc_ess_0', 'smc', 0.), ('smc_ess_0.5', 'smc', .5), ('smc_ess_1', 'smc', 1.)]:
            learner.cfg = replace(cfg, guide_ess_fraction=threshold)
            rows = []
            for seed in range(400000, 400016):
                row = evaluate_preference(task, learner, weight, mode, episodes=1, seed_start=seed)
                row.update(arm=label, diagnostic_ess_fraction=threshold if mode == 'smc' else None)
                append_json(episode_path, row)
                rows.append(row)
            raw_by_arm[(weight, label)] = rows
            summary = aggregate(rows)
            summary.update(weight=weight, arm=label, diagnostic_ess_fraction=threshold if mode == 'smc' else None)
            evaluations.append(summary)
            print(json.dumps({'phase': 'evaluated', 'weight': weight, 'arm': label, 'return_mean': summary['return_mean'],
                              'resampling_events': summary['compute_totals']['resampling_events'], 'wall_seconds': summary['eval_seconds']}), flush=True)
    learner.cfg = cfg
    fields = ('component_returns', 'scalarized_returns', 'original_returns', 'episode_lengths', 'mean_action_count', 'unique_actions')
    exact = all(all(a[key] == b[key] for key in fields) for weight in cfg.validation_preferences
                for a, b in zip(raw_by_arm[(weight, 'terminal_weighting')], raw_by_arm[(weight, 'smc_ess_0')]))
    if not exact:
        raise AssertionError('Threshold-zero SMC does not exactly match terminal weighting')
    paired = []
    for weight in cfg.validation_preferences:
        base = np.asarray([r['scalarized_returns'][0] for r in raw_by_arm[(weight, 'terminal_weighting')]])
        for label in ('smc_ess_0', 'smc_ess_0.5', 'smc_ess_1'):
            delta = np.asarray([r['scalarized_returns'][0] for r in raw_by_arm[(weight, label)]]) - base
            paired.append({'weight': weight, 'arm': label, 'minus_terminal_per_episode': delta.tolist(),
                           'mean_difference': float(delta.mean()), 'sample_sd': float(delta.std(ddof=1)),
                           'episode_mean_standard_error': float(delta.std(ddof=1) / math.sqrt(len(delta)))})
    calibrated = calibration(learner, checkpoint, cfg, output)
    assert source_manifest()['source_sha256'] == source['source_sha256']
    assert file_hash(run / 'checkpoint.pt') == checkpoint_hash and file_hash(run / 'config.json') == config_hash
    assert models_hash(learner) == models_before
    assert all(torch.equal(getattr(learner, name).get_state(), state) for name, state in rng_before.items())
    assert (learner.updates, learner.actor_updates, learner.guide_updates) == updates_before
    result = {'scope': design['scope'], 'provenance': {
        'source_sha256': source['source_sha256'], 'analysis_script_sha256': file_hash(__file__),
        'checkpoint_path': str(run / 'checkpoint.pt'), 'checkpoint_sha256': checkpoint_hash,
        'config_sha256': config_hash, 'instance_hash': env.instance_hash,
        'checkpoint_train_steps': checkpoint['train_steps'], 'checkpoint_env_steps': checkpoint['env_steps'],
        'checkpoint_updates': learner.updates, 'checkpoint_guide_updates': learner.guide_updates,
        'model_tensors_sha256': models_before}, 'design': design, 'evaluations': evaluations,
        'paired_differences': paired, 'calibration': calibrated,
        'sanity': {'smc_zero_exact_episode_outputs': exact, 'source_unchanged': True,
                   'checkpoint_unchanged': True, 'config_unchanged': True, 'model_tensors_unchanged': True,
                   'learner_rngs_unchanged': True, 'update_counts_unchanged': True},
        'runtime': {'wall_seconds': time.perf_counter() - start, 'process_cpu_seconds': time.process_time() - cpu_start,
                    'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    'torch_threads': torch.get_num_threads(), 'torch_version': torch.__version__}}
    atomic_json(output / 'summary.json', result)
    write_report(result, output)
    print(json.dumps({'phase': 'complete', 'output': str(output), 'calibration_groups': len(calibrated['groups']),
                      'wall_seconds': result['runtime']['wall_seconds'], 'sanity': result['sanity']}), flush=True)


if __name__ == '__main__':
    main()
