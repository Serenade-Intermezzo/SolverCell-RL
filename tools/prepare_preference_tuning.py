"""Freeze a declared eight-arm screen or the validation-selected replications."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from solvercell.preference_config import load_preference_config, debug_preference_config
from solvercell.storage import atomic_json, source_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign-id', required=True)
    p.add_argument('--selection', type=Path)
    p.add_argument('--debug', action='store_true')
    args = p.parse_args()
    if Path(args.campaign_id).name != args.campaign_id or args.campaign_id in {'.', '..'}:
        raise ValueError('Use a new single-directory campaign name')
    root = Path(__file__).resolve().parents[1]
    folder = root / 'experiments' / args.campaign_id
    folder.mkdir(parents=True, exist_ok=False)
    configs, parents = [], {}
    if args.selection:
        selection = json.loads(args.selection.read_text())
        assert selection['source_sha256'] == source_manifest()['source_sha256']
        for seed in (1, 2):
            for method in ('sphere', 'diffusion'):
                chosen = selection['selected'][method]
                parent = root / chosen['config']
                cfg = replace(load_preference_config(parent), seed=seed)
                configs.append((f'{method}_seed{seed}', cfg))
        parents['selection'] = {'path': str(args.selection),
                                'sha256': hashlib.sha256(args.selection.read_bytes()).hexdigest()}
    else:
        for method in ('sphere', 'diffusion'):
            path = root / f'experiments/preference_pilot_20261104/{method}.json'
            parents[method] = {'path': str(path.relative_to(root)),
                               'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            base = replace(load_preference_config(path), train_steps=4000,
                           eval_interval=500, checkpoint_interval=500, eval_episodes=8,
                           final_seed_start=600000, profile='preference_tuning_v2')
            for temperature, tag in ((.5, '050'), (.1, '010')):
                if method == 'sphere':
                    configs.append((f'sphere_t{tag}', replace(base, temperature=temperature)))
                else:
                    for beta, beta_tag in ((.2, '020'), (.6, '060')):
                        configs.append((f'diff_b{beta_tag}_t{tag}', replace(
                            base, temperature=temperature, diffusion_beta_end=beta)))
            if method == 'diffusion':
                for objective, tag in (('grouped_absolute', 'group'), ('centered', 'center')):
                    configs.append((f'{tag}_b060_t010', replace(base, temperature=.1,
                        diffusion_beta_end=.6, guide_objective=objective)))
    jobs = []
    for index, (name, cfg) in enumerate(configs):
        if args.debug:
            cfg = debug_preference_config(cfg)
        cfg.validate()
        path = folder / f'{name}.json'
        atomic_json(path, cfg.to_dict())
        jobs.append({'id': name, 'config': str(path.relative_to(root)),
                     'output': f'outputs/{args.campaign_id}/{name}', 'cpu_set': [24 + index % 2],
                     'config_sha256': hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()})
    atomic_json(folder / 'manifest.json', {'campaign_id': args.campaign_id,
        'schema': 'preference_tuning_v2', 'runner_module': 'tools.run_preference',
        'expected_source_sha256': source_manifest()['source_sha256'], 'jobs': jobs,
        'max_parallel': 2, 'device': 'cpu', 'cpu_sets': [[24], [25]], 'parents': parents,
        'stage': 'debug' if args.debug else ('replication' if args.selection else 'screen'),
        'design': 'docs/PREFERENCE_TUNING_V2.md',
        'scope': 'declared source-frozen independent two-lane queue; no retries; retain all arms'})
    print(folder / 'manifest.json')


if __name__ == '__main__':
    main()
