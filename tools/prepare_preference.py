"""Prepare a fresh two-model preference pilot; does not launch workers."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solvercell.preference_config import PreferenceConfig
from solvercell.storage import atomic_json, source_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-id', required=True)
    parser.add_argument('--train-steps', type=int, default=1000)
    parser.add_argument('--warmup-steps', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if Path(args.campaign_id).name != args.campaign_id or args.campaign_id in {'.', '..'}:
        raise ValueError('Campaign id must be one nonempty directory name')
    root = Path(__file__).resolve().parents[1]
    folder = root / 'experiments' / args.campaign_id
    folder.mkdir(parents=True, exist_ok=False)
    jobs = []
    for name in ('sphere', 'diffusion'):
        cfg = PreferenceConfig(method='preference_' + name, train_steps=args.train_steps,
                               warmup_steps=args.warmup_steps, seed=args.seed).validate()
        path = folder / (name + '.json')
        atomic_json(path, cfg.to_dict())
        jobs.append({'id': name, 'config': str(path.relative_to(root)),
                     'output': f'outputs/{args.campaign_id}/{name}',
                     'config_sha256': hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()})
    atomic_json(folder / 'manifest.json', {'campaign_id': args.campaign_id,
                'schema': 'preference_pilot_v1', 'runner_module': 'tools.run_preference',
                'expected_source_sha256': source_manifest()['source_sha256'], 'jobs': jobs,
                'max_parallel': 2, 'device': 'cpu', 'cpu_sets': [[24], [25]],
                'scope': f'two model seed{args.seed} implementation pilot; direct/terminal/SMC share each checkpoint',
                'not_compatible_with_legacy_campaign_runner': True})
    print(folder / 'manifest.json')


if __name__ == '__main__':
    main()
