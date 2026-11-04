"""Lock base configurations using complete validation curves only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solvercell.storage import atomic_json


def select(manifest_path, destination):
    if destination.exists():
        raise FileExistsError('Selection is immutable; use a fresh destination')
    manifest = json.loads(manifest_path.read_text())
    candidates = {}
    for job in manifest['jobs']:
        path = ROOT / job['output']
        cfg = json.loads((ROOT / job['config']).read_text())
        assert cfg == json.loads((path / 'config.json').read_text())
        assert hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest() == job['config_sha256']
        status, exit_record = (json.loads((path / name).read_text()) for name in ('status.json', 'exit.json'))
        assert status['status'] == 'complete' and exit_record['returncode'] == 0
        assert status['source_sha256'] == exit_record['source_sha256'] == manifest['expected_source_sha256']
        if cfg['guide_objective'] != 'absolute':
            continue
        points = {}
        for line in (path / 'evaluations.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['split'] != 'validation' or row['mode'] != 'direct':
                continue
            step = row['train_steps']
            points.setdefault(step, {})[row['low_response_weight']] = row['return_mean']
        expected = [0, *range(cfg['eval_interval'], cfg['train_steps'], cfg['eval_interval']), cfg['train_steps']]
        assert sorted(points) == expected
        assert all(sorted(values) == sorted(cfg['validation_preferences']) for values in points.values())
        curve = [(step, float(np.mean(list(points[step].values())))) for step in expected]
        y = np.asarray([value for _, value in curve]); x = np.asarray(expected)
        assert np.isfinite(y).all()
        window = max(1, int(np.ceil(.2 * cfg['train_steps'] / cfg['eval_interval'])))
        candidates[job['id']] = {'id': job['id'], 'config': job['config'], 'config_sha256': job['config_sha256'],
            'method': cfg['method'], 'temperature': cfg['temperature'], 'diffusion_beta_end': cfg['diffusion_beta_end'],
            'last_window_return': float(y[-window:].mean()), 'validation_auc': float(np.trapz(y, x)/cfg['train_steps']),
            'last_window_steps': expected[-window:], 'validation_curve': curve, 'validation_by_preference': points}
    chosen = {}
    for method, reference in (('sphere', 'sphere_t050'), ('diffusion', 'diff_b020_t050')):
        anchor = candidates[reference]
        group = [row for row in candidates.values() if row['method'] == 'preference_' + method]
        for row in group:
            row['eligible'] = (row['id'] == reference or (
                row['last_window_return'] >= 1.02 * anchor['last_window_return']
                and row['validation_auc'] >= .98 * anchor['validation_auc']))
        eligible = [row for row in group if row['eligible']]
        chosen[method] = max(eligible, key=lambda row: (row['last_window_return'], row['validation_auc'], row['id'] == reference))
    result = {'scope': 'seed0 discovery selection; final values excluded from all decisions',
        'source_sha256': manifest['expected_source_sha256'],
        'parent_manifest': str(manifest_path), 'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'decision_rule': 'Rank eligible arms by last-window direct validation, then AUC. Nonreference eligibility: >=2% endpoint-window gain, AUC>=98% of method reference.',
        'selected': chosen, 'all_candidates': candidates, 'heldout_used': False}
    atomic_json(destination, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    r = select(args.manifest, args.output)
    print(json.dumps({'selected': {key: value['id'] for key, value in r['selected'].items()}, 'heldout_used': False}))
