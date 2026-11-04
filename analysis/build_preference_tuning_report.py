"""Audit all tuning/replication artifacts and render the fixed-decision study."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solvercell.storage import atomic_json


def read(path): return json.loads(path.read_text())


def main():
    out = ROOT / 'reports/preference_tuning_20261104'
    screen_path = ROOT / 'experiments/preference_tuning_20261104/manifest.json'
    repeat_path = ROOT / 'experiments/preference_replication_20261104/manifest.json'
    selection = read(out / 'selection.json')
    assert selection['heldout_used'] is False
    screen, repeat = read(screen_path), read(repeat_path)
    assert screen['expected_source_sha256'] == repeat['expected_source_sha256'] == selection['source_sha256']
    audits = out / 'artifact_audits'
    audits.mkdir(exist_ok=True)
    manifests = [('screen', screen_path)]
    for seed in (1, 2):
        subset = dict(repeat)
        subset['jobs'] = [job for job in repeat['jobs'] if read(ROOT / job['config'])['seed'] == seed]
        assert len(subset['jobs']) == 2
        path = audits / f'seed{seed}_manifest.json'
        atomic_json(path, subset)
        manifests.append((f'seed{seed}', path))
    studies = {}
    for name, path in manifests:
        subprocess.run([sys.executable, '-m', 'tools.report_preference', '--manifest', str(path),
                        '--output', str(audits / name)], cwd=ROOT, check=True)
        studies[name] = read(audits / name / 'study.json')
    screened = {r['id']: r for r in studies['screen']['runs']}
    repeats = [run for name in ('seed1', 'seed2') for run in studies[name]['runs']]
    selected = {method: screened[row['id']] for method, row in selection['selected'].items()}
    comparisons = []
    for seed in (0, 1, 2):
        rows = [r for r in (list(selected.values()) if seed == 0 else repeats) if r['config']['seed'] == seed]
        sphere = next(r for r in rows if r['config']['method'] == 'preference_sphere')
        diff = next(r for r in rows if r['config']['method'] == 'preference_diffusion')
        comparisons.append({'seed': seed, 'scope': 'discovery' if seed == 0 else 'new_training_seed',
            'sphere': sphere['metrics'], 'diffusion': diff['metrics'],
            'last_window_diff_minus_sphere': diff['metrics']['last_window_return']-sphere['metrics']['last_window_return'],
            'auc_diff_minus_sphere': diff['metrics']['validation_auc']-sphere['metrics']['validation_auc'],
            'heldout_direct_diff_minus_sphere': diff['metrics']['heldout_return_mean']-sphere['metrics']['heldout_return_mean']})
    guide = read(out / 'guide_check/summary.json')
    assert guide['base_equivalence'] == 'bitwise_equal' and guide['identical_calibration_references']
    result = {'status': 'complete', 'scope': 'one instance; eight discovery arms and four fixed new-seed replications',
        'source_sha256': selection['source_sha256'], 'selection': selection,
        'screen': studies['screen'], 'replications': {name: studies[name] for name in ('seed1', 'seed2')},
        'comparisons': comparisons, 'guide_check': {'path': 'guide_check/summary.json',
            'base_equivalence': guide['base_equivalence'], 'identical_calibration_references': True},
        'new_seed_summary': {key: {'mean': float(np.mean([r[key] for r in comparisons[1:]])),
            'sample_sd': float(np.std([r[key] for r in comparisons[1:]], ddof=1)), 'training_seeds': [1, 2]}
            for key in ('last_window_diff_minus_sphere', 'auc_diff_minus_sphere', 'heldout_direct_diff_minus_sphere')}}
    atomic_json(out / 'study.json', result)
    lines = ['# Preference tuning: completed controlled study', '',
        'All12 runs passed complete config/source/exit/checkpoint/curve/trajectory audits. '
        'The screen used training seed0; selected configurations were frozen using validation only before new-seed replication or final-result reporting. '
        'The two new training seeds are1/2, on the same instance. No statistical significance or cross-instance generalization is claimed.', '',
        f"Scientific source: `{selection['source_sha256']}`. The prespecified design is [here](../../docs/PREFERENCE_TUNING_V2.md).", '',
        '## Complete discovery screen', '',
        'All arms use4000 updates,256 warmup,9 full validation checkpoints,8 actor NFE and the same validation/final episode seed ranges. '
        'Last-window means checkpoints3500/4000. Final unseen-weight means are descriptive and did not select configurations.', '',
        '| Arm | Direct validation last-window | Validation AUC | Final unseen direct | Wall seconds |',
        '|---|---:|---:|---:|---:|']
    for job in screen['jobs']:
        run = screened[job['id']];m=run['metrics']
        lines.append(f"| {run['id']} | {m['last_window_return']:.3f} | {m['validation_auc']:.3f} | {m['heldout_return_mean']:.3f} | {m['wall_seconds']:.1f} |")
    lines += ['', '## Frozen validation-only selection', '']
    for method, row in selection['selected'].items():
        lines.append(f"- {method}: `{row['id']}` — actor temperature{row['temperature']}, diffusion beta-end{row['diffusion_beta_end']}; guide objective remains original absolute MSE.")
    lines += ['', 'Nonreference promotion required at least2% last-window gain and AUC at least98% of method reference. '
        'This is a discovery rule, not a scientific significance threshold. Sphere and diffusion each retain their strongest eligible validation setting. '
        'The grouped/centered guide arms do not add independent base candidates.', '',
        '## Fixed configurations on discovery and new training seeds', '',
        '| Training seed | Scope | Sphere last-window | Diffusion last-window | Diffusion minus sphere | Sphere AUC | Diffusion AUC | Final unseen direct difference |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in comparisons:
        a,b=row['sphere'],row['diffusion']
        lines.append(f"| {row['seed']} | {row['scope']} | {a['last_window_return']:.3f} | {b['last_window_return']:.3f} | "
            f"{row['last_window_diff_minus_sphere']:+.3f} | {a['validation_auc']:.3f} | {b['validation_auc']:.3f} | {row['heldout_direct_diff_minus_sphere']:+.3f} |")
    new=result['new_seed_summary']
    lines += ['', f"Across only the two new training seeds, last-window diffusion-minus-sphere mean is{new['last_window_diff_minus_sphere']['mean']:+.3f}, "
        f"sample SD{new['last_window_diff_minus_sphere']['sample_sd']:.3f}. Final unseen direct difference mean is{new['heldout_direct_diff_minus_sphere']['mean']:+.3f}, "
        f"sample SD{new['heldout_direct_diff_minus_sphere']['sample_sd']:.3f}. These are small-n descriptive summaries, not confidence intervals. "
        'Per-episode observations within one trained model are not independent training seeds.', '',
        '## Relative guide mechanism', '',
        'The fixed beta-end.6/temperature.1 triplet has bitwise-identical base models, optimizers, training environment/replay/RNG and direct validation trajectories. '
        'Its18 actual-suffix calibration reference groups are also identical across guide losses. '
        'Original vs grouped absolute examines grouped sampling; grouped absolute vs centered isolates the objective. '
        'This triplet is a mechanism fixture and is not automatically the selected base configuration.', '',
        '[All calibration cases, paired validation returns, resampling and compute](guide_check/RESULTS.md). '
        'Learned-critic suffix calibration does not establish real-return accuracy. SMC adds guide work to the same particle/terminal-query budget. '
        'No guide default or additional sweep is selected from final results.', '',
        '## Reproducibility and limits', '',
        '- [Selection lock](selection.json); [full machine-readable study](study.json); [curves](tuning_and_replication.png); [PDF](tuning_and_replication.pdf).',
        '- [Eight-arm audit](artifact_audits/screen/RESULTS.md); [seed1 audit](artifact_audits/seed1/RESULTS.md); [seed2 audit](artifact_audits/seed2/RESULTS.md).',
        '- [90-test log](tests.log); [preflight queue audit](preflight/RESULTS.md); runtime provenance and final remote state are in`runtime/`.',
        '- Fixed transition budgets retain partial final training episodes in checkpoints; every validation/final trajectory completes H20.',
        '- Equal actor NFE is not equal full compute. Guide fitting, suffix labels, Q/decoder calls and measured wall time are reported. Counters exclude independent baselines/warmup decoding.',
        '- Fresh final seeds600000–600007 and unseen interpolation weights do not participate in configuration selection. One benchmark instance and two new seeds limit conclusions.',
        '- All failed/degraded arms are retained. Original historical manifests and sources remain immutable. No larger guide sweep follows without the declared mechanism gate.', '']
    (out / 'RESULTS.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for name,row in selection['all_candidates'].items():
        curve=np.asarray(row['validation_curve'])
        axes[0,0].plot(curve[:,0],curve[:,1],label=name)
    colors={'sphere':'tab:blue','diffusion':'tab:orange'}
    for method in ('sphere','diffusion'):
        model_runs=[selected[method],*[r for r in repeats if r['config']['method']=='preference_'+method]]
        for run in model_runs:
            vals={}
            for row in run['evaluations']:
                if row['split']=='validation' and row['mode']=='direct':
                    vals.setdefault(row['train_steps'],[]).append(row['return_mean'])
            x=sorted(vals);y=[np.mean(vals[k]) for k in x];seed=run['config']['seed']
            axes[0,1].plot(x,y,color=colors[method],linestyle=[':', '-', '--'][seed],label=f'{method} seed{seed}')
        for component,ax in enumerate(axes[1]):
            fresh=[r for r in model_runs if r['config']['seed'] in (1,2)]
            weights=sorted(fresh[0]['config']['heldout_preferences'])
            values=np.asarray([[next(row['component_return_mean'][component] for row in run['evaluations'] if row['split']=='heldout'
                and row['mode']=='direct' and row['low_response_weight']==weight) for weight in weights] for run in fresh])
            ax.plot(weights,values.mean(0),marker='o',color=colors[method],label=f'{method}: new-seed mean')
            for seed,y in zip((1,2),values):ax.plot(weights,y,color=colors[method],alpha=.35,linestyle='--',label=f'{method} seed{seed}')
    axes[0,0].set(title='Seed0 complete direct validation curves',xlabel='Updates',ylabel='Mean scalarized return')
    axes[0,1].set(title='Frozen selected configurations',xlabel='Updates',ylabel='Mean scalarized validation return')
    axes[1,0].set(title='Low-response cohort: new training seeds',xlabel='Unseen low-response preference',ylabel='Raw component return')
    axes[1,1].set(title='High-response cohort: new training seeds',xlabel='Unseen low-response preference',ylabel='Raw component return')
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'tuning_and_replication.png',dpi=170);fig.savefig(out/'tuning_and_replication.pdf');plt.close(fig)
    print(json.dumps({'status':'complete','runs':12,'new_seed_differences':new}))


if __name__ == '__main__':main()
