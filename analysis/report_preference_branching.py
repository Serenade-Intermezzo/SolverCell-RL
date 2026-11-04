"""Audit and summarize read-only shared-prefix policy experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from solvercell.storage import atomic_json


def read(path):return json.loads(Path(path).read_text())


def audit(path,manifest,stage,seed,selection):
    row=read(path/'summary.json');invocation=read(path/'invocation.json');worker=read(path/'worker.json')
    assert row['status']==read(path/'status.json')['status']=='complete'
    assert read(path/'exit.json')['returncode']==0
    assert row['stage']==stage and row['seed']==seed
    assert worker['comm']=='llmagent'
    assert row['source_sha256']==invocation['source_sha256']==manifest['expected_source_sha256']
    assert invocation['analysis_script_sha256']==manifest['runner_script_sha256']
    weights=manifest['screen_preferences'] if stage=='screen' else manifest['final_preferences']
    for parent in row['parents']:
        assert parent['models_rngs_updates_checkpoint_unchanged']
        assert parent['parent']==manifest['parents'][parent['model']]
        for weight in weights:
            values=[v for v in row['evaluations'] if v['model']==parent['model'] and v['low_response_weight']==weight]
            expected=len(manifest['independent_candidates'])+(len(manifest['topologies']) if stage=='screen' else 1) if parent['model'].startswith('diffusion') else len(manifest['independent_candidates'])
            assert len(values)==expected
            for value in values:
                assert value['episodes']==manifest['episodes'] and value['episode_lengths']==[20]*manifest['episodes']
                assert value['seed_start']==manifest['screen_seed_start' if stage=='screen' else 'final_seed_start']
                component=np.asarray(value['component_returns']);pref=np.asarray(value['preference'])
                assert np.isfinite(component).all() and np.allclose(component@pref,value['scalarized_returns'])
                assert np.allclose(component.mean(1),value['original_returns'],atol=1e-3)
                assert value['feasibility_rate']==1
                mode=value['mode'];decisions=20*manifest['episodes'];cost=value['compute_totals']
                if mode.startswith('tree'):
                    spec=manifest['topologies'][mode]
                    nfe=spec['roots']*spec['branch_index']+8*(8-spec['branch_index'])
                    assert cost['nfe']==decisions*nfe and cost['q_calls']==18*decisions and cost['decoder_calls']==9*decisions
                    assert cost['guide_calls']==0
                elif mode.startswith('is'):
                    count=int(mode[2:]);assert cost['nfe']==decisions*count*8
                    assert cost['q_calls']==(2+2*count)*decisions and cost['decoder_calls']==(count+1)*decisions
                else:assert cost['nfe']==decisions*8 and cost['q_calls']==0 and cost['decoder_calls']==decisions
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();out=args.output;manifest=read(args.manifest);selection=read(out/'selection.json')
    assert selection['heldout_used'] is False and selection['source_sha256']==manifest['expected_source_sha256']
    screen=audit(ROOT/f"outputs/{manifest['campaign_id']}/screen",manifest,'screen',0,selection)
    finals=[audit(ROOT/f"outputs/{manifest['campaign_id']}/final_seed{seed}",manifest,'final',seed,selection) for seed in (0,1,2)]
    chosen=selection['evaluated_topology'];comparisons=[]
    for result in finals:
        values=[r for r in result['evaluations'] if r['model'].startswith('diffusion')]
        ref=sorted([r for r in values if r['mode']=='is8'],key=lambda r:r['low_response_weight'])
        branch=sorted([r for r in values if r['mode']==chosen],key=lambda r:r['low_response_weight'])
        diffs=np.mean([np.asarray(a['scalarized_returns'])-b['scalarized_returns'] for a,b in zip(branch,ref)],axis=0)
        comparisons.append({'model_training_seed':result['seed'],'scope':'selection_model' if result['seed']==0 else 'out_of_selection_fixed_model',
            'reference_mean':float(np.mean([r['return_mean'] for r in ref])),
            'branch_mean':float(np.mean([r['return_mean'] for r in branch])),
            'paired_mean_difference':float(diffs.mean()),'episode_mean_se':float(diffs.std(ddof=1)/np.sqrt(len(diffs))),
            'paired_episode_differences':diffs.tolist(),
            'environment_inclusive_wall_ratio':sum(r['eval_seconds'] for r in branch)/sum(r['eval_seconds'] for r in ref),
            'per_preference':[{'weight':a['low_response_weight'],'ratio':a['return_mean']/b['return_mean'],
                              'delta':a['return_mean']-b['return_mean']} for a,b in zip(branch,ref)]})
    study={'status':'complete','scope':'fixed-checkpoint inference only; zero training updates',
        'source_sha256':manifest['expected_source_sha256'],'selection':selection,'screen':screen,
        'finals':finals,'comparisons':comparisons}
    atomic_json(out/'study.json',study)
    lines=['# Shared-prefix preference sampling: completed study','',
        'All runs passed source/parent hash, exit, complete H20 evaluation, scalarization, feasibility and actual compute audits. '
        'The trained v2 models, optimizer/RNG states and checkpoints remain unchanged. No new training was performed.', '',
        f"Source: `{manifest['expected_source_sha256']}`. Parent scientific source: `{manifest['parent_source_sha256']}`. "
        '[Prespecified design](../../docs/PREFERENCE_BRANCHING_V3.md).','',
        '## Seed0 validation screen','',
        '| Topology | Actor NFE | Utility / independent8 | Worst-weight ratio | Median inference latency ratio | Eligible |',
        '|---|---:|---:|---:|---:|---|']
    for name,r in selection['candidates'].items():
        lines.append(f"| {name} | {r['actor_nfe_per_decision']} | {r['utility_fraction_vs_is8']:.4f} | "
            f"{r['minimum_preference_fraction']:.4f} | {r['median_latency_fraction_vs_is8']:.4f} | {r['eligible']} |")
    lines += ['',f"Frozen evaluated topology: **{chosen}**. Promotion eligibility: **{selection['promotion_eligible']}**. "
        f"{selection['reason']}. No final episode or preference selected this topology.",'',
        'Weight ESS measures normalized weight concentration only; it does not account for correlated siblings. '
        'The variance/support audits in study.json separately measure the actual conditional branch structure.', '',
        '## Fresh final episodes at unseen fitting preferences','',
        '| Model training seed | Scope | Independent8 mean | Branch mean | Paired mean difference | Episode-mean SE | Environment-inclusive wall ratio |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for r in comparisons:
        lines.append(f"| {r['model_training_seed']} | {r['scope']} | {r['reference_mean']:.3f} | {r['branch_mean']:.3f} | "
            f"{r['paired_mean_difference']:+.3f} | {r['episode_mean_se']:.3f} | {r['environment_inclusive_wall_ratio']:.3f} |")
    lines += ['', 'Final episode seeds800000–800031 are common across modes and preferences. '
        'Paired SE first averages each episode over the four preferences, preserving within-seed dependence. '
        'Model seeds1/2 were not used for topology selection, but are existing v2 checkpoints, not newly trained replications. '
        'Three known models and one instance cannot establish broad noninferiority or intrinsic sampler speed.', '',
        '## Budget boundary','',
        'Independent8 uses64 actor NFE,18 twin-Q rows and9 decoder rows per action. Branching scores the same8 leaves '
        'with the same Q/decoder budget but fewer actor rows. Independent3/4/5/6/7 controls have lower Q/decoder counts; '
        'matched or bracketing actor NFE is not matched total compute. Direct sphere/diffusion uses only8 actor NFE. '
        'Do not infer superiority to tuned sphere from within-diffusion candidate sharing.', '',
        'The finite importance pool is correlated and normalized selection is not exact utility-tilted sampling. '
        'Root-wise inverse child-count weights preserve unnormalized estimands; consistency requires more independent roots. '
        'All negative topologies and controls are retained.', '',
        '[Selection lock](selection.json) · [Complete study and support/variance audits](study.json) · [Figure](branching_tradeoff.png)', '']
    (out/'RESULTS.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for method in ('sphere','diffusion'):
        parent=next(r for r in screen['parents'] if r['model']==f'{method}_seed0')
        rows=[r for r in screen['evaluations'] if r['model']==f'{method}_seed0']
        names=sorted({r['mode'] for r in rows})
        for name in names:
            vals=[r for r in rows if r['mode']==name];utility=np.mean([r['return_mean'] for r in vals])
            nfe=vals[0]['compute_totals']['nfe']/(manifest['episodes']*20)
            marker='x' if name.startswith('tree') else 'o'
            axes[0].scatter(nfe,utility,marker=marker,color='tab:blue' if method=='sphere' else 'tab:orange')
            if method=='diffusion' and (name in ('is8',chosen)):axes[0].annotate(name,(nfe,utility),fontsize=7)
        for name in names:
            if method=='diffusion' and name.startswith('tree'):
                row=selection['candidates'][name]
                axes[1].scatter(row['median_latency_fraction_vs_is8'],row['utility_fraction_vs_is8'],marker='x')
                axes[1].annotate(name,(row['median_latency_fraction_vs_is8'],row['utility_fraction_vs_is8']),fontsize=6)
    axes[0].set(xlabel='Actor forward rows per action',ylabel='Validation scalarized utility',title='Seed0: independent circles, trees crosses')
    axes[1].axvline(1.,color='gray',linestyle=':');axes[1].axhline(1.,color='gray',linestyle=':')
    axes[1].set(xlabel='Median inference latency / independent8',ylabel='Validation utility / independent8',title='Diffusion branching screen')
    axes[2].errorbar([r['model_training_seed'] for r in comparisons],[r['paired_mean_difference'] for r in comparisons],
        yerr=[r['episode_mean_se'] for r in comparisons],fmt='o',capsize=4)
    axes[2].axhline(0,color='gray',linestyle=':')
    axes[2].set(xticks=[0,1,2],xlabel='Fixed model training seed',ylabel='Branch minus independent8 utility',title='Fresh finals; bars show paired episode SE')
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'branching_tradeoff.png',dpi=170);fig.savefig(out/'branching_tradeoff.pdf');plt.close(fig)
    print(json.dumps({'status':'complete','topology':chosen,'promotion_eligible':selection['promotion_eligible'],'comparisons':comparisons}))


if __name__=='__main__':main()
