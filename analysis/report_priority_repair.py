"""Audit the priority repair campaign and enforce its staged-guidance gate."""
import argparse,hashlib,json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from solvercell.storage import atomic_json


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    m=read(args.manifest);data=np.load(ROOT/m['data_path']);assert sha(ROOT/m['data_path'])==m['data_sha256']
    assert len(data['train_obs'])==64 and len(data['validation_obs'])==32
    assert not {r.tobytes() for r in data['train_obs']} & {r.tobytes() for r in data['validation_obs']}
    rows=[];aggregate=[];curves={};archives={};checkpoints={};rngs={};workers={};fullchecks=[]
    for job in m['jobs']:
        path=ROOT/job['output'];cfg=read(ROOT/job['config']);summary=read(path/'summary.json');worker=read(path/'worker.json');inv=read(path/'invocation.json')
        assert read(path/'exit.json')['returncode']==0 and read(path/'status.json')['status']==summary['status']=='complete'
        assert sha(ROOT/job['config'])==job['config_sha256']==inv['config_sha256']
        assert inv['runner_sha256']==m['runner_sha256'] and inv['manifest_sha256']==sha(args.manifest)
        assert summary['source_sha256']==read(path/'source.json')['source_sha256']==m['source_sha256'];assert summary['data_sha256']==m['data_sha256']
        assert worker['comm']=='llmagent' and worker['cpu_affinity'] in ([24],[25])
        assert summary['steps']==cfg['steps']==2000 and summary['parameters']==30408
        assert [r['step'] for r in summary['curves']]==[0,500,1000,1500,2000]
        assert len((path/'training.jsonl').read_text().splitlines())==20
        curves[job['id']]=summary['curves'];workers[job['id']]=worker
        cp=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=False);assert cp['steps']==2000 and cp['data_sha256']==m['data_sha256'];assert cp['config']==cfg
        assert all(torch.isfinite(v).all() for v in cp['actor'].values());checkpoints[job['id']]=sha(path/'checkpoint.pt')
        rngs[job['id']]=cp['data_generator']
        expected_modes=[False] if cfg['method']=='sphere' else [False,True]
        assert len(summary['evaluations'])==32*4*len(expected_modes)
        counts=[];targets=[]
        for index,row in enumerate(summary['evaluations']):
            assert row['samples']==8192 and len(row['counts'])==268 and sum(row['counts'])==8192
            a=np.array(row['counts'])/8192;b=np.array(row['target_probabilities']);assert np.isfinite(b).all() and np.allclose(b.sum(),1)
            assert np.allclose(row['TV'],np.abs(a-b).sum()/2)
            assert row['actor_rows']==8192*8 and row['decoder_rows']==8192 and row['q_rows']==0
            counts.append(row['counts']);targets.append(row['target_probabilities'])
            rows.append({'job':job['id'],'method':cfg['method'],'prediction':cfg['prediction'],'seed':cfg['seed'],'histogram_row':index,
                         **{k:v for k,v in row.items() if k not in ('counts','target_probabilities')}})
        archives[job['id']+'_counts']=np.array(counts,np.int32);archives[job['id']+'_targets']=np.array(targets,np.float64)
        assert summary['compute_totals']['sampling_actor_rows']==len(summary['evaluations'])*8192*8
        assert summary['compute_totals']['training_actor_rows']==2000*64
        for noise in expected_modes:
            r=[v for v in rows if v['job']==job['id'] and v['final_noise']==noise]
            aggregate.append({'job':job['id'],'method':cfg['method'],'prediction':cfg['prediction'],'seed':cfg['seed'],'final_noise':noise,
                'TV':float(np.mean([v['TV'] for v in r])),'target_sampling_TV_floor':float(np.mean([v['target_sampling_TV_floor_draw'] for v in r])),
                'target_minus_sampled_utility':float(np.mean([v['target_minus_sampled_utility'] for v in r])),'wall_seconds':summary['wall_seconds']})
        fullchecks.append({'job':job['id'],'checkpoint_sha256':checkpoints[job['id']],'compute_totals':summary['compute_totals']})
    for seed in (0,1,2):
        values=[rngs[f'{name}_s{seed}'] for name in ('sphere_epsilon','diffusion_epsilon','diffusion_x0')]
        assert all(torch.equal(values[0],v) for v in values[1:]),'Different endpoint training draws'
    # No new stages or parameter choices may bypass the prespecified fit gate.
    sphere=np.mean([r['TV'] for r in aggregate if r['method']=='sphere']);gates={}
    for prediction in ('epsilon','x0'):
        relevant=[r for r in aggregate if r['method']=='diffusion' and r['prediction']==prediction and not r['final_noise']]
        tv=float(np.mean([r['TV'] for r in relevant]));gates[prediction]={'mean_heldout_TV':tv,'sphere_mean_TV':float(sphere),
            'difference_vs_sphere':tv-float(sphere),'passed':bool(tv<=m['gate']['diffusion_TV_max'] and tv-sphere<=m['gate']['diffusion_minus_sphere_TV_max'])}
    calibration=read(out/'task_audit.json');assert calibration['tradeoff_gate']['passed']
    decision={'task_gate_passed':True,'actor_fit_gates':gates,'staged_guidance_permitted':any(g['passed'] for g in gates.values()),
        'scope':'Prespecified fixed-distribution fit gate, not a statistical superiority claim; no heldout tuning.'}
    decision['action']='prepare a separately prespecified guidance study' if decision['staged_guidance_permitted'] else 'do not launch staged guidance; unresolved finite-step actor fitting remains the next research node'
    atomic_json(out/'decision.json',decision)
    np.savez_compressed(out/'histograms.npz',**archives)
    study={'status':'complete','scope':m['scope'],'source_sha256':m['source_sha256'],'manifest_sha256':sha(args.manifest),'data_sha256':m['data_sha256'],
        'aggregates':aggregate,'evaluations':rows,'curves':curves,'workers':workers,'artifacts':fullchecks,'decision':decision,'frozen_component_scales':calibration['frozen_component_scales'],
        'limitations':'Fixed model-informed finite target, not optimal RL. Exact target probabilities enable diagnostic TV; does not imply tractable enumeration on general benchmarks. Histogram TV has finite-sample bias; target-only floors retained.'}
    atomic_json(out/'study.json',study)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.8));curve=np.array(calibration['normalized_expert_curve'])
    axes[0].plot(curve[:,0],curve[:,1],'o-')
    for w,point in zip(calibration['requested_preferences'],curve):axes[0].annotate(str(w),point,fontsize=8)
    axes[0].set(xlabel='Raw low-response return',ylabel='Raw high-response return',title='Feasible model-informed reference; new episodes')
    labels=[];means=[];std=[]
    for method,prediction,noise in [('sphere','epsilon',False),('diffusion','epsilon',False),('diffusion','epsilon',True),('diffusion','x0',False),('diffusion','x0',True)]:
        values=[r['TV'] for r in aggregate if (r['method'],r['prediction'],r['final_noise'])==(method,prediction,noise)]
        labels.append('sphere' if method=='sphere' else prediction+(' noisy' if noise else ' clean'));means.append(np.mean(values));std.append(np.std(values,ddof=1))
    axes[1].bar(np.arange(5),means,yerr=std,capsize=3);axes[1].axhline(.15,color='red',ls=':',label='Prespecified fit gate')
    axes[1].axhline(np.mean([r['target_sampling_TV_floor'] for r in aggregate]),color='gray',ls='--',label='Target sampling floor')
    axes[1].set(xticks=np.arange(5),xticklabels=labels,ylabel='Mean heldout decision TV',title='8192 samples/context; bars show model-seed SD');axes[1].tick_params(axis='x',rotation=25);axes[1].legend(fontsize=7)
    for family in ('sphere_epsilon','diffusion_epsilon','diffusion_x0'):
        values=np.array([[p['training_probe_TV'] for p in curves[f'{family}_s{s}']] for s in (0,1,2)])
        axes[2].plot([0,500,1000,1500,2000],values.mean(0),'o-',label=family)
    axes[2].set(xlabel='Fixed optimizer steps',ylabel='Train probe TV (512 samples)',title='Fixed-budget curves; no best-step selection');axes[2].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'repair_results.png',dpi=170);fig.savefig(out/'repair_results.pdf');plt.close(fig)
    lines=['# Priority-ordered repair: completed bounded campaign','',
        '104 tests passed; repaired online harness debug and all three fixed-target preflights passed. All nine formal actor-fitting jobs completed2000 optimizer steps. '
        'Scientific source: `'+m['source_sha256']+'`. Source/data/config/runner, actual CPU worker identity, counters, finite checkpoints and common endpoint-draw RNGs passed audit.','',
        '## Task controllability and calibrated objectives','',
        f"Native open-cell action catalog:268, checked against10000 random proposals. Opposing feasible reference policies gain{calibration['tradeoff_gate']['low_extreme_gain']:.3f}/{calibration['tradeoff_gate']['high_extreme_gain']:.3f} raw units. "
        f"Scales frozen from32 calibration episodes per extreme: {calibration['frozen_component_scales']}. Validation uses separate seeds. These are model-informed references, not exact optimal experts.",'',
        'Raw rewards/replay are preserved. New v4 scalarization is requested preference dot(raw component / fixed scale), and both critic TD and evaluation use it. '
        'It is a declared objective extension; new normalized scores are not comparable to old raw-objective scores.','',
        '## Common fixed-endpoint results','',
        '| Seed | Sphere TV | Epsilon clean TV | Epsilon noisy TV | x0 clean TV | x0 noisy TV |','|---:|---:|---:|---:|---:|---:|']
    for seed in (0,1,2):
        def value(method,prediction,noise):return next(r['TV'] for r in aggregate if (r['seed'],r['method'],r['prediction'],r['final_noise'])==(seed,method,prediction,noise))
        vals=[value('sphere','epsilon',False),value('diffusion','epsilon',False),value('diffusion','epsilon',True),value('diffusion','x0',False),value('diffusion','x0',True)]
        lines.append('|'+str(seed)+'|'+'|'.join(f'{v:.4f}' for v in vals)+'|')
    lines+=['','Mean target-only sampling TV is about.072. Each case has8192 samples; target probabilities are exact finite-catalog values. '
        'Final contexts32 and preferences4 were not used for optimizer updates or checkpoints. All seed-level values and degraded x0/noise controls remain. '
        'TV is measured against a fixed soft endpoint target; negative target-minus-model utility gap can coexist with poor distribution fit and is not evidence of optimal RL.','',
        '## Staged-guidance decision','',json.dumps(decision,indent=2),'',
        'No production default was silently changed: legacy configs retain historical noisy epsilon/unit-scale semantics. '
        'The explicit v4 factory declares clean terminal sampling and frozen calibration. Old guide checkpoints are rejected when chain/objective semantics differ. '
        'Positive-variance Gaussian log-density calls reject deterministic terminal kernels; actual suffix and SMC labels follow the declared chain.','',
        '[Design](../../docs/PRIORITY_REPAIR_V4.md) · [Decision](decision.json) · [All context metrics and audits](study.json) · [Exact histograms](histograms.npz) · [Figure](repair_results.png)','']
    (out/'RESULTS.md').write_text('\n'.join(lines));print(json.dumps({'jobs':len(m['jobs']),'decision':decision}))


if __name__=='__main__':main()
