"""Audit post-hoc diagnostic artifacts and render their decision-relevant figure."""
import hashlib,json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from solvercell.storage import atomic_json,source_manifest
OUT=ROOT/'reports/method_diagnosis_20261104'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    d=json.loads((OUT/'diagnostics.json').read_text());n=json.loads((OUT/'final_noise.json').read_text())
    study=json.loads((ROOT/'reports/preference_branching_20261104/study.json').read_text())
    for obj,script,design in ((d,'diagnose_preference_method.py','DESIGN.md'),(n,'diagnose_final_noise.py','FINAL_NOISE_DESIGN.md')):
        assert obj['status']=='complete' and obj['source_sha256']==source_manifest()['source_sha256']
        assert obj['script_sha256']==sha(ROOT/'analysis'/script) and obj['design_sha256']==sha(OUT/design)
    assert d['all_parent_models_optimizers_RNGs_updates_checkpoints_unchanged']
    assert d['parent_study_sha256']==sha(ROOT/'reports/preference_branching_20261104/study.json')
    assert len(n['evaluations'])==27 and len(d['critic_probes'])==12
    for parent in n['parents'].values():
        assert parent['all_states_unchanged']
        for filename,digest in parent['parent']['files_sha256'].items():assert sha(ROOT/parent['parent']['output']/filename)==digest
    for r in n['evaluations']:
        assert r['episodes']==32 and r['episode_lengths']==[20]*32 and r['seed_start']==1100000
        assert r['compute_totals']['nfe']==5120 and r['compute_totals']['q_calls']==0 and r['compute_totals']['decoder_calls']==640
        assert np.allclose(np.array(r['component_returns'])@r['preference'],r['scalarized_returns'])
    assert all(r['crossfit_MC_selected_minus_critic_selected']['episode_se'] is None for r in d['critic_probes'])
    # Repeated probe is identical after excluding the corrected cross-fit-SE and tie-aware rank summary.
    old=json.loads((OUT/'runtime/initial_diagnostics_with_naive_crossfit_se.json').read_text())
    assert old['script_sha256']==sha(OUT/'runtime/initial_diagnostic_script.py')
    for a,b in zip(old['critic_probes'],d['critic_probes']):
        assert a['MC_repetitions']==b['MC_repetitions'] and a['critic_values']==b['critic_values']
    variance=[]
    for r in study['screen']['parents'][1]['variance_audit']:
        if r['branch_index']!=6:continue
        out={'replay_row':r['replay_row'],'preference':r['preference']}
        for kind,comp in r['variance_components'].items():
            a,b=comp['between_clipped'],comp['within']
            out[kind]={'same_eight_leaves_variance_ratio':(4*a+b)/(a+b),
                       'matched_actor_rows_variance_ratio':r['predicted_variance_ratio_matched_actor_NFE']['4'][kind]}
        variance.append(out)
    summary={'status':'audited','scientific_source_unchanged':True,'new_complete_policy_episodes':864,
        'kernel_paths':sum(r['samples'] for rows in d['kernel_probes'].values() for r in rows),
        'critic_first_action_continuations':sum(len(r['actions'])*32 for r in d['critic_probes']),
        'critic_environment_steps':sum(len(r['actions'])*32*r['remaining_steps'] for r in d['critic_probes']),
        'variance_fixed_leaves_vs_fixed_actor':variance,'noise_comparisons':n['comparisons'],
        'statistical_correction':'Original raw crossfit naive SE retained under runtime; corrected probe has no crossfit SE. Underlying critic and MC arrays exactly match.',
        'limits':'Post-hoc diagnostics, three known models, one instance, two fixed states. No parameter/default selection.'}
    atomic_json(OUT/'audit.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(15,4.5))
    for method,color in [('sphere','tab:blue'),('diffusion','tab:orange')]:
        rs=[r for r in d['preference_rescoring'] if r['model'].startswith(method) and r['mode']=='direct']
        ws=sorted({r['requested_weight'] for r in rs});values=np.array([np.mean([r['component_means'] for r in rs if r['requested_weight']==w],0) for w in ws])
        ax[0].plot(values[:,0],values[:,1],'o-',label=method,color=color)
        for w,v in zip(ws,values):ax[0].annotate(str(w),v,fontsize=8,xytext=(3,3),textcoords='offset points')
    ax[0].set(xlabel='Low-response cohort return',ylabel='High-response cohort return',title='Conditioning changes; weak low-cohort gains')
    ax[0].margins(x=.15);ax[0].legend();ax[1].errorbar([r['seed'] for r in n['comparisons']],[r['paired_delta'] for r in n['comparisons']],yerr=[r['episode_mean_se'] for r in n['comparisons']],fmt='o',capsize=4)
    ax[1].axhline(0,color='gray',ls=':');ax[1].set(xticks=[0,1,2],xlabel='Existing model training seed',ylabel='Mean-last minus noisy-last return',title='864 fresh episodes; paired SE bars')
    for method,marker in [('sphere','o'),('diffusion','x')]:
        rs=[r for r in d['critic_probes'] if r['method']==method]
        ax[2].scatter([r['weight']+(-.025 if r['remaining_steps']==20 else .025) for r in rs],[r['Q_MC_pearson'] for r in rs],marker=marker,label=method)
    ax[2].set(xticks=[0,.5,1],xlabel='Low-response preference weight',ylabel='Critic / MC candidate correlation',title='Two fixed states; flat values near weight1')
    ax[2].legend()
    for a in ax:a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'diagnosis.png',dpi=170);fig.savefig(OUT/'diagnosis.pdf');plt.close(fig)
    print(json.dumps({k:summary[k] for k in ('status','new_complete_policy_episodes','kernel_paths','critic_first_action_continuations','critic_environment_steps')}))


if __name__=='__main__':main()
