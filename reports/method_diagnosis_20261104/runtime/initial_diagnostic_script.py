"""Post-hoc fixed-checkpoint diagnostics; no tuning, training or parent mutation."""
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.run_preference_branching import load_parent, models_hash, same
from solvercell.storage import atomic_json,source_manifest

OUT=ROOT/'reports/method_diagnosis_20261104'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def corr(a,b):
    return None if np.std(a)<1e-12 or np.std(b)<1e-12 else float(np.corrcoef(a,b)[0,1])


def paired(values):
    x=np.asarray(values);return {'mean':float(x.mean()),'episode_se':float(x.std(ddof=1)/np.sqrt(len(x))),'values':x.tolist()}


def rescore(study):
    out=[]
    for final in study['finals']:
        rows=final['evaluations']
        for model,mode in sorted({(r['model'],r['mode']) for r in rows}):
            selected=sorted([r for r in rows if r['model']==model and r['mode']==mode],key=lambda r:r['low_response_weight'])
            for own in selected:
                w=own['low_response_weight'];pref=np.array([w,1-w]);self_values=np.array(own['component_returns'])@pref
                candidates={str(r['low_response_weight']):paired(np.array(r['component_returns'])@pref-self_values) for r in selected}
                out.append({'model':model,'mode':mode,'requested_weight':w,'own_mean':own['return_mean'],
                    'component_means':own['component_return_mean'],'counterfactual_differences':candidates,
                    'scope':'post-hoc policy-input diagnostic; same environment episode seeds; no new selected policy'})
    return out


@torch.no_grad()
def kernel_probe(learner,obs_rows):
    out=[];actor=learner.target_actor;n=256
    for row,obs in obs_rows:
        for weight in (0.,.5,1.):
            gen=torch.Generator().manual_seed(910000+int(weight*100)+row)
            ob=obs[None].expand(n,-1);p=obs.new_tensor([[weight,1-weight]]).expand(n,-1)
            path=actor.sample_path(ob,p,gen);mean,std=actor.transition_parameters(ob,p,path[:,-2],actor.steps-1)
            a_mean=learner._decode(mean);a_noisy=learner._decode(path[:,-1]);state=np.rint(obs[:-1].numpy()*3).astype(np.float32)
            step_mean=np.array([learner.task.expected_components(a,state) for a in a_mean.numpy()])
            step_noisy=np.array([learner.task.expected_components(a,state) for a in a_noisy.numpy()])
            qm=(learner._twin_vector(ob,a_mean,p)*p).sum(-1);qn=(learner._twin_vector(ob,a_noisy,p)*p).sum(-1)
            # Training endpoints are unit vectors; E||sqrt(abar)c+sqrt(1-abar)e||^2 is exact.
            abar=float(actor.alpha_bars[-1]);dim=learner.cost_dim
            out.append({'replay_row':row,'weight':weight,'samples':n,'alpha_bar_terminal':abar,'cost_dim':dim,
                'unit_target_forward_terminal_expected_norm_sq':abar+(1-abar)*dim,'actual_prior_expected_norm_sq':dim,
                'path_mean_norms':path.norm(dim=-1).mean(0).tolist(),
                'final_kernel_std':float(std),'final_noise_rms':float(std)*np.sqrt(dim),
                'last_mean_norm':float(mean.norm(dim=-1).mean()),
                'last_noise_action_flip_fraction':float((a_mean!=a_noisy).any(-1).float().mean()),
                'last_noise_expected_one_step_component_delta':(step_noisy-step_mean).mean(0).tolist(),
                'last_noise_critic_utility_delta':float((qn-qm).mean()),
                'scope':'local final-step intervention; not a policy performance result'})
    return out


@torch.no_grad()
def rollout_candidates(task,learner,obs,pref,actions,repetitions=32):
    state=np.rint(obs[:-1].numpy()*3).astype(np.float32);remaining=int(round(float(obs[-1])*task.horizon))
    envs=[]
    for action in actions:
        for rep in range(repetitions):
            env=task.env.clone();env.state=state.copy();env.t=task.horizon-remaining
            env.rng=np.random.default_rng(920000+rep);envs.append(env)
    count=len(actions);values=np.zeros(count*repetitions);discount=1.;gen=torch.Generator().manual_seed(930000)
    for step in range(remaining):
        if step==0:chosen=np.repeat(actions,repetitions,axis=0)
        else:
            encoded=torch.tensor(np.stack([task.encode(e.state,task.horizon-e.t) for e in envs]))
            preferences=torch.tensor(pref).expand(len(envs),-1)
            # CRN actor innovations across initial actions; equal blocks start from the same generator state.
            start_rng=gen.get_state();blocks=[];end_rng=None
            for j in range(count):
                gen.set_state(start_rng);ids=slice(j*repetitions,(j+1)*repetitions)
                blocks.append(learner._decode(learner._sample(encoded[ids],preferences[ids],gen)).numpy())
                end_rng=gen.get_state()
            gen.set_state(end_rng);chosen=np.concatenate(blocks)
        for j,(env,a) in enumerate(zip(envs,chosen)):
            nxt,_,done,_=env.step(a);values[j]+=discount*learner.cfg.reward_scale*np.dot(task.components(nxt),pref)
        discount*=learner.cfg.gamma
    assert all(e.t==task.horizon for e in envs)
    return values.reshape(count,repetitions)


@torch.no_grad()
def critic_probe(parents,obs_rows):
    result=[]
    sphere=parents['sphere_seed0'][3];diffusion=parents['diffusion_seed0'][3]
    for row,obs in obs_rows:
        for weight in (0.,.5,1.):
            p=obs.new_tensor([[weight,1-weight]]);ob=obs[None].expand(8,-1);pb=p.expand(8,-1)
            gen=torch.Generator().manual_seed(940000+row+int(weight*100))
            prior=diffusion._decode(torch.randn((8,diffusion.cost_dim),generator=gen))
            sa=sphere._decode(sphere._sample(ob,pb,gen));da=diffusion._decode(diffusion._sample(ob,pb,gen))
            actions=torch.unique(torch.cat((prior,sa,da)),dim=0);npactions=actions.numpy();n=len(actions)
            state=np.rint(obs[:-1].numpy()*3).astype(np.float32);prefs=p.numpy()[0]
            myopic=np.array([np.dot(diffusion.task.expected_components(a,state),prefs) for a in npactions])
            for kind,learner in (('sphere',sphere),('diffusion',diffusion)):
                q=(learner._twin_vector(obs[None].expand(n,-1),actions,p.expand(n,-1))*p).sum(-1).numpy()
                mc=rollout_candidates(learner.task,learner,obs,prefs,npactions)
                values=mc.mean(1);qsel=int(q.argmax());msel=int(myopic.argmax());half=16
                selected=[int(mc[:,:half].mean(1).argmax()),int(mc[:,half:].mean(1).argmax())]
                cf=np.concatenate((mc[selected[1],:half],mc[selected[0],half:]))
                result.append({'method':kind,'model_seed':0,'replay_row':row,'remaining_steps':int(round(float(obs[-1])*20)),
                    'weight':weight,'actions':npactions.tolist(),'critic_values':q.tolist(),'myopic_values':myopic.tolist(),
                    'MC_repetitions':mc.tolist(),'Q_MC_pearson':corr(q,values),
                    'Q_MC_rank_correlation':corr(np.argsort(np.argsort(q)),np.argsort(np.argsort(values))),
                    'critic_selected_index':qsel,'myopic_selected_index':msel,'crossfit_selected_indices':selected,
                    'critic_selected_minus_pool_average':paired(mc[qsel]-mc.mean(0)),
                    'myopic_selected_minus_pool_average':paired(mc[msel]-mc.mean(0)),
                    'crossfit_MC_selected_minus_critic_selected':paired(cf-mc[qsel]),
                    'critic_bias_mean':float((q-values).mean()),'MC_action_spread':float(values.std()),
                    'MC_mean_se_median':float(np.median(mc.std(1,ddof=1)/np.sqrt(mc.shape[1]))),
                    'scope':'selected common candidate pool; method-specific base continuation; two fixed states, not population calibration'})
            print(json.dumps({'phase':'critic','row':row,'weight':weight,'complete':True}),flush=True)
    return result


def main():
    start=time.perf_counter();target=OUT/'diagnostics.json'
    if target.exists():raise FileExistsError('Diagnostic outputs are immutable')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    manifest=json.loads((ROOT/'experiments/preference_branching_20261104/manifest.json').read_text())
    study=json.loads((ROOT/'reports/preference_branching_20261104/study.json').read_text());parents={}
    for key in manifest['parents']:parents[key]=load_parent(manifest,ROOT,key)
    # Every method/seed uses the same two actual observations, removing own-replay-state confounding.
    cp=parents['diffusion_seed0'][1];size=int(cp['replay']['size'])
    obs_rows=[(i,cp['replay']['data']['obs'][i].float()) for i in (size-256,size-1)]
    global_before=(torch.get_rng_state().clone(),copy.deepcopy(np.random.get_state()),random.getstate())
    kernels={key:kernel_probe(value[3],obs_rows) for key,value in parents.items() if key.startswith('diffusion')}
    ranks=critic_probe(parents,obs_rows)
    for key,(cfg,cp,task,learner,before) in parents.items():
        assert models_hash(learner)==before['models']
        assert before['updates']==(learner.updates,learner.actor_updates,learner.guide_updates)
        for name,state in before['rngs'].items():assert torch.equal(state,getattr(learner,name).get_state())
        for name,state in before['optimizers'].items():same(state,None if getattr(learner,name) is None else getattr(learner,name).state_dict())
        for name,digest in manifest['parents'][key]['files_sha256'].items():assert sha(ROOT/manifest['parents'][key]['output']/name)==digest
    same(global_before,(torch.get_rng_state(),np.random.get_state(),random.getstate()))
    output={'status':'complete','scope':'post-hoc diagnosis only; no training, tuning or new policy performance claim',
            'script_sha256':sha(__file__),'design_sha256':sha(OUT/'DESIGN.md'),'source_sha256':source_manifest()['source_sha256'],
            'parent_manifest_sha256':sha(ROOT/'experiments/preference_branching_20261104/manifest.json'),
            'parent_study_sha256':sha(ROOT/'reports/preference_branching_20261104/study.json'),
            'parents':manifest['parents'],'all_parent_models_optimizers_RNGs_updates_checkpoints_unchanged':True,
            'preference_rescoring':rescore(study),'kernel_probes':kernels,'critic_probes':ranks,'wall_seconds':time.perf_counter()-start}
    atomic_json(target,output);print(json.dumps({'status':'complete','wall_seconds':output['wall_seconds']}))


if __name__=='__main__':main()
