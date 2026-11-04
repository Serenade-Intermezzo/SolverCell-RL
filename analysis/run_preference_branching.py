"""Source-frozen, read-only branching screen/final evaluation with durable exits."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import math
import copy
import random
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from solvercell.environment import InterventionEnv
from solvercell.preference_config import load_preference_config
from solvercell.preference_data import PreferenceTask
from solvercell.preference_learner import PreferenceLearner
from solvercell.preference_branching import BranchingPreferencePolicy
from solvercell.preference_trainer import evaluate_preference
from solvercell.storage import atomic_json, append_json, source_manifest, worker_identity
from analysis.preference_tuning_guide_check import same


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def models_hash(learner):
    digest=hashlib.sha256()
    for name in ('actor','target_actor','q1','q2','tq1','tq2','guide'):
        module=getattr(learner,name)
        digest.update(name.encode())
        if module is None:continue
        for key,value in sorted(module.state_dict().items()):
            digest.update(f'{key}:{value.dtype}:{tuple(value.shape)}'.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class Adapter:
    def __init__(self,learner,topologies):
        self.learner=learner;self.device=learner.device
        self.branching=BranchingPreferencePolicy(learner,topologies)
    def act(self,obs,pref,mode='direct',generator=None):
        if mode.startswith('is'):
            old=self.learner.cfg
            try:
                self.learner.cfg=replace(old,guide_particles=int(mode[2:]))
                return self.learner.act(obs,pref,'terminal_weighting',generator)
            finally:self.learner.cfg=old
        return self.branching.act(obs,pref,mode,generator)


def load_parent(manifest,root,key):
    row=manifest['parents'][key];path=root/row['output']
    for name,digest in row['files_sha256'].items():assert sha(path/name)==digest,(key,name)
    saved=read(path/'source.json');current=source_manifest()
    assert current['source_sha256']==manifest['expected_source_sha256']
    assert saved['source_sha256']==manifest['parent_source_sha256']
    assert all(current['files'].get(name)==value for name,value in saved['files'].items()),'Parent source changed'
    assert read(path/'metrics.json')['status']=='complete' and read(path/'exit.json')['returncode']==0
    cfg=load_preference_config(path/'config.json')
    checkpoint=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert checkpoint['train_steps']==4000 and checkpoint['config']==cfg.to_dict()
    assert cfg.seed==int(key[-1]) and cfg.device=='cpu'
    env=InterventionEnv(cfg.n_arms,cfg.n_actions,cfg.budget,cfg.horizon,instance_seed=cfg.instance_seed,seed=cfg.seed+100)
    assert env.instance_hash==checkpoint['instance_hash']
    task=PreferenceTask(env);learner=PreferenceLearner(cfg,task)
    learner.load_state_dict(checkpoint['learner'])
    before={'models':models_hash(learner),'rngs':{name:getattr(learner,name).get_state().clone()
            for name in ('generator','preference_generator','guide_generator','behavior_generator')},
            'optimizers':{name:(None if getattr(learner,name) is None else copy.deepcopy(getattr(learner,name).state_dict()))
                          for name in ('actor_optim','q_optim','guide_optim')},
            'global_rngs':(torch.get_rng_state().clone(),copy.deepcopy(np.random.get_state()),random.getstate()),
            'updates':(learner.updates,learner.actor_updates,learner.guide_updates)}
    return cfg,checkpoint,task,learner,before


def observations(checkpoint):
    size=int(checkpoint['replay']['size'])
    return [(index,checkpoint['replay']['data']['obs'][index].float()) for index in (size-256,size-1)]


def modes(manifest,method,selection=None):
    independent=['direct' if n==1 else f'is{n}' for n in manifest['independent_candidates']]
    if method=='sphere':return independent
    return independent+(list(manifest['topologies']) if selection is None else [selection['evaluated_topology']])


def benchmark(adapter,obs_rows,weights,mode_list,cycles):
    samples={mode:[] for mode in mode_list};rng=np.random.default_rng(712345)
    for i,(_,obs) in enumerate(obs_rows):
        for j,weight in enumerate(weights):
            pref=np.asarray([weight,1-weight],np.float32)
            generators={mode:torch.Generator().manual_seed(720000+i*100+j*10+n)
                        for n,mode in enumerate(mode_list)}
            for mode in mode_list:
                for _ in range(3):adapter.act(obs.numpy(),pref,mode,generators[mode])
            for _ in range(cycles):
                for index in rng.permutation(len(mode_list)):
                    mode=mode_list[index];start=time.perf_counter_ns()
                    adapter.act(obs.numpy(),pref,mode,generators[mode])
                    samples[mode].append((time.perf_counter_ns()-start)/1e9)
    return {mode:{'seconds':values,'median_seconds':float(np.median(values)),
                  'mean_seconds':float(np.mean(values)),'sample_count':len(values)} for mode,values in samples.items()}


def correlation(a,b):
    return None if np.std(a)<1e-12 or np.std(b)<1e-12 else float(np.corrcoef(a,b)[0,1])


@torch.no_grad()
def support_audit(task,learner,obs_rows,manifest):
    output=[];count=manifest['support_candidates'];cfg=learner.cfg
    for obs_number,(index,obs) in enumerate(obs_rows):
        state=np.rint(obs[:-1].numpy()*3).astype(np.float32)
        remaining=int(round(float(obs[-1])*cfg.horizon))
        for weight in manifest['screen_preferences']:
            pref=obs.new_tensor([weight,1-weight]);obatch=obs[None].expand(count,-1);pbatch=pref[None].expand(count,-1)
            generator=torch.Generator().manual_seed(730000+obs_number*100+int(100*weight))
            costs=learner._sample(obatch,pbatch,generator)
            prior=torch.randn((count,task.cost_dim),generator=generator)
            base_actions=learner._decode(costs);prior_actions=learner._decode(prior)
            q=(learner._twin_vector(obatch,base_actions,pbatch)*pbatch).sum(-1).numpy()
            base_step=np.array([np.dot(task.expected_components(a,state),pref.numpy()) for a in base_actions.numpy()])
            prior_step=np.array([np.dot(task.expected_components(a,state),pref.numpy()) for a in prior_actions.numpy()])
            candidates=[('first_base',base_actions[0]),('max_base_Q',base_actions[int(np.argmax(q))]),
                        ('max_base_myopic',base_actions[int(np.argmax(base_step))]),
                        ('max_prior_myopic',prior_actions[int(np.argmax(prior_step))])]
            chosen=[];seen=set()
            for label,action in candidates:
                key=tuple(action.tolist())
                if key in seen:continue
                seen.add(key)
                values=[]
                predicted=(learner._twin_vector(obs[None],action[None],pref[None])*pref[None]).sum().item()
                for rep in range(manifest['continuation_repetitions']):
                    env=task.env.clone();env.state=state.copy();env.t=cfg.horizon-remaining
                    env.rng=np.random.default_rng(740000+rep)
                    gen=torch.Generator().manual_seed(750000+rep)
                    total=0.;discount=1.
                    for step in range(remaining):
                        if step==0:a=action.numpy()
                        else:a,_,_=learner.act(task.encode(env.state,cfg.horizon-env.t),pref.numpy(),'direct',gen)
                        nxt,_,done,_=env.step(a)
                        total+=discount*cfg.reward_scale*float(np.dot(task.components(nxt),pref.numpy()))
                        discount*=cfg.gamma
                    assert done
                    values.append(total)
                chosen.append({'origin':label,'action':action.tolist(),'critic_scalar':predicted,
                    'discounted_MC_values':values,'MC_mean':float(np.mean(values)),
                    'MC_mean_se':float(np.std(values,ddof=1)/math.sqrt(len(values)))})
            output.append({'replay_row':index,'remaining_steps':remaining,'preference':weight,
                'base_unique_actions':len(torch.unique(base_actions,dim=0)),
                'prior_unique_actions':len(torch.unique(prior_actions,dim=0)),
                'base_Q_std':float(np.std(q)),'base_Q_myopic_correlation':correlation(q,base_step),
                'base_myopic_mean':float(base_step.mean()),'base_myopic_max':float(base_step.max()),
                'prior_myopic_max':float(prior_step.max()),'selected_candidate_MC':chosen,
                'scope':'Selected-action diagnostic, not unbiased ranking estimate; myopic values are not future returns.'})
    return output


def variance_components(g):
    within=g.var(1,unbiased=True).sum(-1).mean()
    between=g.mean(1).var(0,unbiased=True).sum()-within/g.shape[1]
    return {'between_raw':float(between),'between_clipped':max(0.,float(between)),'within':float(within)}


@torch.no_grad()
def variance_audit(task,learner,obs_rows,manifest):
    result=[];m=manifest['variance_prefixes'];r=manifest['variance_suffixes']
    for obs_number,(row,obs) in enumerate(obs_rows):
        for weight in manifest['screen_preferences']:
            prefs=obs.new_tensor([[weight,1-weight]])
            for index in (2,4,6):
                gen=torch.Generator().manual_seed(760000+obs_number*1000+int(weight*100)+index)
                robs=obs[None].expand(m,-1);rprefs=prefs.expand(m,-1)
                x=learner.target_actor._initial(robs,gen)
                for step in range(index):x=learner.target_actor.transition(robs,rprefs,x,step,gen)
                lobs=robs.repeat_interleave(r,0);lprefs=rprefs.repeat_interleave(r,0)
                raw=learner.target_actor.suffix(lobs,lprefs,x.repeat_interleave(r,0),index,gen)
                actions=learner._decode(raw)
                utility=(learner._twin_vector(lobs,actions,lprefs)*lprefs).sum(-1).double().reshape(m,r)
                logits=utility/learner.cfg.guide_temperature
                y=(logits-logits.max()).exp()
                _,inverse=torch.unique(actions,dim=0,return_inverse=True)
                onehot=torch.nn.functional.one_hot(inverse).double().reshape(m,r,-1)
                pi=(y[:,:,None]*onehot).sum((0,1))/y.sum()
                g=y[:,:,None]*(onehot-pi)
                target_mean=(y*utility).sum()/y.sum()
                f=(y*(utility-target_mean))[:,:,None]
                components={'normalizer':variance_components(y[:,:,None]),
                            'action_influence_trace':variance_components(g),
                            'utility_influence':variance_components(f)}
                predicted={}
                for child_count in (2,4):
                    values={}
                    for kind,component in components.items():
                        a,b=component['between_clipped'],component['within']
                        values[kind]=None if a+b<1e-30 else (
                            (a+b/child_count)*(index+child_count*(learner.cfg.sampling_steps-index))
                            /(learner.cfg.sampling_steps*(a+b)))
                    predicted[str(child_count)]=values
                agreement=((onehot.sum(1)*(onehot.sum(1)-1)).sum(-1)/(r*(r-1))).mean()
                result.append({'replay_row':row,'preference':weight,'branch_index':index,
                    'prefixes':m,'suffixes_per_prefix':r,'variance_components':components,
                    'predicted_variance_ratio_matched_actor_NFE':predicted,
                    'leaf_action_count':onehot.shape[-1],'sibling_action_agreement':float(agreement),
                    'logweight_range':float(logits.max()-logits.min())})
    return result


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    identity=worker_identity();atomic_json(args.output/'worker.json',identity)
    manifest=read(args.manifest);assert manifest['expected_source_sha256']==source_manifest()['source_sha256']
    assert sha(__file__)==manifest['runner_script_sha256']
    selection=read(args.selection) if args.selection else None
    if args.stage=='final':assert selection is not None and selection['source_sha256']==manifest['expected_source_sha256']
    atomic_json(args.output/'invocation.json',{'stage':args.stage,'seed':args.seed,'manifest':str(args.manifest),
        'manifest_sha256':sha(args.manifest),'source_sha256':manifest['expected_source_sha256'],
        'analysis_script_sha256':sha(__file__),'selection_sha256':sha(args.selection) if args.selection else None})
    append_path=args.output/'evaluations.jsonl';all_rows=[];records=[];start=time.perf_counter()
    for method in ('sphere','diffusion'):
        key=f'{method}_seed{args.seed}'
        cfg,checkpoint,task,learner,before=load_parent(manifest,args.checkpoint_root,key)
        adapter=Adapter(learner,manifest['topologies'])
        mode_list=modes(manifest,method,selection if args.stage=='final' else None)
        weights=manifest['screen_preferences'] if args.stage=='screen' else manifest['final_preferences']
        seed_start=manifest['screen_seed_start'] if args.stage=='screen' else manifest['final_seed_start']
        rows=[]
        for weight in weights:
            atomic_json(args.output/'status.json',{'status':'running','phase':'evaluation','model':key,'preference':weight})
            for mode in mode_list:
                value=evaluate_preference(task,adapter,weight,mode,manifest['episodes'],seed_start)
                assert value['episode_lengths']==[cfg.horizon]*manifest['episodes']
                value.update(model=key,stage=args.stage,checkpoint_train_steps=4000)
                append_json(append_path,value);rows.append(value);all_rows.append(value)
            print(json.dumps({'model':key,'preference':weight,'complete':True}),flush=True)
        record={'model':key,'parent':manifest['parents'][key],'config':cfg.to_dict()}
        if args.stage=='screen':
            atomic_json(args.output/'status.json',{'status':'running','phase':'support_and_variance_audit','model':key})
            obs_rows=observations(checkpoint)
            record['timing']=benchmark(adapter,obs_rows,weights,mode_list,manifest['timing_cycles'])
            record['support_audit']=support_audit(task,learner,obs_rows,manifest)
            if method=='diffusion':record['variance_audit']=variance_audit(task,learner,obs_rows,manifest)
        assert models_hash(learner)==before['models']
        assert before['updates']==(learner.updates,learner.actor_updates,learner.guide_updates)
        for name,state in before['rngs'].items():assert torch.equal(state,getattr(learner,name).get_state())
        for name,state in before['optimizers'].items():
            same(state,None if getattr(learner,name) is None else getattr(learner,name).state_dict())
        same(before['global_rngs'],(torch.get_rng_state(),np.random.get_state(),random.getstate()))
        assert sha(args.checkpoint_root/manifest['parents'][key]['output']/'checkpoint.pt')==manifest['parents'][key]['files_sha256']['checkpoint.pt']
        record['models_rngs_updates_checkpoint_unchanged']=True;records.append(record)
    result={'status':'complete','stage':args.stage,'seed':args.seed,'source_sha256':manifest['expected_source_sha256'],
        'parent_source_sha256':manifest['parent_source_sha256'],'scope':'fixed-checkpoint inference; zero training updates',
        'parents':records,'evaluations':all_rows,'wall_seconds':time.perf_counter()-start}
    atomic_json(args.output/'summary.json',result)
    atomic_json(args.output/'status.json',{'status':'complete','wall_seconds':result['wall_seconds']})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--checkpoint-root',type=Path,default=ROOT)
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--stage',choices=('screen','final'),default='screen')
    parser.add_argument('--seed',type=int,choices=(0,1,2),default=0)
    args=parser.parse_args();code=1;error=None
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    try:run(args);code=0
    except Exception as exception:
        error=repr(exception);traceback.print_exc()
    finally:
        # Never overwrite an existing experiment's exit record on rejected reuse.
        worker=args.output/'worker.json'
        if (worker.exists() and read(worker).get('pid')==__import__('os').getpid()
                and read(worker).get('start_ticks')==Path('/proc/self/stat').read_text().split()[21]):
            atomic_json(args.output/'exit.json',{'returncode':code,'error':error})
    raise SystemExit(code)
