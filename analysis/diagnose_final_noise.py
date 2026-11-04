"""Single declared, fresh-episode final-noise intervention; no training or sweep."""
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.run_preference_branching import load_parent,models_hash,same
from solvercell.preference_trainer import evaluate_preference
from solvercell.storage import atomic_json,append_json,source_manifest
OUT=ROOT/'reports/method_diagnosis_20261104'


class MeanLast:
    def __init__(self,learner):self.learner=learner;self.device=learner.device
    @torch.no_grad()
    def act(self,obs,pref,mode='direct',generator=None):
        if mode=='direct':return self.learner.act(obs,pref,mode,generator)
        assert mode=='mean_last';learner=self.learner;actor=learner.target_actor
        ob=learner._tensor(obs)[None];p=learner._preferences(pref,1);before=learner.counters.copy()
        x=actor._initial(ob,generator)
        for step in range(actor.steps-1):x=actor.transition(ob,p,x,step,generator)
        mean,std=actor.transition_parameters(ob,p,x,actor.steps-1)
        torch.randn(mean.shape,generator=generator)  # Preserve the direct sampler's random draw schedule.
        cost=F.normalize(mean,dim=-1);learner._record_actor(1);action=learner._decode(cost)
        stats={k:learner.counters[k]-before[k] for k in learner.COUNTER_KEYS}
        return action[0].numpy().copy(),cost[0].numpy().copy(),stats


def main():
    target=OUT/'final_noise.json'
    if target.exists() or (OUT/'final_noise_evaluations.jsonl').exists():raise FileExistsError('Immutable output')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);start=time.perf_counter()
    m=json.loads((ROOT/'experiments/preference_branching_20261104/manifest.json').read_text());rows=[];identities={}
    for key in m['parents']:
        cfg,cp,task,learner,before=load_parent(m,ROOT,key);adapter=MeanLast(learner)
        # Mandatory path equivalence/RNG sanity at a real observation before evaluation.
        if key.startswith('diffusion'):
            ob=cp['replay']['data']['obs'][0].float();p=ob.new_tensor([.5,.5]);g=torch.Generator().manual_seed(991)
            raw=learner.target_actor.sample_path(ob[None],p[None],g)
            mean,_=learner.target_actor.transition_parameters(ob[None],p[None],raw[:,-2],7)
            expected=learner._decode(mean)[0].numpy();state=g.get_state()
            g2=torch.Generator().manual_seed(991);action,_,stats=adapter.act(ob.numpy(),p.numpy(),'mean_last',g2)
            assert np.array_equal(action,expected) and torch.equal(state,g2.get_state()) and stats['nfe']==8
        modes=['direct'] if key.startswith('sphere') else ['direct','mean_last']
        for w in (0.,.5,1.):
            for mode in modes:
                result=evaluate_preference(task,adapter,w,mode,32,1100000);result['model']=key
                assert result['episode_lengths']==[20]*32
                assert result['compute_totals']['nfe']==32*20*8 and result['compute_totals']['decoder_calls']==32*20
                assert result['compute_totals']['q_calls']==0
                append_json(OUT/'final_noise_evaluations.jsonl',result);rows.append(result)
            print(json.dumps({'model':key,'weight':w,'complete':True}),flush=True)
        assert models_hash(learner)==before['models'];assert before['updates']==(learner.updates,learner.actor_updates,learner.guide_updates)
        for name,state in before['rngs'].items():assert torch.equal(state,getattr(learner,name).get_state())
        for name,state in before['optimizers'].items():same(state,None if getattr(learner,name) is None else getattr(learner,name).state_dict())
        same(before['global_rngs'],(torch.get_rng_state(),np.random.get_state(),random.getstate()))
        for name,digest in m['parents'][key]['files_sha256'].items():assert hashlib.sha256((ROOT/m['parents'][key]['output']/name).read_bytes()).hexdigest()==digest
        identities[key]={'all_states_unchanged':True,'parent':m['parents'][key]}
    comparisons=[]
    for seed in (0,1,2):
        r=[x for x in rows if x['model']==f'diffusion_seed{seed}'];a=sorted([x for x in r if x['mode']=='direct'],key=lambda x:x['low_response_weight']);b=sorted([x for x in r if x['mode']=='mean_last'],key=lambda x:x['low_response_weight'])
        diff=np.mean([np.asarray(y['scalarized_returns'])-x['scalarized_returns'] for x,y in zip(a,b)],axis=0)
        comparisons.append({'seed':seed,'direct_mean':float(np.mean([x['return_mean'] for x in a])),
            'mean_last_mean':float(np.mean([x['return_mean'] for x in b])),'paired_delta':float(diff.mean()),
            'episode_mean_se':float(diff.std(ddof=1)/np.sqrt(len(diff))),'paired_episode_deltas':diff.tolist()})
    atomic_json(target,{'status':'complete','scope':'post-hoc fixed-kernel mechanism diagnosis, not new method validation',
        'source_sha256':source_manifest()['source_sha256'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'design_sha256':hashlib.sha256((OUT/'FINAL_NOISE_DESIGN.md').read_bytes()).hexdigest(),
        'parents':identities,'evaluations':rows,'comparisons':comparisons,'wall_seconds':time.perf_counter()-start})
    print(json.dumps({'status':'complete','comparisons':comparisons}))


if __name__=='__main__':main()
