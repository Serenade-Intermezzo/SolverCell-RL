"""Training-side controllability, exact native catalog and frozen scale calibration."""
import argparse,hashlib,itertools,json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from solvercell.environment import InterventionEnv
from solvercell.preference_data import PreferenceTask
from solvercell.storage import atomic_json,source_manifest


def catalog(task):
    p=task.projector;sizes=p.group_sizes.numpy();g=len(sizes);seen={}
    for mask in itertools.product((0,1),repeat=g):
        full=np.array(mask,bool);count=int(sizes[full].sum())
        if count>task.budget:continue
        for cut in [None]+[j for j in range(g) if not full[j] and 0<task.budget-count<sizes[j]]:
            scores=np.where(full,2.,-1.)
            if cut is not None:scores[cut]=1.
            cost=torch.tensor(scores,dtype=torch.float32)/p.squared_norms.sqrt().float();cost=cost/cost.norm()
            action=task.decode(cost[None])[0].numpy();seen[tuple(action)]=(action,cost.numpy())
    return np.stack([v[0] for v in seen.values()]),np.stack([v[1] for v in seen.values()])


def relaxed_values(env,actions,gamma=1.):
    probabilities=np.array([env.transition_probabilities(a) for a in actions]);lo=probabilities.min(0);hi=probabilities.max(0)
    zero=env.transition_probabilities(np.zeros(env.n_actions));ids=np.arange(env.n_arms)
    vmax=np.zeros((env.horizon+1,env.n_arms,4));vmin=vmax.copy();vzero=vmax.copy()
    for h in range(1,env.horizon+1):
        for s in range(4):
            u,d=min(s+1,3),max(s-1,0)
            for values,kind in ((vmax,'max'),(vmin,'min'),(vzero,'zero')):
                up=env.state_r[ids,u]+gamma*values[h-1,ids,u];down=env.state_r[ids,d]+gamma*values[h-1,ids,d]
                low=lo*up+(1-lo)*down;high=hi*up+(1-hi)*down
                values[h,:,s]=(np.maximum(low,high) if kind=='max' else np.minimum(low,high) if kind=='min' else zero*up+(1-zero)*down)
    return probabilities,vmax,vmin,vzero


def action_values(task,state,remaining,probabilities,vmax,gamma=1.):
    ids=np.arange(task.n_arms);up=np.minimum(state.astype(int)+1,3);down=np.maximum(state.astype(int)-1,0)
    gu=task.env.state_r[ids,up]+gamma*vmax[remaining-1,ids,up];gd=task.env.state_r[ids,down]+gamma*vmax[remaining-1,ids,down]
    vals=probabilities*gu+(1-probabilities)*gd
    return np.array([task.n_arms*vals[:,group].mean(1) for group in task.groups]).T


def expert_episodes(task,actions,probabilities,vmax,raw_weights,seeds):
    rows=[]
    for weight in raw_weights:
        returns=[]
        for seed in seeds:
            e=task.env.clone();state=e.reset(seed=seed);total=np.zeros(2)
            for _ in range(e.horizon):
                values=action_values(task,state,e.horizon-e.t,probabilities,vmax)
                choice=int(np.argmax(values@np.array([weight,1-weight])))
                state,_,done,_=e.step(actions[choice]);total+=task.components(state)
            assert done;returns.append(total.tolist())
        rows.append({'raw_low_weight':float(weight),'returns':returns,'component_mean':np.mean(returns,0).tolist()})
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--debug',action='store_true');args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    task=PreferenceTask(InterventionEnv());actions,costs=catalog(task)
    assert torch.equal(task.decode(torch.tensor(costs)).bool(),torch.tensor(actions).bool())
    # Random continuous proposals must lie in this complete open-cell catalog.
    gen=torch.Generator().manual_seed(134);random_actions=task.decode(torch.randn(10000,task.cost_dim,generator=gen)).numpy();keys={tuple(a) for a in actions};assert all(tuple(a) in keys for a in random_actions)
    probabilities,vmax,vmin,vzero=relaxed_values(task.env,actions);n=2 if args.debug else 32
    # Scales use only fixed model-informed reference policies on NEW calibration episodes.
    calibration=expert_episodes(task,actions,probabilities,vmax,[0.,1.],range(1200000,1200000+n))
    high=np.array(calibration[0]['component_mean']);low=np.array(calibration[1]['component_mean']);scales=np.maximum(np.abs(low-high),1.)
    requested=[0.,.25,.5,.75,1.];effective=[(w/scales[0])/(w/scales[0]+(1-w)/scales[1]) for w in requested]
    validation=expert_episodes(task,actions,probabilities,vmax,sorted(set(requested+effective)),range(1300000,1300000+n))
    replay_states=[];parent_hashes={}
    for method,path in [('sphere','sphere_t010'),('diffusion','diff_b020_t010')]:
        root=ROOT/'outputs/preference_tuning_20261104'/path;cp=torch.load(root/'checkpoint.pt',map_location='cpu',weights_only=False)
        parent_hashes[method]=hashlib.sha256((root/'checkpoint.pt').read_bytes()).hexdigest();data=cp['replay']['data']['obs'];size=int(cp['replay']['size'])
        for index in np.linspace(256,size-1,32,dtype=int):
            obs=data[index].numpy();state=np.rint(obs[:-1]*3).astype(int);h=int(round(obs[-1]*20));ids=np.arange(40)
            replay_states.append({'method':method,'replay_row':int(index),'remaining':h,
                'relaxed_upper_minus_zero':[float(2*(vmax[h,g,state[g]]-vzero[h,g,state[g]]).sum()) for g in task.groups],
                'action_Q_component_ranges':np.ptp(action_values(task,state,h,probabilities,vmax),axis=0).tolist()})
    low_gain=float(low[0]-high[0]);high_gain=float(high[1]-low[1]);tradeoff=low_gain>5 and high_gain>5
    rows={r['raw_low_weight']:r for r in validation};curve=[rows[w]['component_mean'] for w in effective]
    result={'status':'complete','scope':'model-informed reference audit, not learned-policy or global optimality evidence',
        'source_sha256':source_manifest()['source_sha256'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'catalog_size':len(actions),'random_catalog_checks':10000,'calibration':calibration,'frozen_component_scales':scales.tolist(),
        'requested_preferences':requested,'effective_raw_preferences':effective,'validation':validation,'normalized_expert_curve':curve,
        'episodes_per_preference':n,'calibration_seed_start':1200000,'validation_seed_start':1300000,
        'tradeoff_gate':{'passed':tradeoff,'low_extreme_gain':low_gain,'high_extreme_gain':high_gain,'threshold_each':5},
        'replay_controllability':replay_states,'parent_checkpoint_sha256':parent_hashes,
        'limits':'Independent-arm DP is a relaxation; the feasible greedy reference is not an optimal expert. Scales are frozen from calibration only.'}
    atomic_json(args.output/'audit.json',result);np.savez_compressed(args.output/'catalog.npz',actions=actions,costs=costs,probabilities=probabilities,vmax=vmax,vmin=vmin,vzero=vzero)
    print(json.dumps({'gate':result['tradeoff_gate'],'scales':scales.tolist(),'effective_weights':effective,'curve':curve}))


if __name__=='__main__':main()
