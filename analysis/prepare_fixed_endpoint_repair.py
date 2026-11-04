"""Freeze a common endpoint target and matched actor-fitting configurations."""
import argparse,hashlib,json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.repair_preference_task import action_values
from solvercell.environment import InterventionEnv
from solvercell.preference_data import PreferenceTask
from solvercell.preference_repair import repaired_preference_config
from solvercell.storage import atomic_json,source_manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign-id',required=True);p.add_argument('--debug',action='store_true');args=p.parse_args()
    if Path(args.campaign_id).name!=args.campaign_id:raise ValueError('Single fresh directory name required')
    out=ROOT/'experiments'/args.campaign_id;out.mkdir(parents=True,exist_ok=False)
    calibration=ROOT/'reports/priority_repair_20261104/task_audit.json';audit=json.loads(calibration.read_text());assert audit['tradeoff_gate']['passed']
    archive=np.load(ROOT/'outputs/priority_repair_task_20261104/catalog.npz');task=PreferenceTask(InterventionEnv());costs=archive['costs'];actions=archive['actions'];scales=np.array(audit['frozen_component_scales'])
    train=[];val=[];parent_hashes={};pools={};used=set()
    for method,old in [('sphere','sphere_t010'),('diffusion','diff_b020_t010')]:
        path=ROOT/'outputs/preference_tuning_20261104'/old/'checkpoint.pt';parent_hashes[method]=hashlib.sha256(path.read_bytes()).hexdigest();cp=torch.load(path,map_location='cpu',weights_only=False);size=int(cp['replay']['size'])
        rng=np.random.default_rng(1400000);ids=rng.permutation(np.arange(256,size));pools[method]=[cp['replay']['data']['obs'][i].numpy() for i in ids]
        count=0
        for row in pools[method]:
            if row.tobytes() in used:continue
            train.append(row);used.add(row.tobytes());count+=1
            if count==32:break
        assert count==32
    for method,pool in pools.items():
        count=0
        for row in pool:
            if row.tobytes() in used:continue
            val.append(row);used.add(row.tobytes());count+=1
            if count==16:break
        assert count==16
    train=np.array(train);val=np.array(val)
    if args.debug:train=train[:2];val=val[:1]
    # Two different provenance rows can encode the same physical context. Remove exact overlap.
    train_keys={row.tobytes() for row in train};assert not any(row.tobytes() in train_keys for row in val),'Physical context leakage'
    def utilities(observations):
        result=[]
        for obs in observations:
            state=np.rint(obs[:-1]*3).astype(int);remaining=int(round(float(obs[-1])*20))
            result.append(action_values(task,state,remaining,archive['probabilities'],archive['vmax'])/scales)
        return np.array(result,dtype=np.float32)
    data=out/'data.npz';np.savez_compressed(data,train_obs=train,validation_obs=val,train_values=utilities(train),validation_values=utilities(val),costs=costs,actions=actions)
    jobs=[]
    for seed in ([0] if args.debug else [0,1,2]):
        for method,prediction in [('sphere','epsilon'),('diffusion','epsilon'),('diffusion','x0')]:
            name=f'{method}_{prediction}_s{seed}';cfg,provenance=repaired_preference_config(calibration,method='preference_'+method,seed=seed,prediction=prediction)
            config={'id':name,'method':method,'prediction':prediction,'seed':seed,'actor_config':cfg.to_dict(),'calibration':provenance,
                    'steps':4 if args.debug else 2000,'batch_size':2 if args.debug else 64,'log_interval':2 if args.debug else 100,
                    'eval_interval':2 if args.debug else 500,'eval_samples':32 if args.debug else 8192,'chunk_size':2048,
                    'temperature':.1,'train_preferences':[0.,.25,.5,.75,1.],'heldout_preferences':[.125,.375,.625,.875],
                    'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest()}
            atomic_json(out/f'{name}.json',config)
            jobs.append({'id':name,'config':str((out/f'{name}.json').relative_to(ROOT)),
                         'output':f'outputs/{args.campaign_id}/{name}','config_sha256':hashlib.sha256((out/f'{name}.json').read_bytes()).hexdigest()})
    manifest={'campaign_id':args.campaign_id,'schema':'fixed_endpoint_repair_v4','debug':args.debug,'jobs':jobs,'max_parallel':2,'cpu_sets':[[24],[25]],
        'source_sha256':source_manifest()['source_sha256'],'runner_sha256':hashlib.sha256((ROOT/'analysis/run_fixed_endpoint_repair.py').read_bytes()).hexdigest(),
        'data_path':str(data.relative_to(ROOT)),'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'parent_checkpoint_sha256':parent_hashes,
        'train_contexts':len(train),'validation_contexts':len(val),'gate':{'diffusion_TV_max':.15,'diffusion_minus_sphere_TV_max':.05},
        'scope':'Common fixed finite endpoint distribution fitting; not RL or cumulative reward experiment.'}
    atomic_json(out/'manifest.json',manifest);print(out/'manifest.json')


if __name__=='__main__':main()
