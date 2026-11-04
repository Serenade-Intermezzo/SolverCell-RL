"""Verify storage optimization left a complete seeded training run unchanged."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--old',required=True);p.add_argument('--new',required=True);p.add_argument('--output',required=True);args=p.parse_args()
old,new=Path(args.old),Path(args.new)
a=torch.load(old/'checkpoint.pt',map_location='cpu',weights_only=False)
b=torch.load(new/'checkpoint.pt',map_location='cpu',weights_only=False)
checks={}
def same(x,y,path):
 if isinstance(x,dict):
  assert x.keys()==y.keys(),path
  for k in x:
   if 'seconds' not in str(k):same(x[k],y[k],path+'.'+str(k))
 elif isinstance(x,(tuple,list)):
  assert len(x)==len(y),path
  for i,(v,w) in enumerate(zip(x,y)):same(v,w,path+'.'+str(i))
 elif isinstance(x,np.ndarray):
  assert np.array_equal(x,np.asarray(y)),path
 elif torch.is_tensor(x):
  assert torch.equal(x,torch.as_tensor(y)),path
 else:assert x==y,path
same(a['config'],b['config'],'config');checks['identical_resolved_configuration']=True
same(a['learner'],b['learner'],'learner');checks['bitwise_identical_learner_and_optimizer_rng']=True
for k in ('env_steps','train_steps','episode_count','episode_return','env_state','env_t','env_rng','replay_rng','behavior_rng','python_rng','numpy_rng','torch_rng','cuda_rng','pretrain_done','instance_hash'):
 same(a[k],b[k],k)
checks['identical_environment_counters_and_all_rng']=True
for k in ('size','pos','capacity'):same(a['replay'][k],b['replay'][k],'replay.'+k)
size=a['replay']['size']
for k in a['replay']['data']:
 same(a['replay']['data'][k][:size],b['replay']['data'][k][:size],'replay.data.'+k)
checks['identical_valid_replay_rows_and_ring_position']=True
for name in ('evaluations.jsonl','episodes.jsonl'):
 ar=[json.loads(l) for l in (old/name).read_text().splitlines()];br=[json.loads(l) for l in (new/name).read_text().splitlines()]
 for row in ar+br:row.pop('eval_seconds',None)
 same(ar,br,name)
checks['identical_complete_validation_heldout_and_training_episodes']=True
r={'verified':all(checks.values()),'old_run':str(old.resolve()),'new_run':str(new.resolve()),'old_source':a['source_sha256'],'new_source':b['source_sha256'],'checks':checks,'old_checkpoint_bytes':(old/'checkpoint.pt').stat().st_size,'new_checkpoint_bytes':(new/'checkpoint.pt').stat().st_size}
Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
