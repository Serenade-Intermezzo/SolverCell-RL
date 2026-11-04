"""Deep read-only artifact audit for a frozen campaign."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from tools.campaign import audit_complete, read_json, resolved_config
from solvercell.config import is_native_method

p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--output',required=True);args=p.parse_args()
root=Path(args.root);manifest=read_json(args.manifest);rows=[]
torch.set_num_threads(1)
def walk_finite(value,path='learner'):
 if torch.is_tensor(value):
  if value.is_floating_point() and not torch.isfinite(value).all():raise ValueError('nonfinite '+path)
 elif isinstance(value,np.ndarray):
  if np.issubdtype(value.dtype,np.floating) and not np.isfinite(value).all():raise ValueError('nonfinite '+path)
 elif isinstance(value,dict):
  for k,v in value.items():walk_finite(v,path+'.'+str(k))
 elif isinstance(value,(list,tuple)):
  for k,v in enumerate(value):walk_finite(v,path+'.'+str(k))
 elif isinstance(value,float) and not math.isfinite(value):raise ValueError('nonfinite '+path)
for job in manifest['jobs']:
 run=root/job['output'];cfg=resolved_config(root/job['config']);row={'id':job['id'],'output':str(run.resolve())}
 okay,reasons=audit_complete(run,cfg,manifest['expected_source_sha256'])
 if okay:
  try:
   c=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False);m=read_json(run/'metrics.json')
   assert c['config']==cfg and c['source_sha256']==manifest['expected_source_sha256']
   assert c['env_steps']==cfg['train_steps']+cfg['warmup_steps'] and c['train_steps']==cfg['train_steps']
   assert c['learner']['updates']==cfg['train_steps'] and c['replay']['size']==min(c['env_steps'],cfg['replay_capacity'])
   expected_actor=cfg['train_steps'] if is_native_method(cfg['method']) else (cfg['train_steps']+cfg['actor_interval']-1)//cfg['actor_interval']
   assert c['learner']['actor_updates']==expected_actor
   if is_native_method(cfg['method']):assert c['learner']['pretrain_updates']==8
   walk_finite(c['learner']);walk_finite(c['replay'])
   vals=[json.loads(line) for line in (run/'evaluations.jsonl').read_text().splitlines()]
   for v in vals:
    assert len(v['episode_returns'])==v['episodes'] and abs(float(np.mean(v['episode_returns']))-v['return_mean'])<1e-8
   validation=[v for v in vals if v['split']=='validation']
   xs=np.array([v['train_steps'] for v in validation]);ys=np.array([v['return_mean'] for v in validation])
   window=max(1,int(np.ceil(.2*cfg['train_steps']/cfg['eval_interval'])))
   assert abs(float(ys[-window:].mean())-m['last_window_return'])<1e-8
   assert abs(float(np.trapz(ys,xs)/cfg['train_steps'])-m['validation_auc'])<1e-8
   episodes=[json.loads(line) for line in (run/'episodes.jsonl').read_text().splitlines()]
   assert len(episodes)==c['env_steps']//cfg['horizon']==c['episode_count']
   assert all(v['length']==cfg['horizon'] for v in episodes)
   row.update(status='verified_complete',env_steps=c['env_steps'],train_steps=c['train_steps'],actor_updates=expected_actor,
              episodes=len(episodes),validation_evaluations=len(validation),heldout_episodes=cfg['final_eval_episodes'],
              all_checkpoint_tensors_finite=True,source_sha256=c['source_sha256'],instance_hash=c['instance_hash'])
  except Exception as e:
   row.update(status='invalid',errors=[type(e).__name__+': '+str(e)])
 else:row.update(status='not_verified_complete',errors=reasons)
 rows.append(row)
result={'campaign_id':manifest['campaign_id'],'runs':rows,'all_verified_complete':all(r['status']=='verified_complete' for r in rows)}
Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'campaign_id':manifest['campaign_id'],'all_verified_complete':result['all_verified_complete'],'runs':[(r['id'],r['status']) for r in rows]}))
