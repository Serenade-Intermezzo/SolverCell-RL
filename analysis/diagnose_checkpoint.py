"""Read-only final-checkpoint diagnostics; never changes training artifacts/RNG."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from solvercell.config import load_config
from solvercell.environment import InterventionEnv
from solvercell.learner import Learner
from solvercell.geometry import project_cell

p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);args=p.parse_args()
run=Path(args.run);cfg=load_config(run/'config.json')
if cfg.method=='lsflow_upstream':raise ValueError('This diagnostic is for matched action-critic methods')
torch.set_num_threads(1)
checkpoint=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False)
env=InterventionEnv(cfg.n_arms,cfg.n_actions,cfg.budget,cfg.horizon,instance_seed=cfg.instance_seed)
learner=Learner(cfg,env);learner.load_state_dict(checkpoint['learner'])
rng=np.random.default_rng(882013);replay=checkpoint['replay'];indices=rng.integers(replay['size'],size=512)
obs=learner.tensor_obs(replay['data']['obs'][indices]);g=torch.Generator().manual_seed(882014)
with torch.no_grad():
 costs,acts,expanded=learner.candidates(obs,cfg.candidates,target=True,generator=g)
 values=torch.minimum(learner.tq1(expanded,acts.flatten(0,1)),learner.tq2(expanded,acts.flatten(0,1))).reshape(len(obs),-1)
 ess={};maxw={}
 for temp in (1.,.25,.1,.05,.025):
  w=torch.softmax(values/temp,-1);ess[str(temp)]=float((1/w.square().sum(-1)).mean());maxw[str(temp)]=float(w.max(-1).values.mean())
 w=torch.softmax(values/cfg.temperature,-1)
 idx=torch.multinomial(w,1,generator=g).squeeze(1);rows=torch.arange(len(obs))
 chosen=acts[rows,idx];old=costs[rows,idx]
 z=torch.randn(old.shape,generator=g)
 projected=project_cell(z,chosen,cfg.budget,cfg.margin)
 old_projected=project_cell(old,chosen,cfg.budget,cfg.margin)
 independent=project_cell(torch.randn(z.shape,generator=g),chosen,cfg.budget,cfg.margin)
 distance=lambda y:float((y-z).square().sum(-1).mean())
 result={'run':str(run.resolve()),'train_steps':checkpoint['train_steps'],'source_sha256':checkpoint['source_sha256'],
         'instance_hash':env.instance_hash,'diagnostic_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'sampling_seed':882014,'replay_sample_seed':882013,'states':len(obs),'candidates':cfg.candidates,
         'candidate_q_std_mean':float(values.std(-1).mean()),'candidate_q_range_mean':float((values.max(-1).values-values.min(-1).values).mean()),
         'ess_by_temperature':ess,'max_weight_by_temperature':maxw,
         'unique_candidates_mean':float(np.mean([len(torch.unique(a,dim=0)) for a in acts])),
         'candidate_action_count_mean':float(acts.sum(-1).mean()),'sampled_cost_norm_mean':float(costs.norm(dim=-1).mean()),
         'transport_paired_cell':distance(projected),'transport_independent_cell':distance(independent),
         'transport_margin_point':distance(old_projected)}
Path(args.output).parent.mkdir(parents=True,exist_ok=True)
Path(args.output).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result))
