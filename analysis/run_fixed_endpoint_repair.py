"""Matched, source/data-frozen conditional actor fitting with exact target law."""
import argparse,hashlib,json,os
from pathlib import Path
import sys,time,traceback
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from solvercell.preference_policy import PreferenceActor
from solvercell.environment import InterventionEnv
from solvercell.preference_data import PreferenceTask
from solvercell.storage import atomic_json,append_json,atomic_checkpoint,source_manifest,worker_identity


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probabilities(values,prefs,temperature):return torch.softmax((values*prefs[:,None]).sum(-1)/temperature,-1)


@torch.no_grad()
def evaluate(actor,task,obs,values,cfg,*,samples,weights,seed,monitor=False):
    rows=[];lookup=np.full(1<<task.action_dim,-1,np.int32);bits=2**np.arange(task.action_dim)
    actions=cfg['_actions'];lookup[(actions*bits).sum(1).astype(int)]=np.arange(len(actions))
    old=actor.final_noise;modes=[False] if actor.kind=='sphere' else ([False] if monitor else [False,True])
    gen=torch.Generator().manual_seed(seed);truth_rng=np.random.default_rng(seed)
    for noise in modes:
        gen=torch.Generator().manual_seed(seed);truth_rng=np.random.default_rng(seed)
        actor.final_noise=noise
        for i,o in enumerate(obs):
            for w in weights:
                pref=torch.tensor([w,1-w]);p=probabilities(values[i:i+1],pref[None],cfg['temperature'])[0].double().numpy()
                counts=np.zeros(len(actions),np.int64);left=samples
                while left:
                    n=min(left,cfg['chunk_size']);cost=actor.sample(o[None].expand(n,-1),pref[None].expand(n,-1),gen)
                    decoded=task.decode(cost).numpy();idx=lookup[(decoded*bits).sum(1).astype(int)]
                    assert (idx>=0).all();counts+=np.bincount(idx,minlength=len(actions));left-=n
                q=counts/samples;u=(values[i]*pref).sum(-1).numpy();truth=truth_rng.multinomial(samples,p/p.sum())/samples
                rows.append({'context':i,'weight':w,'final_noise':noise,'samples':samples,'counts':counts.tolist(),
                    'target_probabilities':p.tolist(),'TV':float(np.abs(q-p).sum()/2),
                    'target_sampling_TV_floor_draw':float(np.abs(truth-p).sum()/2),
                    'target_utility':float(p@u),'sampled_utility':float(q@u),
                    'target_minus_sampled_utility':float((p-q)@u),'unique_actions':int((counts>0).sum()),
                    'actor_rows':samples*actor.steps,'decoder_rows':samples,'q_rows':0})
    actor.final_noise=old
    return rows


def run(args):
    args.output.mkdir(parents=True,exist_ok=False);identity=worker_identity();atomic_json(args.output/'worker.json',identity)
    m=json.loads(args.manifest.read_text());cfg=json.loads(args.config.read_text());job=next(j for j in m['jobs'] if j['id']==cfg['id'])
    assert sha(args.config)==job['config_sha256'];assert source_manifest()['source_sha256']==m['source_sha256'];assert sha(__file__)==m['runner_sha256']
    data_path=ROOT/m['data_path'];assert sha(data_path)==cfg['data_sha256']==m['data_sha256'];data=np.load(data_path)
    atomic_json(args.output/'config.json',cfg);atomic_json(args.output/'source.json',source_manifest());atomic_json(args.output/'invocation.json',{'runner_sha256':sha(__file__),'manifest_sha256':sha(args.manifest),'config_sha256':sha(args.config),'data_sha256':sha(data_path)})
    train_obs=torch.tensor(data['train_obs']);val_obs=torch.tensor(data['validation_obs']);train_values=torch.tensor(data['train_values']);val_values=torch.tensor(data['validation_values']);costs=torch.tensor(data['costs']);cfg['_actions']=data['actions'];task=PreferenceTask(InterventionEnv())
    with torch.random.fork_rng():
        torch.manual_seed(1800000+cfg['seed']);actor=PreferenceActor(task.obs_dim,task.cost_dim,hidden=128,steps=8,kind=cfg['method'],beta_end=.2,final_noise=False,prediction_type=cfg['prediction'])
    opt=torch.optim.Adam(actor.parameters(),lr=.0004);data_gen=torch.Generator().manual_seed(1810000+cfg['seed']);noise_gen=torch.Generator().manual_seed(1820000+cfg['seed']);prefs=torch.tensor(cfg['train_preferences']);start=time.perf_counter();curves=[]
    for step in range(cfg['steps']+1):
        if step:
            i=torch.randint(len(train_obs),(cfg['batch_size'],),generator=data_gen);j=torch.randint(len(prefs),(cfg['batch_size'],),generator=data_gen);w=prefs[j];p=torch.stack((w,1-w),-1)
            target_p=probabilities(train_values[i],p,cfg['temperature']);target_i=torch.multinomial(target_p,1,generator=data_gen).flatten();targets=costs[target_i]
            opt.zero_grad(set_to_none=True);loss,diag=actor.loss(train_obs[i],p,targets,generator=noise_gen);loss.backward();grad=torch.nn.utils.clip_grad_norm_(actor.parameters(),5.);opt.step()
            assert torch.isfinite(loss) and torch.isfinite(grad)
            if step%cfg['log_interval']==0:append_json(args.output/'training.jsonl',{'step':step,'loss':float(loss),'grad_norm':float(grad),'loss_kind':cfg['prediction'] if actor.kind=='diffusion' else 'velocity'})
        if step%cfg['eval_interval']==0 or step==cfg['steps']:
            rows=evaluate(actor,task,train_obs[:4],train_values[:4],cfg,samples=32 if m['debug'] else 512,weights=[0.,.5,1.],seed=1830000+cfg['seed'],monitor=True)
            point={'step':step,'training_probe_TV':float(np.mean([r['TV'] for r in rows])),'training_probe_utility_gap':float(np.mean([r['target_minus_sampled_utility'] for r in rows]))};curves.append(point);append_json(args.output/'curve.jsonl',point)
            atomic_json(args.output/'status.json',{'status':'running','phase':'fitting','step':step});print(json.dumps(point),flush=True)
    atomic_json(args.output/'status.json',{'status':'running','phase':'heldout_distribution_evaluation','step':cfg['steps']})
    # Heldout contexts/preferences are queried only after the fixed training budget.
    rows=evaluate(actor,task,val_obs,val_values,cfg,samples=cfg['eval_samples'],weights=cfg['heldout_preferences'],seed=1840000+cfg['seed'])
    serial_cfg={k:v for k,v in cfg.items() if k!='_actions'}
    atomic_checkpoint(args.output/'checkpoint.pt',{'actor':actor.state_dict(),'optimizer':opt.state_dict(),'data_generator':data_gen.get_state(),'noise_generator':noise_gen.get_state(),'steps':cfg['steps'],'config':serial_cfg,'source_sha256':m['source_sha256'],'data_sha256':sha(data_path)})
    monitor_rows=len(curves)*min(4,len(train_obs))*3*(32 if m['debug'] else 512)
    totals={'monitor_actor_rows':monitor_rows*8,'monitor_decoder_rows':monitor_rows,'training_actor_rows':cfg['steps']*cfg['batch_size'],'sampling_actor_rows':sum(r['actor_rows'] for r in rows),'decoder_rows':sum(r['decoder_rows'] for r in rows),'Q_rows':0}
    result={'status':'complete','job':cfg['id'],'steps':cfg['steps'],'source_sha256':m['source_sha256'],'data_sha256':sha(data_path),'curves':curves,'evaluations':rows,'compute_totals':totals,'parameters':sum(p.numel() for p in actor.parameters()),'wall_seconds':time.perf_counter()-start,'scope':m['scope']}
    atomic_json(args.output/'summary.json',result);atomic_json(args.output/'status.json',{'status':'complete','steps':cfg['steps'],'wall_seconds':result['wall_seconds']})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,type=Path);p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);code=1;error=None
    try:run(args);code=0
    except Exception as e:error=repr(e);traceback.print_exc()
    finally:
        path=args.output/'worker.json'
        if path.exists() and json.loads(path.read_text()).get('pid')==os.getpid() and json.loads(path.read_text()).get('start_ticks')==Path('/proc/self/stat').read_text().split()[21]:atomic_json(args.output/'exit.json',{'returncode':code,'error':error})
    raise SystemExit(code)
