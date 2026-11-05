# Read-only: how linear in the action is the real Dynamic Intervention value? Uses the repo env, no training.
import sys,json,time,numpy as np
sys.path.insert(0,'/workspace/SolverCell-RL')
from solvercell.environment import InterventionEnv
env=InterventionEnv(); A,B=env.n_actions if hasattr(env,'n_actions') else 20,10
rng=np.random.default_rng(20261107)
def uniform(n):
    a=np.zeros((n,A)); 
    for i in range(n): a[i,rng.choice(A,B,replace=False)]=1
    return a
def r2(X,y):
    X0=X-X.mean(0); yy=y-y.mean(); w=np.linalg.lstsq(X0,yy,rcond=None)[0]; return 1-np.var(yy-X0@w)/np.var(yy)
def myopic_top(e,state):
    base=np.zeros(A); v=np.array([e.expected_reward(np.eye(A)[i],state) for i in range(A)])
    a=np.zeros(A); a[np.argsort(-v)[:B]]=1; return a
states=[]
for ep in range(6):
    e=env.clone(seed=1000+ep); s=e.reset(seed=1000+ep); 
    for h in range(20):
        st=e.state.copy() if hasattr(e,'state') else None
        if h in (0,5,10,15): states.append((h,st))
        s,_,d,_=e.step(myopic_top(e,st) if ep%2 else e.random_action(rng))
        if d: break
rows=[]
for h,st in states[:24]:
    X=uniform(1500); y=np.array([env.expected_reward(a,st) for a in X]); rows.append(('myopic',h,float(r2(X,y)),float(y.std())))
print('one-step R2 mean/min',np.mean([r[2] for r in rows]),np.min([r[2] for r in rows]),flush=True)
# long-horizon Q with a fixed continuation, split-half reliability corrected
out=[]
for h,st in [x for x in states if x[0] in (0,5,10)][:9]:
    t=time.time(); X=uniform(160); G=np.zeros((160,8))
    for i,a in enumerate(X):
        for r in range(8):
            e=env.clone(seed=int(rng.integers(1<<31)),copy_state=True); e.state=st.copy(); e.t=h if hasattr(e,'t') else 0
            if hasattr(e,'step_count'): e.step_count=h
            tot=0.; cur=a; done=False
            while not done:
                _,rw,done,_=e.step(cur); tot+=rw
                if not done: cur=myopic_top(e,e.state)
            G[i,r]=tot
    y=G.mean(1); a1,a2=G[:,:4].mean(1),G[:,4:].mean(1); rel=np.corrcoef(a1,a2)[0,1]; rel=2*rel/(1+rel)
    out.append(dict(h=h,r2=float(r2(X,y)),rel=float(rel),std=float(y.std()),sec=time.time()-t)); print(out[-1],flush=True)
json.dump(dict(myopic=rows,q=out),open(sys.argv[1],'w'))
