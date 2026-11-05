# Adversarial controls for gate0, same oracle denoiser and critic. NOT an RL result.
# Key questions: does stochasticity matter (SDE vs probability-flow ODE, the Gaussian-path FM control)?
# Is the chain needed at all (mean-field fixed point on the policy, no generative chain)?
import sys,json,itertools,numpy as np
from gate0 import Base,Sigmoid,Pairs,marginals,sample,gibbs,snis
def grid(T,n,kind): 
    return T*(np.arange(0,n+1)/n)**2 if kind=='quad' else T*np.arange(0,n+1)/n
def topk(score,k):
    idx=np.argsort(-score,axis=1)[:,:k]; a=np.zeros_like(score); np.put_along_axis(a,idx,1,1); return a
def sde(base,C,lam,gr,m,k,N,n,T,rng,relin=True,kind='uniform'):
    y=np.zeros((N,m)); sh=np.repeat(gr[None],N,0); ts=grid(T,n,kind)
    for i in range(n):
        h=ts[i+1]-ts[i]
        if relin: sh=C.grad(base.denoise(2*y+sh))/lam
        y=y+h*(2*base.denoise(2*y+sh)-1)+np.sqrt(h)*rng.normal(size=y.shape)
    return topk(y+sh/2,k)
def ode(base,C,lam,gr,m,k,N,n,T,rng,relin=True):
    # probability-flow ODE of the same localisation path: dy=.5*(D+y/t)dt; randomness only in the start
    ts=grid(T,n,'quad'); t0=ts[1]; sh=np.repeat(gr[None],N,0)
    if relin: sh=C.grad(base.denoise(sh))/lam
    # exact start law at t0: sample e from the tilted policy under the current linearisation, then Gaussian channel
    e=2*base.sample_shift(sh[0],N,rng)-1; y=t0*e+np.sqrt(t0)*rng.normal(size=(N,m))
    for i in range(1,n):
        h=ts[i+1]-ts[i]; t=ts[i]
        if relin: sh=C.grad(base.denoise(2*y+sh))/lam
        D=2*base.denoise(2*y+sh)-1
        y=y+h*.5*(D+y/t)
    return topk(y+sh/2,k)
def meanfield(base,C,lam,gr,m,N,n,rng,rho=.5):
    sh=gr.copy()
    for _ in range(n): sh=(1-rho)*sh+rho*C.grad(base.denoise(sh[None]))[0]/lam
    return base.sample_shift(sh,N,rng)
def enum(m,k):
    A=np.zeros((0,m)); out=[]
    for c in itertools.combinations(range(m),k):
        a=np.zeros(m); a[list(c)]=1; out.append(a)
    return np.array(out)
def tv_small(rng,N=4096,m=14,k=4,n=16,T=16.):
    A=enum(m,k); key=lambda a:(a@(2**np.arange(m))).astype(np.int64)
    pos={int(v):i for i,v in enumerate(key(A))}
    def hist(a):
        h=np.zeros(len(A)); 
        for v in key(a): h[pos[int(v)]]+=1
        return h/h.sum()
    base=Base(np.zeros((1,m)),k); rows=[]
    for qt in ('sigmoid','and','xor'):
      for beta in (2,5,10):
        acc={}
        for inst in range(10):
            Q={'sigmoid':lambda:Sigmoid(m,k,rng),'and':lambda:Pairs(m,k,rng),'xor':lambda:Pairs(m,k,rng,xor=True)}[qt]()
            v=Q(A); lam=v.std()/beta; p=np.exp((v-v.max())/lam); p/=p.sum()
            mu0=base.denoise(np.zeros((1,m))); gr=Q.grad(mu0)[0]/lam
            def rec(name,a):
                h=hist(a); acc.setdefault(name,[]).append((.5*np.abs(h-p).sum(),(h@v-v.mean())/v.std()))
            rec('target_floor',A[rng.choice(len(A),N,p=p)])
            for K in (12,48):
                ax=A[rng.integers(len(A),size=(N,K))]; vv=Q(ax); ww=np.exp(vv/lam-(vv/lam).max(1,keepdims=True)); ww/=ww.sum(1,keepdims=True)
                ch=(ww.cumsum(1)>rng.random((N,1))).argmax(1); rec(f'snis{K}',ax[np.arange(N),ch])
            ax=base.sample_shift(gr,N*12,rng).reshape(N,12,m); vv=Q(ax)/lam-ax@gr; ww=np.exp(vv-vv.max(1,keepdims=True)); ww/=ww.sum(1,keepdims=True)
            ch=(ww.cumsum(1)>rng.random((N,1))).argmax(1); rec('shift_res12',ax[np.arange(N),ch])
            rec('shift',base.sample_shift(gr,N,rng))
            rec('mf16',meanfield(base,Q,lam,gr,m,N,16,rng))
            rec('sde16',sde(base,Q,lam,gr,m,k,N,16,T,rng))
            rec('sde16q',sde(base,Q,lam,gr,m,k,N,16,T,rng,kind='quad'))
            rec('sde64q',sde(base,Q,lam,gr,m,k,N,64,T,rng,kind='quad'))
            rec('ode16',ode(base,Q,lam,gr,m,k,N,16,T,rng))
            rec('ode64',ode(base,Q,lam,gr,m,k,N,64,T,rng))
        r={'q':qt,'beta':beta}; 
        for a,b in acc.items():
            b=np.array(b); r[a]=[float(b[:,0].mean()),float(b[:,1].mean())]
        rows.append(r); print(json.dumps({x:(np.round(y,3).tolist() if isinstance(y,list) else y) for x,y in r.items()}),flush=True)
    return rows
def large(rng,m=120,k=30,n=16,T=16.,N=64):
    base=Base(np.zeros((1,m)),k); rows=[]
    for qt in ('sigmoid','and','xor'):
      for beta in (5,10):
        acc={}
        for inst in range(4):
            Q={'sigmoid':lambda:Sigmoid(m,k,rng),'and':lambda:Pairs(m,k,rng),'xor':lambda:Pairs(m,k,rng,xor=True)}[qt]()
            a0=base.sample_shift(np.zeros(m),4096,rng); q0=Q(a0); mean0,std0=q0.mean(),q0.std(); lam=std0/beta
            gr=Q.grad(base.denoise(np.zeros((1,m))))[0]/lam
            def rec(name,a):
                acc.setdefault(name,[]).append(float((Q(a).mean()-mean0)/std0))
                # diversity: mean pairwise Hamming distance, in item units
                acc.setdefault(name+'_div',[]).append(float(np.abs(a[:32,None]-a[32:64][None]).sum(-1).mean()/2))
            rec('sde16',sde(base,Q,lam,gr,m,k,N,16,T,rng)); rec('ode16',ode(base,Q,lam,gr,m,k,N,16,T,rng))
            rec('sde64q',sde(base,Q,lam,gr,m,k,N,64,T,rng,kind='quad')); rec('ode64',ode(base,Q,lam,gr,m,k,N,64,T,rng))
            rec('mf16',meanfield(base,Q,lam,gr,m,N,16,rng)); rec('shift',base.sample_shift(gr,N,rng))
        r={'q':qt,'beta':beta,'m':m,**{a:float(np.mean(b)) for a,b in acc.items()}}
        rows.append(r); print(json.dumps({x:(round(y,2) if isinstance(y,float) else y) for x,y in r.items()}),flush=True)
    return rows
if __name__=='__main__':
    rng=np.random.default_rng(20261106)
    out={'tv':tv_small(rng),'large':large(rng)}
    json.dump(out,open(sys.argv[1],'w'),indent=1)
