# Gate-0 mechanism check: oracle denoiser on exact-k subsets, known Q, frozen. NOT an RL result.
# Question: at a fixed Q-query budget, how much of the PMD tilt pi*exp(Q/lam) does each update realise?
import sys, json, numpy as np
sig=lambda x:1/(1+np.exp(-x))

def log_esp_tables(eta,k):
    # eta: (b,m) natural params for a in {0,1}. F[i][b,c]=log e_c(eta_1..i); B[i][b,c]=log e_c(eta_{i+1}..m)
    b,m=eta.shape; NEG=-1e18
    F=np.full((m+1,b,k+1),NEG); F[0,:,0]=0
    for i in range(m):
        F[i+1]=F[i].copy()
        F[i+1][:,1:]=np.logaddexp(F[i][:,1:],F[i][:,:-1]+eta[:,i,None])
    B=np.full((m+1,b,k+1),NEG); B[m,:,0]=0
    for i in range(m-1,-1,-1):
        B[i]=B[i+1].copy()
        B[i][:,1:]=np.logaddexp(B[i+1][:,1:],B[i+1][:,:-1]+eta[:,i,None])
    return F,B
def marginals(eta,k):
    F,B=log_esp_tables(eta,k); b,m=eta.shape
    logZ=F[m][:,k]; mu=np.zeros((b,m))
    for i in range(m):
        # item i selected: sum_c F[i][c]+eta_i+B[i+1][k-1-c]
        t=F[i][:,:k]+B[i+1][:,k-1::-1]
        mu[:,i]=np.exp(np.logaddexp.reduce(t,axis=1)+eta[:,i]-logZ)
    return np.clip(mu,0,1),logZ
def sample(eta,k,rng):
    _,B=log_esp_tables(eta,k); b,m=eta.shape
    a=np.zeros((b,m)); left=np.full(b,k); ix=np.arange(b)
    for i in range(m):
        lin=eta[:,i]+B[i+1][ix,np.maximum(left-1,0)]; lout=B[i+1][ix,left]
        p=np.where(left>0,np.exp(lin-np.logaddexp(lin,lout)),0.)
        c=rng.random(b)<p; a[:,i]=c; left=left-c
    return a
class Base:
    # mixture of exact-k exponential families; C=1,eta=0 is uniform on the feasible set
    def __init__(s,etas,k): s.etas=etas; s.k=k; s.logw=-marginals(etas,k)[1]*0+np.log(1/len(etas)); s.logZ0=marginals(etas,k)[1]
    def denoise(s,y):
        # posterior mean of a given natural-parameter offset y (b,m), in {0,1} coords
        ms=[];lz=[]
        for c,e in enumerate(s.etas):
            mu,l=marginals(e[None]+y,s.k); ms.append(mu); lz.append(l-s.logZ0[c]+s.logw[c])
        lz=np.stack(lz,1); w=np.exp(lz-lz.max(1,keepdims=True)); w/=w.sum(1,keepdims=True)
        return (w[:,:,None]*np.stack(ms,1)).sum(1)
    def sample_shift(s,g,n,rng):
        # exact samples from base*exp(<g,a>) for a single g (m,)
        y=np.repeat(g[None],n,0); lz=[]
        for c,e in enumerate(s.etas): lz.append(marginals((e+g)[None],s.k)[1][0]-s.logZ0[c]+s.logw[c])
        lz=np.array(lz); w=np.exp(lz-lz.max()); w/=w.sum()
        comp=rng.choice(len(w),n,p=w); a=np.zeros((n,len(g)))
        for c in range(len(w)):
            ix=np.where(comp==c)[0]
            if len(ix): a[ix]=sample(np.repeat((s.etas[c]+g)[None],len(ix),0),s.k,rng)
        return a
class Sigmoid:
    # benchmark-shaped S-curve link: Q=sum_j A_j sigmoid(b_j (W_j.a) - x_j), mild synergy
    def __init__(s,m,k,rng,steep=1.6):
        N=2*m; s.W=(rng.random((N,m))<6/m)*rng.uniform(.5,1.5,(N,m))
        s.A=rng.uniform(.3,1.,N); s.b=np.full(N,steep); s.x=s.b*s.W.sum(1)*k/m+rng.uniform(-.5,1.5,N)
    def __call__(s,a): return (s.A*sig(s.b*(a@s.W.T)-s.x)).sum(-1)
    def grad(s,a):
        u=sig(s.b*(a@s.W.T)-s.x); return (s.A*s.b*u*(1-u))@s.W
class Pairs:
    # sign=+1 AND-bonus on disjoint pairs (strong synergy); sign=-1 XOR on pairs (gradient vanishes at symmetric mean)
    def __init__(s,m,k,rng,xor=False):
        p=rng.permutation(m)[:m//2*2].reshape(-1,2); s.i,s.j=p[:,0],p[:,1]; s.c=rng.uniform(.5,1.5,len(p)); s.xor=xor
    def __call__(s,a):
        x,y=a[...,s.i],a[...,s.j]
        return ((x+y-2*x*y if s.xor else x*y)*s.c).sum(-1)
    def grad(s,a):
        g=np.zeros_like(a); x,y=a[:,s.i],a[:,s.j]
        g[:,s.i]=((1-2*y) if s.xor else y)*s.c; g[:,s.j]=((1-2*x) if s.xor else x)*s.c; return g
class Noisy:
    # critic = truth + smooth Fourier error; amp is relative to Q std under the base policy
    def __init__(s,Q,m,amp,std,rng,F=64):
        s.Q=Q; s.om=rng.normal(0,1.2,(F,m)); s.ph=rng.uniform(0,6.28,F); s.c=amp*std*np.sqrt(2/F)
    def __call__(s,a): return s.Q(a)+s.c*np.sin(a@s.om.T+s.ph).sum(-1)
    def grad(s,a): return s.Q.grad(a)+s.c*(np.cos(a@s.om.T+s.ph)@s.om)

def gibbs(base,Q,lam,k,m,rng,chains=64,steps=None):
    # long swap-MH on the true target (uniform base only), reference E[Q]
    steps=steps or 25*m; a=sample(np.zeros((chains,m)),k,rng); v=Q(a); ix=np.arange(chains); tr=[]
    for t in range(steps):
        on=np.argsort(rng.random((chains,m))-a,axis=1)  # selected items first
        i=on[:,0]; j=on[:,-1]
        b=a.copy(); b[ix,i]=0; b[ix,j]=1; w=Q(b)
        acc=np.log(rng.random(chains))<(w-v)/lam
        a[acc]=b[acc]; v[acc]=w[acc]
        if t>=steps*2//3 and t%10==0: tr.append(v.mean())
    return float(np.mean(tr))
def snis(a,logw,rng):
    p=np.exp(logw-logw.max()); p/=p.sum(); return a[rng.choice(len(a),p=p)], 1/np.sum(p**2)
def run_state(m,k,qtype,beta,K,amp,rng,mixture=False,reps=256,n=16,T=16.):
    if mixture:
        s=rng.choice([-1.,1.],m)*1.5; etas=np.stack([s,-s])
    else: etas=np.zeros((1,m))
    base=Base(etas,k)
    Q={'sigmoid':lambda:Sigmoid(m,k,rng),'and':lambda:Pairs(m,k,rng),'xor':lambda:Pairs(m,k,rng,xor=True)}[qtype]()
    a0=base.sample_shift(np.zeros(m),4096,rng); q0=Q(a0); mean0,std0=q0.mean(),q0.std()
    lam=std0/beta                                  # Gaussian-Q heuristic: requested gain = beta*std, KL = beta^2/2
    C=Noisy(Q,m,amp,std0,rng) if amp>0 else Q      # methods see only C; scoring uses true Q
    out={}
    g=lambda vals:float((np.mean(vals)-mean0)/std0)
    mu0=base.denoise(np.zeros((1,m)))
    # linear R^2 of true Q under the base policy
    X=a0-a0.mean(0); w_=np.linalg.lstsq(X,q0-mean0,rcond=None)[0]; out['r2']=float(1-np.var(q0-mean0-X@w_)/np.var(q0))
    # M0 plain SNIS(K) from pi; M1 best-of-K (LSFlow-style selection)
    v0,v1,ess=[],[],[]
    A=base.sample_shift(np.zeros(m),K*reps,rng).reshape(reps,K,m); CV=C(A.reshape(-1,m)).reshape(reps,K)
    for r in range(reps):
        pick,e=snis(A[r],CV[r]/lam,rng); v0.append(Q(pick[None])[0]); ess.append(e); v1.append(Q(A[r][np.argmax(CV[r])][None])[0])
    out['snis']=g(v0); out['ess']=float(np.mean(ess)); out['bestK']=g(v1)
    # M2 one critic gradient at the base mean, exact tilt by shift (budget 1 gradient)
    gr=C.grad(mu0)[0]/lam
    a=base.sample_shift(gr,reps,rng); out['shift']=g(Q(a))
    # M3 shift proposal + residual-only SNIS (budget K Q + 1 gradient); pi cancels, no density needed
    v=[];ess=[]
    A=base.sample_shift(gr,K*reps,rng).reshape(reps,K,m); CV=C(A.reshape(-1,m)).reshape(reps,K)
    for r in range(reps):
        pick,e=snis(A[r],CV[r]/lam-A[r]@gr,rng); v.append(Q(pick[None])[0]); ess.append(e)
    out['shift_res']=g(v); out['ess_res']=float(np.mean(ess))
    # M4 SL chain dy=D dt+dW with re-linearised shift; n gradients, 2n denoiser calls, solver only at the end
    def chain(relin):
        y=np.zeros((reps,m)); h=T/n; sh=np.repeat(gr[None],reps,0)
        for t in range(n):
            if relin:
                sh=C.grad(base.denoise(2*y+sh))/lam
            D=2*base.denoise(2*y+sh)-1
            y=y+h*D+np.sqrt(h)*rng.normal(size=y.shape)
        idx=np.argsort(-(y+sh/2),axis=1)[:,:k]; a=np.zeros_like(y); np.put_along_axis(a,idx,1,1); return a
    out['chain_fixed']=g(Q(chain(False)))
    out['chain_relin']=g(Q(chain(True)))
    # M5 unguided chain: base-law reproduction / discretisation sanity (gain must be near 0)
    sv=gr; gr=np.zeros(m); out['chain_base']=g(Q(chain(False))); gr=sv
    if not mixture: out['target']=(gibbs(base,Q,lam,k,m,rng)-mean0)/std0
    return out
if __name__=='__main__':
    rng=np.random.default_rng(20261105); rows=[]
    grid=[(40,10),(120,30)]
    for mix in (False,True):
      for qt in ('sigmoid','and','xor'):
        if mix and qt!='sigmoid': continue
        for m,k in grid:
          for beta in (2,5,10):
            for amp in ((0.,.3) if qt=='sigmoid' else (0.,)):
              res=[run_state(m,k,qt,beta,12,amp,rng,mixture=mix,reps=64) for _ in range(6)]
              r={x:float(np.mean([d[x] for d in res])) for x in res[0]}
              r.update(m=m,k=k,q=qt,beta=beta,amp=amp,mix=mix); rows.append(r)
              print(json.dumps({a:round(b,2) if isinstance(b,float) else b for a,b in r.items()}),flush=True)
    json.dump(rows,open(sys.argv[1],'w'),indent=1)
