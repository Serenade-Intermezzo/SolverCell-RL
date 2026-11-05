# Enumeration check: soft value V=alpha*log E_rho[exp(Q/alpha)] without log pi.
# log Z_lin is integrated along the shift using only posterior means at t=0; residual part uses K particles.
import itertools,math,random
random.seed(20261108)
sig=lambda x:1/(1+math.exp(-x))
m,k=14,4
A=[tuple(1 if i in c else 0 for i in range(m)) for c in itertools.combinations(range(m),k)]
def inst():
    W=[[random.uniform(.5,1.5) if random.random()<6/m else 0 for _ in range(m)] for _ in range(2*m)]
    Am=[random.uniform(.3,1) for _ in W]; x=[1.6*sum(w)*k/m+random.uniform(-.5,1.5) for w in W]
    Q=lambda a:sum(c*sig(1.6*sum(wi*ai for wi,ai in zip(w,a))-xx) for c,w,xx in zip(Am,W,x))
    def grad(a):
        g=[0.]*m
        for c,w,xx in zip(Am,W,x):
            u=sig(1.6*sum(wi*ai for wi,ai in zip(w,a))-xx)
            for i in range(m): g[i]+=c*1.6*u*(1-u)*w[i]
        return g
    return Q,grad
def logmeanexp(v):
    M=max(v); return M+math.log(sum(math.exp(x-M) for x in v)/len(v))
def tilt(rho_lp,h):
    lw=[lp+sum(hi*ai for hi,ai in zip(h,a)) for a,lp in zip(A,rho_lp)]
    M=max(lw); w=[math.exp(x-M) for x in lw]; return w
def mean_of(w):
    Z=sum(w); return [sum(wi*a[i] for wi,a in zip(w,A))/Z for i in range(m)]
K=12; rho=[0.]*len(A)
for beta in (2,5,10):
    errp=[];errs=[];idv=[]
    for inst_i in range(30):
        Q,grad=inst(); v=[Q(a) for a in A]; mu=sum(v)/len(v); alpha=math.sqrt(sum((x-mu)**2 for x in v)/len(v))/beta
        true=logmeanexp([x/alpha for x in v])
        g=[x/alpha for x in grad([k/m]*m)]
        # log E_rho exp<g,a> by trapezoid along tau, means only
        n=8; vals=[sum(gi*mi for gi,mi in zip(g,mean_of(tilt(rho,[tau/n*gi for gi in g])))) for tau in range(n+1)]
        lz=(sum(vals)-.5*(vals[0]+vals[-1]))/n
        idv.append(abs(lz-logmeanexp([sum(gi*ai for gi,ai in zip(g,a)) for a in A])))
        w=tilt(rho,g); R=[x/alpha-sum(gi*ai for gi,ai in zip(g,a)) for x,a in zip(v,A)]
        for _ in range(20):
            errp.append(logmeanexp([v[j]/alpha for j in random.choices(range(len(A)),k=K)])-true)
            errs.append(lz+logmeanexp([R[j] for j in random.choices(range(len(A)),weights=w,k=K)])-true)
    st=lambda e:(round(sum(e)/len(e),3),round(math.sqrt(sum(x*x for x in e)/len(e)),3))
    print(f'beta={beta}: |logZ_lin quad err| mean {sum(idv)/len(idv):.2e}; plain K=12 (bias,RMSE) {st(errp)}; shift+residual K=12 (bias,RMSE) {st(errs)}')
