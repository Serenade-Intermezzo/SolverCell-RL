# Pure-Python check of the linear-tilt shift identity on a feasible set, by exact enumeration.
# Forward channel: z = alpha*e(a) + sigma*eps, e(a)=2a-1. Claim: D+_t(z) = D_t(z + sigma^2 q/(alpha*lam)).
import itertools, math, random
random.seed(0)
def feasible(m,k): # up-to-budget subsets
    return [tuple(1 if i in c else -1 for i in range(m)) for r in range(k+1) for c in itertools.combinations(range(m),r)]
def post_mean(A,logpi,z,alpha,sigma):
    lw=[lp+alpha*sum(zi*ei for zi,ei in zip(z,e))/sigma**2 for e,lp in zip(A,logpi)]
    M=max(lw); w=[math.exp(x-M) for x in lw]; Z=sum(w)
    return [sum(wi*e[i] for wi,e in zip(w,A))/Z for i in range(len(z))]
worst=0.0
for trial in range(200):
    m=random.choice([6,8,10]); k=random.randint(1,m//2)
    A=feasible(m,k)
    logpi=[random.gauss(0,1.5) for _ in A]        # arbitrary prior on A(s)
    q=[random.gauss(0,1) for _ in range(m)]; lam=random.uniform(.2,3)
    alpha=random.uniform(.05,1); sigma=random.uniform(.05,2)
    z=[random.gauss(0,2) for _ in range(m)]
    logpi_tilt=[lp+sum(qi*ei for qi,ei in zip(q,e))/lam for e,lp in zip(A,logpi)]
    lhs=post_mean(A,logpi_tilt,z,alpha,sigma)
    d=sigma**2/(alpha*lam)
    rhs=post_mean(A,logpi,[zi+d*qi for zi,qi in zip(z,q)],alpha,sigma)
    worst=max(worst,max(abs(a-b) for a,b in zip(lhs,rhs)))
print("max |D+(z)-D(z+shift)| over 200 random (A,pi,q,lam,alpha,sigma,z):",worst)
