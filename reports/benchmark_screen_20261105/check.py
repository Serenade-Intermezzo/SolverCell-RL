"""Correctness checks for screen.py on instances small enough to enumerate."""
import itertools, json, math, random, sys, time, warnings
import numpy as np
warnings.filterwarnings('ignore', category=SyntaxWarning)
import screen as S
rng = np.random.default_rng(3); out = {}

# k-subset mean map against enumeration
eta = rng.standard_normal(7); sets = [a for r in range(4) for a in itertools.combinations(range(7), r)]; w = np.array([math.exp(eta[list(a)].sum()) for a in sets]); w /= w.sum()
ex = np.array([sum(p for a, p in zip(sets, w) if j in a) for j in range(7)]); out['k_subset'] = float(np.abs(S.k_subset_mean(eta, 3) - ex).max())

# routing mean map and DP solver against enumeration on the real graph with a short horizon
env = S.upstream_env('routing', 0); task = S.Routing(env); task.T, task.B = 6, 2
walks = [[]]
for t in range(task.T):
    walks = [w + [k] for w in walks for k in np.flatnonzero(task.adj[w[-1] if w else task.src])]
walks = [w for w in walks if w[-1] == task.src]; eta = rng.standard_normal((task.T, task.n)); mu = np.zeros((task.T, task.n)); Z = 0.
for w in walks:
    for r in range(task.B + 1):
        for pulls in itertools.combinations(range(task.T), r):
            v = math.exp(sum(eta[t, w[t]] for t in pulls)); Z += v
            for t in pulls: mu[t, w[t]] += v
out['routing_mean'] = float(np.abs(task.mean_map(eta) - mu / Z).max()); out['routing_walks'] = len(walks)
task = S.Routing(env); gaps = []; feas = 0
for _ in range(6):
    c_ = rng.standard_normal(task.n); a = task.solve(c_); x = task.solve_strict(c_); gaps.append(float((c_ @ x - c_ @ a) / (c_ @ x))); feas += task.solve_strict(np.zeros(task.n), fix=a) is not None
out['routing_beam_vs_strict'] = dict(gaps=gaps, feasible=feas, n=6)
t0 = time.perf_counter(); [task.solve(rng.standard_normal(task.n)) for _ in range(20)]; out['routing_beam_ms'] = (time.perf_counter() - t0) / 20 * 1e3

# assignment mean map against enumeration
cost, cap = np.array([2, 3, 2, 4]), np.array([4, 5]); eta = rng.standard_normal((2, 4)); mu = np.zeros((2, 4)); Z = 0.
for asg in itertools.product([-1, 0, 1], repeat=4):
    use = [0, 0]; v = 1.
    for j, i in enumerate(asg):
        if i >= 0: use[i] += cost[j]; v *= math.exp(eta[i, j])
    if use[0] <= cap[0] and use[1] <= cap[1]:
        Z += v
        for j, i in enumerate(asg):
            if i >= 0: mu[i, j] += v
out['assignment_mean'] = float(np.abs(S.assign_mean(cost, cap, eta) - mu / Z).max())

# scheduling mean map against enumeration of the MIP's feasible (a,x)
env = S.upstream_env('scheduling', 0, n=6, b=2); task = S.Scheduling(env); ea = rng.standard_normal(task.n); ex = {e: rng.standard_normal() for e in task.edges}; marg = np.zeros(task.n); Z = 0.
pairs = {j: list(itertools.combinations([e for e in task.edges if e[1] == j], 2)) for j in range(task.n)}
def rec(j, used, c, w):
    global Z
    if j == task.n:
        Z += w[0]
        for k in w[1]: marg[k] += w[0]
        return
    rec(j + 1, used, c, w)
    if c == task.B: return
    for e1, e2 in pairs[j]:
        b1, b2 = (e1[0], e1[2]), (e2[0], e2[2])
        if b1 == b2 or b1 in used or b2 in used: continue
        rec(j + 1, used | {b1, b2}, c + 1, (w[0] * math.exp(ea[j] + ex[e1] + ex[e2]), w[1] + [j]))
rec(0, frozenset(), 0, (1., []))
out['scheduling_mean'] = float(np.abs(S.sched_mean(task, ea, ex) - marg / Z).max()); out['scheduling_edges'] = len(task.edges)

# rollout step against the upstream expected reward
for name in ('scheduling', 'routing', 'assignment', 'routing_published'):
    t = S.make_task(name, 0); a = t.random_action(rng); s = t.init.copy(); t.env.rmab.set_state(s.astype(float)) if hasattr(t.env.rmab, 'set_state') else None
    want = float(t.env.calc_action_expected_value(a)); got = np.mean([t.reward(np.clip(s + 2 * (rng.random(t.n) < t.p_up(a)).astype(int) - 1, 0, t.ns - 1)) for _ in range(20000)])
    out[f'{name}_reward_vs_upstream'] = dict(upstream=want, mc=float(got))

# R2 pipeline on Q with known structure
n = 40; X = np.array([(rng.permutation(n) < 10).astype(float) for _ in range(150)]); w = rng.standard_normal(n); add = X @ w
out['cv_r2_additive'] = S.cv_r2(X, add, rng); out['cv_r2_additive_noise1'] = S.cv_r2(X, add + add.std() * rng.standard_normal(150) * .5, rng)
# cross-half R2 with known nonlinear share and independent noise
pair = X[:, 0] * X[:, 1] * 6 + X[:, 2] * X[:, 3] * 6 + X[:, 4] * X[:, 5] * 6; X2 = np.array([(rng.permutation(n) < 20).astype(float) for _ in range(150)]); nl = 6 * (X2[:, 0] * X2[:, 1] + X2[:, 2] * X2[:, 3] + X2[:, 4] * X2[:, 5] + X2[:, 6] * X2[:, 7]); q = X2 @ w + nl
ref = S.cv_r2(X2, q, rng); est = [S.cross_r2(X2, q + 2 * rng.standard_normal(150), q + 2 * rng.standard_normal(150), rng)[0] for _ in range(20)]
out['cross_r2'] = dict(noiseless_cv=ref, est_mean=float(np.mean(est)), est_sd=float(np.std(est)))
print(json.dumps(out, indent=1))
