"""Enumeration checks of sched_core on a tiny instance."""
import itertools, numpy as np
from sched_core import *
inst = Instance(3, 8, 3, 2); rng = np.random.default_rng(0)
print('edges', inst.E, 'W', inst.W, 'cfg', inst.n_cfg.tolist(), 'groups', inst.groups)
A = [np.array(c) for c in itertools.product(*[np.where(inst.valid[j])[0] for j in range(inst.n)])]; A = np.array(A); ok = inst.is_feasible(A); A = A[ok]; print('feasible', len(A))
w = np.array([1, 1, 1, 1, 2.]); sc = np.ones(5); dp = ExactDP(inst)
# 1. exact linear: log Z, marginals, entropy
g_a = rng.normal(size=inst.n); g_x = rng.normal(size=inst.E); pot = inst.linear_potential(g_a, g_x)
s = pot[inst.arange[None], A].sum(1); lz = logsumexp(s); p = np.exp(s - lz); mu, lz2, H = dp.linear_marginals(pot)
mu_b = np.zeros_like(pot); np.add.at(mu_b, (np.broadcast_to(inst.arange, A.shape), A), p[:, None])
print('logZ err', abs(lz - lz2), 'H err', abs(H + (p * np.log(p)).sum()), 'mu err', np.abs(mu - mu_b).max())
assert abs(lz - lz2) < 1e-9 and np.abs(mu - mu_b).max() < 1e-9
# 2. exact full Q: the group flag and load terminal
Q = q_value(inst, A, w, sc); lz = logsumexp(Q); lin = inst.linear_potential(w[0] * inst.u, np.zeros(inst.E))
tbl = w[0] * inst.f_service + w[1] * inst.f_punct + w[3] * inst.f_pref; V = dp.backward(tbl, w[2] * dp.bal, w[4]); print('full Q logZ err', abs(lz - dp.log_z(V))); assert abs(lz - dp.log_z(V)) < 1e-9
smp = dp.sample(V, tbl, 200000, rng, w[4]); assert inst.is_feasible(smp[:2000]).all(); q2 = q_value(inst, smp, w, sc); ex = (np.exp(Q - lz) * Q).sum(); print('exact sample mean Q', q2.mean(), 'vs', ex, 'se', q2.std() / 447); assert abs(q2.mean() - ex) < 5 * q2.std() / 447
# 3. Metropolis reaches the same target and rho
S = Sampler(inst, rng); r = S.rho(20000, 100); assert inst.is_feasible(r[:2000]).all(); print('rho mean Q', q_value(inst, r, w, sc).mean(), 'vs', Q.mean())
c, acc, ev = S.run(r[:8000], 2000, log_target=lambda x: q_value(inst, x, w, sc)); print('target mean Q', q_value(inst, c, w, sc).mean(), 'vs', ex, 'acc', acc, 'evals', ev.mean()); assert abs(q_value(inst, c, w, sc).mean() - ex) < .1
c, _, _ = S.run(r[:8000], 2000, linear=pot); e = pot[inst.arange[None], c].sum(1).mean(); print('linear target', e, 'vs', (p * s).sum()); assert abs(e - (p * s).sum()) < .1
# 4. MILP equals brute force
sol = Solver(inst)
for _ in range(20):
    g_a = rng.normal(size=inst.n); g_x = rng.normal(size=inst.E); cfg, val = sol.solve(g_a, g_x); bf = inst.linear_potential(g_a, g_x)[inst.arange[None], A].sum(1).max(); assert abs(val - bf) < 1e-6 and inst.is_feasible(cfg[None]).all()
print('milp ok')
# 5. frontier DP counts, entropy bound holds on the exact family
fr = frontier_dp(inst); print('frontier count', np.exp(fr['log_count']), 'vs', len(A), fr['widths'][:6]); assert abs(np.exp(fr['log_count']) - len(A)) < 1e-6
g_a = rng.normal(size=inst.n); pot = inst.linear_potential(g_a, np.zeros(inst.E)); V = dp.backward(pot); smp = dp.sample(V, pot, 4096, rng); print('H', dp.linear_marginals(pot)[2], 'bound', cross_entropy_bound(inst, smp, dp.order))
e, a = approx_mean(inst, pot); print('approx mean err', np.abs(e - dp.edge_marginals(dp.linear_marginals(pot)[0])).mean())
mf, lf, hf = frontier_mean(inst, pot, frontier_order(inst)); md, ld, hd = dp.linear_marginals(pot); print('frontier mean err', np.abs(mf - md).max(), abs(lf - ld), abs(hf - hd)); assert np.abs(mf - md).max() < 1e-9 and abs(lf - ld) < 1e-9
S5 = Instance(100, **SCALES['S']); d5 = ExactDP(S5); p5 = S5.linear_potential(rng.normal(size=S5.n), rng.normal(size=S5.E)); mf, lf, _ = frontier_mean(S5, p5, frontier_order(S5)); md, ld, _ = d5.linear_marginals(p5); print('S frontier err', np.abs(mf - md).max(), abs(lf - ld)); assert np.abs(mf - md).max() < 1e-8
print('ok')
