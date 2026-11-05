"""Read-only probe of the Style-Routing prototype: solver correctness, scale, style conflict, nonlinearity, Gate-0
comparison with an exact denoiser. Myopic known objective; no learning and no RL result."""
import json, math, sys, time
import numpy as np
from routing_core import Routes

rng = np.random.default_rng(20261109)
out = {}
# 1. DP correctness against brute force on a short horizon
R6 = Routes(T=6); eta = rng.normal(size=(1, 6, R6.m))
paths = []
def dfs(s, t, p):
    if t == 6:
        if R6.start[s]: paths.append(list(p))
        return
    for arc in R6.out[s]:
        dfs(R6.head[arc], t + 1, p + [arc])
for s in np.where(R6.start)[0]: dfs(s, 0, [])
P = np.array(paths); sc = eta[0, np.arange(6)[None], P].sum(1)
mu, lz = R6.marginals(eta); w = np.exp(sc - sc.max()); w /= w.sum()
mu_b = np.zeros((6, R6.m))
for wi, p in zip(w, P): mu_b[np.arange(6), p] += wi
out['correctness'] = dict(paths=len(P), logZ_err=float(abs(lz[0] - np.logaddexp.reduce(sc))), marginal_err=float(np.abs(mu[0] - mu_b).max()),
                          solver_matches_enum=bool(abs(eta[0, np.arange(6), R6.solve(eta)[0]].sum() - sc.max()) < 1e-9))
# 2. scale
RR = Routes(T=20); z0 = np.zeros((1, 20, RR.m))
t0 = time.time(); [RR.solve(rng.normal(size=(1, 20, RR.m))) for _ in range(50)]; ts = (time.time() - t0) / 50
t0 = time.time(); [RR.marginals(rng.normal(size=(1, 20, RR.m))) for _ in range(50)]; tm = (time.time() - t0) / 50
out['scale'] = dict(stations=RR.n, undirected_edges=RR.undirected, line_states=RR.S, arcs_per_step=RR.m, bits=RR.dim,
                    log10_walks=float(RR.log_z(z0)[0] / math.log(10)), solver_ms=ts * 1e3, mean_map_ms=tm * 1e3)
print(out['correctness'], out['scale'], flush=True)

def draw_value():
    # RMAB-flavoured prizes: stations in a high state are worth much more
    level = rng.integers(0, 4, RR.n); table = np.array([.1, .2, .5, 3.]) * rng.uniform(.5, 1.5, RR.n)[:, None]
    return table[np.arange(RR.n), level]

def obj(path, value, w, sc):
    f = RR.features(path, value)
    return (f @ (np.array([1, -1, -1, -1]) * w / sc)), f

def best_route(value, w, sc, rounds=12, pool=256):
    """Heuristic reference: repeated linearised DP plus exact tilted samples; always feasible, not proven optimal."""
    lin = np.array(-w[1] / sc[1] * RR.minutes - w[2] / sc[2] * RR.transfer - w[3] / sc[3] * RR.crowd[RR.node] * (RR.minutes > 0))
    best, bv = None, -np.inf; prize = value.copy()
    for r in range(rounds):
        e = np.repeat((lin + w[0] / sc[0] * prize[RR.node] / 2)[None], RR.T, 0)[None]
        cand = np.vstack([RR.solve(e), RR.sample(np.repeat(e, pool, 0) * 3, rng)])
        v, _ = obj(cand, value, w, sc); i = v.argmax()
        if v[i] > bv: bv, best = v[i], cand[i:i + 1]
        seen = np.unique(RR.node[best]); prize = value.copy(); prize[seen] *= rng.uniform(0, 1, len(seen))
    return best

# 3. style conflict at fixed states, scales frozen from independent calibration states
cal = [RR.features(RR.sample(np.zeros((256, 20, RR.m)), rng), draw_value()) for _ in range(16)]
sc = np.concatenate(cal).std(0) + 1e-9
out['feature_scale'] = sc.tolist()
STYLES = {'service': [1, 0, 0, 0], 'fast': [1, 2, 0, 0], 'smooth': [1, 0, 2, 0], 'comfort': [1, 0, 0, 2], 'mixed': [1, 1, 1, 1]}
rows = {k: [] for k in STYLES}; overlap = []
for st_i in range(16):
    v = draw_value(); routes = {}
    for k, w in STYLES.items():
        routes[k] = best_route(v, np.array(w, float), sc); rows[k].append(RR.features(routes[k], v)[0])
    bits = {k: set(RR.node[routes[k][0]].tolist()) for k in STYLES}
    overlap.append(np.mean([len(bits['service'] & bits[k]) / max(1, len(bits['service'] | bits[k])) for k in ('fast', 'smooth', 'comfort')]))
out['style_table'] = {k: np.mean(r, 0).tolist() for k, r in rows.items()}
out['style_station_jaccard_vs_service'] = float(np.mean(overlap))
print(out['style_table'], out['style_station_jaccard_vs_service'], flush=True)

# 4. nonlinearity of myopic objective in the arc-time bits, heldout ridge R2
def ridge_r2(X, y, n_train):
    Xc = X - X[:n_train].mean(0); yc = y - y[:n_train].mean(); best = -9
    for lam in (1., 10., 100., 1000.):
        K = Xc[:n_train] @ Xc[:n_train].T
        al = np.linalg.solve(K + lam * np.eye(n_train), yc[:n_train])
        pred = Xc[n_train:] @ (Xc[:n_train].T @ al)
        best = max(best, 1 - np.mean((yc[n_train:] - pred) ** 2) / np.var(yc[n_train:]))
    return float(best)
r2u, r2t = [], []
for st_i in range(6):
    v = draw_value(); N = 2400
    p = RR.sample(np.zeros((N, 20, RR.m)), rng); r2u.append(ridge_r2(RR.bits(p).reshape(N, -1), RR.features(p, v)[:, 0], 1800))
    e = np.repeat((v[RR.node] / sc[0])[None], RR.T, 0)[None]
    p = RR.sample(np.repeat(e, N, 0), rng); r2t.append(ridge_r2(RR.bits(p).reshape(N, -1), RR.features(p, v)[:, 0], 1800))
out['service_R2_heldout'] = dict(uniform_walks=r2u, tilted_walks=r2t)
print(out['service_R2_heldout'], flush=True)

# 5. equal-budget Gate 0 on this action set, exact mean map as oracle denoiser, uniform walks as base
def gate(beta, K=12, reps=64, n=16, T_sl=16.):
    res = {}
    for inst in range(6):
        v = draw_value(); w = np.array(STYLES['mixed'], float)
        lin = -w[1] / sc[1] * RR.minutes - w[2] / sc[2] * RR.transfer - w[3] / sc[3] * RR.crowd[RR.node] * (RR.minutes > 0)
        val = v * w[0] / sc[0]
        Qf = lambda p: obj(p, v, w, sc)[0]
        q0 = Qf(RR.sample(np.zeros((1024, 20, RR.m)), rng)); m0, s0 = q0.mean(), q0.std(); lam = s0 / beta
        def grad(mu): return (RR.service_grad(mu, val) + lin[None, None]) / lam
        def rec(name, vals): res.setdefault(name, []).append(float((np.mean(vals) - m0) / s0))
        mu0, _ = RR.marginals(np.zeros((1, 20, RR.m))); g0 = grad(mu0)
        A = RR.sample(np.zeros((K * reps, 20, RR.m)), rng); c = Qf(A).reshape(reps, K)
        ww = np.exp(c / lam - (c / lam).max(1, keepdims=True)); ww /= ww.sum(1, keepdims=True)
        ch = (ww.cumsum(1) > rng.random((reps, 1))).argmax(1)
        rec('snis12', c[np.arange(reps), ch]); rec('best12', c.max(1)); res.setdefault('ess12', []).append(float((1 / (ww ** 2).sum(1)).mean()))
        A = RR.sample(np.repeat(g0, K * reps, 0), rng); rec('shift', Qf(A[:reps]))
        c = Qf(A); lt = (RR.bits(A) * g0).sum((1, 2)); r = (c / lam - lt).reshape(reps, K)
        ww = np.exp(r - r.max(1, keepdims=True)); ww /= ww.sum(1, keepdims=True); ch = (ww.cumsum(1) > rng.random((reps, 1))).argmax(1)
        rec('shift_res12', c.reshape(reps, K)[np.arange(reps), ch]); res.setdefault('ess_res12', []).append(float((1 / (ww ** 2).sum(1)).mean()))
        # large-K reference for the soft target
        A = RR.sample(np.repeat(g0, 8192, 0), rng); c = Qf(A); r = c / lam - (RR.bits(A) * g0).sum((1, 2)); ww = np.exp(r - r.max()); ww /= ww.sum()
        res.setdefault('target_ref', []).append(float((ww @ c - m0) / s0)); res.setdefault('target_ref_ess', []).append(float(1 / (ww ** 2).sum()))
        # log-linear fixed point: no generative chain
        sh = g0.copy()
        for _ in range(16): sh = .5 * sh + .5 * grad(RR.marginals(sh)[0])
        rec('loglinear_fp', Qf(RR.sample(np.repeat(sh, reps, 0), rng)))
        # stochastic localization chain with re-linearisation; solver decodes at the end
        y = np.zeros((reps, 20, RR.m)); h = T_sl / n; sh = np.repeat(g0, reps, 0)
        for _ in range(n):
            sh = grad(RR.marginals(2 * y + sh)[0])
            y = y + h * (2 * RR.marginals(2 * y + sh)[0] - 1) + math.sqrt(h) * rng.normal(size=y.shape)
        rec('sde16', Qf(RR.solve(y + sh / 2)))
    return {k: float(np.mean(x)) for k, x in res.items()}
out['gate'] = {}
for beta in (2, 5, 10):
    out['gate'][beta] = gate(beta); print(beta, {k: round(x, 2) for k, x in out['gate'][beta].items()}, flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
