"""Follow-up to probe.py. (a) Style frontier by weight sweep, since weight 2 collapsed two axes in probe.py.
(b) Equal-budget comparison when the objective has competing district bonuses: is the chain needed here?
Known myopic objective, exact mean map as denoiser, uniform walks as base. No learning and no RL result."""
import json, math, sys
import numpy as np
from routing_core import Routes
rng = np.random.default_rng(20261110); RR = Routes(T=20); out = {}
sc = np.array(json.load(open('probe.json'))['feature_scale'])
def draw_value():
    level = rng.integers(0, 4, RR.n); table = np.array([.1, .2, .5, 3.]) * rng.uniform(.5, 1.5, RR.n)[:, None]
    return table[np.arange(RR.n), level]
SIGN = np.array([1, -1, -1, -1])
def lin_of(w): return -w[1] / sc[1] * RR.minutes - w[2] / sc[2] * RR.transfer - w[3] / sc[3] * RR.crowd[RR.node] * (RR.minutes > 0)
def best_route(value, w, rounds=12, pool=256):
    lin = lin_of(w); best, bv = None, -np.inf; prize = value.copy()
    for r in range(rounds):
        e = np.repeat((lin + w[0] / sc[0] * prize[RR.node] / 2)[None], RR.T, 0)[None]
        cand = np.vstack([RR.solve(e), RR.sample(np.repeat(e, pool, 0) * 3, rng)])
        v = RR.features(cand, value) @ (SIGN * w / sc); i = v.argmax()
        if v[i] > bv: bv, best = v[i], cand[i:i + 1]
        seen = np.unique(RR.node[best]); prize = value.copy(); prize[seen] *= rng.uniform(0, 1, len(seen))
    return best
states = [draw_value() for _ in range(12)]; out['frontier'] = {}
for axis, nm in ((1, 'minutes'), (2, 'transfers'), (3, 'crowding')):
    rows = []
    for wc in (0, .05, .1, .2, .4, .8, 1.6):
        w = np.array([1., 0, 0, 0]); w[axis] = wc
        f = np.mean([RR.features(best_route(v, w), v)[0] for v in states], 0); rows.append(dict(weight=wc, service=f[0], cost=f[axis]))
    out['frontier'][nm] = rows; print(nm, [(r['weight'], round(r['service'], 1), round(r['cost'], 1)) for r in rows], flush=True)
# districts: 4 disjoint groups of 4 connected stations, far from each other where possible
adj = {i: set() for i in range(RR.n)}
for t, n in zip(RR.tail, RR.node):
    u = RR.states[t][0]
    if u != n: adj[u].add(n); adj[n].add(u)
def districts():
    used = {RR.src}; out_ = []
    for _ in range(200):
        s = int(rng.integers(RR.n))
        if s in used: continue
        g = [s]
        while len(g) < 4:
            c = sorted({x for y in g for x in adj[y]} - used - set(g))
            if not c: break
            g.append(int(rng.choice(c)))
        if len(g) == 4:
            out_.append(g); used |= set(g)
        if len(out_) == 4: break
    return out_
def seen_of(p):
    s = np.zeros((p.shape[0], RR.n)); np.put_along_axis(s, RR.node[p], 1, 1); return s
def gate(beta, K=12, reps=64):
    res = {}
    for inst in range(6):
        D = districts(); bonus = rng.uniform(.8, 1.2, len(D)); base = .03 * rng.uniform(.5, 1.5, RR.n)
        tmin = .02 * RR.minutes
        def done(p): s = seen_of(p); return np.stack([s[:, g].all(1) for g in D], 1)
        def Qf(p): return done(p) @ bonus + seen_of(p) @ base - tmin[p].sum(1)
        q0 = Qf(RR.sample(np.zeros((2048, 20, RR.m)), rng)); m0, s0 = q0.mean(), q0.std(); lam = s0 / beta
        def grad(mu):
            b = mu.shape[0]; occ = np.zeros((b, 20, RR.n))
            for t in range(20): np.add.at(occ[:, t], (slice(None), RR.node), mu[:, t])
            occ = np.clip(occ, 0, 1 - 1e-9); lm = np.log1p(-occ); tot = lm.sum(1, keepdims=True)
            pv = 1 - np.exp(tot[:, 0]); dp = np.exp(tot - lm)          # P(visit j), d P/d occ_jt
            val = np.tile(base, (b, 1))
            for g, c in zip(D, bonus):
                for j in g:
                    val[:, j] += c * np.prod(pv[:, [x for x in g if x != j]], axis=1)
            return ((val[:, None, :] * dp)[:, :, RR.node] - tmin[None, None]) / lam
        def rec(n, p):
            res.setdefault(n, []).append(float((Qf(p).mean() - m0) / s0))
            d = done(p); res.setdefault(n + '_rate', []).append(float(d.any(1).mean())); res.setdefault(n + '_modes', []).append(float(d.any(0).sum()))
        def pick(A, lw):
            lw = lw.reshape(reps, K); ww = np.exp(lw - lw.max(1, keepdims=True)); ww /= ww.sum(1, keepdims=True)
            ch = (ww.cumsum(1) > rng.random((reps, 1))).argmax(1); return A.reshape(reps, K, 20)[np.arange(reps), ch]
        g0 = grad(RR.marginals(np.zeros((1, 20, RR.m)))[0])
        A = RR.sample(np.zeros((K * reps, 20, RR.m)), rng); rec('snis12', pick(A, Qf(A) / lam))
        A = RR.sample(np.repeat(g0, K * reps, 0), rng); rec('shift', A[:reps]); rec('shift_res12', pick(A, Qf(A) / lam - (RR.bits(A) * g0).sum((1, 2))))
        A = RR.sample(np.repeat(g0, 4096, 0), rng); lw = Qf(A) / lam - (RR.bits(A) * g0).sum((1, 2)); ww = np.exp(lw - lw.max()); ww /= ww.sum()
        d = done(A); res.setdefault('ref', []).append(float((ww @ Qf(A) - m0) / s0)); res.setdefault('ref_ess', []).append(float(1 / (ww ** 2).sum()))
        res.setdefault('ref_rate', []).append(float(ww @ d.any(1))); res.setdefault('ref_modes', []).append(float(((ww @ d) > .05).sum()))
        sh = g0.copy()
        for _ in range(16): sh = .5 * sh + .5 * grad(RR.marginals(sh)[0])
        rec('loglinear_fp', RR.sample(np.repeat(sh, reps, 0), rng))
        for name, n, T_sl in (('sde16', 16, 16.), ('sde32_T32', 32, 32.)):
            ts = T_sl * (np.arange(n + 1) / n) ** 2; y = np.zeros((reps, 20, RR.m)); sh = np.repeat(g0, reps, 0)
            for i in range(n):
                h = ts[i + 1] - ts[i]; sh = grad(RR.marginals(2 * y + sh)[0])
                y = y + h * (2 * RR.marginals(2 * y + sh)[0] - 1) + math.sqrt(h) * rng.normal(size=y.shape)
            rec(name, RR.solve(y + sh / 2))
    return {k: float(np.mean(x)) for k, x in res.items()}
out['district'] = {}
for beta in (2, 5, 10):
    out['district'][beta] = gate(beta); print(beta, {k: round(x, 2) for k, x in out['district'][beta].items()}, flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
