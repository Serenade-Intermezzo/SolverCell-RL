"""Style-Scheduling: joint (a,x) action set of the released scheduling MIP, exact DP on small scale, Metropolis, HiGHS and the approximate mean map."""
import itertools, time
import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.special import logsumexp

FEATURES = ('service', 'punctual', 'balance', 'pref', 'group')
SCALES = dict(S=dict(n=20, B=5, group=2), L=dict(n=40, B=10, group=4))
N_SLOTS = 5


class Instance:
    """Availability follows environment/scheduling.py lines 58-84; a chosen patient holds exactly two worker-slot edges."""
    def __init__(self, seed, n, B, group, S=N_SLOTS):
        rng = np.random.default_rng(seed); self.n, self.B, self.S = n, B, S
        arm_p = np.sort(rng.random(S)); worker_p = np.sort(rng.random(S))[::-1]
        arm_p, worker_p = arm_p / arm_p.sum(), worker_p / worker_p.sum()
        self.arm_avail = [rng.choice(S, 2, replace=False, p=arm_p).tolist() for _ in range(n)]
        self.worker_avail = [rng.choice(S, 3, replace=False, p=worker_p).tolist() for _ in range(B)]
        self.ws = {(i, s): k for k, (i, s) in enumerate((i, s) for i in range(B) for s in self.worker_avail[i])}; self.W = len(self.ws)
        self.ws_worker = np.array([i for (i, s) in self.ws])
        self.u = rng.uniform(.5, 1.5, n); self.p = rng.uniform(0, 1, (n, S))
        edges = []                       # (arm, worker, slot, ws)
        for j in range(n):
            for s in self.arm_avail[j]:
                for i in range(B):
                    if (i, s) in self.ws: edges.append((j, i, s, self.ws[i, s]))
        self.edges = np.array(edges, dtype=int).reshape(-1, 4); self.E = len(edges)
        arm_edges = [np.where(self.edges[:, 0] == j)[0].tolist() for j in range(n)]
        servable = [j for j in range(n) if len(arm_edges[j]) >= 2]
        members = rng.permutation(servable)[:4 * group]; self.groups = [sorted(int(x) for x in members[k * group:(k + 1) * group]) for k in range(len(members) // group)]
        self.bonus = rng.uniform(.8, 1.2, len(self.groups))
        # configurations per arm: index 0 is 'not selected'
        self.cfg_edges = [[()] + list(itertools.combinations(arm_edges[j], 2)) for j in range(n)]
        self.n_cfg = np.array([len(c) for c in self.cfg_edges]); C = self.n_cfg.max(); self.C = C
        self.cfg_e = -np.ones((n, C, 2), dtype=int); self.cfg_ws = np.full((n, C, 2), self.W, dtype=int)    # W is an always-free dummy column
        self.sel = np.zeros((n, C), dtype=np.int8); self.valid = np.zeros((n, C), dtype=bool)
        self.f_service = np.zeros((n, C)); self.f_punct = np.zeros((n, C)); self.f_pref = np.zeros((n, C)); self.load = np.zeros((n, C, B), dtype=np.int8)
        self.cfg_mask = np.zeros((n, C), dtype=np.int64)
        for j, cs in enumerate(self.cfg_edges):
            self.valid[j, :len(cs)] = True
            for c, es in enumerate(cs):
                if not es: continue
                e0, e1 = es; self.cfg_e[j, c] = es; self.cfg_ws[j, c] = self.edges[[e0, e1], 3]; self.sel[j, c] = 1
                if self.edges[e0, 3] == self.edges[e1, 3]: self.valid[j, c] = False; continue   # one worker-slot cannot host two edges
                self.f_service[j, c] = self.u[j]; self.cfg_mask[j, c] = (1 << int(self.edges[e0, 3])) | (1 << int(self.edges[e1, 3]))
                for e in es:
                    _, i, s, _ = self.edges[e]; self.f_punct[j, c] -= s / (S - 1); self.f_pref[j, c] += self.p[j, s]; self.load[j, c, i] += 1
        self.arange = np.arange(n)

    def features(self, cfg):
        """(N,5) raw style features for configurations cfg (N,n)."""
        ix = self.arange[None]; sel = self.sel[ix, cfg].astype(bool)
        load = self.load[ix, cfg].sum(1, dtype=np.float64); dev = load - load.mean(1, keepdims=True)
        grp = np.zeros(cfg.shape[0])
        for g, c in zip(self.groups, self.bonus): grp += c * sel[:, g].all(1)
        return np.stack([self.f_service[ix, cfg].sum(1), self.f_punct[ix, cfg].sum(1), -(dev ** 2).sum(1), self.f_pref[ix, cfg].sum(1), grp], 1)

    def is_feasible(self, cfg):
        ws = self.cfg_ws[self.arange[None], cfg].reshape(cfg.shape[0], -1); ok = self.sel[self.arange[None], cfg].sum(1) <= self.B
        ok &= self.valid[self.arange[None], cfg].all(1)
        for k, row in enumerate(ws):
            r = row[row < self.W]; ok[k] &= len(np.unique(r)) == len(r)
        return ok

    def linear_potential(self, g_a, g_x):
        """Configuration-level table for a potential linear in (a,x)."""
        pot = self.sel * g_a[:, None]; has = self.cfg_e[:, :, 0] >= 0
        pot = pot + has * (g_x[np.maximum(self.cfg_e[:, :, 0], 0)] + g_x[np.maximum(self.cfg_e[:, :, 1], 0)])
        return pot

    def gradient(self, mu_a, mu_w_load, w, scales):
        """Mean-field gradient of Q_w at (mu_a, expected worker load); Q itself is w.(f/scale)."""
        g_a = w[0] * self.u / scales[0]; i, s = self.edges[:, 1], self.edges[:, 2]; j = self.edges[:, 0]
        for grp, c in zip(self.groups, self.bonus):
            for m in grp: g_a[m] += w[4] / scales[4] * c * np.prod([mu_a[o] for o in grp if o != m])
        dev = mu_w_load - mu_w_load.mean()
        g_x = -w[1] / scales[1] * s / (self.S - 1) + w[3] / scales[3] * self.p[j, s] - w[2] / scales[2] * 2 * dev[i]
        return g_a, g_x


def q_value(inst, cfg, w, scales):
    return inst.features(cfg) / scales @ w


# ------------------------------------------------------------- Metropolis
class Sampler:
    """Single-arm reconfiguration: the feasible choices of one arm given the rest are the same before and after, so the proposal is symmetric."""
    def __init__(self, inst, rng):
        self.i, self.rng = inst, rng

    def _used(self, cfg):
        i = self.i; used = np.zeros((cfg.shape[0], i.W + 1), dtype=bool)
        np.put_along_axis(used, i.cfg_ws[i.arange[None], cfg].reshape(cfg.shape[0], -1), True, 1); used[:, i.W] = False
        return used

    def run(self, cfg, steps, log_target=None, linear=None):
        """log_target(cfg)->(N,) full evaluation; linear: table (n,C) for a linear target. Returns cfg, acceptance rate, Q evaluations per chain."""
        i, rng = self.i, self.rng; cfg = cfg.copy(); N = cfg.shape[0]; ix = np.arange(N); used = self._used(cfg); count = i.sel[i.arange[None], cfg].sum(1)
        lp = log_target(cfg) if log_target is not None else None; acc = 0; evals = np.zeros(N, dtype=int)
        for _ in range(steps):
            j = rng.integers(i.n, size=N); cur = cfg[ix, j]; cw = i.cfg_ws[j, cur]
            used[ix, cw[:, 0]] = False; used[ix, cw[:, 1]] = False; used[:, i.W] = False
            ws = i.cfg_ws[j]; feas = i.valid[j] & ~used[ix[:, None], ws[:, :, 0]] & ~used[ix[:, None], ws[:, :, 1]]
            full = count - i.sel[j, cur] >= i.B; feas[full, 1:] = False; feas[ix, cur] = False
            has = feas.any(1); new = (rng.random(feas.shape) * feas).argmax(1); new = np.where(has, new, cur)
            if log_target is None and linear is None:
                accept = has
            elif linear is not None:
                accept = has & (np.log(rng.random(N)) < linear[j, new] - linear[j, cur])
            else:
                prop = cfg.copy(); prop[ix, j] = new; lq = log_target(prop); accept = has & (np.log(rng.random(N)) < lq - lp); lp = np.where(accept, lq, lp); evals += has
            final = np.where(accept, new, cur); acc += accept.sum()
            count = count - i.sel[j, cur] + i.sel[j, final]; cfg[ix, j] = final; fw = i.cfg_ws[j, final]
            used[ix, fw[:, 0]] = True; used[ix, fw[:, 1]] = True; used[:, i.W] = False
        return cfg, acc / max(1, N * steps), evals

    def rho(self, n, sweeps=200):
        return self.run(np.zeros((n, self.i.n), dtype=np.int64), sweeps * self.i.n)[0]


# ------------------------------------------------------------- HiGHS
class Solver:
    """Max of a potential linear in (x,a) over the MIP of approximator/scheduling_approximator.py lines 78-110."""
    def __init__(self, inst):
        self.i = inst; E, n = inst.E, inst.n; rows, cols, vals = [], [], []; lo, hi = [], []; r = 0
        for j in range(n):                                     # exactly two edges when selected, none otherwise
            for e in np.where(inst.edges[:, 0] == j)[0]: rows.append(r); cols.append(e); vals.append(1.)
            rows.append(r); cols.append(E + j); vals.append(-2.); lo.append(0); hi.append(0); r += 1
        for k in range(inst.W):                                # each worker-slot at most once
            for e in np.where(inst.edges[:, 3] == k)[0]: rows.append(r); cols.append(e); vals.append(1.)
            lo.append(0); hi.append(1); r += 1
        for j in range(n): rows.append(r); cols.append(E + j); vals.append(1.)
        lo.append(0); hi.append(inst.B); r += 1
        mat = sparse.csc_matrix((vals, (rows, cols)), shape=(r, E + n)); mat.indices = mat.indices.astype(np.int32); mat.indptr = mat.indptr.astype(np.int32)
        self.con = LinearConstraint(mat, lo, hi); self.nv = E + n
        self.cfg_of = [{frozenset(es): c for c, es in enumerate(cs) if inst.valid[j, c]} for j, cs in enumerate(inst.cfg_edges)]

    def solve(self, g_a, g_x):
        i = self.i; res = milp(-np.concatenate([g_x, g_a]), constraints=self.con, integrality=np.ones(self.nv), bounds=Bounds(0, 1), options=dict(mip_rel_gap=0.))
        assert res.status == 0, res.message
        on = np.where(res.x[:i.E] > .5)[0]; cfg = np.zeros(i.n, dtype=np.int64); by = {}
        for e in on: by.setdefault(int(i.edges[e, 0]), []).append(int(e))
        for j, es in by.items(): cfg[j] = self.cfg_of[j][frozenset(es)]
        return cfg, -res.fun


def perturbed_map(inst, solver, g_a, g_x, n, rng):
    out = np.zeros((n, inst.n), dtype=np.int64)
    for k in range(n):
        out[k] = solver.solve(g_a + rng.gumbel(size=inst.n), g_x + rng.gumbel(size=inst.E))[0]
    return out


# ------------------------------------------------------------- Exact DP on the small scale (dense worker-slot masks)
class ExactDP:
    """Exact sums over all feasible (a,x); the order puts group members next so one flag tracks group completion, worker load is a function of the mask."""
    def __init__(self, inst):
        assert inst.W <= 20
        self.i = inst; self.M = 1 << inst.W; self.masks = np.arange(self.M); self.pop = np.array([bin(m).count('1') for m in range(self.M)], dtype=np.int16)
        order = [m for g in inst.groups for m in g]; order += [j for j in range(inst.n) if j not in order]; self.order = order
        self.first, self.last = {}, {}
        for g, c in zip(inst.groups, inst.bonus): self.first[g[0]] = True; self.last[g[-1]] = c
        self.in_group = {m for g in inst.groups for m in g}
        loads = np.zeros((self.M, inst.B)); 
        for k in range(inst.W): loads[:, inst.ws_worker[k]] += (self.masks >> k) & 1
        self.bal = -((loads - loads.mean(1, keepdims=True)) ** 2).sum(1)

    def backward(self, pot, terminal=None, w_group=0.):
        """V[pos][mask, flag]: log sum over the remaining arms. pot is the configuration table; w_group multiplies the raw group bonus."""
        i = self.i; V = [None] * (i.n + 1); t = np.zeros(self.M) if terminal is None else terminal.copy(); t[self.pop > 2 * i.B] = -np.inf
        V[i.n] = np.stack([t, t], 1)
        for pos in range(i.n - 1, -1, -1):
            j = self.order[pos]; nxt = V[pos + 1]; terms = [[], []]
            for c in np.where(i.valid[j])[0]:
                mc = int(i.cfg_mask[j, c]); s = int(i.sel[j, c]); bad = (self.masks & mc) != 0
                to = self.masks | mc
                for flag in (0, 1):
                    f = (flag or j in self.first) and s if j in self.in_group else 0
                    nf = 1 if f else 0; v = pot[j, c] + nxt[to, nf]
                    if nf and j in self.last: v = v + w_group * self.last[j]
                    terms[flag].append(np.where(bad, -np.inf, v))
            V[pos] = np.stack([logsumexp(np.stack(terms[0]), 0), logsumexp(np.stack(terms[1]), 0)], 1)
        return V

    def log_z(self, V):
        return float(V[0][0, 0])

    def sample(self, V, pot, n, rng, w_group=0.):
        i = self.i; mask = np.zeros(n, dtype=np.int64); flag = np.zeros(n, dtype=np.int64); cfg = np.zeros((n, i.n), dtype=np.int64); ix = np.arange(n)
        for pos in range(i.n):
            j = self.order[pos]; cs = np.where(i.valid[j])[0]; mc = i.cfg_mask[j, cs]; s = i.sel[j, cs].astype(bool)
            first, ing, last = j in self.first, j in self.in_group, self.last.get(j)
            to = mask[:, None] | mc; ok = (mask[:, None] & mc) == 0
            nf = (((flag[:, None] == 1) | first) & s & ing).astype(np.int64)
            v = pot[j, cs][None] + V[pos + 1][to, nf]
            if last is not None: v = v + w_group * last * nf
            v = np.where(ok, v, -np.inf); pick = (v + rng.gumbel(size=v.shape)).argmax(1)
            cfg[:, j] = cs[pick]; mask = to[ix, pick]; flag = nf[ix, pick]
        return cfg

    def linear_marginals(self, pot):
        """Exact configuration marginals, log Z and Shannon entropy for a potential linear in (a,x)."""
        i = self.i; V = self.backward(pot); F = np.full(self.M, -np.inf); F[0] = 0.; lz = self.log_z(V); mu = np.zeros_like(pot)
        for pos in range(i.n):
            j = self.order[pos]; nxt = V[pos + 1][:, 0]; new = []
            for c in np.where(i.valid[j])[0]:
                mc = int(i.cfg_mask[j, c]); bad = (self.masks & mc) != 0; to = self.masks | mc
                v = np.where(bad, -np.inf, F + pot[j, c]); mu[j, c] = np.exp(logsumexp(v + nxt[to]) - lz)
                add = np.full(self.M, -np.inf); good = ~bad & np.isfinite(v); add[to[good]] = v[good]; new.append(add)
            F = logsumexp(np.stack(new), 0)
        return mu, lz, lz - float((mu * pot).sum())

    def edge_marginals(self, mu):
        i = self.i; out = np.zeros(i.E); on = i.cfg_e[:, :, 0] >= 0
        np.add.at(out, i.cfg_e[:, :, 0][on], mu[on]); np.add.at(out, i.cfg_e[:, :, 1][on], mu[on])
        return out


# ------------------------------------------------------------- Approximate mean map
def approx_mean(inst, pot, iters=20, damp=.5):
    """Dual iterations on worker-slot capacities and the budget; each arm is summed exactly over its configurations. Returns edge and arm marginals."""
    u = np.zeros(inst.W + 1); v = 0.; ws = inst.cfg_ws; sel = inst.sel; has = inst.cfg_e[:, :, 0] >= 0
    for it in range(iters + 1):
        lp = pot - u[ws[:, :, 0]] - u[ws[:, :, 1]] - v * sel; lp = np.where(inst.valid, lp, -np.inf)
        p = np.exp(lp - logsumexp(lp, 1, keepdims=True))
        ew = np.zeros(inst.W + 1); np.add.at(ew, ws[:, :, 0][has], p[has]); np.add.at(ew, ws[:, :, 1][has], p[has]); ew[inst.W] = 0
        if it == iters: break
        u[:inst.W] = np.maximum(0, u[:inst.W] + damp * np.log(np.maximum(ew[:inst.W], 1e-12)))
        v = max(0., v + damp * np.log(max((p * sel).sum(), 1e-12) / inst.B))
    e = np.zeros(inst.E); np.add.at(e, inst.cfg_e[:, :, 0][has], p[has]); np.add.at(e, inst.cfg_e[:, :, 1][has], p[has])
    return e, (p * sel).sum(1)


# ------------------------------------------------------------- Entropy upper bound
def cross_entropy_bound(inst, cfg, order=None, steps=300, lr=.1):
    """Held-out cross-entropy of an autoregressive softmax over feasible configurations, two folds. Valid upper bound on Shannon entropy in expectation."""
    import torch
    order = list(range(inst.n)) if order is None else order; n = cfg.shape[0]; h = n // 2; folds = [(np.arange(h), np.arange(h, n)), (np.arange(h, n), np.arange(h))]
    W = inst.W; total = 0.; mask_b = (1 << np.arange(W)).astype(np.int64)
    masks = np.zeros((n, inst.n), dtype=np.int64); cur = np.zeros(n, dtype=np.int64)
    for pos, j in enumerate(order):
        masks[:, pos] = cur; cur = cur | inst.cfg_mask[j, cfg[:, j]]
    for pos, j in enumerate(order):
        cs = np.where(inst.valid[j])[0]; lookup = -np.ones(inst.C, dtype=int); lookup[cs] = np.arange(len(cs))
        bits = (masks[:, pos, None] & mask_b) != 0; cnt = bits.sum(1, keepdims=True) / 2.
        feat = torch.tensor(np.concatenate([bits, cnt / inst.B], 1), dtype=torch.float32)
        conflict = torch.tensor(((masks[:, pos, None] & inst.cfg_mask[j, cs]) != 0) | ((cnt >= inst.B) & (inst.sel[j, cs] == 1)))
        y = torch.tensor(lookup[cfg[:, j]])
        for tr, te in folds:
            th = torch.zeros(len(cs), requires_grad=True); A = torch.zeros(len(cs), W + 1, requires_grad=True); opt = torch.optim.Adam([th, A], lr=lr)
            def nll(idx):
                lg = (th + feat[idx] @ A.T).masked_fill(conflict[idx], -1e9)
                return torch.nn.functional.cross_entropy(lg, y[idx])
            tr_t, te_t = torch.as_tensor(tr), torch.as_tensor(te)
            for _ in range(steps):
                opt.zero_grad(); (nll(tr_t) + 1e-4 * (A ** 2).sum()).backward(); opt.step()
            with torch.no_grad(): total += float(nll(te_t)) / 2
    return total


def collision_entropy(cfg):
    n = cfg.shape[0]; _, counts = np.unique(cfg, axis=0, return_counts=True); coll = float((counts * (counts - 1) / 2).sum()); pairs = n * (n - 1) / 2
    return (-np.log(coll / pairs) if coll else None), np.log(pairs / 3), counts.shape[0] / n, counts.max() / n


# ------------------------------------------------------------- Frontier DP used only to measure hardness
def frontier_dp(inst, time_cap=600., mem_cap=64e9):
    """Forward log Z for a zero potential with forgetting; greedily picks the next arm that keeps the frontier smallest. Returns timing and the state peak."""
    i = inst; arms = [j for j in range(i.n) if i.n_cfg[j] > 1]; arm_ws = {j: set(int(k) for c in np.where(i.valid[j])[0][1:] for k in i.cfg_ws[j, c]) for j in arms}
    remaining = set(arms); order = []; front = set(); widths = []
    while remaining:
        best = min(remaining, key=lambda j: (len((front | arm_ws[j]) & set().union(*[arm_ws[o] for o in remaining if o != j] or [set()])), j))
        remaining.discard(best); order.append(best); front = (front | arm_ws[best]) & (set().union(*[arm_ws[o] for o in remaining]) if remaining else set()); widths.append(len(front))
    keys = np.zeros(1, dtype=np.int64); vals = np.zeros(1); CNT = 40; t0 = time.time(); peak = 1; rest = {j: arm_ws[j] for j in order}; stored = 1
    def merge(k, v):
        o = np.argsort(k, kind='stable'); k, v = k[o], v[o]; start = np.concatenate([[0], np.where(np.diff(k))[0] + 1])
        return k[start], _log_reduce(v, start)
    def stop(pos):
        return dict(completed=False, sec=time.time() - t0, peak=peak, arms_done=pos, arms=len(order), widths=widths, max_bytes=max_bytes)
    max_bytes = 0
    for pos, j in enumerate(order):
        later = set().union(*[rest[o] for o in order[pos + 1:]]) if pos + 1 < len(order) else set(); forget = np.int64(sum(1 << b for b in rest[j] if b not in later))
        ks, vs = [], []; size = 0; old = keys.shape[0]
        for c in np.where(i.valid[j])[0]:
            mc = int(i.cfg_mask[j, c]); s = int(i.sel[j, c]); keep = (keys & mc) == 0
            if s: keep &= (keys >> CNT) < i.B
            ks.append(((keys[keep] | mc) + (s << CNT)) & ~forget); vs.append(vals[keep]); size += ks[-1].shape[0]
            max_bytes = max(max_bytes, 16 * (size + old))
            if time.time() - t0 > time_cap or 16 * (size + old) > mem_cap: return stop(pos)
            if size > 3e7:
                k, v = merge(np.concatenate(ks), np.concatenate(vs)); ks, vs = [k], [v]; size = k.shape[0]
        keys, vals = merge(np.concatenate(ks), np.concatenate(vs)); peak = max(peak, keys.shape[0])
    return dict(completed=True, sec=time.time() - t0, peak=peak, arms_done=len(order), arms=len(order), widths=widths, max_bytes=max_bytes, log_count=float(logsumexp(vals)))


def frontier_mean(inst, pot, order, time_cap=600.):
    """Exact configuration marginals, log Z and entropy on the frontier DP for a potential linear in (a,x)."""
    t0 = time.time(); i = inst; CNT = 40; arms = [j for j in range(i.n)]; ws = {j: set(int(k) for c in np.where(i.valid[j])[0][1:] for k in i.cfg_ws[j, c]) for j in arms}
    order = list(order) + [j for j in arms if j not in order]; keys = np.zeros(1, dtype=np.int64); F = np.zeros(1); layers = []
    for pos, j in enumerate(order):
        later = set().union(*[ws[o] for o in order[pos + 1:]]) if pos + 1 < len(order) else set(); forget = np.int64(sum(1 << b for b in ws[j] if b not in later))
        ks, vs, rec = [], [], []
        if time.time() - t0 > time_cap: raise TimeoutError
        for c in np.where(i.valid[j])[0]:
            mc = int(i.cfg_mask[j, c]); s = int(i.sel[j, c]); keep = (keys & mc) == 0
            if s: keep &= (keys >> CNT) < i.B
            idx = np.where(keep)[0]; nk = ((keys[idx] | mc) + (s << CNT)) & ~forget; ks.append(nk); vs.append(F[idx] + pot[j, c]); rec.append((c, idx, nk))
        k, v = np.concatenate(ks), np.concatenate(vs); o = np.argsort(k, kind='stable'); k, v = k[o], v[o]; start = np.concatenate([[0], np.where(np.diff(k))[0] + 1]); nkeys = k[start]; nF = _log_reduce(v, start)
        layers.append((j, F, [(c, a, np.searchsorted(nkeys, b)) for c, a, b in rec])); keys, F = nkeys, nF
    lz = float(logsumexp(F)); B = np.zeros_like(F); mu = np.zeros_like(pot)
    for j, Fprev, rec in reversed(layers):
        if time.time() - t0 > time_cap: raise TimeoutError
        nb = np.full(Fprev.shape[0], -np.inf)
        for c, a, b in rec:
            term = pot[j, c] + B[b]; mu[j, c] = np.exp(logsumexp(Fprev[a] + term) - lz)
            nb[a] = np.logaddexp(nb[a], term)
        B = nb
    return mu, lz, lz - float((mu * pot).sum())


def frontier_order(inst):
    i = inst; arms = [j for j in range(i.n) if i.n_cfg[j] > 1]; ws = {j: set(int(k) for c in np.where(i.valid[j])[0][1:] for k in i.cfg_ws[j, c]) for j in arms}
    remaining = set(arms); order = []; front = set()
    while remaining:
        best = min(remaining, key=lambda j: (len((front | ws[j]) & set().union(*[ws[o] for o in remaining if o != j] or [set()])), j))
        remaining.discard(best); order.append(best); front = (front | ws[best]) & (set().union(*[ws[o] for o in remaining]) if remaining else set())
    return order


def _log_reduce(v, start):
    m = np.maximum.reduceat(v, start); rep = np.repeat(m, np.diff(np.concatenate([start, [v.shape[0]]])))
    return m + np.log(np.add.reduceat(np.exp(v - rep), start))
