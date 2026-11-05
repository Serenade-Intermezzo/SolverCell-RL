"""G3 screen of LSFlow's public benchmarks: Q nonlinearity, exact mean map cost, solver cost. Trains nothing. Design: docs/G3_BENCHMARK_SCREEN.md."""
import argparse, contextlib, hashlib, io, json, math, os, platform, random, sys, time, types
from itertools import combinations
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('SOLVERCELL_REPO', HERE.parents[1]))
VENDOR = REPO / 'vendor' / 'lsflow'


def _stub_upstream():
    """Upstream instance builders import Gurobi and plotting at module level but do not call them here."""
    for m in ('gurobipy', 'matplotlib', 'matplotlib.pyplot', 'torch_geometric.transforms'):
        sys.modules.setdefault(m, MagicMock())
    tg = types.ModuleType('torch_geometric'); tg.transforms = sys.modules['torch_geometric.transforms']
    data = types.ModuleType('torch_geometric.data'); data.Data = object; tg.data = data
    sys.modules.setdefault('torch_geometric', tg); sys.modules.setdefault('torch_geometric.data', data)
    try:
        import gymnasium  # noqa: F401
    except ImportError:
        gym = types.ModuleType('gymnasium'); gym.Env = object; sys.modules['gymnasium'] = gym
    if str(VENDOR) not in sys.path: sys.path.insert(0, str(VENDOR))


def upstream_env(kind, seed, n=40, b=10, h=20):
    _stub_upstream(); cwd = os.getcwd(); os.chdir(VENDOR)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            from environment.rmab_instances import get_constrained, get_routing, get_scheduling
            from environment.routing import RoutingRMAB
            RoutingRMAB.plot_graph = lambda *_, **__: None
            random.seed(seed); np.random.seed(seed)
            env = dict(scheduling=get_scheduling, routing=get_routing, assignment=get_constrained)[kind](h, n, b, rmab_type='multistate')[0]
    finally:
        os.chdir(cwd)
    return env


class Rmab:
    """Benchmarks built on upstream MultiStateRMAB: independent arms and additive reward."""
    def __init__(self, env):
        self.env = env; base = env.rmab; self.n = base.n_arms; self.m = self.n; self.B = base.budget; self.H = base.horizon
        self.up = np.asarray(base.transitions)[:, 0].astype(float); self.r = np.asarray(base.state_r, dtype=float); self.ns = self.r.shape[1]; self.init = np.asarray(base.init_state).astype(int)

    def p_up(self, a):
        return self.up[np.arange(self.n), a.astype(int)]

    def reward(self, s):
        return float(self.r[np.arange(self.n), s].sum())

    def coeff(self, s):
        ids = np.arange(self.n)
        return (self.up[:, 1] - self.up[:, 0]) * (self.r[ids, np.minimum(s + 1, self.ns - 1)] - self.r[ids, np.maximum(s - 1, 0)])

    def random_action(self, rng):
        random.seed(int(rng.integers(1 << 31))); np.random.seed(int(rng.integers(1 << 31)))
        return np.asarray(self.env.get_random_action()).astype(float)


class Intervention:
    def __init__(self, seed):
        sys.path.insert(0, str(REPO))
        from solvercell.environment import InterventionEnv
        self.env = InterventionEnv(40, 20, 10, 20, instance_seed=seed); self.n = self.env.n_arms; self.m = self.env.n_actions; self.B = self.env.budget; self.H = self.env.horizon
        self.r = np.asarray(self.env.state_r, dtype=float); self.ns = self.env.n_states; self.init = np.asarray(self.env.wrapper_init_state).astype(int)

    def p_up(self, a):
        return self.env.transition_probabilities(a)

    def reward(self, s):
        return float(self.r[np.arange(self.n), s].sum())

    def coeff(self, s):
        ids = np.arange(self.n); du = self.r[ids, np.minimum(s + 1, self.ns - 1)] - self.r[ids, np.maximum(s - 1, 0)]; p0 = self.p_up(np.zeros(self.m))
        return np.array([((self.p_up(np.eye(self.m)[i]) - p0) * du).sum() for i in range(self.m)])

    def random_action(self, rng):
        return self.env.random_action(rng).astype(float)

    def solve(self, c):
        a = np.zeros(self.m); idx = np.argsort(-c)[:self.B]; a[idx[c[idx] > 0]] = 1
        return a


class Mip:
    """max c.a over the upstream MIP; a occupies the first m variables."""
    def build(self, nv, rows, lo, hi, names=None):
        r, c, v = zip(*rows); self.A = coo_matrix((v, (r, c)), shape=(len(lo), nv)).tocsr(); self.lo, self.hi = np.array(lo, float), np.array(hi, float); self.nv = nv

    def solve_full(self, c, fix=None, tl=2.):
        obj = np.zeros(self.nv); obj[:len(c)] = -np.asarray(c); lb, ub = np.zeros(self.nv), np.ones(self.nv)
        if fix is not None: lb[:len(fix)] = ub[:len(fix)] = fix
        res = milp(obj, integrality=np.ones(self.nv), bounds=Bounds(lb, ub), constraints=LinearConstraint(self.A, self.lo, self.hi), options=dict(time_limit=tl, disp=False))
        self.last = res
        return None if res.x is None else np.round(res.x)

    def solve(self, c):
        x = self.solve_full(c)
        return np.zeros(self.m) if x is None else x[:self.m]


class Scheduling(Rmab, Mip):
    def __init__(self, env):
        Rmab.__init__(self, env); self.nw, self.ts = env.n_workers, env.n_timeslots; self.arm_M, self.worker_M = np.asarray(env.arm_M), np.asarray(env.worker_M)
        self.edges = [(i, j, s) for i in range(self.nw) for j in range(self.n) for s in range(self.ts) if self.arm_M[j, s] == 1 == self.worker_M[i, s]]
        k = {e: self.n + q for q, e in enumerate(self.edges)}; rows, lo, hi = [], [], []
        def add(terms, l, h):
            rows.extend((len(lo), v, c) for v, c in terms); lo.append(l); hi.append(h)
        add([(j, 1.) for j in range(self.n)], -np.inf, self.B)
        for i in range(self.nw):
            for s in range(self.ts):
                t = [(k[e], 1.) for e in self.edges if e[0] == i and e[2] == s]
                if t: add(t, -np.inf, 1)
        for j in range(self.n):
            t = [(k[e], 1.) for e in self.edges if e[1] == j]
            add(t + [(j, 0.)], -np.inf, 2)
            add(t + [(j, -2.)], 0, np.inf)
            for e in self.edges:
                if e[1] == j: add([(j, 1.), (k[e], -1.)], 0, np.inf)
        self.build(self.n + len(self.edges), rows, lo, hi); self.k = k


class Assignment(Rmab, Mip):
    def __init__(self, env, single=False):
        Rmab.__init__(self, env); self.nw = env.n_workers; self.cost, self.cap = np.asarray(env.arm_costs).astype(int), np.asarray(env.worker_capacity).astype(int)
        x = lambda i, j: self.n + i * self.n + j; rows, lo, hi = [], [], []
        def add(terms, l, h):
            rows.extend((len(lo), v, c) for v, c in terms); lo.append(l); hi.append(h)
        for i in range(self.nw): add([(x(i, j), float(self.cost[j])) for j in range(self.n)], -np.inf, self.cap[i])
        for j in range(self.n):
            add([(j, -1.)] + [(x(i, j), 1.) for i in range(self.nw)], 0, np.inf if not single else 0)
            for i in range(self.nw): add([(j, 1.), (x(i, j), -1.)], 0, np.inf)
        self.build(self.n + self.nw * self.n, rows, lo, hi); self.x = x


class Routing(Rmab, Mip):
    def __init__(self, env):
        Rmab.__init__(self, env); self.adj = np.asarray(env.edges) == 1; self.src = env.source; self.T = env.max_path_length; self.mip_built = False

    def _build_mip(self):
        n, T = self.n, self.T; arcs = [(j, k) for j in range(n) for k in range(n) if j == k or self.adj[j, k]]; v = {}
        for q, (j, k) in enumerate(arcs):
            for t in range(T): v[j, k, t] = n + q * T + t
        rows, lo, hi = [], [], []
        def add(terms, l, h):
            rows.extend((len(lo), a, c) for a, c in terms); lo.append(l); hi.append(h)
        add([(j, 1.) for j in range(n)], -np.inf, self.B)
        add([(v[self.src, k, 0], 1.) for k in range(n) if self.adj[self.src, k]], 1, 1)
        add([(v[k, self.src, T - 1], 1.) for k in range(n) if self.adj[k, self.src]], 1, 1)
        edge = [(j, k) for j, k in arcs if self.adj[j, k]]
        for t in range(T): add([(v[j, k, t], 1.) for j, k in edge], 1, 1)
        for j in range(n):
            into = [k for k in range(n) if (k, j, 0) in v]; out = [k for k in range(n) if (j, k, 0) in v]
            for t in range(T - 1): add([(v[k, j, t], 1.) for k in into] + [(v[j, k, t + 1], -1.) for k in out], 0, 0)
            add([(j, -1.)] + [(v[k, j, t], 1.) for k in into for t in range(T)], 0, np.inf)
        self.build(n + len(arcs) * T, rows, lo, hi); self.mip_built = True; self.self_vars = [v[j, j, t] for j in range(n) for t in range(T)]

    def solve_mip(self, c, fix=None):
        if not self.mip_built: self._build_mip()
        return self.solve_full(c, fix=fix, tl=2.)

    def solve_strict(self, c, fix=None, tl=20.):
        """The paper's route constraint: the upstream MIP with self-loop variables disabled."""
        if not self.mip_built: self._build_mip()
        obj = np.zeros(self.nv); obj[:self.n] = -np.asarray(c); lb, ub = np.zeros(self.nv), np.ones(self.nv); ub[self.self_vars] = 0
        if fix is not None: lb[:self.n] = ub[:self.n] = fix
        res = milp(obj, integrality=np.ones(self.nv), bounds=Bounds(lb, ub), constraints=LinearConstraint(self.A, self.lo, self.hi), options=dict(time_limit=tl, disp=False)); self.last = res
        return None if res.x is None else np.round(res.x)[:self.n]

    def _reach(self):
        if not hasattr(self, 'reach'):
            self.reach = np.zeros((self.T + 1, self.n), bool); self.reach[self.T, self.src] = True
            for t in range(self.T - 1, -1, -1): self.reach[t] = (self.adj[:, self.reach[t + 1]]).any(1)
        return self.reach

    def solve(self, c, width=200):
        """Beam search over closed walks; a walk's value is the sum of its B best positive visited scores."""
        reach = self._reach(); pos = np.maximum(c, 0.); order = np.argsort(-pos); nbr = [np.flatnonzero(self.adj[j]) for j in range(self.n)]
        nodes = np.array([self.src]); vis = np.zeros((1, self.n), bool)
        for t in range(1, self.T + 1):
            par = np.concatenate([np.full(len(nbr[j]), q) for q, j in enumerate(nodes)]); nxt = np.concatenate([nbr[j] for j in nodes])
            ok = reach[t, nxt]; par, nxt = par[ok], nxt[ok]; v = vis[par].copy(); v[np.arange(len(nxt)), nxt] = True
            key = np.packbits(np.column_stack([v, (nxt[:, None] >> np.arange(6)) & 1]), axis=1); _, u = np.unique(key, axis=0, return_index=True); nxt, v = nxt[u], v[u]
            val = np.sort(v * pos, 1)[:, -self.B:].sum(1); top = np.argsort(-val)[:width]; nodes, vis = nxt[top], v[top]
        best = vis[np.argmax(np.sort(vis * pos, 1)[:, -self.B:].sum(1))]; a = np.zeros(self.n); chosen = [j for j in order if best[j] and c[j] > 0][:self.B]; a[chosen] = 1
        return a

    def mean_map(self, eta):
        """Forward-backward over (t, node, pulls). eta has shape (T, n): score of pulling node k on arrival at step t+1."""
        n, T, B = self.n, self.T, self.B; A = self.adj.astype(float); e = np.exp(eta)
        F = np.zeros((T + 1, n, B + 1)); F[0, self.src, 0] = 1
        for t in range(T):
            arr = A.T @ F[t]; F[t + 1] = arr; F[t + 1, :, 1:] += e[t][:, None] * arr[:, :-1]
        G = np.zeros((T + 1, n, B + 1)); G[T, self.src] = 1
        for t in range(T - 1, -1, -1):
            nxt = G[t + 1].copy(); nxt[:, :-1] += e[t][:, None] * G[t + 1, :, 1:]; G[t] = A @ nxt
        z = F[T, self.src].sum(); mu = np.zeros((T, n))
        for t in range(T):
            arr = A.T @ F[t]; mu[t] = (arr[:, :-1] * e[t][:, None] * G[t + 1, :, 1:]).sum(1) / z
        return mu


class RoutingPublished(Rmab):
    """What the released routing code optimizes: step only checks the budget and the decoder is top-B."""
    def solve(self, c):
        a = np.zeros(self.m); idx = np.argsort(-c)[:self.B]; a[idx[c[idx] > 0]] = 1
        return a

    def random_action(self, rng):
        a = np.zeros(self.m); a[rng.choice(self.m, self.B, replace=False)] = 1
        return a


def k_subset_mean(eta, B):
    """Marginals of P(a) proportional to exp(eta.a) over |a|<=B."""
    n = len(eta); e = np.exp(eta); F = np.zeros((n + 1, B + 1)); F[0, 0] = 1
    for j in range(n):
        F[j + 1] = F[j]; F[j + 1, 1:] += e[j] * F[j, :-1]
    G = np.zeros((n + 1, B + 1)); G[n] = 1
    for j in range(n - 1, -1, -1):
        G[j, :] = G[j + 1]; G[j, :-1] += e[j] * G[j + 1, 1:]
    z = G[0, 0]
    return np.array([e[j] * (F[j, :-1] * G[j + 1, 1:]).sum() / z for j in range(n)])


def assign_mean(cost, cap, eta):
    """Exact marginals on at-most-one-worker assignments by autograd through the capacity tensor DP."""
    import torch
    eta = torch.as_tensor(eta, dtype=torch.float64).requires_grad_(True); F = torch.zeros(*(int(c) + 1 for c in cap), dtype=torch.float64); F[(0,) * len(cap)] = 1
    for j, c in enumerate(cost):
        F = _assign_step(F, int(c), [int(x) for x in cap], torch.exp(eta[:, j]))
    log_z = torch.log(F.sum()); log_z.backward()
    return eta.grad.numpy()


def _assign_step(F, c, cap, w):
    import torch
    new = F.clone()
    for i, ci in enumerate(cap):
        if c > ci: continue
        new = new + w[i] * torch.nn.functional.pad(F.narrow(i, 0, ci + 1 - c), _pad(len(cap), i, c))
    return new


def _pad(d, i, c):
    p = [0] * (2 * d); p[2 * (d - 1 - i)] = c
    return p


def assign_forward_seconds(cost, cap, eta, limit=120.):
    """Forward pass only, stopping at the limit. Returns seconds, arms finished, and tensor size."""
    import torch
    size = int(np.prod([int(c) + 1 for c in cap], dtype=np.int64)); out = dict(states=size, gb_mean=size * 8 * (len(cost) + 1) / 1e9)
    if size * 8 > 8e9: return dict(out, seconds=None, arms=0, stopped='memory')
    eta = torch.as_tensor(eta, dtype=torch.float64)
    with torch.no_grad():
        F = torch.zeros(*(int(c) + 1 for c in cap), dtype=torch.float64); F[(0,) * len(cap)] = 1; t0 = time.perf_counter()
        for j, c in enumerate(cost):
            F = _assign_step(F, int(c), [int(x) for x in cap], torch.exp(eta[:, j])); F = F / F.sum()
            if time.perf_counter() - t0 > limit: return dict(out, seconds=time.perf_counter() - t0, arms=j + 1, stopped='time')
    return dict(out, seconds=time.perf_counter() - t0, arms=len(cost), stopped=None)


def sched_mean(task, eta_a, eta_x):
    """Exact a marginals by mask DP over worker-slot occupancy. Python dicts: small instances only."""
    bit = {}; pairs = {j: [] for j in range(task.n)}; by = {j: [e for e in task.edges if e[1] == j] for j in range(task.n)}
    for e in task.edges: bit.setdefault((e[0], e[2]), len(bit))
    for j in range(task.n):
        for e1, e2 in combinations(by[j], 2):
            b1, b2 = 1 << bit[e1[0], e1[2]], 1 << bit[e2[0], e2[2]]
            if b1 != b2: pairs[j].append((b1 | b2, math.exp(eta_a[j] + eta_x[e1] + eta_x[e2])))
    F = [{(0, 0): 1.}]
    for j in range(task.n):
        nxt = dict(F[-1])
        for (m, c), w in F[-1].items():
            if c == task.B: continue
            for bits, v in pairs[j]:
                if not m & bits: nxt[m | bits, c + 1] = nxt.get((m | bits, c + 1), 0.) + w * v
        F.append(nxt)
    G = {k: 1. for k in F[-1]}; marg = np.zeros(task.n); Z = sum(F[-1].values())
    for j in range(task.n - 1, -1, -1):
        new = {}; tot = 0.
        for (m, c), w in F[j].items():
            g = G.get((m, c), 0.); s = 0.
            if c < task.B:
                for bits, v in pairs[j]:
                    if not m & bits: s += v * G.get((m | bits, c + 1), 0.)
            new[m, c] = g + s; tot += w * s
        marg[j] = tot / Z; G = new
    return marg


def sched_reachable(task, limit_s=10., limit_states=int(5e7)):
    """How far the (mask, count) DP gets at the published size. Exceeding a limit rules out this DP, not the problem."""
    bit = {}; by = {j: [e for e in task.edges if e[1] == j] for j in range(task.n)}
    for e in task.edges: bit.setdefault((e[0], e[2]), len(bit))
    order = sorted(range(task.n), key=lambda j: len(by[j])); keys = np.array([0], dtype=np.uint64); t0 = time.perf_counter(); peak = 1
    for q, j in enumerate(order):
        masks = sorted({(1 << bit[a[0], a[2]]) | (1 << bit[b[0], b[2]]) for a, b in combinations(by[j], 2) if bit[a[0], a[2]] != bit[b[0], b[2]]}); parts = [keys]
        for b in masks:
            sel = keys[((keys >> np.uint64(5)) & np.uint64(b)) == 0]
            sel = sel[(sel & np.uint64(31)) < task.B]
            parts.append((sel | (np.uint64(b) << np.uint64(5))) + np.uint64(1))
        keys = np.unique(np.concatenate(parts)); peak = max(peak, keys.size)
        if time.perf_counter() - t0 > limit_s or keys.size > limit_states:
            return dict(bits=len(bit), arms=q + 1, of=task.n, states=int(keys.size), seconds=time.perf_counter() - t0, stopped=True)
    return dict(bits=len(bit), arms=task.n, of=task.n, states=int(peak), seconds=time.perf_counter() - t0, stopped=False)


def make_task(name, inst):
    if name == 'intervention': return Intervention(inst)
    if name == 'routing_published': return RoutingPublished(upstream_env('routing', inst))
    env = upstream_env(name, inst)
    return dict(scheduling=Scheduling, routing=Routing, assignment=Assignment)[name](env)


def rollout(task, s, t, a, u, cont):
    """u holds this rollout's transition uniforms, shared by all actions so comparisons use common random numbers."""
    total = 0.
    while True:
        s = np.clip(s + 2 * (u[t] < task.p_up(a)).astype(int) - 1, 0, task.ns - 1); total += task.reward(s); t += 1
        if t >= task.H: return total
        a = cont(s)


def cv_residual(X, y, idx, folds=5, ridge=1e-6):
    keep = X.std(0) > 0; X = X[:, keep]; pred = np.zeros(len(y))
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te); mx, my = X[tr].mean(0), y[tr].mean(); Z = X[tr] - mx
        pred[te] = (X[te] - mx) @ np.linalg.solve(Z.T @ Z + ridge * np.eye(Z.shape[1]), Z.T @ (y[tr] - my)) + my
    return y - pred


def cv_r2(X, y, rng):
    e = cv_residual(X, y, rng.permutation(len(y)))
    return float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum())


def cross_r2(X, ya, yb, rng):
    """Nonlinear share of the true Q: rollout noise is independent between halves, so covariances keep only signal."""
    idx = rng.permutation(len(ya)); ea, eb = cv_residual(X, ya, idx), cv_residual(X, yb, idx)
    signal = float(np.cov(ya, yb)[0, 1])
    return (1 - float(np.cov(ea, eb)[0, 1]) / signal if signal > 0 else None), signal


def r2_block(task, s, h, actions, rollouts, rng, make_cont):
    X = np.array(actions); base = int(rng.integers(1 << 31)); G = np.zeros((len(X), rollouts))
    for r in range(rollouts):
        g = np.random.default_rng([base, r]); u = g.random((task.H, task.n)); seed = int(g.integers(1 << 31))
        for i, a in enumerate(X): G[i, r] = rollout(task, s, h, a, u, make_cont(seed))
    y = G.mean(1); uniq = len({a.tobytes() for a in X}); half = rollouts // 2; ya, yb = G[:, :half].mean(1), G[:, half:].mean(1)
    r = float(np.corrcoef(ya - ya.mean(), yb - yb.mean())[0, 1]) if y.std() > 0 else 0.; rel = 2 * r / (1 + r) if r > 0 else 0.
    r2c, signal = cross_r2(X, ya, yb, rng); usable = rel >= .5 and uniq >= 60 and r2c is not None
    return dict(r2=cv_r2(X, y, rng), r2_corrected=r2c, reliability=rel, usable=usable, unique=uniq, q_std=float(y.std()), signal_var=signal, q_mean=float(y.mean()))


def run_q1(cfg):
    task = make_task(cfg['task'], cfg['instance']); rng = np.random.default_rng(1000 * cfg['instance'] + cfg['h'] + 7); cache = {}
    def greedy(s):
        k = s.tobytes()
        if k not in cache: cache[k] = task.solve(task.coeff(s))
        return cache[k]
    s = task.init.copy()
    for t in range(cfg['h']):
        s = np.clip(s + 2 * (rng.random(task.n) < task.p_up(greedy(s))).astype(int) - 1, 0, task.ns - 1)
    n, ro = cfg['n_actions'], cfg['rollouts']; c = task.coeff(s); res = dict(state=s.tolist(), coeff_std=float(c.std()))
    t0 = time.time(); rand = [task.random_action(rng) for _ in range(n)]; near = [task.solve(c + c.std() * rng.standard_normal(len(c))) for _ in range(n)]; res['action_sec'] = time.time() - t0
    t0 = time.time(); res['control_random_cont'] = r2_block(task, s, cfg['h'], rand, ro, rng, lambda seed: (lambda g: (lambda _: task.random_action(g)))(np.random.default_rng(seed))); res['control_sec'] = time.time() - t0
    t0 = time.time(); res['random'] = r2_block(task, s, cfg['h'], rand, ro, rng, lambda _: greedy); res['near'] = r2_block(task, s, cfg['h'], near, ro, rng, lambda _: greedy); res['greedy_sec'] = time.time() - t0; res['solves'] = len(cache)
    return res


def _topb(c, task):
    a = np.zeros(task.m); a[np.argsort(-c)[:task.B]] = 1
    return a


def timed(fn, cs):
    out, ts = [], []
    for c in cs:
        t = time.perf_counter(); out.append(fn(c)); ts.append(time.perf_counter() - t)
    return out, dict(p50=float(np.percentile(ts, 50)), p95=float(np.percentile(ts, 95)), max=float(np.max(ts)), n=len(ts))


def run_struct(cfg):
    name, inst = cfg['task'], cfg['instance']; task = make_task(name, inst); rng = np.random.default_rng(5000 + inst); res = {}
    cs = [rng.standard_normal(task.m) for _ in range(cfg['n_solver'])]
    # Q3: solver cost
    acts, res['solver'] = timed(task.solve, cs)
    if name == 'routing':
        strict, res['solver_strict_mip'] = timed(task.solve_strict, cs[:cfg['n_mip']]); gaps = []; none = 0
        for c, a, x in zip(cs, acts, strict):
            if x is None: none += 1; continue
            gaps.append(float((c @ x - c @ a) / max(c @ x, 1e-9)))
        res['beam_vs_strict_gap'] = dict(mean=float(np.mean(gaps)), max=float(np.max(gaps)), none=none, n=len(gaps))
        _, res['solver_published_mip'] = timed(task.solve_mip, cs[:cfg['n_mip']])
        res['beam_action_feasible'] = sum(task.solve_strict(np.zeros(task.n), fix=a) is not None for a in acts[:cfg['n_topb']])
        bad = bad_s = 0
        for c in cs[:cfg['n_topb']]:
            a = np.zeros(task.m); a[np.argsort(-c)[:task.B]] = 1
            bad += task.solve_mip(np.zeros(task.m), fix=a) is None; bad_s += task.solve_strict(np.zeros(task.m), fix=a, tl=5.) is None
        res['topb_infeasible'] = dict(published=bad, strict=bad_s, n=cfg['n_topb'])
    if name == 'scheduling':
        bad = sum(task.solve_full(np.zeros(task.m), fix=(lambda a: a)(_topb(c, task))) is None for c in cs[:cfg['n_topb']])
        res['topb_infeasible'] = dict(bad=bad, n=cfg['n_topb'])
    res['action_counts'] = dict(mean=float(np.mean([a.sum() for a in acts])), budget=task.B)
    # Q2: exact mean map at the published size
    if name in ('intervention', 'routing_published'):
        _, res['mean_map'] = timed(lambda c: k_subset_mean(c, task.B), cs[:50])
    elif name == 'routing':
        etas = [rng.standard_normal((task.T, task.n)) for _ in range(20)]; _, res['mean_map'] = timed(task.mean_map, etas); res['extended_bits'] = task.T * task.n
    elif name == 'assignment':
        res['mean_map_forward'] = assign_forward_seconds(task.cost, task.cap, rng.standard_normal((task.nw, task.n)), cfg['forward_limit'])
    elif name == 'scheduling':
        res['mask_dp'] = sched_reachable(task)
    # Q2: generic perturb-and-MAP on small instances
    if name in ('scheduling', 'assignment'):
        small = (Scheduling if name == 'scheduling' else lambda e: Assignment(e, single=True))(upstream_env(name, inst, n=20, b=5)); rows = []
        for sig in (.5, 1., 2.):
            for _ in range(cfg['pm_draws']):
                if name == 'scheduling':
                    ea = sig * rng.standard_normal(small.n); ex = {e: sig * rng.standard_normal() for e in small.edges}; exact = sched_mean(small, ea, ex); base = np.concatenate([ea, [ex[e] for e in small.edges]])
                else:
                    eta = sig * rng.standard_normal((small.nw, small.n)); exact = assign_mean(small.cost, small.cap, eta).sum(0); base = np.concatenate([np.zeros(small.n), eta.flatten()])
                est = np.zeros(small.n); t0 = time.perf_counter()
                for _ in range(cfg['pm_k']):
                    est += small.solve_full(base + rng.gumbel(size=small.nv) - np.euler_gamma)[:small.n]
                rows.append(dict(sigma=sig, mae=float(np.abs(est / cfg['pm_k'] - exact).mean()), sec=time.perf_counter() - t0))
        res['perturb_map'] = rows
    return res


def source_hash():
    return hashlib.sha256((HERE / 'screen.py').read_bytes()).hexdigest()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--output', required=True)
    args = ap.parse_args(); cfg = json.loads(Path(args.config).read_text()); out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    res = (run_q1 if cfg['kind'] == 'q1' else run_struct)(cfg)
    (out / 'result.json').write_text(json.dumps(dict(config=cfg, status='completed', source=source_hash(), host=platform.node(), pid=os.getpid(), scipy=scipy.__version__, **res), indent=1))
