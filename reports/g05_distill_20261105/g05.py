"""G0.5 worker: annealed distillation on Style-Routing district objectives. Design: docs/G05_DISTILLATION_DESIGN.md."""
import argparse, hashlib, json, math, os, platform, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from routing_core import Routes

HERE = Path(__file__).resolve().parent
RR = Routes(T=20)
TAIL, HEAD, NODE = (torch.as_tensor(x) for x in (RR.tail, RR.head, RR.node))
START = torch.as_tensor(RR.start)
T_STEPS, M, S, N_ST = RR.T, RR.m, RR.S, RR.n
MINUTES = torch.as_tensor(RR.minutes, dtype=torch.float32)
NEG = -1e5   # finite stand-in for log 0: an all -inf group makes logsumexp gradients NaN


def _padded(keys):
    groups = [torch.where(keys == s)[0] for s in range(S)]
    width = max(len(g) for g in groups); idx = torch.zeros(S, width, dtype=torch.long); mask = torch.zeros(S, width, dtype=torch.bool)
    for s, g in enumerate(groups):
        idx[s, :len(g)] = g; mask[s, :len(g)] = True
    return idx, mask
IN_IDX, IN_MASK = _padded(HEAD)
OUT_IDX, OUT_MASK = _padded(TAIL)


def _group(vals, idx, mask, reduce='sum'):
    v = vals[:, idx].masked_fill(~mask, NEG)
    return torch.logsumexp(v, 2) if reduce == 'sum' else v.max(2).values


def backward(eta, reduce='sum'):
    b = eta.shape[0]; beta = [None] * (T_STEPS + 1)
    last = torch.full((b, S), NEG, dtype=eta.dtype); last[:, START] = 0; beta[T_STEPS] = last
    for t in range(T_STEPS - 1, -1, -1):
        beta[t] = _group(eta[:, t] + beta[t + 1][:, HEAD], OUT_IDX, OUT_MASK, reduce)
    return beta


def marginals(eta):
    """Exact mean map of the exponential family on closed walks; differentiable in eta."""
    b = eta.shape[0]; beta = backward(eta)
    alpha = torch.full((b, S), NEG, dtype=eta.dtype); alpha[:, START] = 0
    log_z = torch.logsumexp(beta[0][:, START], 1, keepdim=True)
    mu = []
    for t in range(T_STEPS):
        vals = alpha[:, TAIL] + eta[:, t]
        mu.append(torch.exp(vals + beta[t + 1][:, HEAD] - log_z))
        alpha = _group(vals, IN_IDX, IN_MASK)
    return torch.stack(mu, 1)


def walk(eta, gen=None):
    """gen=None gives the exact linear solver; otherwise an exact sample from the exponential family."""
    b = eta.shape[0]; beta = backward(eta, 'max' if gen is None else 'sum'); ix = torch.arange(b)
    sc = beta[0].clone(); sc[:, ~START] = -math.inf
    def noise(x): return -torch.log(-torch.log(torch.rand(x.shape, generator=gen).clamp_min(1e-20)))
    if gen is not None: sc = sc + noise(sc)
    s = sc.argmax(1); path = torch.zeros(b, T_STEPS, dtype=torch.long)
    for t in range(T_STEPS):
        v = eta[:, t] + beta[t + 1][:, HEAD]
        v = v.masked_fill(TAIL[None] != s[:, None], -math.inf)
        if gen is not None: v = v + noise(v)
        path[:, t] = v.argmax(1); s = HEAD[path[:, t]]
    return path


def occupancy(mu):
    out = torch.zeros(mu.shape[0], T_STEPS, N_ST, dtype=mu.dtype)
    return out.index_add(2, NODE, mu)


class Objective:
    """Frozen instance: district bonuses, station prizes, and travel-time cost; all known."""
    def __init__(self, seed):
        rng = np.random.default_rng(seed); adj = {i: set() for i in range(N_ST)}
        for t, n in zip(RR.tail, RR.node):
            u = RR.states[t][0]
            if u != n: adj[u].add(int(n)); adj[int(n)].add(u)
        used, D = {RR.src}, []
        while len(D) < 4:
            s = int(rng.integers(N_ST))
            if s in used: continue
            g = [s]
            while len(g) < 4:
                c = sorted({x for y in g for x in adj[y]} - used - set(g))
                if not c: break
                g.append(int(rng.choice(c)))
            if len(g) == 4: D.append(g); used |= set(g)
        self.D = D; self.bonus = torch.as_tensor(rng.uniform(.8, 1.2, 4), dtype=torch.float32)
        self.base = torch.as_tensor(.03 * rng.uniform(.5, 1.5, N_ST), dtype=torch.float32); self.tmin = .02 * MINUTES

    def done(self, p):
        s = torch.zeros(p.shape[0], N_ST).scatter(1, NODE[p], 1.)
        return torch.stack([s[:, g].all(1) for g in self.D], 1), s

    def __call__(self, p):
        d, s = self.done(p)
        return d.float() @ self.bonus + s @ self.base - self.tmin[p].sum(1)

    def grad(self, mu):
        """Multilinear extension gradient at posterior mean mu (b,T,m)."""
        occ = occupancy(mu).clamp(0, 1 - 1e-6); lm = torch.log1p(-occ); tot = lm.sum(1, keepdim=True)
        pv = 1 - torch.exp(tot[:, 0]); dp = torch.exp(tot - lm); val = self.base[None].repeat(mu.shape[0], 1)
        for g, c in zip(self.D, self.bonus):
            for j in g:
                val[:, j] = val[:, j] + c * pv[:, [x for x in g if x != j]].prod(1)
        return (val[:, None] * dp)[:, :, NODE] - self.tmin[None, None]


class Denoiser(nn.Module):
    def __init__(self, arm, hidden=256):
        super().__init__(); self.arm = arm
        if arm == 'ef':
            self.nt = nn.Parameter(torch.zeros(T_STEPS, N_ST)); self.arc = nn.Parameter(torch.zeros(M))
            return
        out = T_STEPS * M if arm == 'plain' else T_STEPS * N_ST + M
        self.net = nn.Sequential(nn.Linear(T_STEPS * N_ST + 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, out))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def potential(self, y, t):
        if self.arm == 'ef':
            nt, arc = self.nt[None], self.arc[None]
        else:
            with torch.no_grad(): occ = occupancy(marginals(2 * y))
            tt = torch.as_tensor(t, dtype=y.dtype).expand(y.shape[0])[:, None] / 16.
            o = self.net(torch.cat([occ.flatten(1), tt, torch.sqrt(tt)], 1))
            if self.arm == 'plain': return o.view(-1, T_STEPS, M)
            nt, arc = o[:, :T_STEPS * N_ST].view(-1, T_STEPS, N_ST), o[:, T_STEPS * N_ST:]
        return nt[:, :, NODE] + arc[:, None]

    def forward(self, y, t):
        """Posterior mean in {0,1} coordinates; plain is exact at zero output so round 0 is the uniform base law."""
        out = self.potential(y, t)
        if self.arm == 'plain':
            with torch.no_grad(): m0 = marginals(2 * y)
            return (m0 + out).clamp(0, 1)
        return marginals(2 * y + out)


def chain(model, q_fn, n_samples, gen, n=16, T_sl=16., chunk=256):
    """SL chain dy=D dt+dW in e-coordinates with a fixed linear shift; the solver decodes at the end."""
    paths = []; ts = T_sl * (torch.arange(n + 1) / n) ** 2
    for i0 in range(0, n_samples, chunk):
        b = min(chunk, n_samples - i0); y = torch.zeros(b, T_STEPS, M); sh = q_fn.expand(b, -1, -1)
        with torch.no_grad():
            for i in range(n):
                h = ts[i + 1] - ts[i]; d = 2 * model(y + sh, ts[i]) - 1
                y = y + h * d + torch.sqrt(h) * torch.randn(y.shape, generator=gen)
            paths.append(walk(y + sh))
    return torch.cat(paths)


def lin_value(p, g):
    return g[0, torch.arange(T_STEPS)[None], p].sum(1)


def group_resample(p, lw, K, gen):
    lw = lw.view(-1, K); w = torch.softmax(lw, 1); ch = torch.multinomial(w, 1, generator=gen)[:, 0]
    return p.view(-1, K, T_STEPS)[torch.arange(lw.shape[0]), ch], float((1 / (w ** 2).sum(1)).mean())


def score(obj, p, m0, s0):
    d, _ = obj.done(p)
    return dict(gain=float((obj(p).mean() - m0) / s0), rate=float(d.any(1).float().mean()), modes=float(d.any(0).sum()))


def run(cfg, out):
    torch.manual_seed(cfg['fit_seed']); gen = torch.Generator().manual_seed(cfg['fit_seed'])
    obj = Objective(cfg['instance'])
    zero = torch.zeros(1, T_STEPS, M)
    q0 = obj(walk(zero.expand(4096, -1, -1), gen)); m0, s0 = float(q0.mean()), float(q0.std()); lam = s0 / cfg['beta']
    res = dict(m0=m0, s0=s0, lam=lam, rounds=[]); K = cfg['K']; J = cfg['rounds']; delta = 1 / (J * lam)
    # one-shot exact controls
    g_full = obj.grad(marginals(zero)) / lam
    p = walk(g_full.expand(K * cfg['n_out'], -1, -1), gen)
    res['shift'] = score(obj, p[:cfg['n_out']], m0, s0)
    pk, _ = group_resample(p, obj(p) / lam - lin_value(p, g_full), K, gen); res['shift_res12'] = score(obj, pk, m0, s0)
    sh = g_full.clone()
    for _ in range(16): sh = .5 * sh + .5 * obj.grad(marginals(sh)) / lam
    res['loglinear_fp'] = score(obj, walk(sh.expand(cfg['n_eval'], -1, -1), gen), m0, s0)
    lw, cs, dn = [], [], []
    for _ in range(cfg['ref_chunks']):
        p = walk(g_full.expand(4096, -1, -1), gen); c = obj(p); lw.append(c / lam - lin_value(p, g_full)); cs.append(c); dn.append(obj.done(p)[0].float())
    lw, cs, dn = torch.cat(lw), torch.cat(cs), torch.cat(dn); w = torch.softmax(lw, 0)
    res['ref'] = dict(gain=float(((w * cs).sum() - m0) / s0), ess=float(1 / (w ** 2).sum()), rate=float((w @ (dn.sum(1) > 0).float())), modes=float(((w @ dn) > .05).sum()))
    # annealed distillation
    model = Denoiser(cfg['arm']); opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    for j in range(J):
        t0 = time.time()
        with torch.no_grad(): mean = model(torch.zeros(1, T_STEPS, M), torch.tensor(0.))
        g = delta * obj.grad(mean)                       # tilt in a-coordinates; e-coordinate shift is g/2
        p = chain(model, g / 2, K * cfg['n_out'], gen)
        lw = delta * obj(p) - lin_value(p, g)
        pk, ess = group_resample(p, lw, K, gen)
        row = dict(round=j, prop=score(obj, p[:cfg['n_out']], m0, s0), prop_res12=score(obj, pk, m0, s0), ess=ess)
        e = 2 * RR_bits(pk) - 1; losses = []
        for step in range(cfg['fit_steps']):
            i = torch.randint(0, e.shape[0], (cfg['batch'],), generator=gen)
            t = 16. * torch.rand(cfg['batch'], 1, 1, generator=gen)
            y = t * e[i] + torch.sqrt(t) * torch.randn(cfg['batch'], T_STEPS, M, generator=gen)
            pred = batched_forward(model, y, t)
            loss = ((2 * pred - 1 - e[i]) ** 2).sum((1, 2)).mean() / T_STEPS
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step(); losses.append(float(loss))
        row['loss_first'] = float(np.mean(losses[:20])); row['loss_last'] = float(np.mean(losses[-20:]))
        row['direct'] = score(obj, chain(model, torch.zeros(1, T_STEPS, M), cfg['n_eval'], gen), m0, s0)
        row['sec'] = time.time() - t0; res['rounds'].append(row); print(json.dumps(dict(j=j, direct=row['direct'], prop_res12=row['prop_res12'], ess=round(ess, 2), loss=row['loss_last'], sec=round(row['sec'], 1))), flush=True)
        write(out, cfg, res, 'running')
    res['direct'] = res['rounds'][-1]['direct']; res['prop_res12'] = res['rounds'][-1]['prop_res12']
    if cfg['arm'] == 'ef':
        with torch.no_grad(): pot = model.potential(zero, torch.tensor(0.))
        res['ef_exact_direct'] = score(obj, walk(pot.expand(cfg['n_eval'], -1, -1), gen), m0, s0)
    torch.save(model.state_dict(), out / 'model.pt')
    return res


def RR_bits(p):
    return torch.zeros(p.shape[0], T_STEPS, M).scatter(2, p[:, :, None], 1.)


def batched_forward(model, y, t):
    # time differs per row; the network takes a per-row t, the exact-structure ef arm ignores it
    if model.arm == 'ef': return model(y, t)
    return _per_row(model, y, t)


def _per_row(model, y, t):
    tt = t.view(-1)
    with torch.no_grad(): m0 = marginals(2 * y); occ = occupancy(m0)
    v = tt[:, None] / 16.
    o = model.net(torch.cat([occ.flatten(1), v, torch.sqrt(v)], 1))
    if model.arm == 'plain': return (m0 + o.view(-1, T_STEPS, M)).clamp(0, 1)
    nt, arc = o[:, :T_STEPS * N_ST].view(-1, T_STEPS, N_ST), o[:, T_STEPS * N_ST:]
    return marginals(2 * y + nt[:, :, NODE] + arc[:, None])


def source_hash():
    h = hashlib.sha256()
    for f in ('g05.py', 'routing_core.py'): h.update((HERE / f).read_bytes())
    return h.hexdigest()


def write(out, cfg, res, status):
    tmp = out / 'result.tmp'
    tmp.write_text(json.dumps(dict(config=cfg, status=status, source=source_hash(), host=platform.node(), pid=os.getpid(), torch=torch.__version__, threads=torch.get_num_threads(), **res), indent=1))
    tmp.replace(out / 'result.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--output', required=True)
    args = ap.parse_args(); cfg = json.loads(Path(args.config).read_text()); out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False); torch.set_num_threads(1)
    write(out, cfg, {}, 'running'); res = run(cfg, out); write(out, cfg, res, 'completed')
