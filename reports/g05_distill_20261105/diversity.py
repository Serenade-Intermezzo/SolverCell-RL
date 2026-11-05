"""G0.5 post-hoc diagnostic: diversity and a bracket on the max-ent objective for saved models. Decides no gate. Design: docs/G05_DISTILLATION_DESIGN.md."""
import argparse, hashlib, json, math, os, platform
from pathlib import Path
import torch
import g05
from g05 import T_STEPS, M, START, TAIL, Denoiser, Objective, chain, group_resample, lin_value, marginals, walk, backward

HERE = Path(__file__).resolve().parent
N, K, ALPHA, KAPPA = 4096, 12, .1, 5.


def log_z(eta):
    return torch.logsumexp(backward(eta)[0][:, START], 1)


def exact_entropy(eta):
    """Shannon entropy of the exponential family on closed walks: log Z - <eta, mu>."""
    eta = eta.double()
    return float(log_z(eta)[0] - (eta * marginals(eta)).sum())


def markov_probs(path, weight=None):
    """Time-inhomogeneous arc counts per tail state, plus start-state counts."""
    b = path.shape[0]; w = torch.ones(b, dtype=torch.float64) if weight is None else weight
    arcs = torch.zeros(T_STEPS, M, dtype=torch.float64)
    arcs.index_put_((torch.arange(T_STEPS).repeat(b), path.flatten()), w.repeat_interleave(T_STEPS), accumulate=True)
    return arcs


def normalise(c):
    tot = torch.zeros(T_STEPS, g05.S, dtype=torch.float64).index_add(1, TAIL, c)
    q = c / tot[:, TAIL]
    first = c[0] * START[TAIL]                # the first arc also carries the choice of start state
    q[0] = first / first.sum()
    return q


def log_q(path, q):
    return torch.log(q[torch.arange(T_STEPS)[None], path]).sum(1)


def _cond_codes(obj, p):
    return (obj.done(p)[0].long() * torch.tensor([1, 2, 4, 8])).sum(1)


def _fold_bounds(obj, train, test):
    glob = normalise(markov_probs(train) + ALPHA)
    plain = -log_q(test, glob).mean()
    code_tr, code_te = _cond_codes(obj, train), _cond_codes(obj, test)
    log_c = torch.log((torch.bincount(code_tr, minlength=16).double() + 1) / (train.shape[0] + 16))
    lq = torch.zeros(test.shape[0], dtype=torch.float64)
    for c in code_te.unique():
        q = normalise(markov_probs(train[code_tr == c]) + KAPPA * glob); i = code_te == c
        lq[i] = log_q(test[i], q) + log_c[c]
    return float(plain), float(-lq.mean())


def entropy_stats(obj, p):
    """H2 <= Shannon H <= held-out cross-entropy. Cross-entropy of any sub-probability on the action set bounds H from above."""
    n = p.shape[0]; h = n // 2
    a1, b1 = _fold_bounds(obj, p[:h], p[h:]); a2, b2 = _fold_bounds(obj, p[h:], p[:h])
    uniq, counts = torch.unique(p, dim=0, return_counts=True)
    pairs = n * (n - 1) / 2; coll = float((counts * (counts - 1) / 2).sum())
    i = torch.randperm(n, generator=torch.Generator().manual_seed(0)); ham = float((p != p[i]).float().mean())
    pc = torch.bincount(_cond_codes(obj, p), minlength=16).double() / n; pc = pc[pc > 0]
    return dict(h_up=min((a1 + a2) / 2, (b1 + b2) / 2), h_up_markov=(a1 + a2) / 2, h_up_cond=(b1 + b2) / 2,
                h2=-math.log(coll / pairs) if coll else None, h2_floor=math.log(pairs / 3), collisions=coll,
                unique_frac=uniq.shape[0] / n, top1=float(counts.max()) / n, hamming=ham, h_district_set=float(-(pc * pc.log()).sum()))


def report(obj, p, m0, s0, beta, log_a, exact_h=None):
    d = obj.done(p)[0]; gain = float((obj(p).mean() - m0) / s0); st = entropy_stats(obj, p)
    lo = exact_h if exact_h is not None else (st['h2'] if st['h2'] is not None else st['h2_floor'])
    up = exact_h if exact_h is not None else st['h_up']
    # J = beta*gain - KL(pi||rho) with uniform rho, in nats, measured from the base mean; J(rho)=0
    return dict(gain=gain, rate=float(d.any(1).float().mean()), modes=float(d.any(0).sum()), exact_h=exact_h, j_low=beta * gain + lo - log_a, j_up=beta * gain + up - log_a, **st)


def run(cfg, out):
    i, b = cfg['instance'], cfg['beta']; torch.manual_seed(20261105 + 1000 * i + b); gen = torch.Generator().manual_seed(20261105 + 1000 * i + b)
    n = cfg.get('n', N); src = json.loads((Path(cfg['models']) / f'i{i}_b{b}_structured' / 'result.json').read_text())
    m0, s0, lam = src['m0'], src['s0'], src['lam']; obj = Objective(i); zero = torch.zeros(1, T_STEPS, M); res = dict(m0=m0, s0=s0, lam=lam, stored_direct={})
    log_a = float(log_z(zero.double())[0]); res['log_a'] = log_a
    rep = lambda p, h=None: report(obj, p, m0, s0, b, log_a, h)
    res['uniform'] = rep(walk(zero.expand(n, -1, -1), gen), log_a)
    g = obj.grad(marginals(zero)) / lam
    res['shift'] = rep(walk(g.expand(n, -1, -1), gen), exact_entropy(g))
    p = walk(g.expand(K * n, -1, -1), gen); res['shift_res12'] = rep(group_resample(p, obj(p) / lam - lin_value(p, g), K, gen)[0])
    sh = g.clone()
    for _ in range(16): sh = .5 * sh + .5 * obj.grad(marginals(sh)) / lam
    res['loglinear_fp'] = rep(walk(sh.expand(n, -1, -1), gen), exact_entropy(sh))
    # log E_rho exp((Q-m0)/lam) by importance sampling from the exact shift proposal; low ESS biases it low
    lw = []
    for _ in range(32):
        p = walk(g.expand(4096, -1, -1), gen); lw.append(obj(p) / lam - lin_value(p, g))
    lw = torch.cat(lw).double(); w = torch.softmax(lw, 0)
    res['target_j_is'] = dict(value=float(log_z(g.double())[0] - log_a + torch.logsumexp(lw, 0) - math.log(lw.shape[0]) - m0 / lam), ess=float(1 / (w ** 2).sum()))
    for arm in ('structured', 'ef', 'plain'):
        name = f'i{i}_b{b}_{arm}'; model = Denoiser(arm)
        model.load_state_dict(torch.load(Path(cfg['models']) / name / 'model.pt', weights_only=True)); model.eval()
        res['stored_direct'][arm] = json.loads((Path(cfg['models']) / name / 'result.json').read_text())['direct']['gain']
        res[arm] = rep(chain(model, zero, n, gen))
        if arm == 'ef':
            with torch.no_grad(): pot = model.potential(zero, torch.tensor(0.))
            res['ef_exact'] = rep(walk(pot.expand(n, -1, -1), gen), exact_entropy(pot))
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--output', required=True)
    a = ap.parse_args(); cfg = json.loads(Path(a.config).read_text()); out = Path(a.output); out.mkdir(parents=True, exist_ok=False); torch.set_num_threads(1)
    src = hashlib.sha256(b''.join((HERE / f).read_bytes() for f in ('diversity.py', 'g05.py', 'routing_core.py'))).hexdigest()
    res = run(cfg, out); (out / 'result.json').write_text(json.dumps(dict(config=cfg, status='completed', source=src, host=platform.node(), pid=os.getpid(), torch=torch.__version__, threads=torch.get_num_threads(), **res), indent=1))
