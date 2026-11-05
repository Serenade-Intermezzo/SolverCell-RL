"""G0.6 Stage A worker. Design: docs/G06_SCHEDULING_PREREG.md. phase 1: A1 and A4 on one calibration instance; phase 2: the other gates at the chosen w5."""
import argparse, hashlib, json, os, platform, resource, time
from pathlib import Path
import numpy as np, scipy
from sched_core import *

HERE = Path(__file__).resolve().parent
BETAS = (2, 5, 10)


def rho_stats(inst, cfg):
    f = inst.features(cfg); return f, dict(mean=f.mean(0).tolist(), std=f.std(0).tolist(), selected=float(inst.sel[inst.arange[None], cfg].sum(1).mean()))


def mean_field(inst, cfg):
    return inst.sel[inst.arange[None], cfg].mean(0), inst.load[inst.arange[None], cfg].sum(1, dtype=np.float64).mean(0)


def bits(inst, cfg):
    n = cfg.shape[0]; x = np.zeros((n, inst.E + 1)); e = inst.cfg_e[inst.arange[None], cfg]; ix = np.arange(n)[:, None]
    x[ix, e[:, :, 0]] = 1; x[ix, e[:, :, 1]] = 1
    return np.concatenate([x[:, :inst.E], inst.sel[inst.arange[None], cfg].astype(float)], 1)


def r2(xtr, ytr, xte, yte, ridge=1e-3):
    mx, my = xtr.mean(0), ytr.mean(); a = xtr - mx
    th = np.linalg.solve(a.T @ a + ridge * len(a) * np.eye(a.shape[1]), a.T @ (ytr - my))
    return float(1 - ((yte - my - (xte - mx) @ th) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())


def phase1(cfg, res):
    seed = cfg['instance']; rng = np.random.default_rng(20261105 + seed); L = Instance(seed, **SCALES['L'])
    res['size'] = dict(E=L.E, W=L.W, cfg_mean=float((L.n_cfg - 1).mean()))
    sam = Sampler(L, rng); t = time.time(); rho = sam.rho(cfg['n_rho'], cfg['sweeps']); res['rho_sec'] = time.time() - t
    f, res['rho'] = rho_stats(L, rho); sc = np.maximum(f.std(0), .02); res['scales'] = sc.tolist(); res['group_positive'] = float((f[:, 4] > 0).mean())
    half = sam.rho(cfg['n_rho'] // 4, cfg['sweeps'] // 2); res['rho_half_mean'] = rho_stats(L, half)[1]['mean']
    mu_a, load = mean_field(L, rho); res['a4'] = {}
    for w5 in cfg['w5_grid']:
        w = np.array([1, 1, 1, 1, w5]); row = dict()
        q0 = f / sc @ w; s0 = float(q0.std())
        for beta in BETAS:
            lam = s0 / beta; g_a, g_x = L.gradient(mu_a, load, w, sc); pot = L.linear_potential(g_a / lam, g_x / lam)
            c, acc, _ = sam.run(rho[:cfg['n_a4']], cfg['sweeps'] * L.n, linear=pot); y = q_value(L, c, w, sc); X = bits(L, c); h = len(y) // 2
            row[beta] = dict(r2=r2(X[:h], y[:h], X[h:], y[h:]), acc=acc, gain=float((y.mean() - q0.mean()) / s0))
        res['a4'][w5] = row
    res['mu_a'] = mu_a.tolist(); res['load'] = load.tolist(); res['s0_by_w5'] = {w5: float((f / sc @ np.array([1, 1, 1, 1, w5])).std()) for w5 in cfg['w5_grid']}
    return res


def phase_a1(cfg, res):
    seed = cfg['instance']; L = Instance(seed, **SCALES['L']); w5 = cfg['w5']; p1 = json.loads((Path(cfg['phase1']) / f'p1_i{seed}' / 'result.json').read_text())
    try:
        res['a1'] = frontier_dp(L, cfg['a1_time'], cfg['a1_mem'])
    except MemoryError:
        res['a1'] = dict(completed=False, memory_error=True, widths=[0], peak=None, sec=None, max_bytes=None)
    if not res['a1']['completed']:
        res['a1']['hard'] = True; return res
    sc = np.array(p1['scales']); w = np.array([1, 1, 1, 1, w5]); lam = p1['s0_by_w5'][str(w5)] / 5; g_a, g_x = L.gradient(np.array(p1['mu_a']), np.array(p1['load']), w, sc); pot = L.linear_potential(g_a / lam, g_x / lam); order = frontier_order(L); tm = []
    try:
        for _ in range(3):
            t = time.perf_counter(); mu, lz, H = frontier_mean(L, pot, order, cfg['a1_time']); tm.append(time.perf_counter() - t)
            if tm[-1] > 60: break
        res['a1_mean'] = dict(sec=tm, log_z=lz, entropy=H, selected=float((mu * L.sel).sum())); res['a1']['exact_mean_sec'] = float(np.median(tm)); res['a1']['hard'] = res['a1']['exact_mean_sec'] > 10
    except (MemoryError, TimeoutError) as e:
        res['a1_mean'] = dict(sec=tm, error=type(e).__name__); res['a1']['hard'] = True
    return res


def phase2(cfg, res):
    seed = cfg['instance']; rng = np.random.default_rng(20261205 + seed); w5 = cfg['w5']; w = np.array([1, 1, 1, 1, w5])
    L = Instance(seed, **SCALES['L']); p1 = json.loads((Path(cfg['phase1']) / f'p1_i{seed}' / 'result.json').read_text()); sc = np.array(p1['scales'])
    sam = Sampler(L, rng); rho = sam.rho(cfg['n_rho'], cfg['sweeps']); f = L.features(rho); mu_a, load = mean_field(L, rho); s0 = float((f / sc @ w).std()); m0 = float((f / sc @ w).mean())
    # A2
    sol = Solver(L); g_a, g_x = L.gradient(mu_a, load, w, sc); lam = s0 / 5; times = []; feas = 0
    for _ in range(cfg['n_a2']):
        ga, gx = g_a / lam + rng.gumbel(size=L.n), g_x / lam + rng.gumbel(size=L.E); t = time.perf_counter(); c, _ = sol.solve(ga, gx); times.append(time.perf_counter() - t); feas += bool(L.is_feasible(c[None])[0])
    res['a2'] = dict(p50=float(np.percentile(times, 50)), p95=float(np.percentile(times, 95)), max=float(np.max(times)), feasible=feas / cfg['n_a2'], scipy=scipy.__version__)
    # A3
    def target(wt, n):
        s = float((f / sc @ wt).std()); lt = lambda x: q_value(L, x, wt, sc) / (s / 5)
        c1, _, _ = sam.run(rho[:n], cfg['sweeps'] * L.n // 2, log_target=lt); c2, acc, _ = sam.run(c1, cfg['sweeps'] * L.n // 2, log_target=lt)
        return (L.features(c2) / sc).mean(0), (L.features(c1) / sc).mean(0), acc
    names = dict(fast=1, smooth=2, personal=3); style = {}
    for nm, k in names.items():
        wt = w.copy(); wt[k] = 3; m, m1, acc = target(wt, cfg['n_a3']); style[nm] = m; res.setdefault('a3_means', {})[nm] = dict(mean=m.tolist(), half=m1.tolist(), acc=acc)
    pairs = []
    for a, ka in names.items():
        for b, kb in names.items():
            if a < b:
                da = style[a][ka] - style[b][ka]; db = style[a][kb] - style[b][kb]; ok = (da >= 1 and db <= -.5) or (-db >= 1 and -da <= -.5)
                pairs.append(dict(a=a, b=b, d_a=float(da), d_b=float(db), ok=bool(ok)))
    sweep = []
    for t in np.arange(9) / 8:
        wt = w.copy(); wt[1] = 1 + 2 * t; wt[2] = 3 - 2 * t; m, _, _ = target(wt, cfg['n_a3'] // 4); sweep.append([float(m[1]), float(m[2])])
    pts = np.array(sweep); front = [p for k, p in enumerate(pts) if not any((o >= p).all() and (o > p).any() for o in pts)]; dist = []
    for p in front:
        if all(np.abs(p - o).max() >= .05 for o in dist): dist.append(p)
    res['a3'] = dict(pairs=pairs, n_ok=sum(p['ok'] for p in pairs), sweep=sweep, pareto_points=len(dist))
    # S-scale gates: A5, A6, A7
    S = Instance(seed, **SCALES['S']); dp = ExactDP(S); zero = np.zeros_like(S.f_service); V0 = dp.backward(zero); log_a = dp.log_z(V0)
    srho = dp.sample(V0, zero, cfg['n_exact'], rng); sf = S.features(srho); ssc = np.maximum(sf.std(0), .02); res['s'] = dict(log_a=log_a, scales=ssc.tolist(), group_positive=float((sf[:, 4] > 0).mean()))
    mu0, _, _ = dp.linear_marginals(zero); mu_edge0 = dp.edge_marginals(mu0); ssam = Sampler(S, rng)
    mc = ssam.rho(cfg['n_a7_rho'], cfg['sweeps']); xm = bits(S, mc).mean(0); ex = np.concatenate([mu_edge0, (mu0 * S.sel).sum(1)]); res['a7_rho'] = dict(linf=float(np.abs(xm - ex).max()), n=cfg['n_a7_rho'])
    qs = sf / ssc @ w; ms, ss = float(qs.mean()), float(qs.std()); mua = (mu0 * S.sel).sum(1); sload = np.zeros(S.B); np.add.at(sload, S.edges[:, 1], mu_edge0)
    sg_a, sg_x = S.gradient(mua, sload, w, ssc); solS = Solver(S); res['a5'] = {}; res['a6'] = {}; res['a7'] = {}
    for beta in (0,) + BETAS:
        lam = ss / beta if beta else 1.; pot = S.linear_potential(sg_a / lam, sg_x / lam) if beta else zero
        mu, _, H = dp.linear_marginals(pot); ee = dp.edge_marginals(mu); ae, aa = approx_mean(S, pot)
        res['a6'][beta] = dict(mae=float(np.abs(ae - ee).mean()), max=float(np.abs(ae - ee).max()), mean_marginal=float(ee.mean()), arm_mae=float(np.abs(aa - (mu * S.sel).sum(1)).mean()))
        Vp = dp.backward(pot); smp = dp.sample(Vp, pot, cfg['n_a7'], rng); ub = cross_entropy_bound(S, smp, dp.order); res['a7'][beta] = dict(exact=H, bound=ub, slack=ub - H)
        if not beta: continue
        tbl = (w[0] / ssc[0] * S.f_service + w[1] / ssc[1] * S.f_punct + w[3] / ssc[3] * S.f_pref) / lam; term = w[2] / ssc[2] * dp.bal / lam; wg = w[4] / ssc[4] / lam
        V = dp.backward(tbl, term, wg); lz = dp.log_z(V); jstar = lz - log_a - ms / lam
        tg = dp.sample(V, tbl, cfg['n_a7'], rng, wg); tgain = float((q_value(S, tg, w, ssc).mean() - ms) / ss)
        start = perturbed_map(S, solS, sg_a / lam, sg_x / lam, cfg['n_a5'], rng); lt = lambda x: q_value(S, x, w, ssc) / lam; row = dict(j_star=jstar, target_gain=tgain, target_group=float((S.features(tg)[:, 4] > 0).mean()))
        for steps in (0, 12, 48, 192):
            c = start if steps == 0 else ssam.run(start, steps, log_target=lt)[0]; gain = float((q_value(S, c, w, ssc).mean() - ms) / ss); ub = cross_entropy_bound(S, c, dp.order); hc = collision_entropy(c)[0]
            row[f'mcmc{steps}'] = dict(gain=gain, h_up=ub, h2=hc, j_up=beta * gain + ub - log_a, feasible=bool(S.is_feasible(c[:200]).all()), gap_low=jstar - (beta * gain + ub - log_a))
        res['a5'][beta] = row
    # diagnostic only: the same cheap chain on the large scale against a long chain
    res['l_mcmc'] = {}
    for beta in BETAS:
        lam = s0 / beta; start = perturbed_map(L, sol, g_a / lam, g_x / lam, cfg['n_l'], rng); lt = lambda x: q_value(L, x, w, sc) / lam; row = dict()
        gain = lambda c: float((q_value(L, c, w, sc).mean() - m0) / s0)
        row['map'] = gain(start); row['mcmc12'] = gain(sam.run(start, 12, log_target=lt)[0]); row['long'] = gain(sam.run(start, cfg['sweeps'] * L.n, log_target=lt)[0])
        res['l_mcmc'][beta] = row
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--output', required=True)
    a = ap.parse_args(); cfg = json.loads(Path(a.config).read_text()); out = Path(a.output); out.mkdir(parents=True, exist_ok=False); t = time.time()
    resource.setrlimit(resource.RLIMIT_AS, (int(cfg['rlimit']), int(cfg['rlimit'])))
    src = hashlib.sha256(b''.join((HERE / f).read_bytes() for f in ('stage_a.py', 'sched_core.py'))).hexdigest()
    res = {1: phase1, 2: phase2, 'a1': phase_a1}[cfg['phase']](cfg, {})
    (out / 'result.json').write_text(json.dumps(dict(config=cfg, status='completed', source=src, host=platform.node(), pid=os.getpid(), sec=time.time() - t, **res), indent=1, default=float))
