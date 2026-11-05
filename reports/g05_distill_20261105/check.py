"""Pre-launch checks for g05.py against the verified numpy DP in routing_core.py."""
import numpy as np, torch, g05
from g05 import RR, marginals, walk, chain, Denoiser, Objective, T_STEPS, M
torch.manual_seed(0); gen = torch.Generator().manual_seed(0)
eta = torch.randn(3, T_STEPS, M, dtype=torch.float64)
mu_np, _ = RR.marginals(eta.numpy())
print('marginal vs numpy', float(np.abs(marginals(eta).numpy() - mu_np).max()))
e32 = eta.float(); s = walk(e32)
print('solver vs numpy value gap', float(np.abs(e32.gather(2, s[:, :, None]).sum((1, 2)).numpy() - np.take_along_axis(eta.numpy(), RR.solve(eta.numpy())[:, :, None], 2).sum((1, 2))).max()))
p = walk(torch.zeros(4000, T_STEPS, M), gen).numpy()
print('all sampled walks closed', bool(all(RR.start[RR.tail[x[0]]] and RR.start[RR.head[x[-1]]] and all(RR.head[x[i]] == RR.tail[x[i + 1]] for i in range(T_STEPS - 1)) for x in p[:500])))
freq = np.zeros((T_STEPS, M))
for t in range(T_STEPS): np.add.at(freq[t], p[:, t], 1 / len(p))
print('sample freq vs exact uniform mean, max err', float(np.abs(freq - RR.marginals(np.zeros((1, T_STEPS, M)))[0][0]).max()))
obj = Objective(0); x = torch.rand(2, T_STEPS, M) / 8; x.requires_grad_(True)
for arm in ('structured', 'ef', 'plain'):
    m = Denoiser(arm); y = torch.randn(4, T_STEPS, M); t = 4 * torch.rand(4, 1, 1)
    l = g05.batched_forward(m, y, t).sum(); l.backward()
    print(arm, 'grad finite', all(torch.isfinite(q.grad).all() for q in m.parameters()))
    q0 = obj(walk(torch.zeros(4096, T_STEPS, M), gen)); pz = chain(m, torch.zeros(1, T_STEPS, M), 1024, gen)
    print(arm, 'untrained chain Q vs uniform (se units)', round(float((obj(pz).mean() - q0.mean()) / (q0.std() / 1024 ** .5)), 2),
          'chain occupancy max err', round(float(np.abs(g05.RR_bits(pz).mean(0).numpy() - RR.marginals(np.zeros((1, T_STEPS, M)))[0][0]).max()), 3))
