"""Prototype Style-Routing action set on the 40-station tube graph used by the released benchmark.

An action is a T-step closed walk on a time-expanded (station, line) graph. Every action activates exactly T
arc-time bits, so e(a)=2a-1 has constant norm. The linear solver, the exponential-family mean map, log Z and exact
sampling are all dynamic programs on this layered DAG. Declared extension, not an upstream benchmark.
"""
import csv
from pathlib import Path
import numpy as np

NAMES = ("Aldgate|Tottenham Court Road|Angel|Kennington|Tower Hill|Bank|Barbican|King's Cross St. Pancras|"
         "Lambeth North|Warren Street|Waterloo|Leicester Square|Blackfriars|Liverpool Street|Westminster|"
         "London Bridge|Borough|Bond Street|Mansion House|Monument|Moorgate|Cannon Street|Chancery Lane|"
         "Charing Cross|Covent Garden|Old Street|Oxford Circus|Picadilly Circus|Elephant & Castle|Embankment|"
         "Euston|Euston Square|Farringdon|Russell Square|Goodge Street|Southwark|Green Park|St. Paul's|Holborn|"
         "Temple").split('|')
NETWORK = Path(__file__).resolve().parents[2] / 'vendor/lsflow/environment/network'


class Routes:
    def __init__(self, T=20, source='Waterloo'):
        st = {r['id']: r for r in csv.DictReader(open(NETWORK / 'london_stations.csv'))}
        ids = {i: NAMES.index(r['name']) for i, r in st.items() if r['name'] in NAMES}
        self.n = len(NAMES); self.T = T; self.src = NAMES.index(source)
        self.crowd = np.zeros(self.n)            # proxy only: number of lines serving the station
        for i, j in ids.items(): self.crowd[j] = float(st[i]['total_lines'])
        edges = []
        for r in csv.DictReader(open(NETWORK / 'london_connections.csv')):
            if r['station1'] in ids and r['station2'] in ids:
                u, v = ids[r['station1']], ids[r['station2']]
                edges += [(u, v, int(r['line']), float(r['time'])), (v, u, int(r['line']), float(r['time']))]
        self.undirected = len({frozenset(e[:2]) for e in edges})
        lines = {}
        for u, v, l, _ in edges:
            lines.setdefault(u, set()).add(l); lines.setdefault(v, set()).add(l)
        states = [(u, l) for u in sorted(lines) for l in sorted(lines[u])]
        idx = {s: i for i, s in enumerate(states)}
        arcs = []                                  # tail, head, head station, minutes, transfer flag
        for (u, l0) in states:
            for u2, v, l, tm in edges:
                if u2 == u: arcs.append((idx[(u, l0)], idx[(v, l)], v, tm, float(l != l0)))
            if u == self.src: arcs.append((idx[(u, l0)], idx[(u, l0)], u, 0., 0.))   # waiting at the depot ends early
        arcs.sort(key=lambda a: a[1])
        A = np.array(arcs)
        self.tail, self.head, self.node = A[:, 0].astype(int), A[:, 1].astype(int), A[:, 2].astype(int)
        self.minutes, self.transfer = A[:, 3], A[:, 4]
        self.states = states; self.S = len(states); self.m = len(arcs)
        self.dim = self.m * T
        self.groups = np.searchsorted(self.head, np.arange(self.S))
        assert len(set(self.head)) == self.S
        self.start = np.array([u == self.src for u, _ in states])
        self.out = [np.where(self.tail == s)[0] for s in range(self.S)]

    def _reduce(self, vals, reducer):
        return reducer.reduceat(vals, self.groups, axis=1)

    def backward(self, eta, reducer=np.logaddexp):
        # eta: (b,T,m). beta[t][b,s]=log-sum (or max) over completions from s at layer t
        b = eta.shape[0]; beta = np.full((self.T + 1, b, self.S), -np.inf); beta[self.T][:, self.start] = 0
        for t in range(self.T - 1, -1, -1):
            vals = eta[:, t] + beta[t + 1][:, self.head]
            order = np.argsort(self.tail, kind='stable'); starts = np.searchsorted(self.tail[order], np.arange(self.S))
            beta[t] = reducer.reduceat(vals[:, order], starts, axis=1)
        return beta

    def log_z(self, eta):
        beta = self.backward(eta)
        return np.logaddexp.reduce(beta[0][:, self.start], axis=1) - 0.0

    def marginals(self, eta):
        # initial line choice is uniform by construction: all line states at the depot are valid starts
        b = eta.shape[0]; beta = self.backward(eta)
        alpha = np.full((b, self.S), -np.inf); alpha[:, self.start] = 0
        logZ = np.logaddexp.reduce(beta[0][:, self.start], axis=1)
        mu = np.zeros_like(eta)
        for t in range(self.T):
            vals = alpha[:, self.tail] + eta[:, t]
            mu[:, t] = np.exp(vals + beta[t + 1][:, self.head] - logZ[:, None])
            alpha = self._reduce(vals, np.logaddexp)
        return mu, logZ

    def _walk(self, eta, beta, rng, greedy):
        b = eta.shape[0]; path = np.zeros((b, self.T), dtype=int); ix = np.arange(b)
        score0 = beta[0].copy(); score0[:, ~self.start] = -np.inf
        if not greedy: score0 = score0 + rng.gumbel(size=score0.shape)
        s = score0.argmax(1)
        for t in range(self.T):
            sc = eta[:, t] + beta[t + 1][:, self.head]
            sc = np.where(self.tail[None, :] == s[:, None], sc, -np.inf)
            if not greedy: sc = sc + rng.gumbel(size=sc.shape)
            path[:, t] = sc.argmax(1); s = self.head[path[:, t]]
        return path

    def sample(self, eta, rng):
        return self._walk(eta, self.backward(eta), rng, False)

    def solve(self, eta):
        """Exact linear solver: argmax <eta,a> over closed walks. Polynomial; no MIP."""
        return self._walk(eta, self.backward(eta, np.maximum), None, True)

    def bits(self, path):
        a = np.zeros((path.shape[0], self.T, self.m))
        np.put_along_axis(a, path[:, :, None], 1, 2)
        return a

    def features(self, path, value):
        """value: per-station service prize. Returns service, minutes, transfers, crowding (raw, per route)."""
        b = path.shape[0]; nodes = self.node[path]
        seen = np.zeros((b, self.n)); np.put_along_axis(seen, nodes, 1, 1)
        moved = self.minutes[path] > 0
        return np.stack([seen @ value, self.minutes[path].sum(1), self.transfer[path].sum(1),
                         (self.crowd[nodes] * moved).sum(1)], 1)

    def service_grad(self, mu, value):
        """Multilinear extension of coverage: d/dmu_arc,t of sum_j v_j(1-prod_t(1-P(at j at t)))."""
        b = mu.shape[0]; occ = np.zeros((b, self.T, self.n))
        for t in range(self.T):
            np.add.at(occ[:, t], (slice(None), self.node), mu[:, t])
        occ = np.clip(occ, 0, 1 - 1e-9)
        miss = np.log1p(-occ).sum(1, keepdims=True) - np.log1p(-occ)
        g = value[None, None, :] * np.exp(miss)
        return g[:, :, self.node]
