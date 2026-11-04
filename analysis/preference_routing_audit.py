"""Read-only routing interface check on the pinned released graph; no training."""
import ast
import hashlib
import itertools
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor/lsflow'


def shortest_closed_walk(required, distances, source):
    """Exact metric TSP; permits graph vertex repetition, relaxing simple cycles."""
    nodes = sorted(set(required) - {source})
    if not nodes:
        return 0
    n = len(nodes)
    table = np.full((1 << n, n), np.inf)
    for j, node in enumerate(nodes):
        table[1 << j, j] = distances[source][node]
    for mask in range(1, 1 << n):
        for j in range(n):
            if not mask & (1 << j):
                continue
            previous = mask ^ (1 << j)
            if not previous:
                continue
            table[mask, j] = min(table[previous, k] + distances[nodes[k]][nodes[j]]
                                 for k in range(n) if previous & (1 << k))
    return int(min(table[-1, j] + distances[nodes[j]][source] for j in range(n)))


def main():
    source_path = VENDOR / 'approximator/routing_approximator.py'
    parsed = ast.parse(source_path.read_text())
    cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef)
               and n.name == 'RoutingRmabApproximator')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'solve_from_coeffs')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_path), 'exec'), namespace)
    # This trusted pickle is part of the pinned public benchmark source.
    graph_path = VENDOR / 'environment/network/tube_network.pkl'
    with graph_path.open('rb') as handle:
        graph = pickle.load(handle)['G[40]']
    assert not graph.is_directed() and nx.is_connected(graph)
    distances = dict(nx.all_pairs_shortest_path_length(graph))
    source, budget, seed = 10, 10, 20261104
    # Independent small brute-force check of the subset DP.
    check_nodes = [1, 7, 14, 28]
    brute = min(sum(distances[a][b] for a, b in zip([source] + list(p), list(p) + [source]))
                for p in itertools.permutations(check_nodes))
    assert shortest_closed_walk(check_nodes, distances, source) == brute
    dummy = SimpleNamespace(rmab=SimpleNamespace(action_dim=40, n_arms=40, budget=budget))
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(32):
        cost = rng.normal(size=40)
        action = namespace['solve_from_coeffs'](dummy, cost)
        selected = np.flatnonzero(action).tolist()
        assert len(selected) == budget
        length = shortest_closed_walk(selected, distances, source)
        rows.append({'index': index, 'coefficients': cost.tolist(), 'selected_nodes': selected,
                     'minimum_closed_walk_edges_allowing_repeats': length,
                     'exceeds_maximum_20_edges_even_under_relaxation': length > 2 * budget})
    result = {
        'scope': 'pinned solve_from_coeffs on released G[40]; no RL, reward or route-objective extension',
        'source_file': str(source_path.relative_to(ROOT)),
        'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'graph_file': str(graph_path.relative_to(ROOT)),
        'graph_sha256': hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        'graph_nodes': graph.number_of_nodes(), 'graph_edges': graph.number_of_edges(),
        'source_node': source, 'budget': budget, 'maximum_route_edges': 2 * budget,
        'seed': seed, 'samples': len(rows),
        'provably_infeasible_node_sets': sum(r['exceeds_maximum_20_edges_even_under_relaxation'] for r in rows),
        'max_minimum_closed_walk_edges': max(r['minimum_closed_walk_edges_allowing_repeats'] for r in rows),
        'interpretation': 'A lower-bound walk exceeding 20 proves infeasibility. A walk at most20 does not certify a simple cycle.',
        'small_dp_matches_independent_permutation_enumeration': True,
        'rows': rows,
    }
    output = ROOT / 'reports/preference_benchmark_audit'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'routing_interface.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
