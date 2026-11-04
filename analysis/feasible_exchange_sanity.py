"""Tiny exact checks for feasible CTMCs and Doob initial-law bookkeeping.

This enumerates a 42-action toy graph, not the production intervention task,
does not train an actor, and does not change any experiment configuration.
Run: .venv/bin/python analysis/feasible_exchange_sanity.py
"""

import itertools
import json
import math

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def enumerate_graph(n=6, budget=3):
    actions = np.asarray([bits for bits in itertools.product((0, 1), repeat=n)
                          if sum(bits) <= budget], dtype=np.int64)
    difference = np.abs(actions[:, None] - actions[None]).sum(-1)
    counts = actions.sum(-1)
    swap = (difference == 2) & (counts[:, None] == counts[None])
    adjacency = (difference == 1) | swap
    return actions, adjacency.astype(float), swap.astype(float)


def main():
    actions, adjacency, swap = enumerate_graph()
    n_states = len(actions)
    degree = adjacency.sum(-1)
    uniform = np.full(n_states, 1. / n_states)
    jump_chain = adjacency / degree[:, None]
    degree_prior = degree / degree.sum()
    q = adjacency - np.diag(degree)
    uniformized = np.eye(n_states) + q / degree.max()
    assert connected_components(csr_matrix(adjacency), directed=False)[0] == 1
    assert connected_components(csr_matrix(swap), directed=False)[0] == 4
    assert np.max(np.abs(uniform @ q)) < 1e-12
    assert np.max(np.abs(degree_prior @ jump_chain - degree_prior)) < 1e-12
    assert np.max(np.abs(uniform @ uniformized - uniform)) < 1e-12
    assert np.min(uniformized) >= 0

    horizon = 0.08
    transition = expm(horizon * q)
    terminal_weight = np.exp(1.7 * actions[:, 0] - 0.8 * actions[:, 1] + 0.3 * actions.sum(-1))
    h0 = transition @ terminal_weight
    transformed_transition = transition * terminal_weight[None] / h0[:, None]
    np.testing.assert_allclose(transformed_transition.sum(-1), 1., atol=1e-12)
    old_terminal = uniform @ transition
    target = old_terminal * terminal_weight
    target /= target.sum()
    incorrectly_fixed_initial = uniform @ transformed_transition
    tilted_initial = uniform * h0
    tilted_initial /= tilted_initial.sum()
    corrected = tilted_initial @ transformed_transition
    np.testing.assert_allclose(corrected, target, atol=1e-12)
    wrong_tv = 0.5 * np.abs(incorrectly_fixed_initial - target).sum()
    assert wrong_tv > 0.05

    anchor = int(np.flatnonzero(actions.sum(-1) == 0)[0])
    anchored_old = transition[anchor]
    anchored_target = anchored_old * terminal_weight / h0[anchor]
    np.testing.assert_allclose(transformed_transition[anchor], anchored_target, atol=1e-12)
    endpoint_kl = float(np.sum(anchored_target * np.log(anchored_target / anchored_old)))

    def instantaneous_path_kl(t):
        h = expm((horizon - t) * q) @ terminal_weight
        probability = expm(t * q)[anchor] * h / h0[anchor]
        ratio = h[None] / h[:, None]
        rate_cost = np.sum(adjacency * (ratio * np.log(ratio) - ratio + 1.), axis=1)
        return float(probability @ rate_cost)

    path_kl, integration_error = quad(instantaneous_path_kl, 0., horizon, epsabs=1e-10)
    assert abs(path_kl - endpoint_kl) < 1e-9

    # Production-size counts/degrees require no action enumeration.
    n, budget = 20, 10
    counts = [math.comb(n, k) for k in range(budget + 1)]
    degrees = [n + k * (n - k) if k < budget else k + k * (n - k)
               for k in range(budget + 1)]
    production_uniform_cardinality = np.asarray(counts, float) / sum(counts)
    production_neighbor_cardinality = np.asarray(counts, float) * degrees
    production_neighbor_cardinality /= production_neighbor_cardinality.sum()
    result = {
        "scope": "exact toy Markov-chain algebra; no RL or learned model",
        "toy_actions": n_states,
        "full_graph_components": 1,
        "swap_only_components": 4,
        "toy_degree_range": [int(degree.min()), int(degree.max())],
        "uniform_stationarity_residual_ctmc": float(np.max(np.abs(uniform @ q))),
        "uniform_stationarity_residual_neighbor_walk": float(np.max(np.abs(uniform @ jump_chain - uniform))),
        "doob_fixed_wrong_initial_tv": float(wrong_tv),
        "doob_corrected_initial_tv": float(0.5 * np.abs(corrected - target).sum()),
        "anchored_doob_terminal_error": float(np.max(np.abs(transformed_transition[anchor] - anchored_target))),
        "anchored_endpoint_kl": endpoint_kl,
        "anchored_path_kl_by_rate_integral": path_kl,
        "quadrature_error_estimate": integration_error,
        "production_action_count_n20_b10": sum(counts),
        "production_degrees_by_cardinality": degrees,
        "production_uniform_cardinality": production_uniform_cardinality.tolist(),
        "production_neighbor_walk_cardinality": production_neighbor_cardinality.tolist(),
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
