#!/usr/bin/env python3
"""Frozen-oracle checks for exact Gaussian positive-topk branch integration.

No training: the terminal value is exact expected immediate environment reward.
Compare complete mean-score vectors, including an equal-two-Q-query baseline.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.solver_support_audit import expected_reward_matrix
from solvercell.environment import InterventionEnv
from tools.campaign import source_sha256


def decode(costs, budget):
    order = np.argsort(-costs, axis=-1, kind="stable")[:, :budget]
    result = np.zeros_like(costs)
    np.put_along_axis(result, order, (np.take_along_axis(costs, order, axis=1) > 0).astype(float), axis=1)
    return result


def branches(costs, coordinate, budget, sigma):
    outside = np.delete(costs, coordinate, axis=1)
    threshold = np.maximum(0., np.sort(outside, axis=1)[:, -budget])
    below, above = costs.copy(), costs.copy()
    below[:, coordinate], above[:, coordinate] = threshold-sigma, threshold+sigma
    return threshold, decode(below, budget), decode(above, budget)


def stats(vectors):
    return {"mean": vectors.mean(0).tolist(), "variance_diagonal": vectors.var(0, ddof=1).tolist(),
            "variance_trace": float(vectors.var(0, ddof=1).sum())}


def sign_toy():
    gradient = 1/(2*math.sqrt(math.pi))
    raw = .25-1/(4*math.pi)
    rb = 1/12+1/(2*math.pi*math.sqrt(3))-1/(4*math.pi)
    phi = lambda x: math.exp(-x*x/2)/math.sqrt(2*math.pi)
    qm = quad(lambda x: x*(ndtr(x)-.5)*phi(x), -12, 12, epsabs=1e-13)[0]
    qv = quad(lambda x: (x*(ndtr(x)-.5))**2*phi(x), -12, 12, epsabs=1e-13)[0]-qm**2
    assert abs(qm-gradient) < 1e-12 and abs(qv-rb) < 1e-12
    return {"scope": "analytic two-step sign-decoder example; not the benchmark or RL",
            "theta": 0., "sigma_1": 1., "sigma_2": 1., "objective": .5,
            "gradient": gradient, "centered_terminal_score_variance": raw,
            "continuation_score_variance": rb, "variance_ratio": rb/raw,
            "quadrature_mean_error": abs(qm-gradient), "quadrature_variance_error": abs(qv-rb)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "reports/continuation_branch_audit"))
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--pilot-samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=882400)
    args = parser.parse_args()
    if args.samples < 2 or args.pilot_samples < 1:
        raise ValueError("positive sample budgets required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    before = source_sha256(ROOT)
    env = InterventionEnv(instance_seed=0)
    support_path = ROOT / "reports/solver_support_audit/state_results.json"
    source_records = json.loads(support_path.read_text())
    learned = [r for r in source_records if r["source"] == "late_training_visits_of_completed_native_quotient_point_seed0"]
    selected = [learned[index] for index in (0, 200, 400)]
    states = np.array([row["state"] for row in selected])
    sigma, dimension, mean = .5, env.n_actions, np.zeros(env.n_actions)
    coordinates = [0, 5, 10, 15]
    rng = np.random.default_rng(args.seed)
    reward = lambda actions: expected_reward_matrix(env, actions, env.weights_p, states)
    pilot_actions = decode(rng.normal(mean, sigma, (args.pilot_samples, dimension)), env.budget)
    baseline = reward(pilot_actions).mean(0)
    rows, records = [], []
    quadrature_max_error = finite_difference_max_error = oracle_max_error = 0.
    for coordinate in coordinates:
        # Independent draws for the one-/two-query SF comparator and RB method.
        sf_costs = rng.normal(mean, sigma, (2*args.samples, dimension))
        sf_actions = decode(sf_costs, env.budget)
        sf_rewards = reward(sf_actions)
        rb_costs = rng.normal(mean, sigma, (args.samples, dimension))
        threshold, minus, plus = branches(rb_costs, coordinate, env.budget, sigma)
        qminus, qplus = reward(minus), reward(plus)
        h = (mean[coordinate]-threshold)/sigma
        probability, density = ndtr(h), np.exp(-.5*h*h)/math.sqrt(2*math.pi)
        expected = (1-probability[:, None])*qminus+probability[:, None]*qplus
        integrated_coordinate = density[:, None]/sigma*(qplus-qminus)

        # Independently check actual decoder branches and the scalar reward API.
        import torch
        from solvercell.geometry import decode as reference_decode
        np.testing.assert_array_equal(sf_actions[:256], reference_decode(torch.tensor(sf_costs[:256]), env.budget).numpy())
        for context in range(3):
            for offset in (-4., -.1, .1, 4.):
                costs = rb_costs[context].copy()
                costs[coordinate] = threshold[context]+offset*sigma
                actual = reference_decode(torch.tensor(costs), env.budget).numpy()
                np.testing.assert_array_equal(actual, minus[context] if offset < 0 else plus[context])
            for state_index, state in enumerate(states):
                qm, qp = qminus[context, state_index], qplus[context, state_index]
                b, threshold_z = baseline[state_index], -h[context]
                phi = lambda z: math.exp(-z*z/2)/math.sqrt(2*math.pi)
                # Score-function integral over the two exact solver intervals.
                numerical = (quad(lambda z: z/sigma*(qm-b)*phi(z), -np.inf, threshold_z, epsabs=1e-12)[0]
                             + quad(lambda z: z/sigma*(qp-b)*phi(z), threshold_z, np.inf, epsabs=1e-12)[0])
                quadrature_max_error = max(quadrature_max_error, abs(numerical-integrated_coordinate[context, state_index]))
                step = 1e-5
                fd = (ndtr(h[context]+step/sigma)-ndtr(h[context]-step/sigma))*(qp-qm)/(2*step)
                finite_difference_max_error = max(finite_difference_max_error, abs(fd-integrated_coordinate[context, state_index]))
                for action, value in ((minus[context], qm), (plus[context], qp)):
                    oracle_max_error = max(oracle_max_error, abs(env.expected_reward(action, state)-value))

        for state_index, source in enumerate(selected):
            score = (sf_costs-mean)/sigma**2
            raw = score*(sf_rewards[:, state_index]-baseline[state_index])[:, None]
            sf1 = raw[::2]
            sf2 = raw.reshape(args.samples, 2, dimension).mean(1)
            rb = (rb_costs-mean)/sigma**2*(expected[:, state_index]-baseline[state_index])[:, None]
            rb[:, coordinate] = integrated_coordinate[:, state_index]
            measures = {"sf_one_query": stats(sf1), "sf_two_independent_queries": stats(sf2),
                        "rb_full_vector_two_queries": stats(rb)}
            mean_delta = rb.mean(0)-sf2.mean(0)
            delta_se = np.sqrt((rb.var(0, ddof=1)+sf2.var(0, ddof=1))/args.samples)
            ratio = measures["rb_full_vector_two_queries"]["variance_trace"]/measures["sf_two_independent_queries"]["variance_trace"]
            row = {"state_id": source["id"], "coordinate": coordinate, "samples_per_estimator": args.samples,
                   "baseline": float(baseline[state_index]),
                   "sf1_variance_trace": measures["sf_one_query"]["variance_trace"],
                   "sf2_variance_trace": measures["sf_two_independent_queries"]["variance_trace"],
                   "rb2_variance_trace": measures["rb_full_vector_two_queries"]["variance_trace"],
                   "rb2_over_sf2_variance_trace": ratio,
                   "mean_difference_l2": float(np.linalg.norm(mean_delta)),
                   "max_abs_component_mean_difference_over_se": float(np.max(np.abs(mean_delta)/delta_se))}
            rows.append(row)
            records.append({**row, "state": source["state"], "estimators": measures})
    assert quadrature_max_error < 1e-10 and finite_difference_max_error < 1e-8 and oracle_max_error < 1e-10
    after = source_sha256(ROOT)
    if after != before:
        raise RuntimeError("science source changed during diagnostic")
    summary = {
        "scope": "frozen exact myopic action-value oracle; no learned critic, policy optimization, or RL result",
        "science_source_sha256": before, "science_source_unchanged": True,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "state_source_path": str(support_path), "state_source_sha256": hashlib.sha256(support_path.read_bytes()).hexdigest(),
        "selected_learned_state_indices": [0, 200, 400], "state_ids": [row["id"] for row in selected],
        "instance_hash": env.instance_hash, "mean": mean.tolist(), "fixed_sigma": sigma,
        "decoder": "primitive20 strictly positive up-to10 topk; final Gaussian stays enabled",
        "kernel_scope": "declared test kernel; not the trained policy's actual terminal kernel",
        "coordinates": coordinates, "seed": args.seed, "samples_per_estimator": args.samples,
        "pilot_samples": args.pilot_samples, "baseline": "shared state-dependent constant from independent pilot; same for all estimators",
        "logical_Q_queries_per_estimate": {"sf_one_query": 1, "sf_two_independent_queries": 2, "rb_full_vector_two_queries": 2},
        "gradient_scope": "full20 mean-score vector; selected coordinate analytic, all other score coordinates retained",
        "sign_toy": sign_toy(),
        "checks": {"conditional_score_quadrature_max_abs_error": quadrature_max_error,
                   "conditional_mean_finite_difference_max_abs_error": finite_difference_max_error,
                   "scalar_environment_oracle_max_abs_error": oracle_max_error,
                   "decoder_branch_and_torch_reference_checks": True},
        "comparison": {"cases": len(rows), "rb2_trace_lower_than_sf2_cases": sum(row["rb2_over_sf2_variance_trace"] < 1 for row in rows),
                       "rb2_over_sf2_trace_ratio_min": min(row["rb2_over_sf2_variance_trace"] for row in rows),
                       "rb2_over_sf2_trace_ratio_max": max(row["rb2_over_sf2_variance_trace"] for row in rows),
                       "max_abs_component_mean_difference_over_se": max(row["max_abs_component_mean_difference_over_se"] for row in rows)},
        "results": records,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    with (output / "variance_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"comparison": summary["comparison"], "checks": summary["checks"], "rows": rows}, allow_nan=False))


if __name__ == "__main__":
    main()
