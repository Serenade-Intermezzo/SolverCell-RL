#!/usr/bin/env python3
"""Exact solver-support/prior audit and bounded, known-model one-step analysis.

No training, checkpoint, campaign configuration, or scientific source is changed.
The 40/20/10 seed-0 instance has 8 orthogonal equal-norm column groups. Enumerate
group-count effects, independently verify decoder support, and compute the exact
uniform-sphere prior by signs/orderings rather than Monte Carlo inversion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solvercell.environment import InterventionEnv, UPSTREAM_COMMIT
from tools.campaign import source_sha256


def stable_decode_group_scores(scores, inverse, budget):
    primitive_scores = np.asarray(scores)[inverse]
    order = np.argsort(-primitive_scores, kind="stable")[:budget]
    action = np.zeros(len(inverse), dtype=np.int64)
    action[order[primitive_scores[order] > 0]] = 1
    return action


def action_for_counts(counts, groups):
    action = np.zeros(sum(map(len, groups)), dtype=np.int64)
    for count, indices in zip(counts, groups):
        action[indices[:int(count)]] = 1
    return action


def classify(counts, sizes, groups, inverse, budget):
    total = int(sum(counts))
    partial = np.flatnonzero((counts > 0) & (counts < sizes))
    if (total < budget and len(partial) == 0) or (total == budget and len(partial) <= 1):
        return "positive_measure"
    if total == budget:
        # All partial groups must have the same positive cutoff score. Removing
        # full/empty groups from that tie leaves a stable global-index prefix.
        tied_indices = np.sort(np.concatenate([groups[g] for g in partial]))
        selected = tied_indices[:int(counts[partial].sum())]
        tied_counts = np.bincount(inverse[selected], minlength=len(sizes))
        if np.array_equal(tied_counts[partial], counts[partial]):
            return "tie_only"
    return "unreachable"


def exact_positive_prior(sizes, budget):
    """Uniform sphere -> iid Gaussian group-score signs and ordering.

    A prescribed ordered tuple of p positive groups has probability 1/(2^G p!).
    Integer numerators use the common denominator 2^G G!.
    """
    dimension = len(sizes)
    factorial = math.factorial(dimension)
    denominator = 2**dimension * factorial
    numerators = {}
    order_count = 0
    for length in range(dimension + 1):
        weight = factorial // math.factorial(length)
        for order in itertools.permutations(range(dimension), length):
            counts = [0] * dimension
            remaining = budget
            for group in order:
                amount = min(int(sizes[group]), remaining)
                counts[group] = amount
                remaining -= amount
                if remaining == 0:
                    break
            key = tuple(counts)
            numerators[key] = numerators.get(key, 0) + weight
            order_count += 1
    if sum(numerators.values()) != denominator:
        raise AssertionError("exact prior did not normalize")
    return numerators, denominator, order_count


def all_tie_reachable(inverse, budget):
    """Independent decoder enumeration: above/equal/below-cutoff levels suffice."""
    dimension = int(inverse.max()) + 1
    effects = set()
    for scores in itertools.product((-1, 1, 2), repeat=dimension):
        action = stable_decode_group_scores(scores, inverse, budget)
        effects.add(tuple(np.bincount(inverse, weights=action, minlength=dimension).astype(int)))
    return effects


def learned_late_states(path, env, sample_size, tail_size, seed):
    """Read completed-run replay only; sample known chronological valid rows."""
    import torch

    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = json.loads(path.with_name("config.json").read_text())
    metrics = json.loads(path.with_name("metrics.json").read_text())
    worker = json.loads(path.with_name("worker.json").read_text())
    if config != checkpoint["config"] or config != worker["config"]:
        raise AssertionError("learned-state checkpoint/config/worker mismatch")
    if metrics["status"] != "complete" or checkpoint["train_steps"] != config["train_steps"]:
        raise AssertionError("learned-state source is not a completed training run")
    if len({checkpoint["source_sha256"], metrics["source_sha256"], worker["source_sha256"]}) != 1:
        raise AssertionError("learned-state checkpoint/metrics/worker source mismatch")
    if checkpoint["instance_hash"] != env.instance_hash:
        raise AssertionError("learned-state checkpoint has a different instance")
    total = checkpoint["env_steps"]
    if total != metrics["env_steps"] or total != config["warmup_steps"] + config["train_steps"]:
        raise AssertionError("learned-state collection counters disagree")
    replay = checkpoint["replay"]
    capacity, size, position = (int(replay[key]) for key in ("capacity", "size", "pos"))
    if size != min(total, capacity) or position != total % capacity:
        raise AssertionError("replay chronology is inconsistent with collection count")
    if not 1 <= sample_size <= tail_size <= min(size, config["train_steps"]):
        raise ValueError("learned-state sample must fit within the valid late training tail")
    eligible = (position - tail_size + np.arange(tail_size)) % capacity
    offsets = np.random.default_rng(seed).choice(tail_size, size=sample_size, replace=False)
    rows = eligible[offsets]
    states = np.asarray(replay["data"]["obs"])[rows].copy()
    if (not np.isfinite(states).all() or not np.equal(states, np.round(states)).all()
            or states.shape != (sample_size, env.n_arms)
            or (states < 0).any() or (states >= env.n_states).any()):
        raise AssertionError("learned-state observations are not valid environment states")
    ids = [f"learned_quotient_point_envstep{total-tail_size+int(offset)+1}_physicalrow{int(row)}"
           for offset, row in zip(offsets, rows)]
    provenance = {
        "checkpoint_path": str(path), "checkpoint_sha256": digest.hexdigest(),
        "checkpoint_source_sha256": checkpoint["source_sha256"],
        "instance_hash": checkpoint["instance_hash"], "method": config["method"], "seed": config["seed"],
        "train_steps": checkpoint["train_steps"], "warmup_steps": config["warmup_steps"], "env_steps": total,
        "replay_capacity": capacity, "replay_size": size, "replay_pos": position,
        "eligible_tail_size": tail_size, "eligible_env_steps_inclusive": [total-tail_size+1, total],
        "eligible_physical_rows": eligible.tolist(), "sample_size": sample_size,
        "sample_seed": seed, "sampling": "uniform without replacement among tail transitions",
        "sampled_physical_rows": rows.tolist(),
        "scope": "historical late training visits, not fresh final-policy rollouts or independent state samples",
    }
    return states.astype(np.int64), ids, provenance


def expected_reward_matrix(env, effects, unique_columns, states):
    inputs = env.theta_p + effects @ unique_columns.T
    intercept = env.a_vals / (1 + np.exp(env.x0_vals))
    probabilities = 0.98 * (env.a_vals / (1 + np.exp(-(env.b_vals * inputs - env.x0_vals))) - intercept) + 0.01
    ids = np.arange(env.n_arms)
    states = np.asarray(states, dtype=np.int64)
    up = env.state_r[ids[None], np.minimum(states + 1, env.n_states - 1)]
    down = env.state_r[ids[None], np.maximum(states - 1, 0)]
    return probabilities @ (up - down).T + down.sum(-1)[None]


def independent_group_dp(env, sizes, groups, unique_columns, states):
    """Known-model multiple-choice knapsack DP, independent of effect enumeration."""
    states = np.asarray(states, dtype=np.int64)
    ids = np.arange(env.n_arms)
    up = env.state_r[ids[None], np.minimum(states + 1, env.n_states - 1)]
    down = env.state_r[ids[None], np.maximum(states - 1, 0)]
    support = unique_columns != 0
    if np.any(support.sum(-1) > 1):
        raise AssertionError("separable DP requires disjoint group supports")
    inactive = ~support.any(-1)
    base_prob = env.transition_probabilities(np.zeros(env.n_actions))
    base = (down[:, inactive] + base_prob[inactive] * (up - down)[:, inactive]).sum(-1)
    dp = np.full((len(states), env.budget + 1), -np.inf)
    dp[:, 0] = base
    for group, size in enumerate(sizes):
        affected = support[:, group]
        new = np.full_like(dp, -np.inf)
        for count in range(int(size) + 1):
            action = np.zeros(env.n_actions)
            action[groups[group][:count]] = 1
            prob = env.transition_probabilities(action)[affected]
            reward = (down[:, affected] + prob * (up - down)[:, affected]).sum(-1)
            for used in range(count, env.budget + 1):
                new[:, used] = np.maximum(new[:, used], dp[:, used-count] + reward)
        dp = new
    return dp.max(-1)


def numeric_summary(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "max": float(values.max()), "min": float(values.min())}


def evaluate_states(env, effects, labels, groups, unique_columns, states, state_ids, source):
    rewards = expected_reward_matrix(env, effects, unique_columns, states)
    positive = np.flatnonzero(labels == "positive_measure")
    canonical = np.flatnonzero(labels != "unreachable")
    index_all = rewards.argmax(0)
    index_positive = positive[rewards[positive].argmax(0)]
    index_canonical = canonical[rewards[canonical].argmax(0)]
    columns = np.arange(len(states))
    best = rewards[index_all, columns]
    best_positive = rewards[index_positive, columns]
    best_canonical = rewards[index_canonical, columns]
    dp = independent_group_dp(env, np.asarray(list(map(len, groups))), groups, unique_columns, states)
    np.testing.assert_allclose(dp, best, atol=1e-11, rtol=1e-12)
    # Independently call the ordinary scalar environment for optimized and
    # sampled actions, rather than validating vectorization against itself.
    oracle_error = 0.
    for column in np.linspace(0, len(states)-1, min(len(states), 64), dtype=int):
        for row in (index_all[column], index_positive[column], int(column * 149 % len(effects))):
            actual = env.expected_reward(action_for_counts(effects[row], groups), states[column])
            oracle_error = max(oracle_error, abs(actual - rewards[row, column]))
    if oracle_error > 1e-10:
        raise AssertionError(f"scalar environment disagreement: {oracle_error}")
    gap = best - best_positive
    abs_gap = best - best_canonical
    worst = int(gap.argmax())
    records = []
    for i, state in enumerate(states):
        records.append({"id": state_ids[i], "source": source, "state": state.astype(int).tolist(),
                        "best_all": float(best[i]), "best_positive_measure": float(best_positive[i]),
                        "best_any_canonical": float(best_canonical[i]), "support_gap": float(gap[i]),
                        "canonical_gap": float(abs_gap[i]), "best_all_counts": effects[index_all[i]].tolist(),
                        "best_all_category": str(labels[index_all[i]]),
                        "best_positive_counts": effects[index_positive[i]].tolist()})
    example = dict(records[worst])
    example["best_all_primitive_action"] = action_for_counts(effects[index_all[worst]], groups).tolist()
    example["best_positive_primitive_action"] = action_for_counts(effects[index_positive[worst]], groups).tolist()
    summary = {"state_source": source, "states": len(states),
               "best_all_expected_one_step_reward": numeric_summary(best),
               "best_positive_expected_one_step_reward": numeric_summary(best_positive),
               "positive_measure_support_gap": numeric_summary(gap),
               "any_canonical_support_gap": numeric_summary(abs_gap),
               "states_with_positive_measure_gap": int((gap > 1e-10).sum()),
               "states_with_any_canonical_gap": int((abs_gap > 1e-10).sum()),
               "best_all_argmax_categories": {label: int((labels[index_all] == label).sum()) for label in sorted(set(labels))},
               "dp_max_abs_error": float(np.max(np.abs(dp-best))), "scalar_oracle_max_abs_error": oracle_error,
               "largest_positive_measure_gap_example": example}
    return summary, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "reports/solver_support_audit"))
    parser.add_argument("--instance-seed", type=int, default=0)
    parser.add_argument("--synthetic-states", type=int, default=128)
    parser.add_argument("--state-seed", type=int, default=882100)
    parser.add_argument("--random-rollouts", type=int, default=20)
    parser.add_argument("--rollout-seed-start", type=int, default=882200)
    parser.add_argument("--learned-checkpoint", default=str(ROOT / "outputs/native_revision_20261104/quotient_point/checkpoint.pt"))
    parser.add_argument("--learned-state-sample", type=int, default=512)
    parser.add_argument("--learned-tail-size", type=int, default=1000)
    parser.add_argument("--learned-state-seed", type=int, default=882300)
    args = parser.parse_args()
    if args.synthetic_states < 1 or args.random_rollouts < 1:
        raise ValueError("state and rollout budgets must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    before = source_sha256(ROOT)
    env = InterventionEnv(instance_seed=args.instance_seed)
    rows, inverse = np.unique(env.weights_p.T, axis=0, return_inverse=True)
    unique_columns = rows.T
    sizes = np.bincount(inverse)
    groups = [np.flatnonzero(inverse == group) for group in range(len(sizes))]
    gram = rows @ rows.T
    if not np.array_equal(gram, np.eye(len(rows)) * gram[0, 0]) or gram[0, 0] <= 0:
        raise ValueError("exact prior derivation requires orthogonal, equally scaled unique columns")
    effects = np.asarray([counts for counts in itertools.product(*(range(int(size)+1) for size in sizes))
                          if sum(counts) <= env.budget], dtype=np.int64)
    labels = np.asarray([classify(counts, sizes, groups, inverse, env.budget) for counts in effects])
    prior, denominator, ordered_subsets = exact_positive_prior(sizes, env.budget)
    classified_positive = {tuple(counts) for counts in effects[labels == "positive_measure"]}
    if set(prior) != classified_positive:
        raise AssertionError("strict ordering enumeration disagrees with support classification")
    weak_support = all_tie_reachable(inverse, env.budget)
    if weak_support != {tuple(counts) for counts in effects[labels != "unreachable"]}:
        raise AssertionError("three-level score enumeration disagrees with tie classification")
    multiplicities = [math.prod(math.comb(int(size), int(count)) for count, size in zip(counts, sizes)) for counts in effects]
    if sum(multiplicities) != sum(math.comb(env.n_actions, count) for count in range(env.budget + 1)):
        raise AssertionError("group counts do not partition the primitive feasible actions")
    examples = {}
    for label in sorted(set(labels)):
        possible = np.flatnonzero(labels == label)
        if label == "positive_measure":
            possible = possible[effects[possible].sum(-1) > 0]
        index = int(possible[0])
        counts = effects[index]
        example = {"counts": counts.tolist(), "primitive_action": action_for_counts(counts, groups).tolist(),
                   "budget_used": int(counts.sum())}
        if label != "unreachable":
            scores = np.where(counts == 0, -1., np.where(counts == sizes, 2., 1.))
            decoded = stable_decode_group_scores(scores, inverse, env.budget)
            np.testing.assert_array_equal(decoded, action_for_counts(counts, groups))
            coefficient = unique_columns @ (scores / np.diag(gram))
            np.testing.assert_allclose(coefficient @ unique_columns, scores, atol=1e-14, rtol=1e-14)
            example.update(witness_group_scores=scores.tolist(), witness_cost=coefficient.tolist())
        examples[label] = example
    by_budget = {}
    for total in range(env.budget + 1):
        by_budget[str(total)] = {label: int(((effects.sum(-1) == total) & (labels == label)).sum()) for label in sorted(set(labels))}
    probabilities = np.array(list(prior.values()), dtype=float) / denominator
    summary = {
        "upstream_commit": UPSTREAM_COMMIT, "science_source_sha256": before,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "instance_hash": env.instance_hash, "instance_seed": args.instance_seed,
        "shape": list(env.weights_p.shape), "budget": env.budget,
        "group_sizes": sizes.tolist(), "group_primitive_indices": [indices.tolist() for indices in groups],
        "group_squared_norms": np.diag(gram).tolist(), "rank": int(np.linalg.matrix_rank(env.weights_p)),
        "all_unbudgeted_group_counts": int(np.prod(sizes+1)), "feasible_effects": len(effects),
        "feasible_primitive_actions": sum(multiplicities),
        "categories": {label: int((labels == label).sum()) for label in sorted(set(labels))},
        "by_budget": by_budget, "examples": examples,
        "verification": {"strict_ordered_positive_subsets": ordered_subsets,
                         "three_level_score_assignments": 3**len(sizes),
                         "positive_support_matches_strict_order_enumeration": True,
                         "canonical_support_matches_three_level_decoder_enumeration": True},
        "exact_uniform_sphere_prior": {
            "scope": "canonical decoded group-count effects; same for uniform full or quotient sphere",
            "common_denominator": denominator, "numerator_sum": sum(prior.values()),
            "positive_support_size": len(prior), "min_probability": float(probabilities.min()),
            "max_probability": float(probabilities.max()), "max_to_min_ratio": float(probabilities.max()/probabilities.min()),
            "entropy_nats": float(-(probabilities*np.log(probabilities)).sum()),
            "uniform_on_positive_support_entropy_nats": math.log(len(prior)),
        },
    }
    synthetic = np.random.default_rng(args.state_seed).integers(0, env.n_states, size=(args.synthetic_states, env.n_arms))
    random_states, random_ids = [], []
    for episode in range(args.random_rollouts):
        seed = args.rollout_seed_start + episode
        rollout = env.clone(seed=seed)
        obs = rollout.reset(seed=seed)
        action_rng = np.random.default_rng(seed + 1000000)
        for step in range(env.horizon):
            random_states.append(obs.copy())
            random_ids.append(f"random_rollout_seed{seed}_step{step}")
            obs, _, _, _ = rollout.step(rollout.random_action(action_rng))
    learned, learned_ids, learned_provenance = learned_late_states(
        args.learned_checkpoint, env, args.learned_state_sample, args.learned_tail_size, args.learned_state_seed)
    summaries, all_records = [], []
    for states, ids, source in (
        (synthetic, [f"synthetic_{index}" for index in range(len(synthetic))], "uniform_synthetic_four_state_vectors"),
        (np.asarray(random_states), random_ids, "visited_predecision_states_of_uniform_budget_random_policy"),
        (learned, learned_ids, "late_training_visits_of_completed_native_quotient_point_seed0"),
    ):
        stats, records = evaluate_states(env, effects, labels, groups, unique_columns, states, ids, source)
        summaries.append(stats)
        all_records.extend(records)
    summary["one_step_analysis"] = {
        "synthetic_state_seed": args.state_seed, "rollout_seed_start": args.rollout_seed_start,
        "random_rollouts": args.random_rollouts, "horizon": env.horizon,
        "learned_state_provenance": learned_provenance,
        "scope": "Exact known-model expected immediate reward on fixed states; no training, long-term optimality, or RL performance claim",
        "datasets": summaries,
    }
    after = source_sha256(ROOT)
    if before != after:
        raise RuntimeError("scientific source changed during analysis")
    summary["science_source_unchanged"] = True
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output / "state_results.json").write_text(json.dumps(all_records, indent=2, allow_nan=False) + "\n")
    with (output / "group_effects.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([*(f"r{group}" for group in range(len(sizes))), "budget_used", "category",
                         "primitive_action_multiplicity", "prior_numerator", "prior_denominator", "prior_probability"])
        for counts, label, multiplicity in zip(effects, labels, multiplicities):
            numerator = prior.get(tuple(counts), 0)
            writer.writerow([*counts, int(counts.sum()), label, multiplicity, numerator, denominator, numerator/denominator])
    print(json.dumps({"output": str(output.resolve()), "categories": summary["categories"],
                      "prior": summary["exact_uniform_sphere_prior"],
                      "one_step": [{"source": row["state_source"], "states": row["states"],
                                    "gap_states": row["states_with_positive_measure_gap"],
                                    "gap": row["positive_measure_support_gap"],
                                    "max_example": row["largest_positive_measure_gap_example"]}
                                   for row in summaries]}, allow_nan=False))


if __name__ == "__main__":
    main()
