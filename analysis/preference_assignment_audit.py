"""Reproduce the source-grounded Assignment utility/resource conflict witness.

No RL training. Execute pinned upstream class/function bodies without optional
Gym/PyG/Gurobi imports, then solve equivalent projected-feasibility MILPs with
SciPy/HiGHS. Run from any directory; output is an analysis artifact only.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import types

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
import torch


ROOT = Path(__file__).resolve().parents[1]
PINNED_UPSTREAM = "7e50d03e7129974a11061c9ce500bc2dc5593246"
COEFFICIENTS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)
EXTRACT = {
    "environment/base_envs.py": {
        "BaseRMAB", "MultiStateRMAB", "TorchMultiStateRMAB", "StandardRMAB", "TorchStandardRMAB"
    },
    "environment/constrained.py": {"ConstrainedRMAB", "TorchConstrainedRMAB"},
    "environment/rmab_instances.py": {"multistate_rmab", "get_wrapper", "get_constrained"},
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def scientific_source_hash(root):
    files = {}
    for pattern in ("solvercell/*.py", "tools/*.py", "run.py", "configs.py", "aggregate.py", "vendor/lsflow/**/*.py"):
        for path in sorted(root.glob(pattern)):
            files[str(path.relative_to(root))] = sha(path)
    return fingerprint(files)


def load_upstream(vendor):
    namespace = {"np": np, "torch": torch, "gym": types.SimpleNamespace(Env=object), "random": random}
    hashes = {}
    for relative, names in EXTRACT.items():
        path = vendor / relative
        tree = ast.parse(path.read_text())
        definitions = [node for node in tree.body
                       if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        assert {node.name for node in definitions} == names, (relative, names)
        module = ast.Module(body=definitions, type_ignores=[])
        exec(compile(module, str(path), "exec"), namespace)
        hashes[relative] = sha(path)
    return namespace, hashes


def run(root):
    before = scientific_source_hash(root)
    vendor = root / "vendor/lsflow"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=vendor, text=True).strip()
    assert commit == PINNED_UPSTREAM, (commit, PINNED_UPSTREAM)
    assert not subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=vendor, text=True).strip()
    namespace, source_hashes = load_upstream(vendor)
    # driver_lsflow.build_env_quiet reseeds after imports; omitted import-level
    # np.random.seed(42) therefore does not alter this seed-0 construction.
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        env, torch_env = namespace["get_constrained"](20, 40, 10, rmab_type="multistate")
    state = env.reset().astype(int)
    J, N = env.n_arms, env.n_workers
    costs = np.asarray(env.arm_costs)
    capacities = np.asarray(env.worker_capacity)
    rewards = np.asarray(env.rmab.state_r)
    transitions = np.asarray(env.rmab.transitions)
    assert np.array_equal(costs, torch_env.arm_costs)
    assert np.array_equal(capacities, torch_env.worker_capacity)
    p0, p1 = transitions[:, 0, 0], transitions[:, 0, 1]
    up = rewards[np.arange(J), np.minimum(state + 1, env.rmab.n_states - 1)]
    down = rewards[np.arange(J), np.maximum(state - 1, 0)]
    gain = (p1 - p0) * (up - down)
    base = float(np.sum(p0 * up + (1 - p0) * down))
    assert np.all(costs > 0)

    # Variable order [a_0...a_39, x_0,0...x_9,39]. The original permits repeated
    # service, but removing duplicate assignments preserves all capacity bounds
    # and a. Hence sum_i x_ij=a_j has the identical projected feasible set of a.
    # No global cardinality bound is added: budget=10 denotes worker count.
    rows, lower, upper = [], [], []
    for i in range(N):
        row = np.zeros(J + N * J)
        row[J + i * J:J + (i + 1) * J] = costs
        rows.append(row)
        lower.append(-np.inf)
        upper.append(capacities[i])
    for j in range(J):
        row = np.zeros(J + N * J)
        row[j] = -1
        row[J + j::J] = 1
        rows.append(row)
        lower.append(0)
        upper.append(0)
    constraints = LinearConstraint(np.asarray(rows), lower, upper)
    results = []
    for coefficient in COEFFICIENTS:
        objective = np.r_[-gain + coefficient * costs, np.zeros(N * J)]
        solution = milp(objective, integrality=np.ones(J + N * J), bounds=Bounds(0, 1),
                        constraints=constraints, options={"time_limit": 15, "mip_rel_gap": 0.0})
        assert solution.success and solution.status == 0, (coefficient, solution.message)
        binary = np.rint(solution.x).astype(int)
        assert np.max(np.abs(binary - solution.x)) <= 1e-6
        assert np.isin(binary, [0, 1]).all()
        action, assignments = binary[:J], binary[J:].reshape(N, J)
        loads = (assignments * costs).sum(axis=1)
        assert np.all(loads <= capacities)
        assert np.array_equal(assignments.sum(axis=0), action)
        # Check the original OR inequalities as well, not only the equivalent
        # formulation that produced the solution.
        assert np.all(action <= assignments.sum(axis=0))
        assert np.all(assignments <= action[None, :])
        resource_cost = int(costs @ action)
        expected_reward = float(base + gain @ action)
        upstream_expectation = float(env.calc_action_expected_value(action))
        # Upstream accumulates expected rewards in float32; this audit reports
        # the analytic expectation evaluated in float64.
        assert np.isclose(expected_reward, upstream_expectation, rtol=1e-6, atol=1e-6)
        assert np.array_equal(env.observation(), state)
        assert env.rmab.t == 0
        assert np.isclose(float(solution.fun), -float(gain @ action) + coefficient * resource_cost)
        results.append({"lambda": coefficient, "selected_count": int(action.sum()),
                        "selected": np.flatnonzero(action).tolist(), "action": action.tolist(),
                        "expected_original_reward": expected_reward,
                        "upstream_float32_expected_reward": upstream_expectation,
                        "resource_cost": resource_cost,
                        "scalarized_expected_reward": expected_reward - coefficient * resource_cost,
                        "workers": [np.flatnonzero(row).tolist() for row in assignments],
                        "worker_loads": loads.tolist(), "solver_status": int(solution.status),
                        "solver_message": solution.message, "mip_gap": float(solution.mip_gap),
                        "mip_node_count": int(solution.mip_node_count),
                        "feasibility_verified": True})
    instance = {"seed": 0, "n_arms": J, "n_workers": N, "horizon": env.horizon,
                "initial_state": np.asarray(env.rmab.init_state).tolist(),
                "transitions": transitions.tolist(), "state_rewards": rewards.tolist(),
                "service_costs": costs.tolist(), "worker_capacities": capacities.tolist()}
    assert results[0]["expected_original_reward"] > results[2]["expected_original_reward"]
    assert results[0]["resource_cost"] > results[2]["resource_cost"]
    assert results[0]["selected_count"] > N  # Demonstrates absence of cardinality B.
    after = scientific_source_hash(root)
    assert before == after
    return {"scope": "same-state exact one-step oracle conflict witness; no policy learning or RL result",
            "upstream_commit": commit, "upstream_definition_sha256": source_hashes,
            "analysis_script_sha256": sha(Path(__file__).resolve()),
            "scientific_source_sha256": before, "scientific_source_unchanged": before == after,
            "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                         "scipy": scipy.__version__, "torch": torch.__version__},
            "construction": "execute pinned official AST definitions, omit optional imports, reseed0, get_constrained(20,40,10,multistate), first NumPy reset",
            "instance": instance, "instance_sha256": fingerprint(instance),
            "state": state.tolist(), "base_expected_reward": base, "action_marginal_gains": gain.tolist(),
            "resource_objective": "sum(service_cost_j * selected_patient_j); count each patient once",
            "solver": {"backend": "scipy.optimize.milp / HiGHS", "binary_variables": J + N * J,
                       "time_limit_per_solve_seconds": 15, "mip_rel_gap": 0.0,
                       "projected_feasibility_equivalence": "positive costs permit removing all duplicate service assignments, preserving a and capacities",
                       "global_cardinality_bound_added": False},
            "results": results,
            "checks": {"all_solves_success": True, "all_original_constraints_verified": True,
                       "analytic_rewards_match_upstream_float32": True,
                       "no_environment_step_or_training": True, "same_state_for_every_preference": True},
            "limits": ["New resource objective is a preference extension, not the original LSFlow reward.",
                       "One reset state and one-step utility do not establish a long-term Pareto frontier.",
                       "These lambda coefficients are diagnostic, not a selected RL training range.",
                       "HiGHS feasibility is checked; this does not execute the original Gurobi solver or certify its tie behavior.",
                       "No claim about diffusion versus flow matching follows from this diagnostic."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "reports/preference_benchmark_audit/assignment_conflict.json"
    report = run(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "instance_sha256": report["instance_sha256"],
                      "scientific_source_unchanged": report["scientific_source_unchanged"],
                      "witnesses": [{k: row[k] for k in ("lambda", "selected_count", "expected_original_reward", "resource_cost", "mip_gap")}
                                    for row in report["results"]]}, indent=2))


if __name__ == "__main__":
    main()
