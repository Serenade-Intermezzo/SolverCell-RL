#!/usr/bin/env python3
"""Source-grounded reward decomposition/witnesses; no RL or dynamics changes."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solvercell.environment import InterventionEnv, UPSTREAM_COMMIT
from tools.campaign import source_sha256


def main():
    before = source_sha256(ROOT)
    env = InterventionEnv(instance_seed=0)
    groups = [env.bad_arms, env.good_arms]
    state = np.full(env.n_arms, 2, dtype=int)
    witnesses = []
    for name, indices in (("all_low_response_interventions", env.bad_actions),
                          ("all_high_response_interventions", env.good_actions)):
        action = np.zeros(env.n_actions)
        action[indices] = 1
        p = env.transition_probabilities(action)
        expected_arm_rewards = p*env.state_r[:, 3]+(1-p)*env.state_r[:, 1]
        means = [float(expected_arm_rewards[group].mean()) for group in groups]
        total = float(expected_arm_rewards.sum())
        assert abs(total-env.expected_reward(action, state)) < 1e-12
        # These witnesses use whole duplicate groups and are reachable by native W.
        coefficient = np.zeros(env.n_arms)
        coefficient[env.bad_arms] = 1 if name.startswith("all_low") else -1
        coefficient[env.good_arms] = -coefficient[env.bad_arms[0]]
        scores = coefficient @ env.weights_p
        decoded = np.zeros(env.n_actions)
        order = np.argsort(-scores, kind="stable")[:env.budget]
        decoded[order[scores[order] > 0]] = 1
        np.testing.assert_array_equal(decoded, action)
        witnesses.append({"name": name, "selected_primitives": np.flatnonzero(action).tolist(),
                          "action_count": int(action.sum()), "cohort_mean_rewards": means,
                          "original_expected_reward": total, "native_decoder_witness_cost": coefficient.tolist()})
    first, second = (np.array(w["cohort_mean_rewards"]) for w in witnesses)
    difference = first-second
    assert difference[0] > 0 and difference[1] < 0
    cutoff = float(-difference[1]/(difference[0]-difference[1]))
    preference_rows = []
    for weight in (.25, .5, .75):
        values = [float(weight*x[0]+(1-weight)*x[1]) for x in (first, second)]
        preference_rows.append({"low_response_weight": weight, "mean_utility_values": values,
                                "preferred_witness": witnesses[int(np.argmax(values))]["name"]})

    # Existing replay has next_obs and action; no new environment interaction.
    import torch
    checkpoint_path = ROOT / "outputs/native_revision_20261104/quotient_point/checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["instance_hash"] == env.instance_hash
    replay = checkpoint["replay"]
    valid = int(replay["size"])
    nxt = np.asarray(replay["data"]["next_obs"])[:valid].astype(int)
    arm_rewards = env.state_r[np.arange(env.n_arms)[None, :], nxt].astype(np.float32)
    components = np.stack([arm_rewards[:, group].mean(1, dtype=np.float64) for group in groups], axis=1)
    reconstructed = components @ np.array([len(group) for group in groups])
    actual_reward = np.asarray(replay["data"]["reward"])[:valid]
    error = float(np.max(np.abs(reconstructed-actual_reward)))
    assert error < 1e-5
    counts = np.asarray(replay["data"]["action"])[:valid].sum(1)

    root = ROOT / "vendor/lsflow"
    expected_data = ["ICPSR_22140.pkl", "ICPSR_22140/DS0001/22140-0001-Data.tsv",
                     "ICPSR_22140/DS0002/22140-0002-Data.tsv", "ICPSR_22140/DS0003/22140-0003-Data.tsv",
                     "ICPSR_22140/checkpoints/HIV.pkl", "ICPSR22140/checkpoints/HIV.pkl"]
    after = source_sha256(ROOT)
    assert after == before
    result = {
        "scope": "reward-component observability and an exact one-step conflict witness; not MORL/Pareto/RL performance",
        "upstream_commit": UPSTREAM_COMMIT, "science_source_sha256": before, "science_source_unchanged": True,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "instance_hash": env.instance_hash,
        "intervention": {"state": state.tolist(), "group_definition": "existing synthetic low/high sigmoid-response cohorts; not demographic groups",
                         "group_arm_indices": [group.tolist() for group in groups], "witnesses": witnesses,
                         "strict_tradeoff_switch_weight": cutoff, "pairwise_preference_comparison": preference_rows,
                         "scope": "a pair of feasible native-decoder-reachable actions; not a claim of full Pareto-optimality"},
        "existing_replay_decomposition": {
            "path": str(checkpoint_path), "source_sha256": checkpoint["source_sha256"],
            "rows": valid, "reconstruction_max_abs_error": error,
            "roundoff_note": "groupwise float64 mean versus upstream float32 sum",
            "action_count_min": float(counts.min()), "action_count_max": float(counts.max()),
            "mean_components": components.mean(0).tolist(),
        },
        "disease_local_data_presence": {path: (root/path).exists() for path in expected_data},
        "disease_synthetic_witness_only": {
            "scope": "two isolated frontier nodes; analytic independent Bernoulli example, not fitted ICPSR data",
            "budget": 1, "positive_probabilities": [.9, .1],
            "one_batch_group_rewards": {"test_first_group": [.9, 0.], "test_second_group": [0., .1]},
            "weight_on_first_group_switch": .1,
            "complete_episode_expected_total_positives_under_either_order": 1.,
        },
    }
    output = ROOT / "reports/preference_component_audit"
    output.mkdir(parents=True, exist_ok=True)
    (output/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
