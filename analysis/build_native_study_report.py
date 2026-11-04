"""Rebuild the completed native study from audited artifacts; never train or select on heldout."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGNS = ("native_revision_20261104", "native_margin_20261104")
DESTINATION = ROOT / "reports/native_revision_study_20261104"
METRICS = ("last_window_return", "validation_auc", "final_validation_return")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    return {"n": len(values), "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "values": values}


def assert_close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-10), (a, b)


def load_campaign(root, campaign):
    report = root / "reports" / campaign
    summary = read(report / "summary.json")
    audit = read(report / "local_checkpoint_audit.json")
    deployment = read(report / "deployment.json")
    manifest_path = root / "experiments" / campaign / "manifest.json"
    manifest = read(manifest_path)
    assert audit["all_verified_complete"] is True
    assert summary["status_counts"] == {"complete": len(manifest["jobs"])}
    expected = manifest["expected_source_sha256"]
    assert summary["source_sha256"] == expected == deployment["source"]["source_sha256"]
    by_id = {row["id"]: row for row in summary["runs"]}
    audited = {row["id"]: row for row in audit["runs"]}
    rows = []
    for job in manifest["jobs"]:
        row = by_id[job["id"]]
        cfg, metrics = row["config"], row["metrics"]
        output = root / job["output"]
        assert cfg == read(output / "config.json")
        assert metrics == read(output / "metrics.json")
        assert hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest() == job["config_sha256"]
        assert row["status"] == "complete" and not row["audit_errors"]
        assert read(output / "exit.json")["returncode"] == 0
        check = audited[job["id"]]
        assert check["status"] == "verified_complete" and check["all_checkpoint_tensors_finite"]
        assert check["source_sha256"] == expected == metrics["source_sha256"]
        assert metrics["env_steps"] == 5000 and metrics["train_steps"] == 4000
        assert metrics["updates"] == metrics["actor_updates"] == 4000
        assert check["episodes"] == 250 and check["validation_evaluations"] == 9
        assert check["heldout_episodes"] == metrics["heldout_episodes"] == 50
        assert metrics["decoder_calls"] == 9600 and metrics["feasibility_rate"] == 1.0
        journal = [json.loads(line) for line in (output / "evaluations.jsonl").read_text().splitlines()]
        curve = [v for v in journal if v["split"] == "validation"]
        heldout = [v for v in journal if v["split"] == "heldout"]
        assert curve == row["curve"] and len(heldout) == 1
        xs, ys = [v["train_steps"] for v in curve], [v["return_mean"] for v in curve]
        assert xs == list(range(0, 4001, 500))
        assert all(v["seed_start"] == 100000 and v["episodes"] == 20 for v in curve)
        assert heldout[0]["seed_start"] == 200000 and heldout[0]["episodes"] == 50
        assert_close(metrics["last_window_return"], statistics.mean(ys[-2:]))
        auc = sum((b - a) * (u + v) / 2 for a, b, u, v in zip(xs, xs[1:], ys, ys[1:])) / 4000
        assert_close(metrics["validation_auc"], auc)
        assert_close(metrics["final_validation_return"], ys[-1])
        assert_close(metrics["heldout_return_mean"], heldout[0]["return_mean"])
        rows.append({"campaign": campaign, "id": row["id"], "status": "complete",
                     "method": cfg["method"], "seed": cfg["seed"], "margin": cfg["margin"],
                     "config": cfg, "config_sha256": job["config_sha256"],
                     "source_sha256": expected, "output": job["output"],
                     "metrics": metrics, "validation_curve": [{"train_steps": x, "return_mean": y} for x, y in zip(xs, ys)],
                     "deep_audit": check})
    metadata = {"campaign": campaign, "completed_runs": len(rows), "source_sha256": expected,
                "git_commit": deployment["source"]["git_commit"],
                "manifest": str(manifest_path.relative_to(root)), "manifest_sha256": sha(manifest_path),
                "summary_sha256": sha(report / "summary.json"),
                "local_deep_audit_sha256": sha(report / "local_checkpoint_audit.json"),
                "remote_root": deployment["remote_root"], "max_parallel": manifest["max_parallel"],
                "cpu_sets": manifest["cpu_sets"]}
    return rows, metadata, deployment["source"]["files"]


def build(root):
    destination = root / "reports/native_revision_study_20261104"
    a, provenance_a, files_a = load_campaign(root, CAMPAIGNS[0])
    b, provenance_b, files_b = load_campaign(root, CAMPAIGNS[1])
    assert len(a) == 6 and len(b) == 12
    source_changes = [key for key in sorted(set(files_a) | set(files_b)) if files_a.get(key) != files_b.get(key)]
    assert source_changes == ["solvercell/learner.py", "solvercell/trainer.py"]
    eq = {name: read(destination / f"{name}_storage_equivalence.json") for name in ("point", "sphere")}
    assert all(v["verified"] and all(v["checks"].values()) for v in eq.values())
    for value in eq.values():
        assert value["old_source"] == provenance_a["source_sha256"]
        assert value["new_source"] == provenance_b["source_sha256"]
    by_a, by_b = {r["id"]: r for r in a}, {r["id"]: r for r in b}
    for old, new in [("quotient_point", "point_m010_recheck"), ("quotient_sphere", "sphere_m010_recheck")]:
        assert by_a[old]["validation_curve"] == by_b[new]["validation_curve"]
        assert by_a[old]["config"] == by_b[new]["config"]
    margin_groups = []
    for margin, ids, by in [(.1, ["quotient_point", "quotient_cell", "quotient_independent"], by_a),
                            (.3, ["point_m030", "cell_m030", "independent_m030"], by_b),
                            (.6, ["point_m060", "cell_m060", "independent_m060"], by_b)]:
        point, cell, independent = [by[key] for key in ids]
        members = [point, cell, independent]
        assert all(r["seed"] == 0 and r["margin"] == margin for r in members)
        margin_groups.append({"margin": margin, "members": [{"campaign": r["campaign"], "id": r["id"]} for r in members],
                              "mean_last_window_return": statistics.mean(r["metrics"]["last_window_return"] for r in members),
                              "mean_validation_auc": statistics.mean(r["metrics"]["validation_auc"] for r in members),
                              "point_last_window": point["metrics"]["last_window_return"],
                              "cell_last_window": cell["metrics"]["last_window_return"],
                              "independent_last_window": independent["metrics"]["last_window_return"],
                              "contrasts": {metric: {"cell_minus_point": cell["metrics"][metric] - point["metrics"][metric],
                                                      "cell_minus_independent": cell["metrics"][metric] - independent["metrics"][metric]}
                                            for metric in ("last_window_return", "validation_auc")}})
    selected = sorted(margin_groups, key=lambda g: (-g["mean_last_window_return"], -g["mean_validation_auc"], g["margin"]))[0]
    pairs = []
    for seed in (0, 1, 2):
        suffix = "recheck" if seed == 0 else f"seed{seed}"
        point, sphere = by_b[f"point_m010_{suffix}"], by_b[f"sphere_m010_{suffix}"]
        pairs.append({"seed": seed, "scope": "known discovery seed repeated" if seed == 0 else "new training seed",
                      "point_id": point["id"], "sphere_id": sphere["id"],
                      "point": {m: point["metrics"][m] for m in METRICS},
                      "sphere": {m: sphere["metrics"][m] for m in METRICS},
                      "point_minus_sphere": {m: point["metrics"][m] - sphere["metrics"][m] for m in METRICS}})
    replication = {"fixed_margin": .1, "fixed_bridge_sigma": .5, "pairs": pairs,
                   "selection_scope": "parameters fixed from stage A before margin results; seed 0 is discovery, seeds 1/2 are new",
                   "all_three_seeds": {arm: {m: stats([p[arm][m] for p in pairs]) for m in METRICS}
                                       for arm in ("point", "sphere", "point_minus_sphere")},
                   "new_seeds_1_2_only": {m: stats([p["point_minus_sphere"][m] for p in pairs[1:]]) for m in METRICS}}
    heldout = {"used_for_selection": False, "scope": "50 final-checkpoint episodes, seeds 200000:200049; episode SD is not training-seed SD",
               "runs": [{"campaign": r["campaign"], "id": r["id"], "seed": r["seed"],
                         "return_mean": r["metrics"]["heldout_return_mean"],
                         "episode_return_sd": r["metrics"]["heldout_return_std"]} for r in a + b]}
    remote = read(destination / "final_remote_state.json")
    assert not remote["training_workers"]
    assert all(c["status"] == "complete" and set(c["jobs"].values()) == {"complete"} for c in remote["campaigns"].values())
    old = read(root / "reports/study_20261104/study.json")
    diagnostic_path = root / "reports/continuation_branch_audit/summary.json"
    diagnostic = read(diagnostic_path)
    assert diagnostic["science_source_unchanged"] is True
    assert diagnostic["comparison"]["cases"] == 12
    assert diagnostic["comparison"]["rb2_trace_lower_than_sf2_cases"] == 0
    return {"status": "complete", "completed_runs": 18, "failed_runs": 0, "partial_runs": 0,
            "live_runs": 0, "queued_runs": 0, "total_training_updates": 72000, "total_environment_steps": 90000,
            "instance_seed": 0, "instance_hash": a[0]["metrics"]["instance_hash"],
            "protocol": {"n_arms": 40, "n_actions": 20, "budget": 10, "horizon": 20,
                         "warmup_steps": 1000, "train_steps": 4000, "validation_episodes": 20,
                         "last_window": "mean of validation returns at training steps 3500 and 4000",
                         "auc": "trapezoidal validation area over steps 0:500:4000 divided by 4000",
                         "evaluation_policy": "best of 12 native-critic candidates, without behavior execution perturbation",
                         "full_released_budget": 40000, "generalization_scope": "one fixed instance; short-budget training"},
            "provenance": [provenance_a, provenance_b], "source_changes_between_stages": source_changes,
            "storage_equivalence": eq,
            "margin_selection": {"basis": "mean last-window validation across point/cell/independent; exact tie: mean AUC, then smaller margin",
                                 "heldout_used": False, "historical_anchor_gate_passed": True,
                                 "historical_anchor_counted_once": True, "selected_margin": selected["margin"], "groups": margin_groups},
            "fixed_replication": replication, "heldout_final_checkpoint": heldout,
            "runs": a + b, "final_remote_state": remote,
            "historical_study": {"path": "reports/study_20261104/RESULTS.md", "completed_runs": old["completed_arms"],
                                 "source_sha256": old["experiment_source_sha256"], "pooled_with_native_study": False,
                                 "reason": "different adaptation harness; historical tuning context only"},
            "all_campaigns_completed_runs": sum(len(c["jobs"]) for c in remote["campaigns"].values()),
            "subsequent_mechanism_diagnostic": {
                "path": str(diagnostic_path.relative_to(root)), "sha256": sha(diagnostic_path),
                "scope": diagnostic["scope"], "kernel_scope": diagnostic["kernel_scope"],
                "samples_per_estimator": diagnostic["samples_per_estimator"],
                "coordinates": diagnostic["coordinates"], "comparison": diagnostic["comparison"],
                "checks": diagnostic["checks"], "rl_training_launched": False,
                "decision": "two-query branch estimator fails the equal-query variance gate; diagnose actual denoising-step credit before proposing training"},
            "compute_caveats": {"equal_environment_update_budget": True, "equal_nfe": False,
                                "flow_nfe_per_sample": 24, "diffusion_nfe_per_sample": 12,
                                "decoder_calls_each": 9600,
                                "decoder_count_scope": "executed native actions including evaluation; not neural candidate evaluations",
                                "intrinsic_speed_advantage_established": False},
            "conclusions": ["Cell endpoint coupling is not supported as a consistent improvement over point or independent controls.",
                            "Shared margin 0.6 improves seed-0 cell return, but cell remains below point and the pairing sign changes across margins.",
                            "Quotient point wins discovery seed 0 but loses both new seeds to quotient spherical flow; no diffusion superiority or stability claim.",
                            "Short, one-instance results do not establish full-budget convergence, cross-instance generalization or a novel diffusion-specific mechanism.",
                            "A subsequent fixed-kernel oracle diagnostic also fails the equal-two-query variance gate for the simplest conditional-branch estimator; it is not a trained-policy or RL result."]}


def render(study):
    a, b = [[r for r in study["runs"] if r["campaign"] == campaign] for campaign in CAMPAIGNS]
    replication, margin = study["fixed_replication"], study["margin_selection"]
    delta = replication["all_three_seeds"]["point_minus_sphere"]["last_window_return"]
    def pm(v):
        return f'{v["mean"]:.3f} ± {v["sample_sd"]:.3f}'
    lines = ["# Native-structure diffusion study: completed results, 2026-11-04", "",
             "**All 18 new runs are complete and audited: 6 native-structure arms plus 12 margin/replication arms.** Together with the separate historical 14-arm study, all 32 authorized runs across four campaigns are complete. No new RL experiment is queued by this report.", "",
             f"The main scientific result is negative for the current cell-coupling proposal. A shared margin of **{margin['selected_margin']:.1f}** improves the cell arm at seed 0, but it remains below point diffusion. Point diffusion's seed-0 advantage over quotient spherical flow reverses on both new seeds: the paired last-window difference over seeds 0/1/2 is **{pm(delta)}** (mean ± sample training-seed SD, n=3). This does not support a diffusion superiority or stability claim.", "",
             "![Completed learning curves and fixed replication](learning_and_replication.png)", "",
             "[Vector figure](learning_and_replication.pdf) · [Machine-readable study](study.json) · [Rebuild script](../../analysis/build_native_study_report.py)", "",
             "## Protocol and completion", "",
             "All runs use the same released Dynamic Intervention instance: N40/A20/B10/H20, instance seed 0; 1,000 warmup interactions plus 4,000 training interactions and updates. Batch size is 64, actor hidden width 256, learning rate 0.0004, 12 behavior candidates, 16 smoothed Bellman candidates, and 12 sampler steps. Native critic updates, Q normalization, alpha schedule, clipping, execution perturbation, and trimmed Bellman target are retained. Diffusion bridge sigma remains 0.5. The released full budget is 40,000 updates; these are short controlled experiments, not full-budget reproduction claims.", "",
             "Each run has exit 0, the exact config/source/instance identities, 5,000 environment steps, 4,000 critic updates, 4,000 actor updates, 250 complete H20 training episodes, nine scheduled validation evaluations, 50 final heldout episodes, and finite saved scientific state. Local deep audits pass for all 18 mirrored checkpoints. Total new work: 90,000 environment interactions and 72,000 training updates. The current local suite records 54 passing tests; exact storage rechecks provide additional full-run evidence.", "",
             "Validation uses the same 20 episode seeds 100000–100019 at steps 0,500,…,4000, with evaluation RNG isolated from training. **Last-window** is the mean of steps 3500 and 4000; **AUC** is trapezoidal area over the full nine-point validation curve divided by 4000. Evaluation selects the critic's best of 12 candidates without the training behavior's final execution perturbation. Reported returns therefore refer to that evaluation procedure. No checkpoint or hyperparameter is selected using heldout results.", "",
             "## Stage A: native-structure bridge, seed 0", "",
             "| Arm | Last-window validation | Validation AUC | Final validation | Wall seconds |",
             "|---|---:|---:|---:|---:|"]
    for r in a:
        m = r["metrics"]
        lines.append(f"| `{r['id']}` | {m['last_window_return']:.3f} | {m['validation_auc']:.3f} | {m['final_validation_return']:.3f} | {m['wall_seconds']:.1f} |")
    lines += ["", "Point diffusion in the quotient space has the highest last-window score in this discovery screen. The native full-space reference has the highest AUC (299.974), so the point result does not dominate the full learning curve. Coupled cell (289.115) and independent cell (288.648) are nearly tied at margin 0.1 and both trail point (335.431). The native full-space reference remains seed-0-only; quotient replication does not turn it into a replicated baseline.", "",
              "## Stage B: all 12 completed arms", "", "| Arm | Seed | Margin | Last-window validation | Validation AUC | Final validation | Wall seconds |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in b:
        m = r["metrics"]
        lines.append(f"| `{r['id']}` | {r['seed']} | {r['margin']:.1f} | {m['last_window_return']:.3f} | {m['validation_auc']:.3f} | {m['final_validation_return']:.3f} | {m['wall_seconds']:.1f} |")
    lines += ["", "### Shared margin selection", "",
              "The prespecified rule averages last-window validation equally across point, coupled cell and independent cell, then breaks an exact tie with their mean AUC and finally the smaller margin. The historical margin-0.1 group is admitted only after exact storage equivalence passes and is counted once. The new point recheck is not an additional independent observation. Heldout values and favorable cell-minus-control gaps do not select the margin.", "",
              "| Margin | Point | Cell | Independent | Three-method mean | Mean AUC | Cell − point | Cell − independent |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in margin["groups"]:
        c = g["contrasts"]["last_window_return"]
        lines.append(f"| {g['margin']:.1f} | {g['point_last_window']:.3f} | {g['cell_last_window']:.3f} | {g['independent_last_window']:.3f} | {g['mean_last_window_return']:.3f} | {g['mean_validation_auc']:.3f} | {c['cell_minus_point']:+.3f} | {c['cell_minus_independent']:+.3f} |")
    lines += ["", "**Shared margin 0.6 is selected.** Cell improves from 289.115 to 308.283, but point remains higher at 326.995. The cell-minus-independent difference changes sign across margins, and all factorial arms use only seed 0. The result supports a limited margin intervention, not a stable coupling advantage. The earlier fixed-checkpoint execution-flip probe (0.897/0.788/0.603 at margins 0.1/0.3/0.6) motivated the intervention; it is not a retrained-policy or training-curve robustness measurement. These results do not isolate noise robustness from all other consequences of changing endpoint laws.", "",
              "### Fixed point-versus-sphere replication", "",
              "This comparison keeps margin 0.1 and bridge sigma 0.5, specified from Stage A before the margin-factorial results. All six runs use the same Stage B source. Seed 0 is the known discovery seed repeated to check storage equivalence; only seeds 1 and 2 are new training-seed evidence. The selected margin 0.6 is not substituted into this fixed comparison.", "",
              "| Seed | Scope | Point last-window | Sphere last-window | Point − sphere | Point AUC − sphere AUC |",
              "|---:|---|---:|---:|---:|---:|"]
    for p in replication["pairs"]:
        lines.append(f"| {p['seed']} | {p['scope']} | {p['point']['last_window_return']:.3f} | {p['sphere']['last_window_return']:.3f} | {p['point_minus_sphere']['last_window_return']:+.3f} | {p['point_minus_sphere']['validation_auc']:+.3f} |")
    lines += ["", "| Across training seeds 0/1/2 (n=3) | Last-window mean ± sample SD | AUC mean ± sample SD |", "|---|---:|---:|"]
    for arm in ("point", "sphere", "point_minus_sphere"):
        values = replication["all_three_seeds"][arm]
        lines.append(f"| {arm.replace('_', ' ')} | {pm(values['last_window_return'])} | {pm(values['validation_auc'])} |")
    fresh = replication["new_seeds_1_2_only"]
    lines += ["", f"For new seeds 1/2 alone, paired last-window difference is {pm(fresh['last_window_return'])}; paired AUC difference is {pm(fresh['validation_auc'])} (n=2). Both new seeds favor spherical flow. Three pairs, including a discovery seed, support a limited repeatability assessment, not a precise population estimate or a significance claim. These SDs measure training-seed variation on one fixed environment instance; rollout episode SD is reported separately below.", "",
              "## Compute and implementation scope", "",
              "All arms have equal environment/update budgets and 9,600 recorded native decoder calls. That counter counts executed actions, including evaluation; it does not count neural candidate evaluations. The inherited Heun spherical sampler uses **24 network evaluations per sample**, while diffusion uses **12**. Equal sampler-step settings therefore do not match neural evaluation cost. Timing also includes evaluation, checkpointing, projection, and concurrency. The recorded wall times do not establish an intrinsic speed advantage of diffusion.", "",
              "| Actor family | Actor parameters | Critic parameters | NFE per sample |", "|---|---:|---:|---:|",
              "| Native full-space spherical reference | 107112 | 173570 | 24 |",
              "| Full-space point diffusion | 117392 | 173570 | 12 |",
              "| Quotient spherical flow | 90696 | 173570 | 24 |",
              "| Quotient point/cell/independent diffusion | 92752 | 173570 | 12 |", "",
              "Tensor-only native sampling passed strict equivalence tests; its paired local microbenchmark was about 1.008×, which is not a meaningful established acceleration. Compact replay reduces a 5,000/100,000-row replay payload from 57,275,424 to 2,842,248 bytes (95.04%). That is replay serialization, not a 95% reduction in complete model checkpoints or runtime memory. The actual point complete checkpoint changes from 61,457,791 to 7,177,477 bytes; sphere changes from 61,424,181 to 7,140,795 bytes.", "",
              "Both complete seed-0 rechecks exactly reproduce their old resolved configs, every validation/heldout/training-episode return, final learner and optimizer states, environment counters, all RNGs, and valid replay rows/ring positions. Only representation, source identity and timing/artifact fields are excluded. These checks justify pooling the historical margin-0.1 anchors across the storage-only source boundary.", "",
              "[Point equivalence audit](point_storage_equivalence.json) · [Sphere equivalence audit](sphere_storage_equivalence.json) · [Replay benchmark](../../analysis/replay_storage_benchmark_20261104.json) · [Tensor-path benchmark](../../analysis/fast_native_benchmark_20261104.json)", "",
              "## Final heldout results: descriptive only", "",
              "These are final-checkpoint means and within-run episode SDs over 50 seeds 200000–200049. They did not select shared margin, method, checkpoint, or replication parameters. Episode SD is not the uncertainty of an across-training-seed improvement.", "",
              "| Stage | Arm | Mean return | Episode SD |", "|---|---|---:|---:|"]
    for r in study["heldout_final_checkpoint"]["runs"]:
        stage = "A" if r["campaign"] == CAMPAIGNS[0] else "B"
        lines.append(f"| {stage} | `{r['id']}` | {r['return_mean']:.3f} | {r['episode_return_sd']:.3f} |")
    lines += ["", "## Provenance and current completion snapshot", ""]
    for p in study["provenance"]:
        lines += [f"- `{p['campaign']}`: {p['completed_runs']} complete; commit `{p['git_commit']}`; scientific source `{p['source_sha256']}`. [Summary](../{p['campaign']}/summary.json), [local deep audit](../{p['campaign']}/local_checkpoint_audit.json), [deployment manifest](../{p['campaign']}/deployment.json)."]
    remote = study["final_remote_state"]
    lines += ["", "Only `solvercell/learner.py` and `solvercell/trainer.py` differ between the two frozen scientific snapshots, implementing compact Replay serialization/save/restore. Historical sources and manifest configs remain immutable. Run the report rebuild from the repository root with `python analysis/build_native_study_report.py`; it verifies configs, exits, counters, deep-audit outcomes, all validation curves and metrics before writing this report.", "",
              f"The recorded remote snapshot is **{remote['captured_utc']}**: all four campaign controllers report complete, `training_workers=[]`; GPU snapshot is `{remote['gpu']}`; `unrelated_original_worker_exists={str(remote['unrelated_original_worker_exists']).lower()}`. Our experiments used CPU and did not signal or modify the unrelated GPU worker. The prior worker was no longer present at this snapshot; no claim is made about its continued execution. [Snapshot](final_remote_state.json).", "",
              "## Historical results and research decision", "",
              "The initial 8-arm matched-harness screen and 6-arm temperature factorial are a separate completed 14-run historical study under source `258f8ec3afa41bab34e4456f19b114bb09c64d0a56ef22a914b24ad20258c914`. Their critic/decoder adaptation differs from this native-structure study; they are not pooled into the replication or margin statistics. [Historical report](../study_20261104/RESULTS.md).", "",
              "The present results reject treating shorter source-to-cell Euclidean paths as sufficient evidence of better policy improvement. The methodology audit explains why preserving an action label does not preserve the native noisy execution kernel, and why minimum Euclidean transport is not the minimum-KL lift of an action-space PMD step. Current endpoint-weighted denoising uses stochastic-interpolant regression; it has not tested trajectory-level denoising likelihood credit or established an advantage specific to diffusion over flow matching. Best-of-K behavior, fixed execution noise, and the trimmed bootstrap also limit claims about an exact deployment-policy value function.", "",
              "A subsequent inexpensive oracle check also gives a negative efficiency result for the simplest conditional-branch gradient estimator. On three fixed real late-replay states × four declared coordinates (12 cases, 20,000 estimates per case), its two-query full-gradient covariance trace is **1.473–1.862 times** that of averaging two independent score-function samples; all 12 cases are worse at equal Q-query cost. The analytic integration and finite-difference checks pass, which validates the calculation, not efficiency. This uses a declared mean-zero, sigma-0.5 primitive20 Gaussian test kernel and exact immediate-reward oracle; it is neither the trained policy's terminal kernel nor learned-Q/long-horizon RL evidence. It blocks promoting this first estimator into a training sweep. The next diagnostic should first establish the actual denoising-step credit problem.", "",
              "[Conditional-branch diagnostic](../continuation_branch_audit/summary.json) · [Continuation critic derivation and negative gate](../../docs/NOISE_CONTINUATION_CRITIC.md). No additional RL run was launched for this diagnostic.", "",
              "[Methodology and source audit](../../docs/METHODOLOGY_AUDIT.md) · [Support/prior audit](../../docs/SOLVER_SUPPORT_AND_PRIOR.md) · [Feasible-exchange rethink](../../docs/FEASIBLE_EXCHANGE_RETHINK.md) · [Diffusion redesign decision](../../docs/DIFFUSION_REDESIGN_DECISION.md) · [Prespecified margin/replication design](../../docs/NATIVE_MARGIN_FOLLOWUP.md). Design documents preserve their preparation-time statements; this report records actual completion. New mechanism proposals are research designs only, not implemented or validated RL results. Further experiments require a fresh bounded design; none is launched by this analysis.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    study = build(root)
    destination = root / "reports/native_revision_study_20261104"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "study.json").write_text(json.dumps(study, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (destination / "RESULTS.md").write_text(render(study))
    print(json.dumps({"completed_runs": study["completed_runs"],
                      "selected_margin": study["margin_selection"]["selected_margin"],
                      "paired_last_window": study["fixed_replication"]["all_three_seeds"]["point_minus_sphere"]["last_window_return"],
                      "report": str(destination / "RESULTS.md")}, indent=2))


if __name__ == "__main__":
    main()
