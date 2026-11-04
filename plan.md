# SolverCell-RL plan

## Objective and current status

Build a source-grounded diffusion-policy workspace around a released LSFlow combinatorial-RL benchmark, run bounded controlled experiments on wflow-server, and use the completed evidence to decide whether the proposed mechanism merits further work.

**2026-11-04: all32 historical single-objective runs across four campaigns are complete.** The original14-run matched-harness study and subsequent18-run native-structure study remain separate scientific comparisons. Their final remote snapshot was09:36:30 UTC. A new preference harness is now implemented and being validated independently; current preference status is recorded at the end of this file.

The current research direction is documented in [Diffusion redesign decision](docs/DIFFUSION_REDESIGN_DECISION.md). It is a proposed design, not a validated RL result. The current cell-coupling implementation did not establish a consistent advantage, and point diffusion's promising discovery seed did not replicate against quotient spherical flow on new seeds 1 and 2.

## Completed work

1. [done] Create an independent local workspace and pin upstream commit `7e50d03e7129974a11061c9ce500bc2dc5593246`.
2. [done] Audit and port the released Dynamic Intervention environment, preserving N40/A20/B10/H20, the original instance constructor and transition/reward semantics; use instance seed 0 throughout.
3. [done] Implement exact robust-cell projection, stochastic-interpolant diffusion, point and independent-cell controls, a matched spherical reference, and the native LSFlow adapter.
4. [done] Validate projection, upstream environment trajectories, finite gradients, train/evaluation RNG isolation, exact checkpoint recovery, controller accounting and durable exit evidence.
5. [done] Complete and audit the initial eight-arm screen and six-arm temperature factorial; preserve all negative and degraded arms.
6. [done] Diagnose the adaptation boundary, then retain native critics, Q normalization, update schedules, noisy execution and trimmed Bellman targets while replacing the actor.
7. [done] Implement the rank-8 quotient actor, tie-aware original-decoder geometry, and matched native point/cell/independent/quotient-flow controls.
8. [done] Complete and audit the six-arm native revision at training seed 0.
9. [done] Optimize replay serialization without changing scientific state; verify legacy loading, wrapped-buffer behavior, exact next samples/writes, and unchanged RNGs.
10. [done] Complete all 12 margin/replication jobs: six margin-factorial arms, two seed-0 full-run equivalence rechecks, and four fixed point/sphere runs at training seeds 1 and 2.
11. [done] Verify both point and sphere rechecks are bitwise equivalent in final scientific state and exactly equal in complete validation, heldout and training-episode trajectories.
12. [done] Recompute validation-only shared-margin selection and paired training-seed statistics; preserve full curves, separate final heldout results, and document unequal neural evaluation cost.
13. [done] Audit the action/cost KL boundary, noisy execution kernel, solver support and prior, and existing diffusion/flow mechanisms; prepare a revised research direction without launching new experiments.

## Scientific outcomes

- Historical matched-harness study: common temperature 0.1 selected using validation-only across-method means. Last-window returns are cell 147.635, point 181.346 and independent cell 143.648. Native LSFlow is 316.331, while the matched spherical adaptation is 160.845. These figures diagnose the large adaptation boundary; they are not pooled with the native-structure study.
- Native Stage A, seed 0: native reference 316.331, full point 320.364, quotient sphere 318.809, quotient point 335.431, quotient cell 289.115 and independent cell 288.648. The native reference has the highest AUC, 299.974; the point endpoint result does not dominate the full curve.
- Shared margin 0.6 is selected from {0.1, 0.3, 0.6} using the mean last-window return across point/cell/independent, with AUC as the prespecified tie-breaker. At 0.6, point is 326.995, cell is 308.283 and independent cell is 300.864. Cell improves over its 0.1 anchor but still trails point; the cell-minus-independent sign changes across margins. All margin-factorial evidence remains seed-0-only.
- Fixed margin-0.1 replication uses the same new source and the parameters fixed before margin results. Point-minus-sphere last-window differences are +16.622, −26.211 and −35.580 for training seeds 0, 1 and 2. Across all three, the mean is −15.056 and sample SD is 27.832. Seed 0 is the known discovery seed; both newly added seeds favor spherical flow. No diffusion superiority, stable coupling benefit or significance claim is supported.
- One instance and 4,000 training updates per run do not establish cross-instance generalization or full-budget convergence. Final heldout episodes never select margin, method, checkpoint or follow-up parameters.
- Equal interaction/update budgets and 9,600 recorded decoder calls do not imply equal compute: the Heun spherical sampler uses 24 network evaluations per sample, diffusion uses 12. Recorded wall times do not establish an intrinsic speed advantage.
- A later fixed-kernel oracle diagnostic fails the equal-Q-query efficiency gate for the simplest conditional-branch estimator: its full-gradient variance trace is 1.473–1.862 times averaging two independent score-function samples in all 12 cases (three real late-replay states × four coordinates, 20,000 estimates each). Integration/finite-difference checks pass, but this is a declared Gaussian test kernel and myopic oracle, not the trained policy or an RL result. The proposed estimator is not promoted to a training sweep.

## Engineering and source boundaries

- Current local full suite: 54 passing tests. Tensor-only native sampling preserves original computation and RNG behavior; the paired local microbenchmark is approximately 1.008×, without a meaningful acceleration claim.
- A 5,000/100,000-row replay payload shrinks from 57,275,424 to 2,842,248 bytes (95.04%). This is checkpoint replay serialization, not runtime memory or total-model compression. The complete point and sphere rechecks establish exact full-run equivalence across the storage boundary.
- Historical 14-run scientific source: `258f8ec3afa41bab34e4456f19b114bb09c64d0a56ef22a914b24ad20258c914`.
- Native Stage A, 6 runs: source `030e1c799e990310b3c2d44a1872b051baefdc141267d42bf048df5c5d1eb53b`, commit `5e2e764`.
- Native Stage B, 12 runs: source `6eb3126bc49621d34308013091304b3fb46bb17e9c35bec7062d322cf656a66c`, commit `00a7eed`. The hashed source differences from Stage A are `solvercell/learner.py` and `solvercell/trainer.py`, for replay serialization/save/restore. Full-run equality gates passed before historical margin anchors were combined.
- Frozen manifests, source snapshots and remote outputs remain immutable. At most four single-thread CPU workers used CPUs 24–27 in the native stages; no GPU training was launched. The final remote snapshot records GPU 684 MiB / 4% and the previously observed unrelated worker absent. We did not signal or modify that worker, and do not claim it is still running.

## Deliverables and reproducible analysis

- [Completed native 18-run report](reports/native_revision_study_20261104/RESULTS.md), [machine-readable results](reports/native_revision_study_20261104/study.json), [full curves and replication figure](reports/native_revision_study_20261104/learning_and_replication.png), [PDF figure](reports/native_revision_study_20261104/learning_and_replication.pdf).
- [Report builder](analysis/build_native_study_report.py): `python analysis/build_native_study_report.py` verifies saved config/source/exit/counters/curves and deep-audit outcomes before rebuilding the report. It does not train or use heldout values for selection.
- [Point storage equivalence](reports/native_revision_study_20261104/point_storage_equivalence.json), [sphere storage equivalence](reports/native_revision_study_20261104/sphere_storage_equivalence.json), [final remote state](reports/native_revision_study_20261104/final_remote_state.json).
- [Historical completed 14-run report](reports/study_20261104/RESULTS.md) and its [full learning curves](reports/study_20261104/learning_curves.png).
- [Native revision design](docs/NATIVE_REVISION_DESIGN.md), [prespecified margin/replication design](docs/NATIVE_MARGIN_FOLLOWUP.md). These preserve preparation-time statements; completed status is recorded here and in the result report.
- [Methodology audit](docs/METHODOLOGY_AUDIT.md), [solver support/prior audit](docs/SOLVER_SUPPORT_AND_PRIOR.md), [feasible-exchange rethink](docs/FEASIBLE_EXCHANGE_RETHINK.md), [diffusion redesign decision](docs/DIFFUSION_REDESIGN_DECISION.md).
- [Continuation critic and negative efficiency gate](docs/NOISE_CONTINUATION_CRITIC.md), [fixed-kernel branch diagnostic](reports/continuation_branch_audit/summary.json).
- All complete run artifacts and checkpoints are mirrored under Git-ignored `outputs/`. Campaign-specific `reports/` preserve CSV/JSON/LaTeX summaries, deployments and independent audits.

## Preference extension: implementation and bounded validation

The latest user direction is preference-controllable generation: one policy changes its real decision distribution with deployment priorities. See [preference redesign](docs/PREFERENCE_CONTROL_REDESIGN.md), [Scheduling/Assignment audit](docs/PREFERENCE_BENCHMARK_SCHEDULING_ASSIGNMENT.md), [Intervention/Disease audit](docs/PREFERENCE_BENCHMARK_INTERVENTION_DISEASE.md), and [guidance methodology](docs/PREFERENCE_GUIDANCE_METHODOLOGY.md).

- [done] Audit all five benchmark domains: original rewards are scalar; preference vectors are declared extensions, not original multi-objective benchmark results.
- [done] Establish native-decodable, equal-budget conflicting Intervention actions and relabel the two reward components from all5000 saved transitions.
- [done] Establish exact same-state Assignment utility–resource MILP witnesses under the original projected feasible set; worker count is not a global patient cardinality cap.
- [done] Check the released routing cost interface on32 fixed coefficient vectors:30 selections cannot fit even a relaxed20-edge closed walk. This is a decoder probe, not a trained-policy violation rate.
- [done] Document Disease's fixed-world semantics, absent local graph data, sampler/RNG and truncation-metric boundaries before proposing risk/fairness claims.
- [done] Define clean-cell preference tilt and joint noise-level soft continuation; preserve the execution-noise/initial-law and future-preference Bellman conditions. Compare against correctly conditioned FM, not only fixed-reward LSFlow.

- [done] Implement vector reward/replay, time-aware observation, preference relabeling with consistent Bellman weights, and matched conditional sphere/diffusion actors in a separate harness.
- [done] Implement actual-chain suffix targets, a joint log-exponential continuation guide, exact terminal weighting, and SMC with initial weights and ancestor/potential accounting.
- [done] Add isolated evaluation, complete checkpoints, bitwise mid-episode continuation, source/config guards and invocation-owned exit evidence. Preserve the old single-objective code and immutable campaigns.
- [in progress] Freeze the implementation, run two bounded server arms (one model each, seed0,256 warmup+1000 updates), and audit complete curves and unseen preference response. Engineering validation and scientific performance remain separate.
- [pending] Assignment utility–resource implementation and multi-instance/multi-seed replication. Neither is part of this first implementation pilot.

See the [implementation contract](docs/PREFERENCE_IMPLEMENTATION.md) for actual algorithm semantics, compute scope and approximation boundaries. Joint guidance must improve on equal-particle terminal weighting after counting extra guide cost before supporting a diffusion-specific efficiency claim. The earlier two-query branch estimator remains behind its failed cost gate; no RB or margin training sweep is planned.
