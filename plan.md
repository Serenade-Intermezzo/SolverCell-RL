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

- Current local full suite:90 passing tests (54 historical checks plus36 preference checks). Tensor-only native sampling preserves original computation and RNG behavior; the paired local microbenchmark is approximately1.008×, without a meaningful acceleration claim.
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
- [done] Freeze implementation commit`f304a2f`, prepare commit`2382fc1`, scientific source`9b8a1b044b789a34ca7d3898bb3613db85f9e0008163287cdde5cb1b4b626b8c`; complete and audit both server arms (seed0, instance0,256 warmup+1000 updates). All85 tests pass. Configs, worker affinity/comm, source/exit, checkpoint, replay, all five validation points and every final evaluation trajectory pass the audit. Final checkpoint byte hashes match the server.
- [done] Record first pilot results: direct validation AUC sphere131.702/diffusion139.749; unseen-preference direct means157.785/137.240 and terminal-weighted means175.026/168.013. These are separate splits and one-seed descriptive statistics. Default diffusion SMC equals terminal weighting in all32 paired heldout episodes, with zero intermediate resampling and40960 additional guide forward rows. No guide benefit or diffusion superiority is established.
- [done] Complete the [post-hoc fixed-checkpoint validation-only mechanism diagnostic](reports/preference_guide_probe_20261104/RESULTS.md):192 full-H20 episodes and18 within-state/preference/noise-level calibration cases, each16 prefixes×128 suffixes. ESS0/.5 exactly equal terminal weighting; ESS1 activates all intermediate resampling but paired deltas are noisy and do not establish benefit. Guide fit varies substantially: one weight1 late-prefix group has learned/reference correlation−.681 despite reference split-half correlation.991, while other groups are positively aligned. The targets are not universally flat, and guide discrimination/calibration needs attention. No new threshold is selected; original heldout results, source, model/config/checkpoint hashes and learner RNG/update states remain unchanged.
- [pending] Assignment utility–resource implementation and multi-instance/multi-seed replication. Neither is part of this first implementation pilot.

See the [implementation contract](docs/PREFERENCE_IMPLEMENTATION.md) for actual algorithm semantics, compute scope and approximation boundaries. Joint guidance must improve on equal-particle terminal weighting after counting extra guide cost before supporting a diffusion-specific efficiency claim. The earlier two-query branch estimator remains behind its failed cost gate; no RB or margin training sweep is planned.

The [complete preference pilot](reports/preference_pilot_20261104/RESULTS.md), [machine-readable full curves and evaluations](reports/preference_pilot_20261104/study.json), [figure](reports/preference_pilot_20261104/preference_curves.png), and [final remote evidence](reports/preference_pilot_20261104/runtime/final_remote_state.json) remain separate from all32 historical single-objective runs. Both preference workers and their controller have exited; no preference training remains live or queued. Training used CPUs24/25 and launched no GPU worker. Each fixed-budget run retains a16/20-step partial training episode after62 complete episodes; all validation and heldout episodes finish H20.

## Preference tuning v2 and GitHub upload

- [done] Re-audit parent config JSONs, complete curves and fixed-checkpoint prefix calibration. Prespecify [eight-arm tuning design](docs/PREFERENCE_TUNING_V2.md), including strong conditional sphere baselines, the temperature/beta-end factorial, grouped-absolute control and centered relative-potential loss.
- [done] Implement configurable terminal diffusion noise, grouped/centered guide learning, a two-lane bounded queue, exact config/CPU audit and backward-compatible default learner checkpoint loading. All90 tests pass, including matched grouped inputs/base-state equivalence and centered-loss gauge/scale tests.
- [done] Source`33fcff51897f7485ddce4a303d089526d0cada8e165a4c380e7c9509db6680b3`, implementation`b183ef1`, screen preparation`9855144`; all eight debug workers and eight4000-update discovery arms complete. The server lacks matplotlib; complete artifact audits and plotting passed locally. New final episode seeds600000–600007. Historical source/configs remain unchanged.
- [done] Lock validation-only selection at commit`cbc1cfe`: sphere temperature.1; diffusion temperature.1/beta-end.2, original absolute guide. Seed0 last-window returns improve290.735→359.450 and203.018→297.661 versus their matched4000-update references. Beta-end.6 is degraded and retained.
- [done] Complete all four frozen replications at new training seeds1/2. Diffusion-minus-sphere last-window differences are−69.473 and−65.725 (mean−67.599, sample SD2.650); final unseen and AUC also favor sphere. Seed0 remains discovery evidence. No tuning-effect replication claim is made without original-temperature repeats on new seeds.
- [done] Validate bitwise-equivalent base models/optimizers/replay/environment/RNG/direct curves across the guide triplet; all18 calibration references exactly match. Centered guide median relative prefix error.993 versus grouped-absolute1.611; all ESS.5/.9 validation probes still make zero intermediate resampling and equal terminal weighting. No guide gain/default promotion or larger sweep follows.
- [done] [All12-run audits, curves, per-seed comparisons and interpretation](reports/preference_tuning_20261104/RESULTS.md); all90 tests pass and12 mirrored checkpoint hashes match the server. Final remote state has zero preference workers or queued jobs; CPUs24/25 only, no GPU training.
- [done] Upload source/configs/reports and complete commit history to [Serenade-Intermezzo/SolverCell-RL](https://github.com/Serenade-Intermezzo/SolverCell-RL), branch`main`, using the verified Serenade-Intermezzo SSH identity. After GitHub reported the user-created repository as public, the user explicitly authorized public upload, overriding the initial private-repository preference. The first content commit`e44197d` matches the remote ref and all299 tracked blob SHAs match GitHub's complete tree. [Upload verification](reports/preference_tuning_20261104/runtime/github_upload.json) records this first content snapshot. Raw checkpoints remain local/server artifacts with recorded hashes. No experiment was relaunched for publication.

## Shared-prefix diffusion inference v3

- [done] Implement root-corrected terminal importance pools over actual stochastic diffusion suffixes. No parent actor/critic/guide/training code changed. One-child equality, fresh suffix noise, analytic Gaussian leaf covariance, unequal-root weights, actual cost hooks and RNG isolation pass; full suite98 tests passed.
- [done] Prespecify [fixed-checkpoint design and cost/utility gates](docs/PREFERENCE_BRANCHING_V3.md), six topologies, seven independent controls per method, validation-only selection and fresh complete final episodes on existing model seeds0/1/2.
- [done] Immutable server screen from implementation`d21a377`, source`05c3cfffb6e7ed1bad6a077085e7da70bfaef41ad88323108052a261740fcd10`. At most two CPU workers24/25, no training or GPU worker. Selection locked at`0ca1c9f` before final launches. All four workers exited0; all7680 formal episodes completed H20 and all six checkpoints remain unchanged.
- [done] [Audited results](reports/preference_branching_20261104/RESULTS.md), [interpretation](reports/preference_branching_20261104/INTERPRETATION.md), and [remote completion](reports/preference_branching_20261104/runtime/final_remote_state.json). Final deltas vs independent8:+3.986/+8.901/+4.100; actor rows64→28, median latency1.853ms→1.803ms. Tuned sphere remains stronger; branching remains optional. Zero live/queued jobs.
- [done] Publish this completed source/config/report revision to the authorized public GitHub repository; raw checkpoints remain local/server artifacts.

## Post-hoc methodology diagnosis

- [done] [Evidence and methodological decisions](reports/method_diagnosis_20261104/DIAGNOSIS.md): cross-preference rescoring reveals weak low-cohort gains and poor learned tradeoffs; same-leaf branching variance is higher despite matched-actor-budget savings.
- [done] Local read-only probes:4608 actual paths,7808 base-policy candidate continuations over two common states, and864 new complete H20 episodes. Critic ranks are useful in action-relevant goals; final additive noise damages this particular finite policy. Removing only final noise changes mean returns by+17.348/+16.710/+18.126 at equal8 actor rows. Low-cohort-only objective remains flat.
- [done] [Audits and provenance](reports/method_diagnosis_20261104/audit.json) preserve all six parent models/optimizers/training RNGs/update counts/checkpoints and scientific source. No training, server job, production sampler/default or new tuned configuration. Corrected naive crossfit-SE summary; original raw/script retained, underlying MC/Q arrays identical.
- [pending] Task controllability/frontier validation on training-side states and common fixed-endpoint actor fitting, before another online RL sweep or adaptive guidance claim.

## Priority repair v4

- [done] [Task audit](reports/priority_repair_20261104/task_audit.json):268 verified native open-cell decisions; feasible model-informed references show strong opposing gains. Freeze scales87.0718753/563.5656228 from calibration episodes only. Independent validation shows a meaningful reference tradeoff; original learned flatness was not proof of inherent uncontrollability.
- [done] Explicit `repaired_preference_config` factory, consistent normalized TD/evaluation with raw replay retained, clean terminal kernel, x0 ablation, deterministic-density rejection and changed-chain checkpoint guards. Legacy defaults remain loadable in original semantics.104 tests pass; repaired online harness debug and3 fixed-data preflights pass.
- [done] [Nine common fixed-endpoint fits](reports/priority_repair_20261104/RESULTS.md), preparation`af3f1a8`, source`eec36f7e3d63572e34f709e4343ceb0c67f3b216c8c6ba29c9dc3a7a85fdf022`:64 train/32 disjoint contexts, common268 endpoints, exact target, identical per-seed endpoint draws,30408 parameters,2000 optimizer steps each. All9 exit0 and mirrored checkpoints match. No online-RL result follows from fitting.
- [done] [Gate decision](reports/priority_repair_20261104/decision.json): sphere TV.2525, epsilon-clean.2770, x0-clean.2913 vs target-sampling floor~.072. Both diffusion variants fail preset TV<=.15; no new staged guidance or broader online sweep. X0 remains an unpromoted ablation. Zero live/queued jobs, CPU24/25 only.
- [pending] Resolve common fixed-distribution fit: direct classification positive control and known-mixture oracle score/kernel diagnostics before changing budgets or adding guidance. Finite-step fit, online credit assignment/exploration, and scalability beyond268 native actions remain unresolved. See [interpretation](reports/priority_repair_20261104/INTERPRETATION.md).
