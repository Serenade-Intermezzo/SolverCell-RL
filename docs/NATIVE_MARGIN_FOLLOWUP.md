# Native margin follow-up and fixed replication, 2026-11-04

**Status: prepared; not launched.** The bounded campaign contains 12 jobs: six margin-factorial jobs at training seed 0 and six point-versus-sphere runs at fixed margin 0.1 and training seeds 0, 1, 2. The seed-0 point run also checks equivalence of the new checkpoint storage. No scientific source is changed by this preparation.

The manifest requires source SHA256 `6eb3126bc49621d34308013091304b3fb46bb17e9c35bec7062d322cf656a66c`. Its parent native-revision campaign used `030e1c799e990310b3c2d44a1872b051baefdc141267d42bf048df5c5d1eb53b`. The intervening hashed changes concern Replay checkpoint serialization and trainer save/restore. They are intended to preserve learning, but full-run equivalence is a required check before combining historical and new-source results.

## Evidence and falsifiable hypothesis

The completed native-revision quotient runs have the following validation scores. Each uses instance seed 0, training seed 0 and margin 0.1.

| Method | Last-window return | Validation AUC |
|---|---:|---:|
| Point diffusion | 335.431249 | 293.122969 |
| Coupled cell diffusion | 289.115001 | 270.712658 |
| Independent cell diffusion | 288.647501 | 269.200626 |

These results do not show a coupling advantage. The cell and independent methods are nearly tied and both trail point diffusion. All three completed the full budget with finite states. The cell arm's 40 logged actor updates have weight ESS between 40.872 and 58.087 out of batch 64, actor gradient norm between 0.545 and 1.293, and zero cumulative projection fallback. The point and independent logs also have finite gradients, non-collapsed ESS and zero fallback. These observations do not indicate numerical divergence or failed projection as the immediate cause of the cell deficit.

An offline diagnostic on the **fixed final cell checkpoint** changes only the projection margin while retaining matched replay states, initial samples and execution perturbations:

| Projection margin | Execution changes the clean decoded action | Mean action Hamming distance | Normalized squared source-to-endpoint distance |
|---|---:|---:|---:|
| 0.1 | 0.896606 | 4.988464 | 0.773019 |
| 0.3 | 0.788452 | 3.598389 | 0.986273 |
| 0.6 | 0.603149 | 2.235596 | 1.244946 |

The diagnostic uses 512 replay states, 32 execution perturbations per center, native execution concentration kappa=28, replay seed 882013, source seed 882014 and perturbation seed 882015. Each tested margin has zero clean decode mismatches and zero fallback. The increasing distance is a tradeoff: pushing endpoints farther inside the clean decision cell improves execution robustness in this probe while giving up some short transport paths. These are offline endpoint measurements, not retrained policy returns or measurements throughout training.

The falsifiable hypothesis is that boundary-concentrated cell endpoints are vulnerable to the retained native execution noise, and that increasing projection margin improves their trained validation returns while reducing the same offline action-flip diagnostic. The healthy optimizer/projection diagnostics and the direct monotonic margin probe motivate changing margin first. They do not establish that margin is the sole cause or rule out bridge sigma. Bridge sigma therefore stays fixed at 0.5 in this experiment.

Evidence is stored in `reports/native_revision_20261104/native_mechanism_audit/`, particularly `quotient_cell_offline_margin_probe.json`, the three method reports, and `deep_completion_audit.json`. `experiments/native_margin_20261104/design.json` records the evidence file hashes, exact parent configuration hashes, checkpoint hashes, and every changed field. No heldout score is used to motivate or select this follow-up.

## Stage B: controlled margin factorial

| New arm | Parent method | Margin | Training seed | Exact parent-config change |
|---|---|---:|---:|---|
| point_m030 | native_quotient_point_diffusion | 0.3 | 0 | margin only |
| cell_m030 | native_quotient_cell_diffusion | 0.3 | 0 | margin only |
| independent_m030 | native_quotient_independent_diffusion | 0.3 | 0 | margin only |
| point_m060 | native_quotient_point_diffusion | 0.6 | 0 | margin only |
| cell_m060 | native_quotient_cell_diffusion | 0.6 | 0 | margin only |
| independent_m060 | native_quotient_independent_diffusion | 0.6 | 0 | margin only |

Each new configuration is copied from its corresponding `experiments/native_revision_20261104/quotient_*.json`. No method receives a different margin search range or budget. Point, coupled cell and independent cell retain the same native critic and policy-improvement machinery. Cell versus independent changes the pairing between initial noise and endpoint construction; cell versus point also changes the endpoint distribution, so that comparison alone cannot identify a coupling effect.

The historical margin-0.1 point/cell/independent runs provide the third margin group **only after the exact storage-equivalence check below passes**. The new point_m010_recheck is not counted as an extra independent seed or given double weight. Use the historical three-method margin-0.1 group once; its point result must equal the recheck.

Common settings are N=40, A=20, B=10, H=20, instance seed 0, 1,000 warmup interactions followed by 4,000 training interactions, batch 64, hidden width 256, learning rate 0.0004, raw observation/reward scaling, actor update every training step, native target-update schedule, 12 behavior candidates, 16 smoothed Bellman candidates, 12 sampler steps, bridge sigma 0.5, diffusion scale 1.0, epsilon 0, and endpoint jitter 0. Native Q normalization, learned alpha, lambda schedule, weight clipping, execution perturbation and trimmed Bellman reduction remain inherited. The JSON configurations and frozen source define the full protocol.

Select a **shared** margin from {0.1, 0.3, 0.6} using the mean last-window validation score across all three methods. Break an exact tie with their mean validation AUC, then the smaller margin. Last-window is the mean of the scheduled 3,500- and 4,000-step validation returns; AUC is trapezoidal over all nine validation points, divided by 4,000. Do not choose margin by the most favorable cell-minus-control gap, a single peak, final heldout return, or a method-specific search. An incomplete three-method group has no valid shared-margin score; retain failures and degraded arms.

Report the within-margin cell-minus-point and cell-minus-independent contrasts before interpreting mechanism. Improvement over the cell's own 0.1 anchor accompanied by reduced action flips supports the robustness intervention. A coupling claim additionally needs cell to outperform independent under the common chosen margin; lower transport alone is insufficient. If larger margins do not improve cell validation return, revise the boundary-fragility explanation rather than extending the grid automatically. Even a positive seed-0 result remains a tuning observation requiring separate replication.

## Fixed confirmatory replication: point versus quotient sphere

This stage is specified from completed native-revision validation evidence **before any Stage B margin result**. It cannot switch to a margin winner or choose its methods after seeing the new factorial.

| Arm | Method | Seed | Margin | Parent-config change |
|---|---|---:|---:|---|
| point_m010_recheck | native_quotient_point_diffusion | 0 | 0.1 | none; exact copy |
| sphere_m010_recheck | native_quotient_sphere | 0 | 0.1 | none; exact copy |
| point_m010_seed1 | native_quotient_point_diffusion | 1 | 0.1 | seed only |
| sphere_m010_seed1 | native_quotient_sphere | 1 | 0.1 | seed only |
| point_m010_seed2 | native_quotient_point_diffusion | 2 | 0.1 | seed only |
| sphere_m010_seed2 | native_quotient_sphere | 2 | 0.1 | seed only |

All six use the same new source, fixed bridge sigma 0.5, matched seed pairs and the inherited 4,000+1,000 interaction budget. Training seed 0 is the known discovery seed repeated under the new source. Seeds 1 and 2 are new training seeds. Report the newly added pairs separately as well as all three pairs; do not describe all three as previously unseen evidence.

The primary comparison is the point-minus-sphere last-window difference for each seed, followed by the across-seed mean and sample standard deviation of those paired differences. Report the paired AUC differences and all complete validation curves as secondary evidence. A favorable aggregate driven only by the known seed 0 does not establish replication; mixed signs must remain visible. Three pairs support a limited repeatability assessment, not a precise population estimate or a significance claim.

These are quotient-policy comparisons. The historical full-space native reference remains seed-0-only and does not become a multi-seed baseline through this replication. The equal interaction/update budget is also not equal neural evaluation cost: diffusion uses 12 network evaluations per sample while the inherited Heun flow uses 24. Report parameter counts, NFE and wall time alongside returns rather than claiming matched compute.

## Storage equivalence and completion gates

`point_m010_recheck.json` is byte-for-byte identical to the parent `quotient_point.json`; its canonical config SHA256 is `dd4d14084aa421b99eb1193477e46f0d6cfc4a0ce66363b55c434d3a49331189`. The original full training run is repeated from initialization, not resumed from its final checkpoint. Its complete validation episode-return vectors and lengths must match the old run, and final scientific state must match exactly: actor and targets, critics and targets, all optimizers, alpha/Q statistics, RNGs, environment state, counters and replay contents after converting both storage formats to the same runtime Replay representation.

Exclude only source/artifact identity, timestamps/process identity, the explicit checkpoint storage/schema representation, wall/evaluation/projection timings and derived throughput from equality. Do not hide parameter, metric, RNG or trajectory differences behind a numerical tolerance. Compare the old nine-point validation curve ending at 345.2949985265732 and last-window 335.431248986721. The sphere seed-0 recheck is also an exact parent-config copy and provides an additional available control. The existing compact-storage unit tests and 95.0376% replay-payload size reduction are engineering evidence; they do not replace this end-to-end equality audit.

If the point equivalence check fails, do not pool historical margin-0.1 anchors with the new factorial. Diagnose the runtime/source difference first; within-new-source contrasts can still be reported separately. No source change or silent rerun is part of this fixed manifest.

Each completed job must have matching config/source/instance identities, worker exit 0, 5,000 environment steps, 4,000 critic updates, 4,000 actor updates, eight native pretraining updates, 250 complete H20 episodes, nine finite validation evaluations at steps 0:500:4000, one final 50-episode evaluation, finite saved scientific state, and a matching final checkpoint. Validation uses seeds 100000–100019. The runner records heldout seeds 200000–200049, but those scores cannot select margin, method, checkpoint or the next sweep. All jobs remain visible, including failures.

The whole campaign uses one fixed instance. The six margin-factorial jobs use only training seed 0. The point/sphere replication uses three training seeds. None of these results alone establishes cross-instance generalization or full-budget convergence.

## Prepared dispatch

`experiments/native_margin_20261104/manifest.json` lists exactly 12 jobs, each with its resolved-config SHA256 and distinct output directory. It permits at most four one-thread CPU workers, bound to CPUs 24, 25, 26 and 27. Total authorized work is 60,000 environment interactions, including 48,000 training updates and 12,000 warmup interactions; no automatic expansion is configured.

The queue starts with the two seed-0 rechecks and the fixed point/sphere seed pairs, then the six margin-factorial configurations. Independent jobs may run concurrently; historical-anchor pooling still waits for the equality gate. A future dispatcher must verify available slots and avoid overlap with unrelated or previous campaign workers. This preparation does not acquire CPUs or launch a controller.

After freezing the required source and deploying it, the prepared command is:

```bash
/home/li/SolverCell-RL/.venv/bin/python tools/campaign.py run \
  --manifest experiments/native_margin_20261104/manifest.json
```

Run from the new frozen deployment root. Existing native-revision source and outputs remain immutable.
