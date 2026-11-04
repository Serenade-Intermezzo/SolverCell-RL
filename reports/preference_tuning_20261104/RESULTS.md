# Preference tuning: completed controlled study

All12 runs passed complete config/source/exit/checkpoint/curve/trajectory audits. The screen used training seed0; selected configurations were frozen using validation only before new-seed replication or final-result reporting. The two new training seeds are1/2, on the same instance. No statistical significance or cross-instance generalization is claimed.

Scientific source: `33fcff51897f7485ddce4a303d089526d0cada8e165a4c380e7c9509db6680b3`. The prespecified design is [here](../../docs/PREFERENCE_TUNING_V2.md).

## Complete discovery screen

All arms use4000 updates,256 warmup,9 full validation checkpoints,8 actor NFE and the same validation/final episode seed ranges. Last-window means checkpoints3500/4000. Final unseen-weight means are descriptive and did not select configurations.

| Arm | Direct validation last-window | Validation AUC | Final unseen direct | Wall seconds |
|---|---:|---:|---:|---:|
| sphere_t050 | 290.735 | 210.392 | 269.084 | 77.0 |
| sphere_t010 | 359.450 | 297.633 | 343.984 | 77.0 |
| diff_b020_t050 | 203.018 | 173.936 | 193.189 | 98.0 |
| diff_b060_t050 | 180.674 | 162.800 | 180.524 | 98.2 |
| diff_b020_t010 | 297.661 | 247.723 | 283.128 | 98.5 |
| diff_b060_t010 | 266.412 | 228.967 | 254.986 | 98.9 |
| group_b060_t010 | 266.412 | 228.967 | 254.986 | 94.8 |
| center_b060_t010 | 266.412 | 228.967 | 254.986 | 97.2 |

## Frozen validation-only selection

- diffusion: `diff_b020_t010` — actor temperature0.1, diffusion beta-end0.2; guide objective remains original absolute MSE.
- sphere: `sphere_t010` — actor temperature0.1, diffusion beta-end0.2; guide objective remains original absolute MSE.

Nonreference promotion required at least2% last-window gain and AUC at least98% of method reference. This is a discovery rule, not a scientific significance threshold. Sphere and diffusion each retain their strongest eligible validation setting. The grouped/centered guide arms do not add independent base candidates.

## Fixed configurations on discovery and new training seeds

| Training seed | Scope | Sphere last-window | Diffusion last-window | Diffusion minus sphere | Sphere AUC | Diffusion AUC | Final unseen direct difference |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | discovery | 359.450 | 297.661 | -61.789 | 297.633 | 247.723 | -60.856 |
| 1 | new_training_seed | 356.044 | 286.571 | -69.473 | 278.835 | 231.599 | -71.977 |
| 2 | new_training_seed | 361.147 | 295.422 | -65.725 | 293.938 | 242.122 | -61.295 |

Across only the two new training seeds, last-window diffusion-minus-sphere mean is-67.599, sample SD2.650. Final unseen direct difference mean is-66.636, sample SD7.553. These are small-n descriptive summaries, not confidence intervals. Per-episode observations within one trained model are not independent training seeds.

## Relative guide mechanism

The fixed beta-end.6/temperature.1 triplet has bitwise-identical base models, optimizers, training environment/replay/RNG and direct validation trajectories. Its18 actual-suffix calibration reference groups are also identical across guide losses. Original vs grouped absolute examines grouped sampling; grouped absolute vs centered isolates the objective. This triplet is a mechanism fixture and is not automatically the selected base configuration.

[All calibration cases, paired validation returns, resampling and compute](guide_check/RESULTS.md). Learned-critic suffix calibration does not establish real-return accuracy. SMC adds guide work to the same particle/terminal-query budget. No guide default or additional sweep is selected from final results.

## Reproducibility and limits

- [Selection lock](selection.json); [full machine-readable study](study.json); [curves](tuning_and_replication.png); [PDF](tuning_and_replication.pdf).
- [Eight-arm audit](artifact_audits/screen/RESULTS.md); [seed1 audit](artifact_audits/seed1/RESULTS.md); [seed2 audit](artifact_audits/seed2/RESULTS.md).
- [90-test log](tests.log); [preflight queue audit](preflight/RESULTS.md); runtime provenance and final remote state are in`runtime/`.
- Fixed transition budgets retain partial final training episodes in checkpoints; every validation/final trajectory completes H20.
- Equal actor NFE is not equal full compute. Guide fitting, suffix labels, Q/decoder calls and measured wall time are reported. Counters exclude independent baselines/warmup decoding.
- Fresh final seeds600000–600007 and unseen interpolation weights do not participate in configuration selection. One benchmark instance and two new seeds limit conclusions.
- All failed/degraded arms are retained. Original historical manifests and sources remain immutable. No larger guide sweep follows without the declared mechanism gate.
