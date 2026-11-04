# Fixed-checkpoint preference-guide diagnostic

Post-run report note: interpretation and the threshold-0.5 equality audit below were expanded after execution from unchanged saved artifacts. The analysis script is preserved at its exact executed SHA256 recorded in summary.json.

Completed post-hoc validation-only probe. No training, checkpoint selection, heldout tuning, source/config edits, efficiency claim, or method-superiority claim.

Scientific source: `9b8a1b044b789a34ca7d3898bb3613db85f9e0008163287cdde5cb1b4b626b8c`. Checkpoint: `/home/serenade/AI/src/RL/SolverCell-RL/outputs/preference_pilot_20261104/diffusion/checkpoint.pt`; SHA256 `585cd4da5857ac72f704e5c4f977ece10fb01956c4f922bf8ae24d52aabf3861`.

All arms use the same trained target actor, target vector critics, guide, eight particles, eight denoising steps, and fresh paired episode seeds 400000–400015. Only validation preferences 0, 0.5, 1 are queried. ESS threshold changes are read-only inference diagnostics; none is selected as a new configuration.

| Low-response weight | Inference | Return mean ± episode SD | Resampling events | Actor NFE rows | Guide rows | Q rows | Decoder rows | Wall s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | terminal_weighting | 246.675000 ± 67.478288 | 0 | 20480 | 0 | 5760 | 2880 | 0.490 |
| 0 | smc_ess_0 | 246.675000 ± 67.478288 | 0 | 20480 | 20480 | 5760 | 2880 | 0.826 |
| 0 | smc_ess_0.5 | 246.675000 ± 67.478288 | 0 | 20480 | 20480 | 5760 | 2880 | 0.816 |
| 0 | smc_ess_1 | 250.075001 ± 55.274449 | 2560 | 20480 | 20480 | 5760 | 2880 | 0.985 |
| 0.5 | terminal_weighting | 163.331250 ± 26.310776 | 0 | 20480 | 0 | 5760 | 2880 | 0.496 |
| 0.5 | smc_ess_0 | 163.331250 ± 26.310776 | 0 | 20480 | 20480 | 5760 | 2880 | 0.808 |
| 0.5 | smc_ess_0.5 | 163.331250 ± 26.310776 | 0 | 20480 | 20480 | 5760 | 2880 | 0.807 |
| 0.5 | smc_ess_1 | 163.159375 ± 28.327197 | 2560 | 20480 | 20480 | 5760 | 2880 | 1.026 |
| 1 | terminal_weighting | 122.962500 ± 6.071669 | 0 | 20480 | 0 | 5760 | 2880 | 0.488 |
| 1 | smc_ess_0 | 122.962500 ± 6.071669 | 0 | 20480 | 20480 | 5760 | 2880 | 0.817 |
| 1 | smc_ess_0.5 | 122.962500 ± 6.071669 | 0 | 20480 | 20480 | 5760 | 2880 | 0.822 |
| 1 | smc_ess_1 | 126.031250 ± 11.541849 | 2560 | 20480 | 20480 | 5760 | 2880 | 0.978 |

Actor NFE, guide, Q and decoder counts are processed rows, not Python invocations. Episode SD describes rollout variability for one fixed checkpoint, not training-seed stability. SMC adds guide calls even if it never resamples. Threshold 1 resamples almost every strictly nonuniform population; this is a diagnostic intervention, not an efficiency recommendation.

Threshold-0 terminal-telescoping check: **True** (all 48 paired episodes have exactly equal component returns, scalarized returns, original returns, episode lengths, action counts, and unique-action counts). The default threshold 0.5 also has zero resampling and exactly the same episode results. Across the three preferences, each SMC arm incurs 61,440 additional guide rows. Threshold 1 resamples all 7,680 possible intermediate populations.

| Weight | Threshold-1 minus terminal mean | Paired episode mean SE |
|---:|---:|---:|
| 0 | +3.400000 | 24.471756 |
| 0.5 | -0.171875 | 9.987168 |
| 1 | +3.068750 | 3.683406 |

These descriptive differences do not establish benefit. They use one checkpoint and 16 episodes per preference; additional ancestor draws change the sampler random stream. The environment and initial generator seeds remain paired. No threshold is selected.

## Within-group actual-suffix calibration

Each row fixes one replay observation, one preference and one path index, then compares 16 fresh actual prefixes. Each prefix has 128 independent fresh suffixes. The reference is log-mean-exp of the same learned terminal utility used during deployment. It can diagnose guide fit to the fixed critic, not critic correctness or actual return calibration.

| Replay row | Weight | Index | Learned range | MC reference range | Within-group r | Split-half r | Mean MC SE | Centered RMSE / constant RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0.34495 | 0.15061 | 0.641 | 0.789 | 0.01589 | 1.925 |
| 0 | 0 | 4 | 0.26490 | 0.23137 | 0.571 | 0.853 | 0.01522 | 0.888 |
| 0 | 0 | 7 | 0.15258 | 0.53279 | 0.739 | 0.991 | 0.01054 | 0.803 |
| 0 | 0.5 | 0 | 0.40803 | 0.07455 | 0.713 | 0.725 | 0.00872 | 4.074 |
| 0 | 0.5 | 4 | 0.25923 | 0.11604 | 0.352 | 0.927 | 0.00838 | 1.540 |
| 0 | 0.5 | 7 | 0.17001 | 0.28568 | 0.753 | 0.997 | 0.00491 | 0.699 |
| 0 | 1 | 0 | 0.31618 | 0.04309 | -0.274 | 0.596 | 0.00524 | 7.418 |
| 0 | 1 | 4 | 0.26423 | 0.09740 | -0.590 | 0.836 | 0.00512 | 3.960 |
| 0 | 1 | 7 | 0.17294 | 0.15799 | -0.681 | 0.991 | 0.00364 | 1.875 |
| 1255 | 0 | 0 | 0.50286 | 0.52765 | 0.903 | 0.846 | 0.03442 | 0.447 |
| 1255 | 0 | 4 | 0.28345 | 0.60587 | 0.931 | 0.973 | 0.03142 | 0.671 |
| 1255 | 0 | 7 | 0.13031 | 0.80472 | 0.579 | 0.989 | 0.02206 | 0.925 |
| 1255 | 0.5 | 0 | 0.34968 | 0.16270 | 0.783 | 0.748 | 0.01930 | 1.226 |
| 1255 | 0.5 | 4 | 0.29942 | 0.43520 | 0.865 | 0.978 | 0.01695 | 0.568 |
| 1255 | 0.5 | 7 | 0.13325 | 0.61499 | 0.507 | 0.991 | 0.01025 | 0.892 |
| 1255 | 1 | 0 | 0.26278 | 0.03631 | -0.094 | 0.798 | 0.00601 | 6.335 |
| 1255 | 1 | 4 | 0.25962 | 0.08256 | 0.709 | 0.907 | 0.00536 | 1.816 |
| 1255 | 1 | 7 | 0.11960 | 0.17597 | 0.501 | 0.985 | 0.00371 | 0.911 |

The reference is not universally flat. For replay row 0, weight 1, index 7, learned-versus-reference correlation is −0.681 while the independent MC halves correlate 0.991: the measured conditional ranking is wrong in this particular case. Other groups are better: row 1255, weight 0, index 4 has correlations 0.931 and 0.973. Thus absence of default resampling alone would miss heterogeneous guide fit. In these sampled groups, learned variation tends to shrink toward the terminal index while conditional reference variation grows; this is a diagnostic observation, not a population statement or a selected new training recipe.

Log-mean-exp with 128 suffixes has finite-sample noise and downward bias. The split halves use 64 suffixes each; low split-half correlation limits interpretation of a low learned-guide correlation. Group mean offsets are removed for centered RMSE because common additive log-potential offsets do not change normalized particle weights. No cross-state pooled correlation is reported. No correlation threshold is used for tuning.

## Artifacts and scope

- `summary.json`: provenance, exact aggregate statistics, paired episode differences, all calibration prefixes and estimates.
- `episodes.jsonl`: every original single-episode `evaluate_preference` result, including returns, sampler diagnostics and compute.
- `calibration_terminal_logits.npz`: all 128 raw suffix terminal logweights for every prefix, sufficient to recompute MC reference statistics.
- `design.json`: intervention and checks written before running.

Total diagnostic wall time: 10.162 s; process CPU time: 10.219 s; maximum resident set: 634972 KiB; CPU threads: 1.

The scientific source, original config, original checkpoint, all model tensors, learner-owned RNGs and update counts passed before/after immutability checks. Evaluation counters alone are advanced in this separate analysis process. This probe does not update the completed pilot results.
