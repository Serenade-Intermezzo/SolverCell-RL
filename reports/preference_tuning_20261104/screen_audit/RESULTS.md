# Preference Intervention: completed implementation pilot

All reported runs passed config/source/exit, full-curve, checkpoint, reward-vector and trajectory audits.

**Scope:** new two-cohort preference task on the unchanged LSFlow Intervention dynamics and native quotient decoder. This is one training seed and one instance at a short budget. It does not establish diffusion superiority or a Pareto frontier.

The actor NFE per sample is matched: spherical Heun uses half as many integration intervals as Gaussian diffusion. Joint potential fitting and SMC potential evaluations add cost and are recorded separately. Terminal weighting and SMC each use the same configured number of base particles and terminal action/value queries. The direct policy uses one sample and is a cheaper reference.

Checkpoint learner counters cover training, guide fitting, behavior, validation and heldout, and exclude warmup decoding and independent behavioral baselines. They are forward-row counts, not complete experiment FLOPs. Per-evaluation compute totals are additive counts; ESS and log-normalizer diagnostics are reported separately and are not costs. The evaluation actor_nfe field aliases sampling nfe and must not be added to it.

The base-policy critic uses the same relabeled preference at both ends of the Bellman target. Guidance is an inference-time base-policy-value improvement; it is not an exact critic evaluation of the guided policy. Finite suffix log-mean-exp labels and finite-particle SMC are approximations. No target reward, checkpoint or configuration was selected using heldout preferences.

| Model | Updates | Last-window direct validation | Direct validation AUC | Wall seconds |
|---|---:|---:|---:|---:|
| sphere_t050 | 4000 | 290.735 | 210.392 | 77.0 |
| sphere_t010 | 4000 | 359.450 | 297.633 | 77.0 |
| diff_b020_t050 | 4000 | 203.018 | 173.936 | 98.0 |
| diff_b060_t050 | 4000 | 180.674 | 162.800 | 98.2 |
| diff_b020_t010 | 4000 | 297.661 | 247.723 | 98.5 |
| diff_b060_t010 | 4000 | 266.412 | 228.967 | 98.9 |
| group_b060_t010 | 4000 | 266.412 | 228.967 | 94.8 |
| center_b060_t010 | 4000 | 266.412 | 228.967 | 97.2 |

Training uses a fixed transition/update budget. Any unfinished training episode is retained in the final checkpoint; every evaluation trajectory completes the full horizon.

| Model | Complete training episodes (including warmup) | Partial final episode | Last-window evaluation points |
|---|---:|---:|---:|
| sphere_t050 | 212 | 16/20 steps | 2 |
| sphere_t010 | 212 | 16/20 steps | 2 |
| diff_b020_t050 | 212 | 16/20 steps | 2 |
| diff_b060_t050 | 212 | 16/20 steps | 2 |
| diff_b020_t010 | 212 | 16/20 steps | 2 |
| diff_b060_t010 | 212 | 16/20 steps | 2 |
| group_b060_t010 | 212 | 16/20 steps | 2 |
| center_b060_t010 | 212 | 16/20 steps | 2 |

## Unseen preference interpolation: final checkpoints

Raw component returns use original reward units (N times per-capita cohort utility). At equal weights the scalarized return equals the original return. These rows use common final episode seeds and are descriptive; no cross-seed uncertainty is inferred.

| Model | Mode | Low-response weight | Low-group return | High-group return | Scalarized return |
|---|---|---:|---:|---:|---:|
| sphere_t050 | direct | 0.125 | 120.463 | 468.625 | 425.105 |
| sphere_t050 | terminal_weighting | 0.125 | 119.450 | 543.862 | 490.811 |
| sphere_t050 | direct | 0.375 | 121.800 | 408.888 | 301.230 |
| sphere_t050 | terminal_weighting | 0.375 | 119.288 | 481.600 | 345.733 |
| sphere_t050 | direct | 0.625 | 123.500 | 348.950 | 208.044 |
| sphere_t050 | terminal_weighting | 0.625 | 119.813 | 405.300 | 226.870 |
| sphere_t050 | direct | 0.875 | 124.588 | 263.538 | 141.956 |
| sphere_t050 | terminal_weighting | 0.875 | 124.163 | 295.475 | 145.577 |
| sphere_t010 | direct | 0.125 | 121.675 | 577.487 | 520.511 |
| sphere_t010 | terminal_weighting | 0.125 | 119.650 | 607.487 | 546.508 |
| sphere_t010 | direct | 0.375 | 121.650 | 576.137 | 405.705 |
| sphere_t010 | terminal_weighting | 0.375 | 119.575 | 601.737 | 420.927 |
| sphere_t010 | direct | 0.625 | 121.925 | 567.587 | 289.048 |
| sphere_t010 | terminal_weighting | 0.625 | 119.625 | 579.237 | 291.980 |
| sphere_t010 | direct | 0.875 | 121.788 | 432.863 | 160.672 |
| sphere_t010 | terminal_weighting | 0.875 | 119.400 | 474.150 | 163.744 |
| diff_b020_t050 | direct | 0.125 | 122.313 | 300.125 | 277.898 |
| diff_b020_t050 | terminal_weighting | 0.125 | 122.312 | 402.763 | 367.706 |
| diff_b020_t050 | smc | 0.125 | 122.312 | 402.763 | 367.706 |
| diff_b020_t050 | direct | 0.375 | 122.425 | 251.112 | 202.855 |
| diff_b020_t050 | terminal_weighting | 0.375 | 123.700 | 318.400 | 245.388 |
| diff_b020_t050 | smc | 0.375 | 123.700 | 318.400 | 245.388 |
| diff_b020_t050 | direct | 0.625 | 124.150 | 219.262 | 159.817 |
| diff_b020_t050 | terminal_weighting | 0.625 | 125.275 | 261.250 | 176.266 |
| diff_b020_t050 | smc | 0.625 | 125.275 | 261.250 | 176.266 |
| diff_b020_t050 | direct | 0.875 | 126.612 | 171.213 | 132.187 |
| diff_b020_t050 | terminal_weighting | 0.875 | 125.312 | 209.138 | 135.791 |
| diff_b020_t050 | smc | 0.875 | 125.312 | 209.138 | 135.791 |
| diff_b060_t050 | direct | 0.125 | 121.075 | 259.750 | 242.416 |
| diff_b060_t050 | terminal_weighting | 0.125 | 121.225 | 349.325 | 320.812 |
| diff_b060_t050 | smc | 0.125 | 121.225 | 349.325 | 320.812 |
| diff_b060_t050 | direct | 0.375 | 124.350 | 230.900 | 190.944 |
| diff_b060_t050 | terminal_weighting | 0.375 | 124.000 | 293.987 | 230.242 |
| diff_b060_t050 | smc | 0.375 | 124.000 | 293.987 | 230.242 |
| diff_b060_t050 | direct | 0.625 | 124.388 | 204.750 | 154.523 |
| diff_b060_t050 | terminal_weighting | 0.625 | 125.738 | 246.175 | 170.902 |
| diff_b060_t050 | smc | 0.625 | 125.738 | 246.175 | 170.902 |
| diff_b060_t050 | direct | 0.875 | 126.675 | 186.988 | 134.214 |
| diff_b060_t050 | terminal_weighting | 0.875 | 128.750 | 195.800 | 137.131 |
| diff_b060_t050 | smc | 0.875 | 128.750 | 195.800 | 137.131 |
| diff_b020_t010 | direct | 0.125 | 121.763 | 480.000 | 435.220 |
| diff_b020_t010 | terminal_weighting | 0.125 | 120.863 | 528.587 | 477.622 |
| diff_b020_t010 | smc | 0.125 | 120.863 | 528.587 | 477.622 |
| diff_b020_t010 | direct | 0.375 | 119.800 | 443.900 | 322.362 |
| diff_b020_t010 | terminal_weighting | 0.375 | 120.737 | 474.350 | 341.745 |
| diff_b020_t010 | smc | 0.375 | 120.737 | 474.350 | 341.745 |
| diff_b020_t010 | direct | 0.625 | 120.163 | 397.125 | 224.023 |
| diff_b020_t010 | terminal_weighting | 0.625 | 121.363 | 391.612 | 222.706 |
| diff_b020_t010 | smc | 0.625 | 121.363 | 391.612 | 222.706 |
| diff_b020_t010 | direct | 0.875 | 122.363 | 350.700 | 150.905 |
| diff_b020_t010 | terminal_weighting | 0.875 | 125.262 | 353.363 | 153.775 |
| diff_b020_t010 | smc | 0.875 | 125.262 | 353.363 | 153.775 |
| diff_b060_t010 | direct | 0.125 | 120.163 | 427.387 | 388.984 |
| diff_b060_t010 | terminal_weighting | 0.125 | 121.700 | 471.375 | 427.666 |
| diff_b060_t010 | smc | 0.125 | 121.700 | 471.375 | 427.666 |
| diff_b060_t010 | direct | 0.375 | 120.150 | 381.750 | 283.650 |
| diff_b060_t010 | terminal_weighting | 0.375 | 122.238 | 439.038 | 320.238 |
| diff_b060_t010 | smc | 0.375 | 122.238 | 439.038 | 320.238 |
| diff_b060_t010 | direct | 0.625 | 121.000 | 340.100 | 203.163 |
| diff_b060_t010 | terminal_weighting | 0.625 | 122.125 | 375.700 | 217.216 |
| diff_b060_t010 | smc | 0.625 | 122.125 | 375.700 | 217.216 |
| diff_b060_t010 | direct | 0.875 | 121.050 | 305.813 | 144.145 |
| diff_b060_t010 | terminal_weighting | 0.875 | 123.213 | 309.900 | 146.548 |
| diff_b060_t010 | smc | 0.875 | 123.213 | 309.900 | 146.548 |
| group_b060_t010 | direct | 0.125 | 120.163 | 427.387 | 388.984 |
| group_b060_t010 | terminal_weighting | 0.125 | 121.700 | 471.375 | 427.666 |
| group_b060_t010 | smc | 0.125 | 121.700 | 471.375 | 427.666 |
| group_b060_t010 | direct | 0.375 | 120.150 | 381.750 | 283.650 |
| group_b060_t010 | terminal_weighting | 0.375 | 122.238 | 439.038 | 320.238 |
| group_b060_t010 | smc | 0.375 | 122.238 | 439.038 | 320.238 |
| group_b060_t010 | direct | 0.625 | 121.000 | 340.100 | 203.163 |
| group_b060_t010 | terminal_weighting | 0.625 | 122.125 | 375.700 | 217.216 |
| group_b060_t010 | smc | 0.625 | 122.125 | 375.700 | 217.216 |
| group_b060_t010 | direct | 0.875 | 121.050 | 305.813 | 144.145 |
| group_b060_t010 | terminal_weighting | 0.875 | 123.213 | 309.900 | 146.548 |
| group_b060_t010 | smc | 0.875 | 123.213 | 309.900 | 146.548 |
| center_b060_t010 | direct | 0.125 | 120.163 | 427.387 | 388.984 |
| center_b060_t010 | terminal_weighting | 0.125 | 121.700 | 471.375 | 427.666 |
| center_b060_t010 | smc | 0.125 | 121.700 | 471.375 | 427.666 |
| center_b060_t010 | direct | 0.375 | 120.150 | 381.750 | 283.650 |
| center_b060_t010 | terminal_weighting | 0.375 | 122.238 | 439.038 | 320.238 |
| center_b060_t010 | smc | 0.375 | 122.238 | 439.038 | 320.238 |
| center_b060_t010 | direct | 0.625 | 121.000 | 340.100 | 203.163 |
| center_b060_t010 | terminal_weighting | 0.625 | 122.125 | 375.700 | 217.216 |
| center_b060_t010 | smc | 0.625 | 122.125 | 375.700 | 217.216 |
| center_b060_t010 | direct | 0.875 | 121.050 | 305.813 | 144.145 |
| center_b060_t010 | terminal_weighting | 0.875 | 123.213 | 309.900 | 146.548 |
| center_b060_t010 | smc | 0.875 | 123.213 | 309.900 | 146.548 |

Prefer measured response curves and compute totals over a claim from one favorable weight. Failure to change cohort allocation is a controllability failure even if average utility is high. No monotonicity or guidance-efficiency claim is inferred merely from low regression loss.

[Complete machine-readable results and compute totals](study.json) · [Curves](preference_curves.png)
