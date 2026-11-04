# Preference Intervention: completed implementation pilot

All reported runs passed config/source/exit, full-curve, checkpoint, reward-vector and trajectory audits.

**Scope:** new two-cohort preference task on the unchanged LSFlow Intervention dynamics and native quotient decoder. This is one training seed and one instance at a short budget. It does not establish diffusion superiority or a Pareto frontier.

The actor NFE per sample is matched: spherical Heun uses half as many integration intervals as Gaussian diffusion. Joint potential fitting and SMC potential evaluations add cost and are recorded separately. Terminal weighting and SMC each use the same configured number of base particles and terminal action/value queries. The direct policy uses one sample and is a cheaper reference.

Checkpoint learner counters cover training, guide fitting, behavior, validation and heldout, and exclude warmup decoding and independent behavioral baselines. They are forward-row counts, not complete experiment FLOPs. Per-evaluation compute totals are additive counts; ESS and log-normalizer diagnostics are reported separately and are not costs. The evaluation actor_nfe field aliases sampling nfe and must not be added to it.

The base-policy critic uses the same relabeled preference at both ends of the Bellman target. Guidance is an inference-time base-policy-value improvement; it is not an exact critic evaluation of the guided policy. Finite suffix log-mean-exp labels and finite-particle SMC are approximations. No target reward, checkpoint or configuration was selected using heldout preferences.

| Model | Updates | Last-window direct validation | Direct validation AUC | Wall seconds |
|---|---:|---:|---:|---:|
| sphere | 1000 | 142.808 | 131.702 | 21.5 |
| diffusion | 1000 | 149.675 | 139.749 | 27.9 |

Training uses a fixed transition/update budget. Any unfinished training episode is retained in the final checkpoint; every evaluation trajectory completes the full horizon.

| Model | Complete training episodes (including warmup) | Partial final episode | Last-window evaluation points |
|---|---:|---:|---:|
| sphere | 62 | 16/20 steps | 1 |
| diffusion | 62 | 16/20 steps | 1 |

## Unseen preference interpolation: final checkpoints

Raw component returns use original reward units (N times per-capita cohort utility). At equal weights the scalarized return equals the original return. These rows use common final episode seeds and are descriptive; no cross-seed uncertainty is inferred.

| Model | Mode | Low-response weight | Low-group return | High-group return | Scalarized return |
|---|---|---:|---:|---:|---:|
| sphere | direct | 0.125 | 126.850 | 197.962 | 189.073 |
| sphere | terminal_weighting | 0.125 | 125.950 | 237.975 | 223.972 |
| sphere | direct | 0.375 | 126.825 | 188.262 | 165.223 |
| sphere | terminal_weighting | 0.375 | 125.950 | 222.875 | 186.528 |
| sphere | direct | 0.625 | 127.738 | 168.550 | 143.042 |
| sphere | terminal_weighting | 0.625 | 130.100 | 190.475 | 152.741 |
| sphere | direct | 0.875 | 129.188 | 166.100 | 133.802 |
| sphere | terminal_weighting | 0.875 | 131.450 | 174.750 | 136.862 |
| diffusion | direct | 0.125 | 123.950 | 152.212 | 148.680 |
| diffusion | terminal_weighting | 0.125 | 129.275 | 233.100 | 220.122 |
| diffusion | smc | 0.125 | 129.275 | 233.100 | 220.122 |
| diffusion | direct | 0.375 | 123.913 | 151.925 | 141.420 |
| diffusion | terminal_weighting | 0.375 | 130.600 | 195.562 | 171.202 |
| diffusion | smc | 0.375 | 130.600 | 195.562 | 171.202 |
| diffusion | direct | 0.625 | 123.950 | 145.850 | 132.163 |
| diffusion | terminal_weighting | 0.625 | 131.038 | 170.575 | 145.864 |
| diffusion | smc | 0.625 | 131.038 | 170.575 | 145.864 |
| diffusion | direct | 0.875 | 123.963 | 145.837 | 126.697 |
| diffusion | terminal_weighting | 0.875 | 130.838 | 163.062 | 134.866 |
| diffusion | smc | 0.875 | 130.838 | 163.062 | 134.866 |

Prefer measured response curves and compute totals over a claim from one favorable weight. Failure to change cohort allocation is a controllability failure even if average utility is high. No monotonicity or guidance-efficiency claim is inferred merely from low regression loss.

[Complete machine-readable results and compute totals](study.json) · [Curves](preference_curves.png)
