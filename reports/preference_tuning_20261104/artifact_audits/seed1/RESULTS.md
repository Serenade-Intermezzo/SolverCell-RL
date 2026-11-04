# Preference Intervention: completed implementation pilot

All reported runs passed config/source/exit, full-curve, checkpoint, reward-vector and trajectory audits.

**Scope:** new two-cohort preference task on the unchanged LSFlow Intervention dynamics and native quotient decoder. This is one training seed and one instance at a short budget. It does not establish diffusion superiority or a Pareto frontier.

The actor NFE per sample is matched: spherical Heun uses half as many integration intervals as Gaussian diffusion. Joint potential fitting and SMC potential evaluations add cost and are recorded separately. Terminal weighting and SMC each use the same configured number of base particles and terminal action/value queries. The direct policy uses one sample and is a cheaper reference.

Checkpoint learner counters cover training, guide fitting, behavior, validation and heldout, and exclude warmup decoding and independent behavioral baselines. They are forward-row counts, not complete experiment FLOPs. Per-evaluation compute totals are additive counts; ESS and log-normalizer diagnostics are reported separately and are not costs. The evaluation actor_nfe field aliases sampling nfe and must not be added to it.

The base-policy critic uses the same relabeled preference at both ends of the Bellman target. Guidance is an inference-time base-policy-value improvement; it is not an exact critic evaluation of the guided policy. Finite suffix log-mean-exp labels and finite-particle SMC are approximations. No target reward, checkpoint or configuration was selected using heldout preferences.

| Model | Updates | Last-window direct validation | Direct validation AUC | Wall seconds |
|---|---:|---:|---:|---:|
| sphere_seed1 | 4000 | 356.044 | 278.835 | 79.8 |
| diffusion_seed1 | 4000 | 286.571 | 231.599 | 99.2 |

Training uses a fixed transition/update budget. Any unfinished training episode is retained in the final checkpoint; every evaluation trajectory completes the full horizon.

| Model | Complete training episodes (including warmup) | Partial final episode | Last-window evaluation points |
|---|---:|---:|---:|
| sphere_seed1 | 212 | 16/20 steps | 2 |
| diffusion_seed1 | 212 | 16/20 steps | 2 |

## Unseen preference interpolation: final checkpoints

Raw component returns use original reward units (N times per-capita cohort utility). At equal weights the scalarized return equals the original return. These rows use common final episode seeds and are descriptive; no cross-seed uncertainty is inferred.

| Model | Mode | Low-response weight | Low-group return | High-group return | Scalarized return |
|---|---|---:|---:|---:|---:|
| sphere_seed1 | direct | 0.125 | 119.638 | 611.525 | 550.039 |
| sphere_seed1 | terminal_weighting | 0.125 | 119.825 | 619.450 | 556.997 |
| sphere_seed1 | direct | 0.375 | 119.650 | 589.750 | 413.462 |
| sphere_seed1 | terminal_weighting | 0.375 | 119.713 | 606.587 | 424.009 |
| sphere_seed1 | direct | 0.625 | 119.638 | 536.000 | 275.773 |
| sphere_seed1 | terminal_weighting | 0.625 | 119.963 | 567.137 | 287.653 |
| sphere_seed1 | direct | 0.875 | 120.762 | 413.400 | 157.342 |
| sphere_seed1 | terminal_weighting | 0.875 | 120.550 | 439.612 | 160.433 |
| diffusion_seed1 | direct | 0.125 | 120.288 | 486.675 | 440.877 |
| diffusion_seed1 | terminal_weighting | 0.125 | 120.588 | 501.912 | 454.247 |
| diffusion_seed1 | smc | 0.125 | 120.588 | 501.912 | 454.247 |
| diffusion_seed1 | direct | 0.375 | 120.163 | 421.975 | 308.795 |
| diffusion_seed1 | terminal_weighting | 0.375 | 121.463 | 459.712 | 332.869 |
| diffusion_seed1 | smc | 0.375 | 121.463 | 459.712 | 332.869 |
| diffusion_seed1 | direct | 0.625 | 122.513 | 365.987 | 213.816 |
| diffusion_seed1 | terminal_weighting | 0.625 | 121.825 | 380.300 | 218.753 |
| diffusion_seed1 | smc | 0.625 | 121.825 | 380.300 | 218.753 |
| diffusion_seed1 | direct | 0.875 | 121.613 | 310.488 | 145.222 |
| diffusion_seed1 | terminal_weighting | 0.875 | 124.450 | 315.975 | 148.391 |
| diffusion_seed1 | smc | 0.875 | 124.450 | 315.975 | 148.391 |

Prefer measured response curves and compute totals over a claim from one favorable weight. Failure to change cohort allocation is a controllability failure even if average utility is high. No monotonicity or guidance-efficiency claim is inferred merely from low regression loss.

[Complete machine-readable results and compute totals](study.json) · [Curves](preference_curves.png)
