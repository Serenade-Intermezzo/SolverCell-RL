# Preference Intervention: completed implementation pilot

All reported runs passed config/source/exit, full-curve, checkpoint, reward-vector and trajectory audits.

**Scope:** new two-cohort preference task on the unchanged LSFlow Intervention dynamics and native quotient decoder. This is one training seed and one instance at a short budget. It does not establish diffusion superiority or a Pareto frontier.

The actor NFE per sample is matched: spherical Heun uses half as many integration intervals as Gaussian diffusion. Joint potential fitting and SMC potential evaluations add cost and are recorded separately. Terminal weighting and SMC each use the same configured number of base particles and terminal action/value queries. The direct policy uses one sample and is a cheaper reference.

Checkpoint learner counters cover training, guide fitting, behavior, validation and heldout, and exclude warmup decoding and independent behavioral baselines. They are forward-row counts, not complete experiment FLOPs. Per-evaluation compute totals are additive counts; ESS and log-normalizer diagnostics are reported separately and are not costs. The evaluation actor_nfe field aliases sampling nfe and must not be added to it.

The base-policy critic uses the same relabeled preference at both ends of the Bellman target. Guidance is an inference-time base-policy-value improvement; it is not an exact critic evaluation of the guided policy. Finite suffix log-mean-exp labels and finite-particle SMC are approximations. No target reward, checkpoint or configuration was selected using heldout preferences.

| Model | Updates | Last-window direct validation | Direct validation AUC | Wall seconds |
|---|---:|---:|---:|---:|
| sphere_seed2 | 4000 | 361.147 | 293.938 | 78.6 |
| diffusion_seed2 | 4000 | 295.422 | 242.122 | 98.9 |

Training uses a fixed transition/update budget. Any unfinished training episode is retained in the final checkpoint; every evaluation trajectory completes the full horizon.

| Model | Complete training episodes (including warmup) | Partial final episode | Last-window evaluation points |
|---|---:|---:|---:|
| sphere_seed2 | 212 | 16/20 steps | 2 |
| diffusion_seed2 | 212 | 16/20 steps | 2 |

## Unseen preference interpolation: final checkpoints

Raw component returns use original reward units (N times per-capita cohort utility). At equal weights the scalarized return equals the original return. These rows use common final episode seeds and are descriptive; no cross-seed uncertainty is inferred.

| Model | Mode | Low-response weight | Low-group return | High-group return | Scalarized return |
|---|---|---:|---:|---:|---:|
| sphere_seed2 | direct | 0.125 | 119.500 | 600.612 | 540.473 |
| sphere_seed2 | terminal_weighting | 0.125 | 119.613 | 613.450 | 551.720 |
| sphere_seed2 | direct | 0.375 | 119.450 | 593.150 | 415.512 |
| sphere_seed2 | terminal_weighting | 0.375 | 119.563 | 610.650 | 426.492 |
| sphere_seed2 | direct | 0.625 | 122.325 | 567.750 | 289.359 |
| sphere_seed2 | terminal_weighting | 0.625 | 119.375 | 575.950 | 290.591 |
| sphere_seed2 | direct | 0.875 | 123.388 | 444.075 | 163.473 |
| sphere_seed2 | terminal_weighting | 0.875 | 119.113 | 472.900 | 163.336 |
| diffusion_seed2 | direct | 0.125 | 119.750 | 491.837 | 445.327 |
| diffusion_seed2 | terminal_weighting | 0.125 | 122.538 | 545.837 | 492.925 |
| diffusion_seed2 | smc | 0.125 | 122.538 | 545.837 | 492.925 |
| diffusion_seed2 | direct | 0.375 | 119.788 | 460.850 | 332.952 |
| diffusion_seed2 | terminal_weighting | 0.375 | 122.400 | 491.612 | 353.158 |
| diffusion_seed2 | smc | 0.375 | 122.400 | 491.612 | 353.158 |
| diffusion_seed2 | direct | 0.625 | 121.050 | 421.712 | 233.798 |
| diffusion_seed2 | terminal_weighting | 0.625 | 123.875 | 409.725 | 231.069 |
| diffusion_seed2 | smc | 0.625 | 123.875 | 409.725 | 231.069 |
| diffusion_seed2 | direct | 0.875 | 121.375 | 362.875 | 151.563 |
| diffusion_seed2 | terminal_weighting | 0.875 | 126.738 | 346.338 | 154.188 |
| diffusion_seed2 | smc | 0.875 | 126.738 | 346.338 | 154.188 |

Prefer measured response curves and compute totals over a claim from one favorable weight. Failure to change cohort allocation is a controllability failure even if average utility is high. No monotonicity or guidance-efficiency claim is inferred merely from low regression loss.

[Complete machine-readable results and compute totals](study.json) · [Curves](preference_curves.png)
