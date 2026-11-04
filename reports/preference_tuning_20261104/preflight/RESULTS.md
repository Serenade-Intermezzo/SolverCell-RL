# Preference Intervention: completed implementation pilot

All reported runs passed config/source/exit, full-curve, checkpoint, reward-vector and trajectory audits.

**Scope:** new two-cohort preference task on the unchanged LSFlow Intervention dynamics and native quotient decoder. This is one training seed and one instance at a short budget. It does not establish diffusion superiority or a Pareto frontier.

The actor NFE per sample is matched: spherical Heun uses half as many integration intervals as Gaussian diffusion. Joint potential fitting and SMC potential evaluations add cost and are recorded separately. Terminal weighting and SMC each use the same configured number of base particles and terminal action/value queries. The direct policy uses one sample and is a cheaper reference.

Checkpoint learner counters cover training, guide fitting, behavior, validation and heldout, and exclude warmup decoding and independent behavioral baselines. They are forward-row counts, not complete experiment FLOPs. Per-evaluation compute totals are additive counts; ESS and log-normalizer diagnostics are reported separately and are not costs. The evaluation actor_nfe field aliases sampling nfe and must not be added to it.

The base-policy critic uses the same relabeled preference at both ends of the Bellman target. Guidance is an inference-time base-policy-value improvement; it is not an exact critic evaluation of the guided policy. Finite suffix log-mean-exp labels and finite-particle SMC are approximations. No target reward, checkpoint or configuration was selected using heldout preferences.

| Model | Updates | Last-window direct validation | Direct validation AUC | Wall seconds |
|---|---:|---:|---:|---:|
| sphere_t050 | 4 | 125.817 | 125.817 | 1.1 |
| sphere_t010 | 4 | 125.817 | 125.817 | 1.1 |
| diff_b020_t050 | 4 | 137.150 | 137.150 | 1.6 |
| diff_b060_t050 | 4 | 116.550 | 116.550 | 1.5 |
| diff_b020_t010 | 4 | 137.150 | 137.150 | 1.5 |
| diff_b060_t010 | 4 | 116.550 | 116.550 | 1.5 |
| group_b060_t010 | 4 | 116.550 | 116.550 | 1.5 |
| center_b060_t010 | 4 | 116.550 | 116.550 | 1.5 |

Training uses a fixed transition/update budget. Any unfinished training episode is retained in the final checkpoint; every evaluation trajectory completes the full horizon.

| Model | Complete training episodes (including warmup) | Partial final episode | Last-window evaluation points |
|---|---:|---:|---:|
| sphere_t050 | 0 | 12/20 steps | 1 |
| sphere_t010 | 0 | 12/20 steps | 1 |
| diff_b020_t050 | 0 | 12/20 steps | 1 |
| diff_b060_t050 | 0 | 12/20 steps | 1 |
| diff_b020_t010 | 0 | 12/20 steps | 1 |
| diff_b060_t010 | 0 | 12/20 steps | 1 |
| group_b060_t010 | 0 | 12/20 steps | 1 |
| center_b060_t010 | 0 | 12/20 steps | 1 |

## Unseen preference interpolation: final checkpoints

Raw component returns use original reward units (N times per-capita cohort utility). At equal weights the scalarized return equals the original return. These rows use common final episode seeds and are descriptive; no cross-seed uncertainty is inferred.

| Model | Mode | Low-response weight | Low-group return | High-group return | Scalarized return |
|---|---|---:|---:|---:|---:|
| sphere_t050 | direct | 0.250 | 123.050 | 176.000 | 162.763 |
| sphere_t050 | terminal_weighting | 0.250 | 118.350 | 156.800 | 147.188 |
| sphere_t050 | direct | 0.750 | 123.050 | 176.000 | 136.288 |
| sphere_t050 | terminal_weighting | 0.750 | 118.200 | 157.150 | 127.938 |
| sphere_t010 | direct | 0.250 | 123.050 | 176.000 | 162.763 |
| sphere_t010 | terminal_weighting | 0.250 | 118.350 | 156.800 | 147.188 |
| sphere_t010 | direct | 0.750 | 123.050 | 176.000 | 136.288 |
| sphere_t010 | terminal_weighting | 0.750 | 118.200 | 157.150 | 127.938 |
| diff_b020_t050 | direct | 0.250 | 123.200 | 183.450 | 168.388 |
| diff_b020_t050 | terminal_weighting | 0.250 | 121.300 | 148.150 | 141.438 |
| diff_b020_t050 | smc | 0.250 | 121.300 | 148.150 | 141.438 |
| diff_b020_t050 | direct | 0.750 | 123.200 | 183.450 | 138.262 |
| diff_b020_t050 | terminal_weighting | 0.750 | 121.300 | 148.150 | 128.013 |
| diff_b020_t050 | smc | 0.750 | 121.300 | 148.150 | 128.013 |
| diff_b060_t050 | direct | 0.250 | 123.300 | 155.650 | 147.563 |
| diff_b060_t050 | terminal_weighting | 0.250 | 121.400 | 156.300 | 147.575 |
| diff_b060_t050 | smc | 0.250 | 121.400 | 156.300 | 147.575 |
| diff_b060_t050 | direct | 0.750 | 123.300 | 155.650 | 131.388 |
| diff_b060_t050 | terminal_weighting | 0.750 | 121.400 | 155.000 | 129.800 |
| diff_b060_t050 | smc | 0.750 | 121.400 | 155.000 | 129.800 |
| diff_b020_t010 | direct | 0.250 | 123.200 | 183.450 | 168.388 |
| diff_b020_t010 | terminal_weighting | 0.250 | 121.300 | 148.150 | 141.438 |
| diff_b020_t010 | smc | 0.250 | 121.300 | 148.150 | 141.438 |
| diff_b020_t010 | direct | 0.750 | 123.200 | 183.450 | 138.262 |
| diff_b020_t010 | terminal_weighting | 0.750 | 121.300 | 148.150 | 128.013 |
| diff_b020_t010 | smc | 0.750 | 121.300 | 148.150 | 128.013 |
| diff_b060_t010 | direct | 0.250 | 123.300 | 155.650 | 147.563 |
| diff_b060_t010 | terminal_weighting | 0.250 | 121.400 | 156.300 | 147.575 |
| diff_b060_t010 | smc | 0.250 | 121.400 | 156.300 | 147.575 |
| diff_b060_t010 | direct | 0.750 | 123.300 | 155.650 | 131.388 |
| diff_b060_t010 | terminal_weighting | 0.750 | 121.400 | 155.000 | 129.800 |
| diff_b060_t010 | smc | 0.750 | 121.400 | 155.000 | 129.800 |
| group_b060_t010 | direct | 0.250 | 123.300 | 155.650 | 147.563 |
| group_b060_t010 | terminal_weighting | 0.250 | 121.400 | 156.300 | 147.575 |
| group_b060_t010 | smc | 0.250 | 121.400 | 156.300 | 147.575 |
| group_b060_t010 | direct | 0.750 | 123.300 | 155.650 | 131.388 |
| group_b060_t010 | terminal_weighting | 0.750 | 121.400 | 155.000 | 129.800 |
| group_b060_t010 | smc | 0.750 | 121.400 | 155.000 | 129.800 |
| center_b060_t010 | direct | 0.250 | 123.300 | 155.650 | 147.563 |
| center_b060_t010 | terminal_weighting | 0.250 | 121.400 | 156.300 | 147.575 |
| center_b060_t010 | smc | 0.250 | 121.400 | 156.300 | 147.575 |
| center_b060_t010 | direct | 0.750 | 123.300 | 155.650 | 131.388 |
| center_b060_t010 | terminal_weighting | 0.750 | 121.400 | 155.000 | 129.800 |
| center_b060_t010 | smc | 0.750 | 121.400 | 155.000 | 129.800 |

Prefer measured response curves and compute totals over a claim from one favorable weight. Failure to change cohort allocation is a controllability failure even if average utility is high. No monotonicity or guidance-efficiency claim is inferred merely from low regression loss.

[Complete machine-readable results and compute totals](study.json) · [Curves](preference_curves.png)
