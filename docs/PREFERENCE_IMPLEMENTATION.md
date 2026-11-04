# Preference-conditioned Intervention implementation

This is an explicit multi-objective extension of LSFlow Intervention, implemented in a separate harness. It preserves the original N40/A20/B10/H20 dynamics, instance constructor, native score/tie convention and feasible decoder. It does not reproduce upstream DPMD and does not pool results with the historical single-objective experiments.

The first two-model [server pilot](../reports/preference_pilot_20261104/RESULTS.md) is complete and independently audited. Default SMC never resamples and matches terminal weighting with added guide cost. The [interpretation](../reports/preference_pilot_20261104/INTERPRETATION.md) and [fixed-checkpoint validation probe](../reports/preference_guide_probe_20261104/RESULTS.md) record weak preference response and uneven prefix-guide calibration. These are prototype limitations, not evidence of diffusion superiority.

## Decision and reward contract

- Observation: 40 arm states divided by 3, plus remaining-horizon fraction. The time feature makes finite-horizon value prediction well specified.
- Preference: `w=(lambda,1-lambda)`, fixed for each rollout. The two components concern the synthetic low/high response cohorts, not demographic groups.
- Reward vector: `[40 * mean(low cohort reward), 40 * mean(high cohort reward)]`. At equal weights the scalar reward is exactly the original benchmark reward. Replay keeps this vector in original units; both critics use reward scale 0.025.
- Continuous policy output: the audited rank-8 column-space quotient of the original 40-dimensional cost interface, lifted into the original decoder. No cell projection, new discrete support, or final execution perturbation is introduced.
- Training preferences: 0, .25, .5, .75, 1. Validation uses the training weights 0, .5, 1; final unseen interpolation uses .125, .375, .625, .875 and disjoint episode seeds. Heldout weights do not fit the actor, critics or guide and do not select a checkpoint.

The replay buffer stores observation, original action, sampled quotient cost, reward vector, next observation, terminal flag and behavior preference. Updates relabel the preference independently of the behavior preference. The same relabeled preference appears at both ends of the vector Bellman target. A whole critic vector is selected according to the smaller scalarized value; componentwise minima are not mixed across critics.

## Conditional base policies

`preference_sphere` learns geodesic spherical flow matching; `preference_diffusion` learns a Gaussian DDPM epsilon model. Both condition on observation, preference and generation time, have the same actor width and parameter count, share the twin vector critic and soft candidate-weighted policy improvement, and use the same training-preference grid. Their EMA target actor consistently defines behavior, Bellman bootstrap and the reference suffix law.

The configured `sampling_steps` means actor neural function evaluations (NFE), not integration intervals. With NFE=8, diffusion uses eight Gaussian transitions and sphere uses four Heun intervals. Direct inference draws one sample. Candidate selection for the actor loss uses all soft-weighted candidates with the same normalized loss reduction across methods. This is a common preference-learning harness, not the original native LSFlow learner.

The Gaussian chain keeps raw unnormalized states internally and normalizes only the returned cost. Its final transition has strictly positive noise. Guide targets, suffix restarts, terminal weighting and SMC all use this actual implemented chain. The short linear beta schedule (.02 to .20 over eight steps) has cumulative alpha around .38 at the noisiest level, so its standard-normal starting prior is an approximation to the forward terminal marginal. This prototype makes no exact reverse-process or exact data-likelihood claim. The exact-kernel accounting below concerns the implemented reference policy.

## Joint noise-level preference guidance

For the conditional base chain `p_base(. | s,w)` and learned base-policy utility `U_w(c)=w dot Q_base(s,D(c),w)`, the inference target is

```
p_tilt(c | s,w) proportional to p_base(c | s,w) exp(U_w(c) / temperature).
```

The guide learns one joint continuation potential at the actual noisy prefix:

```
log h_k(s,x,w) = log E_base[exp((U_w(c_final)-b(s,w))/temperature) | x_k=x].
```

The baseline is state/preference only. Training labels use fresh target-actor suffixes from the same prefix and a log-mean-exp reduction. This has finite-suffix log-estimation bias; summing separately trained component log guides would not give the joint target. The guide uses a separate RNG and optimizer, so its fitting does not alter base-policy training streams. The moving EMA reference and approximate critic introduce further approximation.

At each checkpoint, the same model is evaluated in these modes:

| Mode | Policy | Cost |
|---|---|---|
| `direct` | One conditional base sample | One actor trajectory |
| `terminal_weighting` | N independent complete samples; categorical exponential-utility weighting | N trajectories and terminal Q queries |
| `smc` (diffusion) | N particles; joint prefix potential, ESS resampling, exact terminal utility weights | Same actor/terminal particle budget, plus guide calls |

SMC includes initial `h_0` weights, copies both state and potential through ancestors, and uses telescoping potential ratios. Without intermediate resampling it exactly reduces to terminal importance weighting for the same reference draws, independently of guide accuracy. With resampling it is a finite-particle approximation to the specified tilt. There is no gradient/drift proposal, so no proposal-density correction is omitted. An inaccurate potential can still reduce finite-particle performance.

Guidance is inference-time improvement using a base-policy critic. This critic does not evaluate the closed-loop guided policy. The code makes no claim that the base Bellman target is exact for the guided rollout. Equal NFE per trajectory also does not imply equal total compute: diffusion guide training and SMC guide inference add work.

## Files and entrypoints

| File | Role |
|---|---|
| `solvercell/preference_config.py` | Separate validated configuration and disjoint preference/episode splits |
| `solvercell/preference_data.py` | Reward components, native quotient decoder, compact vector replay |
| `solvercell/preference_policy.py` | Conditional FM/DDPM, actual reverse kernels, suffix sampling, joint guide, terminal weighting and SMC |
| `solvercell/preference_learner.py` | Vector critics, relabeled Bellman update, matched soft actor improvement and guide fitting |
| `solvercell/preference_trainer.py` | Headless training, isolated full-horizon evaluations and exact recovery |
| `tools/run_preference.py` | Refuses overwrite; writes invocation-owned durable exit records |
| `tools/prepare_preference.py` | Freezes a bounded two-model campaign without launching it |
| `tools/preference_campaign.py` | At most two CPU workers; immutable source/config checks and durable controller state |
| `tools/report_preference.py` | Full artifact audit and preference-response report |

Run from the repository root. These preference manifests deliberately use a separate controller; do not pass them to the historical controller.

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m tools.prepare_preference --campaign-id my_preference_pilot
.venv/bin/python -m tools.run_preference --config experiments/my_preference_pilot/diffusion.json --output outputs/my_preference_debug --debug
.venv/bin/python -m tools.preference_campaign --manifest experiments/my_preference_pilot/manifest.json
.venv/bin/python -m tools.report_preference --manifest experiments/my_preference_pilot/manifest.json --output reports/my_preference_pilot
```

The prepared campaign is CPU only, single-thread workers on CPUs24/25 of wflow-server, 256 warmup plus1000 updates, seed0 and instance0. Adjust available CPU affinities before using another host. New source requires a new manifest; historical output directories are never reused. An individual interrupted worker can use `--resume` only with identical source/config, and completed runs cannot resume.

## Validation and evidence limits

Tests cover the actual Gaussian transition density, weighted loss, suffix recovery, matched actor parameter count/NFE, initial SMC weighting, ancestor copying, guide gauge invariance, no-resampling equivalence, vector Bellman semantics, decoder equivalence and vector replay. Both methods undergo complete short runs, evaluation-RNG isolation and bitwise recovery from a mid-episode checkpoint with wrapped replay. A rejected duplicate invocation cannot overwrite the original successful exit record.

Evaluation reports additive actor/Q/guide/decoder row counts separately from averaged ESS/log-normalizer diagnostics. Learner cumulative counters include training and learned-policy validation/heldout evaluation; warmup direct decoding and independent behavioral-baseline work are outside that counter scope. Complete evaluation trajectories always finish H20. Fixed-step training may finish with a partial episode whose state and accumulated reward are preserved; the default1256 steps contain62 complete episodes plus16 steps.

The pilot tests implementation and preference response at one instance, one training seed and a short budget. It does not establish novelty, convergence, a Pareto frontier or diffusion superiority. A useful next scientific gate is reproducible preference response and a gain over equal-particle terminal weighting after accounting for guide cost. Assignment is a separately audited future extension and is not implemented here. Routing and Disease remain excluded for their documented interface/data boundaries.
