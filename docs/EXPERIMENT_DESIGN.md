# Method and initial experiment design

Recorded before the remote screen on 2026-11-04. Experiment ID: `intervention_screen_20261104`.

## Research question and mathematical contract

Can a policy learn more efficiently when the diffusion endpoint represents an entire solver decision region, and is coupled to the current source noise, instead of retaining a particular historical continuous objective?

Let `D(c)` select up to B strictly positive coordinates of c, with deterministic tie-breaking. For binary action a, define a closed robust cell `C_delta(a)`:

- If `|a| < B`, selected coordinates are at least delta and unselected coordinates are at most -delta.
- If `|a| = B`, selected coordinates are at least delta and every selected-minus-unselected gap is at least 2 delta.

Every point in this cell decodes to a. The cell endpoint is `Y = Proj_C_delta(a)(Z)` for fresh Gaussian Z. For full-budget actions, projection reduces to a scalar convex threshold problem; under-budget cells use coordinate clipping. The implementation is checked against independent constrained quadratic optimization, rather than against another copy of the same formula.

For a fixed target action distribution p(a|s), exact projected endpoints satisfy `D(Y)=a`, so their decoded law is p(a|s). Among all endpoints in the same robust cell, projection minimizes squared distance from this particular Z. These statements do not establish globally optimal transport, finite-network endpoint accuracy, RL convergence, or improved return.

Targets a are sampled from target-policy candidates with normalized exponential target-Q weights. This is a finite-candidate approximation, not an exact Boltzmann action sampler. The primary comparisons use identical target-action machinery and networks:

- Cell: `Y=Proj_C_delta(a)(Z)`.
- Point control: `Y=Proj_C_delta(a)(c_old)`; the margin-repaired old-point endpoint is independent of the freshly drawn Z conditional on the selected old candidate.

The independent-cell control projects another independent Gaussian onto the same selected cell. For a fixed target-action distribution it has the same endpoint marginal as cell, but removes the paired nearest-point coupling. During online RL the action distributions can themselves diverge. Cell versus point also changes the conditional endpoint law and its drift as the target actor changes; that contrast alone does not isolate coupling.

A stochastic interpolant trains velocity and noise regressors on `X_t=(1-t)Z+tY+sigma*sin(pi*t)*epsilon`. The velocity label is `Y-Z+sigma*pi*cos(pi*t)*epsilon`; the noise label is epsilon. For interior t, the noise regressor yields the score `-E[epsilon|X_t,s]/(sigma*sin(pi*t))`. The sampler uses diffusion coefficient `g(t)=diffusion_scale*sigma*sin(pi*t)` and drift `v + 0.5*g(t)^2*score`. It uses finite-step Euler-Maruyama, not an exact sampler. Source-dependent Y is permitted by the conditional regression identities. The finite final Euler step still has nonzero noise, so robust targets do not guarantee exact generated endpoint action probabilities. Endpoint distributions can be singular with zero endpoint jitter; no smooth endpoint-density guarantee is asserted.

The `cell_flow` ablation sets g=0 while retaining the same network and training objective. It tests the effect of stochastic sampling. It does not remove the bridge noise from training, and is not the original spherical flow.

## Task and common settings

Released Dynamic Intervention: population 40, primitive actions 20, budget 10, horizon 20, four engagement states, instance seed 0. Upstream commit `7e50d03e7129974a11061c9ce500bc2dc5593246`. See the separate source audit for missing paper settings and the decoder adaptation.

Matched arms: 1,000 warm-up transitions, 4,000 subsequent one-update-per-transition steps; seed 0; batch 64; two hidden layers of width 128; Adam 4e-4; gamma .99; target rate .005; actor every 2 updates; K=8 target/action candidates; 4 bootstrap candidates; 12 sampling steps; temperature 1; bridge sigma .5; rewards scaled by 1/40 for critic learning; observations divided by 3; epsilon .1 random-objective behavior. Actor initialization stream and critic initialization stream are separate so all matched arms start with identical critics. Return is reported on the original reward scale.

The actual behavior and bootstrap both select the highest-Q candidate. Training targets use exponential resampling from target-policy candidates. The number of candidates is matched within the main ablations, but spherical Heun takes two network evaluations per integration step and diffusion Euler takes one. Report wall time and network evaluations; this screen is not an equal-FLOP comparison.

## Fixed eight-arm screen

| Arm | Change from cell_m010 | Purpose |
|---|---|---|
| cell_m010 | Cell endpoint, margin .10 | Anchor |
| point | Old-point endpoint, same margin .10 | Overall decision-cell endpoint design |
| cell_independent | Project an independent Gaussian, same margin .10 | Same conditional cell endpoint law, independent source/endpoint pairing |
| sphere_matched | Spherical geodesic flow policy | Matched decoder/action-critic reference |
| cell_flow | Diffusion scale 0 | Stochastic versus deterministic sampling |
| cell_m005 | Margin .05 only | One-factor margin tuning |
| cell_m020 | Margin .20 only | One-factor margin tuning |
| lsflow_native | Pinned original algorithm and W-transpose decoder | Contextual native reference; not a matched mechanism ablation |

Native LSFlow retains its released 256-wide actor/critic, reward/observation scales, alpha machinery, spherical smoothing, pretraining and update equations. Its configuration records all overrides. The short screen uses K=12, bootstrap smoothing M=16 and 12 Heun steps rather than the released 36. `algorithm.json` describes actual native settings; generic config fields that the native implementation does not use must not be interpreted as effective settings.

## Evaluation and decisions

Validation: common rollout seeds 100000–100019, every 500 training updates and at step zero. Final held-out evaluation: seeds 200000–200049, used once at the final checkpoint. Training seed and benchmark-instance seed are separate. Random feasible and model-informed one-step greedy heuristics are evaluated on the same validation seeds. The latter has privileged dynamics and is not a matched learned-policy baseline.

Keep all arms, including failed/degraded ones. Rank complete margin candidates by mean of the last two validation points (3500 and 4000), then normalized validation AUC. Inspect the entire trajectory, finite losses/gradients, candidate diversity, target weight ESS and projection cost. Do not choose by a single peak or by held-out evaluation. With only one training seed, episode variation does not estimate training stability.

Advance the best margin to a longer confirmatory comparison only if it is feasible and numerically valid and shows a useful signal. A provisional positive mechanism signal requires at least 5% improvement over point in last-window return without more than 5% loss in validation AUC. This is a triage rule, not a statistical claim. A specific coupling interpretation additionally requires improvement over independent-cell and subsequent paired-seed replication. If absent, first diagnose target weights, candidate collapse and critic progress; retain a negative outcome. A margin winner other than .10 does not isolate a causal coupling effect against the .10 point control; a matched-margin point arm is required before that claim.

A full-budget comparison would use 40,000 training updates and paired method/configuration replication on additional training seeds, followed by distinct instance seeds. These are follow-up stages, not evidence already produced by this screen.

## Resources and completion evidence

`wflow-server` GPU was occupied by an unrelated openpi worker at preflight. This bounded campaign uses two CPU workers, one thread each, pinned to logical CPUs 24 and 25, and hides CUDA from those workers. It does not stop or reconfigure existing work.

The source and each config are hashed before dispatch. Every worker records process/start identity, actual package versions and CPU affinity. Checkpoint recovery restores the environment, replay and RNG states; config/source/instance changes refuse resume. Controller lock and per-worker supervisors prevent duplicate workers after reconnection.

Complete means exact environment/update counters, finite training and evaluation journals, held-out result, final loadable checkpoint, matching source/config and exit zero. Raw logs, curves and unsuccessful runs remain available. All current local tiny tests are engineering checks, not policy-quality evidence.
