# SolverCell-RL plan

## Objective
Implement decision-cell diffusion, select a released LSFlow benchmark, and run controlled tuning on wflow-server.

## Current status (2026-11-04)
- Pinned upstream: `7e50d03e7129974a11061c9ce500bc2dc5593246`.
- Selected source-checked Dynamic Intervention: N=40, A=20, B=10, H=20, instance seed 0.
- Projection, diffusion, matched action critic, native LSFlow adapter, durable controller and aggregation implemented.
- Local 31-test suite passed; independent QP oracle, upstream trajectory parity, RNG isolation, checkpoint continuation and controller recovery included.
- Server preflight for cell diffusion and native LSFlow completed 20 training updates each with real batch/network/sampler settings. These are engineering checks, not performance evidence.
- GPU remains occupied by unrelated openpi training. Screen uses two single-thread CPU workers on CPUs 24/25, with CUDA hidden.
- Eight-arm 4k-update seed-0 campaign prepared; final source/config freeze and dispatch next.

## Steps
1. [done] Pin and audit LSFlow code, environment, solver and experimental recipe.
2. [done] Implement decision-cell coupling, stochastic sampler and matched controls.
3. [done] Validate projection/decoder, gradients, RNG recovery, provenance and end-to-end execution.
4. [in_progress] Freeze and deploy the eight-arm bounded screen on wflow-server.
5. [todo] Audit complete curves and artifacts; use validation metrics for a justified follow-up.

## Decisions and limits
- No novelty or performance superiority claim has been established.
- Finite-candidate exponential resampling approximates policy improvement.
- Region projection is conditional nearest-point coupling, not global optimal transport.
- Cell versus point changes endpoint law and coupling; independent-cell controls the pairing at a fixed conditional endpoint law.
- Native LSFlow is contextual; sphere_matched shares the adapted decoder/action critic.
- The first screen is one training seed and one instance at one tenth of the released training budget.
