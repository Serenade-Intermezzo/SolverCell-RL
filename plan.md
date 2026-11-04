# SolverCell-RL plan

## Objective
Implement decision-cell diffusion, select a released LSFlow benchmark, and run controlled tuning on wflow-server.

## Current status (2026-11-04): completed
- New independent local research workspace and Git repository created.
- Pinned upstream `7e50d03e7129974a11061c9ce500bc2dc5593246`; selected source-checked Dynamic Intervention N40/A20/B10/H20.
- Exact robust-cell projection, stochastic-interpolant diffusion, matched action critic, independent-cell and point controls, spherical reference, native LSFlow adapter, durable controller and aggregation implemented.
- Initial 8-arm screen and subsequent 6-arm temperature factorial finished on wflow-server: **14/14 complete**, zero failed/partial/live/queued arms.
- All final checkpoints mirrored locally; complete configuration/source/instance/update/episode/finite-state audits passed independently on server and local copies.
- Same scientific source SHA for all experiments: `258f8ec3afa41bab34e4456f19b114bb09c64d0a56ef22a914b24ad20258c914`.
- Current development subsequently fixes actor metric logging and protects prior manifests from overwrite. Frozen remote source is preserved. Local full suite: 34 tests pass (experiment version: 32).
- No GPU job was launched. Existing openpi GPU workload was preserved; two CPU workers per campaign, briefly three threads across overlapping stages.

## Steps
1. [done] Pin and audit source, environment, decoder and experimental recipe.
2. [done] Implement decision-cell coupling and stochastic sampler with causal controls.
3. [done] Validate projection using independent QP, upstream trajectories, gradients, RNG recovery and controller behavior.
4. [done] Freeze, deploy and complete the eight-arm initial screen.
5. [done] Diagnose Q weights and complete a one-factor temperature follow-up across all three methods.
6. [done] Audit all completed artifacts, render curves, save report and preserve negative results.

## Result and next research decision
- Common temperature .1 selected from prespecified validation-only cross-method means.
- At T=.1, last-window validation: cell147.635, point181.346, independent-cell143.648.
- Native LSFlow316.331, matched spherical reference160.845.
- Cell failed the positive mechanism triage gate. Pairing gain against independent-cell changes sign across temperatures. No superiority or stability claim.
- Next scientific work should narrow the adaptation boundary by keeping native critic/update/decoder while testing a cell target. The W-transpose decoder has structural ties, requiring a different feasible-cell construction. This is a proposed next experiment, not implemented or queued.

## Deliverables
- `README.md`: setup, fresh campaign generation and frozen-source reproduction boundary.
- `reports/study_20261104/RESULTS.md`: complete scientific report and 14-arm table.
- `reports/study_20261104/learning_curves.png` and `.pdf`: complete trajectories.
- `reports/study_20261104/study.json`: machine-readable status and metrics.
- `reports/{intervention_screen_20261104,temperature_screen_20261104}/`: CSV/JSON/LaTeX and independent audits.
- `outputs/`: all final run artifacts including checkpoints (Git-ignored).
