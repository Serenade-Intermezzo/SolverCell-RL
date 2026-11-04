# Controlled temperature follow-up, 2026-11-04

**Status: prepared; not launched.** This design adds six configuration-only arms. It does not change any scientific source under `solvercell/`, `tools/`, `vendor/`, or the root entrypoints/config generator. The required source SHA256 remains `258f8ec3afa41bab34e4456f19b114bb09c64d0a56ef22a914b24ad20258c914`.

## Evidence motivating the intervention

The completed first-screen cell and point anchors use temperature 1.0, margin 0.1, eight candidate actions and reward scale 0.025. Their validation last-window returns are 138.609 and 140.674 respectively. Their curves improve early, then decline; these results do not show a cell-policy advantage.

Final-checkpoint offline diagnostics were computed from 512 sampled replay states using replay seed 882013 and sampling seed 882014. The saved diagnostic script SHA256 is `fec607c64339a18037bef91962c7a9cb1481a005ff0dd5062c354cd5a01e4e81`. Evidence files:

- `reports/intervention_screen_20261104/diagnostics/cell_m010.json`
- `reports/intervention_screen_20261104/diagnostics/point.json`

| Frozen final-checkpoint diagnostic | Cell | Point |
|---|---:|---:|
| Mean candidate-Q standard deviation | 0.261635 | 0.264172 |
| Weight ESS at T=1, out of 8 | 7.566070 | 7.556188 |
| Mean maximum weight at T=1 | 0.170553 | 0.171665 |
| Weight ESS after recomputing weights at T=0.25 | 4.989851 | 4.952044 |
| Weight ESS after recomputing weights at T=0.10 | 2.682958 | 2.644812 |

These diagnostics suggest that temperature 1.0 produces weak value discrimination relative to the observed critic scale. They motivate testing a stronger exponential tilt. They do **not** establish that temperature caused the poor return, that the critic ranking is accurate, or that either lower temperature will improve training. Reducing temperature can also amplify critic errors and hurt exploration. The T=0.1 and T=0.25 entries are recalculations on the same frozen samples, not results of policies trained at those temperatures.

The current actor logger misses actor-update steps: logging every 100 training steps falls on non-actor updates when actor updates occur every two steps starting at index zero. Consequently, the saved first-screen training journal does not contain an actor-ESS/loss/transport trajectory. The final-checkpoint offline diagnostics are valid endpoint measurements, **not training curves**. This follow-up keeps that limitation because the scientific source is frozen; use independently labeled offline diagnostics after completion rather than inventing historical actor statistics.

## Predeclared factorial and controls

| Arm | Method | Temperature | Margin |
|---|---|---:|---:|
| cell_t010 | cell_diffusion | 0.10 | 0.10 |
| point_t010 | point_diffusion | 0.10 | 0.10 |
| independent_t010 | cell_independent_diffusion | 0.10 | 0.10 |
| cell_t025 | cell_diffusion | 0.25 | 0.10 |
| point_t025 | point_diffusion | 0.25 | 0.10 |
| independent_t025 | cell_independent_diffusion | 0.25 | 0.10 |

Every new configuration differs from its same-method first-screen anchor in **temperature only**. Full parent config hashes and exact changed fields are recorded in `experiments/temperature_screen_20261104/design.json`. The anchor configs are `cell_m010.json`, `point.json`, and `cell_independent.json` in the first-screen directory; they all use T=1.0. Completed anchors may be included as reference rows with their original artifacts. They are not rerun or overwritten.

All other settings stay fixed: population 40, primitive actions 20, budget 10, horizon 20, instance seed 0, training seed 0, 1000 warmup interactions plus 4000 training interactions/critic updates, batch 64, actor interval 2, hidden width 128, learning rate 0.0004, reward scale 0.025, eight improvement candidates, four bootstrap candidates, 12 diffusion sampling steps, bridge sigma 0.5, diffusion scale 1.0, exploration epsilon 0.1, and zero endpoint jitter. The exact JSON files, rather than this abbreviated prose, define each run.

Method comparisons at a fixed temperature remain:

- **Cell versus independent cell:** the cell endpoint is a projection of the actual initial noise in one arm and an independent Gaussian noise in the other. Conditional on the chosen action, the endpoint-generation law is the same; the noise-to-endpoint pairing changes.
- **Cell versus point:** both use the same positive-margin decision cell. Cell projects the new initial noise; point projects the previously sampled cost. This changes the endpoint distribution as well as its coupling, so this pair alone cannot isolate coupling.

The independent-cell arm is included at both temperatures to preserve the mechanism test when policy-improvement strength changes. No arm receives a method-specific training budget or a method-specific temperature search range.

## Selection and completion criteria

The primary score is the mean of the last two scheduled validation returns (training steps 3500 and 4000). The secondary score is the trapezoidal validation-return AUC divided by the 4000-step training budget. Show all nine scheduled validation points, including step zero, and retain failures and degraded arms. A single early peak does not select an arm.

Compare methods **within each temperature** before interpreting differences. If a single common lower temperature is required for a next stage, select between 0.10 and 0.25 using the mean last-window validation score across all three completed methods; break an exact tie with their mean validation AUC. This shared rule avoids choosing a temperature because it maximizes a favorable cell-versus-control gap. If any of the three constituent arms fails or is incomplete, that temperature does not have a complete factorial score and no shared-temperature selection is made from the incomplete set. Report individual method rankings as additional observations.

Temperature selection uses only the validation seeds 100000–100019. The existing frozen runner also evaluates final models on seeds 200000–200049. Preserve those heldout results for descriptive reporting; **do not inspect or use them to choose temperature, methods, checkpoints, or the next sweep**. Because the first-screen heldout results have already been reported, this run is development-stage evidence, not a pristine final benchmark test.

A completed arm must satisfy source/config/instance identity, worker exit zero, exactly 5000 total environment interactions, 4000 critic updates, 2000 actor updates, all 250 complete length-20 episodes, all nine finite validation evaluations and one final 50-episode evaluation, finite checkpoint parameters/state, and a matching saved checkpoint. An incomplete or failed arm remains visible and is not silently replaced.

This is a **single training seed, single instance, short-budget tuning experiment**. It tests whether the lower-temperature intervention changes learning and whether coupling comparisons persist. It cannot establish training-seed stability, generalization across task instances, full-budget convergence, or superiority over LSFlow.

## Prepared execution artifacts

Manifest: `experiments/temperature_screen_20261104/manifest.json`. Each job includes the canonical resolved-config SHA256 and a distinct output directory under `outputs/temperature_screen_20261104/`. The manifest requires the unchanged source hash, CPU execution, at most two workers, and CPU affinities [24] and [25]. The six arms total 24,000 training updates plus 6000 warmup interactions. This preparation does not start a controller or training worker.

The intended controller command, to be executed only when this prepared campaign is dispatched, is:

```bash
/home/li/SolverCell-RL/.venv/bin/python tools/campaign.py run \
  --manifest experiments/temperature_screen_20261104/manifest.json
```

Run it from a frozen deployment root containing these prepared configs and the required scientific source. Existing first-screen workers and unrelated workloads must remain untouched. The first-screen and follow-up outputs remain separate; their controller lifetimes must not create overlapping use of the same two reserved CPU slots.

Resource dispatch: the original native reference is still on CPU 25. This follow-up uses CPUs 24 and 26, at most two one-thread workers, for a temporary host total of three training threads. CUDA remains hidden. All matched first-stage arms are complete before this follow-up starts.
