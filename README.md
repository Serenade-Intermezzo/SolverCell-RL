# SolverCell-RL

Repository: [Serenade-Intermezzo/SolverCell-RL](https://github.com/Serenade-Intermezzo/SolverCell-RL). Source, frozen experiment configurations, reports, figures and full commit history are published on`main`. Raw training checkpoints are retained locally/on the experiment server; their hashes and audit evidence are included in the reports.

**Latest priority repair,2026-11-04:** [implemented repairs and completed nine-fit study](reports/priority_repair_20261104/RESULTS.md). Training-side model-informed references establish a real cohort tradeoff; frozen scales calibrate new v4 TD/evaluation while retaining raw rewards. The explicit v4 factory uses a clean terminal kernel and rejects incompatible old guide states.104 tests pass. Common fixed-data mean heldout TV is sphere.2525, epsilon-clean.2770, x0-clean.2913; both diffusion variants fail the preset.15 gate, so no new staged guidance is launched. Actor fitting and learned online preference quality remain unresolved. [Interpretation](reports/priority_repair_20261104/INTERPRETATION.md) · [Figure](reports/priority_repair_20261104/repair_results.png). All9 server workers completed; no jobs remain live/queued.

**Latest diagnosis,2026-11-04:** [methodology audit](reports/method_diagnosis_20261104/DIAGNOSIS.md) identifies weak learned preference tradeoffs and a harmful last-noise choice. In864 fresh complete episodes, removing only final diffusion noise improves fixed-model means by17.348/16.710/18.126 at unchanged8 actor rows, while tuned sphere remains stronger. This is a post-hoc mechanism probe; production defaults and trained models are unchanged. Common-state MC probes support useful critic ranking for high-response/mixed goals, but low-response-only action values are nearly flat. [Raw data and audit](reports/method_diagnosis_20261104/audit.json) · [Figure](reports/method_diagnosis_20261104/diagnosis.png).

**Latest mechanism study,2026-11-04:** [shared-prefix diffusion inference is complete](reports/preference_branching_20261104/RESULTS.md). The validation-locked two-root/four-suffix configuration reduces actor rows64→28 while retaining final utility on three fixed v2 models; deltas versus independent8 are+3.986/+8.901/+4.100, with episode-paired SE5.804/5.948/5.031. Median CPU latency changes only1.853ms→1.803ms. Tuned sphere remains stronger. Branching is optional, with no standalone novelty claim. See [interpretation and budget controls](reports/preference_branching_20261104/INTERPRETATION.md), [figure](reports/preference_branching_20261104/branching_tradeoff.png) and [design](docs/PREFERENCE_BRANCHING_V3.md). All7680 formal episodes finish H20; all98 tests pass; zero training updates, live workers or queued jobs.

**Latest controlled study,2026-11-04:** [all12 preference tuning/replication runs are complete](reports/preference_tuning_20261104/RESULTS.md). Temperature.1 improves diffusion's seed0 last-window direct validation203.018→297.661 at matched4000 updates; tuned sphere reaches359.450. Both new training seeds favor sphere. Centered relative guide fitting improves a small prefix-calibration fixture but SMC still equals terminal weighting with extra cost. See [interpretation](reports/preference_tuning_20261104/INTERPRETATION.md), [full curves](reports/preference_tuning_20261104/tuning_and_replication.png), and the [prespecified design](docs/PREFERENCE_TUNING_V2.md). There is no established diffusion superiority.

**Latest research focus:** [decision-consistent preference control](docs/PREFERENCE_CONTROL_REDESIGN.md), now implemented in a separate [preference-learning harness](docs/PREFERENCE_IMPLEMENTATION.md). Conditional diffusion and spherical flow share a vector critic and native quotient decoder. Diffusion additionally supports a joint noise-level continuation guide and finite-particle SMC; both methods support equal-particle terminal weighting. The [two-model server pilot is complete and audited](reports/preference_pilot_20261104/RESULTS.md): diffusion's default SMC never resamples and exactly matches terminal weighting while adding guide cost. This is an implementation pilot, not a diffusion advantage. Assignment remains an audited future extension.

**Historical single-objective status, 2026-11-04:** all32 runs are complete, including the [18-run native-structure follow-up](reports/native_revision_study_20261104/RESULTS.md). Cell coupling did not establish a consistent advantage. Point diffusion's promising seed-0 result against quotient spherical flow did not replicate on new seeds1/2. Those frozen experiments remain separate from the new preference extension. The original implementation and design history follow below.

A research workspace for **source-conditioned decision-cell diffusion policies** in combinatorial reinforcement learning. The first experiment uses the released LSFlow Dynamic Intervention environment. This is an experimental method, with no established novelty, convergence or performance claim.

The actor samples continuous objectives, and an exact positive-top-k decoder returns a feasible binary intervention. For a selected target action `a`, the proposed endpoint is the Euclidean projection of the **current initial Gaussian noise** onto the robust decision cell of `a`. The principal control projects the **old sampled objective** onto the same cell. Both share the critic, target-action distribution, margin, stochastic-interpolant loss and stochastic sampler.

- [Method and prespecified experiment design](docs/EXPERIMENT_DESIGN.md)
- [Pinned upstream audit and adaptation boundaries](docs/UPSTREAM_AUDIT.md)
- [Current progress](plan.md)

## Setup

Python 3.10+, NumPy and PyTorch are required. SciPy is used only for independent QP tests; Matplotlib renders report curves. Do not install into an unrelated research environment. The local `.venv` and server `.venv` inherit their existing respective PyTorch environments; each run records its actual versions.

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
mkdir -p vendor
git clone https://github.com/anagha-satish/combinatorial-diffusion.git vendor/lsflow
git -C vendor/lsflow checkout 7e50d03e7129974a11061c9ce500bc2dc5593246
.venv/bin/python -m unittest discover -s tests -v
```

The upstream checkout is preserved unchanged and excluded from this repository. Its Python source hashes are included in every experiment's source manifest. Original code is MIT licensed; see [upstream license](https://github.com/anagha-satish/combinatorial-diffusion/blob/main/LICENSE).

## Run

The preference harness has98 passing tests and a source-frozen eight-arm template. Its separate two-lane controller immediately hands ready independent work to a freed CPU lane. Each arm records exact config, source, worker/CPU identity, exit, full curves and complete checkpoints. To run a fresh declared screen:

```bash
.venv/bin/python -m tools.prepare_preference_tuning --campaign-id my_preference_tuning
.venv/bin/python -m tools.preference_campaign --manifest experiments/my_preference_tuning/manifest.json
.venv/bin/python analysis/select_preference_tuning.py --manifest experiments/my_preference_tuning/manifest.json --output reports/my_preference_selection.json
.venv/bin/python -m tools.prepare_preference_tuning --campaign-id my_preference_replication --selection reports/my_preference_selection.json
.venv/bin/python -m tools.preference_campaign --manifest experiments/my_preference_replication/manifest.json
```

The template uses CPUs24/25 on wflow-server,4000 updates and at most two workers. Adjust available CPU sets before using another host. The selected configuration is locked using validation only; fresh final episode seeds do not select settings. Grouped-absolute and centered guide controls leave base learning bitwise identical. Historical default learner checkpoints remain loadable for inference with their original schedule; scientific training continuation still requires identical source/config. Rebuild the shipped completed study with`python analysis/build_preference_tuning_report.py` when its raw artifacts are present under`outputs/`.

For the new preference extension, use the dedicated entrypoint and controller:

```bash
.venv/bin/python -m tools.prepare_preference --campaign-id my_preference_pilot
.venv/bin/python -m tools.run_preference --config experiments/my_preference_pilot/diffusion.json --output outputs/my_preference_debug --debug
.venv/bin/python -m tools.preference_campaign --manifest experiments/my_preference_pilot/manifest.json
.venv/bin/python -m tools.report_preference --manifest experiments/my_preference_pilot/manifest.json --output reports/my_preference_pilot
```

The template declares two CPU workers on CPUs24/25, seed0, one instance,256 warmup and1000 updates. It freezes source/config identities and refuses existing outputs. The same checkpoint is tested with direct conditioning, terminal weighting and (diffusion only) SMC. Training uses five preferences; final interpolation uses four unseen preferences. See the [implementation contract and approximation boundaries](docs/PREFERENCE_IMPLEMENTATION.md). This is a new multi-objective harness, not an upstream DPMD reproduction.

For the historical single-objective cell study:

Generate a new eight-arm screen from the template. Existing experiment directories are never overwritten:

```bash
.venv/bin/python configs.py --campaign-id my_screen
.venv/bin/python run.py --config experiments/my_screen/cell_m010.json --output outputs/my_debug --debug
.venv/bin/python tools/campaign.py run --manifest experiments/my_screen/manifest.json
.venv/bin/python aggregate.py --manifest experiments/my_screen/manifest.json --output reports/screen
```

The shipped screen uses two single-thread CPU workers pinned to CPUs 24 and 25 on `wflow-server`. Change the CPU affinity before running on another machine. It uses 1,000 warm-up steps plus 4,000 training updates per arm, seed 0, one fixed instance, validation every 500 updates, and 50 held-out rollout episodes. It is a short tuning screen; the released full training budget is 40,000 updates.

Each run writes its resolved configuration, source hashes, instance, effective algorithm settings, process identity, training/evaluation/episode journals and full checkpoint. The controller adds the launch command and actual process exit. A completed run needs the exact configured counters, finite metrics and exit zero. Failed and partial runs remain visible and are never silently retried. The controller is protected by a file lock; independent supervisors preserve exit evidence after controller disconnection.

A deliberately interrupted individual run can be continued with the exact same source and config using `run.py --resume`. Do not resume a completed run or reuse an output folder. The campaign controller does not automatically resume partial workers. Evaluation randomness is isolated from training, and checkpoints restore replay, optimizers, targets, environment and all RNG states.

## Layout

| Path | Role |
|---|---|
| `solvercell/environment.py` | Source-checked intervention environment |
| `solvercell/geometry.py` | Exact decoder and nearest robust-cell projection |
| `solvercell/policies.py` | Stochastic interpolant diffusion and spherical flow |
| `solvercell/learner.py` | Matched action-critic policy improvement |
| `solvercell/upstream.py` | Adapter invoking pinned original LSFlow algorithm |
| `solvercell/trainer.py` | Training, paired evaluation, checkpoint and status |
| `configs.py`, `experiments/` | Declared finite experiments |
| `tools/campaign.py`, `aggregate.py` | Durable queue and artifact-audited reports |
| `tests/` | Numerical, environment, random-stream and controller checks |

`lsflow_native` preserves the original arm-cost decoder and learning update, with explicitly shortened sampling/training settings. `sphere_matched` is a controlled spherical-policy adaptation with the same primitive decoder and action critic as diffusion; it is not an official LSFlow reproduction. Compare cell versus point for the overall endpoint design; compare cell versus independent-cell to isolate source/endpoint pairing at a fixed conditional endpoint law. The native reference provides context across the larger adaptation boundary.

## Shared-prefix diffusion preference pools

The additive [branching sampler](solvercell/preference_branching.py) shares actual stochastic prefixes and generates fresh conditional suffixes before global terminal preference weighting. Unequal fixed branch counts use inverse-count root corrections. It changes no trained actor/critic or Bellman target. See the [prespecified mechanism design](docs/PREFERENCE_BRANCHING_V3.md) for correlation, finite-pool bias, consistency and full compute boundaries.

The frozen v3 campaign uses the six existing v2 checkpoints, which must be present under their recorded `outputs/` paths and match the manifest hashes; raw checkpoints are not distributed through GitHub. Reproduce in a fresh campaign directory, with the pinned upstream checkout and unchanged parent source files:

```bash
.venv/bin/python analysis/prepare_preference_branching.py --campaign-id my_branching
.venv/bin/python analysis/control_preference_branching.py --manifest experiments/my_branching/manifest.json --runtime .runtime/my_branching --checkpoint-root . --python /absolute/path/to/python --stage screen
.venv/bin/python analysis/select_preference_branching.py --manifest experiments/my_branching/manifest.json --screen outputs/my_branching/screen --output reports/my_branching/selection.json
.venv/bin/python analysis/control_preference_branching.py --manifest experiments/my_branching/manifest.json --runtime .runtime/my_branching --checkpoint-root . --python /absolute/path/to/python --stage final --selection reports/my_branching/selection.json
.venv/bin/python analysis/report_preference_branching.py --manifest experiments/my_branching/manifest.json --output reports/my_branching
```

The controller is fixed to the declared server CPU lanes24/25 and admits two workers. Selection is locked before final evaluation. `--debug` on preparation produces an explicitly engineering-only preflight; its report does not establish a performance result.

## Frozen experiments versus current development

The two 2026-11-04 screens use scientific source SHA256 `258f8ec3afa41bab34e4456f19b114bb09c64d0a56ef22a914b24ad20258c914`. Their complete source snapshots are preserved on the server in `deployments/<campaign>/source`. Initial source commit: `ec41ef1`; temperature campaign preparation commit: `f6b7d1d` (same scientific code). The local development version subsequently fixes actor metric logging and prevents config generation from overwriting an existing campaign. These changes are not present in those experimental checkpoints.

For exact old-source reproduction, use the recorded commit or preserved server snapshot, its pinned upstream checkout and a fresh output directory. Running a historical manifest against changed source is deliberately refused. Do not regenerate or edit the historical manifests.

## Native-structure quotient revision

The current method keeps native LSFlow critics, Bellman smoothing, weights and update scheduling. The actor can operate in the8-dimensional column span of the original40-dimensional decoder and lift samples back into the native critic. Exact tie-aware cell projection, point and independent endpoint controls, and an8D spherical reference are implemented. See [revision design](docs/NATIVE_REVISION_DESIGN.md).

```bash
.venv/bin/python analysis/prepare_native_revision.py --campaign-id my_native_revision
.venv/bin/python tools/campaign.py run --manifest experiments/my_native_revision/manifest.json
.venv/bin/python aggregate.py --manifest experiments/my_native_revision/manifest.json --output reports/my_native_revision
```

This template uses at most4 CPU workers pinned to24–27 on wflow-server; adjust affinities before using another host. Historical manifests and source snapshots remain immutable.

Future checkpoints serialize only replay rows that can actually be sampled, using CPU tensors instead of NumPy pickle payloads. Legacy replay checkpoints remain loadable. The native six-arm screen launched before this storage-only change; its frozen source is recorded separately.

## Completed native study and current evidence

The six-arm native revision and twelve-arm margin/replication campaign are complete and all 18 final checkpoints passed independent deep audits. Both seed-0 storage rechecks exactly match their old complete trajectories and final scientific state. Shared margin 0.6 was selected by the prespecified validation-only mean across point/cell/independent. It improves cell return but leaves cell below point; the coupling contrast changes sign across margins. Fixed margin-0.1 point-minus-sphere differences are +16.622, −26.211 and −35.580 across seeds 0/1/2, with mean −15.056 and sample SD 27.832. Seed 0 is discovery evidence, while both new seeds favor spherical flow.

- [Complete 18-run report](reports/native_revision_study_20261104/RESULTS.md), [machine-readable study](reports/native_revision_study_20261104/study.json), [learning curves and replication](reports/native_revision_study_20261104/learning_and_replication.png).
- [Point storage equality](reports/native_revision_study_20261104/point_storage_equivalence.json), [sphere storage equality](reports/native_revision_study_20261104/sphere_storage_equivalence.json), [final remote snapshot](reports/native_revision_study_20261104/final_remote_state.json).
- [Historical 14-run study](reports/study_20261104/RESULTS.md), kept separate because its adaptation harness differs.
- [Methodology audit](docs/METHODOLOGY_AUDIT.md), [solver support/prior audit](docs/SOLVER_SUPPORT_AND_PRIOR.md), [feasible-exchange rethink](docs/FEASIBLE_EXCHANGE_RETHINK.md), [diffusion redesign decision](docs/DIFFUSION_REDESIGN_DECISION.md).
- [Continuation critic and conditional-branch check](docs/NOISE_CONTINUATION_CRITIC.md): the first two-query estimator fails its equal-query variance gate in all 12 fixed test cases. This is an oracle diagnostic, not a trained-policy or RL result; no new training sweep follows from it.

Rebuild the completed report with `python analysis/build_native_study_report.py`. The script verifies saved configs, exits, exact counters, complete validation curves and audit outcomes before computing the validation-only selections. Heldout results are a separate descriptive section. This study matches interactions and updates, while spherical flow uses 24 neural evaluations per sample and diffusion uses 12; it does not establish matched-compute or intrinsic speed superiority. All results are from one environment instance and short 4,000-update runs.

The recorded 2026-11-04 09:36:30 UTC remote snapshot has all four campaigns complete and no experiment workers. Runs used CPU; no GPU training was launched. Frozen sources and manifests remain immutable. See [plan.md](plan.md) for source hashes, complete scope and proposed work.

## Explicit repaired preference profile

Historical PreferenceConfig defaults preserve the legacy unit-scale/noisy terminal policy. Use the new factory for a declared calibrated objective and clean terminal diffusion; it requires the frozen training-side calibration and rejects a failed task gate:

```python
from solvercell.preference_repair import repaired_preference_config
from solvercell.storage import atomic_json
cfg, provenance = repaired_preference_config(
    "reports/priority_repair_20261104/task_audit.json",
    method="preference_diffusion", seed=0, train_steps=4000)
atomic_json("experiments/my_repair/config.json", cfg.to_dict())
atomic_json("experiments/my_repair/calibration.json", provenance)
```

Run a new headless job with `python -m tools.run_preference --config experiments/my_repair/config.json --output outputs/my_repair`; use `--debug` for engineering validation. This example prepares a new experiment, not a shipped performance result. New normalized scores cannot be compared directly to old raw-objective scores. Standalone x0 prediction is available as a declared ablation and has not been promoted. The common fixed-endpoint data/configs are distributed under `experiments/priority_fixed_20261104/`; raw learned checkpoints remain local/server artifacts.
