# SolverCell-RL

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
