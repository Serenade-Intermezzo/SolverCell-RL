# Native-structure quotient diffusion: revision design

Prespecified 2026-11-04 after the completed 14-arm study. Historical source and outputs remain intact. This revision addresses a diagnosed architecture mismatch; no gain is assumed in advance.

## Evidence motivating the revision

The previous cell/point policies lost return after an early peak. Lower temperature helped point (181.346) more than cell (147.635); native LSFlow reached316.331. The old matched harness changed more than the generator: binary-action versus continuous-cost critic, reward scale, observation scale, max versus trimmed-average Bellman value, cross-candidate resampling versus native weighted regression, and actor/target update schedules. That gap cannot establish that diffusion itself is weak.

Source inspection found that native actor weights use one proposal per state, target-Q EMA/batch normalization, clipped exponential weights and learned alpha. Native critics see40-dimensional clean cost centers with execution and Bellman tangent perturbation. All these mechanisms are retained in the revision.

The released intervention W has shape40×20, rank8, with8 unique mutually orthogonal columns and32 decoder-invisible directions. These facts are checked in code. Define an orthonormal basis U for the column span. A quotient actor generates y on S7 and lifts c=Uy before native critic/decoder calls. The spherical quotient actor is a necessary control: a gain from removing the kernel is not automatically a diffusion gain.

## Method

- Original observations, reward scale1, twin256-wide continuous-cost critics, eight myopic pretraining steps, M16/J1 smoothed trimmed Bellman targets, native alpha/lambda/weight clipping, every-step actor updates and every-two-step target EMA.
- Original K12 behavior candidates and12 integration steps. Sphere Heun takes24 network evaluations; diffusion Euler12. This is an equal-step/equal-update comparison, not equal FLOPs.
- Actor initialized only after constructing the original native models/critics, under restored Torch/NumPy RNG. All variants have bitwise-identical initial critics.
- Unit prior and unit endpoints. Euclidean stochastic interpolant, Gaussian interior noise with coordinate standard deviation bridge_sigma/sqrt(actor_dim). The default bridge_sigma=.5 is an RMS vector scale, not per-coordinate .5.
- Velocity coordinate-sum error plus noise coordinate-mean error, multiplied by original raw native weights. A two-head actor with256 hidden units generates normalized final costs. It is not a spherical Brownian diffusion and has no finite-step exact-distribution guarantee.
- The original W-transpose decoder is evaluated through unique group scores and broadcast to enforce stable ties among identical columns. Float32 full GEMM can spuriously break those ties. Tests establish that the selected actions have identical W@a environment effects.

For each target action, project in group scores with metric1/||W_g||², then perform a minimum-norm lift. Full/empty groups use a scalar convex threshold. A partially selected duplicate group sits at the threshold. Multiple distinct partial groups have no strict-margin cell and explicitly fall back to the verified old cost. Fallback counts are logged.

Point, cell and independent-cell share the same unnormalized cell and delta=.1. Point projects old cost, cell projects the actual training source, independent-cell projects another unit source. Endpoints are normalized; **the numerical margin after normalization is not identical across samples/methods**. Normalized score gaps are logged. Nearest-point optimality applies before normalization in Euclidean space, not on the sphere.

Clean decoder equivalence does not imply equal smoothed Q: native execution and Bellman noise depend on the cost center. This is a scientific limitation of relabeling costs inside a clean decision cell. No exact PMD or smoothed-policy invariance claim is made.

## Initial fixed six-arm campaign

| Arm | Purpose | Main comparison |
|---|---|---|
| native_reference | Tensor path with original spherical actor | Reproduce strong reference under current source |
| full_point |40-dimensional point diffusion, same native learning loop | Restore training structure before judging diffusion |
| quotient_sphere |8-dimensional spherical actor lifted to40 | Test quotient representation with original generator |
| quotient_point |8-dimensional point diffusion lifted to40 | Diffusion versus sphere on the same representation |
| quotient_cell |8-dimensional cell endpoint diffusion | Cell endpoint design versus quotient point |
| quotient_independent |Same conditional cell endpoint law, independent source | Coupling versus independent pairing |

All share seed0, instance0, N40/A20/B10/H20,1000 warmup+4000 training, Adam4e-4, batch64, gamma.99,tau.005,K12,M16,12 steps. Validation20 common episodes every500 updates and final heldout50; same disjoint seed ranges as the old study.

## Hypotheses and decision rules

| Hypothesis | Minimal contrast | Evidence sought | Competing explanation | Priority |
|---|---|---|---|---|
| Native training structure restores diffusion learning | full_point versus historical best point | >=20% higher last-window return and improved AUC | Actor architecture/source scale also change; bundled revision claim only | Must-have |
| Removing kernel directions helps | quotient_point versus full_point; quotient_sphere versus native_reference | Consistent learning improvement across generator types | Capacity, exploration geometry and smoothing-center effects | Must-have |
| Diffusion is competitive on the quotient | quotient_point versus quotient_sphere | Last-window within5%, AUC within10% | Integration NFE and two-head parameter count | Must-have |
| Cell representation helps | quotient_cell versus quotient_point | >=5% last-window improvement without >5% AUC loss | Different endpoint marginal and final normalized margins | Must-have |
| Pairing itself helps | quotient_cell versus quotient_independent | Same-sign improvement later repeated across seeds | Online target-action laws diverge during RL | Must-have |

Rank only complete curves, last-window then AUC. Retain all arms. Heldout is never used for hyperparameter selection. Native reference should reproduce the historical curve within float precision/known decoder tie environment equivalence; any major mismatch must be investigated before conclusions.

Conditional follow-up: tune only one diagnosed actor mechanism at a time. If small normalized cell gaps/fragile execution accompany weak cell learning, compare common margins across point/cell. If sampling loss/noise dominates, compare common bridge RMS scale. Use a shared hyperparameter selection rule across compared methods rather than maximizing cell's relative gain. Promising methods require matched additional training seeds before a stability claim.

## Code optimization and validation

Tensor-only native candidates and smoothing remove redundant conversions while retaining original kernels, random calls, diagnostics and update equations. Complete filled optimizer checkpoints and multiple subsequent updates are bitwise equal. A local counterbalanced microbenchmark measured1.008×, which does not establish material CPU speedup. Quotient actor parameter reduction and diffusion NFE are reported separately from wall time.

Independent QP tests validate cell projection. Actual learner tests validate initial critics, quotient kernel removal, actor parameter updates, normalized target actions, checkpoint continuation and evaluation RNG isolation. The entire suite must pass before dispatch.

GPU remains occupied by unrelated openpi training. The bounded initial campaign uses at most4 one-thread CPU workers on physical cores24–27, CUDA hidden. Every worker has durable PID/start identity, config/source hashes, checkpoint and exit record. It does not modify the prior deployments.
