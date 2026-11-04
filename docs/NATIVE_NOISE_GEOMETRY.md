# Native execution noise and decoder-cell margins

This is a theoretical note about the retained native perturbation. It neither changes the running experiment nor reports a new experimental result. The probability below is exact for the stated Gaussian tangent perturbation, **not** for an exact von Mises–Fisher distribution.

## Source contract

The pinned [`_vmf_perturb_torch`](../vendor/lsflow/models/rfm/service.py) normalizes its center, draws a standard Gaussian, subtracts its component along that center, scales by `(kappa + 1e-6)^(-1/2)`, adds the center, and normalizes again. [`normalize`](../vendor/lsflow/models/rfm/policy.py) divides by `norm + 1e-8`. [`NativeLSFlow.act`](../solvercell/upstream.py) executes the perturbed coefficient while returning the **clean center** for replay. [`TensorNativeLSFlow.install_tensor_paths`](../solvercell/fast_native.py) calls the same perturbation in the smoothed Bellman calculation; it retains the native min-critic and trimmed reduction.

The decoder is positive top-k of `c @ W`. The new native bridge uses [`NativeCellProjector.decode`](../solvercell/native_geometry.py), which computes one score per unique column and broadcasts it, so identical columns have exact, stable index ties. This resolves numerical tie differences without distinguishing the environment effects of identical columns.

## Exact probability for one decision boundary

First use an exact unit center \(c\), \(\kappa>0\), and \(\epsilon\sim\mathcal N(0,I)\):

\[
 y=c+\frac{(I-cc^\top)\epsilon}{\sqrt\kappa},
 \qquad \widetilde c=\frac{y}{\|y\|}.
\]

Let \(n\ne0\) be a decision-boundary normal oriented toward the desired action, with \(a=n^\top c\ge0\). Here “boundary” includes redundant defining inequalities, not only irredundant facets. The normalization factor is positive; indeed \(c^\top y=1\), so \(y\ne0\). Consequently

\[
 \operatorname{sign}(n^\top\widetilde c)
 =\operatorname{sign}(n^\top y),\qquad
 n^\top y\sim
 \mathcal N\!\left(a,\frac{\|n\|^2-a^2}{\kappa}\right).
\]

For nonzero variance, the probability of crossing to the wrong side is therefore

\[
 p_n(c)=\Pr(n^\top\widetilde c<0)=\Phi(-z_n(c)),
 \qquad
 z_n(c)=\frac{\sqrt\kappa\,n^\top c}
 {\sqrt{\|n\|^2-(n^\top c)^2}}.
\]

\(\Phi\) is the standard-normal CDF. This is a conditional, one-perturbation probability for one oriented boundary. If \(a=\|n\|\), its tangent variance is zero and the crossing probability is zero. If \(a=0\) and the variance is positive, it is \(1/2\); the clean action at that boundary still depends on tie handling.

**Matching the literal source.** Write \(\mu=c_{\rm input}/(\|c_{\rm input}\|+10^{-8})\) and \(\kappa_* = \kappa+10^{-6}\). Before floating-point rounding, the actual pre-normalization scalar is Gaussian with mean \(n^\top\mu\) and variance

\[
 \frac{\|n\|^2-(2-\|\mu\|^2)(n^\top\mu)^2}{\kappa_*}.
\]

This follows from \(\|(I-\mu\mu^\top)n\|^2\), without assuming an exact orthogonal projector. The final `norm + 1e-8` denominator still preserves signs. The simpler unit-center formula uses \(\|c\|=1\); it is not a bitwise claim about float32 or near-zero scores.

## What a raw cell margin does and does not control

Let \(v_g\) denote one distinct column of the released intervention instance's \(W\). For the current instance these eight columns are orthogonal with \(\|v_g\|^2=27\), as checked by [`NativeCellProjector`](../solvercell/native_geometry.py). Relevant oriented boundaries include:

| Decision condition | Normal \(n\) | Norm |
|---|---|---:|
| A selected group stays positive | \(v_g\) | \(\sqrt{27}\) |
| An excluded group stays negative when the budget is not full | \(-v_g\) | \(\sqrt{27}\) |
| A selected group outranks a distinct excluded group when the budget is full | \(v_g-v_h\) | \(\sqrt{54}\) |

An identical-column pair has \(n=0\). Its tie is structural, has no random sign crossing, and cannot satisfy a positive separating margin. Canonical prefixes within duplicate groups handle these ties. Simultaneously partial distinct groups can force an exact inter-group tie; the positive-margin projector records such cases as fallback rather than claiming an interior solution.

A single raw margin parameter does not equate crossing risks. The current cell construction requires selected scores at least \(\delta\) and distinct selected/excluded score gaps at least \(2\delta\). These bounds have different normal lengths; their standardized Gaussian margins are different even at equality. Furthermore, an unnormalized projected endpoint \(x\) becomes \(c=x/\|x\|\): every raw bound is divided by \(\|x\|\). A matched projection parameter between point and cell methods is therefore not a matched final noise robustness level. Compute the standardized margin **after** all normalization and lifting.

For a desired standardized margin \(\nu\ge0\), the unit-center formula gives the equivalent single-boundary requirement

\[
 z_n(c)\ge\nu
 \quad\Longleftrightarrow\quad
 n^\top c\ge
 \|n\|\frac{\nu}{\sqrt{\kappa+\nu^2}}.
\]

Choosing \(\nu=\Phi^{-1}(1-p)\) targets a per-boundary crossing probability at most \(p\le1/2\), conditional on satisfying this inequality at the final unit center. It suggests scaling bounds by the normal norm instead of treating every raw score gap identically. It does **not** establish that all requested bounds are simultaneously feasible: even \(G\) orthogonal oriented unit normals with common required alignment \(t\) need \(Gt^2\le1\). Ranking constraints can further restrict the intersection. Projecting in Euclidean space and then normalizing does not by itself enforce the desired unit-sphere bounds.

For a collection of inequalities defining the target cell, a union bound gives
\(\Pr(\text{action changes})\le\min\{1,\sum_n\Phi(-z_n(c))\}\), subject to the same nondegeneracy and tie assumptions. The crossings are correlated and many defining inequalities are redundant, so this can be loose or vacuous. One small per-boundary probability is not a guarantee about the complete action, many perturbations, or an entire episode.

## Consequence for policy learning

Two centers can decode to the same clean action while having different \(n^\top c\), tangent covariances and boundary-crossing probabilities. Their noisy executed action distributions can therefore differ. The clean-center critic trained from those noisy transitions has no general reason to be constant within a clean decoder cell. The retained Bellman smoothing likewise depends on the center and is not an action-invariant value lookup.

Quotienting away the kernel preserves the clean linear decision structure. It does not generally preserve the distribution produced by the original 40-dimensional execution perturbation after rescaling a center to unit norm. Consequently, reusing an old center's policy-improvement weight for a different center in the same clean cell is not an exact action-level mirror-descent theorem under the current native noisy-center critic.

Noise-scaled margins are a possible future design or diagnostic, with explicit feasibility and probability checks. They are not implemented by this note, do not replace the declared fixed-margin experiment, and do not establish that robustness alone will improve trained return.
