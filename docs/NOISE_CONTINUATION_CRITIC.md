# Continuation values for the actual denoising policy

**Assessment:** a critic indexed by denoising state is mathematically appropriate, but is not a new algorithm by itself. A more specific candidate is to integrate final solver decisions on small conditional branches. However, the cost-matched diagnostic below is **negative** for its simplest two-query implementation: in all 12 fixed cases, its full-gradient variance exceeds averaging two independent score-function samples. Established conditional-expectation theory therefore does not yet justify an efficiency claim for this candidate.

This is a design and mathematical audit, not an implementation or RL result. It changes no scientific code or experiment input. In particular, it does not reinterpret the completed native actor-replacement experiments as tests of the method below.

## 1. The value must follow the policy's actual continuation law

Fix an environment state \(s\), a frozen old policy, a parameter-independent initial law \(p_K\), and stochastic denoising kernels

\[
Z_K\sim p_K,\qquad Z_{k-1}\sim P_k^{\mathrm{old}}(\cdot\mid s,Z_k),
\qquad k=K,\ldots,1.
\]

Suppose the executed action is exactly \(A=D(Z_0)\). Define the *action* value \(Q_{\mathrm{env}}^{\mathrm{old}}(s,a)\) for taking that environment action and subsequently following the old outer policy. Then

\[
F_s(z)=Q_{\mathrm{env}}^{\mathrm{old}}(s,D(z)),\qquad
C_0(s,z)=F_s(z),\qquad
C_k(s,z)=\mathbb E_{P_k^{\mathrm{old}}}[C_{k-1}(s,Z_{k-1})\mid Z_k=z].
\]

By the tower property, \(C_k=\mathbb E[F_s(Z_0)\mid s,Z_k]\). The kernels are the actual remaining inference steps, including their noise schedule and discretization. Arbitrarily forward-corrupting a clean replay action does not generally produce this conditional law. An exact forward/reverse joint construction would be needed to identify the two.

There is no environment discount between these internal steps: their discount is one when the objective is the unchanged terminal utility. An additional denoising discount is a changed weighting scheme, not this exact recursion. Freeze the old kernels and terminal critic while fitting a batch of continuation targets, or explicitly manage the resulting moving target.

If execution has a further stochastic transformation \(A=D(T_\eta(Z_0))\), the correct boundary is instead

\[
C_0(s,z)=\mathbb E_\eta[Q_{\mathrm{env}}^{\mathrm{old}}(s,D(T_\eta(z)))].
\]

This matters here. [`NativeLSFlow.act`](../solvercell/upstream.py) retains the clean center in replay but executes a perturbed coefficient; [`TensorNativeLSFlow`](../solvercell/fast_native.py) retains center-dependent Bellman smoothing. Its native latent critic is not an action lookup. Substituting that critic for \(Q_{\mathrm{env}}(s,D(z))\) silently changes the object being optimized. See [the execution-noise audit](NATIVE_NOISE_GEOMETRY.md).

## 2. Exact credit assignment, and its approximation boundary

For a realized transition \(z\to y\), the exact internal advantage is

\[
A_k^{\mathrm{old}}(s,z,y)=C_{k-1}(s,y)-C_k(s,z).
\]

Its conditional mean is zero. At the old parameters, the policy gradient for the fixed-state terminal objective is

\[
\nabla_\theta J_s\big|_{\mathrm{old}}=
\mathbb E_{\mathrm{old}}\sum_{k=1}^{K}
\left.\nabla_\theta\log P_k^\theta(Z_{k-1}\mid s,Z_k)\right|_{\mathrm{old}}
A_k^{\mathrm{old}}(s,Z_k,Z_{k-1}).
\]

This follows by conditioning the terminal-return score estimator on the present transition and removing a state-only baseline. It requires no derivative through the solver. The outer environment policy-gradient theorem additionally supplies its state-visitation weighting; a replay-state actor objective is not automatically the same weighting.

For fitting \(C_k\), a complete continuation to \(Z_0\) supplies a Monte Carlo target; a single real denoising transition supplies the bootstrap target \(\operatorname{stopgrad}\widehat C_{k-1}(s,Z_{k-1})\). A learned terminal \(\widehat Q\) and bootstrap error introduce bias. Rao–Blackwellization can reduce conditional sampling variance without correcting either critic error. Past latent transitions also become off-policy when the denoiser changes.

Gaussian transition likelihoods are available only on steps with nonzero variance. A deterministic DDIM/ODE step does not acquire a Gaussian likelihood by calling it a diffusion step. Clipping PPO ratios, taking many optimization steps, or reusing stale continuation values also removes any immediate claim that this exact local-gradient identity guarantees policy improvement.

## 3. What “smooth” means here

Assume bounded terminal values, fixed positive-definite \(\Sigma_k\), and
\(P_k(\cdot\mid z)=\mathcal N(m_k(z),\Sigma_k)\). Even if \(C_{k-1}\) is only bounded and measurable,

\[
C_k(z)=\int C_{k-1}(y)\,\mathcal N(y;m_k(z),\Sigma_k)\,dy
\]

is a Gaussian convolution evaluated at \(m_k(z)\). It is smooth in that Gaussian mean. As a function of \(z\), it is only as smooth as the mean map: a ReLU denoiser does not make the composition globally infinitely differentiable. Smooth, uniformly nondegenerate parameter-dependent covariances admit analogous differentiation under suitable domination conditions.

The clean boundary \(F_s\) remains piecewise constant with jumps across solver cells. A zero-variance final step need not smooth it; a positive Gaussian step farther upstream can smooth the resulting bounded function. Derivatives can become large near boundaries as variance tends to zero. Thus neither universal smoothness at every level nor well-conditioned optimization follows merely from having finitely many Gaussian steps.

This construction estimates the actual probability of each eventual decision:
\(C_k(s,z)=\sum_a Q_{\mathrm{env}}(s,a)\Pr_{\mathrm{old}}(D(Z_0)=a\mid Z_k=z)\).
It is more specific than evaluating an arbitrary noisy cost with the clean action critic, or using one additional tangent perturbation as a stand-in for all remaining inference steps.

## 4. Prior art rules out the generic novelty claim

[DDPO](https://arxiv.org/abs/2305.13301) already formulates denoising as sequential decisions optimized through transition likelihoods. [DPPO](https://arxiv.org/abs/2409.00588) applies a nested denoising/environment MDP to control. More directly, **[DIA, September 2026](https://arxiv.org/html/2609.12245v1#S4.SS3)** defines the expected final-action Q-value conditioned on an intermediate denoising state, with the final Q-value as boundary, and forms an inner denoising advantage. Its equations 15–18 substantially overlap the proposed \(C_k,A_k\) construction; reversing the timestep index does not distinguish it.

[BDPO](https://ai.gaocx.io/bdpo/) also uses a two-time-scale actor-critic construction with intermediate diffusion values and pathwise regularization. [OGPO](https://simchowitzlabpublic.github.io/ogpo-site/) uses an off-policy critic as the generative process's terminal reward. A combinatorial decoder does not make these generic constructions new. The defensible research question must involve a solver-specific estimator or structural result beyond adding a noise-level embedding to a critic.

## 5. A precise solver-specific candidate: integrate the final decision branch

Consider the final actual transition in **primitive-cost coordinates**,

\[
X\mid(s,z_1)\sim\mathcal N(m,\operatorname{diag}(\sigma_1^2,\ldots,\sigma_d^2)),
\qquad \sigma_i>0,
\]

followed immediately by the positive up-to-\(b\) top-k decoder, \(b\ge1\). Freeze \(s,z_1\) and the terminal action critic. Condition on all coordinates \(X_{-i}=v\). Let

\[
\tau(v)=\max\{0,\text{the }b\text{-th largest outside score}\},
\]

with the absent order statistic treated as \(-\infty\). There are exactly two possible actions away from a probability-zero equality:

- \(a^-\): select the positive top-\(b\) outside items, excluding \(i\);
- \(a^+\): select \(i\) and the positive top-\((b-1)\) outside items.

Both are feasible. With \(h=(m_i-\tau)/\sigma_i\), \(p=\Phi(h)\), and \(q^\pm=\widehat Q(s,a^\pm)\),

\[
\overline F_i(v)=(1-p)q^-+pq^+,\qquad
\partial_{m_i}\overline F_i(v)=\frac{\varphi(h)}{\sigma_i}(q^+-q^-).
\]

Thus two feasible critic queries integrate **all** values of one final Gaussian coordinate exactly. No cost-to-action derivative, straight-through decoder, or gradient of \(\widehat Q\) with respect to its action is needed. Positive final normalization preserves this decoder and does not affect the formula. Conditioning on one quotient group score of the native duplicate-column decoder instead gives intervals separated by zero and the other seven group scores: at most nine intervals, each with an exactly computable univariate Gaussian mass and one decoded action. This variant retains that decoder's support restrictions.

For a full mean-score vector \(G_j=(X_j-m_j)(F(X)-\beta)/\sigma_j^2\), with a fixed parent-state baseline \(\beta\), integrating coordinate \(i\) gives

\[
\overline G_i=\frac{\varphi(h)}{\sigma_i}(q^+-q^-),\qquad
\overline G_j=\frac{v_j-m_j}{\sigma_j^2}(\overline F_i(v)-\beta),\quad j\ne i.
\]

This is \(\mathbb E[G\mid X_{-i}]\), so it preserves the expected score gradient and satisfies
\(\operatorname{Cov}(G)-\operatorname{Cov}(\overline G)=\mathbb E[\operatorname{Cov}(G\mid X_{-i})]\succeq0\).
Multiplication by the mean-network Jacobian is valid with the sampled outside scores and critic values held fixed. Backpropagating through those outside samples as well would add unintended terms to this score estimator. For a learned scale, the integrated coordinate's log-scale derivative is \(-h\varphi(h)(q^+-q^-)\).

For a complete trajectory score estimator, condition on the sampled denoising prefix as well. Earlier transition score factors are then fixed, so their terminal-return multipliers can also be replaced by \(\overline F_i\). Marginalizing the **whole** estimator this way gives a genuine conditional expectation, without adding a pathwise derivative through the sampled prefix. A coordinate chosen before the terminal noise, independently of that noise conditional on the parent state, gives an unbiased mixture of these full-vector estimators without inverse-selection weighting. This differs from retaining only one coordinate's gradient, which needs the corresponding coordinate-sampling correction.

One can instead compute each coordinate's separately marginalized mean derivative and sum their parameter contributions. That remains unbiased, but a covariance reduction for the *sum* does not follow from separate per-coordinate variance reductions. Similarly, an arbitrary substitution of \(\overline F\) into a clipped PPO product is not the Rao–Blackwellization of that product: the likelihood ratio and clipping can depend on the integrated coordinate. Integrate the entire estimator, or state that a different surrogate is being used.

The conditional terminal expectation can also train \(C_1\), followed by the actual-kernel recursion for earlier levels. The same two critic queries must be compared with two ordinary continuation samples, not one, when evaluating variance per unit compute. General conditional expectation and action enumeration are established methods; see [Rao–Blackwellized stochastic gradients](https://proceedings.mlr.press/v97/liu19c.html) and [expected policy gradients](https://arxiv.org/abs/1801.03326). The narrow candidate contribution is this feasible, solver-conditioned integration inside generative policy learning, not Rao–Blackwellization itself. Literature coverage here does not certify novelty.

## 6. Exact two-step sanity check, not an RL experiment

Let \(Z_1=\theta+\epsilon_2\), \(Z_0=Z_1+\epsilon_1\), with independent standard Gaussians, decoder \(D(z)=\mathbf1[z>0]\), and terminal reward \(F=D\). Then

\[
C_0(z)=\mathbf1[z>0],\qquad C_1(z)=\Phi(z),\qquad
J(\theta)=\Phi(\theta/\sqrt2).
\]

At \(\theta=0\), use the identical baseline \(1/2\). The sampled terminal estimator is \(G=Z_1(F-1/2)\); the continuation estimator is \(\overline G=Z_1(\Phi(Z_1)-1/2)\). Both have mean \(1/(2\sqrt\pi)=0.2820947918\). Their exact variances are

\[
\operatorname{Var}(G)=\frac14-\frac1{4\pi}=0.1704225285,
\]
\[
\operatorname{Var}(\overline G)=\frac1{12}+\frac1{2\pi\sqrt3}-\frac1{4\pi}
=0.0956440110.
\]

The ratio is 0.56121695. One-dimensional numerical quadrature independently checked the closed-form mean and variance to below \(10^{-15}\). This confirms the known conditional-expectation identity for one transition term. It does not validate a learned critic, a full shared-parameter gradient's variance, a PPO update, or benchmark performance.

## 7. Actual top-k and nonlinear benchmark-oracle check

[`continuation_branch_audit.py`](../analysis/continuation_branch_audit.py) checks the branch formula with the actual primitive-20 positive-top-10 decoder and the benchmark's exact nonlinear sigmoid expected immediate reward. It uses three fixed learned-visit states, selected at indices 0, 200, and 400 of the preceding 512-state audit, and predeclared integrated coordinates 0, 5, 10, and 15. The test kernel is \(X\sim\mathcal N(0,0.5^2 I_{20})\), with nonzero final noise throughout. This is a declared diagnostic kernel, **not** the learned policy's terminal kernel. No noisy-center critic is used.

An independent 2,048-draw pilot supplies one constant baseline per state, shared across all methods. Each of 12 cases uses 20,000 samples per estimator and compares:

| Estimator | Logical Q queries per estimate | Gradient components retained |
|---|---:|---|
| Ordinary score function | 1 | All 20 |
| Mean of two independent score-function draws | 2 | All 20 |
| Integrate one coordinate, retain other conditional score terms | 2 | All 20 |

The conditional two-branch integral agrees with independent Gaussian quadrature within \(6.3\times10^{-15}\); its mean derivative agrees with a central finite difference within \(1.8\times10^{-10}\). Scalar environment reward checks agree within \(5.4\times10^{-15}\), and direct calls to the independent Torch decoder confirm both branch actions. The estimator is unbiased for the declared frozen-oracle objective by the conditional-expectation derivation; these finite checks do not independently prove that theorem.

**All 12 cases favor two independent samples at the equal-query budget.** The ratio of full-vector covariance traces, RB/two-sample SF, is 1.473–1.862. Integrating coordinate 0 or 5 gives ratios around 1.83–1.86; coordinates 10 or 15 give 1.47–1.50. The RB estimator improves over one SF draw as expected, but does not recover the extra query cost in this test. Monte Carlo means are compatible with the shared target; the maximum component difference is 2.77 pooled standard errors across the reported cases, which is a diagnostic, not a formal multiple-comparison test.

The result does not rule out different coordinates, blocks, kernels, baselines, or cheap batched action-value evaluation. It also gives no license to select those after seeing these results and report them as predeclared gains. This initial cost evidence does **not** support recommending the two-query RB estimator as an established improvement. Full state vectors, seeds, gradient means/variances, hashes, and all cases are preserved in [`summary.json`](../reports/continuation_branch_audit/summary.json) and [`variance_comparison.csv`](../reports/continuation_branch_audit/variance_comparison.csv).

Reproduce the bounded CPU diagnostic with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python analysis/continuation_branch_audit.py
```

## 8. What a fair tailored algorithm would have to test

The current [`UnitDiffusionPolicy.sample`](../solvercell/diffusion_service.py) does use explicit Gaussian Euler–Maruyama transitions before final normalization when `diffusion_scale > 0`. However, it then participates in the separately perturbed native execution protocol. The two-branch formula above is not an exact drop-in target for that retained tangent perturbation. A new experiment must either include all execution randomness in its boundary expectation or declare a changed execution protocol and match it across controls.

The narrow design under consideration would use an action-value critic on actual feasible actions, the declared stochastic generative kernel, and conditional final-branch integration, with continuation learning only if it adds measurable value. Before RL implementation, a cost-aware estimator study must explain or overcome the negative diagnostic above. The earlier support audit likewise gives no strong local myopic reason to expect a large gain: the sampled learned late visits had mean missing-support loss only 0.001042. See [the complete support/prior audit](SOLVER_SUPPORT_AND_PRIOR.md).

Three distinctions are essential before making a diffusion claim:

1. Compare with generic terminal-return policy gradients and DIA-style continuation values under the **same** diffusion kernel, critic, action support, and query budget. This isolates solver integration from established credit assignment.
2. Compare primitive-20 and restricted rank-eight representations separately. Restoring support cannot count as a diffusion benefit.
3. Compare flow matching with the **same final Gaussian transition and conditional branch estimator**. The final integration also works there. Multimodality, smoothing, feasible terminal decoding, and this Rao–Blackwell step are not exclusive to diffusion. A diffusion-specific advantage would require evidence that stochastic refinement at multiple earlier levels helps beyond the matched terminal-noise control.

Stochastic diffusion supplies tractable local likelihoods and an actual sequence of conditional decision probabilities; a deterministic ODE sampler does not supply those Gaussian likelihoods internally. This is a concrete algorithmic convenience, not a general superiority theorem. Whether it outweighs longer inference, critic approximation, and extra value queries is an empirical question left open by this note.
