# Shared-prefix preference sampling: prespecified v3 mechanism study

## Motivation and target

The v2 guide has no downstream selection effect. This task tests a different diffusion capability: reuse the computation of an actual stochastic prefix and obtain multiple fresh conditional suffixes. No learned potential, resampling threshold, actor/critic update, artificial flow noise or decoder relaxation is introduced. The trained policies remain exactly the published v2 checkpoints.

For independent roots X_j at generation index k and R_j conditionally independent suffixes, each leaf has the original base-policy marginal. The root-wise estimators of E_base[Y] and E_base[Y f] are

```
Zhat = (1/M) sum_j (1/R_j) sum_r exp(U(D(c_jr))/tau)
Fhat = (1/M) sum_j (1/R_j) sum_r exp(U(D(c_jr))/tau) f(D(c_jr)).
```

Both are unbiased for static allocations chosen before the scored suffixes. Thus normalized candidate selection uses log weight U/tau-log(R_j). A root with more children must not acquire extra probability merely from that count. Selection from a finite weighted pool remains a biased approximation, as in ordinary self-normalized importance sampling. Consistency requires more independent roots, not just more children of a fixed root. Weight ESS does not capture sibling correlation.

With uniform R, A=Var(E[Y|X_k]) and B=E[Var(Y|X_k)],

```
Var(Zhat) = A/M + B/(M R)
actor NFE = M k + M R (K-k).
```

At a matched actor-forward-row budget, the ideal continuous allocation is R*=sqrt(B k/(A (K-k))). This is ordinary nested Monte Carlo reasoning, not a novelty claim. Actual action-probability/utility ratio estimators have their own influence functions; the audit reports those as well. Fewer actor NFE alone does not imply lower wall cost or lower solver/Q cost.

## Parents, controls and immutable boundary

Parent scientific source is`33fcff51897f7485ddce4a303d089526d0cada8e165a4c380e7c9509db6680b3`. Checkpoints are the selected temperature.1 sphere/diffusion models at training seeds0/1/2, instance0,4000 updates. Parent SHA256s/configs are frozen before launch. Every Python file in the parent's recorded source map must still match exactly; this extension only adds a sampler module and analysis entrypoints. These are fixed-checkpoint inference experiments, not new training or new training seeds.

Independent controls use1 direct sample and terminal weighted pools of3/4/5/6/7/8. Diffusion additionally uses roots M in{2,4}, children8/M and branch index k in{2,4,6}. All branching pools score8 leaves. With K8, their actor NFE are52/40/28 (M2) and56/48/40 (M4), versus64 for independent8. Q/decoder budgets match independent8; matched-NFE controls can have fewer Q/decoder calls and must be reported separately. The actual sphere Heun sampler is deterministic conditional on its prefix; it uses independent controls, not duplicated children counted as fresh samples.

## Screen, diagnostics and selection

Seed0 model screen: validation weights0/.5/1,32 complete H20 episodes, common fresh seeds700000–700031. Benchmark inference-only latency separately at two fixed replay observations and all three weights; warm each mode then randomize its order within32 timing cycles. Record latency samples and environment-inclusive wall time. Each mode has its own injected RNG; no learner-owned/global RNG, model parameter, optimizer, environment training state or checkpoint may be modified.

Support diagnostics use the same two replay rows,128 base and128 native Gaussian-prior candidates, exact one-step expected components, and selected-action discounted finite-horizon Monte Carlo continuation with8 CRN repetitions of the actual conditional base policy. One-step ordering is not future-value calibration; candidate selection is diagnostic, not an unbiased population ranking estimate. Finite Monte Carlo error and critic errors remain separate.

Diffusion nested-variance audit uses32 independent prefixes×32 fresh suffixes at indices2/4/6, two fixed observations and weights0/.5/1. Estimate within variance and between-prefix variance with the within/R correction. Also report variance traces for action-probability influence functions and utility influence functions, finite-MC limitations, leaf action diversity, and sibling action agreement. This diagnoses where computation sharing might help; it does not choose settings using final episodes.

Selection is frozen from seed0 validation only. Eligible topology: average utility across weights at least98% of independent8, each weight at least95%, median randomized inference latency no more than105% of independent8, and at least10% lower actor NFE. Among eligible topologies prefer lower NFE, then higher utility, then lower latency. These are engineering screen thresholds, not confidence or significance claims. Paired episode differences/SE must be reported. If none passes, lock a no-promotion decision and still evaluate the best utility topology diagnostically without changing the default.

## Fixed decision evaluation

After selection is locked, evaluate the selected/diagnostic topology and independent controls on all three fixed model seeds, unseen preference weights.125/.375/.625/.875,32 complete H20 episodes with fresh seeds800000–800031. Seed0 was used for method selection; model seeds1/2 provide out-of-selection model checks, not newly trained models. No tuning follows from these final values in this task. Report complete episode vectors, both component responses, actual actor/Q/decoder calls, prefix/leaf counts, weight ESS limitations and wall cost. No claim of beating tuned sphere or intrinsic flow/diffusion speed follows from fewer forward rows.

## Sanity and stopping gates

One child per root must exactly recover the original independent sampler at every split; unequal children must preserve root weights; leaf Gaussian marginals and sibling covariance must match an analytic test kernel; terminal forks create duplicates rather than diversity; actual forward/Q/decoder counts must match hooks; all invalid supports must be rejected. Run the full suite and a bounded debug execution before deployment. Use an immutable server snapshot and at most two single-thread CPU workers with durable metadata/exit records. No GPU training. Retain all negative controls and source/parent identities. Synchronize code and reports to the already-authorized public GitHub repository after verification.
