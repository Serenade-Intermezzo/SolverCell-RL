# Solver support, feasible exchanges, and the meaning of a prior correction

The released intervention feature map has a substantial **support restriction**, not merely unequal decoder-cell volumes. This is a property of the particular map \(\phi(a)=Wa\). It is not an inherent limitation of linear optimization over binary actions, and it does not by itself establish a diffusion advantage. In particular, the exact one-step penalty from the missing support is small on the audited late visits of the learned quotient-point policy: mean 0.001042 over 512 sampled states. The larger synthetic-state gaps below must not be substituted for that observation.

This note audits the pinned 40-arm, 20-action, budget-10, instance-seed-0 benchmark. It changes no training code, campaign input, or running experiment. The upstream commit is `7e50d03e7129974a11061c9ce500bc2dc5593246`; the audited scientific-source SHA256 is `6eb3126bc49621d34308013091304b3fb46bb17e9c35bec7062d322cf656a66c`.

## Exact support of the current decoder

The decoder selects up to ten strictly positive entries of \(c^\top W\), ordered by decreasing score, with ascending primitive index breaking ties. This is the canonical rule in [`NativeCellProjector.decode`](../solvercell/native_geometry.py). The eight distinct columns \(v_g\) satisfy \(v_g^\top v_h=27\mathbf1[g=h]\). Their primitive-index groups are:

| Group | Primitive indices, zero based | Capacity \(m_g\) |
|---:|---|---:|
| 0 | 5, 9, 17 | 3 |
| 1 | 7, 10, 15 | 3 |
| 2 | 0 | 1 |
| 3 | 2, 6, 12 | 3 |
| 4 | 8, 11, 13 | 3 |
| 5 | 1, 3, 4 | 3 |
| 6 | 19 | 1 |
| 7 | 14, 16, 18 | 3 |

An action's group-count vector \(r_g=\sum_{i\in g}a_i\) determines \(Wa=\sum_gv_gr_g\), and hence its transition law. Distinct group counts give distinct \(Wa\). We count these effects, not primitive binary vectors. There are

\[
\mathcal R=\{r\in\mathbb Z^8:0\le r_g\le m_g,\ \sum_g r_g\le10\},
\qquad |\mathcal R|=9{,}318.
\]

They partition 616,666 feasible binary actions, with multiplicity \(M(r)=\prod_g\binom{m_g}{r_g}\). A group is *partial* when \(0<r_g<m_g\).

For distinct nonzero group scores, the decoder consumes whole positive groups in score order and may truncate the final group at the budget. Therefore:

- Below budget, an effect has positive-volume preimage exactly when every group is full or empty.
- At budget ten, it has positive-volume preimage exactly when at most one group is partial.
- Two or more partial groups require an exact positive tie at the cutoff. They are canonically reachable only when their selected primitive indices form the appropriate global-index prefix within the union of partial groups. Fully selected groups can be placed above that tie and empty groups below it.

The conditions are sufficient by assigning strict scores to full, partial, and empty groups as appropriate. They are necessary because any selected item must outrank every unselected item at a full budget; a selected and unselected member of each partial group have the same group score.

| Class of group-count effect | Count | Probability under a diffuse isotropic cost reference |
|---|---:|---:|
| Positive-volume preimage | 268 | Positive |
| Reachable only on distinct-group tie boundaries | 246 | Zero |
| Never reached by the canonical decoder | 8,804 | Zero |
| All feasible effects | 9,318 | — |

Thus 9,050 effects have zero reference probability. “Zero probability” does not forbid a singular deterministic policy or finite-precision computation from landing on a tie. The split between boundary-only and never-canonical effects depends on the stated tie rule; an optimizer with a different tie rule can differ on those boundaries. The 268 positive-volume **effect** support does not depend on which identical primitive columns win a structural duplicate tie.

A simple impossible effect is selecting primitive 14 alone: its group also contains 16 and 18. With unused budget, a positive group score selects all three, while a nonpositive score selects none. Selecting all three is reachable on an open set. This example is about the feature map; the environment permits both actions.

The script independently checks the support classification using all 109,601 ordered positive-group subsets and all \(3^8=6{,}561\) assignments of group scores from \(\{-1,1,2\}\). The latter represent above-cutoff, cutoff, and excluded groups and suffice to enumerate canonical boundary behavior. It does not estimate these counts by sampling.

## An identity feature map removes this obstruction

For any feasible binary action \(a\), choose primitive costs \(c=2a-\mathbf1\in\mathbb R^{20}\). For every other feasible binary \(b\),

\[
c^\top a-c^\top b=\|a-b\|_1>0.
\]

Hence \(a\) is the unique linear maximizer, even when its cardinality is below ten. Positive normalization preserves the maximizer, so a spherical primitive-cost policy can expose every feasible action. The current holes arise because scores are constrained to \(W^\top c\), with identical scores within each duplicate group.

A future feasible-exchange method must therefore be compared with **native DPMD learning using a primitive-20 decoder**, preserving the native critic and update protocol as far as the changed representation permits. The earlier adapted action-critic harness is not that control. Beating only the rank-eight restricted decoder could show restoration of action support, without demonstrating a diffusion-specific benefit. This note prepares that comparison; it does not implement or launch it.

## The clean isotropic prior is exactly computable

Let \(B=[v_0,\ldots,v_7]/\sqrt{27}\), so \(B^\top B=I_8\). A uniform full-sphere cost is \(c=G/\|G\|\), with \(G\sim\mathcal N(0,I_{40})\). Its group scores are a common positive multiple of the eight independent coordinates \(B^\top G\). A uniform quotient-sphere cost \(Bz/\|z\|\), \(z\sim\mathcal N(0,I_8)\), has the same ordering and signs under a direct coupling. Consequently **uniform \(S^{39}\) and uniform quotient \(S^7\) induce exactly the same clean effect prior** \(\mu(r)\). Removing the nullspace does not correct clean decoder-cell volumes; it can change model capacity and the geometry of the retained execution noise.

A prescribed ordered tuple of \(p\) positive groups has probability \(1/(2^8p!)\). Applying the greedy decoder to every such tuple gives exact integer probability numerators with common denominator

\[
2^8 8!=10{,}321{,}920.
\]

The resulting 268 positive probabilities range from 0.0012974330 to 0.0090959821, a ratio of 7.01075. Entropy is 5.39726 nats, compared with \(\log268=5.59099\) for uniform mass on that support. These numbers describe volume imbalance among reachable effects; they do not quantify the long-term cost of missing effects. Every numerator, multiplicity, and classification is saved in [`group_effects.csv`](../reports/solver_support_audit/group_effects.csv).

## Does the missing support matter to the actual reward?

The environment applies a nonlinear sigmoid to \(Wa\), rather than rewarding a linear surrogate. We evaluated exact known-model expected **one-step** reward using [`InterventionEnv.expected_reward`](../solvercell/environment.py), whose transition formula follows the pinned [`MultiActionRMAB`](../vendor/lsflow/environment/multi_action.py). Every feasible group count was evaluated, as were the 268 positive-volume and 514 total canonical effects.

The group columns have disjoint affected-arm supports, making immediate reward separable across group counts. An independently implemented multiple-choice knapsack dynamic program verified the unrestricted optimum, with maximum absolute disagreement \(1.1\times10^{-14}\). Scalar environment calls also verified the vectorized calculation within \(9\times10^{-15}\).

| Fixed state set | States | States with positive-volume support gap | Mean gap | Median gap | 95th percentile | Maximum gap |
|---|---:|---:|---:|---:|---:|---:|
| Uniform synthetic vectors in \(\{0,1,2,3\}^{40}\) | 128 | 124 | 0.965921 | 0.880928 | 2.127699 | 3.222414 |
| Predecision states from 20 uniform-budget random-policy rollouts | 400 | 196 | 0.032228 | 0 | 0.108831 | 0.874865 |
| Sample of late training visits from completed quotient-point, seed 0 | 512 | 20 | 0.001042 | 0 | 0 | 0.067417 |

The gap is \(\max_{r\in\mathcal R}R(s,r)-\max_{\mu(r)>0}R(s,r)\). Synthetic-state seed is 882100; rollout seeds are 882200–882219, horizon 20, with independently seeded action RNGs. The first two diagnostic state sets do not follow the learned policy. If boundary-only canonical effects are also admitted, their mean gaps fall to 0.871674 and 0.018200 respectively; 122 synthetic and 130 random-policy states still have a strictly positive gap.

The third set comes from the final checkpoint of [`native_revision_20261104/quotient_point`](../outputs/native_revision_20261104/quotient_point/metrics.json), trained for 4,000 steps after 1,000 warmup interactions. Replay has capacity 100,000, size 5,000, and next-write position 5,000, so physical rows 4,000–4,999 are the final 1,000 collected transitions, all after warmup. The audit selects 512 of those rows uniformly without replacement using seed 882300. They are historical late training visits, not fresh rollouts of the final policy or independent state draws. The saved checkpoint uses source `030e1c799e990310b3c2d44a1872b051baefdc141267d42bf048df5c5d1eb53b`, distinct from the current audit source; checkpoint/config/worker agreement and the exact instance hash were checked. Its SHA256 is `447367fe5a4d7c9bf1cee993415e626e9c6d458ec28138687878fbe432f94446`.

For those learned visits, the unrestricted mean best immediate reward is 16.203430, versus 16.202388 on positive-volume support. Admitting all canonical effects reduces the mean gap further to 0.0001116, with a positive gap in only 5/512 states. These observations materially weaken the hypothesis that immediate-reward support loss is the main obstacle near this learned trajectory. They do not rule out longer-horizon or other-state effects, but provide no evidence for such effects either.

For visited state `random_rollout_seed882205_step12`, the unrestricted optimum has counts \((3,3,0,0,1,2,0,1)\) and expected reward 14.114255. The best canonical effect, including boundary-only possibilities, obtains 13.239390. Multiple partial groups are useful because sigmoid marginal returns can favor distributing interventions, whereas the linear score decoder consumes groups greedily. Full states, maximizing effects, and scalar values are in [`state_results.json`](../reports/solver_support_audit/state_results.json).

This establishes existence and diagnostic frequency of an immediate-reward obstruction. It does **not** establish long-term optimality gaps, trained return improvements, or that exchange diffusion will learn the unrestricted optimum.

## What feasible-exchange diffusion would need to specify

The feasible effect graph can use legal additions/deletions \(r\leftrightarrow r+e_g\), with optional transfers \(r\leftrightarrow r-e_g+e_h\). Add/delete edges connect every feasible effect through zero. Transfers alone preserve cardinality and cannot cover the full up-to-budget action space.

The reference law is part of the method. Equal symmetric continuous-time rates on each undirected feasible edge yield uniform stationary mass over effects. Choosing a uniformly random neighbor at a fixed total jump rate instead yields degree-weighted mass. Neither should be called uniform without checking its generator. Symmetric primitive-bit flips aggregate to group birth/death rates proportional to \(m_g-r_g\) and \(r_g\), and produce stationary effect mass proportional to \(M(r)\). Uniform feasible primitive actions and uniform feasible effects are different objectives.

These observations identify a concrete feasible-state diffusion construction and the controls needed to test it. They are not a novelty or performance claim.

## What solver-prior correction can actually claim

Fix a cost reference \(\rho(c)\), decoder \(D\), and its pushforward cell mass \(\mu(r)=\int\mathbf1[D(c)=r]\rho(c)\,dc\). For an effect-invariant utility \(Q(r)\), the explicitly chosen objective

\[
\max_\pi\ \mathbb E_\pi Q-\tau\,\mathrm{KL}(\pi\|\mu)
\]

has solution \(\pi(r)\propto\mu(r)e^{Q(r)/\tau}\). If the desired reference is instead \(u(r)\), reweighting costs by

\[
\frac{u(D(c))}{\mu(D(c))}\exp\!\left(\frac{Q(D(c))}{\tau}\right)
\]

produces the desired \(u(r)e^{Q(r)/\tau}\) distribution **only on supported effects**, and only when the needed absolute-continuity and integrability conditions hold. Equivalently, use the correction \(Q+\tau\log u-\tau\log\mu\) on that support. Uniform mass over all 9,318 effects is impossible through importance weighting when 9,050 have \(\mu=0\); the representation or decoder support must change. Uniform primitive-action mass induces \(u(r)=M(r)/616{,}666\), while uniform effect mass uses \(u(r)=1/9{,}318\).

This is a statement about a declared KL-to-reference or entropy objective, not a generic correction to policy mirror descent. A proximal update \(\pi_{t+1}\propto\pi_t e^{\eta Q_t}\) already has the intended previous-policy reference. Repeatedly subtracting \(\log\mu\) changes that update and needs its own objective; the initial geometric prior alone does not make the update biased.

The retained native critic evaluates a **clean center whose executed coefficient is noisy**. Its value need not be constant on a clean decoder cell, as derived in [`NATIVE_NOISE_GEOMETRY.md`](NATIVE_NOISE_GEOMETRY.md). Accordingly, the effect-invariant \(Q\) assumption above cannot be silently applied to the current noisy-center critic. Likewise, \(\mu\) is an integrated cell probability, not a pointwise cost density. A reciprocal Monte Carlo frequency is not generally unbiased and is undefined at an unobserved cell without additional conventions. The exact prior calculation avoids that estimation problem for this fixture only.

## Reproduction and evidence

Run from the workspace root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python analysis/solver_support_audit.py
```

The completed CPU audit writes [`summary.json`](../reports/solver_support_audit/summary.json), the complete effect table, and all state-level results. The default command requires the existing local quotient-point checkpoint plus its config, worker, and final metrics; `--learned-checkpoint` can specify its location. The summary records source/script hashes, instance hash, the learned-state checkpoint hash and sampled physical rows, independent enumeration checks, and oracle errors. It verifies that the scientific-source hash is unchanged. These artifacts support this structural and one-step analysis; training campaign results remain separate evidence.
