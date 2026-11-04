# Preference conditioning: what the released benchmarks actually support

**Recommended first fixture:** split Dynamic Intervention's existing post-transition reward into its two synthetic response cohorts and condition the policy on their relative utility weight. This preserves the exact dynamics, budget, reward observations, and native decoder support. An explicit pair of feasible actions reverses preference as the weight changes. Existing replay can reconstruct both reward components.

Disease Network can support priorities over early detection or coverage of declared node groups, but it is a static hidden-label testing simulator, not an epidemic-control or treatment simulator. The supplied local checkout is also missing its required graph data. No fitted-disease-graph preference conflict or policy improvement has been established here.

This audit uses upstream commit `7e50d03e7129974a11061c9ce500bc2dc5593246`, current source SHA256 `6eb3126bc49621d34308013091304b3fb46bb17e9c35bec7062d322cf656a66c` as recorded in the report. It changes no training code, dynamics, or campaign inputs. Source details are in the [audit artifact](../reports/preference_component_audit/summary.json).

## 1. Dynamic Intervention: existing semantics

[`MultiActionRMAB.step`](../vendor/lsflow/environment/multi_action.py) first computes each arm's upward probability from a sigmoid of \(Wa\), then independently samples an upward/downward move and clips the new state to \(\{0,1,2,3\}\). Reward is the sum of the **new** states' rewards, not a pre-action reward:

\[
r_t=\sum_{j=1}^{40}u_j(s_{t+1,j}),\qquad a_t\in\{0,1\}^{20},\quad\|a_t\|_1\le10.
\]

[`MultiStateRMAB`](../vendor/lsflow/environment/base_envs.py) supplies heterogeneous reward tables. Arms 0–9 have low-state rewards \((0.2,0.15,0.1)\) and terminal-state rewards drawn from 4–6; the other arms have \((0.1,0.15,0.2)\) and terminal rewards from 1–3. Thus even “higher state means higher immediate utility” is not universally true at low states. The episode horizon is 20 in our pinned instance. [`InterventionEnv`](../solvercell/environment.py) preserves these rules and has an exact one-step reward oracle.

The weight constructor independently names two **response** cohorts: `bad_arms` 0–19 have smaller sigmoid amplitudes and lower response thresholds; `good_arms` 20–39 have larger amplitudes and higher thresholds. Each has ten associated primitive interventions. These labels are synthetic construction parameters. They are neither demographic categories nor evidence that one population should receive lower normative priority. The response-cohort boundary also differs from the boundary of the high-terminal-reward arms above.

There is a hard action-count cap. There is no monetary intervention price, capacity carried across rounds, action-switch penalty, adverse-effect state, or collision/safety event in this transition law. Adding any such objective must be labeled as an extension, even when its numerical feature is cheap to compute.

## 2. A minimal preference vector with an actual conflict

Let \(L=\{0,\ldots,19\}\), \(H=\{20,\ldots,39\}\), and define per-capita utility components

\[
f_L(s')=\frac1{20}\sum_{j\in L}u_j(s'_j),\qquad
f_H(s')=\frac1{20}\sum_{j\in H}u_j(s'_j).
\]

For a preference \(w=(\lambda,1-\lambda)\), a scalar reward with the original numerical scale is

\[
r_w(s,a,s')=40\,[\lambda f_L(s')+(1-\lambda)f_H(s')].
\]

At \(\lambda=1/2\), this equals the original scalar reward. Neither the transition law nor feasibility depends on \(w\). Component normalization is fixed before comparing methods; changing it changes the meaning of the preference.

An exact one-step witness uses the valid state \(s=(2,\ldots,2)\):

| Feasible action | Primitive indices | Budget used | Expected \(f_L\) | Expected \(f_H\) | Original expected reward |
|---|---|---:|---:|---:|---:|
| All low-response-target interventions | 1, 3, 4, 8, 11, 13, 14, 16, 18, 19 | 10 | 0.808913 | 0.168500 | 19.548252 |
| All high-response-target interventions | 0, 2, 5, 6, 7, 9, 10, 12, 15, 17 | 10 | 0.184000 | 0.953863 | 22.757262 |

The first is strictly better for the low-response cohort and strictly worse for the high-response cohort. Their weighted ordering changes at \(\lambda=0.5568862\): the second wins at \(\lambda=0.25\) and \(0.5\), while the first wins at \(0.75\). Both are reachable by the original \(W^\top c\) decoder: assign arm costs +1 to the intended cohort and −1 to the other. Thus this conflict does not rely on restoring missing decoder support or inventing a new penalty.

This proves a real conflicting pair under the released dynamics. It does not prove that either action lies on the full long-term Pareto frontier, that this constructed state is typical, or that a conditioned diffusion policy learns the optimal tradeoff. Those are later questions. The earlier [support audit](SOLVER_SUPPORT_AND_PRIOR.md) found only a small mean immediate support gap on learned late visits; missing support is not the main motivation here.

The [component audit](../analysis/preference_component_audit.py) also reconstructs \((f_L,f_H)\) from all 5,000 transitions in the completed quotient-point replay, using `next_obs` and the frozen reward table. Recovering the original scalar reward has maximum error \(2.37\times10^{-6}\), explained by the original float32 sum versus groupwise accumulation. No new environment interaction is needed to label these components.

“Balanced cohort utility” is an allocation preference, not a fairness guarantee. The reward tables attach different utilities to different arms. Equalizing raw utilities, equalizing state-3 occupancy, and equalizing intervention exposure are distinct objectives. Minimum-group expected return, expected minimum group return, and a weighted average are also distinct; a linear weight vector does not silently implement all of them.

## 3. Which intervention extensions are cheap, and which claims would be new?

| Candidate component | Recoverable from current replay? | Status and limitations |
|---|---|---|
| Cohort utility means above | Yes: `next_obs`, instance reward tables | Decomposition of existing scalar reward; new preference weights only |
| State occupancy, e.g. fraction at state 3 | Yes: `next_obs` | New performance definition; not identical to the original reward |
| Intervention count \(\|a\|_1\) | Yes: saved `action` | New soft price under the unchanged hard cap; observed replay uses counts 0–10 |
| Action churn \(\|a_t-a_{t-1}\|_1\) | Needs ordered transitions and episode boundaries | No switching harm exists in the source; a Markov policy/critic for this extension needs previous action in its state |
| Return variance, downside probability, CVaR | Needs complete trajectories and repeated policy evaluation | Not a simple one-step reward component; cannot be inferred from mean critic values |

The first two cohort components are the least invasive choice. A soft cost coefficient can be a later extension, but a hard budget should remain hard rather than be traded away by a preference weight. Source parameters such as `reward_scale`, DPMD's mirror-descent temperature, and execution-noise concentration are not user utility weights.

## 4. Disease Network: detection order in a fixed hidden world

The actual driver uses [`DiseaseDPMDEnv`](../vendor/lsflow/driver_disease_lsflow.py), which exposes a status vector: −1 means unknown, 0/1 means revealed. [`BinaryFrontierEnvBatch.reset/step`](../vendor/lsflow/environment/frontier_batch_env.py) samples one `world_X` at reset and never evolves it during the episode. An action reveals nodes from the **pre-step** frontier; observations within one batch cannot affect the rest of that batch. Frontier expansion follows any tested node, regardless of whether its result is positive. Reward is

\[
r_t=\sum_i a_{t,i}X_i,
\quad a_{t,i}=0\text{ outside the frontier},\quad\sum_i a_{t,i}\le B.
\]

No node can be tested twice. Done means all nodes have been tested. There is no infection spreading during a rollout, no treatment, and no prevention effect from testing. “Safety” cannot therefore mean infections averted without adding new mechanisms.

[`BatchGraphApproximator.solve_from_coeffs`](../vendor/lsflow/approximator/batch_graph_approximator.py) actually chooses exactly \(\min(B,|\mathrm{frontier}|)\) nodes, even when all their costs are negative. This differs from Intervention's positive-only decoder. A per-test price is constant within a state whenever every available action fills that count, and its undiscounted complete-episode total is always \(n\). Similarly, on a given hidden world, complete-episode undiscounted total positives equal \(\sum_i X_i\) for every completing policy. The interesting existing task is **which cases are found early**, through discounting or a genuine finite testing horizon, rather than changing the full-world total.

A switching-cost analogy is especially weak here: successive testing sets are disjoint, so \(\|a_t-a_{t-1}\|_1=\|a_t\|_1+\|a_{t-1}\|_1\). Penalizing this “churn” mainly penalizes testing volume and does not model discomfort or switching an ongoing treatment. No such harm exists in the source.

## 5. Disease preference components available from the same observations

For fixed, verified node groups \(G_g\), define newly tested indicators
\(b_i=\mathbf1[s_i=-1,\ s'_i\in\{0,1\}]\). Then both of the following are recoverable even from replay that saved status transitions and latent costs but omitted the discrete action:

\[
f_g^{\mathrm{detect}}(s,s')=\frac1{|G_g|}\sum_{i\in G_g}b_i s'_i,
\qquad
f_g^{\mathrm{coverage}}(s,s')=\frac1{|G_g|}\sum_{i\in G_g}b_i.
\]

Weighted **discounted early detections across groups** are a natural same-dynamics extension. Coverage priorities are another declared objective, not existing clinical utility. Status already records cumulative coverage, so explicitly specified coverage-gap penalties can be computed from the current transition; their fairness interpretation still needs a definition. Group recall divided by *all infected people in that group* cannot be reconstructed from a truncated replay with untested nodes. It requires evaluator-only access to the episode's hidden world, which must not leak into the policy.

[`ICPSR22140Processor`](../vendor/lsflow/environment/ICPSR_22140_processor.py) names covariates including `RACE`, `ETHN`, and `SEX`, and creates one-hot encodings from sorted observed codes. This establishes that covariates influence the fitted model; it does not provide a ready, verified group-index dictionary in this local checkout. A demographic fairness benchmark must pin the data dictionary and encoded category mapping. Graph-component or explicitly synthetic cohorts are possible alternatives but must be labeled as such.

There is a simple **synthetic**, not ICPSR, conflict: two isolated frontier nodes, budget one, independent positive probabilities 0.9 and 0.1, one node in each group. Testing the first gives expected group rewards \((0.9,0)\); testing the second gives \((0,0.1)\). Their one-batch ordering switches when the first group's weight crosses 0.1. Both complete testing orders have expected undiscounted total reward one. This illustrates the distinction between early allocation and total detections, but establishes no quantitative tradeoff on the released fitted disease graph.

## 6. Disease reproducibility and risk boundaries

The local checkout has none of the three TSV files expected under `ICPSR_22140/DS000*`, no `ICPSR_22140.pkl`, and no ready graph cache. Shipped parameter pickles are under `ICPSR22140/checkpoints`, whereas [`load_disease_graph_instance`](../vendor/lsflow/environment/disease_graph_loader.py) looks under `ICPSR_22140/checkpoints`. Consequently this audit has not executed a fitted Disease benchmark and does not report one as ready.

Two further source details matter before interpreting risk or detection fairness:

- [`LogJunctionTree.sample_full_unconditional`](../vendor/lsflow/environment/log_junction_tree.py) creates a fresh unseeded `default_rng()`. It initializes all nodes to zero, then uses all neighboring entries without tracking which have been sampled, despite its comment about already-sampled neighbors. This is not a verified exact sampler of the fitted joint model. The environment's seed alone does not establish paired hidden worlds or the intended return distribution.
- [`evaluate_detection_curve`](../vendor/lsflow/algos/dpmd_experiment_rf_disease_gnn.py) runs at most \(\lceil n/B\rceil+1\) rounds, which need not complete testing with a narrow frontier. It normalizes detections by positives found by that stopping time, not by all positives in `world_X`; a truncated trajectory can therefore end at a plotted fraction one. Training AUC also increments its test count by the nominal budget, even when the frontier admits fewer tests. These quantities need explicit interpretation before being used as preference objectives or recall measurements.

These are source-audit findings, not fixes applied to the pinned experiment. If a future benchmark repairs the sampler or evaluation protocol, that must be a declared shared benchmark revision.

For Intervention, conditional next-arm moves are independently sampled, but returns over time are correlated through the evolving states and actions. For Disease, the fixed latent world makes successive results dependent. In either setting, lower-tail risk concerns the distribution of the complete scalarized return \(w^\top G\), not the scalar mean \(w^\top\mathbb E G\). Separate marginal quantiles of reward components are insufficient to obtain its CVaR without their dependence structure. The minimum of two value critics is likewise not a defined return-risk measure.

## 7. Replay relabeling and the first conditional-policy task

For the proposed Intervention vector, save or recompute \(f=(f_L,f_H)\) once, use the scaled vector \(\widetilde f=40f\), sample \(w\), and relabel \(r_w\) from that same transition. This is valid because \(w\) changes utility, not physical dynamics. It does **not** mean a replay trajectory generated by one behavior policy is a rollout of every preference-conditioned policy. Off-policy targets must condition their continuation policy on the relabeled preference, for example

\[
y_w=w^\top\widetilde f(s,a,s')+\gamma\,\mathbb E_{a'\sim\pi(\cdot\mid s',w)}Q(s',a',w),
\]

with the declared component scaling and terminal handling. A vector critic evaluated under one fixed policy cannot simply be reweighted and called optimal for every preference. For a finite horizon, time belongs in the state or value convention. Keep \(w\) fixed within an episode initially; changing it mid-episode requires a declared preference-transition process and a corresponding continuation objective.

The first scientific question can therefore be narrow: **can one policy learn the existing two-cohort utility tradeoff, recover the original objective at equal weights, and generalize to held-out preferences under unchanged constraints?** A preference-conditioned spherical-flow baseline, a simpler conditional policy, and separately trained single-preference references are necessary. Conditioning is not a capability exclusive to diffusion. No new RL run, Pareto frontier, risk-sensitive controller, or diffusion advantage is established by this source audit.

The bounded check is reproducible with `.venv/bin/python analysis/preference_component_audit.py`. Its [JSON report](../reports/preference_component_audit/summary.json) records the exact conflict, decoder witnesses, replay decomposition, data availability, script/source hashes, and the unchanged-scientific-source check.
