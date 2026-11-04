# Preference audit: released Scheduling and Assignment

Audited 2026-11-04 against unchanged upstream commit `7e50d03e7129974a11061c9ce500bc2dc5593246`. Scope: source/paper audit and six inexpensive one-step MILPs, with no training or scientific-source changes.

**Recommendation within these two domains: use Dynamic Assignment for an explicitly declared utility–resource preference extension.** Its heterogeneous service costs already exist, can be measured from the returned action, and give a real conflict with engagement utility on the same state. Neither released benchmark is originally multi-objective. Scheduling is a useful later test of richer assignment preferences only if its hidden schedule is exposed; the current selected-patient action cannot express worker fairness or appointment comfort by itself.

## What the released benchmark actually optimizes

The LSFlow paper states a scalar discounted-return objective. Its Appendix J supplies scheduling availability and assignment capacity constraints, not separate safety/speed/comfort objectives. The README points to these as the Xu et al. coRMAB benchmarks. The earlier paper explicitly motivates four-state arms to make short-term and long-term rewards disagree; this temporal tradeoff is not automatically a multi-objective preference task. [LSFlow, Sections 2.1 and J.1](https://arxiv.org/html/2601.22211v2#S2.SS1), [coRMAB, Appendix B and C](https://arxiv.org/html/2503.01919#A2).

Exact local source pointers are listed with one-based lines; every path below is under `vendor/lsflow/`.

| Contract | Source and implication |
|---|---|
| Published launch examples | [`README.md`](../vendor/lsflow/README.md), lines 31–43: `scheduling` or `constrained`, `n_arms=40`, `budget=10`, `horizon=20`. |
| Actual environment dispatch | [`driver_lsflow.py`](../vendor/lsflow/driver_lsflow.py), lines 42–50: both use `rmab_type='multistate'`. The unused `n_actions` CLI argument does not determine these action spaces. |
| Instance construction | [`environment/rmab_instances.py`](../vendor/lsflow/environment/rmab_instances.py), lines 19–48: the wrapper receives the same underlying MultiStateRMAB; budget also sets the number of workers. Scheduling has five timeslots. Calling `get_scheduling/get_constrained` without the `multistate` override hits their `standard` default, which raises at lines 21–22. |
| State and horizon | [`environment/base_envs.py`](../vendor/lsflow/environment/base_envs.py), lines 46–71: the observation contains only arm states, reset draws Bernoulli states in {0,1}, and termination is at H steps. Later states can reach {2,3}. Remaining horizon is not included in observation. |
| Dynamics | [`environment/rmab_instances.py`](../vendor/lsflow/environment/rmab_instances.py), lines 70–81: passive upward probability is U[0,.2), active upward probability is U[.7,.9). [`environment/base_envs.py`](../vendor/lsflow/environment/base_envs.py), lines 144–174: each arm moves one level up/down, clipped to 0…3. Actions affect future states, so this is sequential RL, not repeated independent static optimization. |
| Reward | [`environment/base_envs.py`](../vendor/lsflow/environment/base_envs.py), lines 119–133 and 175–182: the reward is the sum of post-transition per-arm table rewards. First B arms have `[.2,.15,.1,R]`, R∈{4,5,6}; remaining arms have `[.1,.15,.2,R]`, R∈{1,2,3}. `calc_action_expected_value`, lines 194–207, provides the exact one-step expectation. |
| Wrapper transition/reward | [`environment/scheduling.py`](../vendor/lsflow/environment/scheduling.py), lines 105–127; [`environment/constrained.py`](../vendor/lsflow/environment/constrained.py), lines 45–84: wrappers delegate to the same base dynamics and reward. |

Availability, service costs and worker capacities are sampled at instance construction and remain fixed. There is no job-arrival process, deadline counter, travel speed, worker fatigue, accumulating worker load, or comfort attribute in these state transitions. Such quantities would be new simulator components, not latent existing objectives. The reward table is also deliberately nonmonotone at low states for the first B arms: immediately improving engagement can reduce the current scalar reward while enabling a large later reward. This is intentional sequential structure, not evidence of an implementation sign bug. Do not rename its arbitrary units “clinical safety.”

The solver's learned coefficients are distinct from real resource costs. Code maximizes a learned linear score; the paper often writes minimization of costs. These are opposite sign conventions, not by themselves a reward-sign defect.

## Scheduling: genuine feasibility, but assignment information is discarded

Source: [`environment/scheduling.py`](../vendor/lsflow/environment/scheduling.py), lines 58–84, and [`approximator/scheduling_approximator.py`](../vendor/lsflow/approximator/scheduling_approximator.py), lines 78–110.

There are 40 patient-selection bits `a_j`, 10 workers and five slots. Each patient is available in two slots; each worker in three. Binary `x_ijs` exists only on compatible slots. The MIP imposes:

\[
\sum_j a_j\le10,\quad \sum_jx_{ijs}\le1,\quad
\sum_{i,s}x_{ijs}\le2,\quad 2a_j\le\sum_{i,s}x_{ijs},\quad a_j\ge x_{ijs}.
\]

Thus every selected patient has exactly two assignment edges and an unselected patient has none. **The two edges need not involve distinct workers.** A single compatible worker at two different timeslots satisfies these equations. Conversely, the random-action code explicitly requires distinct workers (`environment/scheduling.py`, lines 147–164). The MIP is consistent with the displayed paper equations but weaker than an informal “two distinct workers” interpretation. Enforcing distinct workers would be a declared environment change.

`solve_from_coeffs` returns only `a`, not `x` (`approximator/scheduling_approximator.py`, lines 40–58); `step` then uses only `a` and does not validate the assignment constraints (`environment/scheduling.py`, lines 123–127). The rare numerical budget fallback in solver lines 51–56 selects global top scores without reconstructing a feasible schedule; it must not be treated as a proof of availability feasibility. No failing solver case was observed here; this is a source-level boundary.

Consequences for possible preferences:

- **Total number of worker-slot assignments is exactly `2*sum(a)`.** At fixed patient count it is constant, so utility versus “total work” cannot distinguish schedules serving the same number of patients. Across different counts it is a legitimate intervention-intensity preference, but not a rich scheduling-specific tradeoff.
- **Appointment lateness, worker balance and patient-specific slot comfort depend on `x`.** The current action/replay discards it. Two schedules with the same `a` are the same MDP action, even if an external metric would rate them differently. Returning `(a,x)`, or defining a canonical secondary optimization over x, changes the evaluated policy interface and must be declared. Arbitrary solver tie-breaking cannot be the preference mechanism.
- Availability violations are hard infeasibility, not a soft “safety” objective to trade away. A safety preference requires an independently modeled outcome or risk measure beyond these binary hard constraints.
- [`environment/bipartite.py`](../vendor/lsflow/environment/bipartite.py), lines 9–49, is a PyG data container (`x_action`, `x_arm`, edges), not another assignment environment or reward definition.

## Assignment: the cleanest source-grounded two-objective extension

Source: [`environment/constrained.py`](../vendor/lsflow/environment/constrained.py), lines 19–36, and [`approximator/constrained_approximator.py`](../vendor/lsflow/approximator/constrained_approximator.py), lines 47–86.

Each patient has a fixed service cost \(d_j\in\{2,3,4,5,6\}\), each worker a capacity \(b_i\in\{2,\ldots,7\}\). The feasible set is the projection of

\[
\sum_jd_jx_{ij}\le b_i,\quad a_j\le\sum_ix_{ij},\quad a_j\ge x_{ij},\quad (a,x)\text{ binary}.
\]

**There is no `sum(a) <= budget` constraint here.** B=10 sets the number of workers; a valid action may serve more than ten patients. `step` explicitly uses `allow_excess=True` (`environment/constrained.py`, lines 78–84). Adding a global cardinality limit would silently change this benchmark.

The original MIP permits duplicate service of one patient by multiple workers. It has no benefit in the objective or dynamics, which depend only on `a`; with positive costs, deleting all but one assignment preserves feasibility. Consequently, the set of feasible patient selections is unchanged if a witness uses `sum_i x_ij=a_j`. The source also creates an unused first copy of x before creating the constrained copy (approximator lines 63–70); this is redundant model size, not an extra feasible-action constraint.

For an unambiguous minimal extension, define

\[
\mathbf r(s,a,s')=\left(\underbrace{\sum_jR_j(s'_j)}_{\text{original engagement utility}},
\underbrace{-\sum_jd_ja_j}_{\text{resource use per distinct served patient}}\right),
\qquad r_\lambda= r_1+\lambda r_2.
\]

The second component is measurable from existing instance data and returned action. It counts necessary service resource once per patient; it is not the possibly duplicated hidden x load and not measured dollars. It can be added to logging without changing transitions or feasibility, but **optimizing it is a new preference benchmark extension**, not reproduction of the original single-objective scores. The no-action extreme is legitimate and must remain visible rather than being removed to make a prettier frontier.

A second possible extension is the pair of per-group returns for the existing first-B and remaining-arm groups. Their different reward profiles provide an allocation tension, but these are synthetic reward-table groups, not protected demographic categories. One should not call this fairness without defining the desired group criterion and normalization. No source-supported safety/speed/comfort triple exists here.

## Cheap same-state conflict witness, without training

Reproduce with `.venv/bin/python analysis/preference_assignment_audit.py` from the repository root. The [analysis script](../analysis/preference_assignment_audit.py) writes only [assignment_conflict.json](../reports/preference_benchmark_audit/assignment_conflict.json), preserving other reports in the same directory. The JSON contains the complete generated instance, source hashes, state, solver outcomes, concrete assignments, capacity checks and analytic-versus-upstream expected-reward checks. No Gurobi, Gymnasium or PyG installation is needed for this audit.

We executed the pinned class/function definitions through Python AST, omitting imports requiring Gym/PyG/Gurobi but preserving constructor bodies and their NumPy draw order. Start `np.random.seed(0); random.seed(0)`, call `get_constrained(20,40,10,rmab_type='multistate')`, then call the returned NumPy environment's first `reset()`. Torch wrappers receive the exact upstream parameters. No vendor file was changed. The resulting state is:

```text
[0,0,1,0,0,1,1,0,0,1,1,0,1,1,1,1,1,0,0,0,
 0,1,1,0,0,0,0,0,0,1,0,0,0,0,1,0,1,1,1,0]
```

Service costs and worker capacities are:

```text
d=[5,3,3,6,2,2,5,4,5,4,6,5,5,2,5,2,6,4,5,6,
   2,4,5,5,3,5,6,5,5,3,3,3,5,4,4,6,5,3,4,5]
b=[7,7,7,3,6,4,4,2,3,7]
```

At this state, expected original reward is exactly linear in a:

\[
\mathbb E[r\mid s,a]=r_0(s)+\sum_jg_j(s)a_j,\qquad
g_j=(p_{j,1}-p_{j,0})[R_j(s_j^+)-R_j(s_j^-)],
\]

with \(r_0=5.205763995664590\). We solved `max g@a - lambda*d@a` using SciPy/HiGHS MILP, 40 selection variables plus 400 assignment variables, per-worker capacities, and `sum_i x_ij=a_j`. The equality is equivalent for projected a as proved above. All six solves returned success; every rounded solution was checked for binary selection, per-worker capacity, and assignment/selection consistency. This is a small static oracle audit, not the official Gurobi execution path or a learned-policy result.

| Cost coefficient λ | Selected patients | Expected original reward | Resource cost |
|---:|---:|---:|---:|
| 0 | 14 | 6.043373659544242 | 50 |
| .01 | 13 | 6.034799614316506 | 47 |
| .02 | 4 | 5.489013215854597 | 10 |
| .05 | 0 | 5.205763995664590 | 0 |
| .1 | 0 | 5.205763995664590 | 0 |
| .2 | 0 | 5.205763995664590 | 0 |

Concrete actions, with zero-based patient indices:

```text
λ=0:   [10,12,13,14,15,20,21,24,29,31,33,34,37,38]
λ=.01: [12,13,14,15,20,21,22,24,29,34,36,37,38]
λ=.02: [13,15,29,37]
```

For λ=0, an explicit worker-by-worker witness is `[[12,20],[24,34],[14,15],[37],[10],[21],[38],[13],[29],[31,33]]`; its loads are `[7,7,7,3,6,4,4,2,3,7]`. At λ=.02, workers 4,7,8 can respectively serve `[13,29]`, `[15]`, `[37]`, with loads 5,2,3 within their capacities 6,2,3.

This establishes that the two proposed objectives can conflict for feasible actions of the **same actual reset state**. It does not establish the long-horizon Pareto frontier, a suitable training preference scale, diffusion controllability, or benefit over FM. The reward table changes marginal utility dramatically near state 3; these one-step λ values must not be copied blindly into a long-term preference sweep.

## Targeted algorithm implication and a falsifiable next question

The actual LSFlow implementation has a single scalar reward buffer (`algos/replay_buffer.py`, lines 27–33,55), a scalar Q head (`algos/dpmd_rf.py`, lines 27–37), and an actor conditioned on observation/cost/time without preference (`models/rfm/policy.py`, lines 73–89). This is the relevant limitation for this user request: it learns one utility policy, not an adjustable family of utility–resource distributions. Merely adding λ to the actor/critic is available to spherical FM too and is a necessary baseline, not a diffusion-specific contribution.

An initial research contract should keep original transitions/feasibility, record the two reward components separately, fix λ for an episode, and require a preference-conditioned value/continuation model. New λ values affect optimal future actions, so vector values for one fixed future policy cannot automatically be treated as optimal values for every λ. Changing preference partway through an episode is a separate task and requires specifying what future preferences the critic assumes.

The diffusion-specific hypothesis worth isolating is whether **noise-scale-consistent guidance during multiple denoising stages** can recover the requested return–resource tradeoff more reliably than terminal reranking or directly conditioned generation at equal cost. It must follow the actual denoising continuation, not score a noisy coefficient as if it were an executed action. Use preference-conditioned spherical FM, equally budgeted endpoint reranking, and a stochastic-flow counterpart as controls. If plain conditioning already achieves the same frontier and calibration, multi-step guidance has not added value.

Before RL training, test a few frozen-state preference sweeps for feasibility, response to λ, objective conflicts and query cost. Later RL evaluation should report both raw components over complete episodes, scalarized regret on unseen λ values, seeds/instances, coverage of the reachable tradeoff set, and NFE/Q/solver budgets. Separate objective-scale calibration from performance tuning. No new training or core-code modification is proposed by this audit alone.

## Reproduction and dependency boundary

The shipped official solver classes import Gurobi; the local audit environment does not have Gurobi, Gymnasium or PyG installed. It does have SciPy, and the witness above uses its HiGHS MILP backend only. This does not certify licensed Gurobi availability on a future host. A source-equivalent open-source solver port is feasible but would require its own action-feasibility and tie-behavior parity audit.

The actual CLI defaults are N20/B2/H10, train_updates=2000 **episodes**, 1000 warmup, learning rate 4e-4 and flow_steps=36 (`driver_lsflow.py`, lines 81–132; `algos/dpmd_experiment_rf.py`, lines 90–105). The README dimensions override N/B/H. At H20, 2000 episodes imply 40,000 training interactions, not 2000 updates. Paper and released implementation differ in some architecture/hyperparameter details; the existing [upstream audit](UPSTREAM_AUDIT.md) records those boundaries. None of the completed Intervention runs constitutes a Scheduling/Assignment experiment.
