# LSFlow benchmark audit and adaptation boundary

Audited on 2026-11-04. Upstream: [anagha-satish/combinatorial-diffusion](https://github.com/anagha-satish/combinatorial-diffusion), pinned commit `7e50d03e7129974a11061c9ce500bc2dc5593246`, MIT license. The vendored tree is preserved unchanged. Paper: [Latent Spherical Flow Policy for Reinforcement Learning with Combinatorial Actions, v2](https://arxiv.org/html/2601.22211v2).

## Selected first task

Dynamic Intervention, upstream `--rmab_type sigmoid`, with **40 population arms, 20 primitive interventions, budget 10, horizon 20**. The population has four engagement states and nonlinear intervention synergies. At each decision an action is a binary length-20 vector with sum at most 10. The initial instance seed is 0 and is distinct from every rollout/training seed.

This is an explicitly configured instance of the released benchmark, **not a claim of reproducing the paper's reported number**. The paper gives population 40, budget 10 and horizon 20, but does not specify the required primitive intervention count. The README accidentally gives `routing` for Dynamic Intervention; the actual dispatcher uses `sigmoid`. The CLI default primitive count 3 is incompatible with budget 10. Setting 40 primitives also fails in the released weight constructor because too few untouched population arms remain for a group of three. We retain this failure rather than change the environment's scientific semantics. The supported 40/20/10 configuration passes parameter parity against the source for instance seeds 0, 1 and 7.

## What the port preserves

Relevant sources:

- `environment/rmab_instances.py`: `get_rmab_sigmoid` and `multistate_rmab`.
- `environment/base_envs.py`: `BaseRMAB.fresh_reset` and `MultiStateRMAB.__init__`.
- `environment/multi_action.py`: initialization, `get_sigmoid_weights`, `link_function`, `step`, `calc_action_expected_value`.

`solvercell/environment.py` retains the exact instance construction draw order with a private `numpy.random.RandomState`. Even the unused base-transition and initial-state draws are retained. The generated instance records reward tables, weights, sigmoid parameters, the otherwise unused base transitions and wrapper initial state. Saved JSON instances can be loaded directly so all methods see identical dynamics. Arrays held by the environment are read-only.

Each reset samples independent Bernoulli(0.5) states in {0,1}. Later states can reach {0,1,2,3}. Given action `a`, upward probability for population arm `j` is

`0.98 * (a_j * sigmoid(b_j * (theta_j + W_j @ a) - x0_j) - a_j * sigmoid(-x0_j)) + 0.01`.

Here `a_j` in the scalar sigmoid expression denotes the saved response-amplitude parameter, not a binary action entry. The state moves up or down by one and is clipped to [0,3]. Reward is the sum of the **post-transition** state rewards. At 20 steps the episode terminates. Float32 per-arm reward accumulation matches upstream.

Intentional engineering changes: rollout randomness uses an instance-local `numpy.random.Generator`; training and evaluation use separate environments/seeds. Consequently, equal numerical rollout seeds do not promise the same pseudorandom trajectory as upstream's global legacy RNG. Transition probabilities and rewards are unchanged. The tests temporarily inject the same legacy draws to check exact state/reward parity over a complete 20-step trajectory. Plotting, Torch wrappers, solver imports, and incidental graph construction are omitted. The port validates binary actions and refuses steps after termination.

## Decoder and baseline boundary

The original `algos/dpmd_experiment_rf.py` sets latent dimension to population count, **40**. `approximator/multi_action_rmab_approximator.py:solve_from_coeffs` maps these arm costs to primitive scores with `W.T @ c`, then maximizes the linear score under `sum(a) <= B`. It invokes Gurobi for this cardinality problem. For primitive costs, an exact replacement simply selects up to B strictly positive scores; no commercial solver is necessary.

The primary controlled comparison uses **the same primitive-cost positive-top-k decoder and dimension 20 for every compared policy**, including the LSFlow-style adapted reference. This removes a decoder confound between cell projection and fixed-endpoint training. It is a declared adaptation of the original LSFlow policy, not an unmodified upstream reproduction. The separate `lsflow_native` arm invokes the pinned original learner and arm-cost `W.T @ c` decoder. Upstream W contains repeated columns (for the default instance, 8 unique column patterns and rank 8), so ties are structural; an explicit deterministic tie rule is needed, and it need not match Gurobi's tie choice.

`InterventionEnv.random_action` uniformly selects exactly B interventions, matching upstream's random feasible policy. `greedy_action` is a **model-informed myopic marginal-gain heuristic**, stopping when no positive immediate improvement remains. It is not an exact nonlinear optimizer and differs from the paper's empirical five-step-warmup intervention heuristic. It must be reported under its own name.

## Paper settings versus released defaults

| Item | Paper benchmark description | Current released code |
|---|---|---|
| Population / budget / horizon | 40 / 10 / 20 | CLI defaults 20 / 2 / 10; README overrides to 40 / 10 / 20 |
| Primitive intervention count | Not specified | `--n_actions 3`, incompatible with B=10 |
| Training | 2000 episodes, H=20; warmup 1000 | Same episode interpretation; 40,000 training environment steps plus 1000 warmup; one critic update per training step |
| Batch | 64 | 64 |
| Learning rate | 1e-3 | 4e-4 |
| Gamma / target rate / delay setting | 0.99 / 0.005 / 2 | Gamma and target rate match; code updates actor each step and applies delay to target updates |
| Policy particles | 12 | 12 |
| Temperature schedule | 2.0 to 0.8 over 10,000 | Same named settings; code also has learned-alpha machinery |
| Weight clip / Q clip / EMA beta | 4 / 3 / 0.05 | Same named defaults |
| Smoothing | kappa=28, J=1 | kappa_exec=kappa_smooth=28; M_smooth=16; Gaussian tangent approximation to vMF |
| Policy hidden widths | (32,32) | `models/rfm/policy.py:RFMPolicy` defaults (256,256); critic also (256,256) |
| Integration | Table lists flow_steps=36 and Heun=30 | Heun method; passed `flow_steps=36` overrides method default 30 |
| Evaluation episodes | 50 | CLI default 5 |
| Reported metric | Described as average return | Driver prints average of all per-step rewards; save both episode return and return/H to avoid unit ambiguity |

The source additionally runs a small critic-only myopic pretraining stage after warmup. Official training only saves a loss plot and performs final evaluation. Our harness adds full evaluation curves, per-episode returns, checkpoints, configurations, source hashes and explicit run status; these artifact improvements do not establish algorithmic gains.

## Other benchmarks and dependency findings

- Scheduling: real binary assignment constraints enforce time compatibility and two assignments per selected patient. `approximator/scheduling_approximator.py` supplies a genuine Gurobi model. An open-source MILP port is plausible for a later test of more complex decision cells.
- Assignment: `approximator/constrained_approximator.py` supplies capacity-constrained assignments and a genuine Gurobi model. The source redundantly creates an unused first set of assignment variables; retain source provenance if adapting it.
- Routing: the current `solve_from_coeffs` simply selects top-B scores and optionally calls masks/projections that the published `RoutingRMAB` does not provide. Its `step` checks only the cardinality budget. This driver path does not establish route feasibility and is not selected for the first study.
- All official solver modules import Gurobi 12.0.3. The requirements also install Torch 2.8, TorchRL, PyG and a separate JAX stack. The selected NumPy benchmark and exact cardinality decoder do not require those heavy dependencies or a Gurobi license; policy training still requires PyTorch.

## Validation and reporting contract

`tests/test_environment.py` executes the pinned NumPy class definitions as a parity oracle, replacing only unused visualization/Torch wrappers. It checks source-generated instance equality for seeds 0,1,7, complete common-draw trajectory parity, exact one-step expected reward, independent evaluation randomness, JSON instance roundtrip, action feasibility and termination. Parameter equality is stronger than merely matching nominal dimensions.

The initial campaign is a seed-0 tuning screen on one fixed instance. A complete run requires its configured environment/update budget, finite evaluation curves, final evaluation and durable completion artifacts. It is not evidence of multi-seed stability, benchmark-wide superiority or novelty.
