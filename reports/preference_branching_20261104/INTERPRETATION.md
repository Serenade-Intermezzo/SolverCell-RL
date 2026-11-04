# What the completed branching experiment supports

The previous continuation guide had no downstream selection effect. This revision reuses the actual diffusion chain: sample two independent roots, run six of eight transitions, generate four fresh two-transition suffixes per root, and score all eight native-decoded leaves with the existing preference-conditioned twin critic. No guide, actor/critic training, artificial sampler noise or Bellman change was needed. The fixed spherical Heun policy is deterministic given a prefix, so duplicating its suffix would not create fresh candidates; independent sphere sampling remains the strong control.

This is a clean inference mechanism and a modest engineering improvement. Shared-prefix nested Monte Carlo is not itself a novelty claim. The results do not establish diffusion superiority or a publishable standalone algorithm.

## Locked evidence and completion

Implementation/preparation commit`d21a377` froze the source, six parent hashes, six topologies, seven independent controls, splits and selection gates. Validation-only selection was committed at`0ca1c9f` before final workers launched. All96 parent source files still match their recorded hashes. Only the new sampler module was added to the scientific source map. All six checkpoints and evaluated model/optimizer/RNG/update states remain unchanged.

All four formal workers exited0: one screen and three final evaluations, totaling7680 complete H20 episodes (153600 decisions), feasible-action rate1. No training updates or GPU workers were launched. Actual workers were`llmagent` on CPUs24/25; the controller handed the freed lane to seed2. The [remote audit](runtime/final_remote_state.json) records zero live workers/queued jobs and unchanged parent hashes. All98 tests passed. An initial engineering preflight failed on a sphere model's absent guide; its log is retained. The corrected [full preflight](../preference_branching_preflight_20261104/RESULTS.md) passed and is engineering-only: it did not select the formal topology.

## Result and cost boundary

Validation selected`tree_m2_k6` by the prespecified lowest-actor-NFE rule: utility99.16% of independent8, worst preference98.45%, randomized median inference latency97.27%. One alternative (`tree_m4_k4`) failed the utility gate. Every screened setting and control is retained.

Final values average four held-out preference weights, each with32 complete episodes. Seeds refer to existing trained models, not new training replications.

| Fixed model seed | Diffusion direct | Independent3 (24 actor rows) | Independent4 (32 rows) | Independent8 (64 rows) | Shared prefix (28 rows) | Sphere direct (8 rows) |
|---:|---:|---:|---:|---:|---:|---:|
|0|274.469|303.452|302.474|306.162|310.148|353.549|
|1|269.571|293.941|294.327|294.491|303.391|355.940|
|2|279.968|304.901|306.512|310.221|314.321|359.972|

Shared-prefix minus independent8 deltas are+3.986/+8.901/+4.100; descriptive mean+5.662, sample SD across models2.805. Episode-paired SEs are5.804/5.948/5.031, averaging each episode over preferences. These uncertainties and the small model/instance scope preclude general gain, significance or broad noninferiority claims. Seed0 selected the topology. Seeds1/2 did not select it but were known from v2. One seed1 preference decreases1.359 (ratio.9908); all per-preference vectors remain available.

Actor forward rows, not sequential network invocations, fall64→28 (−56.25%). Both still execute eight transition stages and score eight leaves, using18 twin-Q rows and9 decoder rows/action. Small CPU batches therefore give only a small latency difference: median1.853ms→1.803ms in the randomized screen (192 samples/mode across two states/three preferences). There is no latency significance claim or extrapolation to other devices. Final environment-inclusive wall ratios are.982/.969/.981. Independent3/4 have fewer Q/decoder rows: the table brackets actor budget only, not total compute. Direct sphere remains stronger using only8 actor rows and no scoring queries. The default learner is unchanged; branching is optional.

## Mechanism and remaining weaknesses

Across six fixed state/preference cases, sibling action agreement rises from3.50% at index2 to10.73% at index6; late suffixes still yield diversity. With32 prefixes×32 suffixes per case, median predicted matched-actor-budget variance ratios for four children at index6 are.699 for the normalizer, .548 for action-probability influence trace, and.702 for utility influence. These are finite-MC local estimates with corrected/clipped between-prefix variances, not observed episode-return variances or a population theorem. Weight ESS ignores sibling dependence.

Support is not universally collapsed: diffusion produces51–86 unique native actions from128 base candidates in the six cases, versus93–99 from a native Gaussian prior. Best sampled one-step rewards are similar for base/prior candidates. More raw diversity alone is not a demonstrated solution. Selected-action MC continuations sometimes favor critic selection, but improvements are often small relative to eight-rollout uncertainty; this is not unbiased population critic-ranking calibration. Long-horizon critic offsets occur in both methods. These fixtures do not isolate actor learning, value learning and coverage as the cause of the sphere–diffusion gap.

The next scientific question is whether conditional suffix allocation based on decision-relevant uncertainty helps beyond fixed branching and ordinary candidate weighting. That requires a separate prespecified study with independent pilot/suffix randomness, correct allocation weights, uncertainty-free controls and total-cost accounting. It was not implemented or tuned from final results in this campaign. More independent roots, instances and fresh training replications remain necessary before a research contribution claim.

## Reproducibility

[Design/gates](../../docs/PREFERENCE_BRANCHING_V3.md), [manifest](../../experiments/preference_branching_20261104/manifest.json), [selection](selection.json), [audited results](RESULTS.md), [full episodes/support/variance](study.json), [tests](tests.log). Scientific source`05c3cfffb6e7ed1bad6a077085e7da70bfaef41ad88323108052a261740fcd10`; unchanged parent source`33fcff51897f7485ddce4a303d089526d0cada8e165a4c380e7c9509db6680b3`. Raw checkpoints/journals remain local/server artifacts outside Git. Source, configs, reports and audits are published to the user-authorized public repository.
