# Preference tuning v2: prespecified mechanism screen

## Evidence and hypotheses

Parent commits are implementation `f304a2f`, pilot `6a3a4f2`, and fixed-checkpoint guide audit `106b2f3`. Exact parent JSONs are `experiments/preference_pilot_20261104/{sphere,diffusion}.json`; preparation records their byte hashes. The old pilot has a nearly flat diffusion low-cohort response, no SMC resampling and uneven within-context prefix ranking. It has1000 updates, weakly informative candidate weights and an8-step Gaussian terminal forward alpha-bar around.38. Those observations motivate this bounded screen, not a claim that any fix will help.

H1: decreasing actor candidate temperature from.5 to.1 improves validation utility and preference response after4000 updates. H2: increasing diffusion beta-end from.2 to.6 reduces its forward-terminal prior mismatch and improves validation utility at unchanged8 actor NFE. Neither change guarantees sampler accuracy: coarse reverse transitions and learned-noise error remain.

H3: fitting relative log potentials among prefixes at the same observation/preference/index improves guide calibration. SMC normalized weights are invariant to a common log-potential offset at each such context, so centered MSE removes a nuisance gauge while retaining log-weight scale. The exact terminal correction remains unchanged. This is a proposed training modification, not an established novelty claim.

## Screen, controls and budget

Eight seed0/instance0 arms:

| Arm family | Actor temperature | Diffusion beta-end | Guide |
|---|---|---|---|
| Sphere (2) | .5,.1 | .2 (unused by sphere sampler) | none |
| Diffusion factorial (4) | .5,.1 | .2,.6 | original absolute MSE |
| Grouped control (1) | .1 | .6 | grouped absolute MSE |
| Relative guide (1) | .1 | .6 | grouped centered MSE |

All use256 warmup+4000 updates, batch64, width128, lr.0004, reward scale.025, discount.99, EMA.005,8 candidates,8 actor NFE. Validation is every500 updates,8 episodes at weights0/.5/1 and seeds100000–100007. Five training preference weights are unchanged. Final unseen-preference evaluation uses8 fresh episodes at seeds600000–600007; it does not select settings. Previous pilot finals used200000; the earlier guide probe used400000.

Both grouped losses use exactly the same16 prefixes, grouped into4 independent contexts×4 prefixes, shared per-group path index,4 fresh suffixes per prefix. Original absolute MSE instead uses16 contexts. Thus original vs grouped absolute isolates grouping; grouped absolute vs centered isolates the loss. Their complete base actor/critics, optimizers, behavior trajectories and RNGs must be bitwise identical: guide fitting must not alter the base algorithm. Label path and suffix row budgets are matched in expectation, while realized suffix NFE and wall time are recorded; fewer distinct contexts are a grouping tradeoff.

Two single-thread CPU lanes on24/25 execute all independent arms, handing a lane its next queued arm immediately after exit. There are no automatic retries, overwritten outputs or changes to live sources. Failed/degraded arms remain visible. This is bounded CPU work, with no GPU launch.

## Metrics and selection before reading final results

Primary: mean direct scalarized validation utility over the final two checkpoints (3500/4000), averaged over the three fixed validation preferences. Secondary: normalized full-curve AUC, per-preference utility, endpoint cohort shifts, feasibility, complete trajectories, candidate-weight ESS, true prefix calibration against fresh suffixes, inference counters and wall cost. Scalarization weight changes alone are not controllability evidence.

Select each base policy independently so sphere remains a strong tuned comparator. Sphere candidates are its two temperatures. Diffusion candidates are the four original-absolute-guide factorial arms; the two guide arms are not additional independent base candidates. Rank by primary metric, then AUC, then original configuration preference. Promote a nonreference setting only if primary improves at least2% over the4000-update reference (.5 temperature, diffusion beta-end.2) and AUC is no worse than98% of reference; otherwise retain reference. Freeze selection JSON before reporting heldout values. This is discovery selection, not replication or significance evidence.

**Completion note, added after the screen:** the selector ranks candidates eligible under that gain/AUC guard; the reference always remains eligible. No final-result fields are used. The actual validation selection was frozen at commit`cbc1cfe` before the new-seed workers launched. The original decision rule above is unchanged.

Then run the frozen selected sphere and diffusion settings at new training seeds1/2 (four runs), same budget and instance. Report discovery seed0 separately, full curves, seed-wise differences and uncertainty limits. Do not tune further from replication/heldout values within this task.

## Guide mechanism checks and decision gates

Compare the fixed beta-end.6/temperature.1 guide triplet with the same trained base. Test actual-suffix calibration within fixed state/preference/noise groups; do not pool cross-state correlations. Use a finite Monte Carlo reference with split-half consistency, not a true-environment-value oracle. Report all cases, including negative correlations.

Deployment ESS sensitivity .5/.9 may be measured on validation preferences and fresh seeds500000–500015; terminal weighting uses the same8 particles and terminal Q queries. The guide temperature remains.5, matching guide labels. No temperature transfer or independently summed component guide is introduced. All SMC modes retain initial weights and exact terminal utility correction. Additional guide cost is explicit.

A useful guide must improve within-context relative calibration AND show a reproducible utility improvement over terminal weighting that is not explained solely by extra compute. If it does not, record a negative mechanism result and do not launch a larger guide sweep. A stronger target signal or more resampling alone does not establish a useful guide.

## Sanity gates and scope

Before launch: legacy default kernel equivalence, valid schedules and terminal alpha-bars, centered-loss gauge invariance/gradient scale, shared group conditions, grouped/centered base-state equivalence, full suite, source-frozen debug queue, exact restore and artifact audit. All evaluation episodes finish H20; fixed-step training retains any partial episode.

Only one benchmark instance and short training budgets are used. New seeds test training variability, not cross-instance generalization. Guide fitting, suffix Monte Carlo, finite-particle SMC, approximate critics and coarse diffusion remain approximations. Assignment is outside this tuning task. Upload the code/configs/reports to the user-confirmed private `Serenade-Intermezzo/SolverCell-RL` repository, once its URL is supplied; checkpoints remain local/server artifacts with hashes.

**Publication note, after experiments:** the user subsequently supplied the repository URL and explicitly authorized keeping it public. Source, configs, reports and history are now uploaded to [Serenade-Intermezzo/SolverCell-RL](https://github.com/Serenade-Intermezzo/SolverCell-RL), branch`main`. This overrides only the original visibility preference; the scientific design and checkpoint-storage boundary above remain unchanged.
