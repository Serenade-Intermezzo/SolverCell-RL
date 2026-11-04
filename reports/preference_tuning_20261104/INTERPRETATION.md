# Tuning decision and remaining limitations

## What improved

At the same4000-update discovery budget, lowering actor candidate temperature from.5 to.1 improves diffusion's final-window direct validation from203.018 to297.661 (+46.62%), with AUC173.936→247.723. Sphere improves290.735→359.450 (+23.63%), with AUC210.392→297.633. Fresh unseen-weight direct averages improve193.189→283.128 for diffusion and269.084→343.984 for sphere; those final values were read only after validation selection was frozen.

These comparisons identify candidate weighting as a useful optimization axis in this harness. The tuning-effect comparison itself is seed0 discovery evidence; seeds1/2 repeat the chosen recipes, not each original-temperature reference. Do not claim that the temperature effect size is replicated across seeds.

The selected recipes keep temperature.1, eight actor NFE and the original absolute guide objective. Diffusion retains beta-end.2. Increasing it to.6 reduces cumulative terminal alpha from.386792 to.037108 but worsens validation utility at both temperatures. Better forward-prior moment matching alone does not guarantee a better coarse learned reverse sampler. The degraded arms remain in the table and full artifact audit.

## What did not improve

Tuned spherical flow remains ahead of tuned diffusion at all three training seeds. Discovery last-window difference (diffusion minus sphere) is−61.789; new-seed differences are−69.473 and−65.725. Their mean is−67.599, sample SD2.650. Full-curve AUC and fresh unseen direct utility also favor sphere on both new seeds. This is a stable direction on this instance at this budget, not a cross-instance or significance claim.

Improved scalarized return does not establish stronger preference control. Across fresh final weights.125→.875, the two-new-seed mean low-cohort return changes approximately119.57→122.08 for sphere and120.02→121.49 for diffusion. Much larger changes occur in high-cohort returns. A changing scalarization coefficient alone is not a policy response metric; the component curves remain necessary.

## What the guide control established

All three guide arms at beta-end.6/temperature.1 have bitwise-equal base actor/critics, optimizers, replay, training environment/RNG, and direct validation trajectories. The two grouped objectives see the same grouped inputs and draw budgets. Their18 calibration reference groups use exactly identical fixed-critic suffix draws. This isolates the guide mechanism from changed base learning.

Relative centered MSE reduces median centered prefix error to.03033, compared with.05787 for grouped absolute MSE and.04307 for original ungrouped absolute MSE. Median error relative to a constant within-context predictor is.993, versus1.611 and1.333. Thus the relative objective improves calibration in this small fixture but is only roughly at a constant predictor on that median metric. Group-wise correlations and split-half consistency vary; two replay states do not establish general calibration.

At ESS fractions.5 and.9, all three guides still make zero intermediate resampling decisions on the declared validation probe. Every SMC rollout therefore equals terminal weighting, with extra guide calls. The mechanism gate fails: there is no measured downstream selection or utility benefit. No guide default is changed and no larger guide sweep is launched. Making a guide fit metric smaller must not be reported as a diffusion-specific policy improvement.

## Publication and reproducibility boundary

The completed tuning study comprises eight discovery arms and four fixed new-seed arms, all exit0 with complete expected updates/curves/trajectories and finite checkpoints. Every mirrored checkpoint byte hash matches the server. All90 implementation tests pass. The server source remains frozen at`33fcff51897f7485ddce4a303d089526d0cada8e165a4c380e7c9509db6680b3`; original pilot/historical sources are unchanged. Training used two CPU lanes, no GPU workers, and all experiment workers have exited.

Upload source, exact configs, selection lock, reports and reproducibility scripts to the user-confirmed private GitHub repository. Raw checkpoints remain local/server artifacts with recorded hashes. Do not label a local commit or prepared bundle an uploaded repository; verify the actual remote ref after push. The user chose manual private-repository creation and will supply its URL.
