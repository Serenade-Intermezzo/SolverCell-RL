# Interpretation of the first preference implementation pilot

Both final server runs passed the [artifact audit](RESULTS.md). This establishes a complete, reproducible implementation at the declared short budget. It does not establish an effective diffusion-specific guidance mechanism.

## The conditional policy has weak and asymmetric preference response

Changing the low-response weight from .125 to .875 yields these changes in raw component returns, using the same eight final episode seeds at each endpoint:

| Model / inference | Low-response return change | High-response return change |
|---|---:|---:|
| Sphere / direct | +2.3375 | −31.8625 |
| Sphere / terminal weighting | +5.5000 | −63.2250 |
| Diffusion / direct | +0.0125 | −6.3750 |
| Diffusion / terminal weighting | +1.5625 | −70.0375 |
| Diffusion / SMC | +1.5625 | −70.0375 |

Diffusion's direct low-response returns are123.9500,123.9125,123.9500,123.9625 over the four unseen weights. That component is essentially flat. In the paired endpoint comparison it improves in only1/8 episodes; terminal weighting/SMC improve it in3/8. The response mainly reduces high-cohort utility and does not demonstrate strong capability to increase the requested low-cohort utility. A changing scalarized score alone would not establish preference control because the scalarization weights themselves change.

## Default SMC is inactive as a sequential selection mechanism

All19 SMC evaluation rows, comprising1840 decisions across validation and heldout, have zero intermediate resampling events. On heldout, terminal weighting and SMC have exactly identical two-component and scalarized returns in all32 paired episodes. This matches the tested telescoping identity when no resampling occurs. It is not evidence that the learned continuation potential improves decisions.

On the640 heldout decisions, each weighted mode uses40960 actor NFE rows,11520 twin-Q rows and5760 decoder rows. SMC additionally uses40960 guide forward rows. Measured total heldout evaluation time is1.3259s for terminal weighting versus2.1446s for SMC (about1.617×), on the same CPU worker; this is a prototype measurement rather than an intrinsic runtime ratio. The extra guide currently has no measured utility benefit. Terminal-weighting gains over direct also use eight times as many actor trajectories and terminal value selection; they cannot be attributed to the learned guide.

## Preserve the split and comparator boundaries

Direct validation AUC is131.702 for sphere and139.749 for diffusion, while final unseen-weight direct averages are157.785 and137.240 respectively. Final terminal-weighted unseen averages are175.026 and168.013. The ordering depends on the split/metric, so there is no basis for choosing a favorable comparison or declaring diffusion superiority. These numbers average scalarized scores over the prespecified weight grid, not independent training seeds.

The known-model primitive greedy reference has much higher validation utility, particularly for the high-response objective. It uses model information and a broader primitive-action support than the learned native-decoder policies. It indicates substantial remaining learning headroom and is not a matched algorithm comparator.

Both actors have30408 parameters and use8 NFE per base trajectory. Diffusion has an additional29505-parameter guide trained250 times. Single-trajectory NFE matching does not cover guide fitting, backward passes or all experiment work. See the machine-readable [counter scopes and all evaluation rows](study.json).

The short8-step diffusion prior is approximate, suffix log-mean-exp targets use only4 suffixes during training, and the guide follows a moving target actor/critic. Those are explicit limitations of this first prototype. Stronger research claims require repeatable improvement over terminal weighting, informative intermediate selection and preference response that benefits the requested component across multiple seeds/instances.

## Follow-up mechanism diagnostic

The [fixed-checkpoint probe](../preference_guide_probe_20261104/RESULTS.md) is post-hoc and uses validation preferences only, with a new episode-seed range. It examines resampling sensitivity and noisy-prefix calibration without modifying this pilot's configuration, weights, source or heldout results. It is not an additional training arm and does not select an improved heldout model.

It completed192 full-horizon evaluation episodes and18 calibration groups. ESS thresholds0/.5 still exactly match terminal weighting. Threshold1 activates all possible intermediate resampling, but paired mean utility differences at weights0/.5/1 are+3.400,−0.172,+3.069, with episode-mean standard errors24.472,9.987,3.683 respectively. This small single-checkpoint diagnostic does not establish benefit or justify a threshold recommendation.

The calibration reference uses the same fixed learned critics and actual actor suffix kernel, with128 suffixes for each of16 prefixes within a fixed observation/preference/index. It is not a true-return oracle. It shows that continuation targets are not universally flat and that guide quality is uneven. At replay row0, weight1,index7, learned-vs-reference correlation is−.681 while independent64-suffix halves correlate.991; at row1255, weight0,index4 the corresponding correlations are+.931 and+.973. The former failure cannot plausibly be explained solely by a flat target or large suffix Monte Carlo noise in that group.

The evidence therefore points to within-context guide discrimination and calibration as a concrete next implementation target. Merely increasing resampling frequency does not repair an incorrectly ranked prefix potential. A future grouped or contrastive guide objective could emphasize relative prefix weights within the same observation/preference/noise level; that change has not been implemented or tested in this pilot.
