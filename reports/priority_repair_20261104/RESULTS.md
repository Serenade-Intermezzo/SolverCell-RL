# Priority-ordered repair: completed bounded campaign

104 tests passed; repaired online harness debug and all three fixed-target preflights passed. All nine formal actor-fitting jobs completed2000 optimizer steps. Scientific source: `eec36f7e3d63572e34f709e4343ceb0c67f3b216c8c6ba29c9dc3a7a85fdf022`. Source/data/config/runner, actual CPU worker identity, counters, finite checkpoints and common endpoint-draw RNGs passed audit.

## Task controllability and calibrated objectives

Native open-cell action catalog:268, checked against10000 random proposals. Opposing feasible reference policies gain87.072/563.566 raw units. Scales frozen from32 calibration episodes per extreme: [87.07187530398369, 563.5656227767467]. Validation uses separate seeds. These are model-informed references, not exact optimal experts.

Raw rewards/replay are preserved. New v4 scalarization is requested preference dot(raw component / fixed scale), and both critic TD and evaluation use it. It is a declared objective extension; new normalized scores are not comparable to old raw-objective scores.

## Common fixed-endpoint results

| Seed | Sphere TV | Epsilon clean TV | Epsilon noisy TV | x0 clean TV | x0 noisy TV |
|---:|---:|---:|---:|---:|---:|
|0|0.2556|0.2780|0.2749|0.3058|0.2957|
|1|0.2523|0.2714|0.2712|0.2809|0.2742|
|2|0.2497|0.2817|0.2784|0.2872|0.2806|

Mean target-only sampling TV is about.072. Each case has8192 samples; target probabilities are exact finite-catalog values. Final contexts32 and preferences4 were not used for optimizer updates or checkpoints. All seed-level values and degraded x0/noise controls remain. TV is measured against a fixed soft endpoint target; negative target-minus-model utility gap can coexist with poor distribution fit and is not evidence of optimal RL.

## Staged-guidance decision

{
  "task_gate_passed": true,
  "actor_fit_gates": {
    "epsilon": {
      "mean_heldout_TV": 0.27704092296122934,
      "sphere_mean_TV": 0.25252143953275663,
      "difference_vs_sphere": 0.02451948342847271,
      "passed": false
    },
    "x0": {
      "mean_heldout_TV": 0.29128633584544633,
      "sphere_mean_TV": 0.25252143953275663,
      "difference_vs_sphere": 0.0387648963126897,
      "passed": false
    }
  },
  "staged_guidance_permitted": false,
  "scope": "Prespecified fixed-distribution fit gate, not a statistical superiority claim; no heldout tuning.",
  "action": "do not launch staged guidance; unresolved finite-step actor fitting remains the next research node"
}

No production default was silently changed: legacy configs retain historical noisy epsilon/unit-scale semantics. The explicit v4 factory declares clean terminal sampling and frozen calibration. Old guide checkpoints are rejected when chain/objective semantics differ. Positive-variance Gaussian log-density calls reject deterministic terminal kernels; actual suffix and SMC labels follow the declared chain.

[Design](../../docs/PRIORITY_REPAIR_V4.md) · [Decision](decision.json) · [All context metrics and audits](study.json) · [Exact histograms](histograms.npz) · [Figure](repair_results.png)
