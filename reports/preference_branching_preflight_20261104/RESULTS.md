# Shared-prefix preference sampling: engineering preflight only

All runs passed source/parent hash, exit, complete H20 evaluation, scalarization, feasibility and actual compute audits. The trained v2 models, optimizer/RNG states and checkpoints remain unchanged. No new training was performed.

Source: `05c3cfffb6e7ed1bad6a077085e7da70bfaef41ad88323108052a261740fcd10`. Parent scientific source: `33fcff51897f7485ddce4a303d089526d0cada8e165a4c380e7c9509db6680b3`. [Prespecified design](../../docs/PREFERENCE_BRANCHING_V3.md).

## Seed0 validation screen

| Topology | Actor NFE | Utility / independent8 | Worst-weight ratio | Median inference latency ratio | Eligible |
|---|---:|---:|---:|---:|---|
| tree_m2_k2 | 52 | 0.9465 | 0.9465 | 0.9663 | False |
| tree_m2_k4 | 40 | 0.9406 | 0.9406 | 0.9553 | False |
| tree_m2_k6 | 28 | 0.9014 | 0.9014 | 0.9453 | False |
| tree_m4_k2 | 56 | 1.0326 | 1.0326 | 0.9545 | True |
| tree_m4_k4 | 48 | 0.9401 | 0.9401 | 0.9758 | False |
| tree_m4_k6 | 40 | 1.0000 | 1.0000 | 0.9491 | True |

Frozen evaluated topology: **tree_m4_k6**. Promotion eligibility: **True**. eligible lowest-NFE topology. No final episode or preference selected this topology.

Weight ESS measures normalized weight concentration only; it does not account for correlated siblings. The variance/support audits in study.json separately measure the actual conditional branch structure.

## Fresh final episodes at unseen fitting preferences

| Model training seed | Scope | Independent8 mean | Branch mean | Paired mean difference | Episode-mean SE | Environment-inclusive wall ratio |
|---:|---|---:|---:|---:|---:|---:|
| 0 | selection_model | 433.100 | 431.788 | -1.313 | 6.537 | 0.976 |
| 1 | out_of_selection_fixed_model | 359.138 | 387.350 | +28.212 | 20.663 | 0.980 |
| 2 | out_of_selection_fixed_model | 402.275 | 414.400 | +12.125 | 46.425 | 0.983 |

Final episode seeds800000–800001 are common across modes and preferences. Paired SE first averages each episode over the 1 preferences, preserving within-seed dependence. Model seeds1/2 were not used for topology selection, but are existing v2 checkpoints, not newly trained replications. Three known models and one instance cannot establish broad noninferiority or intrinsic sampler speed.

## Budget boundary

Independent8 uses64 actor NFE,18 twin-Q rows and9 decoder rows per action. Branching scores the same8 leaves with the same Q/decoder budget but fewer actor rows. Independent3/4/5/6/7 controls have lower Q/decoder counts; matched or bracketing actor NFE is not matched total compute. Direct sphere/diffusion uses only8 actor NFE. Do not infer superiority to tuned sphere from within-diffusion candidate sharing.

The finite importance pool is correlated and normalized selection is not exact utility-tilted sampling. Root-wise inverse child-count weights preserve unnormalized estimands; consistency requires more independent roots. All negative topologies and controls are retained.

[Selection lock](selection.json) · [Complete study and support/variance audits](study.json) · [Figure](branching_tradeoff.png)
