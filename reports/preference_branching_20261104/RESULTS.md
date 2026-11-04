# Shared-prefix preference sampling: completed study

All runs passed source/parent hash, exit, complete H20 evaluation, scalarization, feasibility and actual compute audits. The trained v2 models, optimizer/RNG states and checkpoints remain unchanged. No new training was performed.

Source: `05c3cfffb6e7ed1bad6a077085e7da70bfaef41ad88323108052a261740fcd10`. Parent scientific source: `33fcff51897f7485ddce4a303d089526d0cada8e165a4c380e7c9509db6680b3`. [Prespecified design](../../docs/PREFERENCE_BRANCHING_V3.md).

## Seed0 validation screen

| Topology | Actor NFE | Utility / independent8 | Worst-weight ratio | Median inference latency ratio | Eligible |
|---|---:|---:|---:|---:|---|
| tree_m2_k2 | 52 | 0.9999 | 0.9894 | 0.9891 | True |
| tree_m2_k4 | 40 | 0.9804 | 0.9755 | 0.9815 | True |
| tree_m2_k6 | 28 | 0.9916 | 0.9845 | 0.9727 | True |
| tree_m4_k2 | 56 | 0.9847 | 0.9654 | 0.9921 | True |
| tree_m4_k4 | 48 | 0.9703 | 0.9529 | 0.9922 | False |
| tree_m4_k6 | 40 | 0.9883 | 0.9808 | 0.9819 | True |

Frozen evaluated topology: **tree_m2_k6**. Promotion eligibility: **True**. eligible lowest-NFE topology. No final episode or preference selected this topology.

Weight ESS measures normalized weight concentration only; it does not account for correlated siblings. The variance/support audits in study.json separately measure the actual conditional branch structure.

## Fresh final episodes at held-out preference weights

| Model training seed | Scope | Independent8 mean | Branch mean | Paired mean difference | Episode-mean SE | Environment-inclusive wall ratio |
|---:|---|---:|---:|---:|---:|---:|
| 0 | selection_model | 306.162 | 310.148 | +3.986 | 5.804 | 0.982 |
| 1 | out_of_selection_fixed_model | 294.491 | 303.391 | +8.901 | 5.948 | 0.969 |
| 2 | out_of_selection_fixed_model | 310.221 | 314.321 | +4.100 | 5.031 | 0.981 |

Final episode seeds800000–800031 are common across modes and preferences. Paired SE first averages each episode over the 4 preferences, preserving within-seed dependence. Model seeds1/2 were not used for topology selection, but are existing v2 checkpoints, not newly trained replications. Three known models and one instance cannot establish broad noninferiority or intrinsic sampler speed.

## Budget boundary

Independent8 uses64 actor NFE,18 twin-Q rows and9 decoder rows per action. Branching scores the same8 leaves with the same Q/decoder budget but fewer actor rows. Independent3/4/5/6/7 controls have lower Q/decoder counts; matched or bracketing actor NFE is not matched total compute. Direct sphere/diffusion uses only8 actor NFE. Do not infer superiority to tuned sphere from within-diffusion candidate sharing.

The finite importance pool is correlated and normalized selection is not exact utility-tilted sampling. Root-wise inverse child-count weights preserve unnormalized estimands; consistency requires more independent roots. All negative topologies and controls are retained.

[Selection lock](selection.json) · [Complete study and support/variance audits](study.json) · [Figure](branching_tradeoff.png)
