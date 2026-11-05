# Style-Routing probe

Interpretation: [docs/SOLVER_LOCALIZATION_DESIGN.md, section 9](../../docs/SOLVER_LOCALIZATION_DESIGN.md). Known myopic objective with an exact mean map as the denoiser. There is no learning, no MDP dynamics and no RL result. The task is a declared extension, not an upstream benchmark.

| File | Content |
|---|---|
| `routing_core.py` | Time-expanded closed-walk action set, DP solver, mean map, log Z, exact sampling |
| `probe.py`, `probe.json`, `probe.log` | DP correctness, scale, style table, R², equal-budget comparison; seed 20261109 |
| `probe2.py`, `probe2.json`, `probe2.log` | Style frontier sweep, district-bonus comparison; seed 20261110 |

`probe2.py` reads `feature_scale` from `probe.json`. Runs used Python 3.12 and NumPy 2.1.3 on CPU; graph data comes from `vendor/lsflow/environment/network`. Style references come from a heuristic and are not proven optimal.

```bash
python probe.py probe.json
python probe2.py probe2.json
```
