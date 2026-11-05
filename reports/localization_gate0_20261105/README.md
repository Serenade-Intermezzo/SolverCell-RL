# Gate 0: oracle mechanism check for SOLO

Design and interpretation: [docs/SOLVER_LOCALIZATION_DESIGN.md](../../docs/SOLVER_LOCALIZATION_DESIGN.md). This is a frozen-Q oracle-denoiser check. It is not an RL result and trains nothing.

| File | Content |
|---|---|
| `shift_check.py` | Enumeration check of the linear-tilt shift identity; pure Python |
| `gate0.py`, `gate0.json` | Equal-budget gain: SNIS, best-of-K, shift, shift plus residual SNIS, SL chains; seed 20261105 |
| `gate0b.py`, `gate0b.json` | Exact TV on m=14, plus SDE vs probability-flow ODE vs mean-field controls; seed 20261106 |
| `r2_real.py`, `r2_real.json` | Read-only action linearity on Dynamic Intervention instance 0; seed 20261107 |
| `softv.py` | Enumeration check of the soft-value estimator without action density; pure Python; seed 20261108 |

Runs used Python 3.12 and NumPy 2.1.3 on CPU. After the runs, a dead branch in `Sigmoid.grad` and one unused assignment were removed from `gate0.py`; behavior is unchanged and the gradient matches finite differences to 4e-9. The JSON files were not regenerated.

```bash
python gate0.py gate0.json
python gate0b.py gate0b.json
python r2_real.py r2_real.json
```

`r2_real.py` has the workspace path hardcoded in `sys.path`.
