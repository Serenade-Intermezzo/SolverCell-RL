# Preference implementation engineering checks

The frozen implementation is commit `f304a2f`; pilot preparation is commit `2382fc1`. The matching scientific source fingerprint is `9b8a1b044b789a34ca7d3898bb3613db85f9e0008163287cdde5cb1b4b626b8c`.

- `tests_final.log`: all85 tests pass, including both methods' full short training, evaluation isolation, bitwise mid-episode resume with wrapped replay, and duplicate-entrypoint exit-record preservation.
- `tests.log`: earlier84-test pass before the exit-record regression test and final accounting/entrypoint hardening. Historical engineering evidence only.
- `debug_diffusion.log`: earlier4-update local debug run at source `1e438571954a761dcd2d8a9f4edc95d88d343ac71ba3fa5b25efc7318ccc4297`. This is not a final-source pilot or a capability result. Its resolved config is in Git-ignored `outputs/preference_debug_20261104/diffusion/config.json`; `experiments/preference_debug_20261104` is an earlier preparation snapshot and is not the formal pilot manifest.
- Final-source server preflights completed4 updates for both methods under PyTorch2.5.1 before the two formal pilot workers launched. Their complete metrics/exit records and logs are preserved in [pilot runtime evidence](../preference_pilot_20261104/runtime/deployment.json).

The complete1000-update two-model server pilot is audited separately in [RESULTS.md](../preference_pilot_20261104/RESULTS.md). The85 tests establish implementation invariants, not scientific superiority.
