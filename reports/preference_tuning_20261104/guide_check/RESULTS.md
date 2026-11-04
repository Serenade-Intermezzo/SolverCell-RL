# Validation-only guide mechanism check

All three base learners, replay, training environment/RNG and direct validation curves are bitwise equal. Reference suffix draws/targets are also exactly equal across all18 calibration groups. Only the guide data grouping/loss differs. No final preference or final episode is used.

| Guide | Median centered RMSE | Median constant-relative RMSE ratio | Median within-group r |
|---|---:|---:|---:|
| diff_b060_t010 | 0.04307 | 1.333 | 0.058 |
| group_b060_t010 | 0.05787 | 1.611 | 0.407 |
| center_b060_t010 | 0.03033 | 0.993 | 0.424 |

Each group fixes state, preference and noise level. Only two replay states are probed. Finite128-suffix log-mean-exp has Monte Carlo noise/bias; inspect all group values and split-half consistency in summary.json.

| Guide | Weight | ESS fraction | SMC minus terminal | Paired episode-mean SE | Resampling events |
|---|---:|---:|---:|---:|---:|
| diff_b060_t010 | 0.0 | 0.5 | 0.000 | 0.000 | 0 |
| diff_b060_t010 | 0.0 | 0.9 | 0.000 | 0.000 | 0 |
| diff_b060_t010 | 0.5 | 0.5 | 0.000 | 0.000 | 0 |
| diff_b060_t010 | 0.5 | 0.9 | 0.000 | 0.000 | 0 |
| diff_b060_t010 | 1.0 | 0.5 | 0.000 | 0.000 | 0 |
| diff_b060_t010 | 1.0 | 0.9 | 0.000 | 0.000 | 0 |
| group_b060_t010 | 0.0 | 0.5 | 0.000 | 0.000 | 0 |
| group_b060_t010 | 0.0 | 0.9 | 0.000 | 0.000 | 0 |
| group_b060_t010 | 0.5 | 0.5 | 0.000 | 0.000 | 0 |
| group_b060_t010 | 0.5 | 0.9 | 0.000 | 0.000 | 0 |
| group_b060_t010 | 1.0 | 0.5 | 0.000 | 0.000 | 0 |
| group_b060_t010 | 1.0 | 0.9 | 0.000 | 0.000 | 0 |
| center_b060_t010 | 0.0 | 0.5 | 0.000 | 0.000 | 0 |
| center_b060_t010 | 0.0 | 0.9 | 0.000 | 0.000 | 0 |
| center_b060_t010 | 0.5 | 0.5 | 0.000 | 0.000 | 0 |
| center_b060_t010 | 0.5 | 0.9 | 0.000 | 0.000 | 0 |
| center_b060_t010 | 1.0 | 0.5 | 0.000 | 0.000 | 0 |
| center_b060_t010 | 1.0 | 0.9 | 0.000 | 0.000 | 0 |

These16 episodes per context describe a fixed seed0 checkpoint. They do not establish training-seed stability or an efficiency advantage. All SMC calls add guide cost; compare full counters and wall time. No larger guide sweep or default change follows automatically.
