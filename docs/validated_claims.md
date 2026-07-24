# Validated Claims

## Scientifically Evaluated

| Claim | Status | Evidence | Limits |
| --- | --- | --- | --- |
| MultiPhysio tabular benchmark runs with grouped subject CV over 60-second bio features. | Partial/capped LOSO smoke | `docs/multiphysio_benchmark.md` | Only five held-out folds by default; 60-second features and questionnaire labels. |
| WESAD fast current-state stress detection runs on held-out subjects. | Evaluated smoke | `docs/fast_wesad_diagnostics.md` | WESAD protocol stress only, not physical hazard risk. |
| WESAD slow protocol stress forecasting emits separate 5s and 30s horizon outputs. | Evaluated smoke | `docs/slow_forecaster.md` | Protocol blocks are long; horizons are often label-identical. |
| Real-model physiological replay can consume explicitly paired saved held-out prediction artifacts. | Engineering/scientific bridge | `streaming_real_model_physiological_replay_*` runs and manifest compatibility tests | Policy thresholds may emit no warnings even when ranking metrics are nonzero. |

## Engineering Validation

Causal buffering, multi-horizon timestamp alignment, model serialization, prediction artifact validation, replay determinism, physical-kernel priority, emergency latching, reset handling, and state-machine invariants are covered by tests.

## Oracle-Label Simulations

`oracle_label_policy_simulation` replays label-derived probabilities for policy testing only. Its warning coverage is not model performance.

## Synthetic Simulations

`synthetic_physical_safety_simulation` and `combined_integration_simulation` validate software rules only. They are not empirical safety validation.

## Unsupported Claims

The repository does not support claims of complete construction-risk prediction, collision prediction, incident reduction, ISO compliance, safety certification, deployed velocity adaptation, or validated 3DCP worker safety intervention.
