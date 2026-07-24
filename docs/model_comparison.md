# Model Comparison

Results below are separated by dataset and task. Incompatible tasks are not ranked against each other.

## MultiPhysio 60-Second Feature Benchmark

Protocol: capped leave-one-subject-out smoke over `bio_features_60s.csv`, `cv.max_folds: 5`. TFT is disabled because the input is a 60-second tabular feature table.

| Target | Prevalence | Valid fold proportion | Best non-dummy note |
| --- | ---: | ---: | --- |
| STAI stress | 0.183 overall; capped folds had 0 positive test prevalence | 0.0 in shown capped folds | Not interpretable in capped run. |
| NASA-TLX workload | 0.156 overall | 0.2 | AdaBoost/RF/XGB rank above dummy by AUROC/AUPRC in limited folds, but support is weak. |
| SAM Valence | 0.949 overall | 0.0 in capped folds | Extreme prevalence; do not claim performance. |
| SAM Arousal | 0.180 overall | 0.8 | Logistic and XGBoost improve AUPRC over prevalence in capped folds. |

See `experiments/runs/multiphysio_cv_20260723_135339/aggregate_results.csv` for full AUROC, AUPRC, macro F1, balanced accuracy, Brier, and ECE columns.

## WESAD Fast Current-State Detection

Task: `P(current WESAD protocol stress state | recent causal physiology)`.

| Model | Train subjects | Validation subjects | Test subjects | Context | Stride | AUROC | AUPRC | F1 selected | p95 latency |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dummy | S2-S5 | S6-S7 | S8-S9 | 3s | 0.25s | 0.500 | 0.298 | 0.000 | 0.63 ms |
| Logistic Regression | S2-S5 | S6-S7 | S8-S9 | 3s | 0.25s | 0.559 | 0.438 | 0.0069 | 1.44 ms |
| TCN | S2-S5 | S6-S7 | S8-S9 | 3s | 0.25s | 0.623 | 0.325 | 0.000 | 0.61 ms |

The trained models did not yield useful selected-threshold F1 at the configured policy threshold. TCN improved AUROC over dummy but not AUPRC meaningfully.

## WESAD Slow Forecasting

Task: `P(WESAD protocol stress at t + horizon | physiology up to t)`. Context is 30s, stride is 1s. Calibration is horizon-specific Platt scaling.

| Model | Horizon | AUROC | AUPRC | F1 selected | F1 fixed 0.5 | Brier | ECE | Limitation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Logistic | 5s | 0.796 | 0.578 | 0.502 | 0.522 | 0.203 | 0.179 | Protocol labels are long blocks. |
| Random Forest | 5s | 0.965 | 0.919 | 0.785 | 0.800 | 0.142 | 0.160 | Strong ranking, weak real early-warning evidence. |
| XGBoost | 5s | 0.963 | 0.926 | 0.786 | 0.799 | 0.101 | 0.150 | Same limitation. |
| TCN | 5s | 0.917 | 0.824 | 0.455 | 0.514 | 0.488 | 0.561 | Calibration weak. |
| TFT-family | 5s | 0.977 | 0.952 | 0.455 | 0.519 | 0.246 | 0.416 | Not a full production TFT; attention not causal. |
| Logistic | 30s | 0.796 | 0.578 | 0.502 | 0.522 | 0.203 | 0.179 | 30s labels often same as 5s. |
| Random Forest | 30s | 0.965 | 0.919 | 0.785 | 0.800 | 0.142 | 0.160 | Same limitation. |
| XGBoost | 30s | 0.963 | 0.926 | 0.786 | 0.799 | 0.101 | 0.150 | Same limitation. |
| TCN | 30s | 0.190 | 0.192 | 0.345 | 0.034 | 0.261 | 0.336 | Failed against dummy on ranking. |
| TFT-family | 30s | 0.977 | 0.952 | 0.455 | 0.527 | 0.245 | 0.418 | Threshold/calibration still weak. |

## Replay Policy Results

Replay results are policy simulations, not classifier rankings.

| Category | Interpretation | Result summary |
| --- | --- | --- |
| real_model_physiological_replay | Saved held-out WESAD predictions drive policy. | No warnings at current thresholds; model ranking metrics are reported separately. |
| oracle_label_policy_simulation | Label-derived probabilities drive policy. | Software policy test only; not model performance. |
| synthetic_physical_safety_simulation | Physical safety rules only. | Manifest scenarios pass. |
| combined_integration_simulation | Synthetic physical plus synthetic ML probabilities. | Priority, latching, reset, degraded, and hysteresis scenarios pass. |

