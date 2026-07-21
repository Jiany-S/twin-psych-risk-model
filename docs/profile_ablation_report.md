# Profile Ablation Report

Date: 2026-07-21

This report documents the Prompt 4 personalization refactor and smoke ablations. The MultiPhysio and WESAD smoke test folds below contain only negative test labels, so AUROC and AUPRC are undefined. Treat these as execution and feature-list checks, not model-quality evidence.

## Implementation Summary

| Component | Current behavior |
| --- | --- |
| Global normalization | Fits physiology mean/std on training rows only; applies train-fitted statistics to validation/test. |
| Calibration normalization | Fits per-subject baseline/rest calibration statistics from explicit calibration rows only. Held-out subject calibration uses unlabeled baseline/rest rows and ignores target labels. |
| Profile vector | Optional model input controlled by `profiles.include_calibration_features`, `include_role_metadata`, and `include_experience_metadata`. |
| Metadata | Hash-generated specialization and experience values removed. Missing role is `-1`; missing experience is `0`. MultiPhysio can include real `Experience`; role metadata remains disabled because no real role field is loaded. |
| XGBoost | Profile-on appends profile columns to the engineered feature matrix; profile-off omits them. |
| TFT | Profile values are passed as static real covariates when enabled. `worker_id` remains only a sequence grouping key, not a static categorical embedding. No validation/test worker rows are injected into training. |

## Ablation Runs

| Run | Config | Normalization | Profile columns | Test prevalence | Notes |
| --- | --- | --- | --- | ---: | --- |
| `experiments/runs/20260721_154928` | `src/config/multiphysio_ablation_global.yaml` | global | none | 0.000 | MultiPhysio time split, TFT disabled for fast benchmark hierarchy check. |
| `experiments/runs/20260721_155016` | `src/config/multiphysio_ablation_calibration.yaml` | calibration | none | 0.000 | Same split/features as global run. |
| `experiments/runs/20260721_155206` | `src/config/multiphysio_ablation_calibration_features.yaml` | calibration | 16 calibration columns | 0.000 | Adds baseline mean/std/median/IQR for four MultiPhysio modalities. |
| `experiments/runs/20260721_155241` | `src/config/multiphysio_ablation_calibration_metadata.yaml` | calibration | 16 calibration columns + `experience_metadata` | 0.000 | Adds real MultiPhysio experience metadata; role disabled. |
| `experiments/runs/20260721_155518` | `src/config/wesad_tiny.yaml` | global | 12 calibration columns | 0.000 | WESAD tiny raw-signal run, includes TFT execution check. |

## MultiPhysio Metrics

| Run | Model | AUROC | AUPRC | F1 | Specificity | Balanced Acc. | Brier | ECE | Pred. + Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| global | dummy_most_frequent | NaN | NaN | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| global | logistic_regression | NaN | NaN | 0.000 | 0.039 | 0.039 | 0.397 | 0.588 | 0.961 |
| global | random_forest | NaN | NaN | 0.000 | 0.853 | 0.853 | 0.042 | 0.169 | 0.147 |
| global | xgboost | NaN | NaN | 0.000 | 0.804 | 0.804 | 0.078 | 0.178 | 0.196 |
| calibration | dummy_most_frequent | NaN | NaN | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| calibration | logistic_regression | NaN | NaN | 0.000 | 0.000 | 0.000 | 0.855 | 0.916 | 1.000 |
| calibration | random_forest | NaN | NaN | 0.000 | 0.520 | 0.520 | 0.079 | 0.249 | 0.480 |
| calibration | xgboost | NaN | NaN | 0.000 | 0.608 | 0.608 | 0.159 | 0.286 | 0.392 |
| calibration+features | dummy_most_frequent | NaN | NaN | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| calibration+features | logistic_regression | NaN | NaN | 0.000 | 0.000 | 0.000 | 0.855 | 0.916 | 1.000 |
| calibration+features | random_forest | NaN | NaN | 0.000 | 0.500 | 0.500 | 0.081 | 0.251 | 0.500 |
| calibration+features | xgboost | NaN | NaN | 0.000 | 0.431 | 0.431 | 0.225 | 0.344 | 0.569 |
| calibration+metadata | dummy_most_frequent | NaN | NaN | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| calibration+metadata | logistic_regression | NaN | NaN | 0.000 | 0.000 | 0.000 | 0.862 | 0.917 | 1.000 |
| calibration+metadata | random_forest | NaN | NaN | 0.000 | 0.549 | 0.549 | 0.096 | 0.284 | 0.451 |
| calibration+metadata | xgboost | NaN | NaN | 0.000 | 0.794 | 0.794 | 0.227 | 0.351 | 0.206 |

## WESAD Tiny Metrics

| Model | AUROC | AUPRC | F1 | Specificity | Balanced Acc. | Brier | ECE | Pred. + Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dummy_most_frequent | NaN | NaN | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| logistic_regression | NaN | NaN | 0.000 | 0.000 | 0.000 | 0.772 | 0.871 | 1.000 |
| random_forest | NaN | NaN | 0.000 | 0.951 | 0.951 | 0.120 | 0.337 | 0.049 |
| xgboost | NaN | NaN | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| tft | NaN | NaN | 0.000 | 0.110 | 0.110 | 0.699 | 0.744 | 0.890 |

## Exact Feature Lists

The full per-run feature lists are saved in each run's `profile_ablation_report.md` and `profile_ablation_report.json`.

| Run | Feature count by model |
| --- | --- |
| `20260721_154928` | dummy/logistic/random_forest/xgboost: 24 |
| `20260721_155016` | dummy/logistic/random_forest/xgboost: 24 |
| `20260721_155206` | dummy/logistic/random_forest/xgboost: 40 |
| `20260721_155241` | dummy/logistic/random_forest/xgboost: 41 |
| `20260721_155518` | dummy/logistic/random_forest/xgboost: 25; TFT feature list is stored through dataset covariates, not the flat feature list. |

## Blockers

1. The small MultiPhysio and WESAD smoke test folds are single-class, so AUROC/AUPRC are undefined and F1 is not informative.
2. WESAD temporal units remain unresolved from the audit: native rows are not guaranteed to match configured seconds after downsampling.
3. TFT holdout behavior should be rerun on a larger subject-holdout config now that worker-ID embeddings and fake rows are removed.
