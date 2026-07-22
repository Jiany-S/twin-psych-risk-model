# MultiPhysio Benchmark

Run directory: `experiments\runs\multiphysio_cv_20260722_180827`

This benchmark uses grouped subject cross-validation over `bio_features_60s.csv`. One row is a 60-second precomputed physiological feature interval, suitable for slow workload or affect estimation, not immediate safety intervention.

Only physiological features are used: `hrv_mean_nn`, `eda_mean`, `emg_rmse`, and `rrv_mean_bb`. No fabricated role, specialization, experience, or subject-ID metadata is used as a feature.

Targets: STAI-Y1 stress (`STAI >= 40`), NASA-TLX cognitive workload (`NASA >= 40`), SAM Valence (`Valence >= 3`), and SAM Arousal (`Arousal >= 3`).

Models: Dummy most-frequent, Dummy stratified, Logistic Regression, Random Forest, AdaBoost, XGBoost. TFT is disabled because this benchmark uses tabular 60-second feature rows rather than a meaningful raw temporal stream.

## Aggregate Metrics

The table below shows fold means. `aggregate_results.csv` and `aggregate_results.json` include mean, standard deviation, median, and 95% confidence interval columns for every metric. `per_subject_metrics.csv` reports metrics for each held-out subject within each grouped fold.

| target | model | n_folds | auroc_mean | auprc_mean | macro_f1_mean | balanced_accuracy_mean | brier_mean | ece_mean | prevalence_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arousal_sam | adaboost | 5.0000 | 0.4597 | 0.1758 | 0.2610 | 0.4868 | 0.1736 | 0.1615 | 0.1800 |
| arousal_sam | dummy_most_frequent | 5.0000 | 0.5000 | 0.1800 | 0.1465 | 0.5000 | 0.1800 | 0.1800 | 0.1800 |
| arousal_sam | dummy_stratified | 5.0000 | 0.5156 | 0.1862 | 0.2752 | 0.5080 | 0.2997 | 0.2997 | 0.1800 |
| arousal_sam | logistic_regression | 5.0000 | 0.5046 | 0.2139 | 0.2916 | 0.5013 | 0.2532 | 0.3182 | 0.1800 |
| arousal_sam | random_forest | 5.0000 | 0.3680 | 0.1451 | 0.1922 | 0.4778 | 0.1892 | 0.1995 | 0.1800 |
| arousal_sam | xgboost | 5.0000 | 0.3833 | 0.1536 | 0.1827 | 0.4447 | 0.1771 | 0.1797 | 0.1800 |
| cognitive_workload_nasa | adaboost | 5.0000 | 0.5060 | 0.2293 | 0.3007 | 0.4469 | 0.1741 | 0.2187 | 0.1540 |
| cognitive_workload_nasa | dummy_most_frequent | 5.0000 | 0.5000 | 0.1925 | 0.1903 | 0.4000 | 0.1540 | 0.1540 | 0.1540 |
| cognitive_workload_nasa | dummy_stratified | 5.0000 | 0.4863 | 0.1890 | 0.2572 | 0.3946 | 0.2746 | 0.2746 | 0.1540 |
| cognitive_workload_nasa | logistic_regression | 5.0000 | 0.4806 | 0.1864 | 0.3452 | 0.4083 | 0.2487 | 0.3388 | 0.1540 |
| cognitive_workload_nasa | random_forest | 5.0000 | 0.4592 | 0.1817 | 0.3167 | 0.4762 | 0.1935 | 0.2102 | 0.1540 |
| cognitive_workload_nasa | xgboost | 5.0000 | 0.4554 | 0.1906 | 0.3410 | 0.4376 | 0.1675 | 0.1678 | 0.1540 |
| stress_stai | adaboost | 5.0000 | 0.4550 | 0.1933 | 0.2833 | 0.3991 | 0.1875 | 0.2051 | 0.1825 |
| stress_stai | dummy_most_frequent | 5.0000 | 0.5000 | 0.1825 | 0.1492 | 0.5000 | 0.1825 | 0.1825 | 0.1825 |
| stress_stai | dummy_stratified | 5.0000 | 0.5137 | 0.1883 | 0.1492 | 0.5000 | 0.2991 | 0.2991 | 0.1825 |
| stress_stai | logistic_regression | 5.0000 | 0.5479 | 0.2962 | 0.2233 | 0.4670 | 0.2420 | 0.3066 | 0.1825 |
| stress_stai | random_forest | 5.0000 | 0.4177 | 0.1701 | 0.3272 | 0.4789 | 0.1949 | 0.2025 | 0.1825 |
| stress_stai | xgboost | 5.0000 | 0.3858 | 0.1758 | 0.2810 | 0.4105 | 0.1834 | 0.1814 | 0.1825 |
| valence_sam | adaboost | 5.0000 | 0.2870 | 0.9290 | 0.4868 | 0.5000 | 0.1177 | 0.2576 | 0.9491 |
| valence_sam | dummy_most_frequent | 5.0000 | 0.5000 | 0.9491 | 0.4868 | 0.5000 | 0.0509 | 0.0509 | 0.9491 |
| valence_sam | dummy_stratified | 5.0000 | 0.4778 | 0.9473 | 0.4887 | 0.4989 | 0.1033 | 0.1033 | 0.9491 |
| valence_sam | logistic_regression | 5.0000 | 0.3274 | 0.9321 | 0.4707 | 0.4813 | 0.2192 | 0.3947 | 0.9491 |
| valence_sam | random_forest | 5.0000 | 0.4829 | 0.9435 | 0.4868 | 0.5000 | 0.0542 | 0.0549 | 0.9491 |
| valence_sam | xgboost | 5.0000 | 0.3536 | 0.9246 | 0.4868 | 0.5000 | 0.0524 | 0.0498 | 0.9491 |

## Failure Summary

No failures.

## Artifacts

- `fold_manifests/fold_*.csv`
- `per_fold_metrics.csv`
- `per_subject_metrics.csv`
- `aggregate_results.csv`
- `failures.json`
