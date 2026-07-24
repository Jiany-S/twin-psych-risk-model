# MultiPhysio Benchmark

Run directory: `experiments\runs\multiphysio_cv_20260723_195159`

This benchmark uses `leave_one_subject_out` grouped by subject over `bio_features_60s.csv`. This run is capped at `5` held-out folds for smoke validation; set `cv.max_folds: null` for full LOSO. One row is a 60-second precomputed physiological feature interval, suitable for slow workload or affect estimation, not immediate safety intervention.

Only physiological features are used: `hrv_mean_nn`, `eda_mean`, `emg_rmse`, and `rrv_mean_bb`. No fabricated role, specialization, experience, or subject-ID metadata is used as a feature.

Targets: STAI-Y1 stress (`STAI >= 40`), NASA-TLX cognitive workload (`NASA >= 40`), SAM Valence (`Valence >= 3`), and SAM Arousal (`Arousal >= 3`).

Models: Dummy most-frequent, Dummy stratified, Logistic Regression, Random Forest, AdaBoost, XGBoost. TFT is disabled because this benchmark uses tabular 60-second feature rows rather than a meaningful raw temporal stream.

## Aggregate Metrics

The table below shows fold means for validation-selected thresholds. `aggregate_results.csv` and `aggregate_results.json` include mean, standard deviation, median, and 95% confidence interval columns for fixed-0.5 and validation-selected metrics. `per_subject_metrics.csv` reports metrics for each held-out subject within each grouped fold.

| target | model | n_folds | auroc_mean | auprc_mean | macro_f1_mean | balanced_accuracy_mean | brier_mean | ece_mean | prevalence_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arousal_sam | adaboost | 5.0000 | 0.5000 | 0.3478 | 0.3657 | 0.4518 | 0.1802 | 0.1883 | 0.2151 |
| arousal_sam | dummy_most_frequent | 5.0000 | 0.5000 | 0.2689 | 0.5347 | 0.5000 | 0.2151 | 0.2151 | 0.2151 |
| arousal_sam | dummy_stratified | 5.0000 | 0.5368 | 0.2900 | 0.5067 | 0.5170 | 0.2559 | 0.2559 | 0.2151 |
| arousal_sam | logistic_regression | 5.0000 | 0.5884 | 0.3939 | 0.3345 | 0.4710 | 0.2606 | 0.3213 | 0.2151 |
| arousal_sam | random_forest | 5.0000 | 0.5281 | 0.3630 | 0.3783 | 0.4383 | 0.1872 | 0.1969 | 0.2151 |
| arousal_sam | xgboost | 5.0000 | 0.5342 | 0.3996 | 0.3008 | 0.4154 | 0.1750 | 0.1900 | 0.2151 |
| cognitive_workload_nasa | adaboost | 5.0000 | 0.8067 | 0.6074 | 0.2445 | 0.2112 | 0.1230 | 0.2628 | 0.0529 |
| cognitive_workload_nasa | dummy_most_frequent | 5.0000 | 0.5000 | 0.2647 | 0.8847 | 0.5000 | 0.0529 | 0.0529 | 0.0529 |
| cognitive_workload_nasa | dummy_stratified | 5.0000 | 0.5156 | 0.2723 | 0.4761 | 0.4602 | 0.1446 | 0.1446 | 0.0529 |
| cognitive_workload_nasa | logistic_regression | 5.0000 | 0.3867 | 0.2315 | 0.5123 | 0.3933 | 0.2299 | 0.4097 | 0.0529 |
| cognitive_workload_nasa | random_forest | 5.0000 | 0.7000 | 0.5625 | 0.5488 | 0.4404 | 0.1159 | 0.1896 | 0.0529 |
| cognitive_workload_nasa | xgboost | 5.0000 | 0.6711 | 0.5112 | 0.8847 | 0.5000 | 0.0709 | 0.1367 | 0.0529 |
| stress_stai | adaboost | 5.0000 | nan | nan | 0.3409 | 0.2934 | 0.0875 | 0.2866 | 0.0000 |
| stress_stai | dummy_most_frequent | 5.0000 | nan | nan | 1.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| stress_stai | dummy_stratified | 5.0000 | nan | nan | 0.4727 | 0.4483 | 0.1035 | 0.1035 | 0.0000 |
| stress_stai | logistic_regression | 5.0000 | nan | nan | 0.3934 | 0.3616 | 0.1271 | 0.3291 | 0.0000 |
| stress_stai | random_forest | 5.0000 | nan | nan | 1.0000 | 0.5000 | 0.0240 | 0.1067 | 0.0000 |
| stress_stai | xgboost | 5.0000 | nan | nan | 1.0000 | 0.5000 | 0.0113 | 0.0786 | 0.0000 |
| valence_sam | adaboost | 5.0000 | nan | nan | 0.8990 | 0.4980 | 0.1084 | 0.3244 | 1.0000 |
| valence_sam | dummy_most_frequent | 5.0000 | nan | nan | 1.0000 | 0.5000 | 0.0000 | 0.0000 | 1.0000 |
| valence_sam | dummy_stratified | 5.0000 | nan | nan | 1.0000 | 0.5000 | 0.0000 | 0.0000 | 1.0000 |
| valence_sam | logistic_regression | 5.0000 | nan | nan | 0.5036 | 0.3911 | 0.4055 | 0.5740 | 1.0000 |
| valence_sam | random_forest | 5.0000 | nan | nan | 0.6918 | 0.4843 | 0.0014 | 0.0183 | 1.0000 |
| valence_sam | xgboost | 5.0000 | nan | nan | 0.8885 | 0.4794 | 0.0009 | 0.0137 | 1.0000 |

## Target Quality

`target_quality.csv`, `per_subject_prevalence.csv`, and `threshold_sensitivity.csv` report prevalence, per-subject class support, valid AUROC/AUPRC fold proportions, and threshold sensitivity. Extreme-prevalence targets are flagged and should not be described as strong by AUPRC without comparing to prevalence.

## Failure Summary

No failures.

## Artifacts

- `fold_manifests/fold_*.csv`
- `per_fold_metrics.csv`
- `per_subject_metrics.csv`
- `aggregate_results.csv`
- `failures.json`