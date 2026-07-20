# Current Pipeline Audit

Audit date: 2026-07-20

Scope: `src/run_experiment.py`, dataset loaders, windowing/features, worker profiles, XGBoost/TFT training, WESAD and MultiPhysio configs, tracked weekly reports. No architectural changes were made.

## Executive Findings

| Area | Finding | Risk |
| --- | --- | --- |
| MultiPhysio target semantics | `y_stress` is derived from `labels.csv` column `NASA >= 40`. NASA-TLX is workload, not a direct stress label. | High: results should be called workload/high NASA-TLX risk, not stress, unless justified as a proxy. |
| MultiPhysio physiology mapping | `temp` is populated from `EMG_RMSE` by default. This is EMG, not skin temperature. | High: temperature feature names and plots are scientifically wrong for MultiPhysio. |
| WESAD time base | Native WESAD pickle rows are chest sample rows after truncating channels to shared length; configs treat them as 4 Hz and divide by `downsample_factor`. | High: reported seconds, HRV peak distances, forecast horizons, and stride are wrong for native pickles. |
| WESAD downsampling | Loader physically keeps every `downsample_factor`-th row, and `run_experiment.py` also divides `task.sampling_rate_hz` by that same factor. | High: if `sampling_rate_hz` is already intended as post-downsample rate, this double-adjusts; if intended as raw WESAD, it is still wrong because native chest is not 4 Hz. |
| MultiPhysio temporal rows | Each row is already a 60-second precomputed feature window. The pipeline windows those feature rows again. | Medium/high: configured windows are windows of 60-second summaries, not raw physiological windows. |
| Profile ablations | Several profile-on/off runs change normalization mode or add no static features, so "profiles on vs off" is not consistently isolating profile information. | High: profile claims are not reliable without matched config pairs. |
| TFT subject holdout | TFT uses `worker_id` as a static categorical and inserts artificial training rows for validation/test workers. | High: held-out subject evaluation is contaminated by test/val subject IDs and fake train samples. |
| Held-out subject validity | XGBoost subject sets are disjoint; TFT subject sets are disjoint at raw split level but not clean at model-input level because of the encoder workaround. | High for TFT, medium for XGBoost because subject-level prevalence/task composition can dominate. |
| Metric comparability | Weekly report `2026-04-16` compares runs where test prevalence changed from 25/135 to 96/173 positives. | High: F1/AUPRC/calibration changes cannot be interpreted as pure model improvement. |
| Dummy prevalence baselines | Metrics include prevalence and class counts but do not explicitly compare AUPRC to prevalence or F1 to trivial always-positive/always-negative baselines. | Medium/high: shifted prevalence can make F1 and AUPRC look better without a better model. |

## Dataset Inventory

| Dataset/config family | What data actually contains | Current loader representation | Current configured target |
| --- | --- | --- | --- |
| Synthetic/default | Generated worker time series with `ecg`, `eda`, `temp`, optional robot context, synthetic stress and comfort proxy. | One row per synthetic timestep. | `y_stress` binary synthetic risk; `y_comfort_proxy` synthetic comfort proxy. |
| WESAD native pickle | Wearable stress/affect protocol with baseline, stress, amusement and other transitions; chest signals include ECG, EDA, Temp, Resp, ACC. | One row per retained sample after truncating selected channels to a shared minimum length and filtering to labels 1/2/3. | `y_stress`: label 2 = 1, label 1 = 0, label 3 either 0 or excluded. `y_comfort_proxy`: fixed protocol proxy baseline 0.9, stress 0.2, amusement 0.7. |
| WESAD CSV exports | CSVs with schema columns or protocol labels, if no native pickles are found/selected. | One row per CSV row; optional downsample by row stride. | Same protocol mapping when `protocol_label` is present. |
| MultiPhysio-HRC | Multimodal HRC dataset with 256 Hz raw physiology and precomputed features. The repo uses `features/bio_features_60s.csv`, `features/labels.csv`, and task availability. | One row per 60-second bio feature row, sorted by task order/repetition/window and reindexed per participant. | `y_stress`: `NASA >= stress_threshold` (default 40). This is high NASA-TLX workload, not direct stress. `y_comfort_proxy`: SAM Valence normalized `(Valence - 1) / 4`. |

## What One Row Represents

| Dataset | One raw row in pipeline means | Consequence |
| --- | --- | --- |
| Synthetic | One generated timestep at configured `task.sampling_rate_hz`. | Config seconds are meaningful if synthetic generator semantics match the rate. |
| WESAD pickle | One retained native sample row after optional row-stride downsampling and optional `max_rows_per_subject` selection. | `time_idx` is a row counter, not seconds. Configs currently misreport real time. |
| WESAD CSV | One provided CSV row after optional row-stride downsampling. | Time semantics depend on external CSV export. |
| MultiPhysio | One 60-second precomputed feature window from `bio_features_60s.csv`. | A model window of length 10 means 10 feature windows, roughly 10 minutes if non-overlapping; it is not 10 raw samples. |

## Config Audit

Effective seconds below are the pipeline's current calculation: `task.sampling_rate_hz / dataset.downsample_factor` when a downsample factor exists. For WESAD native pickles, these reported seconds should be treated as unreliable because the base rate is not 4 Hz.

| Config | Dataset | Split | Models | Target | Effective Hz | Window | Horizon | Prediction stride | Profiles | Normalization |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `default.yaml` | synthetic | time | XGB + TFT by inherited defaults | synthetic `y_stress`, `y_comfort_proxy` | 4.0 | 15.0 s | 1.25 s | 0.25 s | on, static meta on | global |
| `synthetic_debug.yaml` | synthetic | time | XGB + TFT by inherited defaults | synthetic `y_stress`, comfort disabled | 4.0 | 7.5 s | 0.75 s | 0.25 s | off | global |
| `wesad_tiny.yaml` | WESAD | time | XGB + TFT | protocol stress vs baseline/amusement | 0.8 | 37.5 s reported | 3.75 s reported | 2.5 s reported | on, static meta on | global |
| `wesad_debug_tft.yaml` | WESAD | time | TFT only | protocol stress vs baseline/amusement | 0.2 | 300.0 s reported | 25.0 s reported | 25.0 s reported | on, static meta on | global |
| `wesad_kaggle_api.yaml` | WESAD | subject holdout | XGB + TFT | protocol stress vs baseline/amusement | 0.4 | 50.0 s reported | 2.5 s reported | 12.5 s reported | on, static meta off | online |
| `wesad_pilot_8subj.yaml` | WESAD | subject holdout | XGB + TFT | protocol stress vs baseline/amusement | 0.8 | 37.5 s reported | 3.75 s reported | 1.25 s reported | on, static meta off | online |
| `wesad_pilot_8subj_no_profiles.yaml` | WESAD | subject holdout | XGB + TFT | protocol stress vs baseline/amusement | 0.8 | 37.5 s reported | 3.75 s reported | 1.25 s reported | off | global |
| `wesad_paper_fast.yaml` | WESAD | subject holdout | XGB + TFT | protocol stress vs baseline/amusement | 0.4 | 50.0 s reported | 2.5 s reported | 12.5 s reported | on, static meta off | online |
| `wesad_paper_fast_no_profiles.yaml` | WESAD | subject holdout | XGB + TFT | protocol stress vs baseline/amusement | 0.4 | 50.0 s reported | 2.5 s reported | 12.5 s reported | off | online |
| `multiphysio_debug.yaml` | MultiPhysio | time | XGB only | `NASA >= 40`, Valence proxy | 0.0166667 | 480.0 s | 60.0 s | 60.0 s | off | global |
| `multiphysio_smoke.yaml` | MultiPhysio | subject holdout | XGB + TFT | `NASA >= 40`, comfort disabled | 0.0166667 | 360.0 s | 60.0 s | 60.0 s | off | global |
| `multiphysio_full.yaml` | MultiPhysio | subject holdout | XGB + TFT | `NASA >= 40`, Valence proxy | 0.0166667 | 600.0 s | 60.0 s | 60.0 s | off | global |

## Known Bugs And Scientific Risks

1. MultiPhysio `temp` is actually `EMG_RMSE`.
   - Source: `src/data/load_multiphysio.py` defaults `temp_col` to `EMG_RMSE`.
   - Impact: engineered features named `temp_mean`, `temp_std`, `temp_slope` are EMG-derived in MultiPhysio runs.

2. MultiPhysio `stress` is high NASA-TLX workload.
   - Source: `stress_label_col: NASA`, `stress_threshold: 40.0`.
   - Impact: calling it stress overstates the target. Use "NASA-TLX workload" or "workload/stress proxy".

3. WESAD time units are not trustworthy.
   - Source: WESAD loader creates `time_idx = np.arange(n)`, then config uses `sampling_rate_hz: 4.0` and physical row downsampling.
   - Impact: reported observation windows and horizons are not real seconds for native WESAD.

4. Downsampling semantics are conflated.
   - Source: WESAD loader applies row-stride downsampling; `run_experiment.py` divides the configured sampling rate again.
   - Impact: feature extraction and result summaries can use the wrong rate.

5. `max_rows_per_subject` is another sampling operation.
   - Source: WESAD loader uses `np.linspace` to reduce rows after downsample.
   - Impact: effective time spacing can become non-uniform and unreported.

6. Profile ablations are not clean.
   - Source: normalization is always applied in `_profile_transform`; profile flags control static/profile covariates inconsistently.
   - Impact: profile-on/off metric deltas may reflect normalization changes, not personalization features.

7. Built-in XGBoost ablation may be identical to the main XGBoost run.
   - Source: ablation reuses `X_all` already built from the current profile-normalized pipeline and only removes appended static profiles.
   - Impact: when `use_static_meta: false`, XGBoost profiles-on/off can have the same feature matrix.

8. TFT subject-holdout leakage risk.
   - Source: `worker_id` is a static categorical; unseen validation/test workers are added as artificial training rows.
   - Impact: the model and encoders are exposed to held-out subject IDs before evaluation.

9. TFT stride does not exactly match XGBoost window semantics.
   - Source: XGBoost builds windows with `window_step`; TFT first strides raw rows when `window_step > 1`, then lets `TimeSeriesDataSet` create adjacent windows.
   - Impact: same config can mean different observation content/counts across model families.

10. Metric interpretation is prevalence-sensitive.
   - Source: metrics include F1/AUPRC but no explicit dummy prevalence comparison.
   - Impact: weekly report `2026-04-16` shows large metric gains with a large test prevalence shift, so claims need rewording.

## Recommended Fixes In Priority Order

1. Rename MultiPhysio target and report text to high NASA-TLX workload; reserve "stress" for WESAD protocol stress or explicitly say proxy.
2. Stop mapping `EMG_RMSE` into `temp`; add a dataset-specific physiology schema or rename the third channel/features to `emg`.
3. Define canonical time units per dataset: native WESAD raw Hz, post-downsample Hz, MultiPhysio feature-window cadence.
4. Remove double downsample/rate adjustment by making `sampling_rate_hz` either raw input rate or post-loader effective rate, never both.
5. Make WESAD loader resample aligned channels explicitly or document that only same-rate chest signals are supported.
6. Remove `worker_id` from TFT static categoricals for subject-holdout, or use a true unknown-worker bucket without artificial val/test rows.
7. Rebuild profile ablation configs so only one variable family changes: normalization fixed, static/profile covariates toggled.
8. Add dummy baselines to `metrics.json`: prevalence AUPRC, always-positive F1, always-negative accuracy, validation/test prevalence.
9. Align XGBoost and TFT stride/window semantics and report each model's actual evaluated sample count.
10. Re-run WESAD and MultiPhysio baselines only after the above semantic fixes.
