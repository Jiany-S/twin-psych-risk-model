# Current Pipeline Audit

Audit date: 2026-07-20

Scope: `src/run_experiment.py`, dataset loaders, windowing/features, worker profiles, XGBoost/TFT training, WESAD and MultiPhysio configs, tracked weekly reports. No architectural changes were made.

## Executive Findings

| Area | Finding | Risk |
| --- | --- | --- |
| MultiPhysio target semantics | Phase 2 corrected this: STAI maps to stress/state anxiety, NASA-TLX maps to cognitive workload, SAM Valence maps to comfort proxy, and SAM Arousal maps to arousal. | Remaining risk: historical runs before Phase 2 mislabeled NASA-TLX workload as stress. |
| MultiPhysio physiology mapping | Phase 2 corrected this: `EMG_RMSE` maps to `emg_rmse`; no `temp` column is fabricated for MultiPhysio. | Remaining risk: historical runs before Phase 2 mislabeled EMG as temperature. |
| WESAD time base | Prompt 5 refactored this: native WESAD signals are explicitly resampled from real source rates to `stream.target_sampling_rate_hz`; timestamps are seconds. | Remaining risk: historical artifacts before Prompt 5 used incorrect reported seconds. |
| WESAD downsampling | Prompt 5 removed `dataset.downsample_factor` from configs and no longer divides sampling rate twice. | Remaining risk: old configs/artifacts with downsample factors should not be cited. |
| MultiPhysio temporal rows | Each row is already a 60-second precomputed feature window. The pipeline windows those feature rows again. | Medium/high: configured windows are windows of 60-second summaries, not raw physiological windows. |
| Profile ablations | Prompt 4 refactored this: normalization, calibration, and profile-vector inputs are independently configured. | Remaining risk: old artifacts before Prompt 4 should not be used for profile claims. |
| TFT subject holdout | Prompt 4 removed `worker_id` from TFT static covariates and removed artificial validation/test worker rows. | Remaining risk: rerun larger subject-holdout TFT experiments to verify unseen-group behavior and prevalence stability. |
| Held-out subject validity | XGBoost and TFT subject sets are disjoint at raw split level; Prompt 4 removes direct held-out ID/profile leakage. | Medium: subject-level prevalence/task composition can still dominate. |
| Metric comparability | Weekly report `2026-04-16` compares runs where test prevalence changed from 25/135 to 96/173 positives. | High: F1/AUPRC/calibration changes cannot be interpreted as pure model improvement. |
| Dummy prevalence baselines | Metrics include prevalence and class counts but do not explicitly compare AUPRC to prevalence or F1 to trivial always-positive/always-negative baselines. | Medium/high: shifted prevalence can make F1 and AUPRC look better without a better model. |

## Dataset Inventory

| Dataset/config family | What data actually contains | Current loader representation | Current configured target |
| --- | --- | --- | --- |
| Synthetic/default | Generated worker time series with `ecg`, `eda`, `temp`, optional robot context, synthetic stress and comfort proxy. | One row per synthetic timestep. | `y_stress` binary synthetic risk; `y_comfort_proxy` synthetic comfort proxy. |
| WESAD native pickle | Wearable stress/affect protocol with baseline, stress, amusement and other transitions; chest signals include ECG, EDA, Temp, Resp, ACC. | One row per retained sample after truncating selected channels to a shared minimum length and filtering to labels 1/2/3. | `y_stress`: label 2 = 1, label 1 = 0, label 3 either 0 or excluded. `y_comfort_proxy`: fixed protocol proxy baseline 0.9, stress 0.2, amusement 0.7. |
| WESAD CSV exports | CSVs with schema columns or protocol labels, if no native pickles are found/selected. | One row per CSV row; optional downsample by row stride. | Same protocol mapping when `protocol_label` is present. |
| MultiPhysio-HRC | Multimodal HRC dataset with 256 Hz raw physiology and precomputed features. The repo uses `features/bio_features_60s.csv`, `features/labels.csv`, and task availability. | One row per 60-second bio feature row, sorted by task order/repetition/window and reindexed per participant. | `y_stress`: normalized STAI-Y1 state anxiety. `y_cognitive_load`: normalized NASA-TLX workload. `y_cognitive_load_binary`: NASA-TLX >= 40 only when explicitly configured. `y_comfort_proxy`: normalized SAM Valence. `y_arousal`: normalized SAM Arousal. |

## Target Table

| Dataset | Target | Source | Interpretation |
| --- | --- | --- | --- |
| WESAD | stress | protocol label | experimentally induced stress |
| MultiPhysio | stress | STAI-Y1 (`STAI`) | self-reported state anxiety |
| MultiPhysio | cognitive load | NASA-TLX (`NASA`) | perceived workload |
| MultiPhysio | comfort proxy | SAM Valence (`Valence`) | affective valence |
| MultiPhysio | arousal | SAM Arousal (`Arousal`) | affective activation |

## What One Row Represents

| Dataset | One raw row in pipeline means | Consequence |
| --- | --- | --- |
| Synthetic | One generated timestep at configured `task.sampling_rate_hz`. | Config seconds are meaningful if synthetic generator semantics match the rate. |
| WESAD pickle | One retained native sample row after optional row-stride downsampling and optional `max_rows_per_subject` selection. | `time_idx` is a row counter, not seconds. Configs currently misreport real time. |
| WESAD CSV | One provided CSV row after optional row-stride downsampling. | Time semantics depend on external CSV export. |
| MultiPhysio | One 60-second precomputed feature window from `bio_features_60s.csv`. | A model window of length 10 means 10 feature windows, roughly 10 minutes if non-overlapping; it is not 10 raw samples. |

## Config Audit

Effective seconds below were recorded during the original audit before Prompt 5. Current configs use `stream.*` and duration-based `task.*` fields; see `docs/time_semantics.md`.

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
| `multiphysio_debug.yaml` | MultiPhysio | time | XGB only | primary `cognitive_load_binary` from NASA-TLX >= 40 | 0.0166667 | 480.0 s | 60.0 s | 60.0 s | off | global |
| `multiphysio_smoke.yaml` | MultiPhysio | subject holdout | XGB + TFT | primary `cognitive_load_binary` from NASA-TLX >= 40 | 0.0166667 | 360.0 s | 60.0 s | 60.0 s | off | global |
| `multiphysio_full.yaml` | MultiPhysio | subject holdout | XGB + TFT | primary `cognitive_load_binary` from NASA-TLX >= 40 | 0.0166667 | 600.0 s | 60.0 s | 60.0 s | off | global |

## Known Bugs And Scientific Risks

1. Historical MultiPhysio runs before Phase 2 used `EMG_RMSE` as `temp`.
   - Current status: fixed in code/config; MultiPhysio now exposes `emg_rmse`.
   - Impact: old reports and artifacts must not be interpreted as temperature-based physiology.

2. Historical MultiPhysio runs before Phase 2 used `NASA >= 40` as `y_stress`.
   - Current status: fixed in code/config; NASA-TLX is now `cognitive_load`/`cognitive_load_binary`, and STAI is `stress`.
   - Impact: old reports and artifacts must be relabeled as workload benchmarks.

3. Historical WESAD time units before Prompt 5 are not trustworthy.
   - Current status: Prompt 5 uses explicit resampling and duration-based windows.
   - Impact: old reported observation windows and horizons are not real seconds for native WESAD.

4. Historical downsampling semantics were conflated.
   - Current status: Prompt 5 deprecates `dataset.downsample_factor`; `stream.target_sampling_rate_hz` is the single resampling control.
   - Impact: old feature extraction and result summaries can use the wrong rate.

5. `max_rows_per_subject` is no longer treated as resampling.
   - Current status: Prompt 5 uses only a contiguous debug cap after explicit resampling.
   - Impact: old `np.linspace`-capped artifacts can have non-uniform and unreported spacing.

6. Historical profile ablations before Prompt 4 are not clean.
   - Current status: Prompt 4 separates `normalization.mode` from optional profile-vector flags.
   - Impact: old profile-on/off metric deltas may reflect normalization changes, not personalization features.

7. Historical built-in XGBoost ablation could be identical to the main XGBoost run.
   - Current status: Prompt 4 tests prove profile-on and profile-off matrices differ when profile columns are enabled.
   - Impact: old `use_static_meta: false` ablations should not be cited as profile evidence.

8. Historical TFT subject-holdout leakage risk.
   - Current status: Prompt 4 removes `worker_id` from static categoricals and removes artificial validation/test worker rows.
   - Impact: larger subject-holdout TFT runs still need to be rerun before making generalization claims.

9. TFT stride does not exactly match XGBoost window semantics.
   - Source: XGBoost builds windows with `window_step`; TFT first strides raw rows when `window_step > 1`, then lets `TimeSeriesDataSet` create adjacent windows.
   - Impact: same config can mean different observation content/counts across model families.

10. Metric interpretation is prevalence-sensitive.
   - Source: metrics include F1/AUPRC but no explicit dummy prevalence comparison.
   - Impact: weekly report `2026-04-16` shows large metric gains with a large test prevalence shift, so claims need rewording.

## Recommended Fixes In Priority Order

1. Define canonical time units per dataset: native WESAD raw Hz, post-downsample Hz, MultiPhysio feature-window cadence.
2. Remove double downsample/rate adjustment by making `sampling_rate_hz` either raw input rate or post-loader effective rate, never both.
3. Make WESAD loader resample aligned channels explicitly or document that only same-rate chest signals are supported.
4. Rerun TFT subject-holdout after the Prompt 4 worker-ID leakage fix and verify unseen-group behavior.
5. Rebuild profile ablation configs so only one variable family changes: normalization fixed, static/profile covariates toggled.
6. Add dummy baselines to `metrics.json`: prevalence AUPRC, always-positive F1, always-negative accuracy, validation/test prevalence.
7. Align XGBoost and TFT stride/window semantics and report each model's actual evaluated sample count.
8. Re-run WESAD and MultiPhysio baselines after the remaining semantic and evaluation fixes.
