# Twin Psych Risk Model: XGBoost vs TFT

This repository trains and compares XGBoost and Temporal Fusion Transformer (TFT) for short-horizon cognitive risk forecasting from multimodal time series, with leakage-safe worker personalization.

## Quick Start

### 1) Environment
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Fast sanity check
```bash
python scripts/sanity_check.py
```

### 3) End-to-end run
```bash
python -m src.run_experiment --config src/config/default.yaml
```

## Run Artifacts
Each run writes to `experiments/runs/<timestamp>/`:
- `metrics.json`
- `results.md`
- `config_resolved.yaml`
- `profile_feature_stats.json`
- `numeric_feature_stats.json`
- `engineered_feature_stats.json`
- `models/*`
- `plots/*.png`
- `processed/windows_*.npy`, `processed/meta.csv`, `processed/splits.csv`, `processed/tft_flat.csv`

## Dataset Configuration
Configured under `dataset` in `src/config/default.yaml`.
- `dataset.name`: `wesad | multiphysio | synthetic | csv`
- `dataset.source`: `local | kaggle_api` (WESAD only)
- `dataset.path`: local dataset path for `source: local`
- `dataset.format`: `auto | wesad_pickle | csv`
- `dataset.kaggle_dataset`: Kaggle dataset id for `source: kaggle_api`
- `dataset.kaggle_cache_dir`: extraction/cache location for Kaggle downloads

WESAD loader supports:
- native `S*/S*.pkl`
- CSV exports

Kaggle API mode example:
```yaml
dataset:
  name: wesad
  source: kaggle_api
  kaggle_dataset: orvile/wesad-wearable-stress-affect-detection-dataset
  kaggle_cache_dir: data/wesad/wesad_kaggle
  format: auto
```
Credentials are required via `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`.

MultiPhysio loader example:
```yaml
dataset:
  name: multiphysio
  path: data/multiphysio
  report_name: MultiPhysio-HRC cognitive workload benchmark
  multiphysio:
    feature_columns:
      hrv_mean_nn: HRV_MeanNN
      eda_mean: EDA_mean
      emg_rmse: EMG_RMSE
      rrv_mean_bb: RRV_MeanBB

features:
  physiology: [hrv_mean_nn, eda_mean, emg_rmse, rrv_mean_bb]
  engineering_mode: precomputed

experiment:
  primary_target: cognitive_load_binary

targets:
  stress:
    source_col: STAI
    source_questionnaire: STAI-Y1
    source_range: [20, 80]
    task_type: regression
  cognitive_load_binary:
    source_col: NASA
    source_questionnaire: NASA-TLX
    source_range: [0, 100]
    task_type: classification
    threshold: 40.0
  comfort:
    source_col: Valence
    source_questionnaire: SAM Valence
    source_range: [1, 5]
    task_type: regression
  arousal:
    source_col: Arousal
    source_questionnaire: SAM Arousal
    source_range: [1, 5]
    task_type: regression
```

MultiPhysio `bio_features_60s.csv` contains precomputed 60-second features, not raw physiological waveforms. NASA-TLX is modeled as cognitive workload; it is not silently used as stress. EMG remains `emg_rmse` and is not mapped to temperature.

Config behavior:
- `src/config/default.yaml` is the base config.
- Passing `--config <other.yaml>` applies a deep override on top of defaults.

## Targets
- Configured primary classification target: `experiment.primary_target`
- WESAD stress classification: `y_stress`
- MultiPhysio STAI stress/state-anxiety regression: `y_stress`
- MultiPhysio cognitive workload: `y_cognitive_load`, with explicit binary form `y_cognitive_load_binary`
- Comfort regression proxy: `y_comfort_proxy`

Multi-head training is enabled by default (`targets.multi_head.enabled: true`).

| Dataset | Target | Source | Interpretation |
| --- | --- | --- | --- |
| WESAD | stress | protocol label | experimentally induced stress |
| MultiPhysio | stress | STAI-Y1 (`STAI`) | self-reported state anxiety |
| MultiPhysio | cognitive load | NASA-TLX (`NASA`) | perceived workload |
| MultiPhysio | comfort proxy | SAM Valence (`Valence`) | affective valence |
| MultiPhysio | arousal | SAM Arousal (`Arousal`) | affective activation |

## Leakage-Safe Profile Handling
Worker EMA baselines are fit on train split only:
1. Split chronologically per worker (or disjoint subject holdout).
2. Fit profile baselines using train data only.
3. Transform train/val/test with train-fitted statistics.

Disable profile features via:
```yaml
profiles:
  enabled: false
```

## Core Controls
```yaml
task:
  window_length: 60
  horizon_steps: 5
  sampling_rate_hz: 4.0
```

## Benchmark Models
The primary classification target is evaluated with a benchmark hierarchy:

```yaml
models:
  run_dummy: true
  run_logistic: true
  run_random_forest: true
  run_xgb: true
  run_tft: true

dummy:
  strategies: [most_frequent, stratified]

logistic_regression:
  class_weight: balanced
  calibration: none  # none | platt

random_forest:
  class_weight: balanced
```

All tabular classifiers use the same engineered windows/features, train-fitted imputation, and validation-only threshold selection. XGBoost remains the strong boosted-tree benchmark. Logistic regression uses train-only scaling, and optional Platt calibration is fit on validation probabilities only.

## WESAD Pilot (8 Subjects)
Prepare a local subset:
```bash
python scripts/prepare_wesad_subset.py --subjects "S2,S3,S4,S5,S6,S7,S8,S9"
```

Run profiles on/off:
```bash
python -m src.run_experiment --config src/config/wesad_pilot_8subj.yaml
python -m src.run_experiment --config src/config/wesad_pilot_8subj_no_profiles.yaml
```

Convenience wrappers:
- `scripts/run_wesad_pilot.sh`
- `scripts/run_wesad_pilot.bat`

## Data Layout
- `data/wesad/`: WESAD data (`wesad_subset/`, optional `wesad_kaggle/`)
- `data/multiphysio/`: MultiPhysio-HRC docs and feature tables
- `data/raw/`: synthetic/default raw CSV fallback

## MultiPhysio Debug
```bash
python scripts/debug_multiphysio_dataset.py --config src/config/multiphysio_debug.yaml
python -m src.run_experiment --config src/config/multiphysio_debug.yaml
```

## MultiPhysio Smoke (Both Models)
Run a small subject-holdout smoke test that asserts both XGBoost and TFT complete:
```bash
python scripts/run_multiphysio_smoke.py --config src/config/multiphysio_smoke.yaml
```

Convenience wrappers:
- `scripts/run_multiphysio_smoke.sh`
- `scripts/run_multiphysio_smoke.bat`

## MultiPhysio Full Training
Run full training with deterministic subject-holdout split construction (both models):
```bash
python scripts/run_multiphysio_full.py --config src/config/multiphysio_full.yaml
```

Convenience wrappers:
- `scripts/run_multiphysio_full.sh`
- `scripts/run_multiphysio_full.bat`

## Documentation
- Agent operating contract: `AGENTS.md`
- Problem framing: `docs/problem_definition.md`
- Experiment workflow and improvement loop: `docs/experimentation_workflow.md`
- Historical run notes: `docs/weekly_reports/`

## Code Quality
Optional linting with Ruff:
```bash
pip install ruff
ruff check src scripts
```
Project lint settings live in `pyproject.toml`.

## Notes
- Default config is CPU-friendly (`tft.max_epochs: 3` with early stopping).
- If one model fails, the pipeline still completes and records the failure in `metrics.json`.
