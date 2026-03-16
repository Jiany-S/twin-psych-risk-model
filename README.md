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
- `dataset.name`: `wesad | synthetic | csv`
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
  kaggle_cache_dir: data/raw/wesad_kaggle
  format: auto
```
Credentials are required via `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`.

Config behavior:
- `src/config/default.yaml` is the base config.
- Passing `--config <other.yaml>` applies a deep override on top of defaults.

## Targets
- Stress classification: `y_stress`
- Comfort regression proxy: `y_comfort_proxy`

Multi-head training is enabled by default (`targets.multi_head.enabled: true`).

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
