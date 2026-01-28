# Risk Forecasting with XGBoost + TFT

This repository trains and compares two short-horizon risk forecasting models on multimodal time-series:
1) XGBoost baseline with engineered window features
2) Temporal Fusion Transformer (TFT) using PyTorch Forecasting

The pipeline runs end-to-end with a single command and automatically generates a realistic synthetic dataset if raw data is missing.

## One-command run
```bash
python -m src.run_experiment --config src/config/default.yaml
```

Artifacts are saved under `experiments/runs/<timestamp>/` including metrics, plots, and a markdown report.

## Data schema
Place a single CSV at `data/raw/data.csv` or multiple CSVs under `data/raw/`. Minimum columns:
- `time_idx` (int, increasing per worker)
- `worker_id` (str/int)
- `target` (0/1 or float in [0,1])
- `hr`, `hrv_rmssd`, `gsr`
- `distance_to_robot`, `robot_speed`

Optional columns (auto-filled if missing):
- `hazard_zone` (int/cat, default 0)
- `task_phase` (cat, default "default")

If raw data is absent, synthetic data is generated and stored under `data/raw/`.

## Worker profiles
Per-worker baseline stats (mu/sigma) are tracked using EMA and used by both models.
Static features:
```
experience_level (binned), specialization_id,
mu_hr, sigma_hr, mu_hrv, sigma_hrv, mu_gsr, sigma_gsr
```

Profiles are used as:
- TFT static covariates (categorical + real)
- XGBoost features appended to window-engineered features

## Switching classification vs regression
In `src/config/default.yaml` set:
- `task.task_type: classification` or `regression`
- `task.risk_threshold` (for binarizing continuous targets when classification)

## Outputs
`experiments/runs/<timestamp>/`
- `metrics.json` (per-model metrics + comparison)
- `results.md` (summary report)
- `plots/*.png` (ROC, PR, calibration, confusion matrix, time-series, feature importance)
- model artifacts (XGBoost model, TFT checkpoint)

## Notes
- The default config is CPU-friendly and finishes quickly on a synthetic dataset.
- All splits are chronological within each worker to avoid leakage.
