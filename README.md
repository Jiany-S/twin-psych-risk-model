# Cognitive Risk Forecasting

An end-to-end research template for cognitive risk forecasting in human–robot construction environments. The repository bootstraps preprocessing, worker profiling, Temporal Fusion Transformer (TFT) training, baseline comparisons, and calibration.

## Repository Layout
```
cdt-risk-forecasting/
├── data/
│   ├── raw/            # drop WESAD-like CSV files here (per worker)
│   └── processed/      # generated train.csv
├── docs/
│   ├── problem_definition.md
│   └── weekly_reports/2025-11-17.md
├── notebooks/
│   ├── 01_explore_data.ipynb
│   └── 02_tft_prototype.ipynb
├── src/
│   ├── data/           # loading & preprocessing
│   ├── profiles/       # worker profile store + normalization
│   ├── models/         # baselines + TFT utilities
│   └── training/       # CLI entry points
└── src/config/default.yaml
```

## Data Expectations
Place public physiological data (e.g., WESAD) under `data/raw/` as CSV files with at least these columns:

| Column | Description |
| --- | --- |
| `time_idx` | monotonically increasing integer index per worker (auto-generated if missing) |
| `worker_id` | unique worker identifier (inferred from filename if absent) |
| `target` | future stress/risk label in `[0, 1]` |
| `hr`, `hrv_rmssd`, `gsr` | physiological signals |
| `distance_to_robot`, `robot_speed`, `hazard_zone` | known covariates (auto-filled with zeros when omitted) |
| optional: `experience_level`, `specialization_id`, `safe_flag` |

The preprocessing stage writes `data/processed/train.csv` that already contains:
- normalized physiology per worker (personalized z-score),
- worker profile stats: `baseline_mu_*` / `baseline_sigma_*`,
- static categorical covariates: `worker_id`, `specialization_id`, `experience_level_bin`.

## Worker Profiles
`WorkerProfileStore` maintains per-worker exponential moving averages (EMA) for `hr`, `hrv_rmssd`, and `gsr`. Updates only consider samples flagged as safe (column `safe_flag`), or all samples when the flag is absent. For each worker we export:
```
[experience_level_bin, specialization_id,
 mu_hr, sigma_hr,
 mu_hrv, sigma_hrv,
 mu_gsr, sigma_gsr]
```
These vectors act as static covariates for TFT and provide personalized normalization when z-scoring physiology.

## Quickstart
1. **Preprocess data**
   ```bash
   python -m src.data.preprocess --config src/config/default.yaml
   ```
2. **Train baselines (logistic regression + GRU)**
   ```bash
   python -m src.training.train_baseline --config src/config/default.yaml
   ```
3. **Train TFT**
   ```bash
   python -m src.training.train_tft --config src/config/default.yaml
   ```
4. **Evaluate models + calibration**
   ```bash
   python -m src.training.evaluate --config src/config/default.yaml --model all --calibrate
   ```

## Configuration
All hyperparameters, column names, and paths live in `src/config/default.yaml`. Key entries:
- `paths.*`: raw/processed/artifact directories.
- `data.window_length` / `data.prediction_horizon`: encoder length `T` and forecasting horizon `Δt`.
- `profile.*`: EMA decay and safe flag column.
- `training.*`: batch size, learning rate, epochs, GPU count.
- `baseline.*`: feature columns for logistic/GRU, GRU hidden size/layers.
- `evaluation.ece_bins`: bins for expected calibration error.

## Baselines & TFT
- **Logistic Regression** uses tabular snapshots (physiology, robot context, static profile stats).
- **GRU Baseline** consumes sliding windows of the same features.
- **Temporal Fusion Transformer** (PyTorch Forecasting + Lightning) leverages:
  - static categorical covariates: worker id, specialization id, binned experience level,
  - static real covariates: worker baseline stats,
  - known real covariates: robot context,
  - observed real covariates: personalized physiological streams.

## Calibration & Metrics
`src/training/evaluate.py` reports AUROC, AUPRC, and Expected Calibration Error (ECE). Enable temperature scaling with `--calibrate` to fit a validation-based temperature parameter before scoring the test split. `src/training/calibrate.py` also exposes a standalone CLI for temperature scaling arbitrary prediction files.

## Notebooks
- `01_explore_data.ipynb`: inspect processed CSV, visualize profiles.
- `02_tft_prototype.ipynb`: interactively explore TFT hyper-parameters.

## Development Notes
- Python 3.10+, dependencies in `requirements.txt`.
- No external web calls—datasets must already be local.
- Logging is CLI-friendly via `src/utils/logging.py`.
- Random seeds handled via `src/utils/seed.py`.
