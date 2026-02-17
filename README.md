# Twin Psych Risk Model: XGBoost vs TFT

This repo trains and compares XGBoost and Temporal Fusion Transformer (TFT) for short-horizon risk forecasting from multimodal time-series, with leak-safe worker personalization.

## Run
```bash
python -m src.run_experiment --config src/config/default.yaml
```

Artifacts are saved under `experiments/runs/<timestamp>/`:
- `metrics.json`
- `results.md`
- `config_resolved.yaml`
- `models/*`
- `plots/*.png`
- `processed/windows_*.npy`, `processed/meta.csv`, `processed/splits.csv`, `processed/tft_flat.csv`

## Datasets
Configured via `dataset` in `src/config/default.yaml`.
- `dataset.name: wesad | synthetic | csv`
- `dataset.path`: local path to dataset root/file
- `dataset.format: auto | wesad_pickle | csv`

WESAD loader supports:
- native `S*/S*.pkl`
- CSV exports

Unified columns include:
- `timestamp`, `time_idx`, `worker_id`
- `ecg`, `eda`, `temp` (optional `resp`)
- `distance_to_robot`, `robot_speed`, `hazard_zone`, `task_phase`
- `protocol_label`, `y_stress`, `y_comfort_proxy`

## Targets
- Stress classification: `y_stress`
- Comfort regression proxy: `y_comfort_proxy`

Default is multi-head enabled (both tasks in one run).

## Leakage-safe profiles
Worker EMA baselines are fit on **train split only**:
1. split chronologically per worker
2. fit profiles on train only
3. transform train/val/test using train-fitted profiles

You can disable profile features using:
```yaml
profiles:
  enabled: false
```

## Window and horizon controls
```yaml
task:
  window_length: 60
  horizon_steps: 5
  sampling_rate_hz: 4.0
  window_step: 1
```

## Notes
- Run is CPU-friendly by default (`tft.max_epochs: 3` + early stopping).
- If one model fails, the run still completes and records the error in `metrics.json`.
- Profile ablation can be toggled via `models.run_ablation_profiles`; it is skipped automatically when `models.run_xgb: false`.

## WESAD subject-holdout pilot (15-subject default)
Prepare the local subset from Kaggle-hosted WESAD (default list matches the pilot config):
```bash
python scripts/prepare_wesad_subset.py --subjects "S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S13,S14,S15,S16,S17"
```

Run pilot with profiles enabled:
```bash
python -m src.run_experiment --config src/config/wesad_pilot_8subj.yaml
```

Run pilot with profiles disabled:
```bash
python -m src.run_experiment --config src/config/wesad_pilot_8subj_no_profiles.yaml
```

Generate a paper-ready markdown summary for any run:
```bash
python scripts/make_paper_summary.py --run_dir experiments/runs/<timestamp>
```

Convenience scripts:
- `scripts/run_wesad_pilot.sh`
- `scripts/run_wesad_pilot.bat`
  - both scripts now also generate `paper_summary.md` for the two latest pilot runs.
