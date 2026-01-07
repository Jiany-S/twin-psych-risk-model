# Cognitive Risk Forecasting Problem Definition

## Motivation
Human–robot co-working on construction sites exposes workers to cognitive overload and stress while handling heavy equipment and unpredictable situations. Predicting near-future cognitive risk allows the robot supervisor to adjust task allocation, warn humans, or slow motion plans before unsafe states occur. Reliable forecasting must leverage multimodal physiology, interaction context, and personalized worker profiles because each individual reacts differently to workload and stressors.

## Objective
We frame cognitive risk forecasting as a short-horizon probabilistic prediction task. Given the past `T` timesteps of physiological signals and contextual covariates, we estimate the probability that a worker will enter a high-risk cognitive state at `t + Δt`. Labels are derived from public stress datasets (WESAD-like) and treated as risk proxies for pretraining.

## Data Assumptions
- **Physiology**: heart rate (hr), heart rate variability (hrv_rmssd), galvanic skin response (gsr).
- **Context**: worker-to-robot distance, robot speed, hazard zone indicators.
- **Meta**: worker profile attributes (experience, specialization).
- **Schema**: tabular time-series with `time_idx`, `worker_id`, covariates, and `target` in `[0, 1]`.

## Worker Personalization
Each worker owns a profile vector containing categorical attributes (experience level bucket, specialization id) and continuous baseline stats derived from an exponential moving average (EMA) of "safe" samples. The profile:
```
[experience_level, specialization_id,
 mu_hr, sigma_hr,
 mu_hrv, sigma_hrv,
 mu_gsr, sigma_gsr]
```
Profiles are used twice:
1. **Static covariates** for the Temporal Fusion Transformer.
2. **Normalization context** for per-worker z-score scaling of physiological streams.

## Modeling Scope
1. **Baselines**: logistic regression on summary statistics, gated recurrent unit (GRU) forecaster on sequences.
2. **Temporal Fusion Transformer**: multi-horizon forecasting with static covariates (worker profile), known future inputs (robot context), and observed inputs (physiology).
3. **Calibration**: Expected Calibration Error (ECE) plus optional temperature scaling fitted on validation predictions.

## Deliverables
- Deterministic preprocessing pipeline that transforms raw WESAD-like data stored under `data/raw/` into a normalized `data/processed/train.csv`.
- Training scripts for baselines and TFT models with YAML-driven hyperparameters.
- Evaluation utilities that report AUROC, AUPRC, and calibration metrics.
- Documentation and notebooks describing exploratory analysis and TFT prototyping steps.
