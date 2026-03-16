# Cognitive Risk Forecasting Problem Definition

## Motivation
Human-robot co-working on construction sites can increase cognitive load and stress under heavy equipment operation and dynamic hazards. Predicting near-future risk enables a robot supervisor to adapt behavior before unsafe states escalate.

Reliable forecasting must use:
- multimodal physiology
- interaction/context features
- worker-specific personalization

## Objective
Formulate cognitive risk forecasting as a short-horizon probabilistic prediction task.

Given the past `T` timesteps of physiological and contextual covariates, estimate the probability that a worker enters a high-risk state at `t + delta_t`.

In this repository, labels are stress/comfort proxies derived from WESAD-like data.

## Data Assumptions
- Physiology: `ecg`, `eda`, `temp` (optional `resp`)
- Context: `distance_to_robot`, `robot_speed`, optional hazard/phase indicators
- Meta: worker attributes (experience, specialization)
- Time-series schema: `time_idx`, `worker_id`, covariates, targets

## Worker Personalization
Each worker has profile features that can be used as static covariates and normalization context.

Typical profile components:
```text
[experience_level, specialization_id,
 baseline_mu_signal..., baseline_sigma_signal...]
```

Profiles are used for:
1. static model features (optional)
2. leakage-safe normalization of physiological streams

## Leakage Safety Requirements
- Fit profile baselines on train split only.
- Apply train-fitted transformations to val/test.
- Preserve chronological split constraints per worker.
- Preserve disjoint subject sets when using subject-holdout mode.

## Modeling Scope
1. XGBoost baselines using engineered window features.
2. Temporal Fusion Transformer for sequence forecasting.
3. Calibration and threshold diagnostics for stress classification.

## Deliverables In This Repository
- Deterministic preprocessing and split/windowing pipeline.
- Config-driven training (`src/config/*.yaml`).
- Metrics and diagnostics written per run to `experiments/runs/<timestamp>/`.
- Plots and markdown summaries for comparison/reporting.
- Reproducible scripts for sanity checks, dataset debugging, and pilot runs.
