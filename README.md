# Personalized Multi-Rate Human-State Forecasting for Human-Robot Collaboration

This repository studies physiological stress and workload modeling, benchmark hierarchy, and multi-rate supervisory replay logic for human-robot collaboration research. It does not provide safety certification, ISO compliance, construction deployment validation, hardware actuation, collision prediction, or incident-reduction evidence.

## What Is Validated

| Area | Status | Limits |
| --- | --- | --- |
| MultiPhysio benchmark | Capped grouped subject-CV smoke over 60-second physiological feature rows | Questionnaire/task-level labels; no local raw physiology. |
| WESAD fast detector | Held-out-subject current protocol-stress detection | WESAD stress labels are not construction danger labels. |
| WESAD slow forecaster | 5s and 30s protocol-stress forecasting smoke | Protocol blocks are long; horizons are often label-identical. |
| Replay engine | Timestamp-driven saved-prediction, oracle, synthetic physical, and combined simulations | Synthetic modes validate software logic only. |
| Physical kernel | Deterministic configurable safety logic | Not ISO compliant and not connected to hardware. |

## Datasets And Targets

| Dataset | Local representation | Targets |
| --- | --- | --- |
| WESAD | Raw pickle physiology resampled explicitly from source rates | Protocol stress: baseline/amusement = 0, stress = 1 when configured. |
| MultiPhysio-HRC | `bio_features_60s.csv` precomputed 60-second features | STAI-Y1 stress, NASA-TLX workload, SAM Valence, SAM Arousal. |
| Synthetic replay | Deterministic scenario rows | Software-state expectations only. |

Correct modality mapping:
- WESAD `ecg`, `eda`, `temp` are chest signals where available.
- MultiPhysio uses `hrv_mean_nn`, `eda_mean`, `emg_rmse`, and `rrv_mean_bb`; EMG is not temperature.
- NASA-TLX is workload, not direct stress.

## Time Semantics

Use duration-based configuration. Forecast horizon and inference stride are independent.

```yaml
stream:
  representation: raw_signal
  source_sampling_rate_hz: 700
  target_sampling_rate_hz: 4

task:
  context_seconds: 30
  forecast_horizon_seconds: 5
  inference_stride_seconds: 1
```

For MultiPhysio feature tables:

```yaml
stream:
  representation: precomputed_features
  row_interval_seconds: 60
```

## Benchmark Hierarchy

Tabular benchmark models:
- Dummy baselines
- Logistic Regression
- Random Forest
- AdaBoost where configured
- XGBoost as the strong boosted-tree benchmark

Neural sequence models:
- Fast TCN for current-state WESAD stress detection
- TFT-family slow multi-horizon forecaster for WESAD protocol stress forecasting

Do not compare fast current-state detection and slow forecasting as the same task.

## Personalization And Calibration

Normalization and profiles are separate:

```yaml
normalization:
  mode: global  # global | calibration

profiles:
  include_calibration_features: false
  include_role_metadata: false
  include_experience_metadata: false
```

Subject calibration uses explicit baseline/rest segments only. Synthetic role or experience metadata is not generated for real-data runs.

## Multi-Rate Supervisory Architecture

Layers:
1. Deterministic physical kernel
2. Fast detector adapter
3. Slow multi-horizon forecaster adapter
4. Supervisory state machine

Authority order:

```text
Hard physical emergency
  > physical protective rule
  > fast acute physiological/context detector
  > slow physiological forecast
  > normal operation
```

Physiological ML cannot request emergency stop. The replay engine emits recommendations/actions only; it does not actuate hardware.

## Replay Modes

| Config | Result category |
| --- | --- |
| `src/config/streaming_physiological.yaml` | `real_model_physiological_replay` using saved held-out prediction artifacts |
| `src/config/streaming_physiological_oracle.yaml` | `oracle_label_policy_simulation` |
| `src/config/streaming_synthetic_physical.yaml` | `synthetic_physical_safety_simulation` |
| `src/config/streaming_multirate.yaml` | `combined_integration_simulation` |

Oracle-label and synthetic simulations are not empirical model validation.

## Reproduction Commands

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python scripts/sanity_check.py

python scripts/run_multiphysio_cv.py --config src/config/multiphysio_cv.yaml
python -m src.training.fast_train --config src/config/fast_wesad.yaml
python -m src.training.tft_train --config src/config/slow_tft.yaml

python -m src.streaming.replay --config src/config/streaming_physiological.yaml
python -m src.streaming.replay --config src/config/streaming_physiological_oracle.yaml
python -m src.streaming.replay --config src/config/streaming_synthetic_physical.yaml
python -m src.streaming.replay --config src/config/streaming_multirate.yaml
```

Optional lint:

```bash
ruff check src scripts tests
```

## Key Documentation

- `docs/dataset_limitations.md`
- `docs/model_comparison.md`
- `docs/validated_claims.md`
- `docs/final_architecture.md`
- `docs/multirate_architecture.md`
- `docs/reproducibility_checklist.md`
- `docs/research_roadmap.md`

## Limitations

WESAD has experimentally induced stress protocol labels, not construction hazards. Local MultiPhysio lacks raw/filtered high-rate physiology. Neither dataset contains complete robot distance, velocity, stopping-distance, construction near-miss labels, or physical intervention outcomes. Synthetic replay validates software invariants only.

