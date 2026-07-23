# Fast Detector Data Audit

Date: 2026-07-23

## Decision

The local MultiPhysio-HRC checkout is not suitable for training or validating a fast causal second-level detector. It contains derived feature tables and task/repetition questionnaire labels, but no local raw or filtered high-rate physiological signal files.

The fast detector implementation therefore uses WESAD raw physiology for software and physiological-state detection validation. It predicts WESAD protocol stress state at the end of a causal observation window. It must not be described as a real-time MultiPhysio industrial safety detector.

## Local MultiPhysio Files

| File | Local evidence | Temporal meaning | Fast-detector suitability |
| --- | --- | --- | --- |
| `features/bio_features_60s.csv` | Present; precomputed ECG/HRV, EDA, EMG, RESP features | One row is a 60-second feature interval | Not suitable for sub-second or second-level detection |
| `features/eeg_features_5s.csv` | Present; precomputed EEG features | One row is a 5-second feature interval | Not raw physiology; questionnaire labels are task-level, not per-5-second events |
| `features/aus_data.csv` | Present; facial action-unit feature table | Derived visual features | Not physiological raw-stream input |
| `features/speech_features.csv` | Present; includes speech feature rows and timestamps | Speech segment feature intervals | Not aligned physiological raw-stream input |
| `features/labels.csv` | Present; questionnaire labels by subject, class, repetition | Post-task self-report labels | Not second-level labels |
| `participants_task_overview.csv` | Present | Task availability metadata | No sample-level labels |
| `physiological_data/raw/` | Not present locally | Would be 256 Hz sensor data if available | Blocked |
| `physiological_data/filtered/` | Not present locally | Would be filtered sensor data if available | Blocked |

## Label Semantics

MultiPhysio labels are questionnaire outcomes after tasks:

| Target | Source | Meaning | Fast use |
| --- | --- | --- | --- |
| Stress | STAI-Y1 | State anxiety / stress questionnaire score | Slow task-level state estimation only |
| Cognitive workload | NASA-TLX | Workload questionnaire score | Slow workload estimation only |
| Comfort / valence | SAM Valence | Affective valence | Slow affect estimation only |
| Arousal | SAM Arousal | Affective activation | Slow affect estimation only |

These labels do not identify exact seconds when a risk state starts or ends. Assigning them to every second of a task would fabricate temporal precision.

## WESAD Fast Path

WESAD is used because the local data include raw pickle files with protocol labels and physiological signals. The fast path:

| Property | Value |
| --- | --- |
| Source sampling | WESAD chest signals at 700 Hz |
| Effective sampling | Configured via `stream.target_sampling_rate_hz` |
| Observation window | `fast_model.context_seconds` |
| Forecast horizon | `0.0` seconds in the smoke config |
| Inference stride | `fast_model.inference_stride_seconds` |
| Target | WESAD protocol stress state at causal window end |
| Excluded | Fabricated robot context, fabricated MultiPhysio second-level labels |

## Blockers Before MultiPhysio Fast Detection

1. Obtain raw or filtered MultiPhysio physiological signal files with recoverable timestamps.
2. Verify signal-task-label alignment at sample or interval level.
3. Define a scientifically valid event or state target with known onset/offset timing.
4. Define whether questionnaire labels are acceptable for slow state estimation only.
5. Re-run causality and split-leakage tests on the raw MultiPhysio loader before enabling `src/config/fast_multiphysio.yaml`.

