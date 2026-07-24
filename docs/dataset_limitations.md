# Dataset Limitations

## WESAD

WESAD contains experimentally induced stress and affect protocol segments. Labels are long contiguous protocol blocks such as baseline, stress, and amusement. The dataset has physiological signals, but no robot context, construction tasks, collision hazards, near-miss labels, intervention timestamps, or physical safety events.

WESAD results in this repository support physiological stress-state detection and protocol stress forecasting only. They do not validate construction danger prediction, collision prediction, or safety intervention effectiveness.

## Local MultiPhysio-HRC

The local MultiPhysio checkout contains feature tables and questionnaire labels. The benchmark path uses `features/bio_features_60s.csv`, where one row is a precomputed 60-second physiological feature interval. Labels are questionnaire/task-level measures: STAI-Y1, NASA-TLX, SAM Valence, and SAM Arousal.

The local checkout does not include raw or filtered high-rate physiology directories. It is unsuitable for second-level physiological detection, immediate safety intervention, or replay of raw HRC signals.

## Synthetic Replay

Synthetic physical and combined replay scenarios validate software logic only: priority ordering, stale-input handling, latching, reset behavior, and deterministic replay. They provide no empirical safety performance evidence.

