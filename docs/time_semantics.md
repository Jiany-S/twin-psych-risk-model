# Time Semantics

Date: 2026-07-22

The pipeline now treats time as duration first. Configs should define stream representation and task durations, then the code converts seconds to rows in one place: `src/data/time_semantics.py`.

## Configuration

Raw signal streams:

```yaml
stream:
  representation: raw_signal
  source_sampling_rate_hz: 700.0
  target_sampling_rate_hz: 4.0

task:
  context_seconds: 37.5
  forecast_horizon_seconds: 3.75
  inference_stride_seconds: 2.5
```

Precomputed feature tables:

```yaml
stream:
  representation: precomputed_features
  row_interval_seconds: 60

task:
  context_seconds: 480
  forecast_horizon_seconds: 60
  inference_stride_seconds: 60
```

The centralized conversion is:

| Field | Conversion |
| --- | --- |
| `context_seconds` | `round(context_seconds * effective_rate_hz)` |
| `forecast_horizon_seconds` | `round(forecast_horizon_seconds * effective_rate_hz)` |
| `inference_stride_seconds` | `round(inference_stride_seconds * effective_rate_hz)` |

The effective rate is `stream.target_sampling_rate_hz` for raw signals and `1 / stream.row_interval_seconds` for precomputed feature tables.

## WESAD Example

Native WESAD chest signals are treated as 700 Hz source streams. The loader now resamples each raw signal explicitly to `stream.target_sampling_rate_hz` using real timestamps:

| Signal | Source | Source rate used |
| --- | --- | ---: |
| ECG | chest ECG, wrist BVP fallback | 700 Hz chest, 64 Hz BVP fallback |
| EDA | chest EDA, wrist EDA fallback | 700 Hz chest, 4 Hz wrist fallback |
| Temp | chest Temp, wrist TEMP fallback | 700 Hz chest, 4 Hz wrist fallback |
| Resp | chest Resp | 700 Hz |
| ACC | chest ACC, wrist ACC fallback | 700 Hz chest, 32 Hz wrist fallback |

For this config:

```yaml
stream:
  representation: raw_signal
  source_sampling_rate_hz: 700.0
  target_sampling_rate_hz: 4.0
task:
  context_seconds: 37.5
  forecast_horizon_seconds: 3.75
  inference_stride_seconds: 2.5
```

The runner derives:

| Quantity | Value |
| --- | ---: |
| effective sampling rate | 4 Hz |
| row interval | 0.25 s |
| context rows | 150 |
| horizon rows | 15 |
| stride rows | 10 |
| expected prediction cadence | every 2.5 s |

The label timestamp must equal the last observation timestamp plus `forecast_horizon_seconds`. Windows are not allowed to cross timestamp gaps created by protocol filtering.

## MultiPhysio Example

The current MultiPhysio path uses `features/bio_features_60s.csv`. This is not a raw physiological stream. One row represents one 60-second precomputed feature interval.

For this config:

```yaml
stream:
  representation: precomputed_features
  row_interval_seconds: 60
task:
  context_seconds: 480
  forecast_horizon_seconds: 60
  inference_stride_seconds: 60
```

The runner derives:

| Quantity | Value |
| --- | ---: |
| effective row rate | 1/60 Hz |
| row interval | 60 s |
| context rows | 8 |
| horizon rows | 1 |
| stride rows | 1 |
| expected prediction cadence | every 60 s |

This representation does not support sub-second detection or 5-second detection. A 5-second inference stride is invalid unless the input representation is changed to a table with rows at 5-second cadence or to raw signals resampled at an appropriate rate.

## Run Artifacts

Each run now stores:

| Artifact | Contents |
| --- | --- |
| `time_semantics.json` | effective sampling rate, row interval, configured durations, derived row counts, expected prediction cadence |
| `metrics.json.time_semantics` | same timing block embedded in metrics |
| `processed/meta.csv` | per-window start/end/label timestamps and configured durations |

## Guardrails

1. `dataset.downsample_factor` is deprecated for WESAD pickle loading. Use `stream.target_sampling_rate_hz`.
2. `max_rows_per_subject` is only a contiguous row cap for debug runs. It is not resampling.
3. Raw ECG/HRV feature extraction rejects contexts shorter than 4 seconds.
4. Precomputed feature tables reject inference strides shorter than their row interval.
5. Window overlap is derived from `inference_stride_seconds`, not from an independent row-count field.
