"""Windowing, chronological split, and feature engineering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .features import extract_precomputed_window_features, extract_window_features
from .schema import DataSchema


@dataclass
class WindowedData:
    X_windows: np.ndarray
    y_stress: np.ndarray
    y_comfort: np.ndarray
    meta: pd.DataFrame


def create_time_splits(
    df: pd.DataFrame,
    schema: DataSchema,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train/val/test ratios must sum to 1.")
    train_rows: list[pd.DataFrame] = []
    val_rows: list[pd.DataFrame] = []
    test_rows: list[pd.DataFrame] = []
    split_manifest: list[pd.DataFrame] = []

    for worker_id, worker_df in df.groupby(schema.worker_id, observed=True):
        worker_df = worker_df.sort_values(schema.time_idx).reset_index(drop=True)
        n = len(worker_df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        train_part = worker_df.iloc[:train_end].copy()
        val_part = worker_df.iloc[train_end:val_end].copy()
        test_part = worker_df.iloc[val_end:].copy()
        train_rows.append(train_part)
        val_rows.append(val_part)
        test_rows.append(test_part)
        for split_name, split_df in (("train", train_part), ("val", val_part), ("test", test_part)):
            split_manifest.append(
                pd.DataFrame(
                    {
                        "worker_id": split_df[schema.worker_id].astype(str),
                        "row_time_idx": split_df[schema.time_idx].astype(int),
                        "split": split_name,
                    }
                )
            )

    return (
        pd.concat(train_rows, ignore_index=True),
        pd.concat(val_rows, ignore_index=True),
        pd.concat(test_rows, ignore_index=True),
        pd.concat(split_manifest, ignore_index=True),
    )


def create_subject_holdout_splits(
    df: pd.DataFrame,
    schema: DataSchema,
    train_subjects: list[str],
    val_subjects: list[str],
    test_subjects: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_set = {str(s) for s in train_subjects}
    val_set = {str(s) for s in val_subjects}
    test_set = {str(s) for s in test_subjects}
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("Subject-holdout split has overlapping subject IDs across train/val/test.")

    all_subjects = set(df[schema.worker_id].astype(str).unique())
    missing = (train_set | val_set | test_set) - all_subjects
    if missing:
        raise ValueError(f"Requested subjects not found in dataset: {sorted(missing)}")

    train_df = df[df[schema.worker_id].astype(str).isin(train_set)].copy()
    val_df = df[df[schema.worker_id].astype(str).isin(val_set)].copy()
    test_df = df[df[schema.worker_id].astype(str).isin(test_set)].copy()
    split_manifest = pd.concat(
        [
            pd.DataFrame(
                {
                    "worker_id": train_df[schema.worker_id].astype(str),
                    "row_time_idx": train_df[schema.time_idx].astype(int),
                    "split": "train",
                }
            ),
            pd.DataFrame(
                {
                    "worker_id": val_df[schema.worker_id].astype(str),
                    "row_time_idx": val_df[schema.time_idx].astype(int),
                    "split": "val",
                }
            ),
            pd.DataFrame(
                {
                    "worker_id": test_df[schema.worker_id].astype(str),
                    "row_time_idx": test_df[schema.time_idx].astype(int),
                    "split": "test",
                }
            ),
        ],
        ignore_index=True,
    )
    return train_df, val_df, test_df, split_manifest


def build_windows(
    df: pd.DataFrame,
    schema: DataSchema,
    window_length: int,
    horizon_steps: int,
    window_step: int = 1,
    row_interval_seconds: float | None = None,
    context_seconds: float | None = None,
    forecast_horizon_seconds: float | None = None,
    inference_stride_seconds: float | None = None,
) -> WindowedData:
    feature_cols = list(schema.physiology) + list(schema.robot_context)
    for optional_signal in ("resp", "accel"):
        if optional_signal in df.columns and optional_signal not in feature_cols:
            feature_cols.append(optional_signal)
    if schema.hazard_zone in df.columns:
        feature_cols.append(schema.hazard_zone)

    windows: list[np.ndarray] = []
    y_stress: list[float] = []
    y_comfort: list[float] = []
    metas: list[dict[str, Any]] = []
    for worker_id, worker_df in df.groupby(schema.worker_id, observed=True):
        worker_df = worker_df.sort_values(schema.time_idx).reset_index(drop=True)
        timestamps_all = pd.to_numeric(worker_df[schema.timestamp], errors="coerce").to_numpy(dtype=float)
        if row_interval_seconds is not None and len(timestamps_all) > 1:
            diffs = np.diff(timestamps_all)
            breaks = np.where(~np.isclose(diffs, row_interval_seconds, atol=max(1e-5, row_interval_seconds * 1e-4)))[0] + 1
            segment_bounds = np.r_[0, breaks, len(worker_df)]
        else:
            segment_bounds = np.array([0, len(worker_df)])
        for seg_start, seg_stop in zip(segment_bounds[:-1], segment_bounds[1:]):
            segment_df = worker_df.iloc[int(seg_start):int(seg_stop)].reset_index(drop=True)
            if len(segment_df) < window_length + horizon_steps:
                continue
            values = segment_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
            stress = segment_df[schema.stress_target].to_numpy(dtype=float)
            comfort = segment_df[schema.comfort_target].to_numpy(dtype=float)
            times = segment_df[schema.time_idx].to_numpy(dtype=int)
            timestamps = pd.to_numeric(segment_df[schema.timestamp], errors="coerce").to_numpy(dtype=float)
            for start in range(0, len(segment_df) - window_length - horizon_steps + 1, max(1, window_step)):
                end = start + window_length
                label_idx = end + horizon_steps - 1
                if row_interval_seconds is not None:
                    observed_context = timestamps[end - 1] - timestamps[start] + row_interval_seconds
                    observed_horizon = timestamps[label_idx] - timestamps[end - 1]
                    expected_context = context_seconds if context_seconds is not None else window_length * row_interval_seconds
                    expected_horizon = (
                        forecast_horizon_seconds if forecast_horizon_seconds is not None else horizon_steps * row_interval_seconds
                    )
                    if not np.isclose(observed_context, expected_context, atol=max(1e-5, row_interval_seconds * 1e-4)):
                        raise ValueError(
                            f"Window context duration mismatch for worker {worker_id}: "
                            f"observed {observed_context}, expected {expected_context}."
                        )
                    if not np.isclose(observed_horizon, expected_horizon, atol=max(1e-5, row_interval_seconds * 1e-4)):
                        raise ValueError(
                            f"Label timestamp is not exactly after the observation window by the configured horizon: "
                            f"observed {observed_horizon}, expected {expected_horizon}."
                        )
                windows.append(values[start:end])
                y_stress.append(stress[label_idx])
                y_comfort.append(comfort[label_idx])
                metas.append(
                    {
                        "worker_id": str(worker_id),
                        "start_idx": int(times[start]),
                        "end_idx": int(times[end - 1]),
                        "label_time_idx": int(times[label_idx]),
                        "start_timestamp": float(timestamps[start]),
                        "end_timestamp": float(timestamps[end - 1]),
                        "label_timestamp": float(timestamps[label_idx]),
                        "context_seconds": float(context_seconds if context_seconds is not None else window_length),
                        "forecast_horizon_seconds": float(
                            forecast_horizon_seconds if forecast_horizon_seconds is not None else horizon_steps
                        ),
                        "inference_stride_seconds": float(
                            inference_stride_seconds if inference_stride_seconds is not None else window_step
                        ),
                    }
                )

    if not windows:
        raise ValueError("Insufficient data to build windows. Check window_length and horizon_steps.")
    return WindowedData(
        X_windows=np.stack(windows).astype(np.float32),
        y_stress=np.array(y_stress, dtype=np.float32),
        y_comfort=np.array(y_comfort, dtype=np.float32),
        meta=pd.DataFrame(metas),
    )


def engineer_window_features(
    windows: np.ndarray,
    schema: DataSchema,
    sampling_rate_hz: float,
    include_freq_domain: bool,
    scr_threshold: float,
    min_scr_distance: int,
) -> tuple[np.ndarray, list[str]]:
    cols = list(schema.physiology) + list(schema.robot_context)
    mode = str(getattr(schema, "feature_engineering_mode", "raw_signals"))
    if mode == "precomputed":
        rows: list[list[float]] = []
        names: list[str] | None = None
        for window in windows:
            signal_feats = extract_precomputed_window_features(window[:, : len(cols)], cols)
            if names is None:
                names = list(signal_feats.keys())
            rows.append([signal_feats[k] for k in names])
        return np.array(rows, dtype=np.float32), (names or [])

    if mode != "raw_signals":
        raise ValueError(f"Unsupported feature engineering mode: {mode!r}. Expected 'raw_signals' or 'precomputed'.")
    if sampling_rate_hz <= 0:
        raise ValueError("Raw signal feature extraction requires a positive effective sampling rate.")
    if "ecg" not in cols or "eda" not in cols or "temp" not in cols:
        raise ValueError(
            "Raw signal feature extraction requires physiology columns ['ecg', 'eda', 'temp']; "
            f"got {list(schema.physiology)}."
        )
    duration_seconds = windows.shape[1] / sampling_rate_hz if windows.ndim >= 2 else 0.0
    if duration_seconds < 4.0:
        raise ValueError(
            "Raw ECG/HRV feature extraction requires at least 4 seconds of context; "
            f"got {duration_seconds:.3f}s at {sampling_rate_hz:.3f} Hz."
        )

    ecg_idx = cols.index("ecg")
    eda_idx = cols.index("eda")
    temp_idx = cols.index("temp")
    dist_idx = cols.index("distance_to_robot") if "distance_to_robot" in cols else None
    speed_idx = cols.index("robot_speed") if "robot_speed" in cols else None
    resp_idx = cols.index("resp") if "resp" in cols else None
    accel_idx = cols.index("accel") if "accel" in cols else None

    rows: list[list[float]] = []
    names: list[str] | None = None
    for window in windows:
        signal_feats = extract_window_features(
            ecg_window=window[:, ecg_idx],
            eda_window=window[:, eda_idx],
            temp_window=window[:, temp_idx],
            sampling_rate_hz=sampling_rate_hz,
            include_freq_domain=include_freq_domain,
            scr_threshold=scr_threshold,
            min_scr_distance=min_scr_distance,
        )
        if dist_idx is not None and speed_idx is not None:
            dist = float(window[-1, dist_idx])
            speed = float(window[-1, speed_idx])
            signal_feats["ttc_proxy"] = dist / max(speed, 1e-3)
        signal_feats["distance_last"] = float(window[-1, dist_idx]) if dist_idx is not None else 0.0
        signal_feats["speed_last"] = float(window[-1, speed_idx]) if speed_idx is not None else 0.0
        if resp_idx is not None:
            signal_feats["resp_mean"] = float(np.mean(window[:, resp_idx]))
            signal_feats["resp_std"] = float(np.std(window[:, resp_idx]))
        if accel_idx is not None:
            signal_feats["accel_mean"] = float(np.mean(window[:, accel_idx]))
            signal_feats["accel_std"] = float(np.std(window[:, accel_idx]))
        if names is None:
            names = list(signal_feats.keys())
        rows.append([signal_feats[k] for k in names])
    return np.array(rows, dtype=np.float32), (names or [])
