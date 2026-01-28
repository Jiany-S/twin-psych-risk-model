"""Windowing and feature engineering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .schema import DataSchema


@dataclass
class WindowedData:
    X_windows: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame
    flat: pd.DataFrame


def _encode_task_phase(series: pd.Series) -> pd.Series:
    if series.dtype.name == "category":
        return series.cat.codes
    return series.astype("category").cat.codes


def create_time_splits(
    df: pd.DataFrame,
    schema: DataSchema,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train/val/test ratios must sum to 1.")
    train_rows = []
    val_rows = []
    test_rows = []
    for _, worker_df in df.groupby(schema.worker_id):
        worker_df = worker_df.sort_values(schema.time_idx)
        n = len(worker_df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        train_rows.append(worker_df.iloc[:train_end])
        val_rows.append(worker_df.iloc[train_end:val_end])
        test_rows.append(worker_df.iloc[val_end:])
    return (
        pd.concat(train_rows, ignore_index=True),
        pd.concat(val_rows, ignore_index=True),
        pd.concat(test_rows, ignore_index=True),
    )


def build_windows(
    df: pd.DataFrame,
    schema: DataSchema,
    window_length: int,
    horizon_steps: int,
    task_type: str,
    risk_threshold: float,
) -> WindowedData:
    frame = df.copy()
    frame[schema.task_phase] = _encode_task_phase(frame[schema.task_phase])

    feature_cols = list(schema.physiology) + list(schema.robot_context) + [schema.hazard_zone, schema.task_phase]

    windows = []
    labels = []
    metas = []
    for worker_id, worker_df in frame.groupby(schema.worker_id):
        worker_df = worker_df.sort_values(schema.time_idx).reset_index(drop=True)
        values = worker_df[feature_cols].to_numpy(dtype=float)
        targets = worker_df[schema.target].to_numpy(dtype=float)
        times = worker_df[schema.time_idx].to_numpy(dtype=int)

        for start in range(0, len(worker_df) - window_length - horizon_steps + 1):
            end = start + window_length
            label_idx = end + horizon_steps - 1
            window = values[start:end]
            label = targets[label_idx]
            if task_type == "classification":
                label = float(label >= risk_threshold)
            windows.append(window)
            labels.append(label)
            metas.append(
                {
                    "worker_id": worker_id,
                    "start_idx": int(times[start]),
                    "end_idx": int(times[end - 1]),
                    "label_time_idx": int(times[label_idx]),
                }
            )

    if not windows:
        raise ValueError("Insufficient data to build windows. Check window length and horizon.")

    X = np.stack(windows).astype(np.float32)
    y = np.array(labels).astype(np.float32)
    meta = pd.DataFrame(metas)

    return WindowedData(X_windows=X, y=y, meta=meta, flat=frame)


def engineer_window_features(
    windows: np.ndarray,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    n_windows, length, n_features = windows.shape
    feats = []
    names = []
    x = windows

    last_vals = x[:, -1, :]
    feats.append(last_vals)
    names += [f"{f}_last" for f in feature_names]

    mean_vals = x.mean(axis=1)
    feats.append(mean_vals)
    names += [f"{f}_mean" for f in feature_names]

    std_vals = x.std(axis=1)
    feats.append(std_vals)
    names += [f"{f}_std" for f in feature_names]

    min_vals = x.min(axis=1)
    feats.append(min_vals)
    names += [f"{f}_min" for f in feature_names]

    max_vals = x.max(axis=1)
    feats.append(max_vals)
    names += [f"{f}_max" for f in feature_names]

    # slope via simple linear regression against time index
    time_idx = np.arange(length).reshape(1, -1, 1)
    time_mean = time_idx.mean()
    denom = ((time_idx - time_mean) ** 2).sum()
    slope = ((time_idx - time_mean) * (x - x.mean(axis=1, keepdims=True))).sum(axis=1) / denom
    feats.append(slope)
    names += [f"{f}_slope" for f in feature_names]

    feature_matrix = np.concatenate(feats, axis=1).astype(np.float32)
    return feature_matrix, names


def add_ttc_features(
    feature_matrix: np.ndarray,
    feature_names: list[str],
    windows: np.ndarray,
    feature_cols: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    if "distance_to_robot" not in feature_cols or "robot_speed" not in feature_cols:
        return feature_matrix, feature_names
    dist_idx = feature_cols.index("distance_to_robot")
    speed_idx = feature_cols.index("robot_speed")
    dist_last = windows[:, -1, dist_idx]
    speed_last = windows[:, -1, speed_idx]
    ttc = dist_last / np.clip(speed_last, 1e-3, None)
    feature_matrix = np.concatenate([feature_matrix, ttc.reshape(-1, 1)], axis=1)
    feature_names = feature_names + ["time_to_collision"]
    return feature_matrix, feature_names
