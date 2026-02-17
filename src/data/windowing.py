"""Windowing, chronological split, and feature engineering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .features import extract_window_features
from .schema import DataSchema


def validate_time_indices(df: pd.DataFrame, schema: DataSchema) -> None:
    """Validate that time indices are strictly increasing per worker.

    Raises:
        ValueError: If time indices are not monotonically increasing for any worker.
    """
    for worker_id, worker_df in df.groupby(schema.worker_id, observed=True):
        time_idx = worker_df[schema.time_idx].values
        if len(time_idx) > 1:
            diffs = np.diff(time_idx)
            if not np.all(diffs > 0):
                non_increasing = np.where(diffs <= 0)[0]
                first_bad_idx = non_increasing[0]
                raise ValueError(
                    f"Time indices not strictly increasing for worker {worker_id}: "
                    f"indices[{first_bad_idx}]={time_idx[first_bad_idx]} >= "
                    f"indices[{first_bad_idx + 1}]={time_idx[first_bad_idx + 1]}"
                )


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
        values = worker_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
        stress = worker_df[schema.stress_target].to_numpy(dtype=float)
        comfort = worker_df[schema.comfort_target].to_numpy(dtype=float)
        times = worker_df[schema.time_idx].to_numpy(dtype=int)
        for start in range(0, len(worker_df) - window_length - horizon_steps + 1, max(1, window_step)):
            end = start + window_length
            label_idx = end + horizon_steps - 1
            windows.append(values[start:end])
            y_stress.append(stress[label_idx])
            y_comfort.append(comfort[label_idx])
            metas.append(
                {
                    "worker_id": str(worker_id),
                    "start_idx": int(times[start]),
                    "end_idx": int(times[end - 1]),
                    "label_time_idx": int(times[label_idx]),
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
    if "ecg" not in cols or "eda" not in cols or "temp" not in cols:
        raise ValueError("Expected physiology columns ecg, eda, temp for WESAD feature extraction.")

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
