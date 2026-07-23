"""Causal multi-horizon window construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.streaming.fast_detector import causal_window_features, causal_feature_names


@dataclass(frozen=True)
class MultiHorizonWindows:
    raw: np.ndarray
    engineered: np.ndarray
    targets: np.ndarray
    meta: pd.DataFrame
    feature_names: list[str]
    horizon_seconds: list[float]
    context_seconds: float
    inference_stride_seconds: float
    row_interval_seconds: float


def horizon_seconds_to_steps(horizons_seconds: Sequence[float], rate_hz: float) -> list[int]:
    steps: list[int] = []
    for horizon in horizons_seconds:
        if float(horizon) <= 0:
            raise ValueError("Forecast horizons must be positive seconds for slow forecasting.")
        step = int(round(float(horizon) * rate_hz))
        if step < 1:
            raise ValueError(f"Horizon {horizon}s is shorter than one sample at {rate_hz} Hz.")
        steps.append(step)
    return steps


def build_multi_horizon_windows(
    frame: pd.DataFrame,
    *,
    worker_col: str,
    timestamp_col: str,
    protocol_col: str,
    target_col: str,
    feature_columns: Sequence[str],
    context_steps: int,
    stride_steps: int,
    horizon_steps: Sequence[int],
    horizon_seconds: Sequence[float],
    row_interval_seconds: float,
    split: str,
    context_seconds: float,
    inference_stride_seconds: float,
) -> MultiHorizonWindows:
    if context_steps < 1 or stride_steps < 1:
        raise ValueError("context_steps and stride_steps must be positive.")
    if len(horizon_steps) != len(horizon_seconds):
        raise ValueError("horizon_steps and horizon_seconds must have the same length.")

    raw_rows: list[np.ndarray] = []
    feature_rows: list[np.ndarray] = []
    target_rows: list[list[int]] = []
    meta_rows: list[dict[str, Any]] = []
    feature_cols = list(feature_columns)
    max_horizon = max(int(h) for h in horizon_steps)
    for worker_id, subject in frame.groupby(worker_col, observed=True):
        subject = subject.sort_values(timestamp_col)
        values = subject[feature_cols].to_numpy(dtype=float)
        targets = pd.to_numeric(subject[target_col], errors="coerce").to_numpy(dtype=int)
        timestamps = pd.to_numeric(subject[timestamp_col], errors="coerce").to_numpy(dtype=float)
        protocols = subject[protocol_col].astype(str).to_numpy()
        for origin in range(context_steps - 1, len(subject) - max_horizon, stride_steps):
            start = origin - context_steps + 1
            window_protocols = protocols[start : origin + 1]
            if len(set(window_protocols)) != 1:
                continue
            diffs = np.diff(timestamps[start : origin + 1])
            if len(diffs) and np.any(diffs > row_interval_seconds * 1.5):
                continue
            if any(protocols[origin + int(h)] != protocols[origin] for h in horizon_steps):
                continue
            y = [int(targets[origin + int(h)]) for h in horizon_steps]
            window = values[start : origin + 1]
            raw_rows.append(window.T.astype(np.float32))
            feature_rows.append(causal_window_features(window, feature_cols).astype(np.float32))
            target_rows.append(y)
            meta_rows.append(
                {
                    "worker_id": str(worker_id),
                    "split": split,
                    "protocol_label": str(protocols[origin]),
                    "window_start_timestamp": float(timestamps[start]),
                    "window_end_timestamp": float(timestamps[origin]),
                    "prediction_timestamp": float(timestamps[origin]),
                    "context_seconds": float(context_seconds),
                    "inference_stride_seconds": float(inference_stride_seconds),
                }
            )
    return MultiHorizonWindows(
        raw=np.asarray(raw_rows, dtype=np.float32),
        engineered=np.asarray(feature_rows, dtype=np.float32),
        targets=np.asarray(target_rows, dtype=int),
        meta=pd.DataFrame(meta_rows),
        feature_names=causal_feature_names(feature_cols),
        horizon_seconds=[float(h) for h in horizon_seconds],
        context_seconds=float(context_seconds),
        inference_stride_seconds=float(inference_stride_seconds),
        row_interval_seconds=float(row_interval_seconds),
    )


def windows_to_long_predictions(
    windows: MultiHorizonWindows,
    *,
    raw_logits: np.ndarray,
    uncalibrated_probability: np.ndarray,
    calibrated_probability: np.ndarray,
    threshold_by_horizon: dict[float, float],
    model_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row_idx, meta_row in windows.meta.reset_index(drop=True).iterrows():
        for h_idx, horizon in enumerate(windows.horizon_seconds):
            target_timestamp = float(meta_row["prediction_timestamp"]) + float(horizon)
            prob = float(calibrated_probability[row_idx, h_idx])
            rows.append(
                {
                    "model": model_name,
                    "worker_id": meta_row["worker_id"],
                    "protocol_label": meta_row["protocol_label"],
                    "window_start_timestamp": float(meta_row["window_start_timestamp"]),
                    "window_end_timestamp": float(meta_row["window_end_timestamp"]),
                    "prediction_timestamp": float(meta_row["prediction_timestamp"]),
                    "target_timestamp": target_timestamp,
                    "horizon_seconds": float(horizon),
                    "target": int(windows.targets[row_idx, h_idx]),
                    "raw_logit": float(raw_logits[row_idx, h_idx]),
                    "uncalibrated_probability": float(uncalibrated_probability[row_idx, h_idx]),
                    "calibrated_probability": prob,
                    "prediction": int(prob >= threshold_by_horizon[float(horizon)]),
                }
            )
    return pd.DataFrame(rows)
