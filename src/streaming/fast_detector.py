"""Causal streaming detector primitives."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass
class FastDetectorConfig:
    feature_columns: list[str]
    sample_interval_seconds: float
    context_seconds: float
    inference_stride_seconds: float
    threshold: float = 0.5
    gap_tolerance: float = 1.5

    @property
    def context_samples(self) -> int:
        return max(1, int(round(self.context_seconds / self.sample_interval_seconds)))


class CausalStandardizer:
    """Standardizer fitted only on rows available at or before a cutoff."""

    def __init__(self, feature_columns: Iterable[str]):
        self.feature_columns = list(feature_columns)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.fit_sample_count_: int = 0
        self.max_fit_timestamp_: float | None = None

    def fit(
        self,
        frame: pd.DataFrame,
        timestamp_col: str = "timestamp",
        cutoff_timestamp: float | None = None,
    ) -> "CausalStandardizer":
        fit_frame = frame
        if cutoff_timestamp is not None:
            timestamps = pd.to_numeric(fit_frame[timestamp_col], errors="coerce")
            fit_frame = fit_frame[timestamps <= float(cutoff_timestamp)]
        if fit_frame.empty:
            raise ValueError("No calibration rows are available at or before cutoff_timestamp.")
        values = fit_frame[self.feature_columns].to_numpy(dtype=float)
        self.mean_ = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        self.scale_ = np.where(scale < 1e-8, 1.0, scale)
        self.fit_sample_count_ = int(len(fit_frame))
        self.max_fit_timestamp_ = float(pd.to_numeric(fit_frame[timestamp_col], errors="coerce").max())
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("CausalStandardizer must be fit before transform.")
        return (values - self.mean_) / self.scale_


def causal_window_features(window: np.ndarray, feature_columns: list[str]) -> np.ndarray:
    """Extract compact features from a past-only raw window."""
    if window.ndim != 2:
        raise ValueError("window must have shape [time, channels].")
    feats: list[float] = []
    x_idx = np.arange(window.shape[0], dtype=float)
    x_idx = x_idx - x_idx.mean()
    denom = float(np.sum(x_idx**2)) or 1.0
    for col_idx, _ in enumerate(feature_columns):
        values = window[:, col_idx].astype(float)
        fill = float(np.nanmean(values)) if np.isfinite(np.nanmean(values)) else 0.0
        values = np.nan_to_num(values, nan=fill)
        slope = float(np.sum(x_idx * (values - values.mean())) / denom)
        feats.extend([float(values.mean()), float(values.std()), float(values.min()), float(values.max()), float(values[-1]), slope])
    return np.asarray(feats, dtype=float)


def causal_feature_names(feature_columns: list[str]) -> list[str]:
    suffixes = ["mean", "std", "min", "max", "last", "slope"]
    return [f"{col}_{suffix}" for col in feature_columns for suffix in suffixes]


class FastRiskDetector:
    """Online causal detector with bounded buffer and fixed prediction cadence."""

    def __init__(
        self,
        model: Any,
        config: FastDetectorConfig,
        standardizer: CausalStandardizer | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.standardizer = standardizer
        self._buffer: deque[tuple[float, np.ndarray]] = deque(maxlen=config.context_samples)
        self._last_timestamp: float | None = None
        self._next_prediction_timestamp: float | None = None

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()
        self._last_timestamp = None
        self._next_prediction_timestamp = None

    def update(self, sample: dict[str, Any]) -> dict[str, Any]:
        missing = [col for col in self.config.feature_columns if col not in sample]
        if missing:
            raise ValueError(f"Missing required streaming feature(s): {missing}")
        timestamp = float(sample.get("timestamp", sample.get("time", 0.0)))
        if self._last_timestamp is not None:
            delta = timestamp - self._last_timestamp
            max_gap = self.config.sample_interval_seconds * self.config.gap_tolerance
            if delta < -1e-9:
                raise ValueError("Streaming timestamps must be nondecreasing.")
            if delta > max_gap:
                self.reset()
                self._append(timestamp, sample)
                return {"status": "skipped_gap", "timestamp": timestamp, "reason": "timestamp_gap_reset"}
        self._append(timestamp, sample)
        if len(self._buffer) < self.config.context_samples:
            return {"status": "not_ready", "timestamp": timestamp, "buffer_size": len(self._buffer)}
        if self._next_prediction_timestamp is None:
            self._next_prediction_timestamp = timestamp
        if timestamp + 1e-9 < self._next_prediction_timestamp:
            return {"status": "not_ready", "timestamp": timestamp, "reason": "cadence_wait"}
        started = time.perf_counter()
        window = np.vstack([row for _, row in self._buffer])
        features = causal_window_features(window, self.config.feature_columns).reshape(1, -1)
        if self.standardizer is not None:
            features = self.standardizer.transform(features)
        prob = float(np.asarray(self.model.predict_proba(features))[0, 1])
        prob = float(np.clip(prob, 0.0, 1.0))
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._next_prediction_timestamp = timestamp + self.config.inference_stride_seconds
        return {
            "status": "prediction",
            "timestamp": timestamp,
            "probability": prob,
            "prediction": int(prob >= self.config.threshold),
            "threshold": float(self.config.threshold),
            "latency_ms": float(latency_ms),
            "buffer_size": len(self._buffer),
        }

    def _append(self, timestamp: float, sample: dict[str, Any]) -> None:
        values = np.asarray([float(sample[col]) for col in self.config.feature_columns], dtype=float)
        self._buffer.append((timestamp, values))
        self._last_timestamp = timestamp
