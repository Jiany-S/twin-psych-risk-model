"""Centralized temporal configuration and sample conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalSpec:
    representation: str
    source_sampling_rate_hz: float | None
    target_sampling_rate_hz: float | None
    row_interval_seconds: float
    context_seconds: float
    forecast_horizon_seconds: float
    inference_stride_seconds: float
    context_steps: int
    horizon_steps: int
    stride_steps: int

    @property
    def effective_sampling_rate_hz(self) -> float:
        return 1.0 / self.row_interval_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation": self.representation,
            "source_sampling_rate_hz": self.source_sampling_rate_hz,
            "target_sampling_rate_hz": self.target_sampling_rate_hz,
            "effective_sampling_rate_hz": self.effective_sampling_rate_hz,
            "row_interval_seconds": self.row_interval_seconds,
            "context_seconds": self.context_seconds,
            "forecast_horizon_seconds": self.forecast_horizon_seconds,
            "inference_stride_seconds": self.inference_stride_seconds,
            "expected_prediction_cadence_seconds": self.inference_stride_seconds,
            "context_steps": self.context_steps,
            "horizon_steps": self.horizon_steps,
            "stride_steps": self.stride_steps,
        }


def seconds_to_samples(seconds: float, rate_hz: float, *, name: str) -> int:
    if seconds <= 0:
        raise ValueError(f"{name} must be positive seconds; got {seconds}.")
    if rate_hz <= 0:
        raise ValueError(f"Sampling rate must be positive; got {rate_hz}.")
    samples = int(round(seconds * rate_hz))
    if samples < 1:
        raise ValueError(f"{name}={seconds}s is shorter than one sample at {rate_hz} Hz.")
    return samples


def infer_temporal_spec(cfg: Mapping[str, Any]) -> TemporalSpec:
    stream = cfg.get("stream", {})
    if not isinstance(stream, Mapping):
        stream = {}
    task = cfg.get("task", {})
    if not isinstance(task, Mapping):
        task = {}
    dataset = cfg.get("dataset", {})
    if not isinstance(dataset, Mapping):
        dataset = {}

    representation = str(stream.get("representation", "raw_signal")).lower()
    if representation not in {"raw_signal", "precomputed_features"}:
        raise ValueError("stream.representation must be 'raw_signal' or 'precomputed_features'.")

    if representation == "precomputed_features":
        row_interval = float(stream.get("row_interval_seconds", 60.0))
        if row_interval <= 0:
            raise ValueError("stream.row_interval_seconds must be positive for precomputed feature tables.")
        rate_hz = 1.0 / row_interval
        source_hz = None
        target_hz = rate_hz
    else:
        legacy_rate = float(task.get("sampling_rate_hz", 1.0))
        source_hz = float(stream.get("source_sampling_rate_hz", legacy_rate))
        target_hz = float(stream.get("target_sampling_rate_hz", stream.get("source_sampling_rate_hz", legacy_rate)))
        if source_hz <= 0 or target_hz <= 0:
            raise ValueError("stream source/target sampling rates must be positive.")
        row_interval = 1.0 / target_hz
        rate_hz = target_hz

    context_seconds = task.get("context_seconds")
    horizon_seconds = task.get("forecast_horizon_seconds")
    stride_seconds = task.get("inference_stride_seconds")
    if context_seconds is None:
        context_seconds = float(task.get("window_length", 1)) * row_interval
    if horizon_seconds is None:
        horizon_seconds = float(task.get("horizon_steps", 1)) * row_interval
    if stride_seconds is None:
        stride_seconds = float(task.get("window_step", 1)) * row_interval

    if representation == "precomputed_features" and float(stride_seconds) < row_interval:
        raise ValueError(
            "Precomputed feature tables cannot support inference_stride_seconds smaller than "
            f"row_interval_seconds={row_interval}."
        )

    context_steps = seconds_to_samples(float(context_seconds), rate_hz, name="task.context_seconds")
    horizon_steps = seconds_to_samples(float(horizon_seconds), rate_hz, name="task.forecast_horizon_seconds")
    stride_steps = seconds_to_samples(float(stride_seconds), rate_hz, name="task.inference_stride_seconds")

    return TemporalSpec(
        representation=representation,
        source_sampling_rate_hz=source_hz,
        target_sampling_rate_hz=target_hz,
        row_interval_seconds=row_interval,
        context_seconds=float(context_seconds),
        forecast_horizon_seconds=float(horizon_seconds),
        inference_stride_seconds=float(stride_seconds),
        context_steps=context_steps,
        horizon_steps=horizon_steps,
        stride_steps=stride_steps,
    )


def assert_regular_timestamps(
    frame: pd.DataFrame,
    worker_col: str,
    timestamp_col: str,
    expected_interval_seconds: float,
    *,
    tolerance_seconds: float = 1e-5,
    allow_gaps: bool = True,
) -> None:
    for worker_id, worker_df in frame.groupby(worker_col, observed=True):
        ts = pd.to_numeric(worker_df.sort_values(timestamp_col)[timestamp_col], errors="coerce").to_numpy(dtype=float)
        if len(ts) < 2:
            continue
        diffs = np.diff(ts)
        if allow_gaps:
            multiples = diffs / expected_interval_seconds
            valid = (diffs > 0) & np.isclose(multiples, np.round(multiples), atol=1e-4, rtol=1e-4)
        else:
            valid = np.isclose(diffs, expected_interval_seconds, atol=tolerance_seconds, rtol=1e-4)
        if not np.all(valid):
            raise ValueError(
                f"Irregular timestamps for worker {worker_id}: expected {expected_interval_seconds}s spacing, "
                f"observed range {float(np.nanmin(diffs))}..{float(np.nanmax(diffs))}."
            )
