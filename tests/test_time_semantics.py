from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.schema import DataSchema
from src.data.time_semantics import infer_temporal_spec, seconds_to_samples
from src.data.windowing import build_windows, engineer_window_features


def _schema(mode: str = "raw_signals") -> DataSchema:
    return DataSchema.from_config(
        {
            "features": {
                "timestamp": "timestamp",
                "time_idx": "time_idx",
                "worker_id": "worker_id",
                "protocol_label": "protocol_label",
                "physiology": ["ecg", "eda", "temp"] if mode == "raw_signals" else ["hrv_mean_nn", "eda_mean"],
                "engineering_mode": mode,
                "use_robot_context": False,
                "optional": {"hazard_zone": "hazard_zone", "task_phase": "task_phase"},
            },
            "targets": {
                "stress": {"label_col": "y_stress"},
                "comfort": {"label_col": "y_comfort_proxy"},
            },
        }
    )


def _timeline(n: int = 12, interval: float = 0.5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "worker_id": ["S1"] * n,
            "timestamp": np.arange(n, dtype=float) * interval,
            "time_idx": np.arange(n),
            "protocol_label": ["baseline"] * n,
            "ecg": np.sin(np.arange(n)),
            "eda": np.arange(n, dtype=float),
            "temp": 30.0 + np.arange(n, dtype=float) * 0.01,
            "hazard_zone": [0] * n,
            "task_phase": ["rest"] * n,
            "y_stress": np.arange(n) % 2,
            "y_comfort_proxy": np.ones(n),
        }
    )


def test_seconds_to_samples_is_centralized_and_rounded() -> None:
    assert seconds_to_samples(2.0, 4.0, name="context") == 8
    assert seconds_to_samples(2.49, 2.0, name="context") == 5
    with pytest.raises(ValueError):
        seconds_to_samples(0.0, 4.0, name="context")


def test_duration_config_derives_window_horizon_and_stride_samples() -> None:
    spec = infer_temporal_spec(
        {
            "stream": {"representation": "raw_signal", "source_sampling_rate_hz": 700, "target_sampling_rate_hz": 4},
            "task": {
                "context_seconds": 37.5,
                "forecast_horizon_seconds": 3.75,
                "inference_stride_seconds": 2.5,
            },
        }
    )
    assert spec.context_steps == 150
    assert spec.horizon_steps == 15
    assert spec.stride_steps == 10
    assert spec.effective_sampling_rate_hz == 4.0


def test_window_labels_are_exactly_after_context_by_horizon() -> None:
    schema = _schema()
    windows = build_windows(
        _timeline(n=12, interval=0.5),
        schema,
        window_length=4,
        horizon_steps=2,
        window_step=2,
        row_interval_seconds=0.5,
        context_seconds=2.0,
        forecast_horizon_seconds=1.0,
        inference_stride_seconds=1.0,
    )
    assert list(windows.meta["start_timestamp"]) == [0.0, 1.0, 2.0, 3.0]
    assert np.allclose(windows.meta["label_timestamp"] - windows.meta["end_timestamp"], 1.0)
    assert windows.meta["start_timestamp"].diff().dropna().eq(1.0).all()


def test_windows_do_not_cross_timestamp_gaps() -> None:
    schema = _schema()
    frame = _timeline(n=10, interval=0.5)
    frame.loc[5:, "timestamp"] += 10.0
    windows = build_windows(
        frame,
        schema,
        window_length=3,
        horizon_steps=1,
        window_step=1,
        row_interval_seconds=0.5,
        context_seconds=1.5,
        forecast_horizon_seconds=0.5,
        inference_stride_seconds=0.5,
    )
    assert not ((windows.meta["start_timestamp"] < 2.5) & (windows.meta["label_timestamp"] > 10.0)).any()


def test_precomputed_features_report_sixty_second_rows() -> None:
    spec = infer_temporal_spec(
        {
            "stream": {"representation": "precomputed_features", "row_interval_seconds": 60},
            "task": {
                "context_seconds": 480,
                "forecast_horizon_seconds": 60,
                "inference_stride_seconds": 60,
            },
        }
    )
    assert spec.context_steps == 8
    assert spec.effective_sampling_rate_hz == pytest.approx(1 / 60)
    with pytest.raises(ValueError, match="cannot support"):
        infer_temporal_spec(
            {
                "stream": {"representation": "precomputed_features", "row_interval_seconds": 60},
                "task": {
                    "context_seconds": 60,
                    "forecast_horizon_seconds": 60,
                    "inference_stride_seconds": 5,
                },
            }
        )


def test_impossible_raw_hrv_window_raises() -> None:
    schema = _schema()
    windows = np.zeros((1, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="at least 4 seconds"):
        engineer_window_features(windows, schema, 4.0, True, 0.05, 3)
