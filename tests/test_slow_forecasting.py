from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.multihorizon_windowing import build_multi_horizon_windows, horizon_seconds_to_steps, windows_to_long_predictions
from src.models.tft_model import SlowTFTForecaster, assert_no_worker_static_embedding, resolve_tft_loss
from src.training.tft_train import _PlattCalibrator, _calibration_cutoffs, _filter_after_calibration


def _synthetic_frame(n: int = 80) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "worker_id": ["S1"] * n,
            "timestamp": np.arange(n, dtype=float),
            "protocol_label": ["baseline"] * n,
            "y_stress": (np.arange(n) >= 40).astype(int),
            "ecg": np.sin(np.arange(n) / 5.0),
            "eda": np.cos(np.arange(n) / 7.0),
            "temp": np.linspace(30.0, 31.0, n),
        }
    )


def test_tft_classification_uses_bce_and_rejects_quantile():
    loss = resolve_tft_loss({"loss": "bce"}, task_type="classification")
    assert isinstance(loss, torch.nn.BCEWithLogitsLoss)
    with pytest.raises(ValueError, match="requires loss='bce'"):
        resolve_tft_loss({"loss": "quantile"}, task_type="classification")


def test_slow_tft_outputs_logits_for_every_horizon_without_sigmoid_module():
    model = SlowTFTForecaster(input_channels=3, n_horizons=2, hidden_size=6)
    logits = model(torch.randn(4, 3, 30))
    assert logits.shape == (4, 2)
    assert not any(isinstance(module, torch.nn.Sigmoid) for module in model.modules())


def test_multi_horizon_timestamps_cadence_and_context_are_distinct():
    frame = _synthetic_frame()
    horizons = [5.0, 30.0]
    steps = horizon_seconds_to_steps(horizons, rate_hz=1.0)
    windows = build_multi_horizon_windows(
        frame,
        worker_col="worker_id",
        timestamp_col="timestamp",
        protocol_col="protocol_label",
        target_col="y_stress",
        feature_columns=["ecg", "eda", "temp"],
        context_steps=30,
        stride_steps=1,
        horizon_steps=steps,
        horizon_seconds=horizons,
        row_interval_seconds=1.0,
        split="test",
        context_seconds=30.0,
        inference_stride_seconds=1.0,
    )
    assert np.allclose(np.diff(windows.meta["prediction_timestamp"].head(5)), 1.0)
    assert np.allclose(windows.meta["window_end_timestamp"] - windows.meta["window_start_timestamp"], 29.0)
    logits = np.zeros((len(windows.meta), 2))
    probs = np.full_like(logits, 0.5, dtype=float)
    long = windows_to_long_predictions(
        windows,
        raw_logits=logits,
        uncalibrated_probability=probs,
        calibrated_probability=probs,
        threshold_by_horizon={5.0: 0.5, 30.0: 0.5},
        model_name="tft",
    )
    assert set(long["horizon_seconds"]) == {5.0, 30.0}
    assert np.allclose(long["target_timestamp"] - long["prediction_timestamp"], long["horizon_seconds"])


def test_windows_do_not_cross_protocol_boundaries():
    frame = _synthetic_frame()
    frame.loc[35:45, "protocol_label"] = "stress"
    windows = build_multi_horizon_windows(
        frame,
        worker_col="worker_id",
        timestamp_col="timestamp",
        protocol_col="protocol_label",
        target_col="y_stress",
        feature_columns=["ecg", "eda", "temp"],
        context_steps=30,
        stride_steps=1,
        horizon_steps=[5],
        horizon_seconds=[5.0],
        row_interval_seconds=1.0,
        split="test",
        context_seconds=30.0,
        inference_stride_seconds=1.0,
    )
    for _, row in windows.meta.iterrows():
        span = frame[(frame["timestamp"] >= row["window_start_timestamp"]) & (frame["timestamp"] <= row["window_end_timestamp"])]
        assert span["protocol_label"].nunique() == 1


def test_calibration_cutoff_filters_scored_windows_after_initial_segment():
    frame = _synthetic_frame()
    cutoffs = _calibration_cutoffs(frame, type("Schema", (), {"worker_id": "worker_id", "timestamp": "timestamp", "protocol_label": "protocol_label"}), 10.0)
    windows = build_multi_horizon_windows(
        frame,
        worker_col="worker_id",
        timestamp_col="timestamp",
        protocol_col="protocol_label",
        target_col="y_stress",
        feature_columns=["ecg", "eda", "temp"],
        context_steps=5,
        stride_steps=1,
        horizon_steps=[1],
        horizon_seconds=[1.0],
        row_interval_seconds=1.0,
        split="test",
        context_seconds=5.0,
        inference_stride_seconds=1.0,
    )
    filtered = _filter_after_calibration(windows, cutoffs)
    assert float(filtered.meta["prediction_timestamp"].min()) > float(cutoffs["calibration_cutoff_timestamp"].iloc[0])


def test_calibrators_require_validation_labels_and_worker_id_not_static():
    with pytest.raises(ValueError, match="both validation classes"):
        _PlattCalibrator().fit(np.array([0.1, 0.2]), np.array([0, 0]))
    assert_no_worker_static_embedding("worker_id", ["task_phase"])
    with pytest.raises(ValueError, match="worker_id"):
        assert_no_worker_static_embedding("worker_id", ["worker_id"])

