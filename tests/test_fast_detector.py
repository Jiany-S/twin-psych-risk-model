from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from src.models.fast_tcn import FastTCN
from src.streaming.fast_detector import CausalStandardizer, FastDetectorConfig, FastRiskDetector


class ConstantModel:
    def __init__(self, probability: float = 0.7) -> None:
        self.probability = probability

    def predict_proba(self, x):
        return np.column_stack([np.full(len(x), 1.0 - self.probability), np.full(len(x), self.probability)])


def _sample(t: float, value: float = 1.0) -> dict[str, float]:
    return {"timestamp": t, "ecg": value, "eda": value + 1.0, "temp": value + 2.0}


def test_fast_detector_respects_buffer_and_cadence():
    cfg = FastDetectorConfig(["ecg", "eda", "temp"], sample_interval_seconds=1.0, context_seconds=3.0, inference_stride_seconds=2.0)
    detector = FastRiskDetector(ConstantModel(), cfg)
    assert detector.update(_sample(0.0))["status"] == "not_ready"
    assert detector.update(_sample(1.0))["status"] == "not_ready"
    first = detector.update(_sample(2.0))
    assert first["status"] == "prediction"
    assert 0.0 <= first["probability"] <= 1.0
    assert detector.update(_sample(3.0))["status"] == "not_ready"
    assert detector.update(_sample(4.0))["status"] == "prediction"


def test_fast_detector_gap_reset_and_manual_reset():
    cfg = FastDetectorConfig(["ecg", "eda", "temp"], sample_interval_seconds=1.0, context_seconds=2.0, inference_stride_seconds=1.0)
    detector = FastRiskDetector(ConstantModel(), cfg)
    detector.update(_sample(0.0))
    detector.update(_sample(1.0))
    gap = detector.update(_sample(5.0))
    assert gap["status"] == "skipped_gap"
    assert detector.buffer_size == 1
    detector.reset()
    assert detector.buffer_size == 0


def test_fast_detector_requires_declared_features_and_no_robot_defaults():
    cfg = FastDetectorConfig(["ecg", "eda", "temp"], sample_interval_seconds=1.0, context_seconds=1.0, inference_stride_seconds=1.0)
    detector = FastRiskDetector(ConstantModel(), cfg)
    with pytest.raises(ValueError, match="Missing required"):
        detector.update({"timestamp": 0.0, "ecg": 1.0, "eda": 2.0})


def test_causal_standardizer_uses_only_allowed_calibration_rows():
    frame = pd.DataFrame(
        {
            "timestamp": [0.0, 1.0, 2.0, 99.0],
            "ecg": [1.0, 2.0, 3.0, 1000.0],
            "eda": [2.0, 3.0, 4.0, 1000.0],
        }
    )
    scaler = CausalStandardizer(["ecg", "eda"]).fit(frame, cutoff_timestamp=2.0)
    assert scaler.fit_sample_count_ == 3
    assert scaler.max_fit_timestamp_ == 2.0
    assert np.allclose(scaler.mean_, [2.0, 3.0])


def test_detector_serialization_roundtrip(tmp_path):
    cfg = FastDetectorConfig(["ecg", "eda", "temp"], sample_interval_seconds=1.0, context_seconds=1.0, inference_stride_seconds=1.0)
    path = tmp_path / "detector.joblib"
    joblib.dump(FastRiskDetector(ConstantModel(0.6), cfg), path)
    loaded = joblib.load(path)
    out = loaded.update(_sample(0.0))
    assert out["status"] == "prediction"
    assert out["prediction"] == 1


def test_fast_tcn_batch_size_one_and_probability_range():
    model = FastTCN(input_channels=3, hidden_channels=4, layers=2)
    x = torch.randn(1, 3, 12)
    with torch.no_grad():
        prob = model.predict_proba_tensor(x)
    assert prob.shape == (1,)
    assert float(prob.min()) >= 0.0
    assert float(prob.max()) <= 1.0

