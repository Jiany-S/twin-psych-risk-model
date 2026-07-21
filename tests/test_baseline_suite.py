from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training import baseline_train
from src.training.baseline_train import train_classifier_baselines


def _cfg(**models):
    return {
        "reproducibility": {"seed": 7},
        "models": {
            "run_dummy": models.get("run_dummy", True),
            "run_logistic": models.get("run_logistic", True),
            "run_random_forest": models.get("run_random_forest", True),
        },
        "dummy": {"strategies": ["most_frequent", "stratified"]},
        "logistic_regression": {"max_iter": 200, "class_weight": "balanced", "calibration": "platt"},
        "random_forest": {"n_estimators": 10, "class_weight": "balanced", "n_jobs": 1},
        "thresholding": {"policy": "f1", "min_pred_rate": 0.0, "max_pred_rate": 1.0},
        "report": {"top_k_features": 3},
    }


def _data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(36, 4)).astype(np.float32)
    y = np.array([0, 1] * 18, dtype=np.float32)
    X[30:] = 9999.0
    meta = pd.DataFrame({"worker_id": [f"W{i % 3}" for i in range(len(y))]})
    split = {"train": np.arange(0, 20), "val": np.arange(20, 30), "test": np.arange(30, 36)}
    return X, y, meta, split


def test_dummy_fit_does_not_receive_test_rows(monkeypatch, tmp_path) -> None:
    seen = {}

    class RecordingDummy:
        classes_ = np.array([0.0, 1.0])

        def __init__(self, strategy="most_frequent", random_state=None):
            self.strategy = strategy

        def fit(self, X, y):
            seen[self.strategy] = float(np.max(X))
            self.prob_ = float(np.mean(y))
            return self

        def predict_proba(self, X):
            probs = np.full(len(X), self.prob_)
            return np.column_stack([1.0 - probs, probs])

    monkeypatch.setattr(baseline_train, "DummyClassifier", RecordingDummy)
    monkeypatch.setattr(baseline_train.joblib, "dump", lambda obj, path: path.write_bytes(b"test"))
    X, y, meta, split = _data()
    train_classifier_baselines(
        _cfg(run_logistic=False, run_random_forest=False),
        X,
        ["a", "b", "c", "d"],
        y,
        meta,
        None,
        split,
        tmp_path,
        use_profiles=False,
    )
    assert seen["most_frequent"] < 9999.0
    assert seen["stratified"] < 9999.0


def test_threshold_selection_uses_validation_targets_only(monkeypatch, tmp_path) -> None:
    X, y, meta, split = _data()
    expected_val = y[split["val"]].copy()

    def assert_validation_only(y_true, probs, **kwargs):
        np.testing.assert_array_equal(y_true, expected_val)
        return {
            "threshold": 0.5,
            "policy": "f1",
            "fallback_used": False,
            "fallback_reason": "",
            "val_positive_rate": float(np.mean(probs >= 0.5)),
        }

    monkeypatch.setattr(baseline_train, "select_threshold_from_validation", assert_validation_only)
    train_classifier_baselines(
        _cfg(run_logistic=False, run_random_forest=False),
        X,
        ["a", "b", "c", "d"],
        y,
        meta,
        None,
        split,
        tmp_path,
        use_profiles=False,
    )


def test_dummy_most_frequent_auprc_matches_test_prevalence(tmp_path) -> None:
    X, y, meta, split = _data()
    y[split["train"]] = 0
    y[split["train"][:4]] = 1
    y[split["test"]] = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    out = train_classifier_baselines(
        _cfg(run_logistic=False, run_random_forest=False),
        X,
        ["a", "b", "c", "d"],
        y,
        meta,
        None,
        split,
        tmp_path,
        use_profiles=False,
    )
    metrics = out["dummy_most_frequent"].metrics
    assert metrics["auprc"] == pytest.approx(metrics["prevalence"])


def test_enabled_baselines_write_artifacts_and_metrics(tmp_path) -> None:
    X, y, meta, split = _data()
    out = train_classifier_baselines(
        _cfg(),
        X,
        ["a", "b", "c", "d"],
        y,
        meta,
        None,
        split,
        tmp_path,
        use_profiles=False,
    )
    assert {"dummy_most_frequent", "dummy_stratified", "logistic_regression", "random_forest"} <= set(out)
    for artifact in out.values():
        assert artifact.model_path.exists()
        assert artifact.metrics["probabilities_path"]
        assert artifact.metrics["predictions_path"]
        for key in ["auroc", "auprc", "f1", "specificity", "balanced_accuracy", "brier", "ece", "prevalence"]:
            assert key in artifact.metrics
