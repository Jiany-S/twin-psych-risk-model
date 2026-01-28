"""Train and evaluate XGBoost model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..data.windowing import add_ttc_features, engineer_window_features
from ..models.xgb_model import predict_xgb, train_xgboost
from ..training.metrics import classification_metrics, regression_metrics


@dataclass
class XGBArtifacts:
    predictions: np.ndarray
    metrics: dict[str, Any]
    feature_names: list[str]
    model_path: Path


def prepare_features(
    windows: np.ndarray,
    meta: pd.DataFrame,
    static_profiles: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, list[str]]:
    base_features, feature_names = engineer_window_features(windows, feature_cols)
    base_features, feature_names = add_ttc_features(base_features, feature_names, windows, feature_cols)
    profiles = static_profiles.set_index("worker_id")
    static = meta["worker_id"].astype(str).map(lambda w: profiles.loc[w] if w in profiles.index else None).apply(pd.Series)
    static = static.fillna(0.0)
    static_features = static.drop(columns=["worker_id"], errors="ignore").to_numpy(dtype=float)
    feature_names += [c for c in static.columns if c != "worker_id"]
    feature_matrix = np.concatenate([base_features, static_features], axis=1)
    return feature_matrix, feature_names


def train_xgb_pipeline(
    cfg: dict[str, Any],
    windows: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
    static_profiles: pd.DataFrame,
    feature_cols: list[str],
    split_indices: dict[str, np.ndarray],
    task_type: str,
    run_dir: Path,
) -> XGBArtifacts:
    feature_matrix, feature_names = prepare_features(windows, meta, static_profiles, feature_cols)

    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]

    X_train, y_train = feature_matrix[train_idx], labels[train_idx]
    X_val, y_val = feature_matrix[val_idx], labels[val_idx]
    X_test, y_test = feature_matrix[test_idx], labels[test_idx]

    result = train_xgboost(X_train, y_train, X_val, y_val, task_type, cfg["xgboost"], feature_names)
    preds = predict_xgb(result.model, result.calibrator, X_test, task_type)

    if task_type == "classification":
        metrics = classification_metrics(y_test, preds)
    else:
        metrics = regression_metrics(y_test, preds)

    model_path = run_dir / "models" / "xgboost_model.pkl"
    joblib.dump({"model": result.model, "calibrator": result.calibrator, "feature_names": feature_names}, model_path)

    metrics["feature_importance_topk"] = _feature_importance(result.model, feature_names, cfg["report"]["top_k_features"])
    metrics["best_params"] = cfg.get("xgboost", {})
    return XGBArtifacts(predictions=preds, metrics=metrics, feature_names=feature_names, model_path=model_path)


def _feature_importance(model: Any, feature_names: list[str], top_k: int) -> list[dict[str, float]]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    idx = np.argsort(importances)[-top_k:][::-1]
    return [{"feature": feature_names[i], "importance": float(importances[i])} for i in idx]
