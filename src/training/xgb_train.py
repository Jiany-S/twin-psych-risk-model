"""Train and evaluate XGBoost models for stress and comfort tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..data.features import impute_with_train_statistics
from ..models.xgb_model import predict_xgb, train_xgboost
from .metrics import classification_metrics, regression_metrics


@dataclass
class XGBTaskArtifacts:
    predictions: np.ndarray
    metrics: dict[str, Any]
    model_path: Path
    feature_importance: list[dict[str, float]]


def _append_static_features(
    X: np.ndarray,
    meta: pd.DataFrame,
    static_profiles: pd.DataFrame | None,
    use_profiles: bool,
) -> tuple[np.ndarray, list[str]]:
    if not use_profiles or static_profiles is None or static_profiles.empty:
        return X, []
    profiles = static_profiles.set_index("worker_id")
    mapped = (
        meta["worker_id"]
        .astype(str)
        .map(lambda w: profiles.loc[w] if w in profiles.index else None)
        .apply(pd.Series)
        .fillna(0.0)
    )
    mapped = mapped.drop(columns=["worker_id"], errors="ignore")
    names = list(mapped.columns)
    return np.concatenate([X, mapped.to_numpy(dtype=float)], axis=1), names


def _importance(model: Any, feature_names: list[str], top_k: int) -> list[dict[str, float]]:
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return []
    idx = np.argsort(values)[-top_k:][::-1]
    return [{"feature": feature_names[i], "importance": float(values[i])} for i in idx]


def train_xgb_tasks(
    cfg: dict[str, Any],
    feature_matrix: np.ndarray,
    feature_names: list[str],
    y_stress: np.ndarray,
    y_comfort: np.ndarray,
    meta: pd.DataFrame,
    static_profiles: pd.DataFrame | None,
    split_indices: dict[str, np.ndarray],
    run_dir: Path,
    use_profiles: bool,
    model_prefix: str = "xgb",
) -> dict[str, XGBTaskArtifacts]:
    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]

    X, static_names = _append_static_features(feature_matrix, meta, static_profiles, use_profiles)
    names = feature_names + static_names
    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]
    X_train, X_val, X_test = impute_with_train_statistics(X_train, X_val, X_test)
    if np.isnan(X_train).any() or np.isnan(X_val).any() or np.isnan(X_test).any():
        raise ValueError("NaNs remain after train-fitted imputation in XGBoost features.")

    artifacts: dict[str, XGBTaskArtifacts] = {}

    # Stress classification
    stress = train_xgboost(
        X_train,
        y_stress[train_idx],
        X_val,
        y_stress[val_idx],
        task_type="classification",
        params=cfg["xgboost"],
        feature_names=names,
    )
    stress_pred = predict_xgb(stress.model, stress.calibrator, X_test, task_type="classification")
    stress_metrics = classification_metrics(y_stress[test_idx], stress_pred)
    stress_imp = _importance(stress.model, names, cfg["report"]["top_k_features"])
    stress_path = run_dir / "models" / f"{model_prefix}_stress.pkl"
    joblib.dump({"model": stress.model, "calibrator": stress.calibrator, "feature_names": names}, stress_path)
    artifacts["stress"] = XGBTaskArtifacts(
        predictions=stress_pred,
        metrics=stress_metrics,
        model_path=stress_path,
        feature_importance=stress_imp,
    )

    # Comfort regression
    comfort = train_xgboost(
        X_train,
        y_comfort[train_idx],
        X_val,
        y_comfort[val_idx],
        task_type="regression",
        params=cfg["xgboost"],
        feature_names=names,
    )
    comfort_pred = predict_xgb(comfort.model, calibrator=None, X=X_test, task_type="regression")
    comfort_metrics = regression_metrics(y_comfort[test_idx], comfort_pred)
    comfort_imp = _importance(comfort.model, names, cfg["report"]["top_k_features"])
    comfort_path = run_dir / "models" / f"{model_prefix}_comfort.pkl"
    joblib.dump({"model": comfort.model, "feature_names": names}, comfort_path)
    artifacts["comfort"] = XGBTaskArtifacts(
        predictions=comfort_pred,
        metrics=comfort_metrics,
        model_path=comfort_path,
        feature_importance=comfort_imp,
    )
    return artifacts
