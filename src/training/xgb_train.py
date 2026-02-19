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
from .metrics import classification_metrics, regression_metrics, select_threshold_from_validation


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
    val_probs = predict_xgb(stress.model, stress.calibrator, X_val, task_type="classification")
    threshold_cfg = cfg.get("thresholding", {})
    thresh_policy = str(
        threshold_cfg.get("policy", cfg.get("xgboost", {}).get("threshold_policy", "f1"))
    ).lower()
    target_recall = float(
        threshold_cfg.get("target_recall", cfg.get("xgboost", {}).get("target_recall", 0.7))
    )
    target_precision = float(
        threshold_cfg.get("target_precision", cfg.get("xgboost", {}).get("target_precision", 0.7))
    )
    min_pred_rate = float(threshold_cfg.get("min_pred_rate", 0.02))
    max_pred_rate = float(threshold_cfg.get("max_pred_rate", 0.98))
    allow_pathological = bool(threshold_cfg.get("allow_pathological", False))
    threshold_diag = select_threshold_from_validation(
        y_stress[val_idx],
        val_probs,
        policy=thresh_policy,
        target_recall=target_recall,
        target_precision=target_precision,
        min_pred_rate=min_pred_rate,
        max_pred_rate=max_pred_rate,
        allow_pathological=allow_pathological,
    )
    chosen_thr = float(threshold_diag["threshold"])
    stress_pred = predict_xgb(stress.model, stress.calibrator, X_test, task_type="classification")
    stress_metrics = classification_metrics(y_stress[test_idx], stress_pred, chosen_threshold=chosen_thr)
    stress_metrics["threshold_policy"] = thresh_policy
    stress_metrics["val_selected_threshold"] = float(chosen_thr)
    stress_metrics["val_positive_rate_at_threshold"] = float(np.mean(val_probs >= chosen_thr))
    stress_metrics["threshold_diagnostics"] = threshold_diag
    stress_metrics["val_prob_stats"] = {
        "min": float(np.min(val_probs)),
        "max": float(np.max(val_probs)),
        "mean": float(np.mean(val_probs)),
        "std": float(np.std(val_probs)),
    }
    stress_metrics["test_positive_count_default"] = int(np.sum(stress_pred >= 0.5))
    stress_metrics["test_positive_count_optimal"] = int(np.sum(stress_pred >= chosen_thr))
    stress_metrics["pred_min"] = float(np.min(stress_pred))
    stress_metrics["pred_max"] = float(np.max(stress_pred))
    stress_metrics["n_predictions"] = int(len(stress_pred))
    stress_metrics["n_targets"] = int(len(y_stress[test_idx]))
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
