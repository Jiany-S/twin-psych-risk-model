"""Classical classification baselines for the primary target."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..data.features import impute_with_train_statistics
from .metrics import classification_metrics, select_threshold_from_validation
from .xgb_train import _append_static_features


@dataclass
class ClassifierArtifacts:
    predictions: np.ndarray
    binary_predictions: np.ndarray
    metrics: dict[str, Any]
    model_path: Path
    feature_importance: list[dict[str, float]]


def _positive_proba(model: Any, X: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        return probs[:, classes.index(1)]
    return np.zeros(len(X), dtype=float)


def _fit_platt_from_validation(val_probs: np.ndarray, y_val: np.ndarray) -> LogisticRegression | None:
    if np.unique(y_val).size < 2 or np.unique(val_probs).size < 2:
        return None
    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(val_probs.reshape(-1, 1), y_val)
    return calibrator


def _apply_calibrator(probs: np.ndarray, calibrator: LogisticRegression | None) -> np.ndarray:
    if calibrator is None:
        return probs
    return calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]


def _threshold_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    threshold_cfg = cfg.get("thresholding", {})
    return {
        "policy": str(threshold_cfg.get("policy", cfg.get("xgboost", {}).get("threshold_policy", "f1"))).lower(),
        "target_recall": float(threshold_cfg.get("target_recall", cfg.get("xgboost", {}).get("target_recall", 0.7))),
        "target_precision": float(
            threshold_cfg.get("target_precision", cfg.get("xgboost", {}).get("target_precision", 0.7))
        ),
        "min_pred_rate": float(threshold_cfg.get("min_pred_rate", 0.02)),
        "max_pred_rate": float(threshold_cfg.get("max_pred_rate", 0.98)),
        "allow_pathological": bool(threshold_cfg.get("allow_pathological", False)),
    }


def _evaluate_classifier(
    cfg: dict[str, Any],
    model_name: str,
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    run_dir: Path,
    artifact: dict[str, Any],
    feature_importance: list[dict[str, float]] | None = None,
    calibrator: LogisticRegression | None = None,
) -> ClassifierArtifacts:
    val_probs = _apply_calibrator(_positive_proba(model, X_val), calibrator)
    test_probs = _apply_calibrator(_positive_proba(model, X_test), calibrator)
    tcfg = _threshold_cfg(cfg)
    threshold_diag = select_threshold_from_validation(y_val, val_probs, **tcfg)
    chosen_thr = float(threshold_diag["threshold"])
    binary = test_probs >= chosen_thr
    metrics = classification_metrics(y_test, test_probs, chosen_threshold=chosen_thr)
    metrics["threshold_policy"] = tcfg["policy"]
    metrics["threshold_diagnostics"] = threshold_diag
    metrics["val_selected_threshold"] = chosen_thr
    metrics["val_positive_rate_at_threshold"] = float(np.mean(val_probs >= chosen_thr))
    metrics["val_prob_stats"] = {
        "min": float(np.min(val_probs)),
        "max": float(np.max(val_probs)),
        "mean": float(np.mean(val_probs)),
        "std": float(np.std(val_probs)),
    }
    metrics["test_positive_count_default"] = int(np.sum(test_probs >= 0.5))
    metrics["test_positive_count_optimal"] = int(np.sum(binary))
    metrics["pred_min"] = float(np.min(test_probs))
    metrics["pred_max"] = float(np.max(test_probs))
    metrics["n_predictions"] = int(len(test_probs))
    metrics["n_targets"] = int(len(y_test))
    metrics["model_family"] = model_name

    model_dir = run_dir / "models"
    pred_dir = run_dir / "predictions"
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name}_primary.pkl"
    joblib.dump(artifact | {"model": model, "calibrator": calibrator}, model_path)
    prob_path = pred_dir / f"{model_name}_primary_probs.npy"
    pred_path = pred_dir / f"{model_name}_primary_predictions.npy"
    np.save(prob_path, test_probs)
    np.save(pred_path, binary.astype(np.float32))
    metrics["probabilities_path"] = str(prob_path)
    metrics["predictions_path"] = str(pred_path)

    return ClassifierArtifacts(
        predictions=test_probs,
        binary_predictions=binary.astype(np.float32),
        metrics=metrics,
        model_path=model_path,
        feature_importance=feature_importance or [],
    )


def _importance_from_values(values: np.ndarray, feature_names: list[str], top_k: int) -> list[dict[str, float]]:
    if values.size == 0:
        return []
    idx = np.argsort(values)[-top_k:][::-1]
    return [{"feature": feature_names[i], "importance": float(values[i])} for i in idx]


def train_classifier_baselines(
    cfg: dict[str, Any],
    feature_matrix: np.ndarray,
    feature_names: list[str],
    y_primary: np.ndarray,
    meta: pd.DataFrame,
    static_profiles: pd.DataFrame | None,
    split_indices: dict[str, np.ndarray],
    run_dir: Path,
    use_profiles: bool,
) -> dict[str, ClassifierArtifacts]:
    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]

    X, static_names = _append_static_features(feature_matrix, meta, static_profiles, use_profiles)
    names = feature_names + static_names
    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]
    X_train, X_val, X_test = impute_with_train_statistics(X_train, X_val, X_test)
    y_train = y_primary[train_idx]
    y_val = y_primary[val_idx]
    y_test = y_primary[test_idx]
    top_k = int(cfg["report"]["top_k_features"])
    models_cfg = cfg.get("models", {})
    artifacts: dict[str, ClassifierArtifacts] = {}

    if bool(models_cfg.get("run_dummy", True)):
        dummy_cfg = cfg.get("dummy", {})
        strategies = dummy_cfg.get("strategies", ["most_frequent", "stratified"])
        for strategy in strategies:
            name = f"dummy_{strategy}"
            model = DummyClassifier(strategy=str(strategy), random_state=int(cfg.get("reproducibility", {}).get("seed", 42)))
            model.fit(X_train, y_train)
            artifacts[name] = _evaluate_classifier(
                cfg, name, model, X_val, y_val, X_test, y_test, run_dir, {"feature_names": names, "strategy": strategy}
            )

    if bool(models_cfg.get("run_logistic", True)):
        log_cfg = cfg.get("logistic_regression", {})
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)
        class_weight = log_cfg.get("class_weight")
        model = LogisticRegression(
            max_iter=int(log_cfg.get("max_iter", 1000)),
            C=float(log_cfg.get("C", 1.0)),
            class_weight=class_weight,
            solver=str(log_cfg.get("solver", "lbfgs")),
            random_state=int(cfg.get("reproducibility", {}).get("seed", 42)),
        )
        model.fit(X_train_s, y_train)
        val_probs_uncal = _positive_proba(model, X_val_s)
        calibrator = (
            _fit_platt_from_validation(val_probs_uncal, y_val)
            if str(log_cfg.get("calibration", "none")).lower() == "platt"
            else None
        )
        coef = np.abs(np.ravel(getattr(model, "coef_", np.array([]))))
        artifacts["logistic_regression"] = _evaluate_classifier(
            cfg,
            "logistic_regression",
            model,
            X_val_s,
            y_val,
            X_test_s,
            y_test,
            run_dir,
            {"feature_names": names, "scaler": scaler, "class_weight": class_weight},
            feature_importance=_importance_from_values(coef, names, top_k),
            calibrator=calibrator,
        )

    if bool(models_cfg.get("run_random_forest", True)):
        rf_cfg = cfg.get("random_forest", {})
        class_weight = rf_cfg.get("class_weight")
        model = RandomForestClassifier(
            n_estimators=int(rf_cfg.get("n_estimators", 200)),
            max_depth=rf_cfg.get("max_depth"),
            min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 1)),
            class_weight=class_weight,
            random_state=int(cfg.get("reproducibility", {}).get("seed", 42)),
            n_jobs=int(rf_cfg.get("n_jobs", -1)),
        )
        model.fit(X_train, y_train)
        values = np.asarray(getattr(model, "feature_importances_", np.array([])))
        artifacts["random_forest"] = _evaluate_classifier(
            cfg,
            "random_forest",
            model,
            X_val,
            y_val,
            X_test,
            y_test,
            run_dir,
            {"feature_names": names, "class_weight": class_weight},
            feature_importance=_importance_from_values(values, names, top_k),
        )

    return artifacts
