"""Metrics for classification and regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for start, end in zip(bin_edges[:-1], bin_edges[1:], strict=False):
        mask = (probs >= start) & (probs < end)
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(acc - conf)
    return float(ece)


def optimal_threshold(y_true: np.ndarray, probs: np.ndarray, mode: str = "youden") -> float:
    if mode == "f1":
        thresholds = np.linspace(0.05, 0.95, 50)
        scores = [f1_score(y_true, probs >= t) for t in thresholds]
        return float(thresholds[int(np.argmax(scores))])
    fpr, tpr, thr = roc_curve(y_true, probs)
    youden = tpr - fpr
    return float(thr[int(np.argmax(youden))])


def select_threshold_from_validation(
    y_true: np.ndarray,
    probs: np.ndarray,
    policy: str = "f1",
    target_recall: float = 0.7,
) -> float:
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    thresholds = np.linspace(0.01, 0.99, 99)
    if policy == "target_recall":
        best_thr = 0.5
        best_prec = -1.0
        for t in thresholds:
            pred = probs >= t
            rec = recall_score(y_true, pred, zero_division=0)
            if rec >= target_recall:
                prec = precision_score(y_true, pred, zero_division=0)
                if prec > best_prec:
                    best_prec = prec
                    best_thr = float(t)
        return float(best_thr)
    # default: maximize F1
    best_thr = 0.5
    best_f1 = -1.0
    for t in thresholds:
        pred = probs >= t
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thr = float(t)
    return float(best_thr)


def classification_metrics(y_true: np.ndarray, probs: np.ndarray, chosen_threshold: float | None = None) -> dict[str, Any]:
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    y_pred_default = probs >= 0.5
    threshold = float(chosen_threshold if chosen_threshold is not None else 0.5)
    y_pred_optimal = probs >= threshold
    unique = np.unique(y_true)
    if unique.size < 2:
        auc = float("nan")
        auprc = float("nan")
    else:
        try:
            auc = roc_auc_score(y_true, probs)
        except Exception:
            auc = float("nan")
        auprc = average_precision_score(y_true, probs)
    brier = brier_score_loss(y_true, probs)
    ece = expected_calibration_error(probs, y_true, bins=15)
    default_pos_rate = float(np.mean(y_pred_default))
    optimal_pos_rate = float(np.mean(y_pred_optimal))

    return {
        "auroc": float(auc),
        "auprc": float(auprc),
        "accuracy": float(accuracy_score(y_true, y_pred_optimal)),
        "f1": float(f1_score(y_true, y_pred_optimal, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred_optimal, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_optimal, zero_division=0)),
        "brier": float(brier),
        "ece": float(ece),
        "confusion_matrix_default": confusion_matrix(y_true, y_pred_default, labels=[0, 1]).tolist(),
        "optimal_threshold": float(threshold),
        "chosen_threshold": float(threshold),
        "confusion_matrix_optimal": confusion_matrix(y_true, y_pred_optimal, labels=[0, 1]).tolist(),
        "default_positive_rate": default_pos_rate,
        "optimal_positive_rate": optimal_pos_rate,
        "class_counts": {str(k): int((y_true == k).sum()) for k in np.unique(y_true)},
    }


def regression_metrics(y_true: np.ndarray, preds: np.ndarray) -> dict[str, Any]:
    mae = mean_absolute_error(y_true, preds)
    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    r2 = r2_score(y_true, preds)
    corr = float(np.corrcoef(y_true, preds)[0, 1]) if len(y_true) > 1 else 0.0
    if np.isnan(corr):
        corr = 0.0
    return {"mae": float(mae), "rmse": rmse, "r2": float(r2), "correlation": corr}
