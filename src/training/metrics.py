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


def classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    y_pred = probs >= 0.5
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
    try:
        optimal_thr = optimal_threshold(y_true, probs, mode="youden")
    except Exception:
        optimal_thr = 0.5
    optimal_pred = probs >= optimal_thr

    return {
        "auroc": float(auc),
        "auprc": float(auprc),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier),
        "ece": float(ece),
        "confusion_matrix_default": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "optimal_threshold": float(optimal_thr),
        "confusion_matrix_optimal": confusion_matrix(y_true, optimal_pred, labels=[0, 1]).tolist(),
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
