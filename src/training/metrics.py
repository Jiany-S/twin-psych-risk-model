"""Metrics for classification and regression."""

from __future__ import annotations

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


def _binary_split_stats(y_true: np.ndarray) -> dict[str, float | int]:
    n = int(len(y_true))
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    prev = float(n_pos / n) if n else 0.0
    return {"n_samples": n, "n_pos": n_pos, "n_neg": n_neg, "prevalence": prev}


def _prob_stats(probs: np.ndarray) -> dict[str, float]:
    if len(probs) == 0:
        return {"prob_min": 0.0, "prob_max": 0.0, "prob_mean": 0.0, "prob_std": 0.0}
    return {
        "prob_min": float(np.min(probs)),
        "prob_max": float(np.max(probs)),
        "prob_mean": float(np.mean(probs)),
        "prob_std": float(np.std(probs)),
    }


def select_threshold_from_validation(
    y_true: np.ndarray,
    probs: np.ndarray,
    policy: str = "f1",
    target_recall: float = 0.7,
    target_precision: float = 0.7,
    min_pred_rate: float = 0.02,
    max_pred_rate: float = 0.98,
    allow_pathological: bool = False,
) -> dict[str, Any]:
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    uniq = np.unique(y_true)
    if uniq.size < 2:
        return {
            "threshold": 0.5,
            "policy": policy,
            "fallback_used": True,
            "fallback_reason": "validation_has_single_class",
            "val_positive_rate": float(np.mean(probs >= 0.5)),
        }
    # Use adaptive thresholds from validation probabilities to avoid missing
    # viable operating points when scores are compressed near 0 or 1.
    q = np.linspace(0.001, 0.999, 200)
    quantile_thresholds = np.quantile(probs, q)
    dense_thresholds = np.linspace(0.001, 0.999, 200)
    thresholds = np.unique(np.clip(np.concatenate([quantile_thresholds, dense_thresholds]), 0.0, 1.0))
    candidates: list[dict[str, float]] = []
    for t in thresholds:
        pred = probs >= t
        pr = float(np.mean(pred))
        rec = float(recall_score(y_true, pred, zero_division=0))
        prec = float(precision_score(y_true, pred, zero_division=0))
        f1 = float(f1_score(y_true, pred, zero_division=0))
        if allow_pathological or (min_pred_rate <= pr <= max_pred_rate):
            candidates.append(
                {"threshold": float(t), "pred_rate": pr, "recall": rec, "precision": prec, "f1": f1}
            )

    if not candidates:
        return {
            "threshold": 0.5,
            "policy": policy,
            "fallback_used": True,
            "fallback_reason": "all_thresholds_filtered_by_pred_rate_guardrail",
            "val_positive_rate": float(np.mean(probs >= 0.5)),
        }

    chosen = candidates[0]
    if policy == "recall_at_precision":
        feasible = [c for c in candidates if c["precision"] >= target_precision]
        if feasible:
            chosen = sorted(feasible, key=lambda c: (c["recall"], c["precision"], -abs(c["threshold"] - 0.5)), reverse=True)[
                0
            ]
        else:
            chosen = {"threshold": 0.5, "pred_rate": float(np.mean(probs >= 0.5)), "recall": 0.0, "precision": 0.0, "f1": 0.0}
            return {
                "threshold": 0.5,
                "policy": policy,
                "fallback_used": True,
                "fallback_reason": "no_threshold_meets_target_precision",
                "val_positive_rate": float(np.mean(probs >= 0.5)),
            }
    elif policy == "precision_at_recall":
        feasible = [c for c in candidates if c["recall"] >= target_recall]
        if feasible:
            chosen = sorted(feasible, key=lambda c: (c["precision"], c["recall"], -abs(c["threshold"] - 0.5)), reverse=True)[0]
        else:
            return {
                "threshold": 0.5,
                "policy": policy,
                "fallback_used": True,
                "fallback_reason": "no_threshold_meets_target_recall",
                "val_positive_rate": float(np.mean(probs >= 0.5)),
            }
    else:  # f1
        chosen = sorted(candidates, key=lambda c: (c["f1"], c["precision"], c["recall"]), reverse=True)[0]

    return {
        "threshold": float(chosen["threshold"]),
        "policy": policy,
        "fallback_used": False,
        "fallback_reason": "",
        "val_positive_rate": float(chosen["pred_rate"]),
        "val_f1_at_threshold": float(chosen["f1"]),
        "val_precision_at_threshold": float(chosen["precision"]),
        "val_recall_at_threshold": float(chosen["recall"]),
        "guardrails": {
            "min_pred_rate": float(min_pred_rate),
            "max_pred_rate": float(max_pred_rate),
            "allow_pathological": bool(allow_pathological),
        },
    }


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

    metrics = {
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
    metrics.update(_binary_split_stats(y_true))
    metrics.update(_prob_stats(probs))
    return metrics


def regression_metrics(y_true: np.ndarray, preds: np.ndarray) -> dict[str, Any]:
    mae = mean_absolute_error(y_true, preds)
    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    r2 = r2_score(y_true, preds)
    corr = float(np.corrcoef(y_true, preds)[0, 1]) if len(y_true) > 1 else 0.0
    if np.isnan(corr):
        corr = 0.0
    return {"mae": float(mae), "rmse": rmse, "r2": float(r2), "correlation": corr}
