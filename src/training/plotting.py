"""Plotting utilities using matplotlib."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray | list[tuple[str, np.ndarray]], path: Path, label: str = "model") -> None:
    plt.figure()
    if isinstance(y_prob, list):
        for name, probs in y_prob:
            try:
                fpr, tpr, _ = roc_curve(y_true, probs)
                plt.plot(fpr, tpr, label=name)
            except Exception:
                continue
    else:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            plt.plot(fpr, tpr, label=label)
        except Exception:
            pass
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_pr(y_true: np.ndarray, y_prob: np.ndarray | list[tuple[str, np.ndarray]], path: Path, label: str = "model") -> None:
    plt.figure()
    if isinstance(y_prob, list):
        for name, probs in y_prob:
            try:
                precision, recall, _ = precision_recall_curve(y_true, probs)
                plt.plot(recall, precision, label=name)
            except Exception:
                continue
    else:
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            plt.plot(recall, precision, label=label)
        except Exception:
            pass
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray | list[tuple[str, np.ndarray]],
    path: Path,
    bins: int = 10,
    label: str = "model",
) -> None:
    bins_edges = np.linspace(0.0, 1.0, bins + 1)
    plt.figure()
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    series = y_prob if isinstance(y_prob, list) else [(label, y_prob)]
    for name, probs in series:
        centers = []
        accuracies = []
        for start, end in zip(bins_edges[:-1], bins_edges[1:], strict=False):
            mask = (probs >= start) & (probs < end)
            if not np.any(mask):
                continue
            centers.append((start + end) / 2)
            accuracies.append(y_true[mask].mean())
        plt.plot(centers, accuracies, marker="o", label=name)
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical accuracy")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, path: Path, title: str = "Confusion Matrix") -> None:
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_timeseries(
    meta_df,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    workers: Sequence[str] | None = None,
    max_points: int = 100,
) -> None:
    plt.figure(figsize=(10, 5))
    meta = meta_df.copy()
    meta["y_true"] = y_true
    meta["y_pred"] = y_pred
    worker_ids = workers or list(meta["worker_id"].unique())[:3]
    for worker_id in worker_ids:
        subset = meta[meta["worker_id"] == worker_id].sort_values("label_time_idx").head(max_points)
        plt.plot(subset["label_time_idx"], subset["y_true"], label=f"true-{worker_id}", alpha=0.7)
        plt.plot(subset["label_time_idx"], subset["y_pred"], label=f"pred-{worker_id}", linestyle="--")
    plt.xlabel("Time index")
    plt.ylabel("Risk")
    plt.title("Predicted vs True Risk")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_feature_importance(importances: np.ndarray, feature_names: Sequence[str], path: Path, top_k: int = 15) -> None:
    if importances is None or len(importances) == 0:
        return
    idx = np.argsort(importances)[-top_k:][::-1]
    labels = [feature_names[i] for i in idx]
    values = importances[idx]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=45, ha="right")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    path: Path,
    optimal_threshold: float | None = None,
    num_points: int = 100,
) -> None:
    """Plot precision-recall trade-off across classification thresholds.

    Shows how precision and recall change as the decision threshold varies,
    and optionally marks the optimal operating point.
    """
    thresholds = np.linspace(0.0, 1.0, num_points)
    precisions = []
    recalls = []
    f1_scores = []

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label="Precision", linewidth=2)
    plt.plot(thresholds, recalls, label="Recall", linewidth=2)
    plt.plot(thresholds, f1_scores, label="F1 Score", linewidth=2, linestyle="--")

    if optimal_threshold is not None and np.isfinite(optimal_threshold):
        plt.axvline(optimal_threshold, color="red", linestyle=":", linewidth=2, label=f"Optimal ({optimal_threshold:.3f})")

    plt.xlabel("Classification Threshold")
    plt.ylabel("Score")
    plt.title("Precision-Recall Trade-off vs Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_tft_variable_importance(
    importance_dict: dict[str, float],
    path: Path,
    top_k: int = 15,
    title: str = "TFT Variable Importance",
) -> None:
    """Plot TFT variable importance from encoder/decoder variable selection.

    Args:
        importance_dict: Dictionary mapping variable names to importance scores
        path: Output path for the plot
        top_k: Number of top variables to display
        title: Plot title
    """
    if not importance_dict:
        return

    items = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    names = [item[0] for item in items]
    values = [item[1] for item in items]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(values)), values)
    plt.yticks(range(len(values)), names)
    plt.xlabel("Importance Score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
