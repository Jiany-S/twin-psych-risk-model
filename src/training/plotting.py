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
