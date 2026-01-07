"""Baseline models for cognitive risk forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn


def assemble_tabular_features(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract numpy arrays suitable for classic ML models."""
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for baseline features: {missing}")
    features = df[feature_columns].fillna(0.0).to_numpy(dtype=float)
    targets = df[target_column].to_numpy(dtype=float)
    return features, targets


class LogisticRegressionBaseline:
    """Simple logistic regression classifier on tabular snapshots."""

    def __init__(self, penalty: str = "l2", class_weight: str | None = "balanced", max_iter: int = 500) -> None:
        self.model = LogisticRegression(
            penalty=penalty,
            max_iter=max_iter,
            class_weight=class_weight,
            solver="liblinear" if penalty == "l1" else "lbfgs",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        import joblib

        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "LogisticRegressionBaseline":
        import joblib

        baseline = cls()
        baseline.model = joblib.load(path)
        return baseline


class SequenceGRUClassifier(nn.Module):
    """GRU-based sequence classifier used as a neural baseline."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.gru(x)
        logits = self.head(outputs[:, -1])
        return logits.squeeze(-1)


def build_sequence_tensors(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    worker_column: str,
    window_length: int,
    horizon: int,
    time_col: str = "time_idx",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create sliding windows for sequence models."""
    sequences: list[np.ndarray] = []
    labels: list[float] = []
    for worker_id, worker_df in df.groupby(worker_column):
        worker_df = worker_df.sort_values(time_col)
        values = worker_df[feature_columns].to_numpy(dtype=float)
        targets = worker_df[target_column].to_numpy(dtype=float)
        for start in range(0, len(worker_df) - window_length - horizon + 1):
            end = start + window_length
            label_idx = end + horizon - 1
            seq = values[start:end]
            label = targets[label_idx]
            sequences.append(seq)
            labels.append(label)

    if not sequences:
        raise ValueError("Insufficient data to build GRU sequences. Check window_length and horizon.")

    X = torch.tensor(np.stack(sequences), dtype=torch.float32)
    y = torch.tensor(np.array(labels), dtype=torch.float32)
    return X, y
