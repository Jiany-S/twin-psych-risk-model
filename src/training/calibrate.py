"""Temperature scaling for calibration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch


class TemperatureScaler(torch.nn.Module):
    """Simple temperature scaling module for binary logits."""

    def __init__(self) -> None:
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = torch.clamp(self.temperature, min=1e-3)
        return logits / temperature

    def fit(self, logits: np.ndarray, targets: np.ndarray, lr: float = 0.01, max_iter: int = 500) -> float:
        """Optimize temperature by minimizing negative log-likelihood."""
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def _closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(self.forward(logits_tensor), targets_tensor)
            loss.backward()
            return loss

        optimizer.step(_closure)
        return float(torch.clamp(self.temperature.detach(), min=1e-3))

    def transform_proba(self, probs: np.ndarray) -> np.ndarray:
        logits = probs_to_logits(probs)
        calibrated = self.forward(torch.tensor(logits, dtype=torch.float32))
        return torch.sigmoid(calibrated).numpy()


def probs_to_logits(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(probs, eps, 1 - eps)
    return np.log(clipped / (1 - clipped))


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-logits))


def calibrate_predictions(prediction_file: Path, output_file: Path) -> None:
    data = np.genfromtxt(prediction_file, delimiter=",", names=True)
    if "prob" not in data.dtype.names or "target" not in data.dtype.names:
        raise ValueError("Prediction file must contain 'prob' and 'target' columns.")
    scaler = TemperatureScaler()
    scaler.fit(probs_to_logits(data["prob"]), data["target"])
    calibrated = scaler.transform_proba(data["prob"])
    np.savetxt(
        output_file,
        np.vstack([data["target"], calibrated]).T,
        delimiter=",",
        header="target,calibrated_prob",
        comments="",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temperature scale a set of predictions.")
    parser.add_argument("--predictions", type=str, required=True, help="CSV with columns target,prob.")
    parser.add_argument("--output", type=str, required=True, help="Where to write the calibrated CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibrate_predictions(Path(args.predictions), Path(args.output))


if __name__ == "__main__":
    main()
