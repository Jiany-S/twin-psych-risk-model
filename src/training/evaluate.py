"""Model evaluation script."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from ..data.schema import DataSchema
from ..data.splits import temporal_split
from ..models.baselines import (
    LogisticRegressionBaseline,
    SequenceGRUClassifier,
    assemble_tabular_features,
    build_sequence_tensors,
)
from ..models.tft_model import build_tft_dataset
from ..training.calibrate import TemperatureScaler, probs_to_logits, logits_to_probs
from ..utils.logging import configure_logging


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, num_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    total = len(targets)
    for start, end in zip(bins[:-1], bins[1:], strict=False):
        mask = (probs >= start) & (probs < end)
        if not np.any(mask):
            continue
        bin_prob = probs[mask]
        bin_target = targets[mask]
        accuracy = bin_target.mean()
        confidence = bin_prob.mean()
        ece += (mask.sum() / total) * abs(accuracy - confidence)
    return float(ece)


def maybe_calibrate(val_logits: np.ndarray, val_targets: np.ndarray, test_logits: np.ndarray, enable: bool) -> np.ndarray:
    if not enable:
        return logits_to_probs(test_logits)
    scaler = TemperatureScaler()
    scaler.fit(val_logits, val_targets)
    calibrated_logits = scaler(torch.tensor(test_logits, dtype=torch.float32)).numpy()
    return logits_to_probs(calibrated_logits)


def evaluate_logistic(cfg: dict[str, Any], schema: DataSchema, df: pd.DataFrame, calibrate: bool, logger) -> None:
    baseline_dir = Path(cfg["paths"]["artifacts_dir"]) / "baselines"
    model_path = baseline_dir / "logistic_regression.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Logistic regression model not found at {model_path}.")
    model = LogisticRegressionBaseline.load(model_path.as_posix())
    feature_cols = cfg["baseline"]["feature_columns"]
    split = temporal_split(df, time_col=schema.time_idx, val_ratio=cfg["data"]["val_ratio"], test_ratio=cfg["data"]["test_ratio"])
    X_val, y_val = assemble_tabular_features(split.val, feature_cols, schema.target)
    X_test, y_test = assemble_tabular_features(split.test, feature_cols, schema.target)
    val_probs = model.predict_proba(X_val)
    test_probs = model.predict_proba(X_test)
    val_logits = probs_to_logits(val_probs)
    test_logits = probs_to_logits(test_probs)
    calibrated_probs = maybe_calibrate(val_logits, y_val, test_logits, calibrate)
    report_metrics("Logistic Regression", calibrated_probs, y_test, cfg, logger)


def evaluate_gru(cfg: dict[str, Any], schema: DataSchema, df: pd.DataFrame, calibrate: bool, logger) -> None:
    baseline_dir = Path(cfg["paths"]["artifacts_dir"]) / "baselines"
    model_path = baseline_dir / "gru_baseline.pt"
    metadata_path = baseline_dir / "gru_baseline.json"
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("GRU baseline artifacts are missing. Train the GRU model first.")
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    feature_cols = metadata["feature_columns"]
    window = metadata["window_length"]
    horizon = metadata["prediction_horizon"]
    split = temporal_split(df, time_col=schema.time_idx, val_ratio=cfg["data"]["val_ratio"], test_ratio=cfg["data"]["test_ratio"])
    X_val, y_val = build_sequence_tensors(
        split.val,
        feature_cols,
        schema.target,
        schema.worker_id,
        window,
        horizon,
        time_col=schema.time_idx,
    )
    X_test, y_test = build_sequence_tensors(
        split.test,
        feature_cols,
        schema.target,
        schema.worker_id,
        window,
        horizon,
        time_col=schema.time_idx,
    )

    device = torch.device("cpu")
    model = SequenceGRUClassifier(
        input_size=X_val.shape[-1],
        hidden_size=metadata.get("hidden_size", cfg["baseline"]["gru_hidden_size"]),
        num_layers=metadata.get("num_layers", cfg["baseline"]["gru_layers"]),
        dropout=metadata.get("dropout", cfg["baseline"]["gru_dropout"]),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        val_logits = model(X_val).detach().numpy()
        test_logits = model(X_test).detach().numpy()
    calibrated_probs = maybe_calibrate(val_logits, y_val.numpy(), test_logits, calibrate)
    report_metrics("GRU Baseline", calibrated_probs, y_test.numpy(), cfg, logger)


def evaluate_tft(cfg: dict[str, Any], schema: DataSchema, df: pd.DataFrame, calibrate: bool, logger) -> None:
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting import TemporalFusionTransformer

    artifacts_dir = Path(cfg["paths"]["artifacts_dir"]) / "tft"
    checkpoints = sorted(artifacts_dir.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No TFT checkpoints found in {artifacts_dir}.")
    checkpoint_path = checkpoints[0]

    split = temporal_split(df, time_col=schema.time_idx, val_ratio=cfg["data"]["val_ratio"], test_ratio=cfg["data"]["test_ratio"])
    train_ds, val_ds = build_tft_dataset(
        split.train,
        split.val,
        schema=schema,
        max_encoder_length=cfg["data"]["window_length"],
        max_prediction_length=cfg["data"]["prediction_horizon"],
    )
    test_ds = TimeSeriesDataSet.from_dataset(train_ds, split.test, predict=True, stop_randomization=True)
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]
    val_loader = val_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)
    val_target_loader = val_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)
    test_loader = test_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)
    test_target_loader = test_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)

    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path, map_location="cpu")
    trainer = pl.Trainer(logger=False, enable_checkpointing=False, accelerator="cpu", devices=1)

    def _flatten_prediction(pred_batches: list[torch.Tensor]) -> np.ndarray:
        tensor = torch.cat(pred_batches)
        if tensor.ndim > 2:
            tensor = tensor[..., -1]
        elif tensor.ndim == 2:
            tensor = tensor[:, -1]
        return tensor.detach().cpu().numpy()

    val_preds = _flatten_prediction(trainer.predict(model, dataloaders=val_loader))
    test_preds = _flatten_prediction(trainer.predict(model, dataloaders=test_loader))

    def _collect_targets(loader) -> np.ndarray:
        targets = []
        for _, y in loader:
            targets.append(y[0][:, -1])
        return torch.cat(targets).detach().cpu().numpy()

    val_targets = _collect_targets(val_target_loader)
    test_targets = _collect_targets(test_target_loader)
    val_logits = probs_to_logits(np.clip(val_preds, 1e-6, 1 - 1e-6))
    test_logits = probs_to_logits(np.clip(test_preds, 1e-6, 1 - 1e-6))
    calibrated_probs = maybe_calibrate(val_logits, val_targets, test_logits, calibrate)
    report_metrics("Temporal Fusion Transformer", calibrated_probs, test_targets, cfg, logger)


def report_metrics(model_name: str, probs: np.ndarray, targets: np.ndarray, cfg: dict[str, Any], logger) -> None:
    auc = roc_auc_score(targets, probs)
    auprc = average_precision_score(targets, probs)
    ece = expected_calibration_error(probs, targets, num_bins=cfg["evaluation"]["ece_bins"])
    logger.info("%s — AUROC: %.3f | AUPRC: %.3f | ECE: %.3f", model_name, auc, auprc, ece)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained models.")
    parser.add_argument("--config", type=str, default="src/config/default.yaml")
    parser.add_argument(
        "--model",
        type=str,
        choices=("logistic", "gru", "tft", "all"),
        default="all",
        help="Which model family to evaluate.",
    )
    parser.add_argument("--calibrate", action="store_true", help="Apply temperature scaling using validation data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    schema = DataSchema.from_config(cfg)
    processed_path = Path(cfg["paths"]["processed_train_file"])
    if not processed_path.exists():
        raise FileNotFoundError("Processed dataset missing. Run preprocessing first.")
    df = pd.read_csv(processed_path)
    logger = configure_logging("evaluate")

    if args.model in ("logistic", "all"):
        evaluate_logistic(cfg, schema, df, args.calibrate, logger)
    if args.model in ("gru", "all"):
        evaluate_gru(cfg, schema, df, args.calibrate, logger)
    if args.model in ("tft", "all"):
        evaluate_tft(cfg, schema, df, args.calibrate, logger)


if __name__ == "__main__":
    main()
