"""Train and evaluate TFT model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
try:
    import lightning.pytorch as pl
except Exception:  # fallback for older installs
    import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from ..models.tft_model import build_tft_datasets, create_tft_model
from ..training.metrics import classification_metrics, regression_metrics


@dataclass
class TFTArtifacts:
    predictions: np.ndarray
    targets: np.ndarray
    metrics: dict[str, Any]
    checkpoint_path: Path


def train_tft_pipeline(
    cfg: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema,
    task_type: str,
    run_dir: Path,
    window_length: int,
    horizon: int,
) -> TFTArtifacts:
    tft_cfg = cfg["tft"]
    for df in (train_df, val_df, test_df):
        df["worker_id"] = df["worker_id"].astype(str)
        df["specialization_id"] = df["specialization_id"].astype(str)
        df["experience_level"] = df["experience_level"].astype(str)
        df["task_phase"] = df["task_phase"].astype(str)
    train_ds, val_ds = build_tft_datasets(
        train_df,
        val_df,
        schema=schema,
        window_length=window_length,
        horizon=horizon,
        known_categoricals=["task_phase"],
    )
    test_ds = train_ds.from_dataset(train_ds, test_df, predict=True, stop_randomization=True)

    train_loader = train_ds.to_dataloader(train=True, batch_size=tft_cfg["batch_size"], num_workers=tft_cfg["num_workers"])
    val_loader = val_ds.to_dataloader(train=False, batch_size=tft_cfg["batch_size"], num_workers=tft_cfg["num_workers"])
    test_loader = test_ds.to_dataloader(train=False, batch_size=tft_cfg["batch_size"], num_workers=tft_cfg["num_workers"])

    model = create_tft_model(train_ds, tft_cfg)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=tft_cfg["early_stop_patience"], mode="min"),
        ModelCheckpoint(
            dirpath=run_dir / "models",
            filename="tft-{epoch:02d}-{val_loss:.3f}",
            monitor="val_loss",
            save_top_k=1,
        ),
    ]
    try:
        from pytorch_forecasting.models.base import PredictCallback

        callbacks.append(PredictCallback())
    except Exception:
        pass
    accelerator = "gpu" if tft_cfg.get("gpus", 0) > 0 and torch.cuda.is_available() else "cpu"
    devices = min(int(tft_cfg.get("gpus", 0)), torch.cuda.device_count()) if accelerator == "gpu" else 1
    trainer = pl.Trainer(
        max_epochs=tft_cfg["max_epochs"],
        callbacks=callbacks,
        accelerator=accelerator,
        devices=max(devices, 1),
        logger=False,
        enable_checkpointing=True,
    )
    trainer.fit(model, train_loader, val_loader)

    ckpt_path = Path(callbacks[1].best_model_path) if callbacks[1].best_model_path else run_dir / "models" / "tft_last.ckpt"
    if not ckpt_path.exists():
        trainer.save_checkpoint(ckpt_path)

    # Predict (prefer PF utility for compatibility)
    predictions = None
    y_true = None
    if hasattr(model, "predict"):
        try:
            pred_out = model.predict(test_loader, return_y=True)
            if isinstance(pred_out, tuple) and len(pred_out) == 2:
                pred_tensor, y_batch = pred_out
            else:
                pred_tensor, y_batch = pred_out, None
            if hasattr(pred_tensor, "prediction"):
                pred_tensor = pred_tensor.prediction
            elif hasattr(pred_tensor, "output"):
                pred_tensor = pred_tensor.output
            if isinstance(pred_tensor, torch.Tensor):
                pred_tensor = pred_tensor.detach().cpu()
            if pred_tensor.ndim > 2:
                pred_tensor = pred_tensor[..., -1]
            elif pred_tensor.ndim == 2:
                pred_tensor = pred_tensor[:, -1]
            predictions = pred_tensor.numpy()
            if y_batch is not None:
                y_tensor = y_batch
                if hasattr(y_tensor, "target"):
                    y_tensor = y_tensor.target
                elif hasattr(y_tensor, "y"):
                    y_tensor = y_tensor.y
                if isinstance(y_tensor, (list, tuple)):
                    y_tensor = y_tensor[0]
                if isinstance(y_tensor, torch.Tensor):
                    if y_tensor.ndim > 1:
                        y_tensor = y_tensor[:, -1]
                    y_true = y_tensor.detach().cpu().numpy()
        except Exception as exc:
            raise RuntimeError("TFT prediction failed; model.predict() did not succeed.") from exc
    else:
        raise RuntimeError("TFT model does not support predict(). Update pytorch-forecasting.")

    def _extract_targets(batch) -> torch.Tensor:
        if isinstance(batch, torch.Tensor):
            return batch
        if isinstance(batch, (list, tuple)):
            if len(batch) == 0:
                raise ValueError("Empty target batch from dataloader.")
            return _extract_targets(batch[0])
        if isinstance(batch, dict):
            if "target" in batch:
                return _extract_targets(batch["target"])
            raise ValueError(f"Unknown target dict keys: {list(batch.keys())}")
        raise ValueError(f"Unsupported target batch type: {type(batch)}")

    if y_true is None:
        targets = []
        for batch in test_loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                _, y = batch[0], batch[1]
            else:
                raise ValueError(f"Unexpected batch structure: {type(batch)}")
            try:
                y_tensor = _extract_targets(y)
            except Exception as exc:
                raise ValueError(f"Failed to extract TFT targets. Batch target type: {type(y)}") from exc
            if y_tensor.ndim > 1:
                y_tensor = y_tensor[:, -1]
            targets.append(y_tensor.detach().cpu())
        y_true = torch.cat(targets).numpy()

    if task_type == "classification":
        loss_mode = cfg["tft"].get("tft_loss", "quantile")
        if loss_mode == "bce":
            predictions = 1 / (1 + np.exp(-predictions))
        predictions = np.clip(predictions, 0.0, 1.0)
        metrics = classification_metrics(y_true, predictions)
    else:
        metrics = regression_metrics(y_true, predictions)

    return TFTArtifacts(predictions=predictions, targets=y_true, metrics=metrics, checkpoint_path=ckpt_path)
