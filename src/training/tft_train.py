"""Train and evaluate TFT model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
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
) -> TFTArtifacts:
    tft_cfg = cfg["tft"]
    train_ds, val_ds = build_tft_datasets(
        train_df,
        val_df,
        schema=schema,
        window_length=cfg["task"]["window_length"],
        horizon=cfg["task"]["horizon_steps"],
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

    # Predict
    preds = trainer.predict(model, dataloaders=test_loader)
    pred_tensor = torch.cat([p.detach().cpu() for p in preds])
    if pred_tensor.ndim > 2:
        pred_tensor = pred_tensor[..., -1]
    elif pred_tensor.ndim == 2:
        pred_tensor = pred_tensor[:, -1]
    predictions = pred_tensor.numpy()

    # Extract targets aligned with test loader
    targets = []
    for _, y in test_loader:
        targets.append(y[0][:, -1].detach().cpu())
    y_true = torch.cat(targets).numpy()

    if task_type == "classification":
        metrics = classification_metrics(y_true, predictions)
    else:
        metrics = regression_metrics(y_true, predictions)

    return TFTArtifacts(predictions=predictions, targets=y_true, metrics=metrics, checkpoint_path=ckpt_path)
