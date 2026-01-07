"""Train the Temporal Fusion Transformer model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from ..data.schema import DataSchema
from ..data.splits import temporal_split
from ..models.tft_model import build_tft_dataset, create_tft_model
from ..utils.logging import configure_logging
from ..utils.seed import seed_everything


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def train_tft(config_path: str) -> Path:
    cfg = load_config(config_path)
    seed_everything(cfg["training"]["seed"])
    logger = configure_logging("train_tft")
    processed_path = Path(cfg["paths"]["processed_train_file"])
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {processed_path}. Run preprocessing first.")

    df = pd.read_csv(processed_path)
    schema = DataSchema.from_config(cfg)

    for column in schema.static_categorical:
        if column in df.columns:
            df[column] = df[column].astype(str)
    df[schema.worker_id] = df[schema.worker_id].astype(str)

    split = temporal_split(df, time_col=schema.time_idx, val_ratio=cfg["data"]["val_ratio"], test_ratio=cfg["data"]["test_ratio"])

    training_ds, validation_ds = build_tft_dataset(
        split.train,
        split.val,
        schema=schema,
        max_encoder_length=cfg["data"]["window_length"],
        max_prediction_length=cfg["data"]["prediction_horizon"],
    )
    train_loader = training_ds.to_dataloader(
        train=True,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
    )
    val_loader = validation_ds.to_dataloader(
        train=False,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
    )

    model = create_tft_model(
        training_ds,
        learning_rate=cfg["training"]["learning_rate"],
        hidden_size=cfg["training"]["hidden_size"],
        lstm_layers=cfg["training"]["lstm_layers"],
        dropout=cfg["training"]["dropout"],
    )

    artifacts_dir = Path(cfg["paths"]["artifacts_dir"]) / "tft"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        ModelCheckpoint(dirpath=artifacts_dir, filename="tft-{epoch:02d}-{val_loss:.3f}", monitor="val_loss", save_top_k=1),
    ]
    csv_logger = CSVLogger(save_dir=artifacts_dir)

    accelerator = "gpu" if cfg["training"]["gpus"] > 0 and torch.cuda.is_available() else "cpu"
    devices = min(cfg["training"]["gpus"], torch.cuda.device_count()) if accelerator == "gpu" else 1
    devices = max(devices, 1)

    trainer = pl.Trainer(
        max_epochs=cfg["training"]["max_epochs"],
        gradient_clip_val=cfg["training"]["gradient_clip_val"],
        callbacks=callbacks,
        logger=csv_logger,
        accelerator=accelerator,
        devices=devices,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    checkpoint_path = Path(callbacks[1].best_model_path) if callbacks[1].best_model_path else artifacts_dir / "tft_last.ckpt"
    if not checkpoint_path.exists():
        trainer.save_checkpoint(checkpoint_path)
    logger.info("Saved TFT checkpoint to %s", checkpoint_path)
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Temporal Fusion Transformer on processed data.")
    parser.add_argument("--config", type=str, default="src/config/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_tft(args.config)


if __name__ == "__main__":
    main()
