"""Train and evaluate TFT models for stress/comfort tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    import lightning.pytorch as pl
except Exception:
    import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from ..models.tft_model import build_tft_datasets, create_tft_model
from .metrics import classification_metrics, regression_metrics


@dataclass
class TFTTaskArtifacts:
    predictions: np.ndarray
    targets: np.ndarray
    metrics: dict[str, Any]
    checkpoint_path: Path
    variable_importance: dict[str, float] | None = None


def _to_tensor(pred_obj: Any) -> torch.Tensor:
    if isinstance(pred_obj, torch.Tensor):
        return pred_obj
    if hasattr(pred_obj, "prediction"):
        return _to_tensor(pred_obj.prediction)
    if hasattr(pred_obj, "output"):
        return _to_tensor(pred_obj.output)
    if isinstance(pred_obj, (list, tuple)) and len(pred_obj) > 0:
        return _to_tensor(pred_obj[0])
    raise TypeError(f"Unsupported prediction structure: {type(pred_obj)}")


def _extract_target(y_obj: Any) -> torch.Tensor:
    if isinstance(y_obj, torch.Tensor):
        return y_obj
    if isinstance(y_obj, dict):
        if "target" in y_obj:
            return _extract_target(y_obj["target"])
        raise ValueError(f"Unknown target dict keys: {list(y_obj.keys())}")
    if isinstance(y_obj, (list, tuple)) and len(y_obj) > 0:
        return _extract_target(y_obj[0])
    if hasattr(y_obj, "target"):
        return _extract_target(y_obj.target)
    raise TypeError(f"Unsupported target structure: {type(y_obj)}")


def _extract_tft_importance(model: Any, dataset: Any) -> dict[str, float] | None:
    """Extract variable importance from TFT model's variable selection network.

    Attempts to extract encoder and decoder variable importance weights
    from the TFT model's attention/selection mechanisms.
    """
    try:
        importance = {}

        # Try to get encoder variable selection weights
        if hasattr(model, "encoder_variable_selection"):
            encoder_vars = getattr(dataset, "time_varying_unknown_reals", [])
            encoder_vars += getattr(dataset, "time_varying_known_reals", [])
            if hasattr(model.encoder_variable_selection, "weight_network"):
                weights = model.encoder_variable_selection.weight_network
                if hasattr(weights, "weight"):
                    w = weights.weight.detach().cpu().numpy()
                    if w.ndim >= 2:
                        var_weights = np.abs(w).mean(axis=0)
                        for i, var_name in enumerate(encoder_vars[:len(var_weights)]):
                            importance[f"encoder_{var_name}"] = float(var_weights[i])

        # Try to get static variable importance
        if hasattr(model, "static_variable_selection"):
            static_vars = getattr(dataset, "static_reals", []) + getattr(dataset, "static_categoricals", [])
            if hasattr(model.static_variable_selection, "weight_network"):
                weights = model.static_variable_selection.weight_network
                if hasattr(weights, "weight"):
                    w = weights.weight.detach().cpu().numpy()
                    if w.ndim >= 2:
                        var_weights = np.abs(w).mean(axis=0)
                        for i, var_name in enumerate(static_vars[:len(var_weights)]):
                            importance[f"static_{var_name}"] = float(var_weights[i])

        return importance if importance else None
    except Exception:
        # Silently fail if importance extraction not supported
        return None


def train_tft_task(
    cfg: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema,
    target_col: str,
    task_type: str,
    run_dir: Path,
    window_length: int,
    horizon: int,
    model_name: str,
    use_profiles: bool,
) -> TFTTaskArtifacts:
    tft_cfg = cfg["tft"]
    for df in (train_df, val_df, test_df):
        df["worker_id"] = df["worker_id"].astype(str)
        df["specialization_index"] = df["specialization_index"].astype(str)
        df["experience_level"] = df["experience_level"].astype(str)
        df["task_phase"] = df["task_phase"].astype(str)

    train_ds, val_ds = build_tft_datasets(
        train_df=train_df,
        val_df=val_df,
        schema=schema,
        target_col=target_col,
        window_length=window_length,
        horizon=horizon,
        use_profiles=use_profiles,
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
            filename=f"{model_name}" + "-{epoch:02d}-{val_loss:.3f}",
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
        devices=max(1, devices),
        logger=False,
        enable_checkpointing=True,
    )
    trainer.fit(model, train_loader, val_loader)

    ckpt_path = Path(callbacks[1].best_model_path) if callbacks[1].best_model_path else run_dir / "models" / f"{model_name}.ckpt"
    if not ckpt_path.exists():
        trainer.save_checkpoint(ckpt_path)

    try:
        pred_out = model.predict(test_loader, return_y=True)
    except Exception as exc:
        raise RuntimeError(f"TFT prediction failed for {model_name}.") from exc

    if isinstance(pred_out, tuple) and len(pred_out) == 2:
        pred_obj, y_obj = pred_out
    else:
        pred_obj, y_obj = pred_out, None

    pred_tensor = _to_tensor(pred_obj).detach().cpu()
    if pred_tensor.ndim > 2:
        pred_tensor = pred_tensor[..., -1]
    elif pred_tensor.ndim == 2:
        pred_tensor = pred_tensor[:, -1]
    predictions = pred_tensor.numpy()

    if y_obj is None:
        targets = []
        for batch in test_loader:
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                raise ValueError(f"Unexpected dataloader batch structure for {model_name}: {type(batch)}")
            y_tensor = _extract_target(batch[1]).detach().cpu()
            if y_tensor.ndim > 1:
                y_tensor = y_tensor[:, -1]
            targets.append(y_tensor)
        y_true = torch.cat(targets).numpy()
    else:
        y_true_tensor = _extract_target(y_obj).detach().cpu()
        if y_true_tensor.ndim > 1:
            y_true_tensor = y_true_tensor[:, -1]
        y_true = y_true_tensor.numpy()

    if task_type == "classification":
        if cfg["tft"].get("tft_loss", "quantile") == "bce":
            predictions = 1.0 / (1.0 + np.exp(-predictions))
        predictions = np.clip(predictions, 0.0, 1.0)
        metrics = classification_metrics(y_true, predictions)
    else:
        metrics = regression_metrics(y_true, predictions)

    # Extract variable importance from TFT model
    variable_importance = _extract_tft_importance(model, train_ds)

    return TFTTaskArtifacts(
        predictions=predictions,
        targets=y_true,
        metrics=metrics,
        checkpoint_path=ckpt_path,
        variable_importance=variable_importance,
    )
