"""Train and evaluate TFT models for stress/comfort tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import torch

try:
    import lightning.pytorch as pl
except Exception:
    import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from ..models.tft_model import build_tft_datasets, create_tft_model
from .metrics import classification_metrics, regression_metrics, select_threshold_from_validation


@dataclass
class TFTTaskArtifacts:
    predictions: np.ndarray
    targets: np.ndarray
    metrics: dict[str, Any]
    checkpoint_path: Path


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


def _predict_with_targets(model: Any, loader: Any, model_name: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        pred_out = model.predict(loader, return_y=True)
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
        for batch in loader:
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
    return predictions, y_true


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
    window_step: int,
    model_name: str,
    use_profiles: bool,
) -> TFTTaskArtifacts:
    tft_cfg = cfg["tft"]
    debug = bool(cfg.get("debug", False))
    if tft_cfg.get("gpus", 0) > 0 and not torch.cuda.is_available():
        tft_cfg = dict(tft_cfg)
        tft_cfg["gpus"] = 0
    for df in (train_df, val_df, test_df):
        df["worker_id"] = df["worker_id"].astype(str)
        df["specialization_index"] = df["specialization_index"].astype(str)
        df["experience_level"] = df["experience_level"].astype(str)
        df["task_phase"] = df["task_phase"].astype(str)
        # Reindex time_idx per worker to ensure contiguous steps for TFT.
        df[schema.time_idx] = df.groupby("worker_id", observed=True).cumcount().astype(int)
    if window_step > 1:
        def _stride_rows(frame: pd.DataFrame, step: int) -> pd.DataFrame:
            pos = frame.groupby("worker_id", observed=True).cumcount()
            keep = (pos % step) == 0
            out = frame.loc[keep].copy()
            out[schema.time_idx] = out.groupby("worker_id", observed=True).cumcount().astype(int)
            return out
        train_df = _stride_rows(train_df, window_step)
        val_df = _stride_rows(val_df, window_step)
        test_df = _stride_rows(test_df, window_step)

    # Ensure validation/test categorical values are known to encoders to avoid unknown-class warnings.
    categorical_cols = ["worker_id", "specialization_index", "experience_level", "task_phase"]
    unseen_workers = sorted(set(val_df["worker_id"]).union(set(test_df["worker_id"])) - set(train_df["worker_id"]))
    if unseen_workers:
        base = train_df.iloc[[0]].copy()
        extra_rows: list[pd.DataFrame] = []
        for worker in unseen_workers:
            row = base.copy()
            row["worker_id"] = str(worker)
            row[schema.time_idx] = 0
            row[target_col] = 0.0
            for col in categorical_cols:
                if col not in row.columns:
                    row[col] = "UNK"
            for col in schema.physiology:
                if col in row.columns:
                    row[col] = 0.0
            extra_rows.append(row)
        train_df = pd.concat([train_df, *extra_rows], ignore_index=True)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Min encoder length and/or min_prediction_idx.*not present in the dataset index.*",
            module="pytorch_forecasting.data.timeseries._timeseries",
        )
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
        # Use predict=False so we evaluate on all available windows, not only the last per series.
        test_ds = train_ds.from_dataset(train_ds, test_df, predict=False, stop_randomization=True)
    train_loader = train_ds.to_dataloader(train=True, batch_size=tft_cfg["batch_size"], num_workers=tft_cfg["num_workers"])
    val_loader = val_ds.to_dataloader(train=False, batch_size=tft_cfg["batch_size"], num_workers=tft_cfg["num_workers"])
    test_loader = test_ds.to_dataloader(train=False, batch_size=tft_cfg["batch_size"], num_workers=tft_cfg["num_workers"])
    if debug:
        print(
            f"[TFT DEBUG] {model_name} datasets: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)} "
            f"rows: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

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
    if hasattr(pl, "seed_everything"):
        pl.seed_everything(int(cfg.get("reproducibility", {}).get("seed", 42)), workers=True)
    trainer = pl.Trainer(
        max_epochs=tft_cfg["max_epochs"],
        callbacks=callbacks,
        accelerator=accelerator,
        devices=max(1, devices),
        logger=False,
        enable_checkpointing=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Found .* unknown classes which were set to NaN",
            module="pytorch_forecasting.data.encoders",
        )
        trainer.fit(model, train_loader, val_loader)

    ckpt_path = Path(callbacks[1].best_model_path) if callbacks[1].best_model_path else run_dir / "models" / f"{model_name}.ckpt"
    if not ckpt_path.exists():
        trainer.save_checkpoint(ckpt_path)

    val_pred, y_val = _predict_with_targets(model, val_loader, model_name=f"{model_name}_val")
    predictions, y_true = _predict_with_targets(model, test_loader, model_name=model_name)
    if len(predictions) != len(y_true):
        raise ValueError(f"TFT prediction/target length mismatch for {model_name}: {len(predictions)} vs {len(y_true)}")
    if debug:
        print(
            f"[TFT DEBUG] {model_name} preds: n={len(predictions)} min={predictions.min():.4f} "
            f"max={predictions.max():.4f} y_true_counts={np.unique(y_true, return_counts=True)}"
        )

    if task_type == "classification":
        if cfg["tft"].get("tft_loss", "quantile") == "bce":
            predictions = 1.0 / (1.0 + np.exp(-predictions))
            val_pred = 1.0 / (1.0 + np.exp(-val_pred))
        predictions = np.clip(predictions, 0.0, 1.0)
        val_pred = np.clip(val_pred, 0.0, 1.0)
        threshold_cfg = cfg.get("thresholding", {})
        threshold_diag = select_threshold_from_validation(
            y_val,
            val_pred,
            policy=str(threshold_cfg.get("policy", cfg.get("xgboost", {}).get("threshold_policy", "f1"))).lower(),
            target_recall=float(threshold_cfg.get("target_recall", cfg.get("xgboost", {}).get("target_recall", 0.7))),
            target_precision=float(
                threshold_cfg.get("target_precision", cfg.get("xgboost", {}).get("target_precision", 0.7))
            ),
            min_pred_rate=float(threshold_cfg.get("min_pred_rate", 0.02)),
            max_pred_rate=float(threshold_cfg.get("max_pred_rate", 0.98)),
            allow_pathological=bool(threshold_cfg.get("allow_pathological", False)),
        )
        chosen_thr = float(threshold_diag["threshold"])
        metrics = classification_metrics(y_true, predictions, chosen_threshold=chosen_thr)
        metrics["threshold_policy"] = str(threshold_diag.get("policy", "f1"))
        metrics["threshold_diagnostics"] = threshold_diag
        metrics["val_selected_threshold"] = chosen_thr
        metrics["val_positive_rate_at_threshold"] = float(np.mean(val_pred >= chosen_thr))
        metrics["val_prob_stats"] = {
            "min": float(np.min(val_pred)),
            "max": float(np.max(val_pred)),
            "mean": float(np.mean(val_pred)),
            "std": float(np.std(val_pred)),
        }
        metrics["test_positive_count_default"] = int(np.sum(predictions >= 0.5))
        metrics["test_positive_count_optimal"] = int(np.sum(predictions >= chosen_thr))
        metrics["pred_min"] = float(np.min(predictions))
        metrics["pred_max"] = float(np.max(predictions))
    else:
        metrics = regression_metrics(y_true, predictions)
    metrics["n_predictions"] = int(len(predictions))
    metrics["n_targets"] = int(len(y_true))
    metrics["window_step_used"] = int(window_step)
    metrics["tft_train_dataset_len"] = int(len(train_ds))
    metrics["tft_val_dataset_len"] = int(len(val_ds))
    metrics["tft_test_dataset_len"] = int(len(test_ds))
    metrics["tft_train_rows"] = int(len(train_df))
    metrics["tft_val_rows"] = int(len(val_df))
    metrics["tft_test_rows"] = int(len(test_df))
    return TFTTaskArtifacts(predictions=predictions, targets=y_true, metrics=metrics, checkpoint_path=ckpt_path)
