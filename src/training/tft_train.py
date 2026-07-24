"""Train and evaluate TFT models for stress/comfort tasks."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
import time
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    import lightning.pytorch as pl
except Exception:
    import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from ..data.load_wesad import load_wesad_dataset
from ..data.multihorizon_windowing import (
    MultiHorizonWindows,
    build_multi_horizon_windows,
    horizon_seconds_to_steps,
    windows_to_long_predictions,
)
from ..data.schema import DataSchema
from ..models.fast_tcn import FastTCN
from ..models.tft_model import SlowTFTForecaster, build_tft_datasets, count_parameters, create_tft_model, model_size_bytes, resolve_tft_loss
from ..streaming.prediction_artifacts import write_replay_manifest_fragment
from .metrics import classification_metrics, expected_calibration_error, regression_metrics, select_threshold_from_validation


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
        if "task_phase" in df.columns:
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

    profile_cols = list(cfg.get("profiles", {}).get("tft_static_real_cols", []))
    if not profile_cols and use_profiles:
        profile_cols = [c for c in train_df.columns if c.startswith("calib_") or c in {"role_metadata", "experience_metadata"}]
    profile_cols = [c for c in profile_cols if c in train_df.columns]
    for df in (train_df, val_df, test_df):
        for col in profile_cols:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

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
            static_reals=profile_cols if use_profiles else [],
            time_varying_known_reals=[],
            static_categoricals=[],
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
        default_root_dir=str(run_dir),
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
        pred_dir = run_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        prob_path = pred_dir / f"{model_name}_primary_probs.npy"
        pred_path = pred_dir / f"{model_name}_primary_predictions.npy"
        np.save(prob_path, predictions)
        np.save(pred_path, (predictions >= chosen_thr).astype(np.float32))
        metrics["probabilities_path"] = str(prob_path)
        metrics["predictions_path"] = str(pred_path)
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


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _slow_run_dir(root: str | Path) -> Path:
    run_dir = Path(root) / f"slow_tft_{time.strftime('%Y%m%d_%H%M%S')}"
    for child in ["models", "predictions", "calibrators"]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def _split_subjects(frame: pd.DataFrame, schema: DataSchema, split_cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    train_subjects = {str(s) for s in split_cfg.get("train_subjects", [])}
    val_subjects = {str(s) for s in split_cfg.get("validation_subjects", split_cfg.get("val_subjects", []))}
    test_subjects = {str(s) for s in split_cfg.get("test_subjects", [])}
    if not train_subjects or not val_subjects or not test_subjects:
        raise ValueError("slow_tft split requires train_subjects, validation_subjects, and test_subjects.")
    if train_subjects & val_subjects or train_subjects & test_subjects or val_subjects & test_subjects:
        raise ValueError("Slow forecaster train/validation/test subject sets must be disjoint.")
    worker = frame[schema.worker_id].astype(str)
    assigned = train_subjects | val_subjects | test_subjects
    missing = sorted(set(worker.unique()) - assigned)
    if missing:
        raise ValueError(f"Loaded subjects not assigned to a slow split: {missing}")
    return {
        "train": frame[worker.isin(train_subjects)].copy(),
        "validation": frame[worker.isin(val_subjects)].copy(),
        "test": frame[worker.isin(test_subjects)].copy(),
    }


def _calibration_cutoffs(frame: pd.DataFrame, schema: DataSchema, seconds: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for worker, group in frame.groupby(schema.worker_id, observed=True):
        ordered = group.sort_values(schema.timestamp)
        baseline = ordered[ordered[schema.protocol_label].astype(str).eq("baseline")]
        if baseline.empty:
            cutoff = float(ordered[schema.timestamp].min())
            n_rows = 0
        else:
            start = float(baseline[schema.timestamp].min())
            used = baseline[pd.to_numeric(baseline[schema.timestamp], errors="coerce") <= start + float(seconds)]
            cutoff = float(used[schema.timestamp].max())
            n_rows = int(len(used))
        rows.append({"worker_id": str(worker), "calibration_cutoff_timestamp": cutoff, "calibration_rows": n_rows})
    return pd.DataFrame(rows)


def _filter_after_calibration(windows: MultiHorizonWindows, cutoffs: pd.DataFrame) -> MultiHorizonWindows:
    if windows.meta.empty:
        return windows
    cutoff_map = dict(zip(cutoffs["worker_id"].astype(str), cutoffs["calibration_cutoff_timestamp"], strict=False))
    keep = windows.meta.apply(lambda row: float(row["prediction_timestamp"]) > float(cutoff_map.get(str(row["worker_id"]), -np.inf)), axis=1)
    keep_arr = keep.to_numpy(dtype=bool)
    return MultiHorizonWindows(
        raw=windows.raw[keep_arr],
        engineered=windows.engineered[keep_arr],
        targets=windows.targets[keep_arr],
        meta=windows.meta.loc[keep_arr].reset_index(drop=True),
        feature_names=windows.feature_names,
        horizon_seconds=windows.horizon_seconds,
        context_seconds=windows.context_seconds,
        inference_stride_seconds=windows.inference_stride_seconds,
        row_interval_seconds=windows.row_interval_seconds,
    )


def _build_slow_windows(
    splits: dict[str, pd.DataFrame],
    schema: DataSchema,
    cfg: dict[str, Any],
) -> tuple[dict[str, MultiHorizonWindows], pd.DataFrame]:
    stream = cfg["stream"]
    slow_cfg = cfg["slow_model"]
    rate_hz = float(stream["target_sampling_rate_hz"])
    row_interval = 1.0 / rate_hz
    context_seconds = float(slow_cfg["context_seconds"])
    stride_seconds = float(slow_cfg["inference_stride_seconds"])
    horizons = [float(h) for h in slow_cfg["forecast_horizons_seconds"]]
    context_steps = int(round(context_seconds * rate_hz))
    stride_steps = int(round(stride_seconds * rate_hz))
    horizon_steps = horizon_seconds_to_steps(horizons, rate_hz)
    calibration_seconds = float(cfg.get("calibration", {}).get("initial_seconds", 60.0))
    all_cutoffs = []
    windows: dict[str, MultiHorizonWindows] = {}
    for split, frame in splits.items():
        cutoffs = _calibration_cutoffs(frame, schema, calibration_seconds)
        cutoffs["split"] = split
        all_cutoffs.append(cutoffs)
        built = build_multi_horizon_windows(
            frame,
            worker_col=schema.worker_id,
            timestamp_col=schema.timestamp,
            protocol_col=schema.protocol_label,
            target_col=schema.primary_target,
            feature_columns=schema.physiology,
            context_steps=context_steps,
            stride_steps=stride_steps,
            horizon_steps=horizon_steps,
            horizon_seconds=horizons,
            row_interval_seconds=row_interval,
            split=split,
            context_seconds=context_seconds,
            inference_stride_seconds=stride_seconds,
        )
        windows[split] = _filter_after_calibration(built, cutoffs)
    for split, built in windows.items():
        if built.raw.size == 0:
            raise ValueError(f"Slow {split} split produced no valid windows.")
        for h_idx, horizon in enumerate(built.horizon_seconds):
            y = built.targets[:, h_idx]
            if split in {"train", "validation"} and np.unique(y).size < 2:
                raise ValueError(f"Slow {split} split has single-class targets for horizon {horizon}s.")
    return windows, pd.concat(all_cutoffs, ignore_index=True)


class _PlattCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=200)

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "_PlattCalibrator":
        if np.unique(y).size < 2:
            raise ValueError("Platt calibration requires both validation classes.")
        self.model.fit(scores.reshape(-1, 1), y.astype(int))
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(scores.reshape(-1, 1))[:, 1]


def _logit(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return np.log(probs / (1.0 - probs))


def _metric_row(y: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, Any]:
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    pred = probs >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    recall = float(recall_score(y, pred, zero_division=0))
    specificity = float(tn / max(1, tn + fp))
    has_both = np.unique(y).size == 2
    return {
        "auroc": float(roc_auc_score(y, probs)) if has_both else float("nan"),
        "auprc": float(average_precision_score(y, probs)) if has_both else float("nan"),
        "prevalence": float(np.mean(y)) if len(y) else 0.0,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "brier": float(brier_score_loss(y, probs)),
        "ece": float(expected_calibration_error(probs, y, bins=15)),
        "predicted_positive_rate": float(np.mean(pred)) if len(pred) else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n": int(len(y)),
        "n_pos": int(np.sum(y == 1)),
        "n_neg": int(np.sum(y == 0)),
    }


def _thresholds_and_calibration(
    val_scores: np.ndarray,
    val_y: np.ndarray,
    test_scores: np.ndarray,
    horizons: list[float],
    run_dir: Path,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[float, float], list[dict[str, Any]]]:
    val_probs_uncal = 1.0 / (1.0 + np.exp(-val_scores))
    test_probs_uncal = 1.0 / (1.0 + np.exp(-test_scores))
    val_probs_cal = np.zeros_like(val_probs_uncal, dtype=float)
    test_probs_cal = np.zeros_like(test_probs_uncal, dtype=float)
    thresholds: dict[float, float] = {}
    diagnostics: list[dict[str, Any]] = []
    for h_idx, horizon in enumerate(horizons):
        calibrator = _PlattCalibrator().fit(val_scores[:, h_idx], val_y[:, h_idx])
        joblib.dump(calibrator, run_dir / "calibrators" / f"{model_name}_h{horizon:g}_platt.joblib")
        val_probs_cal[:, h_idx] = calibrator.predict(val_scores[:, h_idx])
        test_probs_cal[:, h_idx] = calibrator.predict(test_scores[:, h_idx])
        diag = select_threshold_from_validation(
            val_y[:, h_idx],
            val_probs_cal[:, h_idx],
            policy="f1",
            allow_pathological=False,
            min_pred_rate=max(0.02, float(np.mean(val_y[:, h_idx])) - 0.3),
            max_pred_rate=min(0.98, float(np.mean(val_y[:, h_idx])) + 0.3),
        )
        threshold = float(diag["threshold"]) if not diag.get("fallback_used") else 0.5
        thresholds[float(horizon)] = threshold
        diagnostics.append({"horizon_seconds": float(horizon), **diag})
    return val_probs_uncal, test_probs_uncal, test_probs_cal, thresholds, diagnostics


def _train_torch_multi_output(
    model: torch.nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    cfg: dict[str, Any],
) -> torch.nn.Module:
    torch.manual_seed(int(cfg.get("reproducibility", {}).get("seed", 42)))
    slow_cfg = cfg["slow_model"]
    batch_size = int(slow_cfg.get("batch_size", 64))
    epochs = int(slow_cfg.get("max_epochs", 2))
    lr = float(slow_cfg.get("learning_rate", 1e-3))
    criterion = resolve_tft_loss({"loss": slow_cfg.get("loss", "bce")}, task_type="classification")
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_loss = float("inf")
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            target = yb.squeeze(-1) if logits.ndim == 1 and yb.ndim == 2 else yb
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(val_x))
            val_target = torch.from_numpy(val_y.astype(np.float32))
            if val_logits.ndim == 1 and val_target.ndim == 2:
                val_target = val_target.squeeze(-1)
            val_loss = criterion(val_logits, val_target).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model


def _latency_torch(model: torch.nn.Module, sample: np.ndarray) -> dict[str, float]:
    durations = []
    for _ in range(64):
        start = time.perf_counter()
        with torch.no_grad():
            model(torch.from_numpy(sample[:1]))
        durations.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(durations)
    return {"latency_mean_ms": float(arr.mean()), "latency_p50_ms": float(np.percentile(arr, 50)), "latency_p95_ms": float(np.percentile(arr, 95)), "latency_p99_ms": float(np.percentile(arr, 99))}


def _latency_sklearn(model: Any, sample: np.ndarray) -> dict[str, float]:
    durations = []
    for _ in range(64):
        start = time.perf_counter()
        model.predict_proba(sample[:1])
        durations.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(durations)
    return {"latency_mean_ms": float(arr.mean()), "latency_p50_ms": float(np.percentile(arr, 50)), "latency_p95_ms": float(np.percentile(arr, 95)), "latency_p99_ms": float(np.percentile(arr, 99))}


def run_slow_forecaster(config_path: str | Path) -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    schema = DataSchema.from_config(cfg)
    dataset_cfg = cfg["dataset"]
    stream = cfg["stream"]
    run_dir = _slow_run_dir(cfg.get("paths", {}).get("run_root", "experiments/runs"))
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    frame = load_wesad_dataset(
        dataset_cfg["path"],
        schema=schema,
        data_format=dataset_cfg.get("format", "auto"),
        subjects=dataset_cfg.get("subjects"),
        max_rows_per_subject=dataset_cfg.get("max_rows_per_subject"),
        target_sampling_rate_hz=float(stream["target_sampling_rate_hz"]),
        stress_include_amusement=bool(dataset_cfg.get("stress_include_amusement", True)),
    )
    frame = frame[frame[schema.primary_target].notna()].copy()
    frame[schema.primary_target] = frame[schema.primary_target].astype(int)
    splits = _split_subjects(frame, schema, cfg["split"])
    windows, cutoffs = _build_slow_windows(splits, schema, cfg)
    cutoffs.to_csv(run_dir / "calibration_cutoffs.csv", index=False)
    (run_dir / "feature_names.json").write_text(json.dumps(windows["train"].feature_names, indent=2), encoding="utf-8")
    horizons = windows["train"].horizon_seconds

    train_w, val_w, test_w = windows["train"], windows["validation"], windows["test"]
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    def record_model(model_name: str, val_scores: np.ndarray, test_scores: np.ndarray, latency: dict[str, float], model_size: int, params: int) -> None:
        _, test_uncal, test_cal, thresholds, threshold_diags = _thresholds_and_calibration(
            val_scores, val_w.targets, test_scores, horizons, run_dir, model_name
        )
        pred_long = windows_to_long_predictions(
            test_w,
            raw_logits=test_scores,
            uncalibrated_probability=test_uncal,
            calibrated_probability=test_cal,
            threshold_by_horizon=thresholds,
            model_name=model_name,
        )
        pred_long.to_csv(run_dir / "predictions" / f"{model_name}_predictions_long.csv", index=False)
        prediction_frames.append(pred_long)
        for h_idx, horizon in enumerate(horizons):
            selected = _metric_row(test_w.targets[:, h_idx], test_cal[:, h_idx], thresholds[float(horizon)])
            fixed = _metric_row(test_w.targets[:, h_idx], test_cal[:, h_idx], 0.5)
            valid_subjects = int(pred_long[pred_long["horizon_seconds"] == float(horizon)]["worker_id"].nunique())
            metric_rows.append(
                {
                    "model": model_name,
                    "horizon_seconds": float(horizon),
                    **selected,
                    **{f"fixed_0_5_{k}": v for k, v in fixed.items() if k != "confusion_matrix"},
                    "validation_selected_threshold": thresholds[float(horizon)],
                    "threshold_diagnostics": threshold_diags[h_idx],
                    "valid_subject_count": valid_subjects,
                    "model_size_bytes": int(model_size),
                    "parameter_count": int(params),
                    "meets_stride_deadline": bool(latency["latency_p95_ms"] <= float(cfg["slow_model"]["inference_stride_seconds"]) * 1000.0),
                    **latency,
                }
            )

    for model_name in cfg["slow_model"].get("comparators", ["dummy", "logistic", "random_forest", "xgboost", "tcn", "tft"]):
        if model_name == "dummy":
            val_scores = np.zeros_like(val_w.targets, dtype=float)
            test_scores = np.zeros_like(test_w.targets, dtype=float)
            for h_idx in range(len(horizons)):
                clf = DummyClassifier(strategy="prior")
                clf.fit(train_w.engineered, train_w.targets[:, h_idx])
                val_scores[:, h_idx] = _logit(clf.predict_proba(val_w.engineered)[:, list(clf.classes_).index(1)])
                test_scores[:, h_idx] = _logit(clf.predict_proba(test_w.engineered)[:, list(clf.classes_).index(1)])
            record_model(model_name, val_scores, test_scores, {"latency_mean_ms": 0.0, "latency_p50_ms": 0.0, "latency_p95_ms": 0.0, "latency_p99_ms": 0.0}, 0, 0)
        elif model_name in {"logistic", "random_forest", "xgboost"}:
            val_scores = np.zeros_like(val_w.targets, dtype=float)
            test_scores = np.zeros_like(test_w.targets, dtype=float)
            first_model = None
            for h_idx, horizon in enumerate(horizons):
                if model_name == "logistic":
                    clf = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=500, class_weight="balanced"))])
                elif model_name == "random_forest":
                    clf = Pipeline([("impute", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=80, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1))])
                else:
                    if XGBClassifier is None:
                        continue
                    clf = Pipeline([("impute", SimpleImputer(strategy="median")), ("model", XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, eval_metric="logloss", random_state=42))])
                clf.fit(train_w.engineered, train_w.targets[:, h_idx])
                first_model = clf if first_model is None else first_model
                joblib.dump(clf, run_dir / "models" / f"{model_name}_h{horizon:g}.joblib")
                classes = list(clf.classes_)
                val_scores[:, h_idx] = _logit(clf.predict_proba(val_w.engineered)[:, classes.index(1)])
                test_scores[:, h_idx] = _logit(clf.predict_proba(test_w.engineered)[:, classes.index(1)])
            latency = _latency_sklearn(first_model, test_w.engineered) if first_model is not None else {"latency_mean_ms": 0.0, "latency_p50_ms": 0.0, "latency_p95_ms": 0.0, "latency_p99_ms": 0.0}
            size = sum((run_dir / "models" / f"{model_name}_h{h:g}.joblib").stat().st_size for h in horizons if (run_dir / "models" / f"{model_name}_h{h:g}.joblib").exists())
            record_model(model_name, val_scores, test_scores, latency, int(size), int(train_w.engineered.shape[1] + 1))
        elif model_name == "tcn":
            val_scores = np.zeros_like(val_w.targets, dtype=float)
            test_scores = np.zeros_like(test_w.targets, dtype=float)
            total_params = 0
            total_size = 0
            first_model = None
            for h_idx, horizon in enumerate(horizons):
                model = FastTCN(input_channels=train_w.raw.shape[1], hidden_channels=int(cfg["slow_model"].get("hidden_size", 16)), layers=2)
                model = _train_torch_multi_output(model, train_w.raw, train_w.targets[:, h_idx : h_idx + 1], val_w.raw, val_w.targets[:, h_idx : h_idx + 1], cfg)
                first_model = model if first_model is None else first_model
                with torch.no_grad():
                    val_scores[:, h_idx] = model(torch.from_numpy(val_w.raw)).numpy()
                    test_scores[:, h_idx] = model(torch.from_numpy(test_w.raw)).numpy()
                torch.save(model.state_dict(), run_dir / "models" / f"tcn_h{horizon:g}.pt")
                total_params += sum(p.numel() for p in model.parameters())
                total_size += sum(p.numel() * p.element_size() for p in model.parameters())
            record_model(model_name, val_scores, test_scores, _latency_torch(first_model, test_w.raw), int(total_size), int(total_params))
        elif model_name == "tft":
            model = SlowTFTForecaster(
                input_channels=train_w.raw.shape[1],
                n_horizons=len(horizons),
                hidden_size=int(cfg["slow_model"].get("hidden_size", 24)),
                dropout=float(cfg["slow_model"].get("dropout", 0.1)),
            )
            model = _train_torch_multi_output(model, train_w.raw, train_w.targets, val_w.raw, val_w.targets, cfg)
            with torch.no_grad():
                val_scores = model(torch.from_numpy(val_w.raw)).numpy()
                test_scores = model(torch.from_numpy(test_w.raw)).numpy()
            torch.save(model.state_dict(), run_dir / "models" / "slow_tft.pt")
            record_model(model_name, val_scores, test_scores, _latency_torch(model, test_w.raw), model_size_bytes(model), count_parameters(model))
        else:
            raise ValueError(f"Unsupported slow model comparator: {model_name}")

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(run_dir / "predictions_long.csv", index=False)
    write_replay_manifest_fragment("slow", run_dir, cfg, "predictions_long.csv", "xgboost")
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(run_dir / "slow_metrics.csv", index=False)
    (run_dir / "slow_metrics.json").write_text(json.dumps(metric_rows, indent=2, default=_json_default), encoding="utf-8")
    per_subject = (
        predictions.groupby(["model", "horizon_seconds", "worker_id"], observed=True)
        .agg(n=("target", "size"), prevalence=("target", "mean"), predicted_positive_rate=("prediction", "mean"))
        .reset_index()
    )
    per_subject.to_csv(run_dir / "per_subject_slow_metrics.csv", index=False)
    summary = [
        "# Slow Multi-Horizon Forecaster",
        "",
        "Dataset: WESAD raw physiology. Target: WESAD protocol stress state, not construction risk.",
        f"Context seconds: `{cfg['slow_model']['context_seconds']}`. Inference stride seconds: `{cfg['slow_model']['inference_stride_seconds']}`. Horizons seconds: `{cfg['slow_model']['forecast_horizons_seconds']}`.",
        "Loss: BCE with logits for neural classification models; sigmoid is applied once to logits before calibration.",
        "",
        metrics[["model", "horizon_seconds", "auroc", "auprc", "f1", "fixed_0_5_f1", "balanced_accuracy", "brier", "ece", "latency_p95_ms"]].to_markdown(index=False),
        "",
        "Attention-style outputs are not produced by this smoke runner; attention, if later extracted from a full TFT implementation, must not be interpreted as causal explanation.",
    ]
    (run_dir / "slow_results.md").write_text("\n".join(summary), encoding="utf-8")
    Path("docs/slow_forecaster.md").write_text("\n".join(summary), encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run slow WESAD multi-horizon forecaster smoke.")
    parser.add_argument("--config", default="src/config/slow_tft.yaml")
    args = parser.parse_args()
    run_dir = run_slow_forecaster(args.config)
    print(f"Slow forecaster artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
