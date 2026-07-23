"""Train and evaluate fast causal detectors on raw WESAD physiology."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.load_wesad import load_wesad_dataset
from src.data.schema import DataSchema
from src.models.fast_tcn import FastTCN, count_parameters, model_size_bytes
from src.streaming.fast_detector import causal_feature_names, causal_window_features
from src.training.metrics import expected_calibration_error, select_threshold_from_validation


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _make_run_dir(root: str | Path) -> Path:
    run_dir = Path(root) / f"fast_wesad_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "models").mkdir()
    return run_dir


def _split_by_time(frame: pd.DataFrame, schema: DataSchema, cfg: dict[str, Any]) -> pd.Series:
    split_cfg = cfg.get("split", {})
    mode = str(split_cfg.get("mode", "time"))
    train_ratio = float(split_cfg.get("train_ratio", 0.6))
    val_ratio = float(split_cfg.get("val_ratio", 0.2))
    labels = pd.Series(index=frame.index, dtype=object)
    if mode == "stratified_time":
        for _, group in frame.groupby([schema.worker_id, schema.primary_target], observed=True):
            ordered = group.sort_values(schema.timestamp)
            n = len(ordered)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            labels.loc[ordered.index[:train_end]] = "train"
            labels.loc[ordered.index[train_end:val_end]] = "validation"
            labels.loc[ordered.index[val_end:]] = "test"
        return labels
    for _, group in frame.groupby(schema.worker_id, observed=True):
        ordered = group.sort_values(schema.timestamp)
        n = len(ordered)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        labels.loc[ordered.index[:train_end]] = "train"
        labels.loc[ordered.index[train_end:val_end]] = "validation"
        labels.loc[ordered.index[val_end:]] = "test"
    return labels


def _build_windows(
    frame: pd.DataFrame,
    split_labels: pd.Series,
    schema: DataSchema,
    feature_columns: list[str],
    context_samples: int,
    stride_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    raw_windows: list[np.ndarray] = []
    engineered: list[np.ndarray] = []
    y_rows: list[int] = []
    split_rows: list[str] = []
    ts_rows: list[float] = []
    for _, subject in frame.groupby(schema.worker_id, observed=True):
        subject = subject.sort_values(schema.timestamp)
        values = subject[feature_columns].to_numpy(dtype=float)
        targets = subject[schema.primary_target].to_numpy(dtype=int)
        timestamps = subject[schema.timestamp].to_numpy(dtype=float)
        subject_splits = split_labels.loc[subject.index].to_numpy(dtype=object)
        for end in range(context_samples - 1, len(subject), stride_samples):
            start = end - context_samples + 1
            window_split = subject_splits[start : end + 1]
            if len(set(window_split)) != 1:
                continue
            window = values[start : end + 1]
            raw_windows.append(window.T.astype(np.float32))
            engineered.append(causal_window_features(window, feature_columns))
            y_rows.append(int(targets[end]))
            split_rows.append(str(window_split[-1]))
            ts_rows.append(float(timestamps[end]))
    return (
        np.asarray(raw_windows, dtype=np.float32),
        np.asarray(engineered, dtype=np.float32),
        np.asarray(y_rows, dtype=int),
        np.asarray(ts_rows, dtype=float),
        split_rows,
    )


def _binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float, prediction_interval_seconds: float) -> dict[str, Any]:
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    pred = probs >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    has_both = np.unique(y_true).size == 2
    recall = float(recall_score(y_true, pred, zero_division=0))
    specificity = float(tn / max(1, tn + fp))
    duration_hours = len(y_true) * prediction_interval_seconds / 3600.0
    return {
        "auroc": float(roc_auc_score(y_true, probs)) if has_both else float("nan"),
        "auprc": float(average_precision_score(y_true, probs)) if has_both else float("nan"),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "brier": float(brier_score_loss(y_true, probs)),
        "ece": float(expected_calibration_error(probs, y_true, bins=15)),
        "class_prevalence": float(np.mean(y_true)) if len(y_true) else 0.0,
        "predicted_positive_rate": float(np.mean(pred)) if len(pred) else 0.0,
        "false_alarms_per_hour": float(fp / duration_hours) if duration_hours > 0 else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "n_pos": int(np.sum(y_true == 1)),
        "n_neg": int(np.sum(y_true == 0)),
    }


def _fit_tcn(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
) -> FastTCN:
    torch.manual_seed(seed)
    tcn_cfg = cfg.get("fast_model", {}).get("tcn", {})
    model = FastTCN(
        input_channels=train_x.shape[1],
        hidden_channels=int(tcn_cfg.get("hidden_channels", 12)),
        kernel_size=int(tcn_cfg.get("kernel_size", 3)),
        layers=int(tcn_cfg.get("layers", 2)),
        dropout=float(tcn_cfg.get("dropout", 0.1)),
    )
    batch_size = int(tcn_cfg.get("batch_size", 64))
    epochs = int(tcn_cfg.get("epochs", 2))
    lr = float(tcn_cfg.get("learning_rate", 1e-3))
    pos = max(1, int(np.sum(train_y == 1)))
    neg = max(1, int(np.sum(train_y == 0)))
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
    )
    best_state = None
    best_loss = float("inf")
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(torch.from_numpy(val_x)), torch.from_numpy(val_y.astype(np.float32))).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _latency_summary(estimator: Any, samples: np.ndarray, mode: str) -> dict[str, float]:
    n = min(64, len(samples))
    durations = []
    for i in range(n):
        start = time.perf_counter()
        if mode == "torch":
            with torch.no_grad():
                torch.sigmoid(estimator(torch.from_numpy(samples[i : i + 1]))).numpy()
        else:
            estimator.predict_proba(samples[i : i + 1])
        durations.append((time.perf_counter() - start) * 1000.0)
    if not durations:
        return {"latency_p50_ms": 0.0, "latency_p95_ms": 0.0, "latency_p99_ms": 0.0}
    arr = np.asarray(durations)
    return {
        "latency_p50_ms": float(np.percentile(arr, 50)),
        "latency_p95_ms": float(np.percentile(arr, 95)),
        "latency_p99_ms": float(np.percentile(arr, 99)),
    }


def run(config_path: str | Path) -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    seed = int(cfg.get("reproducibility", {}).get("seed", 42))
    np.random.seed(seed)
    schema = DataSchema.from_config(cfg)
    stream = cfg.get("stream", {})
    target_rate = float(stream.get("target_sampling_rate_hz", 4.0))
    fast_cfg = cfg.get("fast_model", {})
    context_samples = max(1, int(round(float(fast_cfg.get("context_seconds", 3.0)) * target_rate)))
    stride_seconds = float(fast_cfg.get("inference_stride_seconds", 0.25))
    stride_samples = max(1, int(round(stride_seconds * target_rate)))
    dataset_cfg = cfg.get("dataset", {})
    frame = load_wesad_dataset(
        dataset_cfg.get("path", "data/wesad/wesad_subset"),
        schema=schema,
        data_format=dataset_cfg.get("format", "auto"),
        subjects=dataset_cfg.get("subjects"),
        max_rows_per_subject=dataset_cfg.get("max_rows_per_subject"),
        target_sampling_rate_hz=target_rate,
        stress_include_amusement=bool(dataset_cfg.get("stress_include_amusement", True)),
    )
    frame = frame[frame[schema.primary_target].notna()].copy()
    frame[schema.primary_target] = frame[schema.primary_target].astype(int)
    split_labels = _split_by_time(frame, schema, cfg)
    feature_columns = list(schema.physiology)
    raw_x, feat_x, y, timestamps, window_splits = _build_windows(
        frame, split_labels, schema, feature_columns, context_samples, stride_samples
    )
    split_arr = np.asarray(window_splits)
    train_mask = split_arr == "train"
    val_mask = split_arr == "validation"
    test_mask = split_arr == "test"
    if min(train_mask.sum(), val_mask.sum(), test_mask.sum()) == 0:
        raise ValueError("Fast WESAD split produced an empty train, validation, or test window set.")
    if np.unique(y[train_mask]).size < 2 or np.unique(y[val_mask]).size < 2:
        raise ValueError("Fast WESAD train and validation splits must each contain both classes.")

    run_dir = _make_run_dir(cfg.get("paths", {}).get("run_root", "experiments/runs"))
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (run_dir / "feature_names.json").write_text(json.dumps(causal_feature_names(feature_columns), indent=2), encoding="utf-8")
    metrics_rows: list[dict[str, Any]] = []
    models_to_run = fast_cfg.get("architectures", ["dummy", "logistic", "tcn"])

    for name in models_to_run:
        if name == "dummy":
            estimator = DummyClassifier(strategy="prior", random_state=seed)
            estimator.fit(feat_x[train_mask], y[train_mask])
            val_probs = estimator.predict_proba(feat_x[val_mask])[:, 1]
            test_probs = estimator.predict_proba(feat_x[test_mask])[:, 1]
            joblib.dump(estimator, run_dir / "models" / "dummy.joblib")
            model_size = int((run_dir / "models" / "dummy.joblib").stat().st_size)
            params = 0
            latency = _latency_summary(estimator, feat_x[test_mask], "sklearn")
        elif name == "logistic":
            estimator = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("model", LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)),
                ]
            )
            estimator.fit(feat_x[train_mask], y[train_mask])
            val_probs = estimator.predict_proba(feat_x[val_mask])[:, 1]
            test_probs = estimator.predict_proba(feat_x[test_mask])[:, 1]
            joblib.dump(estimator, run_dir / "models" / "logistic.joblib")
            model_size = int((run_dir / "models" / "logistic.joblib").stat().st_size)
            params = int(len(causal_feature_names(feature_columns)) + 1)
            latency = _latency_summary(estimator, feat_x[test_mask], "sklearn")
        elif name == "tcn":
            model = _fit_tcn(raw_x[train_mask], y[train_mask], raw_x[val_mask], y[val_mask], cfg, seed)
            model.eval()
            with torch.no_grad():
                val_probs = torch.sigmoid(model(torch.from_numpy(raw_x[val_mask]))).numpy()
                test_probs = torch.sigmoid(model(torch.from_numpy(raw_x[test_mask]))).numpy()
            torch.save(model.state_dict(), run_dir / "models" / "fast_tcn.pt")
            model_size = model_size_bytes(model)
            params = count_parameters(model)
            latency = _latency_summary(model, raw_x[test_mask], "torch")
        else:
            raise ValueError(f"Unknown fast architecture: {name}")

        threshold_diag = select_threshold_from_validation(
            y[val_mask],
            val_probs,
            policy=str(fast_cfg.get("threshold_policy", "f1")),
            allow_pathological=False,
            min_pred_rate=max(0.02, float(np.mean(y[val_mask])) - 0.3),
            max_pred_rate=min(0.98, float(np.mean(y[val_mask])) + 0.3),
        )
        threshold = float(threshold_diag["threshold"]) if not threshold_diag.get("fallback_used") else 0.5
        metrics = _binary_metrics(y[test_mask], test_probs, threshold, stride_seconds)
        fixed = _binary_metrics(y[test_mask], test_probs, 0.5, stride_seconds)
        metrics.update({f"fixed_0_5_{k}": v for k, v in fixed.items() if k != "confusion_matrix"})
        metrics.update(latency)
        metrics.update(
            {
                "model": name,
                "threshold_diagnostics": threshold_diag,
                "model_size_bytes": int(model_size),
                "parameter_count": int(params),
                "context_seconds": float(fast_cfg.get("context_seconds", 3.0)),
                "forecast_horizon_seconds": float(fast_cfg.get("forecast_horizon_seconds", 0.0)),
                "inference_stride_seconds": stride_seconds,
                "effective_sampling_rate_hz": target_rate,
                "detection_delay_seconds": None,
                "target_semantics": "WESAD protocol stress state at causal window end",
            }
        )
        metrics_rows.append(metrics)
        pd.DataFrame(
            {
                "timestamp": timestamps[test_mask],
                "y_true": y[test_mask],
                "probability": test_probs,
                "prediction": test_probs >= threshold,
                "prediction_fixed_0_5": test_probs >= 0.5,
            }
        ).to_csv(run_dir / f"predictions_{name}.csv", index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(run_dir / "fast_metrics.csv", index=False)
    (run_dir / "fast_metrics.json").write_text(json.dumps(metrics_rows, indent=2, default=_json_default), encoding="utf-8")
    cols = ["model", "auroc", "auprc", "f1", "balanced_accuracy", "brier", "ece", "latency_p95_ms", "model_size_bytes"]
    summary = [
        "# Fast WESAD Detector Run",
        "",
        "This run uses raw WESAD physiology resampled to the configured effective rate.",
        "It predicts the WESAD protocol stress state at the end of a causal observation window; it is not a MultiPhysio second-level safety detector.",
        "",
        metrics_df[cols].to_markdown(index=False),
        "",
    ]
    (run_dir / "fast_results.md").write_text("\n".join(summary), encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/config/fast_wesad.yaml")
    args = parser.parse_args()
    run_dir = run(args.config)
    print(f"Fast detector artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
