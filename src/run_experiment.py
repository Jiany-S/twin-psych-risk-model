"""Run end-to-end experiment: data, preprocessing, XGBoost, TFT, evaluation, plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data.loader import load_or_generate
from .data.preprocess import preprocess_dataframe
from .data.schema import DataSchema
from .data.windowing import build_windows, create_time_splits
from .profiles.worker_profile import WorkerProfileStore
from .training.metrics import classification_metrics, regression_metrics
from .training.plotting import (
    plot_calibration,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr,
    plot_roc,
    plot_timeseries,
)
from .training.tft_train import train_tft_pipeline
from .training.xgb_train import train_xgb_pipeline
from .utils.io import load_yaml, save_json
from .utils.logging import setup_logger
from .utils.paths import create_run_dir
from .utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the risk forecasting experiment.")
    parser.add_argument("--config", type=str, default="src/config/default.yaml")
    return parser.parse_args()


def _prepare_flat_for_tft(frame: pd.DataFrame, profiles: WorkerProfileStore, schema: DataSchema) -> pd.DataFrame:
    static = profiles.get_static_profile_table()
    flat = frame.merge(static, on="worker_id", how="left")
    flat["specialization_id"] = flat["specialization_id"].fillna(0).astype(int)
    flat["experience_level"] = flat["experience_level"].fillna(1).astype(int)
    for col in schema.physiology:
        flat[f"baseline_mu_{col}"] = flat[f"baseline_mu_{col}"].fillna(0.0)
        flat[f"baseline_sigma_{col}"] = flat[f"baseline_sigma_{col}"].fillna(1.0)
    return flat


def _save_processed(processed_dir: Path, windows: np.ndarray, labels: np.ndarray, meta: pd.DataFrame, flat: pd.DataFrame) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    np.save(processed_dir / "windows_X.npy", windows)
    np.save(processed_dir / "windows_y.npy", labels)
    meta.to_csv(processed_dir / "meta.csv", index=False)
    flat.to_csv(processed_dir / "tft_flat.csv", index=False)


def _comparison_summary(metrics: dict[str, Any], task_type: str) -> dict[str, Any]:
    if task_type == "classification":
        xgb_auc = metrics.get("xgboost", {}).get("auroc")
        tft_auc = metrics.get("tft", {}).get("auroc")
        winner = "xgboost" if xgb_auc is not None and tft_auc is not None and xgb_auc >= tft_auc else "tft"
        delta_auc = None if xgb_auc is None or tft_auc is None else float(tft_auc - xgb_auc)
        delta_auprc = None
        if metrics.get("xgboost", {}).get("auprc") is not None and metrics.get("tft", {}).get("auprc") is not None:
            delta_auprc = float(metrics["tft"]["auprc"] - metrics["xgboost"]["auprc"])
        return {
            "winner_by_auroc": winner,
            "delta_auroc": delta_auc,
            "delta_auprc": delta_auprc,
            "notes": "Winner chosen by AUROC on the test split.",
        }
    xgb_rmse = metrics.get("xgboost", {}).get("rmse")
    tft_rmse = metrics.get("tft", {}).get("rmse")
    winner = "xgboost" if xgb_rmse is not None and tft_rmse is not None and xgb_rmse <= tft_rmse else "tft"
    delta_rmse = None if xgb_rmse is None or tft_rmse is None else float(tft_rmse - xgb_rmse)
    return {"winner_by_rmse": winner, "delta_rmse": delta_rmse, "notes": "Winner chosen by RMSE."}


def _write_report(run_dir: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# Experiment Results",
        "",
        "## Summary",
        json.dumps(metrics.get("comparison", {}), indent=2),
        "",
        "## XGBoost Metrics",
        json.dumps(metrics.get("xgboost", {}), indent=2),
        "",
        "## TFT Metrics",
        json.dumps(metrics.get("tft", {}), indent=2),
    ]
    (run_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(config_path: str) -> Path:
    cfg = load_yaml(config_path)
    logger = setup_logger()
    seed_everything(int(cfg["split"]["seed"]))

    schema = DataSchema.from_config(cfg)
    raw_df = load_or_generate(cfg, schema)
    frame, profiles = preprocess_dataframe(cfg, raw_df, schema)

    task_type = cfg["task"]["task_type"]
    risk_threshold = float(cfg["task"]["risk_threshold"])
    if task_type == "classification":
        frame[schema.target] = (frame[schema.target] >= risk_threshold).astype(float)

    flat = _prepare_flat_for_tft(frame, profiles, schema)

    train_df, val_df, test_df = create_time_splits(
        flat,
        schema,
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        test_ratio=cfg["split"]["test_ratio"],
    )

    # Windowing across splits for XGBoost
    window_cfg = cfg["task"]
    window_length = int(window_cfg["window_length"])
    horizon = int(window_cfg["horizon_steps"])
    train_windows = build_windows(train_df, schema, window_length, horizon, task_type, risk_threshold)
    val_windows = build_windows(val_df, schema, window_length, horizon, task_type, risk_threshold)
    test_windows = build_windows(test_df, schema, window_length, horizon, task_type, risk_threshold)

    windows_all = np.concatenate([train_windows.X_windows, val_windows.X_windows, test_windows.X_windows], axis=0)
    labels_all = np.concatenate([train_windows.y, val_windows.y, test_windows.y], axis=0)
    meta_all = pd.concat(
        [
            train_windows.meta.assign(split="train"),
            val_windows.meta.assign(split="val"),
            test_windows.meta.assign(split="test"),
        ],
        ignore_index=True,
    )
    split_indices = {
        "train": np.arange(len(train_windows.y)),
        "val": np.arange(len(train_windows.y), len(train_windows.y) + len(val_windows.y)),
        "test": np.arange(len(train_windows.y) + len(val_windows.y), len(labels_all)),
    }

    # Save processed data
    processed_dir = Path(cfg["paths"]["processed_dir"])
    _save_processed(processed_dir, windows_all, labels_all, meta_all, flat)

    run_paths = create_run_dir(cfg["paths"]["run_root"])
    metrics: dict[str, Any] = {"config": cfg}

    # Train XGBoost
    try:
        feature_cols = list(schema.physiology) + list(schema.robot_context) + [schema.hazard_zone, schema.task_phase]
        xgb_artifacts = train_xgb_pipeline(
            cfg,
            windows_all,
            labels_all,
            meta_all,
            profiles.get_static_profile_table(),
            feature_cols,
            split_indices,
            task_type,
            run_paths.root,
        )
        metrics["xgboost"] = xgb_artifacts.metrics | {"model_path": str(xgb_artifacts.model_path)}
        plot_feature_importance(
            np.array([item["importance"] for item in xgb_artifacts.metrics.get("feature_importance_topk", [])]),
            [item["feature"] for item in xgb_artifacts.metrics.get("feature_importance_topk", [])],
            run_paths.plots / "feature_importance_xgb.png",
            top_k=cfg["report"]["top_k_features"],
        )
    except Exception as exc:
        logger.exception("XGBoost training failed: %s", exc)
        metrics["xgboost"] = {"error": str(exc)}
        xgb_artifacts = None

    # Train TFT
    try:
        tft_artifacts = train_tft_pipeline(
            cfg,
            train_df,
            val_df,
            test_df,
            schema,
            task_type,
            run_paths.root,
        )
        metrics["tft"] = tft_artifacts.metrics | {"checkpoint_path": str(tft_artifacts.checkpoint_path)}
    except Exception as exc:
        logger.exception("TFT training failed: %s", exc)
        metrics["tft"] = {"error": str(exc)}
        tft_artifacts = None

    # Plotting and comparison
    if task_type == "classification":
        test_labels = labels_all[split_indices["test"]]
        roc_series = []
        pr_series = []
        cal_series = []
        if xgb_artifacts is not None:
            roc_series.append(("XGBoost", xgb_artifacts.predictions))
            pr_series.append(("XGBoost", xgb_artifacts.predictions))
            cal_series.append(("XGBoost", xgb_artifacts.predictions))
        if tft_artifacts is not None and len(tft_artifacts.targets) == len(tft_artifacts.predictions):
            if len(tft_artifacts.targets) == len(test_labels):
                roc_series.append(("TFT", tft_artifacts.predictions))
                pr_series.append(("TFT", tft_artifacts.predictions))
                cal_series.append(("TFT", tft_artifacts.predictions))

        if roc_series:
            plot_roc(test_labels, roc_series, run_paths.plots / "roc_curve.png")
            plot_pr(test_labels, pr_series, run_paths.plots / "pr_curve.png")
            plot_calibration(test_labels, cal_series, run_paths.plots / "calibration_curve.png")

        cm_source = metrics.get("xgboost", {}).get("confusion_matrix_default")
        if cm_source is None and metrics.get("tft", {}).get("confusion_matrix_default") is not None:
            cm_source = metrics["tft"]["confusion_matrix_default"]
        if cm_source is not None:
            plot_confusion_matrix(np.array(cm_source), run_paths.plots / "confusion_matrix.png")

        if xgb_artifacts is not None:
            plot_timeseries(
                meta_all.loc[split_indices["test"]],
                test_labels,
                xgb_artifacts.predictions,
                run_paths.plots / "timeseries_predictions.png",
            )
        elif tft_artifacts is not None:
            meta = test_windows.meta.copy()
            n = min(len(meta), len(tft_artifacts.predictions))
            plot_timeseries(
                meta.iloc[:n],
                test_windows.y[:n],
                tft_artifacts.predictions[:n],
                run_paths.plots / "timeseries_predictions.png",
            )
    else:
        if xgb_artifacts is not None:
            plot_timeseries(
                meta_all.loc[split_indices["test"]],
                labels_all[split_indices["test"]],
                xgb_artifacts.predictions,
                run_paths.plots / "timeseries_predictions.png",
            )
        elif tft_artifacts is not None:
            meta = test_windows.meta.copy()
            n = min(len(meta), len(tft_artifacts.predictions))
            plot_timeseries(
                meta.iloc[:n],
                test_windows.y[:n],
                tft_artifacts.predictions[:n],
                run_paths.plots / "timeseries_predictions.png",
            )

    metrics["comparison"] = _comparison_summary(metrics, task_type)
    save_json(metrics, run_paths.root / "metrics.json")
    _write_report(run_paths.root, metrics)

    logger.info("Experiment complete. Artifacts saved to %s", run_paths.root)
    return run_paths.root


def main() -> None:
    args = parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
