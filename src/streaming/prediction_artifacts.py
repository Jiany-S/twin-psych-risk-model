"""Prediction artifact loading and validation for replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.utils.artifacts import config_hash, file_sha256, git_commit, latest_run


REQUIRED_COLUMNS = {
    "worker_id",
    "session_id",
    "prediction_timestamp",
    "target_timestamp",
    "horizon_seconds",
    "model_name",
    "probability",
    "calibrated_probability",
    "target",
    "split",
    "model_version",
    "config_hash",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def _subjects_from_config(cfg: dict[str, Any], split_name: str) -> list[str]:
    split = cfg.get("split", {})
    if split_name == "train":
        return [str(s) for s in split.get("train_subjects", [])]
    if split_name == "validation":
        return [str(s) for s in split.get("validation_subjects", split.get("val_subjects", []))]
    return [str(s) for s in split.get("test_subjects", [])]


def standardize_fast_predictions(run_dir: str | Path, model_name: str) -> pd.DataFrame:
    run = Path(run_dir)
    path = run / f"predictions_{model_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Fast prediction artifact not found: {path}")
    df = pd.read_csv(path)
    required = {"worker_id", "prediction_timestamp", "target_timestamp", "predicted_probability", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fast prediction artifact {path} missing columns: {sorted(missing)}")
    cfg = _load_yaml(run / "config_resolved.yaml")
    out = pd.DataFrame(
        {
            "worker_id": df["worker_id"].astype(str),
            "session_id": df.get("session_id", df["worker_id"]).astype(str),
            "prediction_timestamp": pd.to_numeric(df["prediction_timestamp"], errors="raise"),
            "target_timestamp": pd.to_numeric(df["target_timestamp"], errors="raise"),
            "horizon_seconds": 0.0,
            "model_name": model_name,
            "probability": pd.to_numeric(df["predicted_probability"], errors="raise"),
            "calibrated_probability": pd.to_numeric(df["predicted_probability"], errors="raise"),
            "target": pd.to_numeric(df["target"], errors="raise").astype(int),
            "split": df.get("split", "test"),
            "model_version": f"fast_{model_name}",
            "config_hash": config_hash(cfg),
            "source_artifact_path": str(path),
            "source_artifact_hash": file_sha256(path),
            "training_git_commit": git_commit(),
            "training_subjects": ",".join(_subjects_from_config(cfg, "train")),
            "validation_subjects": ",".join(_subjects_from_config(cfg, "validation")),
            "test_subjects": ",".join(_subjects_from_config(cfg, "test")),
            "target_definition": "WESAD protocol stress state; class 1=stress, class 0=baseline/amusement",
            "context_seconds": cfg.get("fast_model", {}).get("context_seconds"),
            "forecast_horizon_seconds": cfg.get("fast_model", {}).get("forecast_horizon_seconds", 0.0),
            "inference_stride_seconds": cfg.get("fast_model", {}).get("inference_stride_seconds"),
        }
    )
    return validate_prediction_contract(out, expected_test_subjects=_subjects_from_config(cfg, "test"))


def standardize_slow_predictions(run_dir: str | Path, model_name: str) -> pd.DataFrame:
    run = Path(run_dir)
    path = run / "predictions_long.csv"
    if not path.exists():
        raise FileNotFoundError(f"Slow prediction artifact not found: {path}")
    df = pd.read_csv(path)
    df = df[df["model"].astype(str) == str(model_name)].copy()
    if df.empty:
        raise ValueError(f"No slow predictions for model {model_name} in {path}")
    cfg = _load_yaml(run / "config_resolved.yaml")
    out = pd.DataFrame(
        {
            "worker_id": df["worker_id"].astype(str),
            "session_id": df.get("session_id", df["worker_id"]).astype(str),
            "prediction_timestamp": pd.to_numeric(df["prediction_timestamp"], errors="raise"),
            "target_timestamp": pd.to_numeric(df["target_timestamp"], errors="raise"),
            "horizon_seconds": pd.to_numeric(df["horizon_seconds"], errors="raise"),
            "model_name": model_name,
            "probability": pd.to_numeric(df["uncalibrated_probability"], errors="raise"),
            "calibrated_probability": pd.to_numeric(df["calibrated_probability"], errors="raise"),
            "target": pd.to_numeric(df["target"], errors="raise").astype(int),
            "split": "test",
            "model_version": f"slow_{model_name}",
            "config_hash": config_hash(cfg),
            "source_artifact_path": str(path),
            "source_artifact_hash": file_sha256(path),
            "training_git_commit": git_commit(),
            "training_subjects": ",".join(_subjects_from_config(cfg, "train")),
            "validation_subjects": ",".join(_subjects_from_config(cfg, "validation")),
            "test_subjects": ",".join(_subjects_from_config(cfg, "test")),
            "target_definition": "WESAD protocol stress forecast; class 1=stress",
            "context_seconds": cfg.get("slow_model", {}).get("context_seconds"),
            "forecast_horizon_seconds": df["horizon_seconds"],
            "inference_stride_seconds": cfg.get("slow_model", {}).get("inference_stride_seconds"),
        }
    )
    return validate_prediction_contract(out, expected_test_subjects=_subjects_from_config(cfg, "test"))


def validate_prediction_contract(df: pd.DataFrame, expected_test_subjects: list[str] | None = None) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Prediction artifact missing required columns: {sorted(missing)}")
    out = df.copy()
    out["worker_id"] = out["worker_id"].astype(str)
    out["session_id"] = out["session_id"].astype(str)
    out["prediction_timestamp"] = pd.to_numeric(out["prediction_timestamp"], errors="raise")
    out["target_timestamp"] = pd.to_numeric(out["target_timestamp"], errors="raise")
    out["horizon_seconds"] = pd.to_numeric(out["horizon_seconds"], errors="raise")
    out["calibrated_probability"] = pd.to_numeric(out["calibrated_probability"], errors="raise")
    if not out["calibrated_probability"].between(0.0, 1.0).all():
        raise ValueError("Prediction artifact contains probabilities outside [0, 1].")
    if not np.allclose(out["target_timestamp"] - out["prediction_timestamp"], out["horizon_seconds"], atol=1e-5):
        raise ValueError("Prediction target timestamps do not align with horizon_seconds.")
    dup_cols = ["worker_id", "session_id", "prediction_timestamp", "target_timestamp", "horizon_seconds", "model_name"]
    if out.duplicated(dup_cols).any():
        raise ValueError("Prediction artifact contains duplicate prediction keys.")
    if expected_test_subjects:
        expected = {str(s) for s in expected_test_subjects}
        observed = set(out["worker_id"].astype(str).unique())
        if not observed.issubset(expected):
            raise ValueError(f"Prediction artifact includes non-test subjects: {sorted(observed - expected)}")
    for _, group in out.groupby(["worker_id", "session_id", "model_name", "horizon_seconds"], observed=True):
        if not group["prediction_timestamp"].is_monotonic_increasing:
            raise ValueError("Prediction timestamps must be monotonic within worker/session/model/horizon.")
    return out.reset_index(drop=True)


def discover_prediction_artifacts(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    replay_cfg = cfg.get("replay", {})
    root = cfg.get("paths", {}).get("run_root", "experiments/runs")
    fast_dir = replay_cfg.get("fast_prediction_run_dir") or str(latest_run(root, "fast_wesad_"))
    slow_dir = replay_cfg.get("slow_prediction_run_dir") or str(latest_run(root, "slow_tft_"))
    fast_model = str(replay_cfg.get("fast_model_name", "tcn"))
    slow_model = str(replay_cfg.get("slow_model_name", "xgboost"))
    return standardize_fast_predictions(fast_dir, fast_model), standardize_slow_predictions(slow_dir, slow_model)

