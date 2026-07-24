"""Prediction artifact loading and validation for replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.utils.artifacts import config_hash, file_sha256, git_commit


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


@dataclass(frozen=True)
class ReplayArtifactPair:
    fast: pd.DataFrame
    slow: pd.DataFrame
    report: dict[str, Any]


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
            "target_name": cfg.get("experiment", {}).get("primary_target", "stress"),
            "positive_class": cfg.get("targets", {}).get("stress", {}).get("positive_class", "stress"),
            "timestamp_unit": "seconds",
            "stream_representation": cfg.get("stream", {}).get("representation"),
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
            "target_name": cfg.get("experiment", {}).get("primary_target", "stress"),
            "positive_class": cfg.get("targets", {}).get("stress", {}).get("positive_class", "stress"),
            "timestamp_unit": "seconds",
            "stream_representation": cfg.get("stream", {}).get("representation"),
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


def write_replay_manifest_fragment(role: str, run_dir: str | Path, cfg: dict[str, Any], prediction_file: str, model_name: str) -> None:
    manifest_path = cfg.get("paths", {}).get("replay_manifest")
    if not manifest_path:
        return
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _load_yaml(path)
    manifest.setdefault("replay_manifest_version", 1)
    manifest[role] = {
        "run_dir": str(Path(run_dir)),
        "prediction_file": prediction_file,
        "metadata_file": "config_resolved.yaml",
        "model_name": model_name,
    }
    split = cfg.get("split", {})
    manifest["alignment"] = {
        "subjects": [str(s) for s in split.get("test_subjects", [])],
        "target_name": cfg.get("experiment", {}).get("primary_target", "stress"),
        "positive_class": cfg.get("targets", {}).get("stress", {}).get("positive_class", "stress"),
        "probability_column": "calibrated_probability",
        "timestamp_unit": "seconds",
        "stream_representation": cfg.get("stream", {}).get("representation", "raw_signal"),
        "join_policy": "exact_prediction_timestamp",
        "minimum_overlap_fraction": 0.01,
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def load_prediction_artifacts_from_manifest(cfg: dict[str, Any]) -> ReplayArtifactPair:
    replay_cfg = cfg.get("replay", {})
    manifest_path = replay_cfg.get("manifest_path")
    if not manifest_path:
        raise ValueError(
            "Real-model physiological replay requires replay.manifest_path. "
            "Do not rely on implicit latest-run artifact discovery."
        )
    manifest = _load_yaml(Path(manifest_path))
    if int(manifest.get("replay_manifest_version", 0)) != 1:
        raise ValueError(f"Unsupported replay manifest version in {manifest_path}.")
    for section in ("fast", "slow", "alignment"):
        if section not in manifest:
            raise ValueError(f"Replay manifest {manifest_path} missing required section: {section}")
    fast_cfg = manifest["fast"]
    slow_cfg = manifest["slow"]
    for role, section in (("fast", fast_cfg), ("slow", slow_cfg)):
        for key in ("run_dir", "prediction_file", "metadata_file", "model_name"):
            if key not in section:
                raise ValueError(f"Replay manifest {manifest_path} missing {role}.{key}")
        metadata = Path(section["run_dir"]) / str(section["metadata_file"])
        if not metadata.exists():
            raise FileNotFoundError(f"Replay manifest {manifest_path} references missing metadata file: {metadata}")
    fast = standardize_fast_predictions(fast_cfg["run_dir"], str(fast_cfg["model_name"]))
    slow = standardize_slow_predictions(slow_cfg["run_dir"], str(slow_cfg["model_name"]))
    report = validate_artifact_pairing(fast, slow, manifest)
    return ReplayArtifactPair(fast=fast, slow=slow, report=report)


def validate_artifact_pairing(fast: pd.DataFrame, slow: pd.DataFrame, manifest: dict[str, Any]) -> dict[str, Any]:
    alignment = manifest.get("alignment", {})
    requested_subjects = {str(s) for s in alignment.get("subjects", [])}
    if not requested_subjects:
        raise ValueError("Replay manifest alignment.subjects must be non-empty.")
    report: dict[str, Any] = {
        "validation_checks": {},
        "shared_subjects": [],
        "timestamp_coverage": {},
        "dropped_prediction_count": {},
        "join_policy": alignment.get("join_policy"),
        "compatibility_decision": "rejected",
    }
    fast_subjects = set(fast["worker_id"].astype(str).unique())
    slow_subjects = set(slow["worker_id"].astype(str).unique())
    shared = sorted((fast_subjects & slow_subjects) & requested_subjects)
    report["shared_subjects"] = shared
    checks = report["validation_checks"]
    checks["subjects_overlap"] = bool(shared)
    checks["requested_subjects_available"] = requested_subjects.issubset(fast_subjects) and requested_subjects.issubset(slow_subjects)
    checks["target_name_match"] = _single_value(fast, "target_name") == _single_value(slow, "target_name") == alignment.get("target_name")
    checks["positive_class_match"] = _single_value(fast, "positive_class") == _single_value(slow, "positive_class") == alignment.get("positive_class")
    checks["timestamp_unit_match"] = _single_value(fast, "timestamp_unit") == _single_value(slow, "timestamp_unit") == alignment.get("timestamp_unit")
    checks["stream_representation_match"] = _single_value(fast, "stream_representation") == _single_value(slow, "stream_representation") == alignment.get("stream_representation")
    checks["config_hashes_present"] = fast["config_hash"].astype(str).str.len().all() and slow["config_hash"].astype(str).str.len().all()
    if alignment.get("join_policy") != "exact_prediction_timestamp":
        raise ValueError("Only exact_prediction_timestamp replay join policy is supported.")
    fast_aligned = fast[fast["worker_id"].astype(str).isin(shared)]
    slow_aligned = slow[slow["worker_id"].astype(str).isin(shared)]
    fast_keys = fast_aligned[["worker_id", "session_id", "prediction_timestamp"]].drop_duplicates()
    slow_keys = slow_aligned[["worker_id", "session_id", "prediction_timestamp"]].drop_duplicates()
    overlap = fast_keys.merge(slow_keys, on=["worker_id", "session_id", "prediction_timestamp"], how="inner")
    denominator = max(1, min(len(fast_keys), len(slow_keys)))
    overlap_fraction = len(overlap) / denominator
    minimum_overlap = float(alignment.get("minimum_overlap_fraction", 0.01))
    checks["timestamp_overlap_sufficient"] = overlap_fraction >= minimum_overlap
    report["timestamp_coverage"] = {
        "fast_prediction_origins": int(len(fast_keys)),
        "slow_prediction_origins": int(len(slow_keys)),
        "aligned_prediction_origins": int(len(overlap)),
        "overlap_fraction": float(overlap_fraction),
        "minimum_overlap_fraction": minimum_overlap,
    }
    report["dropped_prediction_count"] = {
        "fast": int(len(fast_keys) - len(overlap)),
        "slow": int(len(slow_keys) - len(overlap)),
    }
    merged_targets = fast_aligned.merge(
        slow_aligned,
        on=["worker_id", "session_id", "prediction_timestamp", "target_timestamp", "horizon_seconds"],
        suffixes=("_fast", "_slow"),
    )
    if not merged_targets.empty:
        checks["overlapping_targets_agree"] = bool((merged_targets["target_fast"] == merged_targets["target_slow"]).all())
    else:
        checks["overlapping_targets_agree"] = True
    failed = [name for name, passed in checks.items() if not bool(passed)]
    if failed:
        raise ValueError(f"Incompatible replay artifacts: {failed}")
    report["compatibility_decision"] = "accepted"
    return report


def _single_value(df: pd.DataFrame, column: str) -> Any:
    if column not in df:
        return None
    values = {str(v) for v in df[column].dropna().unique()}
    if len(values) != 1:
        raise ValueError(f"Prediction artifact column {column} must contain exactly one value, found {sorted(values)}")
    return next(iter(values))
