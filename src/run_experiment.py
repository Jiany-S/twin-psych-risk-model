"""Run end-to-end experiment: WESAD/synthetic -> features -> XGBoost/TFT -> artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .data.loader import load_or_generate
from .data.preprocess import preprocess_dataframe
from .data.schema import DataSchema
from .data.windowing import (
    build_windows,
    create_subject_holdout_splits,
    create_time_splits,
    engineer_window_features,
)
from .data.time_semantics import assert_regular_timestamps, infer_temporal_spec
from .profiles.worker_profile import (
    CalibrationPolicy,
    apply_calibration_normalization,
    apply_global_normalization,
    attach_calibration_columns,
    build_profile_feature_table,
    fit_calibration_table,
    fit_global_normalization,
)
from .training.plotting import (
    plot_calibration,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr,
    plot_roc,
    plot_timeseries,
)
from .training.baseline_train import train_classifier_baselines
from .training.tft_train import train_tft_task
from .training.xgb_train import train_xgb_tasks
from .utils.io import load_merged_yaml, load_yaml, save_json
from .utils.logging import setup_logger
from .utils.paths import create_run_dir
from .utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the risk forecasting experiment.")
    parser.add_argument("--config", type=str, default="src/config/default.yaml")
    return parser.parse_args()


def _assert_chronological(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, schema: DataSchema) -> None:
    for worker_id in train_df[schema.worker_id].unique():
        tr = train_df[train_df[schema.worker_id] == worker_id][schema.time_idx]
        va = val_df[val_df[schema.worker_id] == worker_id][schema.time_idx]
        te = test_df[test_df[schema.worker_id] == worker_id][schema.time_idx]
        if len(tr) and len(va) and int(va.min()) <= int(tr.max()):
            raise ValueError(f"Chronological split violated for worker {worker_id}: val <= train.")
        if len(va) and len(te) and int(te.min()) <= int(va.max()):
            raise ValueError(f"Chronological split violated for worker {worker_id}: test <= val.")


def _assert_disjoint_subjects(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, schema: DataSchema) -> None:
    train_ids = set(train_df[schema.worker_id].astype(str).unique())
    val_ids = set(val_df[schema.worker_id].astype(str).unique())
    test_ids = set(test_df[schema.worker_id].astype(str).unique())
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Subject sets overlap across train/val/test.")


def _impute_raw_split(split_df: pd.DataFrame, schema: DataSchema) -> pd.DataFrame:
    frame = split_df.copy()
    cols = list(schema.physiology) + list(schema.robot_context)
    for optional_signal in ("resp", "accel"):
        if optional_signal in frame.columns:
            cols.append(optional_signal)
    for col in cols:
        if col not in frame.columns:
            frame[col] = 0.0
        frame[col] = frame.groupby(schema.worker_id, observed=True)[col].ffill().bfill().fillna(0.0)
    frame[schema.hazard_zone] = frame[schema.hazard_zone].fillna(0).astype(int)
    frame[schema.task_phase] = frame[schema.task_phase].fillna("default")
    return frame


def _profile_transform(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, schema: DataSchema, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    profiles_cfg = cfg.get("profiles", {})
    normalization_cfg = cfg.get("normalization", {})
    calibration_cfg = cfg.get("calibration", {})
    mode = str(normalization_cfg.get("mode", "global")).lower()
    if mode == "online":
        raise ValueError("normalization.mode='online' was removed; use 'global' or 'calibration'.")
    if mode not in {"global", "calibration"}:
        raise ValueError("normalization.mode must be one of: global, calibration.")

    policy = CalibrationPolicy(
        protocol_labels=tuple(str(v).lower() for v in calibration_cfg.get("protocol_labels", ["baseline", "rest"])),
        task_phases=tuple(str(v).lower() for v in calibration_cfg.get("task_phases", [])),
        max_rows_per_subject=calibration_cfg.get("max_rows_per_subject"),
        include_median=bool(calibration_cfg.get("include_median", True)),
        robust_scale=str(calibration_cfg.get("robust_scale", "iqr")).lower(),
    )
    global_stats = fit_global_normalization(train_df, schema.physiology)
    train_cal, train_cal_diag = fit_calibration_table(
        train_df, schema.worker_id, schema.protocol_label, schema.task_phase, schema.physiology, policy, global_stats
    )
    val_cal, val_cal_diag = fit_calibration_table(
        val_df, schema.worker_id, schema.protocol_label, schema.task_phase, schema.physiology, policy, global_stats
    )
    test_cal, test_cal_diag = fit_calibration_table(
        test_df, schema.worker_id, schema.protocol_label, schema.task_phase, schema.physiology, policy, global_stats
    )

    train_attached = attach_calibration_columns(train_df, train_cal, schema.physiology)
    val_attached = attach_calibration_columns(val_df, val_cal, schema.physiology)
    test_attached = attach_calibration_columns(test_df, test_cal, schema.physiology)

    if mode == "calibration":
        train_z = apply_calibration_normalization(train_attached, schema.physiology)
        val_z = apply_calibration_normalization(val_attached, schema.physiology)
        test_z = apply_calibration_normalization(test_attached, schema.physiology)
    else:
        train_z = apply_global_normalization(train_attached, schema.physiology, global_stats)
        val_z = apply_global_normalization(val_attached, schema.physiology, global_stats)
        test_z = apply_global_normalization(test_attached, schema.physiology, global_stats)

    combined_cal = pd.concat([train_cal, val_cal, test_cal], ignore_index=True).drop_duplicates("worker_id")
    source_df = pd.concat([train_z, val_z, test_z], ignore_index=True)
    static, static_cols = build_profile_feature_table(
        combined_cal,
        source_df,
        worker_col=schema.worker_id,
        specialization_col=schema.specialization_col,
        experience_col=schema.experience_col,
        include_calibration_features=bool(profiles_cfg.get("include_calibration_features", False)),
        include_role_metadata=bool(profiles_cfg.get("include_role_metadata", False)),
        include_experience_metadata=bool(profiles_cfg.get("include_experience_metadata", False)),
    )
    if not static_cols:
        static = pd.DataFrame(columns=["worker_id"])

    return train_z, val_z, test_z, static, {
        "enabled": bool(static_cols),
        "normalization_mode": mode,
        "global_fit_subjects": sorted(train_df[schema.worker_id].astype(str).unique().tolist()),
        "calibration_policy": policy.__dict__,
        "calibration_diagnostics": {"train": train_cal_diag, "val": val_cal_diag, "test": test_cal_diag},
        "profile_feature_columns": static_cols,
        "include_calibration_features": bool(profiles_cfg.get("include_calibration_features", False)),
        "include_role_metadata": bool(profiles_cfg.get("include_role_metadata", False)),
        "include_experience_metadata": bool(profiles_cfg.get("include_experience_metadata", False)),
    }


def _attach_static(df: pd.DataFrame, static: pd.DataFrame, schema: DataSchema, use_profile_inputs: bool) -> pd.DataFrame:
    frame = df.copy()
    profile_cols = [c for c in static.columns if c != "worker_id"] if not static.empty else []
    merge_cols = ["worker_id", *[c for c in profile_cols if c not in frame.columns]]
    if use_profile_inputs and len(merge_cols) > 1:
        frame = frame.merge(static[merge_cols], on="worker_id", how="left")
    for col in profile_cols:
        if col not in frame.columns:
            continue
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    return frame


def _save_processed(run_dir: Path, train_w, val_w, test_w, split_manifest: pd.DataFrame, flat_df: pd.DataFrame) -> None:
    processed_dir = run_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    windows_all = np.concatenate([train_w.X_windows, val_w.X_windows, test_w.X_windows], axis=0)
    y_stress = np.concatenate([train_w.y_stress, val_w.y_stress, test_w.y_stress], axis=0)
    y_comfort = np.concatenate([train_w.y_comfort, val_w.y_comfort, test_w.y_comfort], axis=0)
    meta = pd.concat(
        [
            train_w.meta.assign(split="train"),
            val_w.meta.assign(split="val"),
            test_w.meta.assign(split="test"),
        ],
        ignore_index=True,
    )
    np.save(processed_dir / "windows_X.npy", windows_all)
    np.save(processed_dir / "windows_y_stress.npy", y_stress)
    np.save(processed_dir / "windows_y_comfort.npy", y_comfort)
    meta.to_csv(processed_dir / "meta.csv", index=False)
    split_manifest.to_csv(processed_dir / "splits.csv", index=False)
    split_manifest.to_csv(run_dir / "splits.csv", index=False)
    flat_df.to_csv(processed_dir / "tft_flat.csv", index=False)


def _safe_git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
            .strip()
        )
    except Exception:
        return "unknown"


def _profile_feature_stats(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    cols = [c for c in train_df.columns if c.startswith("calib_") or c.startswith("norm_")]
    cols += [c for c in ["role_metadata", "experience_metadata"] if c in train_df.columns]
    cols = sorted(set(cols))
    out: dict[str, Any] = {}
    for split_name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
        split_stats: dict[str, Any] = {}
        for col in cols:
            series = pd.to_numeric(frame[col], errors="coerce")
            non_na = series.dropna()
            unique_ratio = float(non_na.nunique() / max(1, len(non_na)))
            split_stats[col] = {
                "mean": float(non_na.mean()) if len(non_na) else 0.0,
                "std": float(non_na.std()) if len(non_na) else 0.0,
                "min": float(non_na.min()) if len(non_na) else 0.0,
                "max": float(non_na.max()) if len(non_na) else 0.0,
                "unique_ratio": unique_ratio,
                "missing_pct": float(series.isna().mean()),
            }
        out[split_name] = split_stats
    return out


def _numeric_feature_stats(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = sorted(
        set(train_df.select_dtypes(include=[np.number]).columns)
        | set(val_df.select_dtypes(include=[np.number]).columns)
        | set(test_df.select_dtypes(include=[np.number]).columns)
    )

    def split_stats(frame: pd.DataFrame) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        for col in numeric_cols:
            if col not in frame.columns:
                continue
            arr = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
            finite = arr[np.isfinite(arr)]
            stats[col] = {
                "min": float(np.min(finite)) if finite.size else 0.0,
                "max": float(np.max(finite)) if finite.size else 0.0,
                "mean": float(np.mean(finite)) if finite.size else 0.0,
                "std": float(np.std(finite)) if finite.size else 0.0,
                "nan_pct": float(np.mean(np.isnan(arr))),
                "inf_pct": float(np.mean(np.isinf(arr))),
            }
        return stats

    return {
        "train": split_stats(train_df),
        "val": split_stats(val_df),
        "test": split_stats(test_df),
    }


def _engineered_feature_stats(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, feature_names: list[str]
) -> dict[str, Any]:
    def _stats(arr: np.ndarray) -> dict[str, Any]:
        finite = np.isfinite(arr)
        return {
            "shape": list(arr.shape),
            "nan_count": int(np.isnan(arr).sum()),
            "inf_count": int(np.isinf(arr).sum()),
            "finite_ratio": float(finite.mean()) if arr.size else 1.0,
        }

    def _per_feature(arr: np.ndarray) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for idx, name in enumerate(feature_names):
            col = arr[:, idx]
            finite = col[np.isfinite(col)]
            out[name] = {
                "min": float(np.min(finite)) if finite.size else 0.0,
                "max": float(np.max(finite)) if finite.size else 0.0,
                "mean": float(np.mean(finite)) if finite.size else 0.0,
                "std": float(np.std(finite)) if finite.size else 0.0,
                "nan_pct": float(np.mean(np.isnan(col))),
                "inf_pct": float(np.mean(np.isinf(col))),
            }
        return out

    return {
        "global": {
            "train": _stats(X_train),
            "val": _stats(X_val),
            "test": _stats(X_test),
        },
        "per_feature": {
            "train": _per_feature(X_train),
            "val": _per_feature(X_val),
            "test": _per_feature(X_test),
        },
    }


def _comparison(metrics: dict[str, Any], include_comfort: bool) -> dict[str, Any]:
    def get(path: tuple[str, ...], default=None):
        ptr = metrics
        for key in path:
            if not isinstance(ptr, dict) or key not in ptr:
                return default
            ptr = ptr[key]
        return ptr

    model_scores = {}
    for model_name, block in metrics.items():
        if isinstance(block, dict) and isinstance(block.get("stress"), dict) and "auroc" in block["stress"]:
            score = float(block["stress"].get("auroc", np.nan))
            if np.isfinite(score):
                model_scores[model_name] = score
    winner = max(model_scores, key=model_scores.get) if model_scores else "n/a"
    xgb_auroc = float(get(("xgboost", "stress", "auroc"), np.nan))
    tft_auroc = float(get(("tft", "stress", "auroc"), np.nan))
    result = {
        "primary_winner_by_auroc": winner,
        "delta_primary_auroc": float(tft_auroc - xgb_auroc),
        "primary_auroc_by_model": model_scores,
    }
    if include_comfort:
        result["comfort_winner_by_rmse"] = (
            "xgboost" if (get(("xgboost", "comfort", "rmse"), 1e9) <= get(("tft", "comfort", "rmse"), 1e9)) else "tft"
        )
        result["delta_comfort_rmse"] = float(
            get(("tft", "comfort", "rmse"), np.nan) - get(("xgboost", "comfort", "rmse"), np.nan)
        )
    return result


def _write_results_md(
    run_dir: Path,
    metrics: dict[str, Any],
    ablation: dict[str, Any],
    dataset_name: str,
    split_desc: str,
    profiles_enabled: bool,
) -> None:
    tft_stress = metrics.get("tft", {}).get("stress", {})
    primary_label = str(metrics.get("primary_target", {}).get("target_name", "primary target"))
    tft_warning = ""
    if isinstance(tft_stress, dict):
        if str(tft_stress.get("auroc", "")).lower() == "nan":
            tft_warning = "TFT AUROC is NaN; test split may contain a single class or too few windows."
    cfg = metrics.get("config", {})
    temporal = metrics.get("time_semantics", {})
    effective_hz = float(temporal.get("effective_sampling_rate_hz", 1.0))
    test_balance = metrics.get("class_balance", {}).get("test", {})
    threshold_policy = metrics.get("xgboost", {}).get("stress", {}).get("threshold_policy", "n/a")
    model_rows = []
    for model_name, block in metrics.items():
        if isinstance(block, dict) and isinstance(block.get("stress"), dict) and "auroc" in block["stress"]:
            vals = block["stress"]
            model_rows.append(
                "| {name} | {auroc} | {auprc} | {f1} | {precision} | {recall} | {specificity} | {balanced_accuracy} | {brier} | {ece} | {prevalence} | {predicted_positive_rate} |".format(
                    name=model_name,
                    auroc=vals.get("auroc", "n/a"),
                    auprc=vals.get("auprc", "n/a"),
                    f1=vals.get("f1", "n/a"),
                    precision=vals.get("precision", "n/a"),
                    recall=vals.get("recall", "n/a"),
                    specificity=vals.get("specificity", "n/a"),
                    balanced_accuracy=vals.get("balanced_accuracy", "n/a"),
                    brier=vals.get("brier", "n/a"),
                    ece=vals.get("ece", "n/a"),
                    prevalence=vals.get("prevalence", "n/a"),
                    predicted_positive_rate=vals.get("predicted_positive_rate", "n/a"),
                )
            )
    lines = [
        "# Experiment Results",
        "",
        "## Setup",
        f"- Dataset: {dataset_name}",
        f"- Split: {split_desc}",
        f"- Profiles enabled: {profiles_enabled}",
        f"- Task: {metrics.get('task_name', 'stress')}",
        f"- Effective sampling rate (Hz): {effective_hz:.3f}",
        "- Context/Horizon/Stride (seconds): "
        f"{float(temporal.get('context_seconds', 0.0)):.2f} / "
        f"{float(temporal.get('forecast_horizon_seconds', 0.0)):.2f} / "
        f"{float(temporal.get('inference_stride_seconds', 0.0)):.2f}",
        f"- Expected prediction cadence (seconds): {float(temporal.get('expected_prediction_cadence_seconds', 0.0)):.2f}",
        f"- Test prevalence: {test_balance}",
        f"- Threshold policy: {threshold_policy}",
        "",
        "## Key Findings",
        "- No-leakage profile fitting: baselines fit on train split only and reused on val/test.",
        f"- {primary_label} AUROC winner: {metrics.get('comparison', {}).get('primary_winner_by_auroc', 'n/a')}",
        f"- Comfort RMSE winner: {metrics.get('comparison', {}).get('comfort_winner_by_rmse', 'n/a')}",
        f"- TFT sanity note: {tft_warning}" if tft_warning else "- TFT sanity note: n/a",
        "",
        "## Primary Target Comparison Table",
        "| Model | AUROC | AUPRC | F1 | Precision | Recall | Specificity | Balanced Acc. | Brier | ECE | Prevalence | Pred. + Rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *(model_rows or ["| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"]),
        "",
        "## Metrics JSON Snapshot",
        "```json",
        json.dumps({k: v for k, v in metrics.items() if k != "config"}, indent=2),
        "```",
        "",
        "## Ablation (Profiles ON vs OFF)",
        "```json",
        json.dumps(ablation, indent=2),
        "```",
    ]
    (run_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")


def _write_profile_ablation_report(run_dir: Path, metrics: dict[str, Any]) -> None:
    info = metrics.get("profiles_info", {})
    lines = [
        "# Profile Ablation Report",
        "",
        f"- Normalization mode: {info.get('normalization_mode', 'n/a')}",
        f"- Include calibration features: {info.get('include_calibration_features', False)}",
        f"- Include role metadata: {info.get('include_role_metadata', False)}",
        f"- Include experience metadata: {info.get('include_experience_metadata', False)}",
        f"- Profile feature columns: {info.get('profile_feature_columns', [])}",
        "",
        "## Model Feature Lists",
    ]
    report_json: dict[str, Any] = {"profiles_info": info, "models": {}}
    for model_name, block in metrics.items():
        if isinstance(block, dict) and isinstance(block.get("stress"), dict) and "feature_names" in block["stress"]:
            feature_names = block["stress"].get("feature_names", [])
            model_metrics = {
                key: block["stress"].get(key)
                for key in ["auroc", "auprc", "f1", "precision", "recall", "specificity", "balanced_accuracy", "brier", "ece"]
            }
            report_json["models"][model_name] = {"feature_names": feature_names, "metrics": model_metrics}
            lines += [
                "",
                f"### {model_name}",
                f"- Feature count: {len(feature_names)}",
                f"- Features: {feature_names}",
                f"- Metrics: {model_metrics}",
            ]
    (run_dir / "profile_ablation_report.md").write_text("\n".join(lines), encoding="utf-8")
    save_json(report_json, run_dir / "profile_ablation_report.json")


def run_experiment(config_path: str) -> Path:
    default_cfg_path = Path(__file__).resolve().parent / "config" / "default.yaml"
    if Path(config_path).resolve() == default_cfg_path.resolve():
        cfg = load_yaml(config_path)
    else:
        cfg = load_merged_yaml(default_cfg_path, config_path)
    logger = setup_logger()
    seed_everything(int(cfg.get("reproducibility", {}).get("seed", cfg["split"]["seed"])))
    schema = DataSchema.from_config(cfg)
    debug = bool(cfg.get("debug", False))
    temporal = infer_temporal_spec(cfg)

    raw_df = load_or_generate(cfg, schema)
    target_metadata = raw_df.attrs.get("target_metadata", {})
    feature_metadata = raw_df.attrs.get("feature_metadata", {})
    loader_time_metadata = raw_df.attrs.get("time_metadata", {})
    frame = preprocess_dataframe(cfg, raw_df, schema)
    assert_regular_timestamps(frame, schema.worker_id, schema.timestamp, temporal.row_interval_seconds)
    split_mode = str(cfg.get("split", {}).get("mode", "time")).lower()
    split_desc = "time-per-worker"
    if split_mode == "subject_holdout":
        subjects_cfg = cfg.get("dataset", {}).get("subjects", [])
        if not subjects_cfg:
            raise ValueError("split.mode=subject_holdout requires dataset.subjects list.")
        subjects = [str(s) for s in subjects_cfg]
        split_cfg = cfg.get("split", {})
        train_subjects = [str(s) for s in split_cfg.get("train_subjects", subjects[:5])]
        val_subjects = [str(s) for s in split_cfg.get("val_subjects", subjects[5:6])]
        test_subjects = [str(s) for s in split_cfg.get("test_subjects", subjects[6:8])]
        train_df, val_df, test_df, split_manifest = create_subject_holdout_splits(
            frame, schema, train_subjects, val_subjects, test_subjects
        )
        _assert_disjoint_subjects(train_df, val_df, test_df, schema)
        split_desc = f"subject-holdout train={train_subjects}, val={val_subjects}, test={test_subjects}"
    else:
        train_df, val_df, test_df, split_manifest = create_time_splits(
            frame, schema, cfg["split"]["train_ratio"], cfg["split"]["val_ratio"], cfg["split"]["test_ratio"]
        )
        _assert_chronological(train_df, val_df, test_df, schema)

    train_df = _impute_raw_split(train_df, schema)
    val_df = _impute_raw_split(val_df, schema)
    test_df = _impute_raw_split(test_df, schema)
    if debug:
        for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
            counts = df[schema.stress_target].value_counts(dropna=False).to_dict()
            logger.info("Split %s raw rows=%d primary_target_counts=%s", name, len(df), counts)

    train_df, val_df, test_df, static_profiles, profiles_info = _profile_transform(train_df, val_df, test_df, schema, cfg)
    use_profile_inputs = not static_profiles.empty and len(static_profiles.columns) > 1
    train_df = _attach_static(train_df, static_profiles, schema, use_profile_inputs=use_profile_inputs)
    val_df = _attach_static(val_df, static_profiles, schema, use_profile_inputs=use_profile_inputs)
    test_df = _attach_static(test_df, static_profiles, schema, use_profile_inputs=use_profile_inputs)
    numeric_diag = _numeric_feature_stats(train_df, val_df, test_df)
    required_numeric = list(schema.physiology) + list(schema.robot_context) + [schema.hazard_zone]
    for split_name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
        for col in required_numeric:
            if col not in frame.columns:
                continue
            values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
            if np.isnan(values).any() or np.isinf(values).any():
                raise ValueError(f"{split_name} has NaN/Inf in required numeric column {col}.")
    if debug:
        logger.info("Profiles: %s", profiles_info)
    flat_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    window_length = temporal.context_steps
    horizon = temporal.horizon_steps
    window_step = temporal.stride_steps
    min_split_len = min(
        min(len(part[part[schema.worker_id] == wid]) for wid in part[schema.worker_id].unique())
        for part in (train_df, val_df, test_df)
    )
    if min_split_len <= horizon or min_split_len <= window_length + horizon:
        raise ValueError(
            "Split segments are too short for the configured context_seconds and forecast_horizon_seconds."
        )
    window_kwargs = {
        "row_interval_seconds": temporal.row_interval_seconds,
        "context_seconds": temporal.context_seconds,
        "forecast_horizon_seconds": temporal.forecast_horizon_seconds,
        "inference_stride_seconds": temporal.inference_stride_seconds,
    }
    train_w = build_windows(train_df, schema, window_length, horizon, window_step=window_step, **window_kwargs)
    val_w = build_windows(val_df, schema, window_length, horizon, window_step=window_step, **window_kwargs)
    test_w = build_windows(test_df, schema, window_length, horizon, window_step=window_step, **window_kwargs)
    if debug:
        logger.info(
            "Window counts train=%d val=%d test=%d",
            len(train_w.y_stress),
            len(val_w.y_stress),
            len(test_w.y_stress),
        )

    run_paths = create_run_dir(cfg["paths"]["run_root"])
    (run_paths.root / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    _save_processed(run_paths.root, train_w, val_w, test_w, split_manifest, flat_df)

    sampling_rate = temporal.effective_sampling_rate_hz
    include_freq = bool(cfg["features"].get("hrv", {}).get("include_freq_domain", True))
    scr_threshold = float(cfg["features"].get("eda", {}).get("scr_threshold", 0.05))
    min_scr_distance = int(cfg["features"].get("eda", {}).get("min_scr_distance", 3))
    X_train, feat_names = engineer_window_features(
        train_w.X_windows, schema, sampling_rate, include_freq, scr_threshold, min_scr_distance
    )
    X_val, _ = engineer_window_features(
        val_w.X_windows, schema, sampling_rate, include_freq, scr_threshold, min_scr_distance
    )
    X_test, _ = engineer_window_features(
        test_w.X_windows, schema, sampling_rate, include_freq, scr_threshold, min_scr_distance
    )
    engineered_diag = _engineered_feature_stats(X_train, X_val, X_test, feat_names)
    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    y_stress_all = np.concatenate([train_w.y_stress, val_w.y_stress, test_w.y_stress], axis=0)
    y_comfort_all = np.concatenate([train_w.y_comfort, val_w.y_comfort, test_w.y_comfort], axis=0)
    meta_all = pd.concat(
        [train_w.meta.assign(split="train"), val_w.meta.assign(split="val"), test_w.meta.assign(split="test")],
        ignore_index=True,
    )
    split_idx = {
        "train": np.arange(len(train_w.y_stress)),
        "val": np.arange(len(train_w.y_stress), len(train_w.y_stress) + len(val_w.y_stress)),
        "test": np.arange(len(train_w.y_stress) + len(val_w.y_stress), len(y_stress_all)),
    }
    window_counts = {"train": int(len(train_w.y_stress)), "val": int(len(val_w.y_stress)), "test": int(len(test_w.y_stress))}
    def _balance(arr: np.ndarray) -> dict[str, int]:
        vals, counts = np.unique(arr, return_counts=True)
        return {str(v): int(c) for v, c in zip(vals, counts)}

    class_balance = {
        "train": _balance(train_w.y_stress),
        "val": _balance(val_w.y_stress),
        "test": _balance(test_w.y_stress),
    }
    if window_counts["test"] < 100:
        logger.warning("Test window count is very small (%d). Metrics may be unreliable.", window_counts["test"])

    include_comfort = bool(cfg.get("targets", {}).get("multi_head", {}).get("enabled", True))
    models_cfg = cfg.get("models", {})
    run_xgb = bool(models_cfg.get("run_xgb", True))
    run_tft = bool(models_cfg.get("run_tft", True))
    metrics: dict[str, Any] = {
        "config": cfg,
        "git_commit": _safe_git_hash(),
        "window_counts": window_counts,
        "class_balance": class_balance,
        "profiles_info": profiles_info,
        "task_name": str(cfg.get("experiment", {}).get("task_name", schema.primary_target_name)),
        "primary_target": {
            "target_name": schema.primary_target_name,
            "label_col": schema.primary_target,
            "task_type": schema.primary_task_type,
            "metadata": target_metadata.get(schema.primary_target_name, {}),
        },
        "target_metadata": target_metadata,
        "feature_metadata": feature_metadata,
        "loader_time_metadata": loader_time_metadata,
        "time_semantics": temporal.to_dict(),
    }
    xgb_out = None
    baseline_out = {}
    tft_stress = None
    tft_comfort = None

    try:
        baseline_out = train_classifier_baselines(
            cfg=cfg,
            feature_matrix=X_all,
            feature_names=feat_names,
            y_primary=y_stress_all,
            meta=meta_all,
            static_profiles=static_profiles,
            split_indices=split_idx,
            run_dir=run_paths.root,
            use_profiles=use_profile_inputs,
        )
        for model_name, artifact in baseline_out.items():
            metrics[model_name] = {"stress": artifact.metrics | {"model_path": str(artifact.model_path)}}
            if artifact.feature_importance:
                importance_path = run_paths.root / "models" / f"{model_name}_feature_importance.json"
                importance_path.write_text(json.dumps(artifact.feature_importance, indent=2), encoding="utf-8")
                metrics[model_name]["stress"]["feature_importance_path"] = str(importance_path)
                plot_feature_importance(
                    np.array([item["importance"] for item in artifact.feature_importance]),
                    [item["feature"] for item in artifact.feature_importance],
                    run_paths.plots / f"feature_importance_{model_name}_primary.png",
                    top_k=cfg["report"]["top_k_features"],
                )
    except Exception as exc:
        logger.exception("Classical baseline suite failed: %s", exc)
        metrics["baselines"] = {"error": str(exc)}

    try:
        if not run_xgb:
            raise RuntimeError("XGBoost disabled by config (models.run_xgb=false).")
        xgb_out = train_xgb_tasks(
            cfg=cfg,
            feature_matrix=X_all,
            feature_names=feat_names,
            y_stress=y_stress_all,
            y_comfort=y_comfort_all,
            meta=meta_all,
            static_profiles=static_profiles,
            split_indices=split_idx,
            run_dir=run_paths.root,
            use_profiles=use_profile_inputs,
            model_prefix="xgb",
        )
        xgb_metrics = {"stress": xgb_out["stress"].metrics | {"model_path": str(xgb_out["stress"].model_path)}}
        if include_comfort:
            xgb_metrics["comfort"] = xgb_out["comfort"].metrics | {"model_path": str(xgb_out["comfort"].model_path)}
        metrics["xgboost"] = xgb_metrics
        plot_feature_importance(
            np.array([item["importance"] for item in xgb_out["stress"].feature_importance]),
            [item["feature"] for item in xgb_out["stress"].feature_importance],
            run_paths.plots / "feature_importance_xgb_stress.png",
            top_k=cfg["report"]["top_k_features"],
        )
    except Exception as exc:
        logger.exception("XGBoost pipeline failed: %s", exc)
        metrics["xgboost"] = {"error": str(exc)}

    try:
        if not run_tft:
            raise RuntimeError("TFT disabled by config (models.run_tft=false).")
        tft_stress = train_tft_task(
            cfg=cfg,
            train_df=train_df.copy(),
            val_df=val_df.copy(),
            test_df=test_df.copy(),
            schema=schema,
            target_col=schema.stress_target,
            task_type="classification",
            run_dir=run_paths.root,
            window_length=window_length,
            horizon=horizon,
            window_step=window_step,
            model_name="tft_stress",
            use_profiles=use_profile_inputs,
        )
        tft_metrics = {"stress": tft_stress.metrics | {"checkpoint_path": str(tft_stress.checkpoint_path)}}
        if include_comfort:
            tft_comfort = train_tft_task(
                cfg=cfg,
                train_df=train_df.copy(),
                val_df=val_df.copy(),
                test_df=test_df.copy(),
                schema=schema,
                target_col=schema.comfort_target,
                task_type="regression",
                run_dir=run_paths.root,
                window_length=window_length,
                horizon=horizon,
                window_step=window_step,
                model_name="tft_comfort",
                use_profiles=use_profile_inputs,
            )
            tft_metrics["comfort"] = tft_comfort.metrics | {"checkpoint_path": str(tft_comfort.checkpoint_path)}
        metrics["tft"] = tft_metrics
    except Exception as exc:
        logger.exception("TFT pipeline failed: %s", exc)
        metrics["tft"] = {"error": str(exc)}

    # Ablation: profile input columns OFF using XGBoost only for fast comparison.
    ablation: dict[str, Any] = {}
    try:
        off_cfg = json.loads(json.dumps(cfg))
        off_cfg.setdefault("profiles", {})
        off_cfg["profiles"]["include_calibration_features"] = False
        off_cfg["profiles"]["include_role_metadata"] = False
        off_cfg["profiles"]["include_experience_metadata"] = False
        off_xgb = train_xgb_tasks(
            cfg=off_cfg,
            feature_matrix=X_all,
            feature_names=feat_names,
            y_stress=y_stress_all,
            y_comfort=y_comfort_all,
            meta=meta_all,
            static_profiles=None,
            split_indices=split_idx,
            run_dir=run_paths.root,
            use_profiles=False,
            model_prefix="xgb_profiles_off",
        )
        ablation = {"profiles_on": {"stress_auroc": metrics.get("xgboost", {}).get("stress", {}).get("auroc")}}
        ablation["profiles_off"] = {"stress_auroc": off_xgb["stress"].metrics.get("auroc")}
        if include_comfort:
            ablation["profiles_on"]["comfort_rmse"] = metrics.get("xgboost", {}).get("comfort", {}).get("rmse")
            ablation["profiles_off"]["comfort_rmse"] = off_xgb["comfort"].metrics.get("rmse")
    except Exception as exc:
        ablation = {"error": str(exc)}

    # Plots
    y_stress_test = y_stress_all[split_idx["test"]]
    y_comfort_test = y_comfort_all[split_idx["test"]]
    primary_plot_series = [(name, artifact.predictions) for name, artifact in baseline_out.items()]
    if xgb_out is not None:
        primary_plot_series.append(("xgboost", xgb_out["stress"].predictions))
        cm = np.array(metrics.get("xgboost", {}).get("stress", {}).get("confusion_matrix_default", [[0, 0], [0, 0]]))
        plot_confusion_matrix(cm, run_paths.plots / "confusion_stress.png")
        plot_timeseries(
            meta_all.iloc[split_idx["test"]],
            y_stress_test,
            xgb_out["stress"].predictions,
            run_paths.plots / "timeseries_stress_overlay.png",
        )
        if include_comfort:
            plot_timeseries(
                meta_all.iloc[split_idx["test"]],
                y_comfort_test,
                xgb_out["comfort"].predictions,
                run_paths.plots / "timeseries_comfort_overlay.png",
            )

    if tft_stress is not None and len(tft_stress.targets) == len(tft_stress.predictions):
        if len(tft_stress.predictions) == len(y_stress_test):
            primary_plot_series.append(("tft", tft_stress.predictions))
        else:
            plot_roc(
                tft_stress.targets,
                [("tft", tft_stress.predictions)],
                run_paths.plots / "roc_curve_stress_tft.png",
            )
            plot_pr(
                tft_stress.targets,
                [("tft", tft_stress.predictions)],
                run_paths.plots / "pr_curve_stress_tft.png",
            )
            plot_calibration(
                tft_stress.targets,
                [("tft", tft_stress.predictions)],
                run_paths.plots / "calibration_stress_tft.png",
            )

    plot_roc(y_stress_test, primary_plot_series, run_paths.plots / "roc_curve_stress.png")
    plot_pr(y_stress_test, primary_plot_series, run_paths.plots / "pr_curve_stress.png")
    plot_calibration(y_stress_test, primary_plot_series, run_paths.plots / "calibration_stress.png")

    metrics["comparison"] = _comparison(metrics, include_comfort=include_comfort)
    metrics["ablation_profiles"] = ablation
    profile_stats = _profile_feature_stats(train_df, val_df, test_df)
    save_json(profile_stats, run_paths.root / "profile_feature_stats.json")
    metrics["profile_feature_stats_path"] = str(run_paths.root / "profile_feature_stats.json")
    save_json(temporal.to_dict(), run_paths.root / "time_semantics.json")
    metrics["time_semantics_path"] = str(run_paths.root / "time_semantics.json")
    save_json(numeric_diag, run_paths.root / "numeric_feature_stats.json")
    save_json(engineered_diag, run_paths.root / "engineered_feature_stats.json")
    metrics["numeric_feature_stats_path"] = str(run_paths.root / "numeric_feature_stats.json")
    metrics["engineered_feature_stats_path"] = str(run_paths.root / "engineered_feature_stats.json")
    save_json(metrics, run_paths.root / "metrics.json")
    _write_results_md(
        run_paths.root,
        metrics,
        ablation,
        dataset_name=str(
            cfg.get("dataset", {}).get("report_name", cfg.get("dataset", {}).get("name", "unknown"))
        ),
        split_desc=split_desc,
        profiles_enabled=use_profile_inputs,
    )
    _write_profile_ablation_report(run_paths.root, metrics)
    logger.info("Experiment complete. Artifacts saved to %s", run_paths.root)
    return run_paths.root


def main() -> None:
    args = parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
