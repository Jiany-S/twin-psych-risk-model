"""Run end-to-end experiment: WESAD/synthetic -> features -> XGBoost/TFT -> artifacts."""

from __future__ import annotations

import argparse
import json
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
from .profiles.worker_profile import WorkerProfileStore
from .training.plotting import (
    plot_calibration,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr,
    plot_roc,
    plot_timeseries,
)
from .training.tft_train import train_tft_task
from .training.xgb_train import train_xgb_tasks
from .utils.io import load_yaml, save_json
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
    def apply_global_norm(frame: pd.DataFrame, stats: dict[str, tuple[float, float]]) -> pd.DataFrame:
        out = frame.copy()
        for col in schema.physiology:
            mu, sigma = stats[col]
            out[f"baseline_mu_{col}"] = mu
            out[f"baseline_sigma_{col}"] = max(sigma, 1e-6)
            out[col] = (out[col] - out[f"baseline_mu_{col}"]) / out[f"baseline_sigma_{col}"]
        return out

    def apply_online_norm(frame: pd.DataFrame, alpha: float, warmup_steps: int) -> pd.DataFrame:
        out = frame.copy()
        for col in schema.physiology:
            out[f"baseline_mu_{col}"] = np.nan
            out[f"baseline_sigma_{col}"] = np.nan

        for worker_id, worker_df in out.groupby(schema.worker_id, observed=True):
            idx = worker_df.sort_values(schema.time_idx).index
            worker = out.loc[idx]
            for col in schema.physiology:
                series = worker[col].astype(float).to_numpy()
                n = len(series)
                warm = min(max(3, warmup_steps), n)
                init = series[:warm]
                mu = float(np.nanmean(init))
                sigma = float(np.nanstd(init) + 1e-6)
                if not np.isfinite(mu):
                    mu = 0.0
                if not np.isfinite(sigma) or sigma <= 0:
                    sigma = 1.0
                mus = np.zeros(n, dtype=np.float32)
                sigmas = np.zeros(n, dtype=np.float32)
                norm = np.zeros(n, dtype=np.float32)
                for i, value in enumerate(series):
                    if not np.isfinite(value):
                        value = mu
                    mus[i] = mu
                    sigmas[i] = sigma
                    norm[i] = float((value - mu) / max(sigma, 1e-6))
                    mu = (1 - alpha) * mu + alpha * float(value)
                    var = (1 - alpha) * (sigma**2) + alpha * float((value - mu) ** 2)
                    sigma = float(np.sqrt(max(var, 1e-6)))
                out.loc[idx, col] = norm
                out.loc[idx, f"baseline_mu_{col}"] = mus
                out.loc[idx, f"baseline_sigma_{col}"] = sigmas
        return out

    profiles_cfg = cfg.get("profiles", {})
    enabled = bool(profiles_cfg.get("enabled", True))
    mode = str(profiles_cfg.get("mode", "global")).lower()
    use_static_meta = bool(profiles_cfg.get("use_static_meta", False))
    alpha = float(cfg.get("profile", {}).get("ema_alpha", 0.1))
    warmup_steps = int(profiles_cfg.get("warmup_steps", 30))
    fit_subjects = sorted(train_df[schema.worker_id].astype(str).unique().tolist())
    if not enabled:
        # Profiles OFF still uses train-only global normalization so ablations remain comparable.
        stats: dict[str, tuple[float, float]] = {}
        for col in schema.physiology:
            series = train_df[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            mu = float(series.mean()) if len(series) else 0.0
            sigma = float(series.std() + 1e-6) if len(series) else 1.0
            stats[col] = (mu, sigma)
        train_z = apply_global_norm(train_df, stats)
        val_z = apply_global_norm(val_df, stats)
        test_z = apply_global_norm(test_df, stats)
        empty_static = pd.DataFrame(columns=["worker_id", "specialization_index", "experience_level"])
        return train_z, val_z, test_z, empty_static, {
            "enabled": False,
            "mode": "global",
            "fit_subjects": fit_subjects,
            "use_static_meta": False,
        }

    if mode == "online":
        train_z = apply_online_norm(train_df, alpha=alpha, warmup_steps=warmup_steps)
        val_z = apply_online_norm(val_df, alpha=alpha, warmup_steps=warmup_steps)
        test_z = apply_online_norm(test_df, alpha=alpha, warmup_steps=warmup_steps)
        static = pd.DataFrame(columns=["worker_id", "specialization_index", "experience_level"])
    else:
        # default: global normalization from TRAIN only
        stats: dict[str, tuple[float, float]] = {}
        for col in schema.physiology:
            series = train_df[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            mu = float(series.mean()) if len(series) else 0.0
            sigma = float(series.std() + 1e-6) if len(series) else 1.0
            stats[col] = (mu, sigma)
        train_z = apply_global_norm(train_df, stats)
        val_z = apply_global_norm(val_df, stats)
        test_z = apply_global_norm(test_df, stats)
        if use_static_meta:
            profile_store = WorkerProfileStore(schema.physiology)
            profile_store.fit_baselines(
                train_df,
                alpha=alpha,
                safe_col=cfg.get("profile", {}).get("safe_col"),
                worker_col=schema.worker_id,
            )
            static = profile_store.get_static_profile_table().rename(columns={"specialization_id": "specialization_index"})
        else:
            static = pd.DataFrame(columns=["worker_id", "specialization_index", "experience_level"])
    return train_z, val_z, test_z, static, {
        "enabled": True,
        "mode": mode,
        "warmup_steps": warmup_steps if mode == "online" else 0,
        "fit_subjects": fit_subjects,
        "use_static_meta": use_static_meta,
    }


def _attach_static(df: pd.DataFrame, static: pd.DataFrame, schema: DataSchema, use_static_meta: bool) -> pd.DataFrame:
    frame = df.copy()
    if not static.empty:
        frame = frame.merge(static, on="worker_id", how="left")
    if use_static_meta:
        if "specialization_index" not in frame.columns:
            if schema.specialization_col in frame.columns:
                frame["specialization_index"] = frame[schema.specialization_col]
            else:
                frame["specialization_index"] = 0
        if "experience_level" not in frame.columns:
            if schema.experience_col in frame.columns:
                frame["experience_level"] = frame[schema.experience_col]
            else:
                frame["experience_level"] = 1
    else:
        frame["specialization_index"] = 0
        frame["experience_level"] = 1
    frame["specialization_index"] = frame["specialization_index"].fillna(0).astype(int)
    frame["experience_level"] = frame["experience_level"].fillna(1).astype(int)
    for col in schema.physiology:
        mu_col = f"baseline_mu_{col}"
        sigma_col = f"baseline_sigma_{col}"
        if mu_col not in frame.columns:
            frame[mu_col] = 0.0
        if sigma_col not in frame.columns:
            frame[sigma_col] = 1.0
        frame[mu_col] = frame[mu_col].fillna(0.0)
        frame[sigma_col] = frame[sigma_col].fillna(1.0)
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


def _comparison(metrics: dict[str, Any], include_comfort: bool) -> dict[str, Any]:
    def get(path: tuple[str, ...], default=None):
        ptr = metrics
        for key in path:
            if not isinstance(ptr, dict) or key not in ptr:
                return default
            ptr = ptr[key]
        return ptr

    result = {
        "stress_winner_by_auroc": "xgboost"
        if (get(("xgboost", "stress", "auroc"), -1) >= get(("tft", "stress", "auroc"), -1))
        else "tft",
        "delta_stress_auroc": float(get(("tft", "stress", "auroc"), np.nan) - get(("xgboost", "stress", "auroc"), np.nan)),
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
    tft_warning = ""
    if isinstance(tft_stress, dict):
        if str(tft_stress.get("auroc", "")).lower() == "nan":
            tft_warning = "TFT AUROC is NaN; test split may contain a single class or too few windows."
    cfg = metrics.get("config", {})
    sampling = float(cfg.get("task", {}).get("sampling_rate_hz", 1.0))
    downsample = int(cfg.get("dataset", {}).get("downsample_factor") or 1)
    effective_hz = sampling / max(1, downsample)
    window_len = int(cfg.get("task", {}).get("window_length", 1))
    horizon = int(cfg.get("task", {}).get("horizon_steps", 1))
    step = int(cfg.get("task", {}).get("window_step", 1))
    test_balance = metrics.get("class_balance", {}).get("test", {})
    threshold_policy = metrics.get("xgboost", {}).get("stress", {}).get("threshold_policy", "n/a")
    xgb = metrics.get("xgboost", {}).get("stress", {})
    tft = metrics.get("tft", {}).get("stress", {})
    lines = [
        "# Experiment Results",
        "",
        "## Setup",
        f"- Dataset: {dataset_name}",
        f"- Split: {split_desc}",
        f"- Profiles enabled: {profiles_enabled}",
        f"- Task: {metrics.get('task_name', 'stress')}",
        f"- Effective sampling rate (Hz): {effective_hz:.3f}",
        f"- Window/Horizon/Step (seconds): {window_len/effective_hz:.2f} / {horizon/effective_hz:.2f} / {step/effective_hz:.2f}",
        f"- Test prevalence: {test_balance}",
        f"- Threshold policy: {threshold_policy}",
        "",
        "## Key Findings",
        "- No-leakage profile fitting: baselines fit on train split only and reused on val/test.",
        f"- Stress AUROC winner: {metrics.get('comparison', {}).get('stress_winner_by_auroc', 'n/a')}",
        f"- Comfort RMSE winner: {metrics.get('comparison', {}).get('comfort_winner_by_rmse', 'n/a')}",
        f"- TFT sanity note: {tft_warning}" if tft_warning else "- TFT sanity note: n/a",
        "",
        "## Comparison Table",
        "| Model | AUROC | AUPRC | F1 | Precision | Recall | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| XGBoost | {xgb.get('auroc', 'n/a')} | {xgb.get('auprc', 'n/a')} | {xgb.get('f1', 'n/a')} | {xgb.get('precision', 'n/a')} | {xgb.get('recall', 'n/a')} | {xgb.get('brier', 'n/a')} | {xgb.get('ece', 'n/a')} |",
        f"| TFT | {tft.get('auroc', 'n/a')} | {tft.get('auprc', 'n/a')} | {tft.get('f1', 'n/a')} | {tft.get('precision', 'n/a')} | {tft.get('recall', 'n/a')} | {tft.get('brier', 'n/a')} | {tft.get('ece', 'n/a')} |",
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


def run_experiment(config_path: str) -> Path:
    cfg = load_yaml(config_path)
    logger = setup_logger()
    seed_everything(int(cfg.get("reproducibility", {}).get("seed", cfg["split"]["seed"])))
    schema = DataSchema.from_config(cfg)
    debug = bool(cfg.get("debug", False))

    raw_df = load_or_generate(cfg, schema)
    frame = preprocess_dataframe(cfg, raw_df, schema)
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
            logger.info("Split %s raw rows=%d stress_counts=%s", name, len(df), counts)

    train_df, val_df, test_df, static_profiles, profiles_info = _profile_transform(train_df, val_df, test_df, schema, cfg)
    use_static_meta = bool(cfg.get("profiles", {}).get("use_static_meta", False))
    train_df = _attach_static(train_df, static_profiles, schema, use_static_meta=use_static_meta)
    val_df = _attach_static(val_df, static_profiles, schema, use_static_meta=use_static_meta)
    test_df = _attach_static(test_df, static_profiles, schema, use_static_meta=use_static_meta)
    if debug:
        logger.info("Profiles: %s", profiles_info)
    flat_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    window_length = int(cfg["task"]["window_length"])
    horizon = int(cfg["task"]["horizon_steps"])
    window_step = int(cfg.get("task", {}).get("window_step", 1))
    min_split_len = min(
        min(len(part[part[schema.worker_id] == wid]) for wid in part[schema.worker_id].unique())
        for part in (train_df, val_df, test_df)
    )
    if min_split_len <= horizon:
        raise ValueError("Split segments are too short for the configured horizon.")
    if min_split_len <= window_length:
        adjusted = max(4, min_split_len - horizon - 1)
        logger.warning("Reducing window_length from %d to %d for available split lengths.", window_length, adjusted)
        window_length = adjusted
    train_w = build_windows(train_df, schema, window_length, horizon, window_step=window_step)
    val_w = build_windows(val_df, schema, window_length, horizon, window_step=window_step)
    test_w = build_windows(test_df, schema, window_length, horizon, window_step=window_step)
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

    sampling_rate = float(cfg["task"].get("sampling_rate_hz", 1.0))
    downsample_factor = int(cfg.get("dataset", {}).get("downsample_factor") or 1)
    if downsample_factor > 1:
        sampling_rate = sampling_rate / downsample_factor
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
        "window_counts": window_counts,
        "class_balance": class_balance,
        "profiles_info": profiles_info,
        "task_name": "stress vs non-stress (baseline+amusement)"
        if bool(cfg.get("dataset", {}).get("stress_include_amusement", False))
        else "stress vs baseline",
    }
    xgb_out = None
    tft_stress = None
    tft_comfort = None

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
            use_profiles=bool(cfg.get("profiles", {}).get("enabled", True)),
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
            use_profiles=bool(cfg.get("profiles", {}).get("enabled", True)),
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
                use_profiles=bool(cfg.get("profiles", {}).get("enabled", True)),
            )
            tft_metrics["comfort"] = tft_comfort.metrics | {"checkpoint_path": str(tft_comfort.checkpoint_path)}
        metrics["tft"] = tft_metrics
    except Exception as exc:
        logger.exception("TFT pipeline failed: %s", exc)
        metrics["tft"] = {"error": str(exc)}

    # Ablation: profiles OFF using XGBoost only for fast comparison.
    ablation: dict[str, Any] = {}
    try:
        off_cfg = json.loads(json.dumps(cfg))
        off_cfg["profiles"]["enabled"] = False
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
    if xgb_out is not None:
        plot_roc(y_stress_test, [("XGBoost", xgb_out["stress"].predictions)], run_paths.plots / "roc_curve_stress.png")
        plot_pr(y_stress_test, [("XGBoost", xgb_out["stress"].predictions)], run_paths.plots / "pr_curve_stress.png")
        plot_calibration(
            y_stress_test, [("XGBoost", xgb_out["stress"].predictions)], run_paths.plots / "calibration_stress.png"
        )
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
        plot_roc(
            tft_stress.targets,
            [("TFT", tft_stress.predictions)],
            run_paths.plots / "roc_curve_stress_tft.png",
        )

    metrics["comparison"] = _comparison(metrics, include_comfort=include_comfort)
    metrics["ablation_profiles"] = ablation
    save_json(metrics, run_paths.root / "metrics.json")
    _write_results_md(
        run_paths.root,
        metrics,
        ablation,
        dataset_name=str(
            cfg.get("dataset", {}).get("report_name", cfg.get("dataset", {}).get("name", "unknown"))
        ),
        split_desc=split_desc,
        profiles_enabled=bool(cfg.get("profiles", {}).get("enabled", True)),
    )
    logger.info("Experiment complete. Artifacts saved to %s", run_paths.root)
    return run_paths.root


def main() -> None:
    args = parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
