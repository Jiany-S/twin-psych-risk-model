"""Normalization, calibration, and optional profile feature utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


UNKNOWN_CATEGORY = -1
UNKNOWN_EXPERIENCE = 0


@dataclass(frozen=True)
class CalibrationPolicy:
    protocol_labels: tuple[str, ...] = ("baseline", "rest")
    task_phases: tuple[str, ...] = ()
    max_rows_per_subject: int | None = None
    include_median: bool = True
    robust_scale: str = "iqr"  # none | iqr | mad


def _finite_stats(values: pd.Series, include_median: bool, robust_scale: str) -> dict[str, float]:
    arr = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 1.0, "median": 0.0, "iqr": 1.0, "mad": 1.0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1) + 1e-6) if arr.size > 1 else 1.0
    median = float(np.median(arr)) if include_median else mean
    q75, q25 = np.percentile(arr, [75, 25])
    iqr = float(max(q75 - q25, 1e-6))
    mad = float(max(np.median(np.abs(arr - median)), 1e-6))
    return {"mean": mean, "std": std, "median": median, "iqr": iqr, "mad": mad}


def fit_global_normalization(train_df: pd.DataFrame, physiology_cols: Sequence[str]) -> dict[str, dict[str, float]]:
    return {
        col: _finite_stats(train_df[col], include_median=True, robust_scale="iqr")
        for col in physiology_cols
        if col in train_df.columns
    }


def apply_global_normalization(
    frame: pd.DataFrame, physiology_cols: Sequence[str], stats: dict[str, dict[str, float]]
) -> pd.DataFrame:
    out = frame.copy()
    for col in physiology_cols:
        col_stats = stats.get(col, {"mean": 0.0, "std": 1.0})
        mu = float(col_stats.get("mean", 0.0))
        sigma = max(float(col_stats.get("std", 1.0)), 1e-6)
        out[f"norm_mu_{col}"] = mu
        out[f"norm_sigma_{col}"] = sigma
        out[col] = (pd.to_numeric(out[col], errors="coerce") - mu) / sigma
    return out


def calibration_mask(
    frame: pd.DataFrame,
    protocol_col: str,
    task_phase_col: str,
    policy: CalibrationPolicy,
) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    if policy.protocol_labels and protocol_col in frame.columns:
        labels = {v.lower() for v in policy.protocol_labels}
        mask |= frame[protocol_col].astype(str).str.lower().isin(labels)
    if policy.task_phases and task_phase_col in frame.columns:
        phases = {v.lower() for v in policy.task_phases}
        mask |= frame[task_phase_col].astype(str).str.lower().isin(phases)
    return mask


def fit_calibration_table(
    frame: pd.DataFrame,
    worker_col: str,
    protocol_col: str,
    task_phase_col: str,
    physiology_cols: Sequence[str],
    policy: CalibrationPolicy,
    fallback_stats: dict[str, dict[str, float]] | None = None,
    allowed_subjects: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"subjects": {}, "policy": policy.__dict__}
    fallback = fallback_stats or {}
    cal_mask = calibration_mask(frame, protocol_col, task_phase_col, policy)
    for worker_id, worker_df in frame.groupby(worker_col, observed=True):
        worker = str(worker_id)
        if allowed_subjects is not None and worker not in allowed_subjects:
            continue
        subset = worker_df.loc[cal_mask.loc[worker_df.index]].sort_values("time_idx")
        if policy.max_rows_per_subject and policy.max_rows_per_subject > 0:
            subset = subset.head(policy.max_rows_per_subject)
        source = "calibration_segment"
        if subset.empty:
            subset = worker_df.iloc[0:0]
            source = "train_global_fallback"
        row: dict[str, Any] = {"worker_id": worker}
        for col in physiology_cols:
            stats = (
                _finite_stats(subset[col], include_median=policy.include_median, robust_scale=policy.robust_scale)
                if not subset.empty and col in subset.columns
                else fallback.get(col, {"mean": 0.0, "std": 1.0, "median": 0.0, "iqr": 1.0, "mad": 1.0})
            )
            prefix = f"calib_{col}"
            row[f"{prefix}_mean"] = float(stats["mean"])
            row[f"{prefix}_std"] = max(float(stats["std"]), 1e-6)
            if policy.include_median:
                row[f"{prefix}_median"] = float(stats["median"])
            if policy.robust_scale == "iqr":
                row[f"{prefix}_iqr"] = max(float(stats["iqr"]), 1e-6)
            elif policy.robust_scale == "mad":
                row[f"{prefix}_mad"] = max(float(stats["mad"]), 1e-6)
        rows.append(row)
        diagnostics["subjects"][worker] = {"source": source, "n_calibration_rows": int(len(subset))}
    return pd.DataFrame(rows), diagnostics


def attach_calibration_columns(frame: pd.DataFrame, table: pd.DataFrame, physiology_cols: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    if not table.empty:
        out = out.merge(table, on="worker_id", how="left")
    calib_cols = [c for c in table.columns if c != "worker_id"] if not table.empty else []
    for col in physiology_cols:
        for suffix, default in (("mean", 0.0), ("std", 1.0), ("median", 0.0), ("iqr", 1.0), ("mad", 1.0)):
            name = f"calib_{col}_{suffix}"
            if name in calib_cols or name in out.columns:
                out[name] = pd.to_numeric(out.get(name), errors="coerce").fillna(default)
    return out


def apply_calibration_normalization(frame: pd.DataFrame, physiology_cols: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in physiology_cols:
        mu_col = f"calib_{col}_mean"
        sigma_col = f"calib_{col}_std"
        if mu_col not in out.columns or sigma_col not in out.columns:
            raise ValueError(f"Calibration normalization requires columns {mu_col!r} and {sigma_col!r}.")
        mu = pd.to_numeric(out[mu_col], errors="coerce").fillna(0.0)
        sigma = pd.to_numeric(out[sigma_col], errors="coerce").fillna(1.0).clip(lower=1e-6)
        out[col] = (pd.to_numeric(out[col], errors="coerce") - mu) / sigma
    return out


def build_profile_feature_table(
    calibration_table: pd.DataFrame,
    source_df: pd.DataFrame,
    worker_col: str,
    specialization_col: str,
    experience_col: str,
    include_calibration_features: bool,
    include_role_metadata: bool,
    include_experience_metadata: bool,
) -> tuple[pd.DataFrame, list[str]]:
    rows = pd.DataFrame({"worker_id": sorted(source_df[worker_col].astype(str).unique())})
    feature_cols: list[str] = []
    if include_calibration_features and not calibration_table.empty:
        rows = rows.merge(calibration_table, on="worker_id", how="left")
        feature_cols.extend([c for c in calibration_table.columns if c != "worker_id"])
    if include_role_metadata:
        role = (
            source_df[[worker_col, specialization_col]]
            .drop_duplicates(subset=[worker_col])
            .rename(columns={worker_col: "worker_id", specialization_col: "role_metadata"})
        )
        rows = rows.merge(role, on="worker_id", how="left")
        rows["role_metadata"] = pd.to_numeric(rows["role_metadata"], errors="coerce").fillna(UNKNOWN_CATEGORY).astype(int)
        feature_cols.append("role_metadata")
    if include_experience_metadata:
        exp = (
            source_df[[worker_col, experience_col]]
            .drop_duplicates(subset=[worker_col])
            .rename(columns={worker_col: "worker_id", experience_col: "experience_metadata"})
        )
        rows = rows.merge(exp, on="worker_id", how="left")
        rows["experience_metadata"] = (
            pd.to_numeric(rows["experience_metadata"], errors="coerce").fillna(UNKNOWN_EXPERIENCE).astype(float)
        )
        feature_cols.append("experience_metadata")
    for col in feature_cols:
        default = UNKNOWN_EXPERIENCE if col == "experience_metadata" else UNKNOWN_CATEGORY if col == "role_metadata" else 0.0
        rows[col] = pd.to_numeric(rows[col], errors="coerce").fillna(default)
    return rows[["worker_id", *feature_cols]], feature_cols
