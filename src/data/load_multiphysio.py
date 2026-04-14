"""Loader for MultiPhysio-HRC precomputed feature tables."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import DataSchema


def _normalize_class_label(text: Any) -> str:
    value = str(text).strip().lower()
    value = value.replace("vr-job simulator", "vr-job-sim")
    value = value.replace("n-back", "n-back")
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    # Some files carry per-repetition class labels like "cobot-task-1".
    value = re.sub(r"-(\d+)$", "", value)
    return value


def _task_key(class_norm: str, repetition: int) -> str:
    if class_norm in {"cobot-task", "manual-task"}:
        return f"{class_norm}-{int(repetition)}"
    return f"{class_norm}_0"


def _read_participants_overview(path: Path) -> pd.DataFrame:
    # The source file contains a checkmark that can be mojibake under cp1252;
    # we treat anything different from "-" as available.
    table = pd.read_csv(path, encoding="cp1252")
    table["ID"] = table["ID"].astype(str).str.strip()
    return table


def _build_task_order(overview: pd.DataFrame) -> dict[str, int]:
    order: dict[str, int] = {}
    for idx, col in enumerate(overview.columns[1:]):
        order[col.strip().lower()] = idx
    return order


def load_multiphysio_dataset(
    dataset_path: str | Path,
    schema: DataSchema,
    dataset_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = dataset_cfg or {}
    root = Path(dataset_path)
    if not root.exists():
        raise FileNotFoundError(f"MultiPhysio dataset path does not exist: {root}")

    features_dir = root / "features"
    bio_path = features_dir / "bio_features_60s.csv"
    labels_path = features_dir / "labels.csv"
    overview_path = root / "participants_task_overview.csv"

    if not bio_path.exists():
        raise FileNotFoundError(f"Missing required file: {bio_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing required file: {labels_path}")
    if not overview_path.exists():
        raise FileNotFoundError(f"Missing required file: {overview_path}")

    ecg_col = str(cfg.get("ecg_col", "HRV_MeanNN"))
    eda_col = str(cfg.get("eda_col", "EDA_mean"))
    # MultiPhysio feature tables do not provide a skin-temperature series;
    # EMG_RMSE is used as a stable third physiological signal placeholder.
    temp_col = str(cfg.get("temp_col", "EMG_RMSE"))
    resp_col = str(cfg.get("resp_col", "RRV_MeanBB"))
    repetition_offset = int(cfg.get("repetition_offset", 1))
    stress_label_col = str(cfg.get("stress_label_col", "NASA"))
    stress_threshold = float(cfg.get("stress_threshold", 40.0))
    comfort_label_col = str(cfg.get("comfort_label_col", "Valence"))
    keep_classes = {str(c).strip().lower() for c in cfg.get("classes", [])}
    min_rows_per_worker = int(cfg.get("min_rows_per_worker", 20))

    bio_cols = ["ID", "Class", "Repetition", "Window", ecg_col, eda_col, temp_col]
    if resp_col:
        bio_cols.append(resp_col)
    bio = pd.read_csv(bio_path, usecols=[c for c in bio_cols if c])

    labels = pd.read_csv(labels_path)
    required_label_cols = {"ID", "Class", "Repetition", stress_label_col}
    missing_label = required_label_cols - set(labels.columns)
    if missing_label:
        raise ValueError(f"labels.csv missing required columns: {sorted(missing_label)}")
    if comfort_label_col and comfort_label_col not in labels.columns:
        comfort_label_col = ""

    for frame in (bio, labels):
        frame["ID"] = frame["ID"].astype(str).str.strip()
        frame["class_norm"] = frame["Class"].map(_normalize_class_label)
        frame["rep_norm"] = pd.to_numeric(frame["Repetition"], errors="coerce").fillna(0).astype(int) + repetition_offset

    if keep_classes:
        bio = bio[bio["class_norm"].isin(keep_classes)].copy()
        labels = labels[labels["class_norm"].isin(keep_classes)].copy()

    labels = labels.drop_duplicates(subset=["ID", "class_norm", "rep_norm"], keep="last")
    merged = bio.merge(
        labels[
            [
                "ID",
                "class_norm",
                "rep_norm",
                stress_label_col,
                *([comfort_label_col] if comfort_label_col else []),
                *([schema.experience_col] if schema.experience_col in labels.columns else []),
            ]
        ],
        on=["ID", "class_norm", "rep_norm"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("No rows left after merging MultiPhysio bio features with labels.")

    overview = _read_participants_overview(overview_path)
    task_order = _build_task_order(overview)
    available = overview.melt(id_vars=["ID"], var_name="task_key", value_name="available")
    available["task_key"] = available["task_key"].astype(str).str.strip().str.lower()
    available["available"] = available["available"].astype(str).str.strip().ne("-")
    available = available[available["available"]][["ID", "task_key"]].drop_duplicates()

    merged["task_key"] = merged.apply(lambda r: _task_key(str(r["class_norm"]), int(r["rep_norm"])), axis=1)
    merged["task_order"] = merged["task_key"].map(task_order).fillna(10_000).astype(int)
    merged = merged.merge(available, on=["ID", "task_key"], how="inner")
    if merged.empty:
        raise ValueError("No rows left after applying participants_task_overview availability filter.")

    merged = merged.sort_values(["ID", "task_order", "rep_norm", "Window"], kind="mergesort").reset_index(drop=True)
    merged[schema.worker_id] = merged["ID"]
    merged[schema.protocol_label] = merged["class_norm"]
    merged[schema.time_idx] = merged.groupby("ID", observed=True).cumcount().astype(int)
    merged[schema.timestamp] = merged[schema.time_idx].astype(float)

    merged[schema.stress_target] = (
        pd.to_numeric(merged[stress_label_col], errors="coerce").fillna(0.0) >= stress_threshold
    ).astype(float)
    if comfort_label_col:
        comfort = pd.to_numeric(merged[comfort_label_col], errors="coerce")
        # SAM Valence is 1..5 in this dataset; normalize to 0..1.
        merged[schema.comfort_target] = ((comfort - 1.0) / 4.0).clip(0.0, 1.0).fillna(0.5).astype(float)
    else:
        merged[schema.comfort_target] = (1.0 - merged[schema.stress_target]).astype(float)

    merged["ecg"] = pd.to_numeric(merged[ecg_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    merged["eda"] = pd.to_numeric(merged[eda_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    merged["temp"] = pd.to_numeric(merged[temp_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if resp_col in merged.columns:
        merged["resp"] = pd.to_numeric(merged[resp_col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    merged[schema.hazard_zone] = merged["class_norm"].isin({"cobot-task", "manual-task"}).astype(int)
    merged[schema.task_phase] = merged["class_norm"].astype(str)
    if schema.experience_col in merged.columns:
        merged[schema.experience_col] = pd.to_numeric(merged[schema.experience_col], errors="coerce").fillna(1).astype(int)
    else:
        merged[schema.experience_col] = 1
    merged[schema.specialization_col] = pd.Categorical(merged["class_norm"]).codes.astype(int)

    keep_cols = [
        schema.worker_id,
        schema.timestamp,
        schema.time_idx,
        schema.protocol_label,
        schema.stress_target,
        schema.comfort_target,
        "ecg",
        "eda",
        "temp",
        schema.hazard_zone,
        schema.task_phase,
        schema.specialization_col,
        schema.experience_col,
    ]
    if "resp" in merged.columns:
        keep_cols.append("resp")
    for context_col in schema.robot_context:
        merged[context_col] = 0.0
        keep_cols.append(context_col)

    out = merged[keep_cols].copy()
    out = out.dropna(subset=["ecg", "eda", "temp"]).reset_index(drop=True)
    if min_rows_per_worker > 1:
        worker_sizes = out.groupby(schema.worker_id, observed=True).size()
        keep_workers = worker_sizes[worker_sizes >= min_rows_per_worker].index.astype(str)
        out = out[out[schema.worker_id].astype(str).isin(keep_workers)].reset_index(drop=True)
    if out.empty:
        raise ValueError("No rows left after dropping NaN core physiology columns in MultiPhysio dataset.")
    return out
