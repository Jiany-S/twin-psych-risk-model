"""Loader for MultiPhysio-HRC precomputed feature tables."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .schema import DataSchema


QUESTIONNAIRE_INTERPRETATION = {
    "stress": "self-reported state anxiety",
    "cognitive_load": "perceived workload",
    "comfort": "affective valence",
    "arousal": "affective activation",
    "dominance": "affective dominance",
}


def _normalize_class_label(text: Any) -> str:
    value = str(text).strip().lower()
    value = value.replace("vr-job simulator", "vr-job-sim")
    value = value.replace("n-back", "n-back")
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return re.sub(r"-(\d+)$", "", value)


def _task_key(class_norm: str, repetition: int) -> str:
    if class_norm in {"cobot-task", "manual-task"}:
        return f"{class_norm}-{int(repetition)}"
    return f"{class_norm}_0"


def _read_participants_overview(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, encoding="cp1252")
    table["ID"] = table["ID"].astype(str).str.strip()
    return table


def _build_task_order(overview: pd.DataFrame) -> dict[str, int]:
    return {col.strip().lower(): idx for idx, col in enumerate(overview.columns[1:])}


def _target_block(targets_cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    block = targets_cfg.get(name, {})
    return block if isinstance(block, Mapping) else {}


def _range_tuple(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    lo = float(value[0])
    hi = float(value[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"Invalid target source_range {value!r}; expected [min, max] with max > min.")
    return lo, hi


def _normalize_questionnaire(series: pd.Series, source_range: tuple[float, float] | None) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if source_range is None:
        non_na = values.dropna()
        if non_na.empty:
            raise ValueError(f"Cannot normalize target column {series.name!r}; all values are missing.")
        lo = float(non_na.min())
        hi = float(non_na.max())
        if hi <= lo:
            raise ValueError(f"Cannot normalize target column {series.name!r}; observed range is degenerate.")
    else:
        lo, hi = source_range
    return ((values - lo) / (hi - lo)).clip(0.0, 1.0)


def _resolve_feature_mapping(cfg: Mapping[str, Any], schema: DataSchema, bio_columns: set[str]) -> dict[str, str]:
    legacy = {
        "ecg_col": "hrv_mean_nn",
        "eda_col": "eda_mean",
        "emg_col": "emg_rmse",
        "resp_col": "rrv_mean_bb",
    }
    mapping: dict[str, str] = {}
    configured = cfg.get("feature_columns", {})
    if isinstance(configured, Mapping):
        mapping.update({str(k): str(v) for k, v in configured.items()})
    for legacy_key, canonical in legacy.items():
        if legacy_key in cfg and canonical not in mapping:
            mapping[canonical] = str(cfg[legacy_key])

    defaults = {
        "hrv_mean_nn": "HRV_MeanNN",
        "eda_mean": "EDA_mean",
        "emg_rmse": "EMG_RMSE",
        "rrv_mean_bb": "RRV_MeanBB",
    }
    for internal in schema.physiology:
        mapping.setdefault(internal, defaults.get(internal, internal))

    invalid_temp = mapping.get("temp")
    if invalid_temp and invalid_temp.upper().startswith("EMG"):
        raise ValueError("Invalid MultiPhysio modality mapping: EMG cannot be mapped to temperature.")

    missing = {src for internal, src in mapping.items() if internal in schema.physiology and src not in bio_columns}
    if missing:
        raise ValueError(
            "MultiPhysio bio_features_60s.csv missing configured physiological source columns: "
            f"{sorted(missing)}"
        )
    return {internal: source for internal, source in mapping.items() if internal in schema.physiology}


def _add_targets(
    merged: pd.DataFrame,
    labels: pd.DataFrame,
    targets_cfg: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    source_cols: set[str] = set()
    target_names = [
        name
        for name, block in targets_cfg.items()
        if name not in {"multi_head", "primary"} and isinstance(block, Mapping) and block.get("source_col")
    ]
    if not target_names:
        raise ValueError("MultiPhysio requires explicit targets.*.source_col entries; no target columns were configured.")

    for name in target_names:
        block = _target_block(targets_cfg, name)
        source_col = str(block.get("source_col", ""))
        if source_col not in labels.columns:
            raise ValueError(f"Configured MultiPhysio target {name!r} source_col {source_col!r} is missing from labels.csv.")
        source_cols.add(source_col)

    labels_subset_cols = ["ID", "class_norm", "rep_norm", *sorted(source_cols)]
    if "Experience" in labels.columns:
        labels_subset_cols.append("Experience")
    labels_subset = labels[labels_subset_cols].drop_duplicates(subset=["ID", "class_norm", "rep_norm"], keep="last")
    merged = merged.merge(labels_subset, on=["ID", "class_norm", "rep_norm"], how="inner")
    if merged.empty:
        raise ValueError("No rows left after merging MultiPhysio bio features with configured label columns.")

    for name in target_names:
        block = _target_block(targets_cfg, name)
        source_col = str(block["source_col"])
        task_type = str(block.get("task_type", "regression")).lower()
        threshold = block.get("threshold")
        source_range = _range_tuple(block.get("source_range"))
        raw = pd.to_numeric(merged[source_col], errors="coerce")
        observed = raw.dropna()
        if observed.empty:
            raise ValueError(f"Configured MultiPhysio target {name!r} source_col {source_col!r} has no numeric values.")

        if name.endswith("_binary"):
            if threshold is None:
                raise ValueError(f"Binary target {name!r} requires an explicit threshold.")
            label_col = str(block.get("label_col", f"y_{name}"))
            merged[label_col] = (raw >= float(threshold)).astype(float)
            transformed_range = [0.0, 1.0]
        else:
            label_col = str(block.get("label_col", f"y_{name}"))
            merged[label_col] = _normalize_questionnaire(raw, source_range).astype(float)
            transformed_range = [0.0, 1.0]
            if task_type == "classification":
                if threshold is None:
                    raise ValueError(f"Classification target {name!r} requires an explicit threshold.")
                binary_col = str(block.get("binary_label_col", f"y_{name}_binary"))
                merged[binary_col] = (raw >= float(threshold)).astype(float)
                metadata[f"{name}_binary"] = {
                    "target_name": f"{name}_binary",
                    "source_questionnaire": str(block.get("source_questionnaire", source_col)),
                    "source_column": source_col,
                    "task_type": "classification",
                    "threshold": float(threshold),
                    "original_range": [float(observed.min()), float(observed.max())],
                    "normalization_range": list(source_range) if source_range else [float(observed.min()), float(observed.max())],
                    "transformed_range": [0.0, 1.0],
                    "label_col": binary_col,
                    "interpretation": QUESTIONNAIRE_INTERPRETATION.get(name, name),
                }

        metadata[name] = {
            "target_name": name,
            "source_questionnaire": str(block.get("source_questionnaire", source_col)),
            "source_column": source_col,
            "task_type": "classification" if name.endswith("_binary") else task_type,
            "threshold": None if threshold is None else float(threshold),
            "original_range": [float(observed.min()), float(observed.max())],
            "normalization_range": list(source_range) if source_range else [float(observed.min()), float(observed.max())],
            "transformed_range": transformed_range,
            "label_col": label_col,
            "interpretation": QUESTIONNAIRE_INTERPRETATION.get(name.replace("_binary", ""), name),
        }
    return merged, metadata


def load_multiphysio_dataset(
    dataset_path: str | Path,
    schema: DataSchema,
    dataset_cfg: dict[str, Any] | None = None,
    targets_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = dataset_cfg or {}
    targets = targets_cfg or {}
    root = Path(dataset_path)
    if not root.exists():
        raise FileNotFoundError(f"MultiPhysio dataset path does not exist: {root}")

    features_dir = root / "features"
    bio_path = features_dir / "bio_features_60s.csv"
    labels_path = features_dir / "labels.csv"
    overview_path = root / "participants_task_overview.csv"
    for required in (bio_path, labels_path, overview_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing required file: {required}")

    repetition_offset = int(cfg.get("repetition_offset", 1))
    keep_classes = {str(c).strip().lower() for c in cfg.get("classes", [])}
    min_rows_per_worker = int(cfg.get("min_rows_per_worker", 20))

    bio_header = pd.read_csv(bio_path, nrows=0)
    feature_mapping = _resolve_feature_mapping(cfg, schema, set(bio_header.columns))
    bio_cols = ["ID", "Class", "Repetition", "Window", *feature_mapping.values()]
    bio = pd.read_csv(bio_path, usecols=list(dict.fromkeys(bio_cols)))
    labels = pd.read_csv(labels_path)

    for frame in (bio, labels):
        frame["ID"] = frame["ID"].astype(str).str.strip()
        frame["class_norm"] = frame["Class"].map(_normalize_class_label)
        frame["rep_norm"] = pd.to_numeric(frame["Repetition"], errors="coerce").fillna(0).astype(int) + repetition_offset

    if keep_classes:
        bio = bio[bio["class_norm"].isin(keep_classes)].copy()
        labels = labels[labels["class_norm"].isin(keep_classes)].copy()

    merged = bio
    merged, target_metadata = _add_targets(merged, labels, targets)

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

    for internal, source in feature_mapping.items():
        merged[internal] = pd.to_numeric(merged[source], errors="coerce").replace([np.inf, -np.inf], np.nan)

    merged[schema.hazard_zone] = merged["class_norm"].isin({"cobot-task", "manual-task"}).astype(int)
    merged[schema.task_phase] = merged["class_norm"].astype(str)
    if "Experience" in merged.columns:
        merged[schema.experience_col] = pd.to_numeric(merged["Experience"], errors="coerce").fillna(1).astype(int)
    else:
        merged[schema.experience_col] = 1
    # MultiPhysio labels include experience, but no worker role/specialization metadata.
    merged[schema.specialization_col] = -1

    keep_cols = [
        schema.worker_id,
        schema.timestamp,
        schema.time_idx,
        schema.protocol_label,
        *schema.configured_target_columns(),
        *list(schema.physiology),
        schema.hazard_zone,
        schema.task_phase,
        schema.specialization_col,
        schema.experience_col,
    ]
    for context_col in schema.robot_context:
        merged[context_col] = 0.0
        keep_cols.append(context_col)

    keep_cols = [c for c in dict.fromkeys(keep_cols) if c in merged.columns]
    out = merged[keep_cols].copy()
    out = out.dropna(subset=list(schema.physiology) + [schema.primary_target]).reset_index(drop=True)
    if min_rows_per_worker > 1:
        worker_sizes = out.groupby(schema.worker_id, observed=True).size()
        keep_workers = worker_sizes[worker_sizes >= min_rows_per_worker].index.astype(str)
        out = out[out[schema.worker_id].astype(str).isin(keep_workers)].reset_index(drop=True)
    if out.empty:
        raise ValueError("No rows left after dropping NaN configured physiology/primary target columns.")

    out.attrs["target_metadata"] = target_metadata
    out.attrs["feature_metadata"] = {
        "feature_kind": "precomputed_60s",
        "feature_mapping": feature_mapping,
        "source_file": str(bio_path),
    }
    return out
