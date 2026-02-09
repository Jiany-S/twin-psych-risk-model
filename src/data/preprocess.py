"""Preprocessing and normalization pipeline."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from .schema import DataSchema


def _ensure_columns(df: pd.DataFrame, schema: DataSchema) -> pd.DataFrame:
    frame = df.copy()
    if schema.stress_target not in frame.columns and "target" in frame.columns:
        frame[schema.stress_target] = (pd.to_numeric(frame["target"], errors="coerce").fillna(0.0) >= 0.5).astype(float)
    if schema.comfort_target not in frame.columns and "target" in frame.columns:
        frame[schema.comfort_target] = (1.0 - pd.to_numeric(frame["target"], errors="coerce").fillna(0.0)).clip(0.0, 1.0)
    if schema.worker_id not in frame.columns:
        raise ValueError(f"Missing required column '{schema.worker_id}'.")
    if schema.timestamp not in frame.columns:
        if schema.time_idx in frame.columns:
            frame[schema.timestamp] = frame[schema.time_idx]
        else:
            frame = frame.sort_values(schema.worker_id).reset_index(drop=True)
            frame[schema.timestamp] = frame.groupby(schema.worker_id).cumcount().astype(float)
    if schema.time_idx not in frame.columns:
        frame = frame.sort_values([schema.worker_id, schema.timestamp]).reset_index(drop=True)
        frame[schema.time_idx] = frame.groupby(schema.worker_id).cumcount().astype(int)
    legacy_signal_map = {"ecg": "hr", "eda": "gsr", "temp": "hrv_rmssd"}
    for expected, legacy in legacy_signal_map.items():
        if expected not in frame.columns and legacy in frame.columns:
            frame[expected] = frame[legacy]
    for col in schema.required_columns():
        if col not in frame.columns:
            raise ValueError(f"Missing required column '{col}'.")
    if schema.hazard_zone not in frame.columns:
        frame[schema.hazard_zone] = 0
    if schema.task_phase not in frame.columns:
        frame[schema.task_phase] = "default"
    if schema.specialization_col not in frame.columns:
        frame[schema.specialization_col] = frame[schema.worker_id].astype(str).map(lambda x: abs(hash(x)) % 5)
    if schema.experience_col not in frame.columns:
        frame[schema.experience_col] = frame[schema.worker_id].astype(str).map(lambda x: 1 + abs(hash(x)) % 5)
    return frame


def preprocess_dataframe(cfg: Mapping[str, object], df: pd.DataFrame, schema: DataSchema) -> pd.DataFrame:
    """Schema/type enforcement only. Baselines are fit after split to avoid leakage."""
    frame = _ensure_columns(df, schema)

    # Keep only required/optional columns early to reduce memory footprint.
    keep_cols = (
        [schema.worker_id, schema.timestamp, schema.time_idx, schema.protocol_label, schema.stress_target, schema.comfort_target]
        + list(schema.physiology)
        + list(schema.robot_context)
        + [schema.hazard_zone, schema.task_phase, schema.specialization_col, schema.experience_col]
    )
    for col in ("resp", "accel"):
        if col in frame.columns:
            keep_cols.append(col)
    frame = frame[[c for c in keep_cols if c in frame.columns]].copy()

    frame[schema.worker_id] = frame[schema.worker_id].astype(str).astype("category")
    frame[schema.timestamp] = pd.to_numeric(frame[schema.timestamp], errors="coerce").fillna(0.0).astype("float32")
    frame[schema.time_idx] = pd.to_numeric(frame[schema.time_idx], errors="coerce").fillna(0).astype("int32")
    frame[schema.stress_target] = (
        pd.to_numeric(frame[schema.stress_target], errors="coerce").fillna(0.0).clip(0.0, 1.0).astype("float32")
    )
    frame[schema.comfort_target] = (
        pd.to_numeric(frame[schema.comfort_target], errors="coerce").fillna(0.0).clip(0.0, 1.0).astype("float32")
    )
    for col in schema.physiology:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float32")
    for col in schema.robot_context:
        if col not in frame.columns:
            frame[col] = 0.0
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float32")
    frame[schema.hazard_zone] = pd.to_numeric(frame[schema.hazard_zone], errors="coerce").fillna(0).astype("int16")
    frame[schema.task_phase] = frame[schema.task_phase].fillna("default").astype(str)
    frame[schema.protocol_label] = frame.get(schema.protocol_label, "unknown")
    frame[schema.specialization_col] = pd.to_numeric(frame[schema.specialization_col], errors="coerce").fillna(0).astype("int16")
    frame[schema.experience_col] = pd.to_numeric(frame[schema.experience_col], errors="coerce").fillna(1).astype("int16")
    frame = frame.sort_values([schema.worker_id, schema.time_idx], kind="mergesort", ignore_index=True)
    return frame
