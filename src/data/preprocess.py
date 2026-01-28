"""Preprocessing and normalization pipeline."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .schema import DataSchema
from ..profiles.worker_profile import WorkerProfileStore


def _ensure_columns(df: pd.DataFrame, schema: DataSchema) -> pd.DataFrame:
    frame = df.copy()
    if schema.worker_id not in frame.columns:
        raise ValueError(f"Missing required column '{schema.worker_id}'.")
    if schema.time_idx not in frame.columns:
        frame = frame.sort_values(schema.worker_id).reset_index(drop=True)
        frame[schema.time_idx] = frame.groupby(schema.worker_id).cumcount().astype(int)
    for col in schema.required_columns():
        if col not in frame.columns:
            raise ValueError(f"Missing required column '{col}'.")
    if schema.hazard_zone not in frame.columns:
        frame[schema.hazard_zone] = 0
    if schema.task_phase not in frame.columns:
        frame[schema.task_phase] = "default"
    return frame


def preprocess_dataframe(cfg: Mapping[str, object], df: pd.DataFrame, schema: DataSchema) -> tuple[pd.DataFrame, WorkerProfileStore]:
    frame = _ensure_columns(df, schema)
    frame[schema.worker_id] = frame[schema.worker_id].astype(str)
    frame[schema.time_idx] = frame[schema.time_idx].astype(int)
    frame[schema.target] = frame[schema.target].astype(float).clip(0.0, 1.0)

    profile_cfg = cfg.get("profile", {})
    store = WorkerProfileStore(schema.physiology)
    store.fit_baselines(
        frame,
        alpha=float(profile_cfg.get("ema_alpha", 0.1)),
        safe_col=profile_cfg.get("safe_col"),
        worker_col=schema.worker_id,
    )
    frame = store.transform_zscore(frame, worker_col=schema.worker_id)

    for col in schema.robot_context:
        frame[col] = frame[col].astype(float)
    for col in schema.physiology:
        frame[col] = frame[col].astype(float)
    frame[schema.hazard_zone] = frame[schema.hazard_zone].fillna(0).astype(int)
    frame[schema.task_phase] = frame[schema.task_phase].fillna("default").astype(str)

    frame = frame.sort_values([schema.worker_id, schema.time_idx]).reset_index(drop=True)
    return frame, store
