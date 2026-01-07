"""Feature engineering helpers for the preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .schema import DataSchema
from ..profiles.worker_profile import WorkerProfileStore


ROBOT_CONTEXT_COLUMNS = ("distance_to_robot", "robot_speed", "hazard_zone")


def _hash_bucket(value: str, modulo: int) -> int:
    return abs(hash(str(value))) % modulo


def ensure_robot_context(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee that robot context columns exist."""
    frame = df.copy()
    for column in ROBOT_CONTEXT_COLUMNS:
        if column not in frame.columns:
            if column == "hazard_zone":
                frame[column] = 0
            else:
                frame[column] = 0.0
    frame["hazard_zone"] = frame["hazard_zone"].astype(int)
    return frame


def ensure_worker_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Populate worker-level metadata if missing."""
    frame = df.copy()
    if "experience_level" not in frame.columns:
        frame["experience_level"] = frame["worker_id"].apply(lambda w: 1 + _hash_bucket(w, 3))
    if "specialization_id" not in frame.columns:
        frame["specialization_id"] = frame["worker_id"].apply(lambda w: _hash_bucket(w, 5))

    if "experience_level" in frame.columns:
        frame["experience_level"] = frame["experience_level"].astype(int)
        frame["experience_level_bin"] = frame["experience_level"].clip(1, 5)
    else:
        frame["experience_level_bin"] = 1

    if "specialization_id" in frame.columns:
        frame["specialization_id"] = frame["specialization_id"].astype(int)

    return frame


def attach_worker_profiles(
    df: pd.DataFrame,
    store: WorkerProfileStore,
) -> pd.DataFrame:
    """Append static worker-level baseline statistics to every row."""
    frame = df.copy()
    profile_df = store.as_static_frame()
    frame = frame.merge(profile_df, on="worker_id", how="left")
    return frame


def finalize_feature_table(df: pd.DataFrame, schema: DataSchema) -> pd.DataFrame:
    """Sort by worker/time and ensure dtype consistency."""
    frame = df.copy()
    frame[schema.worker_id] = frame[schema.worker_id].astype(str)
    frame[schema.time_idx] = frame[schema.time_idx].astype(int)
    frame[schema.target] = frame[schema.target].astype(float)
    for column in schema.known_covariates + schema.observed_covariates:
        if column in frame.columns:
            frame[column] = frame[column].astype(float)
    frame = frame.sort_values([schema.worker_id, schema.time_idx]).reset_index(drop=True)
    return frame
