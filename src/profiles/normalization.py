"""Normalization helpers that leverage worker profiles."""

from __future__ import annotations

import pandas as pd

from .worker_profile import WorkerProfileStore


def normalize_physiology(
    df: pd.DataFrame,
    store: WorkerProfileStore,
    feature_cols: tuple[str, ...] | list[str],
    worker_col: str = "worker_id",
) -> pd.DataFrame:
    """Z-score physiological features using per-worker baselines."""
    frame = df.copy()
    profile_frame = store.as_static_frame()
    baseline_columns = [
        col for col in profile_frame.columns if col.startswith("baseline_")
    ]
    merge_frame = profile_frame[[worker_col, *baseline_columns]]
    frame = frame.merge(merge_frame, on=worker_col, how="left")

    for feature in feature_cols:
        mu_col = f"baseline_mu_{feature}"
        sigma_col = f"baseline_sigma_{feature}"
        global_stats = store.global_baseline(feature)
        frame[mu_col] = frame[mu_col].fillna(global_stats.mu)
        frame[sigma_col] = frame[sigma_col].fillna(max(global_stats.sigma, 1e-6))
        frame[feature] = (frame[feature] - frame[mu_col]) / frame[sigma_col]

    return frame
