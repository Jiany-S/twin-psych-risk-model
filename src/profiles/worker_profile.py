"""Worker profile definitions and utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


BASELINE_FEATURES = ("hr", "hrv_rmssd", "gsr")


@dataclass
class BaselineStats:
    mu: float
    sigma: float


@dataclass
class WorkerProfile:
    worker_id: str
    experience_level: int
    specialization_id: int
    baselines: Dict[str, BaselineStats]

    def to_vector(self, feature_order: Sequence[str]) -> list[float]:
        vector: list[float] = [float(self.experience_level), float(self.specialization_id)]
        for feature in feature_order:
            stats = self.baselines.get(feature)
            if stats is None:
                vector.extend([0.0, 1.0])
            else:
                vector.extend([float(stats.mu), float(stats.sigma)])
        return vector


class WorkerProfileStore:
    """Store per-worker baseline statistics and metadata."""

    def __init__(
        self,
        profiles: Dict[str, WorkerProfile],
        global_baselines: Mapping[str, BaselineStats],
        feature_order: Sequence[str] = BASELINE_FEATURES,
    ) -> None:
        self._profiles = profiles
        self._global_baselines = dict(global_baselines)
        self._feature_order = tuple(feature_order)

    @property
    def feature_order(self) -> tuple[str, ...]:
        return self._feature_order

    def get(self, worker_id: str) -> WorkerProfile | None:
        return self._profiles.get(worker_id)

    def get_baseline(self, worker_id: str, feature: str) -> BaselineStats:
        profile = self._profiles.get(worker_id)
        if profile and feature in profile.baselines:
            return profile.baselines[feature]
        return self._global_baselines.get(feature, BaselineStats(mu=0.0, sigma=1.0))

    def get_profile_vector(self, worker_id: str) -> list[float]:
        profile = self._profiles.get(worker_id)
        if profile is None:
            fallback_profile = WorkerProfile(
                worker_id=worker_id,
                experience_level=0,
                specialization_id=0,
                baselines={f: self.get_baseline(worker_id, f) for f in self._feature_order},
            )
            return fallback_profile.to_vector(self._feature_order)
        return profile.to_vector(self._feature_order)

    def global_baseline(self, feature: str) -> BaselineStats:
        return self._global_baselines.get(feature, BaselineStats(mu=0.0, sigma=1.0))

    def as_static_frame(self) -> pd.DataFrame:
        """Return a dataframe containing static covariates for each worker."""
        rows: list[dict[str, float]] = []
        for worker_id, profile in self._profiles.items():
            vector = self.get_profile_vector(worker_id)
            columns = ["experience_level", "specialization_id"]
            for feature in self._feature_order:
                columns.extend([f"baseline_mu_{feature}", f"baseline_sigma_{feature}"])
            row = dict(zip(columns, vector, strict=False))
            row["worker_id"] = worker_id
            rows.append(row)

        if not rows:
            return pd.DataFrame(
                columns=["worker_id", "experience_level", "specialization_id"]
                + [
                    col
                    for feature in self._feature_order
                    for col in (f"baseline_mu_{feature}", f"baseline_sigma_{feature}")
                ]
            )

        df = pd.DataFrame(rows)
        return df

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        feature_cols: Sequence[str] = BASELINE_FEATURES,
        worker_col: str = "worker_id",
        experience_col: str = "experience_level_bin",
        specialization_col: str = "specialization_id",
        safe_flag_column: str | None = None,
        alpha: float = 0.1,
        min_sigma: float = 1e-3,
    ) -> "WorkerProfileStore":
        feature_order = tuple(feature_cols)
        safe_mask = pd.Series(True, index=df.index)
        if safe_flag_column and safe_flag_column in df.columns:
            safe_mask = df[safe_flag_column].fillna(1).astype(bool)

        profiles: dict[str, WorkerProfile] = {}
        global_baselines: dict[str, BaselineStats] = {
            feature: BaselineStats(mu=0.0, sigma=1.0) for feature in feature_order
        }

        global_values: dict[str, list[float]] = {feature: [] for feature in feature_order}
        for feature in feature_order:
            values = df.loc[safe_mask, feature].dropna().astype(float).tolist()
            if values:
                global_values[feature] = values
                mu = float(np.mean(values))
                sigma = float(np.std(values) + min_sigma)
                global_baselines[feature] = BaselineStats(mu=mu, sigma=max(sigma, min_sigma))

        for worker_id, worker_df in df.groupby(worker_col):
            mask = safe_mask.loc[worker_df.index]
            worker_safe = worker_df[mask.values]
            if worker_safe.empty:
                worker_safe = worker_df
            baselines = cls._compute_worker_baselines(worker_safe, feature_order, alpha, min_sigma)
            experience = int(worker_df[experience_col].iloc[0]) if experience_col in worker_df.columns else 0
            specialization = (
                int(worker_df[specialization_col].iloc[0]) if specialization_col in worker_df.columns else 0
            )
            profiles[worker_id] = WorkerProfile(
                worker_id=str(worker_id),
                experience_level=experience,
                specialization_id=specialization,
                baselines=baselines,
            )

        return cls(profiles=profiles, global_baselines=global_baselines, feature_order=feature_order)

    @staticmethod
    def _compute_worker_baselines(
        df: pd.DataFrame,
        feature_order: Sequence[str],
        alpha: float,
        min_sigma: float,
    ) -> Dict[str, BaselineStats]:
        baselines: dict[str, BaselineStats] = {}
        for feature in feature_order:
            series = df[feature].dropna().astype(float)
            if series.empty:
                baselines[feature] = BaselineStats(mu=0.0, sigma=1.0)
                continue

            mu = series.iloc[0]
            second = mu**2
            for value in series.iloc[1:]:
                mu = (1 - alpha) * mu + alpha * value
                second = (1 - alpha) * second + alpha * (value**2)
            variance = max(second - mu**2, min_sigma**2)
            sigma = float(np.sqrt(variance))
            baselines[feature] = BaselineStats(mu=float(mu), sigma=sigma)
        return baselines
