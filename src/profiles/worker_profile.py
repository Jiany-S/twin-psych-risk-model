"""Worker profile store and personalization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass
class BaselineStats:
    mu: float
    sigma: float


class WorkerProfileStore:
    """Maintains per-worker baseline stats with EMA updates."""

    def __init__(self, physiology_cols: Sequence[str]) -> None:
        self.physiology_cols = tuple(physiology_cols)
        self._profiles: dict[str, dict[str, BaselineStats]] = {}
        self._meta: dict[str, dict[str, int]] = {}
        self._global: dict[str, BaselineStats] = {
            col: BaselineStats(mu=0.0, sigma=1.0) for col in self.physiology_cols
        }

    def fit_baselines(
        self,
        df: pd.DataFrame,
        alpha: float = 0.1,
        safe_col: str | None = None,
        worker_col: str = "worker_id",
    ) -> None:
        mask = pd.Series(True, index=df.index)
        if safe_col and safe_col in df.columns:
            mask = df[safe_col].fillna(1).astype(bool)

        for col in self.physiology_cols:
            values = df.loc[mask, col].dropna().astype(float)
            if not values.empty:
                mu = float(values.mean())
                sigma = float(values.std() + 1e-6)
                self._global[col] = BaselineStats(mu=mu, sigma=sigma)

        for worker_id, worker_df in df.groupby(worker_col):
            worker_mask = mask.loc[worker_df.index]
            safe_df = worker_df[worker_mask.values]
            if safe_df.empty:
                safe_df = worker_df
            self._profiles[str(worker_id)] = self._ema_stats(safe_df, alpha)
            self._meta[str(worker_id)] = self._derive_meta(worker_df)

    def _ema_stats(self, df: pd.DataFrame, alpha: float) -> dict[str, BaselineStats]:
        stats: dict[str, BaselineStats] = {}
        for col in self.physiology_cols:
            series = df[col].dropna().astype(float)
            if series.empty:
                stats[col] = self._global[col]
                continue
            mu = float(series.iloc[0])
            second = mu**2
            for value in series.iloc[1:]:
                mu = (1 - alpha) * mu + alpha * float(value)
                second = (1 - alpha) * second + alpha * float(value) ** 2
            variance = max(second - mu**2, 1e-6)
            stats[col] = BaselineStats(mu=mu, sigma=float(np.sqrt(variance)))
        return stats

    def _derive_meta(self, df: pd.DataFrame) -> dict[str, int]:
        if "specialization_id" in df.columns:
            specialization_id = int(df["specialization_id"].iloc[0])
        else:
            specialization_id = abs(hash(str(df["worker_id"].iloc[0]))) % 5
        if "experience_level" in df.columns:
            exp = int(df["experience_level"].iloc[0])
        else:
            exp = 1 + abs(hash(str(df["worker_id"].iloc[0]))) % 5
        experience_level = min(max(exp, 1), 5)
        return {"specialization_id": specialization_id, "experience_level": experience_level}

    def transform_zscore(self, df: pd.DataFrame, worker_col: str = "worker_id") -> pd.DataFrame:
        frame = df.copy()
        for col in self.physiology_cols:
            mu_col = f"baseline_mu_{col}"
            sigma_col = f"baseline_sigma_{col}"
            frame[mu_col] = frame[worker_col].astype(str).map(
                lambda w: self._profiles.get(w, {}).get(col, self._global[col]).mu
            )
            frame[sigma_col] = frame[worker_col].astype(str).map(
                lambda w: self._profiles.get(w, {}).get(col, self._global[col]).sigma
            )
            frame[sigma_col] = frame[sigma_col].replace(0, 1e-6)
            frame[col] = (frame[col] - frame[mu_col]) / frame[sigma_col]
        return frame

    def get_static_profile_table(self) -> pd.DataFrame:
        rows = []
        for worker_id, stats in self._profiles.items():
            meta = self._meta.get(worker_id, {"specialization_id": 0, "experience_level": 1})
            row = {
                "worker_id": worker_id,
                "specialization_id": meta["specialization_id"],
                "experience_level": meta["experience_level"],
            }
            for col in self.physiology_cols:
                row[f"baseline_mu_{col}"] = stats[col].mu
                row[f"baseline_sigma_{col}"] = stats[col].sigma
            rows.append(row)
        return pd.DataFrame(rows)
