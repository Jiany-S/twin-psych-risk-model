"""Definitions for the canonical data schema used throughout the project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class DataSchema:
    """Container object describing dataset column semantics."""

    worker_id: str
    time_idx: str
    target: str
    known_covariates: Sequence[str] = field(default_factory=tuple)
    observed_covariates: Sequence[str] = field(default_factory=tuple)
    static_categorical: Sequence[str] = field(default_factory=tuple)
    static_real: Sequence[str] = field(default_factory=tuple)
    safe_flag: str | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, object]) -> "DataSchema":
        """Instantiate schema from a parsed YAML config dictionary."""
        data_cfg = cfg.get("data")
        if data_cfg is None:
            raise ValueError("Config is missing the 'data' section.")

        def _as_tuple(key: str) -> tuple[str, ...]:
            values = data_cfg.get(key, [])
            if isinstance(values, str):
                return (values,)
            if isinstance(values, Iterable):
                return tuple(str(v) for v in values)
            raise TypeError(f"Expected iterable for config key '{key}', got {type(values)}.")

        profile_cfg = cfg.get("profile", {})
        safe_flag = None
        if isinstance(profile_cfg, Mapping):
            safe_flag_candidate = profile_cfg.get("safe_flag_column")
            if safe_flag_candidate:
                safe_flag = str(safe_flag_candidate)

        return cls(
            worker_id=str(data_cfg.get("worker_id", "worker_id")),
            time_idx=str(data_cfg.get("time_idx", "time_idx")),
            target=str(data_cfg.get("target", "target")),
            known_covariates=_as_tuple("known_covariates"),
            observed_covariates=_as_tuple("observed_covariates"),
            static_categorical=_as_tuple("static_categorical"),
            static_real=_as_tuple("static_real"),
            safe_flag=safe_flag,
        )

    def required_columns(self) -> tuple[str, ...]:
        """Return the set of mandatory columns expected in processed data."""
        return (
            self.worker_id,
            self.time_idx,
            self.target,
            *self.known_covariates,
            *self.observed_covariates,
        )
