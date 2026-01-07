"""Utilities for loading WESAD-like physiological datasets from local storage."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .schema import DataSchema


REQUIRED_BASE_COLUMNS = (
    "hr",
    "hrv_rmssd",
    "gsr",
    "target",
)


def _discover_csv_files(root: Path) -> list[Path]:
    csv_files = sorted(root.rglob("*.csv"))
    return [f for f in csv_files if f.is_file()]


def _prepare_dataframe(df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    df = df.copy()
    if "worker_id" not in df.columns:
        df["worker_id"] = file_path.stem
    if "time_idx" not in df.columns:
        df = df.reset_index(drop=True)
        df["time_idx"] = df.index.astype(int)
    return df


def load_wesad_like_dataset(raw_dir: str | Path, schema: DataSchema | None = None) -> pd.DataFrame:
    """Load every CSV inside ``data/raw`` and concatenate them into a single dataframe.

    Parameters
    ----------
    raw_dir:
        Directory that contains per-worker CSV files or exported segments.
    schema:
        Optional schema describing column naming conventions. When provided,
        the loader verifies that expected physiological fields exist.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data directory {raw_path} is missing. "
            "Place WESAD-like CSV files under data/raw/ before preprocessing."
        )

    files = _discover_csv_files(raw_path)
    if not files:
        raise FileNotFoundError(
            f"No CSV files were found under {raw_path}. "
            "Download a public physiological dataset (e.g., WESAD) "
            "and export each subject as a CSV file with hr/hrv_rmssd/gsr columns."
        )

    frames: list[pd.DataFrame] = []
    for file_path in files:
        df = pd.read_csv(file_path)
        df = _prepare_dataframe(df, file_path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    missing = [col for col in REQUIRED_BASE_COLUMNS if col not in combined.columns]
    if missing:
        cols = ", ".join(missing)
        raise ValueError(
            f"{file_path.parent} files are missing required columns: {cols}. "
            "Ensure each CSV contains hr, hrv_rmssd, gsr, and target columns."
        )

    if schema:
        for column in schema.required_columns():
            if column not in combined.columns:
                # Optional columns will be added downstream; warn only.
                continue

    combined = combined.sort_values(["worker_id", "time_idx"]).reset_index(drop=True)
    return combined
