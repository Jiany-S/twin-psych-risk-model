"""Data loader and dataset router."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from .load_wesad import load_wesad_dataset
from .schema import DataSchema
from .synthetic import generate_synthetic_dataset


def _load_csv_files(raw_dir: Path) -> list[pd.DataFrame]:
    return [pd.read_csv(file_path) for file_path in sorted(raw_dir.rglob("*.csv"))]


def _load_generic_csv(path: Path, schema: DataSchema) -> pd.DataFrame:
    if path.is_file():
        return pd.read_csv(path)
    frames = _load_csv_files(path)
    if not frames:
        raise FileNotFoundError(f"No CSV files found at {path}")
    df = pd.concat(frames, ignore_index=True)
    if schema.worker_id not in df.columns:
        df[schema.worker_id] = "0"
    return df


def load_or_generate(cfg: Mapping[str, object], schema: DataSchema) -> pd.DataFrame:
    dataset_cfg = cfg.get("dataset", {})
    dataset_name = str(dataset_cfg.get("name", "synthetic")).lower()
    dataset_path = dataset_cfg.get("path")
    dataset_format = str(dataset_cfg.get("format", "auto"))

    paths = cfg.get("paths", {})
    raw_dir = Path(paths.get("raw_dir", "data/raw"))
    raw_file = Path(paths.get("raw_file", "data/raw/data.csv"))
    raw_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name == "wesad":
        if not dataset_path:
            raise FileNotFoundError("dataset.path is required when dataset.name is 'wesad'.")
        subjects = dataset_cfg.get("subjects")
        subject_list = [str(s) for s in subjects] if isinstance(subjects, list) else None
        return load_wesad_dataset(dataset_path, schema=schema, data_format=dataset_format, subjects=subject_list)

    if dataset_name == "csv":
        if not dataset_path:
            if raw_file.exists():
                dataset_path = raw_file
            else:
                dataset_path = raw_dir
        return _load_generic_csv(Path(dataset_path), schema=schema)

    # synthetic default/fallback
    synth_cfg = cfg.get("synthetic", {})
    use_existing_raw = bool(synth_cfg.get("use_existing_raw", False))
    if raw_file.exists() and use_existing_raw:
        return pd.read_csv(raw_file)
    frames = _load_csv_files(raw_dir)
    if frames and use_existing_raw:
        return pd.concat(frames, ignore_index=True)
    if not synth_cfg.get("enabled_if_missing", True):
        raise FileNotFoundError(
            f"No raw CSVs found under {raw_dir}. Provide data/raw/data.csv or enable synthetic generation."
        )
    df = generate_synthetic_dataset(cfg, schema)
    df.to_csv(raw_file, index=False)
    return df
