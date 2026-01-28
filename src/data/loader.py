"""Data loader and fallback to synthetic dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from .schema import DataSchema
from .synthetic import generate_synthetic_dataset


def _load_csv_files(raw_dir: Path) -> list[pd.DataFrame]:
    files = sorted(raw_dir.rglob("*.csv"))
    frames = []
    for file_path in files:
        frames.append(pd.read_csv(file_path))
    return frames


def load_or_generate(cfg: Mapping[str, object], schema: DataSchema) -> pd.DataFrame:
    paths = cfg.get("paths", {})
    raw_dir = Path(paths.get("raw_dir", "data/raw"))
    raw_file = Path(paths.get("raw_file", "data/raw/data.csv"))
    raw_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    if raw_file.exists():
        frames.append(pd.read_csv(raw_file))
    else:
        frames = _load_csv_files(raw_dir)

    if frames:
        df = pd.concat(frames, ignore_index=True)
        return df

    synth_cfg = cfg.get("synthetic", {})
    if not synth_cfg.get("enabled_if_missing", True):
        raise FileNotFoundError(
            f"No raw CSVs found under {raw_dir}. Provide data/raw/data.csv or enable synthetic generation."
        )

    df = generate_synthetic_dataset(cfg, schema)
    df.to_csv(raw_file, index=False)
    return df
