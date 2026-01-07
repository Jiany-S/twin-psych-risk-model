"""End-to-end preprocessing pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .features import ensure_robot_context, ensure_worker_metadata, finalize_feature_table
from .load_wesad import load_wesad_like_dataset
from .schema import DataSchema
from ..profiles.worker_profile import WorkerProfileStore
from ..profiles.normalization import normalize_physiology
from ..utils.logging import configure_logging


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def preprocess(config_path: str) -> Path:
    config = load_config(config_path)
    logger = configure_logging("preprocess")

    paths_cfg = config.get("paths", {})
    raw_dir = paths_cfg.get("raw_data_dir", "data/raw")
    processed_file = Path(paths_cfg.get("processed_train_file", "data/processed/train.csv"))

    schema = DataSchema.from_config(config)
    logger.info("Loading raw data from %s", raw_dir)
    df = load_wesad_like_dataset(raw_dir, schema=schema)
    df = ensure_robot_context(df)
    df = ensure_worker_metadata(df)

    profile_cfg = config.get("profile", {})
    store = WorkerProfileStore.from_dataframe(
        df,
        feature_cols=schema.observed_covariates,
        worker_col=schema.worker_id,
        experience_col="experience_level_bin",
        specialization_col="specialization_id",
        safe_flag_column=schema.safe_flag,
        alpha=float(profile_cfg.get("ema_alpha", 0.1)),
        min_sigma=float(profile_cfg.get("min_sigma", 1e-3)),
    )
    logger.info("Computed worker profiles for %d workers.", len(store.as_static_frame()))

    df = normalize_physiology(df, store, schema.observed_covariates, worker_col=schema.worker_id)
    df = finalize_feature_table(df, schema)

    processed_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_file, index=False)
    logger.info("Saved processed dataset to %s (%d rows).", processed_file, len(df))
    return processed_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw WESAD-like data.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/default.yaml",
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess(args.config)


if __name__ == "__main__":
    main()
