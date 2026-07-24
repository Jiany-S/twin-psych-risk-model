"""Inspect TFT dataset sizes and reproduce predict=True vs predict=False behavior."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running as a script without installing the package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_or_generate  # noqa: E402
from src.data.preprocess import preprocess_dataframe  # noqa: E402
from src.data.schema import DataSchema  # noqa: E402
from src.data.windowing import create_subject_holdout_splits, create_time_splits  # noqa: E402
from src.models.tft_model import build_tft_datasets  # noqa: E402
from src.utils.io import load_yaml  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_yaml(args.config)
    schema = DataSchema.from_config(cfg)
    if str(cfg.get("dataset", {}).get("name", "")).lower() == "synthetic":
        from src.data.synthetic import generate_synthetic_dataset  # noqa: E402

        df = generate_synthetic_dataset(cfg, schema)
    else:
        df = load_or_generate(cfg, schema)
    df = preprocess_dataframe(cfg, df, schema)

    split_mode = str(cfg.get("split", {}).get("mode", "time")).lower()
    if split_mode == "subject_holdout":
        split_cfg = cfg.get("split", {})
        train_df, val_df, test_df, _ = create_subject_holdout_splits(
            df,
            schema,
            [str(s) for s in split_cfg.get("train_subjects", [])],
            [str(s) for s in split_cfg.get("val_subjects", [])],
            [str(s) for s in split_cfg.get("test_subjects", [])],
        )
    else:
        split_cfg = cfg.get("split", {})
        train_df, val_df, test_df, _ = create_time_splits(
            df,
            schema,
            split_cfg.get("train_ratio", 0.7),
            split_cfg.get("val_ratio", 0.15),
            split_cfg.get("test_ratio", 0.15),
        )

    for df_part in (train_df, val_df, test_df):
        df_part["worker_id"] = df_part["worker_id"].astype(str)
        df_part["specialization_index"] = df_part["specialization_index"].astype(str)
        df_part["experience_level"] = df_part["experience_level"].astype(str)
        df_part["task_phase"] = df_part["task_phase"].astype(str)

    train_ds, val_ds = build_tft_datasets(
        train_df=train_df,
        val_df=val_df,
        schema=schema,
        target_col=schema.stress_target,
        window_length=int(cfg["task"]["window_length"]),
        horizon=int(cfg["task"]["horizon_steps"]),
        use_profiles=bool(cfg.get("profiles", {}).get("enabled", True)),
        known_categoricals=["task_phase"],
    )
    test_pred_ds = train_ds.from_dataset(train_ds, test_df, predict=True, stop_randomization=True)
    test_full_ds = train_ds.from_dataset(train_ds, test_df, predict=False, stop_randomization=True)

    print(f"train_ds len={len(train_ds)} val_ds len={len(val_ds)}")
    print(f"test_ds predict=True len={len(test_pred_ds)} (one per series)")
    print(f"test_ds predict=False len={len(test_full_ds)} (all windows)")
    print("test subjects:", test_df[schema.worker_id].nunique())
    vals, counts = np.unique(test_df[schema.stress_target], return_counts=True)
    print("test class balance:", dict(zip(vals, counts)))


if __name__ == "__main__":
    main()
