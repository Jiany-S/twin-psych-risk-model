"""Quick diagnostics for MultiPhysio-HRC dataset ingestion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_or_generate  # noqa: E402
from src.data.preprocess import preprocess_dataframe  # noqa: E402
from src.data.schema import DataSchema  # noqa: E402
from src.utils.io import load_merged_yaml, load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MultiPhysio loader outputs.")
    parser.add_argument("--config", type=str, default="src/config/multiphysio_debug.yaml")
    parser.add_argument("--out", type=str, default="experiments/multiphysio_debug_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_cfg_path = ROOT / "src" / "config" / "default.yaml"
    config_path = Path(args.config)
    cfg = load_yaml(config_path) if config_path.resolve() == default_cfg_path.resolve() else load_merged_yaml(default_cfg_path, config_path)

    schema = DataSchema.from_config(cfg)
    raw_df = load_or_generate(cfg, schema)
    frame = preprocess_dataframe(cfg, raw_df, schema)

    stress_counts = frame[schema.stress_target].value_counts(dropna=False).to_dict()
    by_class = (
        frame.groupby(schema.protocol_label, observed=True)
        .agg(
            rows=(schema.protocol_label, "size"),
            workers=(schema.worker_id, "nunique"),
            stress_rate=(schema.stress_target, "mean"),
        )
        .sort_values("rows", ascending=False)
    )
    per_worker = (
        frame.groupby(schema.worker_id, observed=True)
        .agg(
            rows=(schema.worker_id, "size"),
            first_time=(schema.time_idx, "min"),
            last_time=(schema.time_idx, "max"),
            stress_rate=(schema.stress_target, "mean"),
        )
        .sort_values("rows", ascending=False)
    )

    summary = {
        "rows": int(len(frame)),
        "workers": int(frame[schema.worker_id].nunique()),
        "classes": int(frame[schema.protocol_label].nunique()),
        "stress_counts": {str(k): int(v) for k, v in stress_counts.items()},
        "time_idx_min": int(frame[schema.time_idx].min()),
        "time_idx_max": int(frame[schema.time_idx].max()),
        "missing_core_pct": {
            "ecg": float(frame["ecg"].isna().mean()) if "ecg" in frame.columns else 1.0,
            "eda": float(frame["eda"].isna().mean()) if "eda" in frame.columns else 1.0,
            "temp": float(frame["temp"].isna().mean()) if "temp" in frame.columns else 1.0,
        },
        "top_classes": by_class.head(10).reset_index().to_dict(orient="records"),
        "top_workers": per_worker.head(10).reset_index().to_dict(orient="records"),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved debug summary to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
