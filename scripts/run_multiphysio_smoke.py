"""Run a small MultiPhysio experiment and assert both models complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.run_experiment import run_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run smoke test on MultiPhysio dataset.")
    parser.add_argument("--config", type=str, default="src/config/multiphysio_smoke.yaml")
    return parser.parse_args()


def _require_model_ok(metrics: dict, model_key: str) -> None:
    block = metrics.get(model_key, {})
    if not isinstance(block, dict):
        raise RuntimeError(f"{model_key} metrics block is missing or invalid.")
    if "error" in block:
        raise RuntimeError(f"{model_key} failed: {block['error']}")


def main() -> None:
    args = parse_args()
    run_dir = run_experiment(args.config)
    metrics_path = Path(run_dir) / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    _require_model_ok(metrics, "xgboost")
    _require_model_ok(metrics, "tft")

    test_balance = metrics.get("class_balance", {}).get("test", {})
    has_neg = "0.0" in test_balance
    has_pos = "1.0" in test_balance
    if not (has_neg and has_pos):
        raise RuntimeError(f"Smoke run completed but test split is single-class: {test_balance}")

    print(f"Smoke run OK: {run_dir}")
    print(f"Test class balance: {test_balance}")
    print(f"XGBoost stress AUROC: {metrics.get('xgboost', {}).get('stress', {}).get('auroc')}")
    print(f"TFT stress AUROC: {metrics.get('tft', {}).get('stress', {}).get('auroc')}")


if __name__ == "__main__":
    main()
