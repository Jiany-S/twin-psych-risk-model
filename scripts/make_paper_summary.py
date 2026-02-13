"""Generate a paper-ready summary from a run's metrics.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a paper-ready summary from metrics.json.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to experiments/runs/<timestamp>")
    return parser.parse_args()


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json at {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    cfg = metrics.get("config", {})
    dataset = cfg.get("dataset", {}).get("report_name", cfg.get("dataset", {}).get("name", "unknown"))
    split = metrics.get("config", {}).get("split", {})
    split_desc = f"train={split.get('train_subjects', 'n/a')}, val={split.get('val_subjects', 'n/a')}, test={split.get('test_subjects', 'n/a')}"
    window_counts = metrics.get("window_counts", {})

    xgb = metrics.get("xgboost", {}).get("stress", {})
    tft = metrics.get("tft", {}).get("stress", {})
    ablation = metrics.get("ablation_profiles", {})

    lines = [
        "# Paper Summary",
        "",
        "## Setup",
        f"- Dataset: {dataset}",
        f"- Split: {split_desc}",
        f"- Window counts (train/val/test): {window_counts}",
        "",
        "## Stress Classification (Test)",
        f"- XGBoost AUROC: {_fmt(xgb.get('auroc'))}, AUPRC: {_fmt(xgb.get('auprc'))}, F1: {_fmt(xgb.get('f1'))}",
        f"- TFT AUROC: {_fmt(tft.get('auroc'))}, AUPRC: {_fmt(tft.get('auprc'))}, F1: {_fmt(tft.get('f1'))}",
        "",
        "## Calibration",
        f"- XGBoost Brier: {_fmt(xgb.get('brier'))}, ECE: {_fmt(xgb.get('ece'))}",
        f"- TFT Brier: {_fmt(tft.get('brier'))}, ECE: {_fmt(tft.get('ece'))}",
        "",
        "## Personalization Ablation (XGBoost AUROC)",
        f"- Profiles ON: {_fmt(ablation.get('profiles_on', {}).get('stress_auroc'))}",
        f"- Profiles OFF: {_fmt(ablation.get('profiles_off', {}).get('stress_auroc'))}",
        "",
        "## Notes",
        "- If any AUROC/AUPRC are n/a, the test split likely contained only one class or too few windows.",
        "- Use plots in the run folder for figures.",
    ]

    out_path = run_dir / "paper_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
