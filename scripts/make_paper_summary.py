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
    class_balance = metrics.get("class_balance", {})
    task_name = metrics.get("task_name", "stress")
    task_cfg = cfg.get("task", {})
    ds_cfg = cfg.get("dataset", {})
    sampling = float(task_cfg.get("sampling_rate_hz", 1.0))
    downsample = int(ds_cfg.get("downsample_factor") or 1)
    effective_hz = sampling / max(1, downsample)
    window_sec = float(task_cfg.get("window_length", 1)) / max(effective_hz, 1e-6)
    horizon_sec = float(task_cfg.get("horizon_steps", 1)) / max(effective_hz, 1e-6)
    step_sec = float(task_cfg.get("window_step", 1)) / max(effective_hz, 1e-6)

    xgb = metrics.get("xgboost", {}).get("stress", {})
    tft = metrics.get("tft", {}).get("stress", {})
    ablation = metrics.get("ablation_profiles", {})

    lines = [
        "# Paper Summary",
        "",
        "## Setup",
        f"- Dataset: {dataset}",
        f"- Split: {split_desc}",
        f"- Task: {task_name}",
        f"- Window/Horizon/Step seconds: {window_sec:.2f} / {horizon_sec:.2f} / {step_sec:.2f}",
        f"- Window counts (train/val/test): {window_counts}",
        f"- Class balance (train/val/test): {class_balance}",
        f"- Threshold policy (XGBoost): {xgb.get('threshold_policy', 'n/a')}",
        "",
        "## Stress Classification (Test)",
        f"- XGBoost AUROC: {_fmt(xgb.get('auroc'))}, AUPRC: {_fmt(xgb.get('auprc'))}, F1: {_fmt(xgb.get('f1'))}",
        f"- TFT AUROC: {_fmt(tft.get('auroc'))}, AUPRC: {_fmt(tft.get('auprc'))}, F1: {_fmt(tft.get('f1'))}",
        f"- XGBoost pred+ rate default/optimal: {_fmt(xgb.get('default_positive_rate'))} / {_fmt(xgb.get('optimal_positive_rate'))}",
        f"- TFT pred+ rate default/optimal: {_fmt(tft.get('default_positive_rate'))} / {_fmt(tft.get('optimal_positive_rate'))}",
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
