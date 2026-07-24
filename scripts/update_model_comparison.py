"""Regenerate docs/model_comparison.md from latest run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import pandas as pd


RUN_ROOT = Path("experiments/runs")
OUTPUT = Path("docs/model_comparison.md")


def _latest_run(prefix: str) -> Path:
    candidates = sorted([p for p in RUN_ROOT.iterdir() if p.is_dir() and p.name.startswith(prefix)], key=lambda p: p.name)
    if not candidates:
        raise FileNotFoundError(f"No run directory found for prefix {prefix!r} under {RUN_ROOT}")
    return candidates[-1]


def _git_commit(run_dir: Path | None = None) -> str:
    if run_dir is not None:
        metadata_path = run_dir / "run_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("git_commit"):
                return str(metadata["git_commit"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _source_line(run_dir: Path, commit: str) -> str:
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        return f"Source run: `{run_dir.name}`. Git commit: `{commit}`."
    return (
        f"Source run: `{run_dir.name}`. Git commit: `{commit}` "
        "(documentation-generation commit; this run artifact does not include `run_metadata.json`)."
    )


def _multiphysio_section(run_dir: Path, commit: str) -> str:
    path = run_dir / "aggregate_results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    rows = []
    for target, group in df.groupby("target", observed=True):
        non_dummy = group[~group["model"].astype(str).str.startswith("dummy")].copy()
        best = non_dummy.sort_values("auprc_mean", ascending=False).iloc[0] if not non_dummy.empty else group.iloc[0]
        dummy = group[group["model"].astype(str) == "dummy_most_frequent"]
        dummy_auprc = dummy["auprc_mean"].iloc[0] if not dummy.empty else None
        rows.append(
            [
                target,
                _fmt(best.get("prevalence_mean")),
                _fmt(best.get("auroc_valid_fold_proportion")),
                str(best["model"]),
                _fmt(best.get("auroc_mean")),
                _fmt(best.get("auprc_mean")),
                _fmt(dummy_auprc),
            ]
        )
    rows.sort(key=lambda r: r[0])
    return "\n".join(
        [
            "## MultiPhysio 60-Second Feature Benchmark",
            "",
            "Protocol: capped leave-one-subject-out smoke over `bio_features_60s.csv`, `cv.max_folds: 5`. TFT is disabled because the input is a 60-second tabular feature table.",
            "",
            _source_line(run_dir, commit),
            "",
            _table(
                ["Target", "Prevalence", "Valid AUROC fold proportion", "Best non-dummy by AUPRC", "AUROC mean", "AUPRC mean", "Dummy AUPRC mean"],
                rows,
            ),
            "",
            f"Full metrics: `{run_dir / 'aggregate_results.csv'}`.",
        ]
    )


def _fast_section(run_dir: Path, commit: str) -> str:
    path = run_dir / "fast_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                str(row["model"]),
                "S2-S5",
                "S6-S7",
                "S8-S9",
                f"{_fmt(row.get('context_seconds'))}s",
                f"{_fmt(row.get('inference_stride_seconds'))}s",
                _fmt(row.get("auroc")),
                _fmt(row.get("auprc")),
                _fmt(row.get("f1")),
                f"{_fmt(row.get('latency_p95_ms'))} ms",
            ]
        )
    return "\n".join(
        [
            "## WESAD Fast Current-State Detection",
            "",
            "Task: `P(current WESAD protocol stress state | recent causal physiology)`.",
            "",
            _source_line(run_dir, commit),
            "",
            _table(
                ["Model", "Train subjects", "Validation subjects", "Test subjects", "Context", "Stride", "AUROC", "AUPRC", "F1 selected", "p95 latency"],
                rows,
            ),
        ]
    )


def _slow_section(run_dir: Path, commit: str) -> str:
    path = run_dir / "slow_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                str(row["model"]),
                f"{_fmt(row.get('horizon_seconds'), digits=0)}s",
                _fmt(row.get("auroc")),
                _fmt(row.get("auprc")),
                _fmt(row.get("f1")),
                _fmt(row.get("fixed_0_5_f1")),
                _fmt(row.get("brier")),
                _fmt(row.get("ece")),
                f"{_fmt(row.get('latency_p95_ms'))} ms",
            ]
        )
    return "\n".join(
        [
            "## WESAD Slow Forecasting",
            "",
            "Task: `P(WESAD protocol stress at t + horizon | physiology up to t)`. Context is 30s, stride is 1s. Calibration is horizon-specific Platt scaling.",
            "",
            _source_line(run_dir, commit),
            "",
            _table(
                ["Model", "Horizon", "AUROC", "AUPRC", "F1 selected", "F1 fixed 0.5", "Brier", "ECE", "p95 latency"],
                rows,
            ),
            "",
            "These are the current S8/S9-aligned slow results. They supersede the earlier smoke table where the TFT-family rows showed AUROC around 0.97 on a different held-out split.",
        ]
    )


def _replay_section(run_dirs: list[Path]) -> str:
    rows = []
    for run_dir in run_dirs:
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8")) if (run_dir / "run_metadata.json").exists() else {}
        summary = json.loads((run_dir / "summary_metrics.json").read_text(encoding="utf-8"))
        category = metadata.get("result_category") or summary.get("result_category") or run_dir.name
        if category == "real_model_physiological_replay":
            phys = summary.get("physiological_replay_metrics", {}).get("combined_policy_metrics", {})
            result = f"F1 {_fmt(phys.get('f1'))}; AUPRC {_fmt(phys.get('auprc'))}; detected episodes {phys.get('detected_episodes', '')}/{phys.get('number_of_stress_episodes', '')}"
        elif (run_dir / "scenario_validation.json").exists():
            validation = json.loads((run_dir / "scenario_validation.json").read_text(encoding="utf-8"))
            result = f"Scenario manifest passed: {validation.get('passed')}"
        else:
            result = "Summary metrics saved"
        rows.append([category, run_dir.name, metadata.get("git_commit", _git_commit(run_dir)), result])
    return "\n".join(
        [
            "## Replay Policy Results",
            "",
            "Replay results are policy simulations, not classifier rankings.",
            "",
            _table(["Category", "Run ID", "Git commit", "Result summary"], rows),
        ]
    )


def main() -> None:
    multiphysio = _latest_run("multiphysio_cv_")
    fast = _latest_run("fast_wesad_")
    slow = _latest_run("slow_tft_")
    replay_runs = [
        _latest_run("streaming_real_model_physiological_replay_"),
        _latest_run("streaming_oracle_label_policy_simulation_"),
        _latest_run("streaming_synthetic_physical_safety_simulation_"),
        _latest_run("streaming_combined_integration_simulation_"),
    ]
    commit = _git_commit()
    text = "\n\n".join(
        [
            "# Model Comparison",
            "<!-- Generated by `python scripts/update_model_comparison.py`. Do not manually copy result tables from old runs. -->",
            "Results below are regenerated from the latest run artifacts under `experiments/runs/`. Incompatible tasks are not ranked against each other.",
            _multiphysio_section(multiphysio, commit),
            _fast_section(fast, commit),
            _slow_section(slow, commit),
            "## Historical Slow Table",
            "The earlier WESAD slow forecasting table with TFT-family AUROC near 0.97 is historical. It came from a pre-alignment smoke run with different held-out subjects and must not be presented as the current S8/S9-aligned result.",
            _replay_section(replay_runs),
            "",
        ]
    )
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"Updated {OUTPUT}")


if __name__ == "__main__":
    main()
