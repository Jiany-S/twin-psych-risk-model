"""Filesystem helpers for experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class RunPaths:
    root: Path
    plots: Path
    models: Path
    data: Path


def create_run_dir(root: str | Path) -> RunPaths:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(root) / timestamp
    plots = run_root / "plots"
    models = run_root / "models"
    data = run_root / "data"
    for path in (run_root, plots, models, data):
        path.mkdir(parents=True, exist_ok=True)
    return RunPaths(root=run_root, plots=plots, models=models, data=data)
