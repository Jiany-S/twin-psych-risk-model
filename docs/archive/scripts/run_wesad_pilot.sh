#!/usr/bin/env bash
set -euo pipefail

SUBJECTS="${1:-S2,S3,S4,S5,S6,S7,S8,S9}"

python scripts/prepare_wesad_subset.py --subjects "${SUBJECTS}"
python -m src.run_experiment --config src/config/wesad_pilot_8subj.yaml
python -m src.run_experiment --config src/config/wesad_pilot_8subj_no_profiles.yaml

python - <<'PY'
from pathlib import Path

run_root = Path("experiments/runs")
runs = sorted([p for p in run_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
if len(runs) < 2:
    print("Could not find two run directories under experiments/runs.")
else:
    on_run = runs[1]
    off_run = runs[0]
    print(f"Profiles ON run dir : {on_run}")
    print(f"Profiles OFF run dir: {off_run}")
    print(f"Metrics ON : {on_run / 'metrics.json'}")
    print(f"Metrics OFF: {off_run / 'metrics.json'}")
PY
