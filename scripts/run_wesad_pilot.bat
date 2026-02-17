@echo off
setlocal

set SUBJECTS=%1
if "%SUBJECTS%"=="" set SUBJECTS=S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S13,S14,S15,S16,S17

python scripts\prepare_wesad_subset.py --subjects "%SUBJECTS%"
if errorlevel 1 exit /b 1

python -m src.run_experiment --config src/config/wesad_pilot_8subj.yaml
if errorlevel 1 exit /b 1

python -m src.run_experiment --config src/config/wesad_pilot_8subj_no_profiles.yaml
if errorlevel 1 exit /b 1

python -c "from pathlib import Path; import subprocess, sys; runs=sorted([p for p in Path('experiments/runs').iterdir() if p.is_dir()], key=lambda p:p.stat().st_mtime, reverse=True); on_run=runs[1] if len(runs)>1 else None; off_run=runs[0] if len(runs)>0 else None; print(f'Profiles ON run dir : {on_run}' if on_run else 'Profiles ON run dir : n/a'); print(f'Profiles OFF run dir: {off_run}' if off_run else 'Profiles OFF run dir: n/a'); print(f'Metrics ON : {on_run/\"metrics.json\"}' if on_run else 'Metrics ON : n/a'); print(f'Metrics OFF: {off_run/\"metrics.json\"}' if off_run else 'Metrics OFF: n/a'); [subprocess.check_call([sys.executable, 'scripts/make_paper_summary.py', '--run_dir', str(run)]) for run in (on_run, off_run) if run is not None]; print(f'Paper summary ON : {on_run/\"paper_summary.md\"}' if on_run else 'Paper summary ON : n/a'); print(f'Paper summary OFF: {off_run/\"paper_summary.md\"}' if off_run else 'Paper summary OFF: n/a')"

endlocal
