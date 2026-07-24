@echo off
setlocal

set SUBJECTS=%1
if "%SUBJECTS%"=="" set SUBJECTS=S2,S3,S4,S5,S6,S7,S8,S9

python scripts\prepare_wesad_subset.py --subjects "%SUBJECTS%"
if errorlevel 1 exit /b 1

python -m src.run_experiment --config src/config/wesad_pilot_8subj.yaml
if errorlevel 1 exit /b 1

python -m src.run_experiment --config src/config/wesad_pilot_8subj_no_profiles.yaml
if errorlevel 1 exit /b 1

python -c "from pathlib import Path; runs=sorted([p for p in Path('experiments/runs').iterdir() if p.is_dir()], key=lambda p:p.stat().st_mtime, reverse=True); print(f'Profiles ON run dir : {runs[1]}' if len(runs)>1 else 'Profiles ON run dir : n/a'); print(f'Profiles OFF run dir: {runs[0]}' if len(runs)>0 else 'Profiles OFF run dir: n/a'); print(f'Metrics ON : {runs[1]/\"metrics.json\"}' if len(runs)>1 else 'Metrics ON : n/a'); print(f'Metrics OFF: {runs[0]/\"metrics.json\"}' if len(runs)>0 else 'Metrics OFF: n/a')"

endlocal
