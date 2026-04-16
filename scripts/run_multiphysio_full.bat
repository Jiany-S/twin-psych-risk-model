@echo off
setlocal

python scripts\run_multiphysio_full.py --config src/config\multiphysio_full.yaml
if errorlevel 1 exit /b %errorlevel%

endlocal
