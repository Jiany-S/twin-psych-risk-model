@echo off
setlocal

python scripts\run_multiphysio_smoke.py --config src/config\multiphysio_smoke.yaml
if errorlevel 1 exit /b %errorlevel%

endlocal
