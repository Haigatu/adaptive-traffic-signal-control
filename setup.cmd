@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements-demo.txt
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m atsc.prepare_demo
if errorlevel 1 exit /b 1
echo Ready. Install SUMO 1.27.1, set SUMO_HOME, then run demo.cmd.
