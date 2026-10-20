@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv. See DEMO_WINDOWS.md for setup instructions.
    exit /b 1
)
".venv\Scripts\python.exe" -u -m atsc.compare %*
exit /b %errorlevel%
