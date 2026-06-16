@echo off
setlocal
cd /d "%~dp0health_monitor"
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
echo.
echo Health monitor is ready. Run: health_monitor\run_health_monitor.bat
