@echo off
setlocal
cd /d "%~dp0"
set "VENV_DIR=.venv311"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating Python 3.11 virtual environment...
    py -3.11 -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Python 3.11 is required. Install it from https://www.python.org/downloads/
        exit /b 1
    )
)

"%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt
"%VENV_DIR%\Scripts\python.exe" health_monitor.py --camera 0 --signal-csv raw_rgb.csv --cleaned-signal-csv cleaned_rgb.csv --features-csv extracted_features.csv
