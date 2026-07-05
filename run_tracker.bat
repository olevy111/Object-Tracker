@echo off
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo First run: creating a Python virtual environment ^(one-time setup^)...
    python -m venv venv
    if errorlevel 1 py -3 -m venv venv
)
if not exist venv\Scripts\python.exe (
    echo ERROR: Python 3 was not found. Install it from https://www.python.org and try again.
    pause
    exit /b 1
)

venv\Scripts\python.exe -c "import cv2, numpy" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages ^(one-time setup, may take a minute^)...
    venv\Scripts\python.exe -m pip install --upgrade pip
    venv\Scripts\python.exe -m pip install -r requirements.txt
)

start "" venv\Scripts\pythonw.exe app.py
exit
