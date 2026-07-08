@echo off
setlocal
cd /d "%~dp0"

if exist venv\Scripts\python.exe goto :deps

echo First run: one-time setup...

rem -- find a working Python 3 (the Microsoft Store alias fails the --version check) --
set "PYCMD="
py -3 --version >nul 2>&1
if not errorlevel 1 set "PYCMD=py -3"
if not defined PYCMD (
    python --version >nul 2>&1
    if not errorlevel 1 set "PYCMD=python"
)

if defined PYCMD goto :makevenv

echo Python 3 was not found on this computer.
echo Downloading and installing Python 3.12 now ^(one time, a few minutes^)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = 'Tls12'; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -OutFile '%TEMP%\pixeltracker_python_setup.exe'"
if not exist "%TEMP%\pixeltracker_python_setup.exe" (
    echo ERROR: could not download Python. Check the internet connection and try again,
    echo or install Python 3 manually from https://www.python.org
    pause
    exit /b 1
)
echo Installing Python ^(no admin needed^)...
start /wait "" "%TEMP%\pixeltracker_python_setup.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=0 Include_test=0
del "%TEMP%\pixeltracker_python_setup.exe" >nul 2>&1
set "PYCMD="%LocalAppData%\Programs\Python\Python312\python.exe""
%PYCMD% --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python installation did not complete. Install Python 3 manually
    echo from https://www.python.org and run this file again.
    pause
    exit /b 1
)

:makevenv
echo Creating a Python virtual environment...
%PYCMD% -m venv venv
if not exist venv\Scripts\python.exe (
    echo ERROR: could not create the virtual environment.
    pause
    exit /b 1
)

:deps
venv\Scripts\python.exe -c "import cv2, numpy" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages ^(one time, may take a minute^)...
    venv\Scripts\python.exe -m pip install --upgrade pip
    venv\Scripts\python.exe -m pip install -r requirements.txt
    venv\Scripts\python.exe -c "import cv2, numpy" >nul 2>&1
    if errorlevel 1 (
        echo ERROR: package installation failed. Check the internet connection and try again.
        pause
        exit /b 1
    )
    echo.
    echo Verifying the installation...
    venv\Scripts\python.exe src\check_setup.py
    if errorlevel 1 (
        echo ERROR: setup check failed ^(see the report above^).
        pause
        exit /b 1
    )
    echo.
)

start "" venv\Scripts\pythonw.exe src\app.py
exit
