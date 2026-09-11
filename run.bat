@echo off
echo ============================================
echo PDF Stamp Splitter - Install and Run
echo ============================================
echo.

echo Checking Python...
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.7 or higher from python.org
    pause
    exit /b 1
)

echo.
echo Installing dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo Starting PDF Stamp Splitter...
echo.

REM Use pythonw to avoid console window
start "" pythonw split_pdf_gui.py

exit
