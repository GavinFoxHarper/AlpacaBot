@echo off
REM Quick start for AlpacaBot - Goes directly to live monitoring dashboard
cls
color 0A
title AlpacaBot - Quick Start

echo ============================================
echo      ALPACABOT QUICK START
echo ============================================
echo.
echo Starting Live Monitoring Dashboard...
echo (Press Ctrl+C to stop)
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Start the live monitoring dashboard
python live_monitoring_dashboard.py

if errorlevel 1 (
    echo.
    echo [ERROR] Application crashed!
    echo Check the logs folder for details.
    pause
)

exit /b