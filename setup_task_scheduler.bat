@echo off
REM Windows Task Scheduler Setup for LAEF Trading System
REM This script creates a scheduled task to run the bot every weekday at 9:25 AM

echo ========================================
echo LAEF Windows Task Scheduler Setup
echo ========================================

REM Get current directory
set "SCRIPT_DIR=%~dp0"
set "PYTHON_PATH=%SCRIPT_DIR%automated_daily_trader.py"

echo Setting up daily automated trading...
echo Script location: %PYTHON_PATH%

REM Create the scheduled task
schtasks /create ^
    /tn "LAEF Daily Trading Bot" ^
    /tr "python \"%PYTHON_PATH%\"" ^
    /sc weekly ^
    /d MON,TUE,WED,THU,FRI ^
    /st 09:25 ^
    /ru "%USERNAME%" ^
    /rl highest ^
    /f

if %ERRORLEVEL% == 0 (
    echo ========================================
    echo SUCCESS: Scheduled task created!
    echo ========================================
    echo Task Name: LAEF Daily Trading Bot
    echo Schedule: Monday-Friday at 9:25 AM
    echo Script: %PYTHON_PATH%
    echo.
    echo The bot will now start automatically every weekday
    echo at 9:25 AM, wait for market open at 9:30 AM,
    echo and run throughout the trading day.
    echo.
    echo To manage this task:
    echo - Open Task Scheduler (taskschd.msc)
    echo - Look for "LAEF Daily Trading Bot"
    echo.
    echo To test the task:
    echo - Right-click the task and select "Run"
    echo.
    echo To disable/delete the task:
    echo - schtasks /delete /tn "LAEF Daily Trading Bot" /f
) else (
    echo ========================================
    echo ERROR: Failed to create scheduled task
    echo ========================================
    echo Please run this script as Administrator
    echo or create the task manually in Task Scheduler
)

echo.
echo Additional Setup:
echo 1. Ensure Python is in your PATH
echo 2. Verify .env file has all required settings
echo 3. Test email configuration
echo 4. Run a manual test first

pause