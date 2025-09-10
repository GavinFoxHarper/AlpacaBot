@echo off
REM Setup script for LAEF AlpacaBot Windows Task Scheduler
REM Configures automated daily trading at 9:00 AM ET

echo ========================================
echo LAEF AlpacaBot Scheduler Setup
echo ========================================
echo.

REM Check for admin privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Administrator privileges required!
    echo Please run this script as Administrator.
    pause
    exit /b 1
)

REM Set variables
set TASK_NAME=LAEF_AlpacaBot_Daily
set TASK_PATH=\LAEF\
set PYTHON_EXE=C:\Users\jclif\AlpacaBot\venv\Scripts\python.exe
set SCRIPT_PATH=C:\Users\jclif\AlpacaBot\orchestrator.py
set WORKING_DIR=C:\Users\jclif\AlpacaBot

REM Check if Python virtual environment exists
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python virtual environment not found!
    echo Please create virtual environment first:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

REM Delete existing task if it exists
echo Checking for existing task...
schtasks /query /tn "%TASK_PATH%%TASK_NAME%" >nul 2>&1
if %errorLevel% equ 0 (
    echo Removing existing task...
    schtasks /delete /tn "%TASK_PATH%%TASK_NAME%" /f
)

REM Create new scheduled task for daily trading at 9:00 AM ET
echo Creating daily trading task (9:00 AM ET)...
schtasks /create ^
    /tn "%TASK_PATH%%TASK_NAME%_Trading" ^
    /tr "\"%PYTHON_EXE%\" \"%SCRIPT_PATH%\" --paper" ^
    /sc weekly ^
    /d MON,TUE,WED,THU,FRI ^
    /st 09:00 ^
    /ru %USERNAME% ^
    /rl HIGHEST ^
    /f

REM Create pre-market preparation task at 8:30 AM ET
echo Creating pre-market preparation task (8:30 AM ET)...
schtasks /create ^
    /tn "%TASK_PATH%%TASK_NAME%_PreMarket" ^
    /tr "\"%PYTHON_EXE%\" \"%SCRIPT_PATH%\" --paper --pre-market" ^
    /sc weekly ^
    /d MON,TUE,WED,THU,FRI ^
    /st 08:30 ^
    /ru %USERNAME% ^
    /rl HIGHEST ^
    /f

REM Create end-of-day report task at 4:30 PM ET
echo Creating end-of-day report task (4:30 PM ET)...
schtasks /create ^
    /tn "%TASK_PATH%%TASK_NAME%_EOD" ^
    /tr "\"%PYTHON_EXE%\" \"%SCRIPT_PATH%\" --generate-report" ^
    /sc weekly ^
    /d MON,TUE,WED,THU,FRI ^
    /st 16:30 ^
    /ru %USERNAME% ^
    /rl HIGHEST ^
    /f

echo.
echo ========================================
echo Scheduled Tasks Created Successfully!
echo ========================================
echo.
echo Tasks created:
echo   - %TASK_NAME%_Trading (9:00 AM ET)
echo   - %TASK_NAME%_PreMarket (8:30 AM ET)
echo   - %TASK_NAME%_EOD (4:30 PM ET)
echo.
echo To view tasks: schtasks /query /tn "%TASK_PATH%*"
echo To test now: schtasks /run /tn "%TASK_PATH%%TASK_NAME%_Trading"
echo.
echo NOTE: Tasks are configured for paper trading mode.
echo To enable live trading, edit the task and remove --paper flag.
echo.
pause