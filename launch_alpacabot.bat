@echo off
title AlpacaBot Launcher
color 0A

echo.
echo ============================================
echo    AlpacaBot LAEF Trading System Launcher
echo ============================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ❌ Virtual environment not found!
    echo Please run setup first or create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Check Python availability
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not available in virtual environment
    pause
    exit /b 1
)

echo ✅ Virtual environment activated
echo.

REM Check dependencies
echo 🔍 Checking dependencies...
python utils\check_dependencies.py
if errorlevel 1 (
    echo.
    echo ❌ Dependency check failed
    echo Would you like to install missing dependencies? (y/n)
    set /p install_deps=
    if /i "%install_deps%"=="y" (
        echo 📦 Installing dependencies...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo ❌ Failed to install dependencies
            pause
            exit /b 1
        )
        echo ✅ Dependencies installed successfully
    ) else (
        echo ⚠️ Continuing without installing dependencies...
    )
)

echo.
echo 🚀 Starting AlpacaBot LAEF System...
echo Press Ctrl+C at any time to exit
echo.

REM Launch the main system
python start_laef_interactive.py

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo ❌ AlpacaBot exited with error
    pause
)

echo.
echo 👋 AlpacaBot session ended
pause