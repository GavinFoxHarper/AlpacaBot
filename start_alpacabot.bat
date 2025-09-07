@echo off
REM ============================================
REM AlpacaBot Trading System Launcher
REM ============================================
cls
color 0A
title AlpacaBot Trading System

echo ============================================
echo         ALPACABOT TRADING SYSTEM
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to PATH
    pause
    exit /b 1
)

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [INFO] No virtual environment found, using system Python
)

REM Install/Update dependencies
echo.
echo [INFO] Checking dependencies...
pip install -q --upgrade pip
pip install -q -r requirements.txt 2>nul
if not exist "requirements.txt" (
    echo [INFO] No requirements.txt found, installing core packages...
    pip install -q pandas numpy yfinance alpaca-trade-api python-dotenv
    pip install -q scikit-learn matplotlib seaborn openpyxl
)

REM Create necessary directories
echo.
echo [INFO] Setting up directories...
if not exist "logs" mkdir logs
if not exist "models" mkdir models
if not exist "reports" mkdir reports
if not exist "data" mkdir data
if not exist "config_profiles" mkdir config_profiles

REM Check for .env file
if not exist ".env" (
    echo.
    echo [WARNING] No .env file found!
    echo Please create a .env file with your API keys:
    echo   ALPACA_API_KEY=your_api_key
    echo   ALPACA_SECRET_KEY=your_secret_key
    echo.
    pause
)

REM Display menu
:menu
echo.
echo ============================================
echo            SELECT OPERATION MODE
echo ============================================
echo.
echo   1. Live Trading Dashboard
echo   2. Backtest Mode
echo   3. Training Mode (ML/Q-Learning)
echo   4. Configuration Manager
echo   5. Performance Analysis
echo   6. Exit
echo.
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto live_trading
if "%choice%"=="2" goto backtest
if "%choice%"=="3" goto training
if "%choice%"=="4" goto config
if "%choice%"=="5" goto analysis
if "%choice%"=="6" goto end

echo [ERROR] Invalid choice. Please select 1-6.
goto menu

:live_trading
echo.
echo ============================================
echo         STARTING LIVE TRADING MODE
echo ============================================
echo.
echo [WARNING] Make sure you're using PAPER trading!
echo Starting in 5 seconds... (Press Ctrl+C to cancel)
timeout /t 5 /nobreak >nul
echo.
python live_monitoring_dashboard.py
if errorlevel 1 (
    echo.
    echo [ERROR] Live trading dashboard crashed!
    echo Check logs\alpacabot_%date:~-4,4%%date:~-10,2%%date:~-7,2%.log for details
    pause
)
goto menu

:backtest
echo.
echo ============================================
echo         STARTING BACKTEST MODE
echo ============================================
echo.
set /p symbols="Enter symbols to backtest (comma-separated, e.g., AAPL,MSFT,GOOGL): "
set /p start_date="Enter start date (YYYY-MM-DD): "
set /p end_date="Enter end date (YYYY-MM-DD): "
echo.
echo Running backtest for %symbols% from %start_date% to %end_date%...
python -c "from trading.enhanced_backtest_engine import EnhancedBacktestEngine; engine = EnhancedBacktestEngine(); results = engine.run_backtest(['%symbols%'.split(',')], '%start_date%', '%end_date%'); print(f'\nBacktest complete! Results saved to reports/')"
if errorlevel 1 (
    echo.
    echo [ERROR] Backtest failed!
    pause
)
goto menu

:training
echo.
echo ============================================
echo         STARTING TRAINING MODE
echo ============================================
echo.
echo Select training type:
echo   1. Q-Learning Agent
echo   2. ML Models (Random Forest, XGBoost)
echo   3. Comprehensive Training (All models)
echo   4. Back to main menu
echo.
set /p train_choice="Enter your choice (1-4): "

if "%train_choice%"=="1" (
    echo.
    echo Training Q-Learning agent...
    python -c "from training.comprehensive_trainer import ComprehensiveTrainer; trainer = ComprehensiveTrainer(); trainer.train_q_learning()"
) else if "%train_choice%"=="2" (
    echo.
    echo Training ML models...
    python -c "from training.ml_trainer import train_all_models; train_all_models()"
) else if "%train_choice%"=="3" (
    echo.
    echo Running comprehensive training...
    python -c "from training.comprehensive_trainer import ComprehensiveTrainer; trainer = ComprehensiveTrainer(); trainer.run_comprehensive_training()"
) else if "%train_choice%"=="4" (
    goto menu
) else (
    echo [ERROR] Invalid choice.
    goto training
)

if errorlevel 1 (
    echo.
    echo [ERROR] Training failed!
    pause
)
goto menu

:config
echo.
echo ============================================
echo        CONFIGURATION MANAGER
echo ============================================
echo.
echo   1. View current configuration
echo   2. Edit configuration
echo   3. Load configuration profile
echo   4. Save configuration profile
echo   5. Validate configuration
echo   6. Back to main menu
echo.
set /p config_choice="Enter your choice (1-6): "

if "%config_choice%"=="1" (
    echo.
    python -c "from config import *; print(f'Paper Trading: {PAPER_TRADING}'); print(f'Initial Cash: ${INITIAL_CASH}'); print(f'Max Positions: {MAX_POSITIONS}'); print(f'Stop Loss: {STOP_LOSS_PERCENT}%%'); print(f'Take Profit: {TAKE_PROFIT_PERCENT}%%')"
) else if "%config_choice%"=="2" (
    echo.
    echo Opening config.py in default editor...
    start notepad config.py
) else if "%config_choice%"=="3" (
    echo.
    set /p profile="Enter profile name to load: "
    python -c "from config import load_profile; load_profile('%profile%')"
) else if "%config_choice%"=="4" (
    echo.
    set /p profile="Enter profile name to save: "
    python -c "from config import save_profile; save_profile('%profile%')"
) else if "%config_choice%"=="5" (
    echo.
    python -c "from config import validate_config; is_valid, issues = validate_config(); print('Configuration is VALID' if is_valid else f'Issues found: {issues}')"
) else if "%config_choice%"=="6" (
    goto menu
) else (
    echo [ERROR] Invalid choice.
)

pause
goto config

:analysis
echo.
echo ============================================
echo        PERFORMANCE ANALYSIS
echo ============================================
echo.
echo   1. Show latest backtest results
echo   2. Analyze trading performance
echo   3. Generate detailed report
echo   4. View trade logs
echo   5. Back to main menu
echo.
set /p analysis_choice="Enter your choice (1-5): "

if "%analysis_choice%"=="1" (
    echo.
    python show_backtest_results.py
) else if "%analysis_choice%"=="2" (
    echo.
    python detailed_results_analyzer.py
) else if "%analysis_choice%"=="3" (
    echo.
    echo Generating comprehensive report...
    python -c "from utils.report_generator import generate_comprehensive_report; generate_comprehensive_report()"
) else if "%analysis_choice%"=="4" (
    echo.
    if exist "logs\trades.log" (
        type logs\trades.log | more
    ) else (
        echo No trade logs found.
    )
) else if "%analysis_choice%"=="5" (
    goto menu
) else (
    echo [ERROR] Invalid choice.
    goto analysis
)

pause
goto menu

:end
echo.
echo ============================================
echo    Thank you for using AlpacaBot!
echo ============================================
echo.
timeout /t 3 /nobreak >nul
exit /b 0