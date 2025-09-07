@echo off
REM Debug mode launcher for AlpacaBot - Verbose output for troubleshooting
cls
color 0E
title AlpacaBot - Debug Mode

echo ============================================
echo      ALPACABOT DEBUG MODE
echo ============================================
echo.

REM Set debug environment variables
set DEBUG_MODE=true
set LOG_LEVEL=DEBUG
set PYTHONPATH=%CD%

echo [DEBUG] Working Directory: %CD%
echo [DEBUG] Python Path: %PYTHONPATH%
echo.

REM Check Python version
echo [DEBUG] Python Version:
python --version
echo.

REM Check installed packages
echo [DEBUG] Checking required packages...
python -c "import pandas; print(f'  pandas: {pandas.__version__}')" 2>nul || echo   pandas: NOT INSTALLED
python -c "import numpy; print(f'  numpy: {numpy.__version__}')" 2>nul || echo   numpy: NOT INSTALLED
python -c "import yfinance; print(f'  yfinance: {yfinance.__version__}')" 2>nul || echo   yfinance: NOT INSTALLED
python -c "import alpaca_trade_api; print(f'  alpaca_trade_api: OK')" 2>nul || echo   alpaca_trade_api: NOT INSTALLED
echo.

REM Check configuration
echo [DEBUG] Testing configuration import...
python -c "from config import validate_config; is_valid, issues = validate_config(); print('Config Status: VALID' if is_valid else f'Config Issues: {issues}')"
echo.

REM Check API connection
echo [DEBUG] Testing API connection...
python -c "from trading.alpaca_broker import AlpacaBroker; broker = AlpacaBroker(); print('API Connection: OK' if broker else 'API Connection: FAILED')" 2>nul || echo API Connection: ERROR
echo.

REM Run with debug output
echo [DEBUG] Starting application with verbose logging...
echo ============================================
echo.

python -u live_monitoring_dashboard.py

echo.
echo [DEBUG] Application terminated.
pause
exit /b