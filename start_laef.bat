@echo off
cd /d "C:\Users\jclif\AlpacaBot"
call "venv\Scripts\activate.bat"
echo.
echo ====================================
echo    LAEF Trading System Activated
echo ====================================
echo Virtual Environment: venv
echo Working Directory: %CD%
echo.
echo Available Commands:
echo   python laef_unified_system.py     - Start LAEF main system
echo   python start_laef_interactive.py  - Start interactive mode
echo   python run_live_monitoring.py     - Start live monitoring
echo   python -m pytest tests           - Run tests
echo   deactivate                       - Exit virtual environment
echo.
cmd /k