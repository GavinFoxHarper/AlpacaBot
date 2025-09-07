@echo off
title Virtual Mouse Controller
color 0B
cls

echo Starting Virtual Mouse Controller...
echo.

REM Install pynput if needed and run the virtual mouse
python virtual_mouse.py

if errorlevel 1 (
    echo.
    echo Error running Virtual Mouse!
    echo Installing required package...
    pip install pynput
    echo.
    echo Retrying...
    python virtual_mouse.py
)

pause