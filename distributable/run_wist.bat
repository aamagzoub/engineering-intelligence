@echo off
title Sudanese Wist
cd /d "%~dp0\.."
python gui_wist\main.py
if errorlevel 1 (
    echo.
    echo Python not found or error occurred.
    echo Install Python from python.org and pygame-ce: pip install pygame-ce
    pause
)
