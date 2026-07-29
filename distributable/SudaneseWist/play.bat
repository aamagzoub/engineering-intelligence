@echo off
title Sudanese Wist
cd /d "%~dp0\game"
python -m gui_pygame.main
if errorlevel 1 (
    echo.
    echo Error! Run setup.bat first.
    pause
)
