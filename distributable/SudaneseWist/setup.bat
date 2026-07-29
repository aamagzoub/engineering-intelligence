@echo off
title Sudanese Wist - Setup
echo ============================================
echo  Sudanese Wist - First Time Setup
echo ============================================
echo.
echo Checking Python...

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed.
    echo Download from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo Python found. Installing pygame-ce...
python -m pip install pygame-ce --quiet
echo.
echo ============================================
echo  Setup complete! Run "play.bat" to start.
echo ============================================
pause
