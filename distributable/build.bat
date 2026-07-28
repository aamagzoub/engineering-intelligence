@echo off
echo Building Sudanese Wist - Play Against AI...
echo.

cd /d "%~dp0\.."

pip install pyinstaller

pyinstaller --onefile --windowed --name "SudaneseWist" ^
    --add-data "agents;agents" ^
    --add-data "environments;environments" ^
    --add-data "intelligence;intelligence" ^
    --add-data "gui;gui" ^
    distributable\play_wist.py

echo.
echo Done! Find the .exe in dist\SudaneseWist.exe
pause
