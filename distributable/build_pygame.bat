@echo off
echo ============================================
echo  Building Sudanese Wist - PyGame Edition
echo ============================================
echo.

cd /d "%~dp0\.."

echo Installing PyInstaller...
python -m pip install pyinstaller --quiet

echo.
echo Building .exe (this may take a few minutes)...
echo.

pyinstaller --onefile --windowed --name "SudaneseWist_PyGame" ^
    --add-data "agents;agents" ^
    --add-data "environments;environments" ^
    --add-data "intelligence;intelligence" ^
    --add-data "gui_pygame;gui_pygame" ^
    --hidden-import "pygame" ^
    --hidden-import "agents.rule_based.rule_based_agent" ^
    --hidden-import "agents.learning.learning_agent" ^
    --hidden-import "environments.wist" ^
    --hidden-import "intelligence.core" ^
    gui_pygame\main.py

echo.
echo ============================================
if exist "dist\SudaneseWist_PyGame.exe" (
    echo  SUCCESS! Find your .exe at:
    echo  dist\SudaneseWist_PyGame.exe
) else (
    echo  BUILD FAILED - check errors above
)
echo ============================================
pause
