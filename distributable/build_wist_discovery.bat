@echo off
echo ============================================
echo  Building Wist Discovery Watcher - PyGame
echo ============================================
echo.

cd /d "%~dp0\.."

echo Installing PyInstaller...
python -m pip install pyinstaller --quiet

echo.
echo Building .exe (this may take a few minutes)...
echo.

pyinstaller --onefile --windowed --name "WistDiscovery" ^
    --add-data "agents;agents" ^
    --add-data "environments;environments" ^
    --add-data "intelligence;intelligence" ^
    --add-data "gui_wist_discovery;gui_wist_discovery" ^
    --add-data "gui_wist;gui_wist" ^
    --hidden-import "pygame" ^
    --hidden-import "agents.wist_discovery.discovery_agent" ^
    --hidden-import "environments.wist" ^
    --hidden-import "environments.wist.environment" ^
    --hidden-import "environments.wist.round" ^
    --hidden-import "environments.wist.rules" ^
    --hidden-import "environments.wist.scoring" ^
    --hidden-import "environments.wist.setup" ^
    --hidden-import "environments.wist.tasmiya_engine" ^
    --hidden-import "environments.wist.trick" ^
    --hidden-import "intelligence.core" ^
    --hidden-import "intelligence.core.cards" ^
    gui_wist_discovery\main.py

echo.
echo ============================================
if exist "dist\WistDiscovery.exe" (
    echo  SUCCESS! Find your .exe at:
    echo  dist\WistDiscovery.exe
) else (
    echo  BUILD FAILED - check errors above
)
echo ============================================
pause
