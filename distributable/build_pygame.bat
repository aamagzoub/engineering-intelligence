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

pyinstaller --onefile --windowed --name "Play_Wist_v4.0.0" ^
    --add-data "agents;agents" ^
    --add-data "environments;environments" ^
    --add-data "intelligence;intelligence" ^
    --add-data "tibrain;tibrain" ^
    --add-data "gui_wist;gui_wist" ^
    --hidden-import "pygame" ^
    --hidden-import "tibrain" ^
    --hidden-import "tibrain.agent" ^
    --hidden-import "tibrain.q_learning" ^
    --hidden-import "tibrain.q_table" ^
    --hidden-import "tibrain.policy" ^
    --hidden-import "tibrain.replay_buffer" ^
    --hidden-import "tibrain.neural_net" ^
    --hidden-import "tibrain.mcts" ^
    --hidden-import "tibrain.reward" ^
    --hidden-import "tibrain.evaluation" ^
    --hidden-import "tibrain.persistence" ^
    --hidden-import "tibrain.training" ^
    --hidden-import "tibrain.discovery" ^
    --hidden-import "tibrain.discovery.discovery_engine" ^
    --hidden-import "tibrain.discovery.pattern" ^
    --hidden-import "agents.wist_rule_based.rule_based_agent" ^
    --hidden-import "agents.wist_learning.learning_agent" ^
    --hidden-import "environments.wist" ^
    --hidden-import "intelligence.core" ^
    gui_wist\main.py

echo.
echo ============================================
if exist "dist\Play_Wist_v4.0.0.exe" (
    echo  SUCCESS! Find your .exe at:
    echo  dist\Play_Wist_v4.0.0.exe
) else (
    echo  BUILD FAILED - check errors above
)
echo ============================================
pause
