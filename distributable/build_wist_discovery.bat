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

pyinstaller --onefile --windowed --name "Wist_Learning_and_Discovering_Agent_v4.0.0" ^
    --add-data "agents;agents" ^
    --add-data "environments;environments" ^
    --add-data "intelligence;intelligence" ^
    --add-data "tibrain;tibrain" ^
    --add-data "gui_wist_discovery;gui_wist_discovery" ^
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
    --hidden-import "agents.wist_discovery.discovery_agent" ^
    --hidden-import "agents.wist_discovery.neural_net" ^
    --hidden-import "agents.wist_discovery.mcts" ^
    --hidden-import "agents.wist_discovery.insight_pipeline" ^
    --hidden-import "agents.wist_discovery.insight_pipeline.schema" ^
    --hidden-import "agents.wist_discovery.insight_pipeline.observation_store" ^
    --hidden-import "agents.wist_discovery.insight_pipeline.snapshot_extender" ^
    --hidden-import "agents.wist_discovery.insight_pipeline.evidence_aggregator" ^
    --hidden-import "agents.wist_discovery.insight_pipeline.promotion_pipeline" ^
    --hidden-import "agents.wist_discovery.insight_pipeline.generality_validator" ^
    --hidden-import "agents.wist_discovery.insight_pipeline.text_generator" ^
    --hidden-import "agents.wist_discovery.insight_pipeline.duplicate_merger" ^
    --hidden-import "agents.wist_discovery.insight_pipeline.confidence_dynamics" ^
    --hidden-import "agents.wist_discovery.insight_pipeline.migration" ^
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
    --hidden-import "gui_wist_discovery.game_engine" ^
    --hidden-import "gui_wist_discovery.training" ^
    --hidden-import "gui_wist_discovery.milestones" ^
    --hidden-import "gui_wist_discovery.insights" ^
    --hidden-import "gui_wist_discovery.renderer" ^
    --hidden-import "gui_wist_discovery.constants" ^
    gui_wist_discovery\main.py

echo.
echo ============================================
if exist "dist\Wist_Learning_and_Discovering_Agent_v4.0.0.exe" (
    echo  SUCCESS! Find your .exe at:
    echo  dist\Wist_Learning_and_Discovering_Agent_v4.0.0.exe
) else (
    echo  BUILD FAILED - check errors above
)
echo ============================================
pause
