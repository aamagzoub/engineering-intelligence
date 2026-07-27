"""
Stats & Lab tab — extracted from app.py for maintainability.

This module contains the batch runner, learning progress chart,
and model management functionality. It's used as a mixin by the
main app class.
"""

# This file documents the Stats tab structure.
# The actual implementation remains in app.py for now
# to avoid breaking the tightly-coupled widget references
# (self._stat_labels, self._chart_canvas, etc.)
#
# Future refactoring plan:
# 1. Create a StatsTab class that takes (parent_frame, root, stats) 
# 2. Move _build_stats_tab, _update_stats_display, _draw_win_chart,
#    _run_batch, _batch_worker, _batch_progress_update, _batch_done,
#    _reset_stats, _save_model, _load_model, _reset_brain
# 3. StatsTab holds its own widget references
# 4. App creates StatsTab instance like HumanTab and AdvisorTab
#
# This is deferred to avoid breaking the running application.
# The current code works and is tested.
