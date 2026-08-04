# GUI Refactoring Plan

## Current State (to be refactored)

| File | Lines | Responsibility |
|------|-------|---------------|
| `controller.py` | 1696 | Everything: commands, phases, bidding, tricks, scoring, helpers |
| `app.py` | 1173 | Main app + Game Table tab + Stats tab all in one |

## Target State

### controller.py → split into:

1. **`controller.py`** (~350 lines) — Core state + commands
   - `__init__`, `cmd_*` methods, `start`, `stop`, `pause`, `reset`
   - `continue_simulation` (phase dispatcher only)
   - `_schedule`, `_cancel_timer`
   - `_make_agent`, `start_auto`, `start_step_mode`

2. **`controller_bidding.py`** (~400 lines) — Bidding phase
   - `_run_one_tasmiya`
   - `_init_step_bidding`, `_show_next_bid`
   - `prepare_shota_setup`
   - `_log_tasmiya_result`, `_update_gui_after_tasmiya`
   - `_deal_and_check_dak`

3. **`controller_tricks.py`** (~400 lines) — Trick play phase
   - `play_one_trick`, `play_next_trick_auto`, `play_next_trick_step`
   - `_play_next_card_step`, `_play_one_card_in_trick`
   - `_finish_current_trick`, `_get_current_trick_cards`
   - `finish_simulation`
   - `_start_next_shota_auto`, `_auto_next_shota`, `_reset_for_new_shota`

4. **`controller_helpers.py`** (~150 lines) — Display helpers ✅ DONE
   - `show_player_hands`, `sort_cards`, `format_card*`
   - `get_team_index`, `set_*_safe`, `_explain_card_play`
   - `extract_played_cards_from_trick`, `log_trick_cards`

### app.py → split into:

1. **`app.py`** (~200 lines) — Main shell
   - `WistAILabApp.__init__`, `_build_layout`, `run`
   - Tab creation (delegates to tab modules)
   - Top bar, controls

2. **`game_tab.py`** (~400 lines) — Game Table tab
   - `_build_game_tab`, `_build_centre`, `_build_top_bar`
   - `set_player_hand`, `set_played_cards`, `_redraw_centre_cards`
   - `set_player_status`, `set_player_bid`, etc.
   - Agent selector, load model for game

3. **`stats_tab.py`** (~450 lines) — Stats & Lab tab
   - `_build_stats_tab`
   - `_update_stats_display`, `_draw_win_chart`, `_draw_bar_chart`
   - `_run_batch`, `_batch_worker`, `_batch_progress_update`, `_batch_done`
   - `_reset_stats`, `_save_model`, `_load_model`, `_reset_brain`

## Implementation Strategy

Use **mixin classes** (multiple inheritance):
```python
# controller.py
from gui.controller_bidding import BiddingMixin
from gui.controller_tricks import TricksMixin
from gui.controller_helpers import ControllerHelpersMixin

class SimulationController(BiddingMixin, TricksMixin, ControllerHelpersMixin):
    ...
```

Each mixin defines methods that use `self.app`, `self.round`, etc.
— shared state is on the main class, logic is in mixins.

## Status

- [x] `controller_helpers.py` — Created
- [x] `human_tab.py` — Standalone module
- [x] `advisor_tab.py` — Standalone module  
- [x] `card_widget.py` — Standalone module
- [x] `colors.py` — Standalone module
- [x] `stats.py` — GameStats dataclass
- [ ] `controller_bidding.py` — To be extracted
- [ ] `controller_tricks.py` — To be extracted
- [ ] `game_tab.py` — To be extracted
- [ ] `stats_tab.py` — To be extracted
