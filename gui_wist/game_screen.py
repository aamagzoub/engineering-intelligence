"""
Game screen — the main playing table for the PyGame version.

Handles rendering the table, players, cards, and user interaction.
Integrates with the game engine for actual gameplay.

All 22 features implemented:
1. Trump hidden until first card
2. Dak ceremony (pass-based)
3. Card-based Dak display
4. Qabool rotation display
5. Confirm step in bidding (number → trump → confirm)
6. Pass-based Dak counter (max 2, 3rd forces play)
7. Bid display persistence
8. Game log panel (right side, 200px)
9. Trick winner highlight (gold border 1s)
10. Turn indicator (pulsing border)
11. Suit spacing in hand (15px gap)
12. Score breakdown per Shota in game log
13. Team labels on centre trick
14. Qabool/Shooter borders (gold/green)
15. Empty slot placeholders in centre (dashed rect)
16. Card dealing animation (lerp 30 frames)
17. Card play animation (slide to centre)
18. (Sound effects removed)
19. Bid number validation feedback
20. Responsive hand sizing (CARD_LARGE when ≤6 cards)
21. ESC during game (quit overlay Y/N)
22. Load AI model button (file dialog)
"""

import pygame
import math
import os
import random
from tkinter import Tk, filedialog

from gui_wist.constants import *
from gui_wist.card_renderer import create_card_surface, create_card_back

from agents.wist_rule_based.rule_based_agent import RuleBasedAgent
from environments.wist.environment import WistEnvironment
from environments.wist.actions import PlayCardAction
from environments.wist.round import Round
from environments.wist.rules import legal_cards, trick_winner, rank_value
from environments.wist.setup import create_standard_players
from environments.wist.tasmiya_engine import TasmiyaEngine, determine_first_shota_qabool, determine_first_shota_qabool_with_cards, determine_trump_suit
from environments.wist.trick import Trick
from intelligence.core.cards.card import Card
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit


SUIT_SYMBOLS = {Suit.SPADES: "♠", Suit.HEARTS: "♥", Suit.CLUBS: "♣", Suit.DIAMONDS: "♦"}
RANK_SYMBOLS = {r: s for r, s in zip(Rank, ["2","3","4","5","6","7","8","9","10","J","Q","K","A"])}
HUMAN_ID = 2  # Internal ID for human player.

# Display names.
DISPLAY_NAMES = {2: "Omer", 1: "Gaafar", 0: "Ibrahim", 3: "Musaab"}

# Stats panel width (right side, dedicated area).
STATS_PANEL_WIDTH = 200
# Table area width (minus stats panel).
TABLE_WIDTH = SCREEN_WIDTH - STATS_PANEL_WIDTH


def card_key(card: Card) -> tuple[str, str]:
    return RANK_SYMBOLS[card.rank], SUIT_SYMBOLS[card.suit]


class AnimatingCard:
    """A card animating from one position to another over N frames."""

    def __init__(self, surface: pygame.Surface, start_pos: tuple, end_pos: tuple, frames: int = 30,
                 start_scale: float = 1.2, end_scale: float = 1.0, delay: int = 0, pid: int | None = None):
        self.surface = surface
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.total_frames = frames
        self.frame = 0
        self.done = False
        self.start_scale = start_scale
        self.end_scale = end_scale
        self.delay = delay  # frames to wait before starting
        self.pid = pid  # Player ID this animation belongs to (for play animations).

    def update(self):
        if self.delay > 0:
            self.delay -= 1
            return
        self.frame += 1
        if self.frame >= self.total_frames:
            self.done = True

    @property
    def progress(self) -> float:
        if self.delay > 0:
            return 0.0
        t = min(1.0, self.frame / self.total_frames)
        # Ease-out quad.
        return 1 - (1 - t) ** 2

    @property
    def current_pos(self) -> tuple[float, float]:
        t = self.progress
        x = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * t
        y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * t
        return (x, y)

    @property
    def current_scale(self) -> float:
        t = self.progress
        return self.start_scale + (self.end_scale - self.start_scale) * t

    def render(self, screen: pygame.Surface):
        if self.delay > 0:
            return
        pos = self.current_pos
        scale = self.current_scale

        # Motion trail — draw 2 fading ghosts at previous positions.
        t = self.progress
        if 0.1 < t < 0.95:
            for trail_i, alpha in [(0.7, 40), (0.85, 70)]:
                trail_t = t * trail_i
                tx = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * trail_t
                ty = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * trail_t
                ghost = self.surface.copy()
                ghost.set_alpha(alpha)
                screen.blit(ghost, (tx, ty))

        if abs(scale - 1.0) > 0.01:
            w = int(self.surface.get_width() * scale)
            h = int(self.surface.get_height() * scale)
            scaled_surf = pygame.transform.smoothscale(self.surface, (w, h))
            offset_x = (w - self.surface.get_width()) // 2
            offset_y = (h - self.surface.get_height()) // 2
            screen.blit(scaled_surf, (pos[0] - offset_x, pos[1] - offset_y))
        else:
            screen.blit(self.surface, pos)


class GameScreen:
    """Manages the game table — rendering + interaction."""

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.fonts = {
            "title": pygame.font.SysFont("Segoe UI", 24, bold=True),
            "large": pygame.font.SysFont("Segoe UI", 16, bold=True),
            "medium": pygame.font.SysFont("Segoe UI", 13),
            "small": pygame.font.SysFont("Segoe UI", 10),
            "card": pygame.font.SysFont("Consolas", 11, bold=True),
            "log": pygame.font.SysFont("Consolas", 10),
            "chip": pygame.font.SysFont("Segoe UI", 8, bold=True),
            "chip_light": pygame.font.SysFont("Segoe UI", 8),
            "chip_label": pygame.font.SysFont("Segoe UI", 7),
        }

        # Card cache.
        self._card_cache: dict[str, pygame.Surface] = {}
        self._card_back = create_card_back(CARD_WIDTH, CARD_HEIGHT)
        self._card_back_mini = create_card_back(CARD_MINI_W, CARD_MINI_H)
        self._card_back_large = create_card_back(CARD_LARGE_W, CARD_LARGE_H)

        # Player rating/points (persistent JSON).
        self._player_points = 0
        self._player_stats_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "player_stats.json")
        self._load_player_stats()

        # Game state.
        self.phase = "idle"  # idle, dealing, bidding, playing, shota_end, game_over
        self.players = None
        self.round = None
        self.environment = None
        self.agents = None
        self.trump_suit = None
        self.qabool_id = 0
        self.shooter_id = 0
        self.bid_value = 0
        self.trick_number = 0
        self.team_tricks = [0, 0]
        self.game_scores = [0, 0]
        self.shota_number = 0

        # Feature 1: Trump hidden until first card of first trick.
        self._trump_revealed = False
        self._trump_flip_timer = 0  # Card flip animation (20 frames).

        # Button press feedback.
        self._button_press_timer = 0
        self._button_press_id = None  # Which button is pressed.

        # Feature 6: Pass-based Dak counter (max 2 per game).
        self._dak_count = 0

        # Feature 7: Bid display persistence.
        self._player_bids_display = {0: "", 1: "", 2: "", 3: ""}

        # Feature 8: Game log.
        self._game_log: list[str] = []
        self._log_scroll_offset = 0

        # Feature 9: Trick winner highlight.
        self._trick_winner_id: int | None = None
        self._trick_winner_timer = 0

        # Feature 10: Turn indicator (pulsing).
        self._pulse_frame = 0

        # Feature 16/17: Animations.
        self._deal_animations: list[AnimatingCard] = []
        self._play_animations: list[AnimatingCard] = []

        # Feature 21: ESC quit overlay.
        self._show_quit_overlay = False

        # Feature 22: Loaded AI model path.
        self._ai_model_path: str | None = None
        self._ai_advisor: object | None = None  # Loaded agent for recommendations.
        self._ai_advisor_type: str | None = None  # "learning" or "discovery"
        self._ai_gameplay_agent: object | None = None  # Loaded agent for actual gameplay.
        self._ai_recommendation: str = ""  # Current recommendation text.
        self._ai_rec_card: tuple | None = None  # (rank, suit) of recommended card.

        # Animation state.
        self._trick_played: dict[int, tuple[str, str]] = {}
        self._last_trick_cards: dict[int, tuple[str, str]] = {}  # Previous trick for peek.
        self._hover_card_idx = -1
        self._message = ""
        self._message_timer = 0

        # Bid chip pop-in animation (Feature: bid chip pop-in).
        self._bid_chip_anim_timer: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}

        # Player turn glow.
        self._active_turn_pid: int | None = None

        # Victory confetti.
        self._confetti_particles: list[dict] = []

        # Fan card data for hit detection.
        self._fan_card_data: list[dict] = []

        # Trick resolution state.
        self._pending_trick_winner: int | None = None
        self._pending_trick = None

        # Vignette cache.
        self._vignette_cache: pygame.Surface | None = None
        self._vignette_cache_key: tuple = (0, 0)

        # Restart signal for main app.
        self._restart_to_name = False
        self._shota_only_mode = False  # Endless shota mode.
        self._qabool_draw_done = False  # Only draw once per application run.

        # Stats panel animation state.
        self._score_pulse_timer = 0  # Frames remaining for score pulse.
        self._score_pulse_prev = [0, 0]  # Previous scores to detect change.
        self._stat_highlight_timers: dict[str, int] = {}  # key -> frames remaining.
        self._dak_shake_timer = 0  # Frames remaining for dak shake.
        self._last_trick_winner_team: int | None = None  # For momentum tracking.
        self._team_streak = [0, 0]  # Consecutive tricks per team.

        # Timing.
        self._ai_timer = 0
        self._play_order = []
        self._play_idx = 0

        # Bidding state.
        self._bid_step = "number"  # "number" → "trump" → "confirm"
        self._selected_bid: int | None = None
        self._selected_trump_idx: int | None = None

    # ----------------------------------------------------------
    # Player Points (persistent)
    # ----------------------------------------------------------

    def _load_player_stats(self):
        """Load player stats from JSON file."""
        import json
        try:
            if os.path.exists(self._player_stats_file):
                with open(self._player_stats_file, 'r') as f:
                    data = json.load(f)
                self._player_points = data.get("points", 0)
                self._player_games_played = data.get("games_played", 0)
                self._player_games_won = data.get("games_won", 0)
            else:
                self._player_points = 0
                self._player_games_played = 0
                self._player_games_won = 0
        except Exception:
            self._player_points = 0
            self._player_games_played = 0
            self._player_games_won = 0

    def _save_player_stats(self):
        """Save player stats to JSON file."""
        import json
        data = {
            "name": DISPLAY_NAMES.get(HUMAN_ID, "Player"),
            "points": self._player_points,
            "games_played": self._player_games_played,
            "games_won": self._player_games_won,
        }
        try:
            with open(self._player_stats_file, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass

    def _award_points(self, event: str, value: int = 0):
        """Award points based on game events."""
        pts = 0
        if event == "game_won":
            pts = 10
        elif event == "game_lost":
            pts = 2  # Participation.
        elif event == "bid_met":
            pts = 3
        elif event == "shooter_bid_met":
            pts = 5
        elif event == "seek":
            pts = 15
        elif event == "seek_against":
            pts = -5
        self._player_points += pts
        self._save_player_stats()

    def _load_ai_advisor(self):
        """Load the AI model file for recommendations. Supports LearningAgent and Discovery agent formats."""
        if not self._ai_model_path:
            self._ai_advisor = None
            self._ai_advisor_type = None
            return
        try:
            import json
            with open(self._ai_model_path, "r") as f:
                data = json.load(f)

            if "play_q" in data:
                # Discovery agent format.
                from agents.wist_discovery.discovery_agent import WistDiscoveryAgent
                agent = WistDiscoveryAgent(training=False)
                agent.load(self._ai_model_path)
                self._ai_advisor = agent
                self._ai_advisor_type = "discovery"
                self._ai_recommendation = f"Discovery model ({agent.episodes_trained} episodes)"
            else:
                # LearningAgent format.
                from agents.wist_learning.learning_agent import LearningAgent
                self._ai_advisor = LearningAgent.load(self._ai_model_path, training=False)
                self._ai_advisor_type = "learning"
                self._ai_recommendation = f"Model loaded ({self._ai_advisor.q_table_size} entries)"
        except Exception as e:
            self._ai_advisor = None
            self._ai_advisor_type = None
            self._ai_recommendation = f"Load failed: {str(e)[:30]}"

    def _get_ai_recommendation(self):
        """Query for a recommendation. Uses model if loaded, otherwise rule-based."""
        if not self.players:
            return

        try:
            if self.phase == "playing" and self._play_idx < 4:
                pid = self._play_order[self._play_idx] if self._play_idx < len(self._play_order) else -1
                if pid == HUMAN_ID:
                    obs = self.environment.observe(HUMAN_ID)

                    # Decide which agent to use.
                    if self._ai_advisor and getattr(self, '_ai_advisor_type', None) == "discovery":
                        # Discovery agent — call act() directly.
                        action = self._ai_advisor.act(obs)
                    elif self._ai_advisor and getattr(self, '_ai_advisor_type', None) == "learning":
                        from agents.wist_learning.learning_agent import encode_play_state
                        state = encode_play_state(obs, set())
                        has_data = any(v != 0 for v in self._ai_advisor.q_table.get(state, {}).values())
                        if has_data:
                            action = self._ai_advisor.act(obs)
                        else:
                            action = RuleBasedAgent().act(obs)
                    else:
                        action = RuleBasedAgent().act(obs)

                    if hasattr(action, 'card'):
                        card = action.card
                        r, s = card_key(card)
                        self._ai_rec_card = (r, s)
                        # Generate strategic teaching reason.
                        leading = obs.current_trick.leading_suit if obs.current_trick else None
                        n_played = len(obs.current_trick.played_cards) if obs.current_trick else 0
                        trump = obs.trump_suit
                        hand = obs.hand
                        trump_left = sum(1 for c in hand if c.suit == trump)

                        if obs.must_lead_trump:
                            if rank_value(card.rank) == 14:
                                reason = "Lead Ace of trump: forces opponents to spend high trumps early"
                            elif rank_value(card.rank) >= 12:
                                reason = "High trump lead: draws out their trumps so yours dominate later"
                            else:
                                reason = "Opening trump: reveals the suit and sets your team's tempo"
                        elif leading and card.suit == trump and card.suit != leading:
                            # Whipping.
                            if rank_value(card.rank) >= 13:
                                reason = "High trump whip: unbeatable, secures trick + demoralizes opponents"
                            else:
                                reason = "Trump whip: void in led suit, steal the trick cheaply"
                        elif leading and card.suit == leading:
                            # Following suit.
                            if rank_value(card.rank) == 14:
                                reason = "Ace wins guaranteed: no point saving it, take the trick now"
                            elif rank_value(card.rank) >= 12:
                                reason = "Play high to win: if you don't take it now, opponents will"
                            elif n_played == 3:
                                reason = "4th to play: you see all cards, play minimum needed to win"
                            else:
                                if trump_left >= 3:
                                    reason = "Play low: save trumps for whipping later tricks"
                                elif rank_value(card.rank) <= 6:
                                    reason = "Sacrifice low card: can't win, keep high cards for later"
                                else:
                                    reason = "Mid card: contest but don't overspend on uncertain trick"
                        elif not leading:
                            # Leading a new trick.
                            if card.suit == trump:
                                if trump_left >= 3:
                                    reason = "Lead trump: you hold many, force opponents to waste theirs"
                                else:
                                    reason = "Lead last trump: after this, your side suits are safe"
                            elif card.rank == Rank.ACE:
                                reason = "Lead Ace: guaranteed win without spending trump"
                            elif card.rank == Rank.KING:
                                reason = "Lead King: likely winner, tests if opponents still have Ace"
                            else:
                                suit_count = sum(1 for c in hand if c.suit == card.suit)
                                if suit_count >= 4:
                                    reason = "Lead long suit: opponents likely void, partner may trump"
                                elif suit_count == 1:
                                    reason = "Clear short suit: create void for future whipping"
                                else:
                                    reason = "Probe opponents: see who's void, plan future plays"
                        else:
                            reason = "Best available: balances risk vs reward"
                        self._ai_recommendation = reason
                    else:
                        self._ai_rec_card = None
                        self._ai_recommendation = "Thinking..."
                    return

            if self.phase == "bidding":
                is_human_turn = False
                if self._bid_index < len(self._bid_order) and self._bid_order[self._bid_index] == HUMAN_ID:
                    is_human_turn = True
                if self.qabool_id == HUMAN_ID and self._bid_index >= len(self._bid_order):
                    is_human_turn = True

                if is_human_turn:
                    from environments.wist.observation import BiddingObservation
                    from environments.wist.tasmiya_engine import determine_trump_suit
                    obs = BiddingObservation(
                        player_id=HUMAN_ID,
                        hand=list(self.players[HUMAN_ID].hand),
                        previous_bids=list(self._bid_history),
                        current_highest_bid=(self._bidding_engine.highest_bid.value
                                             if self._bidding_engine.highest_bid else None),
                        is_sahib_al_qabool=(self.qabool_id == HUMAN_ID and self._bid_index >= len(self._bid_order)),
                        is_opening_bid=(not self._has_opening_bid),
                    )

                    # Decide which agent to use.
                    if self._ai_advisor and getattr(self, '_ai_advisor_type', None) == "discovery":
                        action = self._ai_advisor.act(obs)
                    elif self._ai_advisor and getattr(self, '_ai_advisor_type', None) == "learning":
                        from agents.wist_learning.learning_agent import encode_bid_state
                        bid_state = encode_bid_state(obs)
                        has_bid_data = any(v != 0 for v in self._ai_advisor.bid_q.get(bid_state, {}).values())
                        if has_bid_data:
                            action = self._ai_advisor.act(obs)
                        else:
                            action = RuleBasedAgent().act(obs)
                    else:
                        action = RuleBasedAgent().act(obs)

                    from environments.wist.actions import BidAction
                    if isinstance(action, BidAction):
                        trump = determine_trump_suit(self.players[HUMAN_ID].hand)
                        suit_sym = SUIT_SYMBOLS.get(trump, "?")
                        self._ai_rec_card = None
                        self._ai_recommendation = f"Bid {action.value} with {suit_sym}"
                    else:
                        self._ai_rec_card = None
                        self._ai_recommendation = "Pass"
                    return

            self._ai_rec_card = None
        except Exception:
            self._ai_rec_card = None
            self._ai_recommendation = "Error in inference"

    def _get_rank_name(self) -> str:
        """Get military rank based on points."""
        ranks = [
            (8000, "Field Marshal"),
            (6000, "General"),
            (4500, "Lt. General"),
            (3500, "Major General"),
            (2700, "Brigadier"),
            (2100, "Colonel"),
            (1600, "Lt. Colonel"),
            (1200, "Major"),
            (900, "Captain"),
            (650, "1st Lieutenant"),
            (450, "2nd Lieutenant"),
            (300, "Warrant Officer"),
            (200, "Staff Sergeant"),
            (120, "Sergeant"),
            (60, "Corporal"),
            (25, "Private 1st Class"),
            (0, "Private"),
        ]
        for threshold, name in ranks:
            if self._player_points >= threshold:
                return name
        return "Private"

    def _get_rank_progress(self) -> float:
        """Get progress (0.0 to 1.0) toward the next rank."""
        ranks = [
            (8000, "Field Marshal"),
            (6000, "General"),
            (4500, "Lt. General"),
            (3500, "Major General"),
            (2700, "Brigadier"),
            (2100, "Colonel"),
            (1600, "Lt. Colonel"),
            (1200, "Major"),
            (900, "Captain"),
            (650, "1st Lieutenant"),
            (450, "2nd Lieutenant"),
            (300, "Warrant Officer"),
            (200, "Staff Sergeant"),
            (120, "Sergeant"),
            (60, "Corporal"),
            (25, "Private 1st Class"),
            (0, "Private"),
        ]
        pts = self._player_points
        for i, (threshold, _) in enumerate(ranks):
            if pts >= threshold:
                if i == 0:
                    return 1.0  # Max rank.
                next_threshold = ranks[i - 1][0]
                return (pts - threshold) / (next_threshold - threshold)
        return 0.0

    # ----------------------------------------------------------
    # Game lifecycle
    # ----------------------------------------------------------

    def start_game(self):
        """Start a new full game."""
        self.game_scores = [0, 0]
        self.shota_number = 0
        self._shota_scores = []  # List of (t1_score, t2_score) per shota.
        self._dak_count = 0
        self._is_first_deal = True  # Only True for the very first deal of the game.
        self._game_log = []
        self._log_game_event("=== NEW GAME ===")
        self._show_quit_overlay = False

        # Load AI advisor if model path is set.
        self._load_ai_advisor()

        # Live game stats.
        self._game_stats = {
            "shotas_played": 0,
            "seeks": 0,
            "daks": 0,
            "bids_met": 0,
            "player_tricks": {0: 0, 1: 0, 2: 0, 3: 0},
            "player_bids_won": {0: 0, 1: 0, 2: 0, 3: 0},
            "highest_bid": 0,
            "highest_bidder": None,
            "total_tricks_played": 0,
        }

        self._start_new_shota()

    def _log_game_event(self, text: str):
        """Add an entry to the game log (Feature 8)."""
        self._game_log.append(text)
        # Keep max 200 lines.
        if len(self._game_log) > 200:
            self._game_log = self._game_log[-200:]

    def _start_new_shota(self):
        """Deal and start bidding for a NEW shota (increments number + rotates Qabool)."""
        self.shota_number += 1
        # Qabool draw only on the very first run of the application.
        if self.shota_number == 1 and not self._qabool_draw_done:
            self._qabool_draw_done = True
            self.qabool_id, self._qabool_draw_cards = determine_first_shota_qabool_with_cards()
            # Show the Qabool draw phase — interactive.
            self.phase = "qabool_draw"
            self._qd_step = "picking"  # Skip intro, go straight to picking.
            self._qd_picked = {}  # pid → card index in the row
            self._qd_flip_timers = {}  # pid → frames remaining for flip anim
            self._qd_pick_order = [HUMAN_ID, 0, 3, 1]  # Human picks first.
            self._qd_pick_idx = 0  # Index into pick_order.
            self._qd_ai_timer = 0  # Delay between AI picks.
            self._qd_num_cards = 52  # Full deck displayed in the row.
            # Pre-assign which row index each player will pick.
            import random as _rng
            available_indices = list(range(self._qd_num_cards))
            _rng.shuffle(available_indices)
            self._qd_assigned_indices = {}
            for i, pid in enumerate(self._qd_pick_order):
                self._qd_assigned_indices[pid] = available_indices[i]
            # Human's visual slot (updated when they click).
            self._qd_human_visual_slot = None
            self._message = ""
            self._message_timer = 0
            self._log_game_event("=== QABOOL DRAW ===")
            for i, card in enumerate(self._qabool_draw_cards):
                r, s = card_key(card)
                self._log_game_event(f"  {DISPLAY_NAMES[i]} draws {r}{s}")
            self._log_game_event(f"  Qabool: {DISPLAY_NAMES[self.qabool_id]}")
            return
        else:
            self.qabool_id = (self.qabool_id + 1) % 4
        self._start_shota_common()

    def _redeal_after_dak(self):
        """Re-deal after pass-based Dak — Qabool moves to next player.
        From 2nd shota onward, Dak counts as a shota."""
        # Pass-based Dak: Qabool rotates.
        self.qabool_id = (self.qabool_id + 1) % 4
        # From 2nd shota onward, Dak counts as one of the 5.
        if self.shota_number >= 2:
            self.shota_number += 1
        self._start_shota_common()

    def _start_shota_common(self):
        """Common shota setup (shared between new shota and re-deal)."""
        self.trick_number = 0
        self.team_tricks = [0, 0]
        self.bid_value = 0
        self.shooter_id = 0
        self._trick_played = {}
        self._last_trick_cards = {}  # Clear last trick from previous shota.
        self._trump_revealed = False
        self._trick_winner_id = None
        self._trick_winner_timer = 0
        self._player_bids_display = {0: "", 1: "", 2: "", 3: ""}

        self.players = create_standard_players()
        self.round = Round(self.players)
        self.round.deal()

        # Feature 3: Card-based Dak detection with display.
        dak_player_id = self.round.first_card_based_dak_player_id()
        if dak_player_id is not None:
            # Capture proof before re-dealing.
            from environments.wist.dak import has_picture_card, has_eight_or_more_in_one_suit
            dak_hand = list(self.players[dak_player_id].hand)
            # Determine the reason.
            if has_eight_or_more_in_one_suit(dak_hand):
                # Find the suit with 8+ cards.
                from collections import Counter as _Ctr
                suit_counts = _Ctr(c.suit for c in dak_hand)
                dak_suit = max(suit_counts, key=suit_counts.get)
                proof_cards = [c for c in dak_hand if c.suit == dak_suit]
                dak_reason = "8_plus"
            else:
                # No picture cards.
                proof_cards = dak_hand
                dak_reason = "no_pictures"
                dak_suit = None

            # Store for the notice screen.
            self._card_dak_info = {
                "player_id": dak_player_id,
                "reason": dak_reason,
                "proof_cards": proof_cards,
                "suit": dak_suit,
            }
            self.phase = "card_dak_notice"
            self._log_game_event(
                f"Card Dak! {DISPLAY_NAMES[dak_player_id]} — "
                f"{'8+ in one suit' if dak_reason == '8_plus' else 'no picture cards'}")
            return

        self._finish_shota_setup()

    def _finish_shota_setup(self):
        """Complete shota setup after card-based Dak is resolved (or skipped)."""
        # Use trained LearningAgent if available, fall back to RuleBasedAgent.
        ai_agent = self._get_gameplay_agent()
        self.agents = [ai_agent, ai_agent, None, ai_agent]

        self._log_game_event(f"--- Shota {self.shota_number} ---")
        self._log_game_event(f"Qabool: {DISPLAY_NAMES[self.qabool_id]}")

        self._message = f"Qabool: {DISPLAY_NAMES[self.qabool_id]}"
        self._message_timer = 60

        # Feature 16: Deal animation.
        self._start_deal_animation()

        self.phase = "dealing"
        self._ai_timer = 35

    def _get_gameplay_agent(self):
        """
        Get the best available AI agent for gameplay.
        Priority: user-loaded model > trained LearningAgent > Discovery agent > RuleBasedAgent.
        """
        if self._ai_gameplay_agent is not None:
            return self._ai_gameplay_agent

        # If user loaded a model via dialog, use it for gameplay too.
        if self._ai_advisor is not None:
            self._ai_gameplay_agent = self._ai_advisor
            return self._ai_advisor

        from pathlib import Path

        # Try to load a trained LearningAgent model.
        model_path = Path("agents/wist_learning/wist_model.json")
        if model_path.exists():
            try:
                from agents.wist_learning.learning_agent import LearningAgent
                agent = LearningAgent.load(model_path, training=False)
                # Only use if it has meaningful training (at least 1000 episodes).
                if agent.episodes_trained >= 1000:
                    self._ai_gameplay_agent = agent
                    self._log_game_event(
                        f"AI: Learning Agent ({agent.episodes_trained} episodes trained)")
                    return agent
            except Exception:
                pass

        # Try to load a Discovery agent model.
        discovery_path = Path("agents/wist_discovery/wist_discovery_model.json")
        if discovery_path.exists():
            try:
                from agents.wist_discovery.discovery_agent import WistDiscoveryAgent
                agent = WistDiscoveryAgent(training=False)
                agent.load(str(discovery_path))
                if agent.episodes_trained >= 1000:
                    self._ai_gameplay_agent = agent
                    self._log_game_event(
                        f"AI: Discovery Agent ({agent.episodes_trained} episodes trained)")
                    return agent
            except Exception:
                pass

        # Fallback to rule-based.
        return RuleBasedAgent()

    def _handle_card_dak_continue(self):
        """After user acknowledges card-based Dak, re-deal and check again."""
        # From 2nd shota onward, card-based Dak counts as a shota.
        if self.shota_number >= 2:
            self.shota_number += 1

        # Re-deal and check again.
        self.players = create_standard_players()
        self.round = Round(self.players)
        self.round.deal()

        dak_player_id = self.round.first_card_based_dak_player_id()
        if dak_player_id is not None:
            from environments.wist.dak import has_picture_card, has_eight_or_more_in_one_suit
            from collections import Counter as _Ctr
            dak_hand = list(self.players[dak_player_id].hand)
            if has_eight_or_more_in_one_suit(dak_hand):
                suit_counts = _Ctr(c.suit for c in dak_hand)
                dak_suit = max(suit_counts, key=suit_counts.get)
                proof_cards = [c for c in dak_hand if c.suit == dak_suit]
                dak_reason = "8_plus"
            else:
                proof_cards = dak_hand
                dak_reason = "no_pictures"
                dak_suit = None
            self._card_dak_info = {
                "player_id": dak_player_id,
                "reason": dak_reason,
                "proof_cards": proof_cards,
                "suit": dak_suit,
            }
            self.phase = "card_dak_notice"
            self._log_game_event(
                f"Card Dak again! {DISPLAY_NAMES[dak_player_id]}")
            return

        self._finish_shota_setup()

    def _start_deal_animation(self):
        """Animate cards from dealer (right of Qabool) to all players."""
        self._deal_animations = []
        cx, cy = TABLE_WIDTH // 2, SCREEN_HEIGHT // 2

        # Dealer = player to the right of Sahib Al-Qabool.
        dealer_id = (self.qabool_id + 3) % 4
        dealer_positions = {
            0: (cx, 80),
            3: (50, cy),
            1: (TABLE_WIDTH - 100, cy),
            2: (cx, SCREEN_HEIGHT - 140),
        }
        deck_x, deck_y = dealer_positions.get(dealer_id, (cx, cy))

        # Human hand target area — fan arc positions matching _render_human_hand.
        n = 13
        card_w = CARD_WIDTH
        card_h = CARD_HEIGHT

        # Fan parameters (same as _render_human_hand).
        max_spread = 30.0
        fan_spread = min(max_spread, n * 2.5)
        half_spread = fan_spread / 2.0
        fan_cx = TABLE_WIDTH // 2
        fan_radius = 480
        base_y = SCREEN_HEIGHT - card_h - 40

        surf = self._card_back_mini
        for i in range(n):
            if n == 1:
                angle_deg = 0.0
            else:
                angle_deg = -half_spread + (i / (n - 1)) * fan_spread
            angle_rad = math.radians(angle_deg)
            end_x = fan_cx + fan_radius * math.sin(angle_rad) - card_w // 2
            end_y = base_y
            anim = AnimatingCard(surf, (deck_x, deck_y), (end_x, end_y),
                                 frames=25, start_scale=1.0, end_scale=1.0, delay=i * 2)
            self._deal_animations.append(anim)

        # Also animate cards to each opponent.
        target_positions = {
            0: (cx, 80),
            3: (50, cy),
            1: (TABLE_WIDTH - 100, cy),
        }
        for pid in (0, 3, 1):
            end_x, end_y = target_positions[pid]
            for j in range(3):
                end = (end_x + j * 15, end_y)
                anim = AnimatingCard(surf, (deck_x, deck_y), end, frames=20,
                                     start_scale=1.0, end_scale=1.0, delay=j * 3 + 5)
                self._deal_animations.append(anim)

    def _run_bidding(self):
        """Run bidding step by step — AI bids instantly, human gets UI."""
        from environments.wist.tasmiya_engine import tasmiya_order
        from environments.wist.bidding_engine import BiddingEngine

        self._bidding_engine = BiddingEngine()
        self._bid_history = []
        self._bid_order = tasmiya_order(self.qabool_id)
        self._bid_index = 0
        self._has_opening_bid = False
        self._human_trump_choice = None
        self._bidding_done = False
        self._bid_step = "number"
        self._selected_bid = None
        self._selected_trump_idx = None

        self.phase = "bidding"
        self._ai_timer = 30

    def _update_bidding(self):
        """Process bidding one player at a time."""
        if self._bidding_done:
            return

        if self._ai_timer > 0:
            self._ai_timer -= 1
            return

        # All regular players done → Qabool decides.
        if self._bid_index >= len(self._bid_order):
            self._bid_qabool()
            return

        pid = self._bid_order[self._bid_index]

        if pid == HUMAN_ID:
            # Wait for human — handled in _handle_click during bidding.
            pass
        else:
            # AI bids.
            from environments.wist.observation import BiddingObservation
            from environments.wist.bidding import Bid, Pass
            from environments.wist.actions import BidAction, PassAction

            obs = BiddingObservation(
                player_id=pid,
                hand=list(self.players[pid].hand),
                previous_bids=list(self._bid_history),
                current_highest_bid=(self._bidding_engine.highest_bid.value
                                     if self._bidding_engine.highest_bid else None),
                is_sahib_al_qabool=False,
                is_opening_bid=(not self._has_opening_bid),
            )
            action = self.agents[pid].act(obs)

            if isinstance(action, BidAction):
                bid = Bid(player_id=pid, value=action.value)
                self._bidding_engine.apply_bid(bid)
                self._bid_history.append((pid, action.value))
                self._has_opening_bid = True
                self._player_bids_display[pid] = f"Bid {action.value}"
                self._bid_chip_anim_timer[pid] = 15
                self._message = f"{DISPLAY_NAMES[pid]} bids {action.value}"
                self._log_game_event(f"{DISPLAY_NAMES[pid]} bids {action.value}")
            else:
                self._bidding_engine.apply_pass(Pass(player_id=pid))
                self._bid_history.append((pid, None))
                self._player_bids_display[pid] = "Pass"
                self._bid_chip_anim_timer[pid] = 15
                self._message = f"{DISPLAY_NAMES[pid]} passes"
                self._log_game_event(f"{DISPLAY_NAMES[pid]} passes")

            self._message_timer = 60
            self._bid_index += 1

            # Bid of 13 stops bidding — skip remaining players, go to Qabool.
            if isinstance(action, BidAction) and action.value == 13:
                self._bid_index = len(self._bid_order)

            # First-deal-only special rule: if all 3 regular players pass,
            # it's automatic Dak — Qabool has NO say (even human).
            # Only applies on the very first deal of the entire game.
            if (self._is_first_deal and self._bid_index == 3
                    and not self._has_opening_bid
                    and self._bidding_engine.highest_bid is None):
                passes_count = sum(1 for _, v in self._bid_history if v is None)
                if passes_count >= 2 and self._bid_index == 3:
                    self._is_first_deal = False  # Never triggers again.
                    self._dak_count += 1
                    if hasattr(self, '_game_stats'):
                        self._game_stats["daks"] = self._dak_count
                    self._dak_shake_timer = 30
                    self._log_game_event("Auto-DAK! 3rd player declared in first shota.")

                    if self.qabool_id == HUMAN_ID:
                        # Human is Qabool — show explanation, wait for click.
                        self.phase = "auto_dak_notice"
                        self._message = ""
                        self._message_timer = 0
                    else:
                        # AI Qabool — also show notice so user sees what happened.
                        self.phase = "auto_dak_notice"
                        self._message = ""
                        self._message_timer = 0
                    return

            self._ai_timer = 30

    def _bid_qabool(self):
        """Sahib Al-Qabool decides."""
        from environments.wist.observation import BiddingObservation
        from environments.wist.actions import BidAction, PassAction
        from environments.wist.bidding import Bid, Pass

        qid = self.qabool_id

        # Feature 6: If dak_count >= 2, force Qabool to play on 3rd.
        force_play = (self._dak_count >= 2)

        if qid == HUMAN_ID:
            if force_play and self._bidding_engine.highest_bid is None:
                # Human is forced to bid (3rd Dak).
                self._message = "3rd Dak! YOU must bid (forced)."
                self._message_timer = 999
            else:
                self._message = "YOU are Qabool! Select bid + trump + confirm."
                self._message_timer = 999
            return

        obs = BiddingObservation(
            player_id=qid, hand=list(self.players[qid].hand),
            previous_bids=list(self._bid_history),
            current_highest_bid=(self._bidding_engine.highest_bid.value
                                 if self._bidding_engine.highest_bid else None),
            is_sahib_al_qabool=True,
            is_opening_bid=(not self._has_opening_bid),
        )
        action = self.agents[qid].act(obs)

        from environments.wist.actions import BidAction as BA
        if isinstance(action, BA) or force_play:
            if isinstance(action, BA):
                bid_value = action.value
            else:
                # Force a minimum bid — must match or exceed current highest.
                current = (self._bidding_engine.highest_bid.value
                           if self._bidding_engine.highest_bid else 0)
                bid_value = max(7, current)
            bid = Bid(player_id=qid, value=bid_value)
            try:
                self._bidding_engine.apply_bid(bid, is_sahib_al_qabool=True)
            except ValueError:
                # If bid still fails, just accept.
                self._bidding_engine.apply_pass(Pass(player_id=qid))
                self._bid_history.append((qid, None))
                self._player_bids_display[qid] = "Accepts"
                self._bid_chip_anim_timer[qid] = 15
                self._log_game_event(f"{DISPLAY_NAMES[qid]} (Qabool) accepts (forced)")
                self._finalize_bidding()
                return
            self._bid_history.append((qid, bid_value))
            self._player_bids_display[qid] = f"Bid {bid_value}"
            self._bid_chip_anim_timer[qid] = 15
            self._log_game_event(f"{DISPLAY_NAMES[qid]} (Qabool) bids {bid_value}")
        else:
            self._bidding_engine.apply_pass(Pass(player_id=qid))
            self._bid_history.append((qid, None))
            if self._bidding_engine.highest_bid is None:
                # Feature 2: Dak ceremony — show announcement.
                self._dak_count += 1
                if hasattr(self, '_game_stats'):
                    self._game_stats["daks"] = self._dak_count
                self._dak_shake_timer = 30
                self._log_game_event(f"DAK #{self._dak_count}! All passed.")
                self.phase = "pass_dak_notice"
                self._pass_dak_declarer = qid
                return
            self._player_bids_display[qid] = "Accepts"
            self._bid_chip_anim_timer[qid] = 15
            self._log_game_event(f"{DISPLAY_NAMES[qid]} (Qabool) accepts")

        self._finalize_bidding()

    def _human_bid_with_number_and_suit(self):
        """Human selected bid number + suit + confirmed."""
        suits = [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]
        chosen_suit = suits[self._selected_trump_idx]
        bid_value = self._selected_bid
        trump_count = sum(1 for c in self.players[HUMAN_ID].hand if c.suit == chosen_suit)

        from environments.wist.bidding import Bid
        is_qabool = (self.qabool_id == HUMAN_ID and self._bid_index >= len(self._bid_order))
        someone_bid = (self._bidding_engine.highest_bid is not None)

        # Validate: bid must be >= trump_count + 3.
        # Qabool advantage: bid >= trump_count + 2 (ONLY when someone else bid — matching/outbidding).
        # When all passed, standard formula applies (no advantage).
        if is_qabool and someone_bid:
            min_bid = trump_count + 2  # Qabool extra card advantage (matching).
        else:
            min_bid = trump_count + 3  # Standard formula.

        if bid_value < max(7, min_bid):
            self._log_game_event(f"  Invalid: Bid {bid_value} < min {max(7, min_bid)} for {trump_count} cards")
            self._bid_step = "number"
            self._selected_bid = None
            self._selected_trump_idx = None
            return

        # Validate: trump can't have 8+ cards.
        if trump_count >= 8:
            self._bid_step = "number"
            self._selected_bid = None
            self._selected_trump_idx = None
            return

        try:
            bid = Bid(player_id=HUMAN_ID, value=bid_value)
            self._bidding_engine.apply_bid(bid, is_sahib_al_qabool=is_qabool)
        except ValueError as e:
            self._log_game_event(f"  Bid error: {e}")
            self._bid_step = "number"
            self._selected_bid = None
            self._selected_trump_idx = None
            return

        self._bid_history.append((HUMAN_ID, bid_value))
        self._has_opening_bid = True
        self._human_trump_choice = chosen_suit
        self._player_bids_display[HUMAN_ID] = f"Bid {bid_value}"
        self._bid_chip_anim_timer[HUMAN_ID] = 15
        self._log_game_event(f"You bid {bid_value} ({SUIT_SYMBOLS[chosen_suit]})")

        # Clear recommendation after bidding.
        self._ai_rec_card = None
        self._ai_recommendation = ""

        # Reset bid step.
        self._bid_step = "number"
        self._selected_bid = None
        self._selected_trump_idx = None

        if is_qabool:
            self._finalize_bidding()
        else:
            self._bid_index += 1
            # Bid of 13 stops bidding — skip to Qabool.
            if bid_value == 13:
                self._bid_index = len(self._bid_order)
            self._ai_timer = 60

    def _human_pass_action(self):
        """Human passes."""
        from environments.wist.bidding import Pass
        is_qabool = (self.qabool_id == HUMAN_ID and self._bid_index >= len(self._bid_order))

        # Feature 6: Block pass if 3rd Dak forced.
        if is_qabool and self._dak_count >= 2 and self._bidding_engine.highest_bid is None:
            self._message = "3rd Dak! You MUST bid."
            self._message_timer = 50
            return

        self._bidding_engine.apply_pass(Pass(player_id=HUMAN_ID))
        self._bid_history.append((HUMAN_ID, None))
        self._player_bids_display[HUMAN_ID] = "Pass"
        self._bid_chip_anim_timer[HUMAN_ID] = 15
        self._log_game_event("You pass")

        # Clear recommendation after passing.
        self._ai_rec_card = None
        self._ai_recommendation = ""

        if is_qabool:
            if self._bidding_engine.highest_bid is None:
                # Feature 2: Dak ceremony — show announcement.
                self._dak_count += 1
                if hasattr(self, '_game_stats'):
                    self._game_stats["daks"] = self._dak_count
                self._dak_shake_timer = 30
                self._log_game_event(f"DAK #{self._dak_count}!")
                self.phase = "pass_dak_notice"
                self._pass_dak_declarer = HUMAN_ID
                return
            self._player_bids_display[HUMAN_ID] = "Accepts"
            self._bid_chip_anim_timer[HUMAN_ID] = 15
            self._log_game_event("You accept as Qabool")
            self._finalize_bidding()
        else:
            self._bid_step = "number"
            self._selected_bid = None
            self._selected_trump_idx = None
            self._bid_index += 1
            self._ai_timer = 60

    def _finalize_bidding(self):
        """Bidding resolved — set up for play."""
        self._is_first_deal = False  # First deal is over once bidding resolves.
        winning_bid = self._bidding_engine.highest_bid
        if winning_bid is None:
            # Dak — re-deal same shota.
            self._redeal_after_dak()
            return

        self.shooter_id = winning_bid.player_id
        self.bid_value = winning_bid.value
        self._bidding_done = True

        # Track bid stats.
        if hasattr(self, '_game_stats'):
            self._game_stats["player_bids_won"][self.shooter_id] = self._game_stats["player_bids_won"].get(self.shooter_id, 0) + 1
            if self.bid_value > self._game_stats.get("highest_bid", 0):
                self._game_stats["highest_bid"] = self.bid_value
                self._game_stats["highest_bidder"] = self.shooter_id

        if self.shooter_id == HUMAN_ID and self._human_trump_choice:
            self.trump_suit = self._human_trump_choice
        else:
            self.trump_suit = determine_trump_suit(self.players[self.shooter_id].hand)

        self.round.state.trump_suit = self.trump_suit
        self.round.state.winning_bidder_id = self.shooter_id
        self.round.next_leading_player_id = self.shooter_id
        self.environment = WistEnvironment(self.round.state)

        # Feature 1: Trump stays hidden until first card played.
        self._trump_revealed = False

        self._message = f"{DISPLAY_NAMES[self.shooter_id]} shoots! Bid: {self.bid_value}"
        self._message_timer = 60
        self._ai_timer = 60
        self._log_game_event(f"Shooter: {DISPLAY_NAMES[self.shooter_id]}, Bid: {self.bid_value}")

        self.phase = "playing"
        # Clear bid chips now that bidding is done.
        self._player_bids_display = {0: "", 1: "", 2: "", 3: ""}
        # Use frame-based timer instead of OS timer.
        self._play_idx = 99
        self._ai_timer = 30  # Then _start_next_trick fires.

    def _start_next_trick(self):
        """Start a new trick."""
        if self.trick_number >= 13:
            self._end_shota()
            return

        self.trick_number += 1
        self._trick_played = {}
        self._trick_winner_id = None
        self._trick_winner_timer = 0

        leader = self.round.next_leading_player_id
        self.round.state.current_trick = Trick(leading_player_id=leader)
        self._play_order = [(leader + i) % 4 for i in range(4)]
        self._play_idx = 0
        self._ai_timer = 15

    def _end_shota(self):
        """Score the Shota and start next or end game."""
        from environments.wist.scoring import score_shota, detect_seek

        playing_team = self.players[self.shooter_id].team_id
        defending = 1 if playing_team == 0 else 0
        total = self.team_tricks[0] + self.team_tricks[1]

        # Seek detection.
        self._seek_team = None
        team_tricks_dict = {0: self.team_tricks[0], 1: self.team_tricks[1]}
        seek_team = detect_seek(team_tricks_dict)
        if seek_team is not None:
            self._seek_team = seek_team
            self._log_game_event(f"  *** SEEK! Team {seek_team + 1} won all 13! ***")

        shota_score_t1 = 0
        shota_score_t2 = 0
        try:
            if total == 13:
                score_delta = score_shota(
                    playing_team_id=playing_team, defending_team_id=defending,
                    bid=self.bid_value,
                    playing_team_tricks=self.team_tricks[playing_team],
                    defending_team_tricks=self.team_tricks[defending])
                shota_score_t1 = score_delta.get(0, 0)
                shota_score_t2 = score_delta.get(1, 0)
                self.game_scores[0] += shota_score_t1
                self.game_scores[1] += shota_score_t2
            else:
                shota_score_t1 = self.team_tricks[0]
                shota_score_t2 = self.team_tricks[1]
                self.game_scores[0] += shota_score_t1
                self.game_scores[1] += shota_score_t2
        except Exception:
            pass

        # Trigger score pulse animation.
        self._score_pulse_timer = 40

        # Record per-shota scores AND tricks for the scoreboard.
        self._shota_scores.append({
            "score": (shota_score_t1, shota_score_t2),
            "tricks": (self.team_tricks[0], self.team_tricks[1]),
            "bid": self.bid_value,
            "playing_team": playing_team,
            "bid_met": self.team_tricks[playing_team] >= self.bid_value,
        })

        # Track stats.
        if not hasattr(self, '_game_stats'):
            self._game_stats = {"shotas_played": 0, "seeks": 0, "daks": 0, "bids_met": 0}
        self._game_stats["shotas_played"] += 1
        if seek_team is not None:
            self._game_stats["seeks"] += 1
            self._stat_highlight_timers["seeks"] = 40
        bid_met = self.team_tricks[playing_team] >= self.bid_value
        if bid_met:
            self._game_stats["bids_met"] += 1
            if playing_team == 0:
                self._game_stats["bids_met_t1"] = self._game_stats.get("bids_met_t1", 0) + 1
            else:
                self._game_stats["bids_met_t2"] = self._game_stats.get("bids_met_t2", 0) + 1
            self._stat_highlight_timers["bids_met"] = 30

        # Award player points.
        human_team = 0  # Human is on team 0 (players 0, 2).
        if bid_met and playing_team == human_team:
            self._award_points("bid_met")
            if self.shooter_id == HUMAN_ID:
                self._award_points("shooter_bid_met")

        # Score breakdown in game log.
        result_str = "SUCCESS" if bid_met else "FAILED"
        self._log_game_event(f"Shota {self.shota_number}: Bid {self.bid_value} → {result_str}")
        self._log_game_event(f"  Tricks: T1={self.team_tricks[0]} T2={self.team_tricks[1]}")
        self._log_game_event(f"  Score: T1 +{shota_score_t1}, T2 +{shota_score_t2}")
        self._log_game_event(f"  Total: T1={self.game_scores[0]} T2={self.game_scores[1]}")

        # Seek = instant game over regardless of score or shota count.
        # Seek = instant game over always.
        if seek_team is not None:
            self._player_games_played += 1
            human_team = 0
            if seek_team == human_team:
                self._award_points("seek")
                self._award_points("game_won")
                self._player_games_won += 1
            else:
                self._award_points("seek_against")
                self._award_points("game_lost")
            self._save_player_stats()
            self.phase = "game_over"
            self._log_game_event("=== GAME OVER (SEEK) ===")
            self._spawn_confetti()
            return

        if not self._shota_only_mode and (self.shota_number >= 5 or self.game_scores[0] >= 25 or self.game_scores[1] >= 25):
            # Award game-level points.
            self._player_games_played += 1
            if self.game_scores[0] > self.game_scores[1]:
                self._award_points("game_won")
                self._player_games_won += 1
            else:
                self._award_points("game_lost")
            self._save_player_stats()

            self.phase = "game_over"
            self._log_game_event("=== GAME OVER ===")
            self._spawn_confetti()
        else:
            self.phase = "shota_end"
            self._ai_timer = 120

    # ----------------------------------------------------------
    # Update (called each frame)
    # ----------------------------------------------------------

    def update(self):
        """Update game logic each frame."""
        if self._message_timer > 0:
            self._message_timer -= 1

        # AI recommendation (query every 30 frames).
        if self._pulse_frame % 30 == 0:
            self._get_ai_recommendation()

        # Feature 10: Pulse animation.
        self._pulse_frame = (self._pulse_frame + 1) % 60

        # Stats panel animation timers.
        if self._score_pulse_timer > 0:
            self._score_pulse_timer -= 1
        if self._dak_shake_timer > 0:
            self._dak_shake_timer -= 1
        for key in list(self._stat_highlight_timers.keys()):
            if self._stat_highlight_timers[key] > 0:
                self._stat_highlight_timers[key] -= 1
            else:
                del self._stat_highlight_timers[key]

        # Update animations.
        for anim in self._deal_animations:
            anim.update()
        self._deal_animations = [a for a in self._deal_animations if not a.done]

        for anim in self._play_animations:
            anim.update()
        self._play_animations = [a for a in self._play_animations if not a.done]

        # Feature 9: Trick winner timer.
        if self._trick_winner_timer > 0:
            self._trick_winner_timer -= 1
            if self._trick_winner_timer <= 0:
                self._trick_winner_id = None

        # Bid chip pop-in timers.
        for pid in list(self._bid_chip_anim_timer.keys()):
            if self._bid_chip_anim_timer[pid] > 0:
                self._bid_chip_anim_timer[pid] -= 1

        # Trump flip animation timer.
        if self._trump_flip_timer > 0:
            self._trump_flip_timer -= 1

        # Button press feedback timer.
        if self._button_press_timer > 0:
            self._button_press_timer -= 1
            if self._button_press_timer <= 0:
                self._button_press_id = None

        # Victory confetti update.
        for p in self._confetti_particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.15  # gravity
            p["life"] -= 1
            p["alpha"] = max(0, int(255 * (p["life"] / p["max_life"])))
        self._confetti_particles = [p for p in self._confetti_particles if p["life"] > 0]

        if self.phase == "qabool_draw":
            # Handle AI picking with delays.
            if self._qd_step == "picking":
                # Update flip animations.
                for pid in list(self._qd_flip_timers.keys()):
                    if self._qd_flip_timers[pid] > 0:
                        self._qd_flip_timers[pid] -= 1

                # AI auto-pick logic.
                if self._qd_pick_idx < len(self._qd_pick_order):
                    current_pid = self._qd_pick_order[self._qd_pick_idx]
                    if current_pid != HUMAN_ID:
                        self._qd_ai_timer -= 1
                        if self._qd_ai_timer <= 0:
                            # AI picks their assigned card.
                            idx = self._qd_assigned_indices[current_pid]
                            self._qd_picked[current_pid] = idx
                            self._qd_flip_timers[current_pid] = 20  # Flip anim frames.
                            self._qd_pick_idx += 1
                            self._qd_ai_timer = 40  # Delay before next AI pick.
                else:
                    # All picks done — wait for flips to finish, then show result.
                    all_flipped = all(
                        self._qd_flip_timers.get(pid, 0) <= 0
                        for pid in self._qd_pick_order)
                    if all_flipped:
                        self._qd_step = "result"
            elif self._qd_step == "flipping":
                # Update flip animations.
                for pid in list(self._qd_flip_timers.keys()):
                    if self._qd_flip_timers[pid] > 0:
                        self._qd_flip_timers[pid] -= 1
        elif self.phase == "dealing":
            self._ai_timer -= 1
            if self._ai_timer <= 0:
                self._run_bidding()
        elif self.phase == "bidding":
            self._update_bidding()
            # After bidding done and Dak, re-deal same shota.
            if self._bidding_done and self._bidding_engine.highest_bid is None:
                self._ai_timer -= 1
                if self._ai_timer <= 0:
                    self._redeal_after_dak()
        elif self.phase == "playing":
            self._update_playing()
        elif self.phase == "shota_end":
            pass  # Wait for player to click "Next Shota" button.

    def _update_playing(self):
        """Handle AI turns and timing."""
        if self._ai_timer > 0:
            self._ai_timer -= 1
            return

        # State 99 = waiting to start next trick (not a real play index).
        if self._play_idx == 99:
            self._active_turn_pid = None
            self._start_next_trick()
            return

        # State 98 = highlight pause done, now collect cards to winner.
        if self._play_idx == 98:
            self._collect_trick_to_winner()
            return

        if self._play_idx >= 4:
            # Trick complete — show highlight then resolve.
            self._active_turn_pid = None
            self._start_trick_highlight()
            return

        # Safety: if no play order or index out of bounds, skip to resolve.
        if not self._play_order or self._play_idx >= len(self._play_order):
            self._play_idx = 4
            return

        pid = self._play_order[self._play_idx]

        # Set active turn for glow effect.
        if pid not in self._trick_played:
            self._active_turn_pid = pid
        else:
            self._active_turn_pid = None

        if pid == HUMAN_ID:
            # Safety checks — if state is broken, skip human turn.
            if self.round.state.current_trick is None:
                self._play_idx += 1
                self._ai_timer = 5
                return
            if not self.players[HUMAN_ID].hand:
                self._play_idx += 1
                self._ai_timer = 5
                return
            # Show "your turn" message so player knows game isn't frozen.
            if self._message_timer <= 0:
                self._message = "Your turn — click a card"
                self._message_timer = 30
            # Otherwise wait for click.
        else:
            # AI plays.
            try:
                obs = self.environment.observe(pid)
                action = self.agents[pid].act(obs)
                self.environment.apply_action(action)
                r, s = card_key(action.card)
                self._trick_played[pid] = (r, s)

                # Notify learning agent of card played (maintains card memory).
                if hasattr(self.agents[pid], 'observe_card_played'):
                    self.agents[pid].observe_card_played(action.card)

                # Feature 1: Reveal trump on first card of first trick.
                if not self._trump_revealed:
                    self._trump_revealed = True
                    self._trump_flip_timer = 20  # Start flip animation.

                # Feature 17: Play animation.
                self._start_play_animation(pid, r, s)
            except Exception as e:
                # Fallback: try to play any legal card.
                try:
                    hand = self.players[pid].hand
                    if hand and self.round.state.current_trick is not None:
                        leading = self.round.state.current_trick.leading_suit
                        must_trump = None
                        if (self.round.state.is_first_trick
                                and self.round.state.winning_bidder_id == pid
                                and len(self.round.state.current_trick.played_cards) == 0):
                            must_trump = self.trump_suit
                        playable = legal_cards(hand, leading, must_trump)
                        if playable:
                            fallback_card = playable[0]
                            fb_action = PlayCardAction(player_id=pid, card=fallback_card)
                            self.environment.apply_action(fb_action)
                            r, s = card_key(fallback_card)
                            self._trick_played[pid] = (r, s)
                            if not self._trump_revealed:
                                self._trump_revealed = True
                except Exception:
                    pass
            self._play_idx += 1
            self._ai_timer = 15

    def _start_play_animation(self, pid: int, rank: str, suit: str):
        """Start a card play animation with swoosh scale effect (Feature 17)."""
        cx, cy = TABLE_WIDTH // 2, SCREEN_HEIGHT // 2
        # Start positions by player.
        start_positions = {
            0: (cx, 80),
            3: (50, cy),
            1: (TABLE_WIDTH - 100, cy),
            2: (cx, SCREEN_HEIGHT - 140),
        }

        # For the human player, use the actual card position from the fan.
        if pid == HUMAN_ID and hasattr(self, '_fan_card_data') and self._fan_card_data:
            for data in self._fan_card_data:
                card = data['card']
                r, s = card_key(card)
                if r == rank and s == suit:
                    start = (int(data['cx'] - data['w'] // 2), int(data['cy']))
                    break
            else:
                start = start_positions[pid]
        else:
            start = start_positions.get(pid, (cx, cy))

        # End positions (centre trick slots).
        offsets = {0: (0, -70), 1: (90, 0), 2: (0, 50), 3: (-90, 0)}
        dx, dy = offsets.get(pid, (0, 0))
        end = (cx + dx - CARD_WIDTH // 2, cy + dy - CARD_HEIGHT // 2)

        # Detect whipping: trump card played when leading suit is not trump.
        # NOT whipping if: trick led with trump (just following suit), or player IS the leader.
        is_whip = False
        if (self.trump_suit is not None and self.round.state.current_trick
                and self.round.state.current_trick.leading_suit is not None
                and self.round.state.current_trick.leading_suit != self.trump_suit
                and suit == SUIT_SYMBOLS.get(self.trump_suit, "")
                and len(self.round.state.current_trick.played_cards) > 1):
            # Only whip if this is NOT the first card (leader can't whip their own lead).
            is_whip = True
            self._play_whip_sound()

        if is_whip:
            # Create yellow card for whipping animation.
            yellow_card = pygame.Surface((CARD_WIDTH, CARD_HEIGHT), pygame.SRCALPHA)
            pygame.draw.rect(yellow_card, (255, 240, 100),
                             yellow_card.get_rect(), border_radius=CARD_RADIUS)
            pygame.draw.rect(yellow_card, (200, 180, 50),
                             yellow_card.get_rect(), width=1, border_radius=CARD_RADIUS)
            suit_color = RED_SUIT if suit in ("\u2665", "\u2666") else BLACK_SUIT
            sym_font = pygame.font.SysFont("Segoe UI", 18, bold=True)
            yellow_card.blit(sym_font.render(rank, True, suit_color), (5, 3))
            yellow_card.blit(sym_font.render(suit, True, suit_color), (5, 20))
            big_font = pygame.font.SysFont("Segoe UI", 32, bold=True)
            big_s = big_font.render(suit, True, suit_color)
            yellow_card.blit(big_s, big_s.get_rect(center=(CARD_WIDTH // 2, CARD_HEIGHT // 2)))
            surf = yellow_card
        else:
            surf = self._get_card_surface(rank, suit)

        # Swoosh: start at 1.2x scale, animate down to 1.0x.
        anim = AnimatingCard(surf, start, end, frames=15, start_scale=1.2, end_scale=1.0, pid=pid)
        self._play_animations.append(anim)

    def _start_trick_highlight(self):
        """Phase 1: Determine winner, highlight the winning card, pause."""
        trick = self.round.state.current_trick
        if trick is None or len(trick.played_cards) < 4:
            self.round.state.current_trick = None
            self._ai_timer = 20
            self._play_idx = 99
            return

        winner = trick_winner(trick, self.trump_suit)

        # Detect "whipping" — winner played trump on a non-trump-led trick.
        leading_suit = trick.leading_suit
        winner_card = None
        trump_count_in_trick = 0
        for pc in trick.played_cards:
            if pc.player_id == winner:
                winner_card = pc.card
            if pc.card.suit == self.trump_suit:
                trump_count_in_trick += 1

        self._trick_is_whip = False
        self._trick_is_double_whip = False
        if winner_card and leading_suit != self.trump_suit and winner_card.suit == self.trump_suit:
            self._trick_is_whip = True
            if trump_count_in_trick >= 2:
                self._trick_is_double_whip = True

        # Store winner info for phase 2.
        self._pending_trick_winner = winner
        self._pending_trick = trick

        # Feature 9: Gold highlight the winning card on the table.
        self._trick_winner_id = winner
        self._trick_winner_timer = 60

        # Pause with all 4 cards visible + gold highlight, then move to phase 2.
        self._ai_timer = 30  # Pause at 60fps.
        self._play_idx = 98  # Signals: next tick after timer → _collect_trick_to_winner.

    def _collect_trick_to_winner(self):
        """Phase 2: Animate cards sliding to winner, update scores."""
        winner = self._pending_trick_winner
        trick = self._pending_trick

        # Commit game state.
        self.round.state.completed_tricks.append(trick)
        self.round.state.current_trick = None
        self.round.next_leading_player_id = winner

        team = 0 if winner in (0, 2) else 1
        self.team_tricks[team] += 1

        # Track per-player trick wins for live stats.
        if hasattr(self, '_game_stats'):
            self._game_stats["player_tricks"][winner] = self._game_stats["player_tricks"].get(winner, 0) + 1
            self._game_stats["total_tricks_played"] = self._game_stats.get("total_tricks_played", 0) + 1

        # Stats panel animations: highlight + momentum.
        self._stat_highlight_timers["tricks"] = 30
        other_team = 1 - team
        self._team_streak[team] += 1
        self._team_streak[other_team] = 0

        # Animate cards sliding to winner's pile.
        cx_t, cy_t = TABLE_WIDTH // 2, SCREEN_HEIGHT // 2
        winner_positions = {
            0: (cx_t, 90), 3: (60, cy_t), 1: (TABLE_WIDTH - 80, cy_t), 2: (cx_t, SCREEN_HEIGHT - 130),
        }
        offsets = {0: (0, -70), 1: (90, 0), 2: (0, 50), 3: (-90, 0)}
        win_dest = winner_positions.get(winner, (cx_t, cy_t))
        for pid, (r, s) in self._trick_played.items():
            dx, dy = offsets.get(pid, (0, 0))
            start = (cx_t + dx - CARD_WIDTH // 2, cy_t + dy - CARD_HEIGHT // 2)
            surf = self._get_card_surface(r, s)
            anim = AnimatingCard(surf, start, win_dest, frames=20)
            self._play_animations.append(anim)

        winner_name = DISPLAY_NAMES[winner]
        team_name = "Team 1" if team == 0 else "Team 2"
        self._message = f"{winner_name} wins! ({team_name}: {self.team_tricks[team]})"
        self._message_timer = 30
        self._ai_timer = 25
        self._play_idx = 99
        self._log_game_event(f"Trick {self.trick_number}: {DISPLAY_NAMES[winner]} wins")

        # Save last trick for peek, then clear.
        self._last_trick_cards = dict(self._trick_played)
        self._trick_played = {}

        # Frame-based delay — _ai_timer counts down, then update loop calls _start_next_trick.

    # ----------------------------------------------------------
    # Event handling
    # ----------------------------------------------------------

    def handle_event(self, event):
        """Handle PyGame events."""
        # Window close — let it propagate (handled by main app).
        if event.type == pygame.QUIT:
            return

        # Feature 21: ESC quit overlay.
        if event.type == pygame.KEYDOWN:
            if self._show_quit_overlay:
                if event.key == pygame.K_y:
                    self._show_quit_overlay = False
                    self.phase = "idle"
                    return
                elif event.key == pygame.K_n or event.key == pygame.K_ESCAPE:
                    self._show_quit_overlay = False
                    return
            else:
                if event.key == pygame.K_ESCAPE:
                    self._show_quit_overlay = True
                    return
                if event.key == pygame.K_SPACE and self.phase == "game_over":
                    self.start_game()
                # Qabool draw keyboard shortcuts.
                if self.phase == "qabool_draw":
                    if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        if self._qd_step == "result":
                            self._start_shota_common()
                            return
                # Auto-Dak notice keyboard shortcut.
                if self.phase == "auto_dak_notice":
                    if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        self._redeal_after_dak()
                        return
                # Card-based Dak notice keyboard shortcut.
                if self.phase == "card_dak_notice":
                    if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        self._handle_card_dak_continue()
                        return
                # Pass-based Dak notice keyboard shortcut.
                if self.phase == "pass_dak_notice":
                    if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        self._redeal_after_dak()
                        return

        if self._show_quit_overlay:
            # Still allow Restart/Exit clicks on the panel even with overlay.
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                panel_x = TABLE_WIDTH
                pad = 10
                btn_w = STATS_PANEL_WIDTH - pad * 2
                btn_h = 28
                btn_y = SCREEN_HEIGHT - 75
                restart_btn = pygame.Rect(panel_x + pad, btn_y, btn_w, btn_h)
                exit_btn = pygame.Rect(panel_x + pad, btn_y + btn_h + 6, btn_w, btn_h)
                if restart_btn.collidepoint(event.pos):
                    self._show_quit_overlay = False
                    self._restart_to_name = True
                    self.phase = "idle"
                    return
                if exit_btn.collidepoint(event.pos):
                    pygame.event.post(pygame.event.Event(pygame.QUIT))
                    return
            return  # Block all other input while overlay shown.

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._handle_click(event.pos)

    def _handle_click(self, pos):
        """Handle mouse click — bidding or card selection."""
        # Restart and Exit buttons ALWAYS work regardless of phase.
        panel_x = TABLE_WIDTH
        pad = 10
        btn_w = STATS_PANEL_WIDTH - pad * 2
        btn_h = 28
        btn_y = SCREEN_HEIGHT - 75

        restart_btn = pygame.Rect(panel_x + pad, btn_y, btn_w, btn_h)
        if restart_btn.collidepoint(pos):
            self._restart_to_name = True
            self.phase = "idle"
            return

        exit_btn = pygame.Rect(panel_x + pad, btn_y + btn_h + 6, btn_w, btn_h)
        if exit_btn.collidepoint(pos):
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            return

        # Qabool draw phase — interactive card picking.
        if self.phase == "qabool_draw":
            self._handle_qabool_draw_click(pos)
            return

        # Auto-Dak notice — click Continue to re-deal.
        if self.phase == "auto_dak_notice":
            cx = TABLE_WIDTH // 2
            cy = SCREEN_HEIGHT // 2
            panel_h = 220
            panel_bottom = cy + panel_h // 2
            btn_rect = pygame.Rect(cx - 70, panel_bottom - 52, 140, 40)
            if btn_rect.collidepoint(pos):
                self._redeal_after_dak()
            return

        # Card-based Dak notice — click Continue to re-deal.
        if self.phase == "card_dak_notice":
            cx = TABLE_WIDTH // 2
            btn_rect = pygame.Rect(cx - 70, SCREEN_HEIGHT - 80, 140, 42)
            if btn_rect.collidepoint(pos):
                self._handle_card_dak_continue()
            return

        # Pass-based Dak notice — click Continue to re-deal.
        if self.phase == "pass_dak_notice":
            cx = TABLE_WIDTH // 2
            cy = SCREEN_HEIGHT // 2
            panel_h = 200
            panel_bottom = cy + panel_h // 2
            btn_rect = pygame.Rect(cx - 70, panel_bottom - 52, 140, 40)
            if btn_rect.collidepoint(pos):
                self._redeal_after_dak()
            return

        # Shota end — "Next Shota" button.
        if self.phase == "shota_end":
            cx = TABLE_WIDTH // 2
            cy = SCREEN_HEIGHT // 2
            btn_rect = pygame.Rect(cx - 90, cy + 100, 180, 45)
            if btn_rect.collidepoint(pos):
                self._button_press_timer = 4
                self._button_press_id = "next_shota"
                self._start_new_shota()
            return

        # Bidding phase.
        if self.phase == "bidding":
            is_human_turn = False
            if self._bid_index < len(self._bid_order) and self._bid_order[self._bid_index] == HUMAN_ID:
                is_human_turn = True
            if self.qabool_id == HUMAN_ID and self._bid_index >= len(self._bid_order):
                is_human_turn = True

            if is_human_turn:
                self._handle_bid_click(pos)
                return

        # Playing phase — card selection.
        if self.phase == "playing" and self._play_idx < 4:
            pid = self._play_order[self._play_idx]
            if pid == HUMAN_ID:
                card = self._get_clicked_card(pos)
                if card:
                    self._human_play(card)

    def _open_model_dialog(self):
        """Feature 22: Open file dialog to load AI model."""
        try:
            root = Tk()
            root.withdraw()
            path = filedialog.askopenfilename(
                title="Load AI Model",
                filetypes=[("JSON Model", "*.json"), ("All files", "*.*")]
            )
            root.destroy()
            if path:
                self._ai_model_path = path
                self._message = f"Model loaded: {os.path.basename(path)}"
                self._message_timer = 90
                self._log_game_event(f"AI Model: {os.path.basename(path)}")
        except Exception:
            pass

    def _handle_qabool_draw_click(self, pos):
        """Handle clicks during the Qabool draw phase."""
        cx, cy = TABLE_WIDTH // 2, SCREEN_HEIGHT // 2

        if self._qd_step == "picking":
            # Human picks a card from the grid.
            if self._qd_pick_idx < len(self._qd_pick_order):
                current_pid = self._qd_pick_order[self._qd_pick_idx]
                if current_pid == HUMAN_ID:
                    card_w, card_h = CARD_MINI_W, CARD_MINI_H
                    n = self._qd_num_cards
                    cols = 13
                    h_spacing = card_w + 4
                    v_spacing = card_h + 8
                    grid_w = cols * h_spacing - 4
                    grid_x = cx - grid_w // 2
                    grid_y = 170
                    picked_indices = set(self._qd_picked.values())

                    # Check from last to first (top card = higher index in overlaps, but grid has none).
                    for i in range(n - 1, -1, -1):
                        if i in picked_indices:
                            continue
                        row = i // cols
                        col = i % cols
                        x = grid_x + col * h_spacing
                        y = grid_y + row * v_spacing
                        cr = pygame.Rect(x, y, card_w, card_h)
                        if cr.collidepoint(pos):
                            self._qd_picked[HUMAN_ID] = i
                            self._qd_flip_timers[HUMAN_ID] = 20
                            self._qd_pick_idx += 1
                            self._qd_ai_timer = 50
                            return
            return

        if self._qd_step == "result":
            # "Continue" button click.
            rv_w, rv_h = CARD_LARGE_W, CARD_LARGE_H
            reveal_y = SCREEN_HEIGHT - rv_h - 140
            btn_rect = pygame.Rect(cx - 70, reveal_y + rv_h + 40, 140, 42)
            if btn_rect.collidepoint(pos):
                self._start_shota_common()
            return

    def _handle_bid_click(self, pos):
        """Handle bidding click — all visible at once: bids, suits, confirm."""
        cx = TABLE_WIDTH // 2
        cy = SCREEN_HEIGHT // 2

        # Row 1: Pass button (first in row) then Bid numbers.
        # Centred layout: row1_start = cx - 250
        row1_start = cx - 250
        pass_rect = pygame.Rect(row1_start, cy - 10, 60, 45)
        if pass_rect.collidepoint(pos):
            self._button_press_timer = 4
            self._button_press_id = "pass"
            self._selected_bid = None
            self._selected_trump_idx = None
            self._human_pass_action()
            return

        for i in range(7):
            chip_cx = row1_start + 65 + i * 60 + 27
            chip_cy = cy + 12
            chip_r = 22
            dx = pos[0] - chip_cx
            dy = pos[1] - chip_cy
            if dx * dx + dy * dy <= chip_r * chip_r:
                self._selected_bid = 7 + i
                return

        # Row 2: Suit buttons. row2_start = cx - 145
        row2_start = cx - 145
        for i in range(4):
            rect = pygame.Rect(row2_start + i * 75, cy + 50, 65, 55)
            if rect.collidepoint(pos):
                self._selected_trump_idx = i
                return

        # Row 3: Confirm (only if valid).
        if self._is_bid_valid():
            confirm_rect = pygame.Rect(cx - 70, cy + 120, 140, 40)
            if confirm_rect.collidepoint(pos):
                self._button_press_timer = 4
                self._button_press_id = "confirm"
                self._human_bid_with_number_and_suit()
                return

    def _is_bid_valid(self) -> bool:
        """Check if current bid + suit selection is valid."""
        if self._selected_bid is None or self._selected_trump_idx is None:
            return False

        suits = [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]
        chosen_suit = suits[self._selected_trump_idx]
        bid_value = self._selected_bid
        trump_count = sum(1 for c in self.players[HUMAN_ID].hand if c.suit == chosen_suit)

        # Can't use suit with 8+ cards.
        if trump_count >= 8:
            return False

        is_qabool = (self.qabool_id == HUMAN_ID and self._bid_index >= len(self._bid_order))
        someone_bid = (self._bidding_engine.highest_bid is not None)

        # Bid must be >= trump_count + 3 (or +2 for Qabool when matching someone's bid).
        if is_qabool and someone_bid:
            min_bid = trump_count + 2  # Advantage only when matching.
        else:
            min_bid = trump_count + 3
        if bid_value < max(7, min_bid):
            return False

        # Opening bid max 11 (not Qabool).
        if not is_qabool and not self._has_opening_bid and bid_value > 11:
            return False

        # Must beat current highest (not Qabool).
        current_highest = (self._bidding_engine.highest_bid.value
                           if self._bidding_engine.highest_bid else None)
        if not is_qabool and current_highest and bid_value <= current_highest:
            return False

        return True

    def _get_bid_warning(self) -> str:
        """Get warning text explaining why bid+suit is invalid."""
        if self._selected_bid is None or self._selected_trump_idx is None:
            return ""

        suits = [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]
        chosen_suit = suits[self._selected_trump_idx]
        bid_value = self._selected_bid
        trump_count = sum(1 for c in self.players[HUMAN_ID].hand if c.suit == chosen_suit)
        sym = SUIT_SYMBOLS[chosen_suit]

        if trump_count >= 8:
            return f"8+ cards in {sym} = Dak (cannot bid)"

        is_qabool = (self.qabool_id == HUMAN_ID and self._bid_index >= len(self._bid_order))
        someone_bid = (self._bidding_engine.highest_bid is not None)

        if is_qabool and someone_bid:
            min_bid = trump_count + 2
        else:
            min_bid = trump_count + 3

        if bid_value < max(7, min_bid):
            return f"{trump_count} {sym} cards = min bid {max(7, min_bid)}"

        if not is_qabool and not self._has_opening_bid and bid_value > 11:
            return "Opening bid cannot exceed 11"

        current_highest = (self._bidding_engine.highest_bid.value
                           if self._bidding_engine.highest_bid else None)
        if not is_qabool and current_highest and bid_value <= current_highest:
            return f"Current highest is {current_highest} — you must bid {current_highest + 1}+"

        return ""

    def _human_play(self, card: Card):
        """Human plays a card."""
        # Guard: must have an active trick.
        if self.round.state.current_trick is None:
            return
        if card not in self.players[HUMAN_ID].hand:
            return

        # Server-side legality check.
        leading_suit = self.round.state.current_trick.leading_suit
        must_trump = None
        if (self.round.state.is_first_trick
                and self.round.state.winning_bidder_id == HUMAN_ID
                and len(self.round.state.current_trick.played_cards) == 0):
            must_trump = self.trump_suit
        legal = legal_cards(self.players[HUMAN_ID].hand, leading_suit, must_trump)
        if card not in legal:
            self._message = "Illegal move!"
            self._message_timer = 40
            return

        try:
            action = PlayCardAction(player_id=HUMAN_ID, card=card)
            self.environment.apply_action(action)
        except Exception as e:
            self._message = f"Can't play that card"
            self._message_timer = 40
            return

        self._play_idx += 1
        r, s = card_key(card)
        self._trick_played[HUMAN_ID] = (r, s)
        self._ai_timer = 15

        # Clear recommendation after playing.
        self._ai_rec_card = None
        self._ai_recommendation = ""

        # Feature 1: Reveal trump on first card.
        if not self._trump_revealed:
            self._trump_revealed = True
            self._trump_flip_timer = 20

        # Feature 17: Animation.
        self._start_play_animation(HUMAN_ID, r, s)

    def _get_clicked_card(self, pos) -> Card | None:
        """Check if pos is on a card in the human hand (fan arc layout)."""
        hand = self.players[HUMAN_ID].hand
        if not hand:
            return None

        # Use the pre-computed fan card data from rendering.
        if not hasattr(self, '_fan_card_data') or not self._fan_card_data:
            return None

        mx, my = pos

        # Check cards in reverse order (rightmost/topmost drawn last = on top).
        for data in reversed(self._fan_card_data):
            if not data['legal']:
                continue

            # Rotate click point into card's local coordinate system.
            card_cx = data['cx']
            card_cy = data['cy']
            angle_rad = math.radians(data['angle_deg'])

            # Translate click relative to card centre.
            dx = mx - card_cx
            dy = my - card_cy

            # Rotate by negative angle to align with card's axis.
            cos_a = math.cos(-angle_rad)
            sin_a = math.sin(-angle_rad)
            local_x = dx * cos_a - dy * sin_a
            local_y = dx * sin_a + dy * cos_a

            # Check if local coordinates are within card bounds.
            half_w = data['w'] / 2
            half_h = data['h'] / 2
            if -half_w <= local_x <= half_w and -half_h <= local_y <= half_h:
                return data['card']

        return None

    # ----------------------------------------------------------
    # Render
    # ----------------------------------------------------------

    def render(self):
        """Render the full game screen."""
        # Feature 21: Quit overlay takes priority.
        if self._show_quit_overlay:
            self._render_quit_overlay()
            return

        if self.phase == "game_over":
            self._render_game_over()
            self._render_confetti()
            return

        if self.phase == "shota_end":
            self._render_shota_end()
            return

        if self.phase == "qabool_draw":
            self._render_qabool_draw()
            return

        if self.phase == "auto_dak_notice":
            self._render_auto_dak_notice()
            return

        if self.phase == "pass_dak_notice":
            self._render_pass_dak_notice()
            return

        if self.phase == "card_dak_notice":
            self._render_card_dak_notice()
            return

        # Table area (left of stats panel).
        table_rect = pygame.Rect(10, 10, TABLE_WIDTH - 20, SCREEN_HEIGHT - 20)
        pygame.draw.rect(self.screen, TABLE_FELT, table_rect, border_radius=12)
        pygame.draw.rect(self.screen, TABLE_BORDER, table_rect, width=2, border_radius=12)

        # Gold corner ornaments.
        self._draw_table_corners(table_rect)

        # Table felt texture — subtle diagonal lines.
        felt_texture = pygame.Surface((table_rect.width, table_rect.height), pygame.SRCALPHA)
        line_color = (255, 255, 255, 12)
        for offset in range(-table_rect.height, table_rect.width, 20):
            start = (max(0, offset), max(0, -offset))
            end = (min(table_rect.width, offset + table_rect.height),
                   min(table_rect.height, table_rect.height - offset) if offset >= 0
                   else min(table_rect.height, table_rect.width - offset))
            # Draw 45-degree lines.
            x1 = offset
            y1 = 0
            x2 = offset + table_rect.height
            y2 = table_rect.height
            # Clip to surface.
            pygame.draw.line(felt_texture, line_color,
                             (max(0, x1), max(0, y1)),
                             (min(table_rect.width, x2), min(table_rect.height, y2)))
        self.screen.blit(felt_texture, table_rect.topleft)

        # Table vignette — radial gradient overlay for depth.
        self._render_table_vignette(table_rect)

        cx, cy = TABLE_WIDTH // 2, SCREEN_HEIGHT // 2

        # Opponent cards with role borders (Feature 14).
        self._render_opponent(0, cx, 75, horizontal=True)
        self._render_opponent(3, 45, cy - 60, horizontal=False)
        self._render_opponent(1, TABLE_WIDTH - 45 - CARD_MINI_W, cy - 60, horizontal=False)

        # Player labels with roles.
        self._render_player_labels(cx, cy)

        # Persistent "Sahib Al-Qabool" indicator — only show during dealing/bidding.
        # (Removed — info bar shows Qabool info, gold dot during play)

        # Won tricks piles.
        self._render_tricks_won(cx, cy)

        # Feature 1: Trump display (hidden until revealed).
        self._render_trump_display()

        # Feature 15: Empty slot placeholders + Feature 13: labels + centre trick.
        self._render_centre_trick(cx, cy)

        # Last trick (bottom-left, face-down, reveals on hover).
        if self._last_trick_cards:
            self._render_last_trick_area()

        # Human hand.
        self._render_human_hand()

        # Feature 10: Turn indicator border around active player.
        # (Removed — no green boxes)

        # Bidding UI.
        if self.phase == "bidding":
            self._render_bidding_ui(cx, cy)

        # Deal animation (card fan).
        if self.phase == "dealing":
            for anim in self._deal_animations:
                anim.render(self.screen)

        # Play animations (Feature 17).
        for anim in self._play_animations:
            anim.render(self.screen)

        # Player turn glow.
        if self.phase == "playing" and self._active_turn_pid is not None:
            self._render_turn_glow(cx, cy)

        # Message — removed entirely.

        # Feature 8: Game log panel.
        self._render_game_log()

        # Feature 7: Bid display persistence — show during bidding AND playing.
        if self.phase in ("bidding", "playing"):
            self._render_bid_labels(cx, cy)

        # Victory confetti.
        self._render_confetti()

    def _render_bid_labels(self, cx, cy):
        """Render bid badges — uniform size, aligned horizontally for side players,
        vertically for top/bottom players."""
        cx_table = TABLE_WIDTH // 2
        right_x = TABLE_WIDTH - 45 - CARD_MINI_W + CARD_MINI_W // 2

        # Fixed chip size for all players.
        CHIP_W = 110
        CHIP_H = 26

        # Positions — Musaab & Gaafar on SAME horizontal line (cy + 130).
        # Push Musaab lower to avoid card overlap (his cards stack taller on left).
        # Hima & You on SAME vertical centre line (cx_table).
        positions = {
            0: (cx_table, 162),                        # Hima: centred below his cards
            3: (45 + CARD_MINI_W // 2, cy + 130),     # Musaab: further below card stack
            1: (right_x, cy + 130),                    # Gaafar: same Y as Musaab
            2: (cx_table, SCREEN_HEIGHT - 210),        # You: above "your turn"
        }

        bid_font = pygame.font.SysFont("Consolas", 12, bold=True)

        for pid, text in self._player_bids_display.items():
            if not text or pid not in positions:
                continue
            px, py = positions[pid]

            is_bid = "Bid" in text
            is_shooter = (pid == self.shooter_id)

            if is_shooter:
                bg_color = (40, 80, 40)
                border_color = HIGHLIGHT_GREEN
                text_color = TEXT_GREEN
                icon = ""
            else:
                bg_color = (50, 40, 10)
                border_color = TEXT_GOLD
                text_color = TEXT_GOLD
                icon = ""

            display_text = f"{icon}{text}"
            surf = bid_font.render(display_text, True, text_color)

            # Bid chip pop-in scale.
            timer = self._bid_chip_anim_timer.get(pid, 0)
            if timer > 0:
                # Ease from 1.3x down to 1.0x over 15 frames.
                t = timer / 15.0
                scale = 1.0 + 0.3 * t
            else:
                scale = 1.0

            # Fixed-size chip centred at position.
            chip_w = int(CHIP_W * scale)
            chip_h = int(CHIP_H * scale)
            chip_rect = pygame.Rect(0, 0, chip_w, chip_h)
            chip_rect.center = (px, py)

            chip_surf = pygame.Surface((chip_w, chip_h), pygame.SRCALPHA)
            pygame.draw.rect(chip_surf, (*bg_color, 220), chip_surf.get_rect(),
                             border_radius=10)
            pygame.draw.rect(chip_surf, border_color, chip_surf.get_rect(),
                             width=2, border_radius=10)
            self.screen.blit(chip_surf, chip_rect.topleft)

            # Text centred inside chip.
            text_rect = surf.get_rect(center=chip_rect.center)
            self.screen.blit(surf, text_rect)

    def _render_game_log(self):
        """Right-side stats panel — game info and live stats with stable layout."""
        panel_x = TABLE_WIDTH
        panel_w = STATS_PANEL_WIDTH
        pygame.draw.rect(self.screen, BG_DARK,
                         pygame.Rect(panel_x, 0, panel_w, SCREEN_HEIGHT))
        pygame.draw.line(self.screen, TABLE_BORDER,
                         (panel_x, 0), (panel_x, SCREEN_HEIGHT), 2)

        # Fonts (cached on first call).
        if not hasattr(self, '_panel_fonts'):
            self._panel_fonts = {
                'title': pygame.font.SysFont("Segoe UI", 15, bold=True),
                'label': pygame.font.SysFont("Segoe UI", 13),
                'value': pygame.font.SysFont("Consolas", 14, bold=True),
                'btn': pygame.font.SysFont("Segoe UI", 12, bold=True),
                'header': pygame.font.SysFont("Segoe UI", 11, bold=True),
            }
        title_font = self._panel_fonts['title']
        label_font = self._panel_fonts['label']
        value_font = self._panel_fonts['value']
        btn_font = self._panel_fonts['btn']
        header_font = self._panel_fonts['header']

        pad = 10
        inner_w = panel_w - pad * 2
        y = 8

        # --- Logo + Title ---
        if not hasattr(self, '_logo_surface'):
            self._logo_surface = None
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
            try:
                if os.path.exists(icon_path):
                    self._logo_surface = pygame.transform.smoothscale(
                        pygame.image.load(icon_path), (22, 22))
            except Exception:
                pass

        if self._logo_surface:
            self.screen.blit(self._logo_surface, (panel_x + pad, y))
            self.screen.blit(header_font.render("Sudanese Wist", True, TEXT_WHITE),
                             (panel_x + pad + 26, y + 4))
        else:
            self.screen.blit(header_font.render("Sudanese Wist", True, TEXT_WHITE),
                             (panel_x + pad, y + 2))
        y += 28

        # --- Rank & Points ---
        self.screen.blit(title_font.render(self._get_rank_name(), True, TEXT_GOLD),
                         (panel_x + pad, y))
        y += 22
        self.screen.blit(label_font.render(f"{self._player_points} pts", True, TEXT_LIGHT),
                         (panel_x + pad, y))
        games_text = f"Games: {self._player_games_played}"
        games_surf = label_font.render(games_text, True, TEXT_LIGHT)
        self.screen.blit(games_surf, (panel_x + panel_w - pad - games_surf.get_width(), y))
        y += 20
        # Rank progress bar.
        prog = self._get_rank_progress()
        prog_x = panel_x + pad
        prog_w = inner_w
        pygame.draw.rect(self.screen, (30, 50, 30), (prog_x, y, prog_w, 4), border_radius=2)
        if prog > 0:
            pygame.draw.rect(self.screen, TEXT_GOLD, (prog_x, y, int(prog_w * prog), 4), border_radius=2)
        y += 18

        # ========== GAME INFO CARD (fixed height) ==========
        game_card_y = y
        game_card_h = 130
        pygame.draw.rect(self.screen, (22, 40, 22),
                         (panel_x + 5, game_card_y, panel_w - 10, game_card_h), border_radius=8)
        pygame.draw.rect(self.screen, (45, 90, 45),
                         (panel_x + 5, game_card_y, panel_w - 10, game_card_h), width=1, border_radius=8)
        y += 8

        self.screen.blit(title_font.render("Game", True, TEXT_WHITE), (panel_x + pad + 4, y))
        y += 24

        # Info rows — only Shota, Trick, Qabool, Shooter.
        rows = [
            ("Shota", f"{self.shota_number}/5", TEXT_LIGHT),
            ("Trick", f"{self.trick_number}/13", TEXT_LIGHT),
            ("Qabool", DISPLAY_NAMES[self.qabool_id], TEXT_GOLD),
            ("Shooter", DISPLAY_NAMES.get(self.shooter_id, "-") if self.bid_value > 0 else "-", HIGHLIGHT_GREEN),
        ]
        for lbl, val, clr in rows:
            self._draw_stat_row(panel_x, pad + 4, panel_w, y, label_font, value_font, lbl, val, clr)
            y += 19

        # ========== RECOMMENDATION BOX (always visible, fixed size) ==========
        y = game_card_y + game_card_h + 12
        ai_box_h = 145
        ai_box_rect = pygame.Rect(panel_x + 5, y, panel_w - 10, ai_box_h)
        pygame.draw.rect(self.screen, (22, 40, 22), ai_box_rect, border_radius=8)
        pygame.draw.rect(self.screen, (45, 90, 45), ai_box_rect, width=1, border_radius=8)
        # Title.
        box_title = "Expert-Model Rec." if self._ai_model_path else "Rule-Based Rec."
        self.screen.blit(header_font.render(box_title, True, TEXT_GREEN),
                         (panel_x + pad + 6, y + 5))

        # Reason text only in box — left aligned, wrapped with label_font.
        content_x = panel_x + pad + 6
        content_w = panel_w - pad * 2 - 12
        rec_text = self._ai_recommendation if self._ai_recommendation else ""
        if rec_text:
            words = rec_text.split()
            lines = []
            current_line = ""
            for word in words:
                test = current_line + (" " if current_line else "") + word
                if label_font.size(test)[0] <= content_w:
                    current_line = test
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            text_y = y + 24
            for line in lines[:5]:
                ls = label_font.render(line, True, TEXT_LIGHT)
                self.screen.blit(ls, (content_x, text_y))
                text_y += 16

        y += ai_box_h + 8

        # Recommended card — to the right of user's hand, vertically centered with cards.
        rec_card = getattr(self, '_ai_rec_card', None)
        if rec_card:
            r, s = rec_card
            rc_w, rc_h = 50, 72
            # Position: right of the user's hand area, vertically centered.
            # User cards are at base_y (SCREEN_HEIGHT - CARD_LARGE_H - 30).
            hand_base_y = SCREEN_HEIGHT - CARD_LARGE_H - 30
            hand_center_y = hand_base_y + CARD_LARGE_H // 2
            rc_x = TABLE_WIDTH - rc_w - 30
            rc_y = hand_center_y - rc_h // 2
            # White background box.
            box_pad = 5
            bg_rect = pygame.Rect(rc_x - box_pad, rc_y - box_pad,
                                  rc_w + box_pad * 2, rc_h + box_pad * 2)
            pygame.draw.rect(self.screen, (240, 240, 240), bg_rect, border_radius=6)
            pygame.draw.rect(self.screen, TEXT_GREEN, bg_rect, width=2, border_radius=6)
            card_surf = self._get_card_surface_sized(r, s, rc_w, rc_h)
            self.screen.blit(card_surf, (rc_x, rc_y))
            # "REC" label below.
            rec_lbl = self.fonts["small"].render("REC", True, TEXT_GREEN)
            self.screen.blit(rec_lbl, rec_lbl.get_rect(centerx=rc_x + rc_w // 2, y=bg_rect.bottom + 2))

        # ========== SHOTA SCOREBOARD TABLE (always visible) ==========
        self.screen.blit(header_font.render("Shota Scores", True, TEXT_WHITE),
                         (panel_x + pad + 4, y))
        y += 16
        # Header row — 4 columns: Shota, Bid, T1, T2 (centered).
        col_w = (inner_w - 10) // 4
        hx = panel_x + pad + 4
        for ci, (hdr, hdr_c) in enumerate([("Shota", TEXT_WHITE), ("Bid", TEXT_WHITE),
                                            ("T1", TEAM1_BLUE), ("T2", TEAM2_ORANGE)]):
            s = header_font.render(hdr, True, hdr_c)
            self.screen.blit(s, s.get_rect(centerx=hx + col_w * ci + col_w // 2, y=y))
        y += 16
        # Per-shota rows (centered, red underline for shota winner).
        if self._shota_scores:
            for idx, entry in enumerate(self._shota_scores):
                tricks = entry["tricks"]
                bid = entry.get("bid", "-")
                playing_team = entry.get("playing_team", 0)
                bid_met = entry.get("bid_met", False)
                # Show team prefix before bid: T1-9, T2-10, etc.
                team_prefix = "T1" if playing_team == 0 else "T2"
                bid_display = f"{team_prefix}-{bid}" if bid != "-" else "-"
                # Winner: playing team if bid met, defending team if failed.
                winner_team = playing_team if bid_met else (1 - playing_team)

                s = label_font.render(str(idx + 1), True, TEXT_WHITE)
                self.screen.blit(s, s.get_rect(centerx=hx + col_w * 0 + col_w // 2, y=y))
                s = label_font.render(bid_display, True, TEXT_WHITE)
                self.screen.blit(s, s.get_rect(centerx=hx + col_w * 1 + col_w // 2, y=y))
                t1_surf = label_font.render(str(tricks[0]), True, TEAM1_BLUE)
                t1_rect = t1_surf.get_rect(centerx=hx + col_w * 2 + col_w // 2, y=y)
                self.screen.blit(t1_surf, t1_rect)
                if winner_team == 0:
                    pygame.draw.line(self.screen, BUTTON_RED,
                                     (t1_rect.left, t1_rect.bottom + 1),
                                     (t1_rect.right, t1_rect.bottom + 1), 2)
                t2_surf = label_font.render(str(tricks[1]), True, TEAM2_ORANGE)
                t2_rect = t2_surf.get_rect(centerx=hx + col_w * 3 + col_w // 2, y=y)
                self.screen.blit(t2_surf, t2_rect)
                if winner_team == 1:
                    pygame.draw.line(self.screen, BUTTON_RED,
                                     (t2_rect.left, t2_rect.bottom + 1),
                                     (t2_rect.right, t2_rect.bottom + 1), 2)
                y += 20
            # Total row.
            y += 4
            pygame.draw.line(self.screen, TEXT_DIM, (hx, y), (hx + inner_w - 12, y))
            y += 6
            s = label_font.render("Tot", True, TEXT_GOLD)
            self.screen.blit(s, s.get_rect(centerx=hx + col_w * 0 + col_w // 2, y=y))
            s = value_font.render(str(self.game_scores[0]), True, TEAM1_BLUE)
            self.screen.blit(s, s.get_rect(centerx=hx + col_w * 2 + col_w // 2, y=y))
            s = value_font.render(str(self.game_scores[1]), True, TEAM2_ORANGE)
            self.screen.blit(s, s.get_rect(centerx=hx + col_w * 3 + col_w // 2, y=y))

        # ========== BUTTONS (fixed at bottom) ==========
        btn_y = SCREEN_HEIGHT - 75
        btn_w = panel_w - pad * 2
        btn_h = 28
        mx, my = pygame.mouse.get_pos()

        # Restart button.
        restart_rect = pygame.Rect(panel_x + pad, btn_y, btn_w, btn_h)
        rbg = (50, 120, 50) if restart_rect.collidepoint(mx, my) else (40, 95, 40)
        pygame.draw.rect(self.screen, rbg, restart_rect, border_radius=5)
        pygame.draw.rect(self.screen, (30, 70, 30), restart_rect, width=1, border_radius=5)
        restart_text = btn_font.render("Restart Game", True, TEXT_LIGHT)
        self.screen.blit(restart_text, restart_text.get_rect(center=restart_rect.center))

        # Exit button.
        exit_rect = pygame.Rect(panel_x + pad, btn_y + btn_h + 6, btn_w, btn_h)
        ebg = (100, 40, 40) if exit_rect.collidepoint(mx, my) else (70, 30, 30)
        pygame.draw.rect(self.screen, ebg, exit_rect, border_radius=5)
        pygame.draw.rect(self.screen, (50, 20, 20), exit_rect, width=1, border_radius=5)
        exit_text = btn_font.render("Exit the Game", True, (200, 100, 100))
        self.screen.blit(exit_text, exit_text.get_rect(center=exit_rect.center))

    def _draw_stat_row(self, panel_x, pad, panel_w, y, label_font, value_font,
                       label: str, value: str, value_color):
        """Draw a label on the left, value on the right."""
        self.screen.blit(label_font.render(label, True, TEXT_LIGHT), (panel_x + pad, y))
        val_surf = value_font.render(value, True, value_color)
        self.screen.blit(val_surf, (panel_x + panel_w - pad - 4 - val_surf.get_width(), y))

    def _draw_table_corners(self, table_rect):
        """Draw gold decorative corner flourishes on the table."""
        gold = (200, 170, 50)
        gold_dim = (140, 115, 30)
        length = 35
        offset = 18  # Inset from the corner.

        corners = [
            (table_rect.left + offset, table_rect.top + offset, 1, 1),      # Top-left
            (table_rect.right - offset, table_rect.top + offset, -1, 1),    # Top-right
            (table_rect.left + offset, table_rect.bottom - offset, 1, -1),  # Bottom-left
            (table_rect.right - offset, table_rect.bottom - offset, -1, -1),  # Bottom-right
        ]

        for cx, cy, dx, dy in corners:
            # L-shaped corner flourish.
            pygame.draw.line(self.screen, gold, (cx, cy), (cx + length * dx, cy), 2)
            pygame.draw.line(self.screen, gold, (cx, cy), (cx, cy + length * dy), 2)
            # Small decorative dot at the corner vertex.
            pygame.draw.circle(self.screen, gold, (cx, cy), 3)
            # Short inner accent lines.
            pygame.draw.line(self.screen, gold_dim,
                             (cx + 6 * dx, cy + 6 * dy),
                             (cx + 18 * dx, cy + 6 * dy), 1)
            pygame.draw.line(self.screen, gold_dim,
                             (cx + 6 * dx, cy + 6 * dy),
                             (cx + 6 * dx, cy + 18 * dy), 1)

    def _render_quit_overlay(self):
        """Feature 21: ESC quit overlay."""
        # Render game underneath.
        table_rect = pygame.Rect(10, 10, TABLE_WIDTH - 20, SCREEN_HEIGHT - 20)
        pygame.draw.rect(self.screen, TABLE_FELT, table_rect, border_radius=12)

        # Dark overlay.
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        font = pygame.font.SysFont("Segoe UI", 28, bold=True)
        text = font.render("Quit game?", True, TEXT_WHITE)
        self.screen.blit(text, text.get_rect(centerx=cx, y=cy - 40))

        hint_font = pygame.font.SysFont("Segoe UI", 18)
        hint = hint_font.render("Y = Yes, return to menu  |  N = No, keep playing", True, TEXT_LIGHT)
        self.screen.blit(hint, hint.get_rect(centerx=cx, y=cy + 10))

    def _render_game_over(self):
        """Render game over screen with fanned cards and stats."""
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))

        if self.game_scores[0] > self.game_scores[1]:
            winner_text = "YOUR TEAM WINS!"
            color = HIGHLIGHT_GREEN
        elif self.game_scores[1] > self.game_scores[0]:
            winner_text = "TEAM 2 WINS!"
            color = TEAM2_ORANGE
        else:
            winner_text = "IT'S A DRAW!"
            color = TEXT_WHITE

        # Fanned victory cards behind the text.
        fan_cards = [("A", "♠"), ("K", "♥"), ("Q", "♣"), ("J", "♦"), ("A", "♥")]
        fan_cx = cx
        fan_y = cy - 60
        for i, (r, s) in enumerate(fan_cards):
            angle = (i - 2) * 12  # -24, -12, 0, 12, 24 degrees.
            card_surf = create_card_surface(r, s, CARD_LARGE_W, CARD_LARGE_H)
            rotated = pygame.transform.rotate(card_surf, -angle)
            rot_rect = rotated.get_rect(center=(fan_cx + (i - 2) * 40, fan_y))
            rotated.set_alpha(120)
            self.screen.blit(rotated, rot_rect)

        # Title.
        trophy_font = pygame.font.SysFont("Segoe UI", 42, bold=True)
        trophy = trophy_font.render("GAME OVER", True, TEXT_GOLD)
        self.screen.blit(trophy, trophy.get_rect(centerx=cx, y=cy - 120))

        # Winner.
        win_font = pygame.font.SysFont("Segoe UI", 28, bold=True)
        win = win_font.render(winner_text, True, color)
        self.screen.blit(win, win.get_rect(centerx=cx, y=cy - 60))

        # Score.
        score_font = pygame.font.SysFont("Segoe UI", 18)
        score = score_font.render(
            f"T1 ({DISPLAY_NAMES[2]} + {DISPLAY_NAMES[0]}): {self.game_scores[0]}  |  "
            f"T2 ({DISPLAY_NAMES[1]} + {DISPLAY_NAMES[3]}): {self.game_scores[1]}",
            True, TEXT_LIGHT)
        self.screen.blit(score, score.get_rect(centerx=cx, y=cy + 5))

        # Stats summary.
        stats = getattr(self, '_game_stats', {})
        stat_font = pygame.font.SysFont("Segoe UI", 14)
        stat_line = (f"Shotas: {stats.get('shotas_played', self.shota_number)}  |  "
                     f"Daks: {self._dak_count}  |  "
                     f"Seeks: {stats.get('seeks', 0)}  |  "
                     f"Bids Met: {stats.get('bids_met', 0)}")
        self.screen.blit(stat_font.render(stat_line, True, TEXT_LIGHT),
                         stat_font.render(stat_line, True, TEXT_LIGHT)
                         .get_rect(centerx=cx, y=cy + 40))

        hint = self.fonts["medium"].render(
            "Press SPACE for new game  |  ESC for menu", True, TEXT_LIGHT)
        self.screen.blit(hint, hint.get_rect(centerx=cx, y=cy + 80))

        self._render_game_log()

    def _render_shota_end(self):
        """Render Shota end summary — chapter break style."""
        cx, cy = TABLE_WIDTH // 2, SCREEN_HEIGHT // 2

        # Dark overlay.
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        # Card-style panel in the centre.
        panel_w, panel_h = 420, 280
        panel_rect = pygame.Rect(cx - panel_w // 2, cy - panel_h // 2, panel_w, panel_h)
        pygame.draw.rect(self.screen, (20, 35, 20), panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, TEXT_GOLD, panel_rect, width=2, border_radius=12)

        # Gold corner dots on the panel.
        for corner_x, corner_y in [(panel_rect.left + 12, panel_rect.top + 12),
                                    (panel_rect.right - 12, panel_rect.top + 12),
                                    (panel_rect.left + 12, panel_rect.bottom - 12),
                                    (panel_rect.right - 12, panel_rect.bottom - 12)]:
            pygame.draw.circle(self.screen, TEXT_GOLD, (corner_x, corner_y), 3)

        playing_team = self.players[self.shooter_id].team_id
        bid_met = self.team_tricks[playing_team] >= self.bid_value

        # Title.
        title_font = pygame.font.SysFont("Segoe UI", 26, bold=True)
        title = title_font.render(f"Shota {self.shota_number} Complete", True, TEXT_WHITE)
        self.screen.blit(title, title.get_rect(centerx=cx, y=panel_rect.top + 20))

        # Result — based on bid outcome, not raw trick count.
        human_team = 0  # Human is on team 0.
        if bid_met:
            # Playing team won the shota.
            your_team_won = (playing_team == human_team)
        else:
            # Playing team failed — defending team won.
            your_team_won = (playing_team != human_team)

        if your_team_won:
            result_text = "Your Team WON"
            result_color = HIGHLIGHT_GREEN
        else:
            result_text = "Your Team LOST"
            result_color = BUTTON_RED
        result_font = pygame.font.SysFont("Segoe UI", 20, bold=True)
        self.screen.blit(result_font.render(result_text, True, result_color),
                         result_font.render(result_text, True, result_color)
                         .get_rect(centerx=cx, y=panel_rect.top + 55))

        # Seek announcement.
        info_y = panel_rect.top + 85
        if hasattr(self, '_seek_team') and self._seek_team is not None:
            seek_font = pygame.font.SysFont("Segoe UI", 18, bold=True)
            seek_surf = seek_font.render(
                f"SEEK! Team {self._seek_team + 1} won all 13!", True, TEXT_GOLD)
            self.screen.blit(seek_surf, seek_surf.get_rect(centerx=cx, y=info_y))
            info_y += 28

        # Next shota preview.
        next_qabool = (self.qabool_id + 1) % 4
        next_font = pygame.font.SysFont("Segoe UI", 13)
        next_surf = next_font.render(
            f"Next Shota {self.shota_number + 1}  •  Qabool: {DISPLAY_NAMES[next_qabool]}",
            True, TEXT_LIGHT)
        self.screen.blit(next_surf, next_surf.get_rect(centerx=cx, y=info_y))

        # "Next Shota" button.
        ns_press = 2 if (self._button_press_timer > 0 and self._button_press_id == "next_shota") else 0
        btn_rect = pygame.Rect(cx - 90, panel_rect.bottom - 55 + ns_press, 180, 40 - ns_press)
        mx, my = pygame.mouse.get_pos()
        hover = btn_rect.collidepoint(mx, my)
        bg = (56, 142, 60) if hover else BUTTON_GREEN
        pygame.draw.rect(self.screen, bg, btn_rect, border_radius=10)
        if hover:
            pygame.draw.rect(self.screen, (100, 200, 100), btn_rect, width=2, border_radius=10)
        btn_font = pygame.font.SysFont("Segoe UI", 15, bold=True)
        btn_text = btn_font.render("Next Shota", True, TEXT_WHITE)
        self.screen.blit(btn_text, btn_text.get_rect(center=btn_rect.center))

        self._render_game_log()

    def _render_auto_dak_notice(self):
        """Render the auto-Dak explanation screen for the human Qabool on first shota."""
        cx, cy = TABLE_WIDTH // 2, SCREEN_HEIGHT // 2

        # Table background.
        table_rect = pygame.Rect(10, 10, TABLE_WIDTH - 20, SCREEN_HEIGHT - 20)
        pygame.draw.rect(self.screen, TABLE_FELT, table_rect, border_radius=12)
        pygame.draw.rect(self.screen, TABLE_BORDER, table_rect, width=2, border_radius=12)
        self._draw_table_corners(table_rect)

        # Dark overlay panel.
        panel_w, panel_h = 500, 220
        panel_rect = pygame.Rect(cx - panel_w // 2, cy - panel_h // 2, panel_w, panel_h)
        pygame.draw.rect(self.screen, (20, 35, 20), panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, TEXT_GOLD, panel_rect, width=2, border_radius=12)

        title_font = pygame.font.SysFont("Segoe UI", 22, bold=True)
        body_font = pygame.font.SysFont("Segoe UI", 14)
        subtitle_font = pygame.font.SysFont("Segoe UI", 16, bold=True)

        title = title_font.render("3-Passes-Based DAK", True, BUTTON_RED)
        self.screen.blit(title, title.get_rect(centerx=cx, y=panel_rect.top + 18))
        subtitle = subtitle_font.render("Automatic Re-deal", True, TEXT_GOLD)
        self.screen.blit(subtitle, subtitle.get_rect(centerx=cx, y=panel_rect.top + 46))

        lines = [
            "All three players passed on the first shota.",
            "By rule, the 3rd player (dealer) declares automatic Dak.",
            "As Sahib Al-Qabool, you have no say on the first",
            "shota in this case. Cards will be re-dealt.",
        ]
        for i, line in enumerate(lines):
            surf = body_font.render(line, True, TEXT_LIGHT)
            self.screen.blit(surf, surf.get_rect(centerx=cx, y=panel_rect.top + 78 + i * 22))

        # Continue button (inside panel, near bottom).
        btn_rect = pygame.Rect(cx - 70, panel_rect.bottom - 52, 140, 40)
        mx, my = pygame.mouse.get_pos()
        hover = btn_rect.collidepoint(mx, my)
        bg = (56, 142, 60) if hover else BUTTON_GREEN
        pygame.draw.rect(self.screen, bg, btn_rect, border_radius=10)
        if hover:
            pygame.draw.rect(self.screen, (100, 200, 100), btn_rect, width=2, border_radius=10)
        btn_font = pygame.font.SysFont("Segoe UI", 15, bold=True)
        btn_text = btn_font.render("Continue", True, TEXT_WHITE)
        self.screen.blit(btn_text, btn_text.get_rect(center=btn_rect.center))

        self._render_game_log()

    def _render_pass_dak_notice(self):
        """Render pass-based Dak announcement — all four players passed."""
        cx, cy = TABLE_WIDTH // 2, SCREEN_HEIGHT // 2

        # Table background.
        table_rect = pygame.Rect(10, 10, TABLE_WIDTH - 20, SCREEN_HEIGHT - 20)
        pygame.draw.rect(self.screen, TABLE_FELT, table_rect, border_radius=12)
        pygame.draw.rect(self.screen, TABLE_BORDER, table_rect, width=2, border_radius=12)
        self._draw_table_corners(table_rect)

        # Panel.
        panel_w, panel_h = 460, 200
        panel_rect = pygame.Rect(cx - panel_w // 2, cy - panel_h // 2, panel_w, panel_h)
        pygame.draw.rect(self.screen, (20, 35, 20), panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, BUTTON_RED, panel_rect, width=2, border_radius=12)

        title_font = pygame.font.SysFont("Segoe UI", 22, bold=True)
        body_font = pygame.font.SysFont("Segoe UI", 14)
        subtitle_font = pygame.font.SysFont("Segoe UI", 16, bold=True)

        title = title_font.render("4-Passes-DAK", True, BUTTON_RED)
        self.screen.blit(title, title.get_rect(centerx=cx, y=panel_rect.top + 18))
        subtitle = subtitle_font.render("Automatic Re-deal", True, TEXT_GOLD)
        self.screen.blit(subtitle, subtitle.get_rect(centerx=cx, y=panel_rect.top + 46))

        lines = [
            "All four players passed during Al-Tasmiya.",
            "Qabool now rotates to the next player.",
            f"(Dak #{self._dak_count} of max 2 per game)",
        ]
        for i, line in enumerate(lines):
            surf = body_font.render(line, True, TEXT_LIGHT)
            self.screen.blit(surf, surf.get_rect(centerx=cx, y=panel_rect.top + 78 + i * 22))

        # Continue button (inside panel, near bottom).
        btn_rect = pygame.Rect(cx - 70, panel_rect.bottom - 52, 140, 40)
        mx, my = pygame.mouse.get_pos()
        hover = btn_rect.collidepoint(mx, my)
        bg = (56, 142, 60) if hover else BUTTON_GREEN
        pygame.draw.rect(self.screen, bg, btn_rect, border_radius=10)
        if hover:
            pygame.draw.rect(self.screen, (100, 200, 100), btn_rect, width=2, border_radius=10)
        btn_font = pygame.font.SysFont("Segoe UI", 15, bold=True)
        btn_text = btn_font.render("Continue", True, TEXT_WHITE)
        self.screen.blit(btn_text, btn_text.get_rect(center=btn_rect.center))

        self._render_game_log()

    def _render_card_dak_notice(self):
        """Render card-based Dak notice — show why and which cards as proof."""
        cx, cy = TABLE_WIDTH // 2, SCREEN_HEIGHT // 2

        # Table background.
        table_rect = pygame.Rect(10, 10, TABLE_WIDTH - 20, SCREEN_HEIGHT - 20)
        pygame.draw.rect(self.screen, TABLE_FELT, table_rect, border_radius=12)
        pygame.draw.rect(self.screen, TABLE_BORDER, table_rect, width=2, border_radius=12)
        self._draw_table_corners(table_rect)

        info = self._card_dak_info
        pid = info["player_id"]
        reason = info["reason"]
        proof_cards = info["proof_cards"]

        title_font = pygame.font.SysFont("Segoe UI", 22, bold=True)
        subtitle_font = pygame.font.SysFont("Segoe UI", 16, bold=True)
        body_font = pygame.font.SysFont("Segoe UI", 14)
        name_font = pygame.font.SysFont("Segoe UI", 13, bold=True)

        # Use smaller cards for the proof display.
        card_w, card_h = CARD_WIDTH, CARD_HEIGHT  # 70x100 instead of 85x120

        # Calculate vertical layout: text block above cards, cards centered vertically.
        # Cards will be centered in the middle of the screen.
        cards_y = cy - card_h // 2 + 20  # Slightly below vertical center

        # Text lines above the cards with generous line spacing.
        line_spacing = 28  # Space between text lines

        # Build text content.
        player_text = f"{DISPLAY_NAMES[pid]} declares Dak!"
        if reason == "8_plus":
            suit_sym = SUIT_SYMBOLS.get(info["suit"], "?")
            reason_text = f"Reason: 8 or more cards in {suit_sym} ({len(proof_cards)} cards)"
        else:
            reason_text = "Reason: No picture cards (no A, K, Q, or J)"
        proof_label = "Proof cards shown:"

        # Position text block above cards with spacing.
        text_block_bottom = cards_y - 20  # Gap between text and cards
        text_lines = [
            (title_font, "Card-Based DAK", BUTTON_RED),
            (subtitle_font, "Re-deal Required", TEXT_GOLD),
            (body_font, player_text, TEXT_GOLD),
            (body_font, reason_text, TEXT_LIGHT),
            (name_font, proof_label, TEXT_WHITE),
        ]
        # Render text lines from the bottom up to position above cards.
        text_y = text_block_bottom - len(text_lines) * line_spacing
        for font, text, color in text_lines:
            surf = font.render(text, True, color)
            self.screen.blit(surf, surf.get_rect(centerx=cx, y=text_y))
            text_y += line_spacing

        # Display proof cards in a row — centered horizontally and vertically.
        n = len(proof_cards)
        spacing = min(card_w + 6, (TABLE_WIDTH - 120) // max(1, n))
        total_w = (n - 1) * spacing + card_w
        start_x = cx - total_w // 2

        # Sort proof cards by rank for readability.
        sorted_proof = sorted(proof_cards, key=lambda c: (c.suit.value, -rank_value(c.rank)))
        for i, card in enumerate(sorted_proof):
            r, s = card_key(card)
            card_surf = self._get_card_surface_sized(r, s, card_w, card_h)
            self.screen.blit(card_surf, (start_x + i * spacing, cards_y))

        # Explanation below cards.
        explain_y = cards_y + card_h + 25
        explain_lines = [
            "Qabool stays the same. Cards will be re-dealt from scratch.",
        ]
        if reason == "8_plus":
            explain_lines.insert(0, "A player with 8+ cards in one suit must always declare Dak.")
        else:
            explain_lines.insert(0, "A player with no picture cards must declare Dak and show their hand.")

        for i, line in enumerate(explain_lines):
            surf = body_font.render(line, True, TEXT_LIGHT)
            self.screen.blit(surf, surf.get_rect(centerx=cx, y=explain_y + i * 24))

        # Continue button.
        btn_rect = pygame.Rect(cx - 70, SCREEN_HEIGHT - 80, 140, 42)
        mx, my = pygame.mouse.get_pos()
        hover = btn_rect.collidepoint(mx, my)
        bg = (56, 142, 60) if hover else BUTTON_GREEN
        pygame.draw.rect(self.screen, bg, btn_rect, border_radius=10)
        if hover:
            pygame.draw.rect(self.screen, (100, 200, 100), btn_rect, width=2, border_radius=10)
        btn_font = pygame.font.SysFont("Segoe UI", 15, bold=True)
        btn_text = btn_font.render("Continue", True, TEXT_WHITE)
        self.screen.blit(btn_text, btn_text.get_rect(center=btn_rect.center))

        self._render_game_log()

    def _render_qabool_draw(self):
        """Render the interactive Qabool draw — 52 cards in 4 rows (13 per row)."""
        cx, cy = TABLE_WIDTH // 2, SCREEN_HEIGHT // 2

        # Table background.
        table_rect = pygame.Rect(10, 10, TABLE_WIDTH - 20, SCREEN_HEIGHT - 20)
        pygame.draw.rect(self.screen, TABLE_FELT, table_rect, border_radius=12)
        pygame.draw.rect(self.screen, TABLE_BORDER, table_rect, width=2, border_radius=12)
        self._draw_table_corners(table_rect)

        # 4-row grid layout: 13 cards per row, 4 rows.
        card_w, card_h = CARD_MINI_W, CARD_MINI_H
        n = self._qd_num_cards
        cols = 13
        rows = 4
        h_spacing = card_w + 4
        v_spacing = card_h + 8
        grid_w = cols * h_spacing - 4
        grid_h = rows * v_spacing - 8
        grid_x = cx - grid_w // 2
        grid_y = 170

        def _card_grid_pos(i):
            """Get (x, y) for card index i in 4-row grid. No rotation."""
            row = i // cols
            col = i % cols
            x = grid_x + col * h_spacing + card_w // 2
            y = grid_y + row * v_spacing + card_h // 2
            return x, y, 0  # x, y, rotation=0
        title_font = pygame.font.SysFont("Segoe UI", 22, bold=True)
        subtitle_font = pygame.font.SysFont("Segoe UI", 13)
        name_font = pygame.font.SysFont("Segoe UI", 12, bold=True)

        rv_w, rv_h = CARD_LARGE_W, CARD_LARGE_H
        reveal_y = SCREEN_HEIGHT - rv_h - 140
        reveal_spacing = 130
        total_reveal_w = (len(self._qd_pick_order) - 1) * reveal_spacing
        reveal_start_x = cx - total_reveal_w // 2
        reveal_positions = {}
        for i, pid in enumerate(self._qd_pick_order):
            reveal_positions[pid] = (int(reveal_start_x + i * reveal_spacing - rv_w // 2), reveal_y)

        if self._qd_step == "intro":
            title = title_font.render("Determining First Qabool", True, TEXT_GOLD)
            self.screen.blit(title, title.get_rect(centerx=cx, y=30))
            lines = [
                "Before the first Shota, each player draws one card from the deck.",
                "The team with the highest card earns the first Sahib Al-Qabool.",
                "Within that team, the player who drew higher becomes Qabool.",
            ]
            for i, line in enumerate(lines):
                surf = subtitle_font.render(line, True, TEXT_LIGHT)
                self.screen.blit(surf, surf.get_rect(centerx=cx, y=58 + i * 20))

            btn_rect = pygame.Rect(cx - 60, 125, 120, 38)
            mx, my = pygame.mouse.get_pos()
            hover = btn_rect.collidepoint(mx, my)
            bg = (56, 142, 60) if hover else BUTTON_GREEN
            pygame.draw.rect(self.screen, bg, btn_rect, border_radius=10)
            if hover:
                pygame.draw.rect(self.screen, (100, 200, 100), btn_rect, width=2, border_radius=10)
            btn_font = pygame.font.SysFont("Segoe UI", 15, bold=True)
            btn_text = btn_font.render("OK", True, TEXT_WHITE)
            self.screen.blit(btn_text, btn_text.get_rect(center=btn_rect.center))

            for i in range(n):
                x, y, rot = _card_grid_pos(i)
                rotated = pygame.transform.rotate(self._card_back_mini, rot)
                self.screen.blit(rotated, rotated.get_rect(center=(x, y)))

        elif self._qd_step == "picking":
            if self._qd_pick_idx < len(self._qd_pick_order):
                current_pid = self._qd_pick_order[self._qd_pick_idx]
                if current_pid == HUMAN_ID:
                    prompt = "Your turn - click a card to draw!"
                    prompt_color = TEXT_GREEN
                else:
                    prompt = f"{DISPLAY_NAMES[current_pid]} is picking..."
                    prompt_color = TEXT_LIGHT
            else:
                prompt = "All cards drawn!"
                prompt_color = TEXT_GOLD

            title = title_font.render("Determining First Qabool", True, TEXT_GOLD)
            self.screen.blit(title, title.get_rect(centerx=cx, y=30))
            # Explanation lines.
            lines = [
                "Each player draws one card. Highest card's team earns Qabool.",
            ]
            for i, line in enumerate(lines):
                surf = subtitle_font.render(line, True, TEXT_LIGHT)
                self.screen.blit(surf, surf.get_rect(centerx=cx, y=55 + i * 18))
            # Prompt.
            prompt_surf = subtitle_font.render(prompt, True, prompt_color)
            self.screen.blit(prompt_surf, prompt_surf.get_rect(centerx=cx, y=78))

            mx, my = pygame.mouse.get_pos()
            picked_indices = set(self._qd_picked.values())
            hovered_idx = -1
            is_human_turn = (self._qd_pick_idx < len(self._qd_pick_order)
                             and self._qd_pick_order[self._qd_pick_idx] == HUMAN_ID)
            if is_human_turn:
                for i in range(n - 1, -1, -1):
                    if i in picked_indices:
                        continue
                    x, y, rot = _card_grid_pos(i)
                    cr = pygame.Rect(x - card_w // 2, y - card_h // 2, card_w, card_h)
                    if cr.collidepoint(mx, my):
                        hovered_idx = i
                        break

            for i in range(n):
                x, y, rot = _card_grid_pos(i)
                if i in picked_indices:
                    # Find the picker and check if their flip is still active.
                    picker_pid = None
                    for pid, idx in self._qd_picked.items():
                        if idx == i:
                            picker_pid = pid
                            break
                    flip_active = (picker_pid is not None
                                   and self._qd_flip_timers.get(picker_pid, 0) > 0)
                    if flip_active:
                        # Show the card lifted + highlighted while flipping.
                        draw_y = y - 14
                        rotated = pygame.transform.rotate(self._card_back_mini, rot)
                        self.screen.blit(rotated, rotated.get_rect(center=(x, draw_y)))
                        glow = pygame.Surface((card_w + 6, card_h + 6), pygame.SRCALPHA)
                        pygame.draw.rect(glow, (*HIGHLIGHT_GREEN, 180),
                                         glow.get_rect(), width=2, border_radius=6)
                        glow_rot = pygame.transform.rotate(glow, rot)
                        self.screen.blit(glow_rot, glow_rot.get_rect(center=(x, draw_y)))
                    else:
                        # Already resolved — faded gap.
                        faded = self._card_back_mini.copy()
                        faded.set_alpha(25)
                        rotated = pygame.transform.rotate(faded, rot)
                        self.screen.blit(rotated, rotated.get_rect(center=(x, y)))
                else:
                    draw_y = y - 12 if i == hovered_idx else y
                    rotated = pygame.transform.rotate(self._card_back_mini, rot)
                    self.screen.blit(rotated, rotated.get_rect(center=(x, draw_y)))
                    if i == hovered_idx:
                        glow = pygame.Surface((card_w + 6, card_h + 6), pygame.SRCALPHA)
                        pygame.draw.rect(glow, (*TEXT_GOLD, 180),
                                         glow.get_rect(), width=2, border_radius=6)
                        glow_rot = pygame.transform.rotate(glow, rot)
                        self.screen.blit(glow_rot, glow_rot.get_rect(center=(x, draw_y)))

            # Revealed cards below.
            for pid in self._qd_picked:
                dest_x, dest_y = reveal_positions[pid]
                flip_remaining = self._qd_flip_timers.get(pid, 0)
                if flip_remaining > 0:
                    total_flip = 20
                    fp = 1.0 - (flip_remaining / total_flip)
                    if fp < 0.5:
                        sw = max(2, int(rv_w * (1.0 - fp * 2.0)))
                        squeezed = pygame.transform.smoothscale(self._card_back_large, (sw, rv_h))
                    else:
                        sw = max(2, int(rv_w * ((fp - 0.5) * 2.0)))
                        card_obj = self._qabool_draw_cards[pid]
                        r, s = card_key(card_obj)
                        face = self._get_card_surface_sized(r, s, rv_w, rv_h)
                        squeezed = pygame.transform.smoothscale(face, (sw, rv_h))
                    sq_rect = squeezed.get_rect(centerx=dest_x + rv_w // 2, centery=dest_y + rv_h // 2)
                    self.screen.blit(squeezed, sq_rect)
                else:
                    card_obj = self._qabool_draw_cards[pid]
                    r, s = card_key(card_obj)
                    face = self._get_card_surface_sized(r, s, rv_w, rv_h)
                    self.screen.blit(face, (dest_x, dest_y))
                name_surf = name_font.render(DISPLAY_NAMES[pid], True, TEXT_WHITE)
                self.screen.blit(name_surf, name_surf.get_rect(centerx=dest_x + rv_w // 2, y=dest_y + rv_h + 5))

        elif self._qd_step == "result":
            title = title_font.render("Drawing for Qabool", True, TEXT_GOLD)
            self.screen.blit(title, title.get_rect(centerx=cx, y=30))

            # Faded fan.
            for i in range(n):
                x, y, rot = _card_grid_pos(i)
                faded = self._card_back_mini.copy()
                faded.set_alpha(20)
                rotated = pygame.transform.rotate(faded, rot)
                self.screen.blit(rotated, rotated.get_rect(center=(x, y)))

            # Revealed cards.
            for pid in self._qd_pick_order:
                dest_x, dest_y = reveal_positions[pid]
                card_obj = self._qabool_draw_cards[pid]
                r, s = card_key(card_obj)
                face = self._get_card_surface_sized(r, s, rv_w, rv_h)
                self.screen.blit(face, (dest_x, dest_y))
                is_winner = (pid == self.qabool_id)
                if is_winner:
                    pulse = 0.5 + 0.5 * math.sin(pygame.time.get_ticks() * 0.005)
                    ga = int(140 + 80 * pulse)
                    gs = pygame.Surface((rv_w + 10, rv_h + 10), pygame.SRCALPHA)
                    pygame.draw.rect(gs, (*TEXT_GOLD, ga), gs.get_rect(), width=3, border_radius=10)
                    self.screen.blit(gs, (dest_x - 5, dest_y - 5))
                nc = TEXT_GOLD if is_winner else TEXT_WHITE
                name_surf = name_font.render(DISPLAY_NAMES[pid], True, nc)
                self.screen.blit(name_surf, name_surf.get_rect(centerx=dest_x + rv_w // 2, y=dest_y + rv_h + 5))

            # Announcement.
            winner_card = self._qabool_draw_cards[self.qabool_id]
            wr, ws = card_key(winner_card)
            winner_team = 0 if self.qabool_id in (0, 2) else 1
            team_label = "Team 1" if winner_team == 0 else "Team 2"
            announce_font = pygame.font.SysFont("Segoe UI", 17, bold=True)
            explain_font = pygame.font.SysFont("Segoe UI", 13)
            announce_surf = announce_font.render(
                f"{DISPLAY_NAMES[self.qabool_id]} is the first Sahib Al-Qabool!", True, TEXT_GOLD)
            self.screen.blit(announce_surf, announce_surf.get_rect(centerx=cx, y=reveal_y - 50))
            explain_surf = explain_font.render(
                f"{DISPLAY_NAMES[self.qabool_id]} drew {wr}{ws} - "
                f"highest card in {team_label}, which had the best draw overall.", True, TEXT_LIGHT)
            self.screen.blit(explain_surf, explain_surf.get_rect(centerx=cx, y=reveal_y - 26))

            # Continue button.
            btn_rect = pygame.Rect(cx - 70, reveal_y + rv_h + 40, 140, 42)
            mx, my = pygame.mouse.get_pos()
            hover = btn_rect.collidepoint(mx, my)
            bg = (56, 142, 60) if hover else BUTTON_GREEN
            pygame.draw.rect(self.screen, bg, btn_rect, border_radius=10)
            if hover:
                pygame.draw.rect(self.screen, (100, 200, 100), btn_rect, width=2, border_radius=10)
            btn_font = pygame.font.SysFont("Segoe UI", 15, bold=True)
            btn_text = btn_font.render("Continue", True, TEXT_WHITE)
            self.screen.blit(btn_text, btn_text.get_rect(center=btn_rect.center))

        # Game log.
        self._render_game_log()

    def _render_player_labels(self, cx, cy):
        """Render player names only (no team labels)."""
        font = pygame.font.SysFont("Segoe UI", 13, bold=True)

        cx_table = TABLE_WIDTH // 2
        qabool_radius = 5
        qabool_color = (220, 50, 50)

        # Top player (pid 0 = Ibrahim).
        surf = font.render(f"{DISPLAY_NAMES[0]}", True, TEAM1_BLUE)
        name_rect = surf.get_rect(centerx=cx_table, y=57)
        self.screen.blit(surf, name_rect)
        if 0 == self.qabool_id:
            pygame.draw.circle(self.screen, qabool_color, (name_rect.left - 10, name_rect.centery), qabool_radius)

        # Left player (pid 3 = Musaab).
        surf = font.render(f"{DISPLAY_NAMES[3]}", True, TEAM2_ORANGE)
        name_rect = surf.get_rect(centerx=45 + CARD_MINI_W // 2, y=cy - 145)
        self.screen.blit(surf, name_rect)
        if 3 == self.qabool_id:
            pygame.draw.circle(self.screen, qabool_color, (name_rect.left - 10, name_rect.centery), qabool_radius)

        # Right player (pid 1 = Gaafar).
        surf = font.render(f"{DISPLAY_NAMES[1]}", True, TEAM2_ORANGE)
        right_x = TABLE_WIDTH - 45 - CARD_MINI_W + CARD_MINI_W // 2
        name_rect = surf.get_rect(centerx=right_x, y=cy - 145)
        self.screen.blit(surf, name_rect)
        if 1 == self.qabool_id:
            pygame.draw.circle(self.screen, qabool_color, (name_rect.left - 10, name_rect.centery), qabool_radius)

        # Human (pid 2).
        surf = font.render(f"{DISPLAY_NAMES[HUMAN_ID]} (You)", True, TEXT_GOLD)
        name_rect = surf.get_rect(centerx=cx_table, y=SCREEN_HEIGHT - 195)
        self.screen.blit(surf, name_rect)
        if HUMAN_ID == self.qabool_id:
            pygame.draw.circle(self.screen, qabool_color, (name_rect.left - 10, name_rect.centery), qabool_radius)

    def _render_qabool_label(self, cx, cy):
        """Render 'Qabool: Name' directly beneath the trump card."""
        qabool_font = pygame.font.SysFont("Segoe UI", 11, bold=True)
        qabool_name = DISPLAY_NAMES.get(self.qabool_id, "?")
        text = f"Qabool: {qabool_name}"
        surf = qabool_font.render(text, True, TEXT_GOLD)
        # Position: beneath trump display (trump is at TABLE_WIDTH-100, y=60, height ~85+label).
        x = TABLE_WIDTH - 100 + 30  # centred with trump card (trump card is 60px wide)
        y = 160
        self.screen.blit(surf, surf.get_rect(centerx=x, y=y))

    def _render_tricks_won(self, cx, cy):
        """Render won tricks count — on the table, top-left corner, big font."""
        count_font = pygame.font.SysFont("Segoe UI", 20, bold=True)
        label_font = pygame.font.SysFont("Segoe UI", 10)

        # Seek risk chips — one per team.
        chip_w = 68
        chip_h = 16
        chip_font = self.fonts["chip"]
        chip_font_light = self.fonts["chip_light"]

        # Top-left of table area.
        x = 35
        y = 55

        t1_risk = self._calc_seek_risk(team_id=0)
        t2_risk = self._calc_seek_risk(team_id=1)

        # Render Team 1 score centered above its seek chip.
        chip1_x = x
        chip1_center = chip1_x + chip_w // 2
        l1 = label_font.render("Team 1", True, TEAM1_BLUE)
        t1 = count_font.render(str(self.team_tricks[0]), True, TEAM1_BLUE)
        self.screen.blit(l1, l1.get_rect(centerx=chip1_center, y=y))
        self.screen.blit(t1, t1.get_rect(centerx=chip1_center, y=y + 14))

        # Render Team 2 score centered above its seek chip.
        chip2_x = x + chip_w + 4
        chip2_center = chip2_x + chip_w // 2
        l2 = label_font.render("Team 2", True, TEAM2_ORANGE)
        t2 = count_font.render(str(self.team_tricks[1]), True, TEAM2_ORANGE)
        self.screen.blit(l2, l2.get_rect(centerx=chip2_center, y=y))
        self.screen.blit(t2, t2.get_rect(centerx=chip2_center, y=y + 14))

        # Seek risk chips below team scores.
        chip_y = y + 42

        for i, (risk, team_color) in enumerate([(t1_risk, TEAM1_BLUE), (t2_risk, TEAM2_ORANGE)]):
            chip_x = x + i * (chip_w + 4)
            chip_surf = pygame.Surface((chip_w, chip_h), pygame.SRCALPHA)

            if risk == 0:
                pygame.draw.rect(chip_surf, (50, 50, 50, 160), (0, 0, chip_w, chip_h), border_radius=8)
                chip_text = chip_font_light.render("0% Seek", True, TEXT_WHITE)
            elif risk <= 15:
                pygame.draw.rect(chip_surf, (30, 100, 30, 200), (0, 0, chip_w, chip_h), border_radius=8)
                chip_text = chip_font.render(f"{risk}% Seek", True, TEXT_WHITE)
            elif risk <= 40:
                pygame.draw.rect(chip_surf, (160, 120, 20, 200), (0, 0, chip_w, chip_h), border_radius=8)
                chip_text = chip_font.render(f"{risk}% Seek", True, TEXT_WHITE)
            else:
                pygame.draw.rect(chip_surf, (180, 40, 40, 200), (0, 0, chip_w, chip_h), border_radius=8)
                chip_text = chip_font.render(f"{risk}% Seek", True, TEXT_WHITE)

            chip_surf.blit(chip_text, chip_text.get_rect(center=(chip_w // 2, chip_h // 2)))
            self.screen.blit(chip_surf, (chip_x, chip_y))

    def _calc_seek_risk(self, team_id: int) -> int:
        """
        Estimate seek risk % for a team based only on what the human can see:
        - Their own hand (card strength, voids)
        - Tricks won so far (actual performance)
        - Cards played in completed tricks (visible to all)
        - Bid information (public knowledge)
        Returns 0 when seek is impossible, otherwise 1-95.
        """
        other_team = 1 - team_id

        # Seek is dead if the other team already won a trick.
        if self.team_tricks[other_team] > 0:
            return 0

        if self.players is None or self.phase not in ("playing", "shota_end"):
            return 0
        if self.trump_suit is None:
            return 0

        tricks_played = self.team_tricks[0] + self.team_tricks[1]
        tricks_remaining = 13 - tricks_played
        if tricks_remaining <= 0:
            return 0

        # --- Signal 1: Trick-winning rate (strongest observable signal) ---
        if tricks_played > 0:
            win_rate = self.team_tricks[team_id] / tricks_played
        else:
            win_rate = 0.0

        # --- Signal 2: Human hand strength (only what user can see) ---
        trump = self.trump_suit
        human_hand = self.players[HUMAN_ID].hand
        probable_winners = 0.0
        suits_seen = set()

        # Build set of high cards already played (visible from completed tricks).
        played_high_cards = set()  # (suit, rank_value) of cards already seen
        if hasattr(self.round.state, 'completed_tricks'):
            for trick in self.round.state.completed_tricks:
                for pc in trick.played_cards:
                    rv = rank_value(pc.card.rank)
                    if rv >= 12:  # Track Queens and above
                        played_high_cards.add((pc.card.suit, rv))

        if human_hand:
            for c in human_hand:
                rv = rank_value(c.rank)
                suits_seen.add(c.suit)
                if c.suit == trump:
                    # Promote if higher cards in this suit have been played.
                    higher_trumps_gone = sum(
                        1 for (s, r) in played_high_cards if s == trump and r > rv
                    )
                    if rv >= 14:
                        probable_winners += 0.95
                    elif rv >= 13:
                        probable_winners += 0.85 if higher_trumps_gone >= 1 else 0.75
                    elif rv >= 12:
                        probable_winners += 0.70 if higher_trumps_gone >= 2 else 0.55
                    elif rv >= 11:
                        probable_winners += 0.45 if higher_trumps_gone >= 2 else 0.30
                    else:
                        probable_winners += 0.15
                else:
                    # Non-trump: promoted if higher cards in that suit are gone.
                    higher_gone = sum(
                        1 for (s, r) in played_high_cards if s == c.suit and r > rv
                    )
                    if rv >= 14:
                        probable_winners += 0.70
                    elif rv >= 13:
                        probable_winners += 0.50 if higher_gone >= 1 else 0.35
                    elif rv >= 12:
                        probable_winners += 0.30 if higher_gone >= 2 else 0.15

            voids = 4 - len(suits_seen)
        else:
            voids = 0

        strength_ratio = probable_winners / max(1, tricks_remaining)

        # --- Signal 3: Bid context (public info) ---
        bid_signal = 0.0
        team_pids = [0, 2] if team_id == 0 else [1, 3]
        if hasattr(self, '_bid_history') and self._bid_history:
            team_bids = [v for pid, v in self._bid_history if pid in team_pids and v is not None]
            total_team_bid = sum(team_bids) if team_bids else 0
            # Low bid + winning all tricks = classic seek indicator.
            if total_team_bid <= 2 and self.team_tricks[team_id] >= 3:
                bid_signal = 0.15
            elif total_team_bid == 0 and self.team_tricks[team_id] >= 2:
                # All passed but winning everything — suspicious.
                bid_signal = 0.10

        # --- Combine signals based on game phase ---
        if tricks_played >= 4:
            # Mid/late game: actual trick performance dominates.
            raw = (win_rate * 55) + (strength_ratio * 25) + (voids * 4) + (bid_signal * 80)
        elif tricks_played >= 1:
            # Early game: blend hand strength and performance.
            raw = (win_rate * 35) + (strength_ratio * 40) + (voids * 5) + (bid_signal * 80)
        else:
            # Before any tricks played: pure hand analysis.
            if team_id == 0:  # Human's team — use hand directly.
                raw = (strength_ratio * 55) + (voids * 7) + (bid_signal * 80)
            else:  # Opponent team — infer weakness from human hand.
                weakness = 1.0 - min(1.0, strength_ratio + voids * 0.05)
                raw = weakness * 35 + (bid_signal * 80)

        # Late-game escalation: winning 9/9 or 10/10 etc. is very strong signal.
        if tricks_played >= 7 and self.team_tricks[team_id] == tricks_played:
            late_boost = (tricks_played / 13.0) * 35
            raw += late_boost

        risk = int(min(95, max(0, raw)))
        return risk

    def _render_trump_display(self):
        """Feature 1: Trump hidden until first card played. Card flip animation on reveal."""
        x, y = TABLE_WIDTH - 100, 60
        card_w, card_h = 60, 85

        # Flip animation: squish horizontally then expand.
        flip_scale_x = 1.0
        if self._trump_flip_timer > 0:
            t = self._trump_flip_timer / 20.0
            if t > 0.5:
                # First half: squish to 0.
                flip_scale_x = (t - 0.5) * 2
            else:
                # Second half: expand back.
                flip_scale_x = (0.5 - t) * 2
            flip_scale_x = max(0.05, flip_scale_x)

        actual_w = int(card_w * flip_scale_x)
        offset_x = (card_w - actual_w) // 2

        card_surf = pygame.Surface((actual_w, card_h), pygame.SRCALPHA)
        pygame.draw.rect(card_surf, CARD_WHITE, card_surf.get_rect(), border_radius=max(2, int(6 * flip_scale_x)))
        pygame.draw.rect(card_surf, (180, 180, 180), card_surf.get_rect(), width=1, border_radius=max(2, int(6 * flip_scale_x)))

        # Show content only in second half of flip (or when not flipping).
        show_content = (self._trump_flip_timer <= 10)

        if show_content and self._trump_revealed and self.trump_suit is not None:
            sym = SUIT_SYMBOLS.get(self.trump_suit, "?")
            color = RED_SUIT if self.trump_suit in (Suit.HEARTS, Suit.DIAMONDS) else BLACK_SUIT
            font_size = max(12, int(36 * flip_scale_x))
            big_font = pygame.font.SysFont("Segoe UI", font_size)
            suit_surf = big_font.render(sym, True, color)
            card_surf.blit(suit_surf, suit_surf.get_rect(center=(actual_w // 2, card_h // 2)))
        elif show_content:
            font_size = max(12, int(36 * flip_scale_x))
            big_font = pygame.font.SysFont("Segoe UI", font_size, bold=True)
            q_surf = big_font.render("?", True, TEXT_DIM)
            card_surf.blit(q_surf, q_surf.get_rect(center=(actual_w // 2, card_h // 2)))

        self.screen.blit(card_surf, (x + offset_x, y))

        label_text = "TRUMP" if self._trump_revealed else "TRUMP (hidden)"
        label = self.fonts["small"].render(label_text, True, TEXT_GOLD)
        self.screen.blit(label, label.get_rect(centerx=x + card_w // 2, y=y + card_h + 3))

        # Show bid number beneath the trump card.
        if self.bid_value > 0:
            bid_font = pygame.font.SysFont("Segoe UI", 13, bold=True)
            bid_surf = bid_font.render(f"Bid: {self.bid_value}", True, HIGHLIGHT_GREEN)
            self.screen.blit(bid_surf, bid_surf.get_rect(centerx=x + card_w // 2, y=y + card_h + 17))

    def _render_last_trick_area(self):
        """Render last trick in bottom-left: face-down cards that reveal on hover."""
        x_start = 30
        y_start = SCREEN_HEIGHT - 130
        card_w, card_h = CARD_MINI_W, CARD_MINI_H

        # Label.
        label_font = pygame.font.SysFont("Segoe UI", 9)
        label = label_font.render("Last Trick", True, TEXT_WHITE)
        self.screen.blit(label, (x_start, y_start - 14))

        # Check if mouse is hovering over the area.
        area_w = 4 * (card_w + 5)
        area_rect = pygame.Rect(x_start, y_start, area_w, card_h)
        mx, my = pygame.mouse.get_pos()
        hovering = area_rect.collidepoint(mx, my)

        # Draw 4 cards left to right starting from the shooter (bid winner).
        trick_order = [(self.shooter_id + i) % 4 for i in range(4)]

        for i, pid in enumerate(trick_order):
            if pid not in self._last_trick_cards:
                continue
            cx = x_start + i * (card_w + 5)
            if hovering:
                # Reveal: show face-up card.
                r, s = self._last_trick_cards[pid]
                card_surf = self._get_card_surface_sized(r, s, card_w, card_h)
                self.screen.blit(card_surf, (cx, y_start))
            else:
                # Face-down.
                self.screen.blit(self._card_back_mini, (cx, y_start))

    def _render_centre_trick(self, cx, cy):
        """Render centre trick with placeholders (Feature 15) and labels (Feature 13)."""
        # Positions: P0=top, P1=right, P2=bottom, P3=left.
        offsets = {
            0: (0, -70),
            1: (90, 0),
            2: (0, 50),
            3: (-90, 0),
        }

        # Determine which pids are still animating (card sliding in).
        animating_pids = set()
        for anim in self._play_animations:
            if not anim.done and anim.pid is not None:
                animating_pids.add(anim.pid)

        for pid in range(4):
            dx, dy = offsets[pid]
            slot_x = cx + dx - CARD_WIDTH // 2
            slot_y = cy + dy - CARD_HEIGHT // 2

            if pid in self._trick_played:
                # Skip rendering at final position if animation is still in flight.
                if pid in animating_pids:
                    continue

                # Render played card.
                r, s = self._trick_played[pid]
                card_surf = self._get_card_surface(r, s)
                rect = card_surf.get_rect(center=(cx + dx, cy + dy))

                # Shadow.
                shadow = pygame.Surface((CARD_WIDTH + 4, CARD_HEIGHT + 4), pygame.SRCALPHA)
                pygame.draw.rect(shadow, (0, 0, 0, 40), shadow.get_rect(), border_radius=6)
                self.screen.blit(shadow, (rect.x - 2, rect.y + 2))

                # Feature 9: Gold border on winning card + pulsing glow.
                # Fire effect for whipping (trump on non-trump trick).
                if self._trick_winner_id == pid and self._trick_winner_timer > 0:
                    is_whip = getattr(self, '_trick_is_whip', False)
                    is_double_whip = getattr(self, '_trick_is_double_whip', False)

                    if is_double_whip:
                        # Double whip — intense fire (red-orange-yellow flicker).
                        t = self._trick_winner_timer * 0.4
                        r_val = int(255)
                        g_val = int(80 + 100 * abs(math.sin(t)))
                        b_val = int(20 + 30 * abs(math.sin(t * 1.7)))
                        fire_color = (r_val, g_val, b_val)
                        pulse_alpha = int(180 + 75 * math.sin(t * 2))
                        # Outer fire glow.
                        glow_surf = pygame.Surface((CARD_WIDTH + 20, CARD_HEIGHT + 20), pygame.SRCALPHA)
                        pygame.draw.rect(glow_surf, (*fire_color, min(255, pulse_alpha)),
                                         glow_surf.get_rect(), border_radius=12)
                        self.screen.blit(glow_surf, (rect.x - 10, rect.y - 10))
                        # Inner hot glow.
                        inner = pygame.Surface((CARD_WIDTH + 10, CARD_HEIGHT + 10), pygame.SRCALPHA)
                        pygame.draw.rect(inner, (255, 200, 50, int(pulse_alpha * 0.7)),
                                         inner.get_rect(), width=4, border_radius=9)
                        self.screen.blit(inner, (rect.x - 5, rect.y - 5))
                    elif is_whip:
                        # Single whip — fire glow (orange-red).
                        t = self._trick_winner_timer * 0.3
                        r_val = int(255)
                        g_val = int(100 + 80 * abs(math.sin(t)))
                        fire_color = (r_val, g_val, 30)
                        pulse_alpha = int(150 + 80 * math.sin(t * 1.5))
                        glow_surf = pygame.Surface((CARD_WIDTH + 16, CARD_HEIGHT + 16), pygame.SRCALPHA)
                        pygame.draw.rect(glow_surf, (*fire_color, min(255, pulse_alpha)),
                                         glow_surf.get_rect(), border_radius=10)
                        self.screen.blit(glow_surf, (rect.x - 8, rect.y - 8))
                        # Border.
                        glow = pygame.Surface((CARD_WIDTH + 8, CARD_HEIGHT + 8), pygame.SRCALPHA)
                        pygame.draw.rect(glow, (255, 140, 0, 220), glow.get_rect(),
                                         width=4, border_radius=8)
                        self.screen.blit(glow, (rect.x - 4, rect.y - 4))
                    else:
                        # Normal win — gold glow.
                        pulse_alpha = int(120 + 80 * math.sin(self._trick_winner_timer * 0.3))
                        pulse_alpha = max(0, min(255, pulse_alpha))
                        glow_surf = pygame.Surface((CARD_WIDTH + 16, CARD_HEIGHT + 16), pygame.SRCALPHA)
                        pygame.draw.rect(glow_surf, (*HIGHLIGHT_GOLD, pulse_alpha),
                                         glow_surf.get_rect(), border_radius=10)
                        self.screen.blit(glow_surf, (rect.x - 8, rect.y - 8))
                        glow = pygame.Surface((CARD_WIDTH + 8, CARD_HEIGHT + 8), pygame.SRCALPHA)
                        pygame.draw.rect(glow, (*HIGHLIGHT_GOLD, 200), glow.get_rect(),
                                         width=4, border_radius=8)
                        self.screen.blit(glow, (rect.x - 4, rect.y - 4))

                # For whipping trump cards, render with yellow background.
                is_whip_card = (getattr(self, '_trick_is_whip', False)
                                and self.trump_suit is not None
                                and s == SUIT_SYMBOLS.get(self.trump_suit, ""))

                if is_whip_card:
                    # Create a yellow-background card surface.
                    yellow_card = pygame.Surface((CARD_WIDTH, CARD_HEIGHT), pygame.SRCALPHA)
                    pygame.draw.rect(yellow_card, (255, 240, 100),
                                     yellow_card.get_rect(), border_radius=CARD_RADIUS)
                    pygame.draw.rect(yellow_card, (200, 180, 50),
                                     yellow_card.get_rect(), width=1, border_radius=CARD_RADIUS)
                    # Draw rank and suit symbols on the yellow card.
                    sym_font = pygame.font.SysFont("Segoe UI", 18, bold=True)
                    suit_color = RED_SUIT if s in ("♥", "♦") else BLACK_SUIT
                    rank_surf = sym_font.render(r, True, suit_color)
                    suit_surf = sym_font.render(s, True, suit_color)
                    yellow_card.blit(rank_surf, (5, 3))
                    yellow_card.blit(suit_surf, (5, 20))
                    # Big centre symbol.
                    big_font = pygame.font.SysFont("Segoe UI", 32, bold=True)
                    big_surf = big_font.render(s, True, suit_color)
                    yellow_card.blit(big_surf, big_surf.get_rect(center=(CARD_WIDTH // 2, CARD_HEIGHT // 2)))
                    self.screen.blit(yellow_card, rect)
                else:
                    self.screen.blit(card_surf, rect)
            else:
                # Feature 15: Empty slot placeholder (dashed rectangle).
                placeholder = pygame.Surface((CARD_WIDTH, CARD_HEIGHT), pygame.SRCALPHA)
                dash_color = (100, 140, 100, 100)
                # Draw dashed border.
                for i in range(0, CARD_WIDTH, 8):
                    pygame.draw.line(placeholder, dash_color, (i, 0), (min(i+4, CARD_WIDTH), 0))
                    pygame.draw.line(placeholder, dash_color, (i, CARD_HEIGHT-1),
                                     (min(i+4, CARD_WIDTH), CARD_HEIGHT-1))
                for i in range(0, CARD_HEIGHT, 8):
                    pygame.draw.line(placeholder, dash_color, (0, i), (0, min(i+4, CARD_HEIGHT)))
                    pygame.draw.line(placeholder, dash_color, (CARD_WIDTH-1, i),
                                     (CARD_WIDTH-1, min(i+4, CARD_HEIGHT)))
                self.screen.blit(placeholder, (slot_x, slot_y))

    def _render_human_hand(self):
        """Render the human player hand as a flat overlapping row, grouped by suit."""
        if self.players is None:
            return
        hand = self.players[HUMAN_ID].hand
        if not hand:
            return
        suit_order = {Suit.SPADES: 0, Suit.HEARTS: 1, Suit.CLUBS: 2, Suit.DIAMONDS: 3}
        sorted_hand = sorted(hand, key=lambda c: (suit_order[c.suit], -rank_value(c.rank)))
        legal = set(hand)
        if self.phase == "playing" and self._play_idx < 4:
            pid = self._play_order[self._play_idx] if self._play_idx < len(self._play_order) else -1
            if pid == HUMAN_ID and self.round.state.current_trick:
                leading_suit = self.round.state.current_trick.leading_suit
                must_trump = None
                if (self.round.state.is_first_trick and
                        self.round.state.winning_bidder_id == HUMAN_ID and
                        len(self.round.state.current_trick.played_cards) == 0):
                    must_trump = self.trump_suit
                legal = set(legal_cards(hand, leading_suit, must_trump))
        n = len(sorted_hand)
        card_w = CARD_LARGE_W
        card_h = CARD_LARGE_H
        overlap = 30
        suit_gap = 12
        suit_gaps = []
        for i in range(1, n):
            if sorted_hand[i].suit != sorted_hand[i - 1].suit:
                suit_gaps.append(i)
        total_w = (n - 1) * overlap + card_w + len(suit_gaps) * suit_gap
        start_x = (TABLE_WIDTH - total_w) // 2
        base_y = SCREEN_HEIGHT - card_h - 30
        mx, my = pygame.mouse.get_pos()
        hovered_idx = -1
        for i in range(n - 1, -1, -1):
            gaps_before = sum(1 for g in suit_gaps if g <= i)
            card_x = start_x + i * overlap + gaps_before * suit_gap
            cr = pygame.Rect(card_x, base_y, card_w, card_h)
            if cr.collidepoint(mx, my) and sorted_hand[i] in legal:
                hovered_idx = i
                break
        self._fan_card_data = []
        for i, card in enumerate(sorted_hand):
            gaps_before = sum(1 for g in suit_gaps if g <= i)
            card_x = start_x + i * overlap + gaps_before * suit_gap
            draw_y = base_y - 14 if i == hovered_idx else base_y
            is_legal = card in legal
            r, s = card_key(card)
            card_surf = self._get_card_surface_sized(r, s, card_w, card_h)
            self.screen.blit(card_surf, (card_x, draw_y))
            self._fan_card_data.append({
                "card": card,
                "cx": card_x + card_w // 2,
                "cy": draw_y + card_h // 2,
                "w": card_w,
                "h": card_h,
                "angle_deg": 0,
                "legal": is_legal,
            })

    def _render_bidding_ui(self, cx, cy):
        """Render bidding interface — 3 steps: number → trump → confirm (Feature 5)."""
        is_human_turn = False
        if self._bid_index < len(self._bid_order) and self._bid_order[self._bid_index] == HUMAN_ID:
            is_human_turn = True
        if self.qabool_id == HUMAN_ID and self._bid_index >= len(self._bid_order):
            is_human_turn = True

        # Bids are shown by _render_bid_labels (called after this).

        if not is_human_turn:
            wait = self.fonts["large"].render("Bidding...", True, TEXT_DIM)
            self.screen.blit(wait, wait.get_rect(centerx=cx, y=cy))
            return

        # ---- HUMAN BIDDING UI ----
        title_font = pygame.font.SysFont("Segoe UI", 18, bold=True)
        btn_font = pygame.font.SysFont("Segoe UI", 16, bold=True)
        mx, my = pygame.mouse.get_pos()

        if self._bid_step == "number":

            # All three rows visible at once.
            # Row 1: Pass + Bid numbers (7-13).
            is_qabool_turn = (self.qabool_id == HUMAN_ID and self._bid_index >= len(self._bid_order))
            no_bids = (self._bidding_engine.highest_bid is None)
            if is_qabool_turn and no_bids:
                pass_label = "Dak!"
                pass_bg = BUTTON_RED
            else:
                pass_label = "Pass"
                pass_bg = BUTTON_GREY

            # Pass button first in row.
            # Row 1: Total ~500px centred at cx. start_x = cx - 250.
            # Pass: 60px, gap 5px, then 7 buttons of 55px with 5px gap = 60 + 5 + 7*55 + 6*5 = 480.
            row1_start = cx - 250
            pass_press = 2 if (self._button_press_timer > 0 and self._button_press_id == "pass") else 0
            pass_rect = pygame.Rect(row1_start, cy - 10 + pass_press, 60, 45 - pass_press)
            hover_pass = pass_rect.collidepoint(mx, my)
            bg = (100, 50, 50) if (hover_pass and pass_label == "Dak!") else ((80, 80, 80) if hover_pass else pass_bg)
            pygame.draw.rect(self.screen, bg, pass_rect, border_radius=6)
            pass_surf = btn_font.render(pass_label, True, TEXT_GREEN)
            self.screen.blit(pass_surf, pass_surf.get_rect(center=pass_rect.center))

            # Bid numbers — poker chip style.
            for i, val in enumerate(range(7, 14)):
                chip_cx = row1_start + 65 + i * 60 + 27
                chip_cy = cy + 12
                chip_r = 22
                hover = (mx - chip_cx) ** 2 + (my - chip_cy) ** 2 <= chip_r ** 2
                selected = (self._selected_bid == val)

                if selected:
                    pygame.draw.circle(self.screen, (30, 100, 30), (chip_cx, chip_cy), chip_r)
                    pygame.draw.circle(self.screen, TEXT_GREEN, (chip_cx, chip_cy), chip_r, 2)
                    pygame.draw.circle(self.screen, TEXT_GREEN, (chip_cx, chip_cy), chip_r - 5, 1)
                elif hover:
                    pygame.draw.circle(self.screen, (56, 142, 60), (chip_cx, chip_cy), chip_r)
                    pygame.draw.circle(self.screen, TEXT_GREEN, (chip_cx, chip_cy), chip_r, 1)
                else:
                    pygame.draw.circle(self.screen, BUTTON_GREEN, (chip_cx, chip_cy), chip_r)
                    pygame.draw.circle(self.screen, (40, 120, 40), (chip_cx, chip_cy), chip_r - 5, 1)
                num_surf = btn_font.render(str(val), True, TEXT_WHITE)
                self.screen.blit(num_surf, num_surf.get_rect(center=(chip_cx, chip_cy)))

            # Row 2: Suit buttons. 4 buttons 65px each with 10px gap = 290px centred at cx.
            suits = [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]
            row2_start = cx - 145
            for i, suit in enumerate(suits):
                sym = SUIT_SYMBOLS[suit]
                count = sum(1 for c in self.players[HUMAN_ID].hand if c.suit == suit)
                rect = pygame.Rect(row2_start + i * 75, cy + 50, 65, 55)
                hover = rect.collidepoint(mx, my)
                selected = (self._selected_trump_idx == i)
                color = RED_SUIT if suit in (Suit.HEARTS, Suit.DIAMONDS) else BLACK_SUIT

                if selected:
                    bg = (220, 255, 220)
                    pygame.draw.rect(self.screen, bg, rect, border_radius=8)
                    pygame.draw.rect(self.screen, HIGHLIGHT_GREEN, rect, width=2, border_radius=8)
                elif hover:
                    pygame.draw.rect(self.screen, (255, 255, 240), rect, border_radius=8)
                    pygame.draw.rect(self.screen, HIGHLIGHT_GREEN, rect, width=1, border_radius=8)
                else:
                    pygame.draw.rect(self.screen, CARD_WHITE, rect, border_radius=8)
                    pygame.draw.rect(self.screen, (180, 180, 180), rect, width=1, border_radius=8)

                sym_font = pygame.font.SysFont("Segoe UI", 24, bold=True)
                sym_surf = sym_font.render(sym, True, color)
                self.screen.blit(sym_surf, sym_surf.get_rect(centerx=rect.centerx, y=rect.y + 4))
                cnt_surf = self.fonts["small"].render(f"({count})", True, TEXT_DIM)
                self.screen.blit(cnt_surf, cnt_surf.get_rect(centerx=rect.centerx, y=rect.y + 38))

            # Row 3: Confirm button.
            valid = self._is_bid_valid()
            confirm_press = 2 if (self._button_press_timer > 0 and self._button_press_id == "confirm") else 0
            confirm_rect = pygame.Rect(cx - 70, cy + 120 + confirm_press, 140, 40 - confirm_press)
            hover_confirm = confirm_rect.collidepoint(mx, my)
            if valid:
                bg = (56, 142, 60) if hover_confirm else BUTTON_GREEN
                pygame.draw.rect(self.screen, bg, confirm_rect, border_radius=8)
                if hover_confirm:
                    pygame.draw.rect(self.screen, TEXT_GREEN, confirm_rect, width=2, border_radius=8)
                confirm_surf = btn_font.render("Confirm", True, TEXT_WHITE)
            else:
                pygame.draw.rect(self.screen, (50, 50, 50), confirm_rect, border_radius=8)
                confirm_surf = btn_font.render("Confirm", True, TEXT_DIM)
            self.screen.blit(confirm_surf, confirm_surf.get_rect(center=confirm_rect.center))

            # Warning line under confirm when invalid.
            if not valid and self._selected_bid is not None and self._selected_trump_idx is not None:
                warn_text = self._get_bid_warning()
                if warn_text:
                    warn_font = pygame.font.SysFont("Segoe UI", 14, bold=True)
                    warn_surf = warn_font.render(warn_text, True, (255, 100, 50))
                    self.screen.blit(warn_surf, warn_surf.get_rect(centerx=cx, y=cy + 165))

    def _render_table_vignette(self, table_rect: pygame.Rect):
        """Render a radial vignette over the table for depth and focus.
        
        Creates an elliptical gradient that darkens the edges of the table,
        drawing the eye toward the centre where the action happens.
        """
        # Use a cached vignette surface to avoid regenerating every frame.
        cache_key = (table_rect.width, table_rect.height)
        if not hasattr(self, '_vignette_cache') or self._vignette_cache_key != cache_key:
            w, h = table_rect.width, table_rect.height
            vignette = pygame.Surface((w, h), pygame.SRCALPHA)

            cx, cy = w // 2, h // 2
            # Semi-axes of the inner bright ellipse (where vignette starts fading in).
            a_inner = w * 0.42
            b_inner = h * 0.42
            # Semi-axes of the outer edge (full darkness).
            a_outer = w * 0.55
            b_outer = h * 0.55

            max_alpha = 90  # Max darkness at the very edges.

            # Draw concentric elliptical rings from outside in.
            # Use ~30 bands for smooth gradient without being too expensive.
            num_bands = 30
            for band in range(num_bands):
                # t goes from 0 (outermost) to 1 (innermost edge of vignette).
                t = band / num_bands
                # Ellipse size interpolates from outer to inner.
                a = a_outer + (a_inner - a_outer) * t
                b = b_outer + (b_inner - b_outer) * t
                # Alpha fades from max at edge to 0 at inner boundary.
                alpha = int(max_alpha * (1 - t) ** 1.5)
                if alpha <= 0:
                    continue
                # Draw a filled ellipse ring by drawing outer then cutting inner.
                ring = pygame.Surface((w, h), pygame.SRCALPHA)
                ellipse_rect = pygame.Rect(cx - int(a), cy - int(b), int(a * 2), int(b * 2))
                # Only draw pixels outside this ellipse as dark.
                # Approach: fill entire surface with dark, then clear the ellipse.
                ring.fill((0, 0, 0, alpha))
                # Cut out the inner ellipse (transparent).
                inner_a = a_outer + (a_inner - a_outer) * ((band + 1) / num_bands)
                inner_b = b_outer + (b_inner - b_outer) * ((band + 1) / num_bands)
                inner_rect = pygame.Rect(cx - int(inner_a), cy - int(inner_b),
                                         int(inner_a * 2), int(inner_b * 2))
                pygame.draw.ellipse(ring, (0, 0, 0, 0), inner_rect)
                vignette.blit(ring, (0, 0))

            # Also add a soft overall edge darkening.
            edge_surf = pygame.Surface((w, h), pygame.SRCALPHA)
            edge_surf.fill((0, 0, 0, max_alpha))
            # Clear centre ellipse.
            clear_rect = pygame.Rect(cx - int(a_inner), cy - int(b_inner),
                                     int(a_inner * 2), int(b_inner * 2))
            pygame.draw.ellipse(edge_surf, (0, 0, 0, 0), clear_rect)
            vignette.blit(edge_surf, (0, 0))

            self._vignette_cache = vignette
            self._vignette_cache_key = cache_key

        self.screen.blit(self._vignette_cache, table_rect.topleft)

    def _render_opponent(self, pid, x, y, horizontal=True):
        """Render face-down cards as a flat overlapping row (no fan)."""
        if self.players is None:
            return
        count = len(self.players[pid].hand)
        if count == 0:
            return

        card_w, card_h = 60, 85
        if not hasattr(self, '_card_back_opp'):
            from gui_wist.card_renderer import create_card_back
            self._card_back_opp = create_card_back(card_w, card_h)
        if not hasattr(self, '_card_back_opp_landscape'):
            self._card_back_opp_landscape = pygame.transform.rotate(self._card_back_opp, 90)

        # Fixed overlap and suit gap (face-down, so we just space evenly).
        if horizontal:
            overlap = 14
            total_w = (count - 1) * overlap + card_w
            start_x = x - total_w // 2 + card_w // 2
            for i in range(count):
                self.screen.blit(self._card_back_opp, (start_x + i * overlap, y))
        else:
            overlap = 10
            total_h = (count - 1) * overlap + card_w  # card_w because rotated.
            start_y = y - total_h // 2 + card_w // 2
            lw, lh = card_h, card_w  # Landscape dimensions.
            for i in range(count):
                self.screen.blit(self._card_back_opp_landscape, (x, start_y + i * overlap))

    def _render_turn_glow(self, cx, cy):
        """Render a soft radial glow behind the active player's card area."""
        if self._active_turn_pid is None:
            return
        pid = self._active_turn_pid
        # Positions for the glow centre per player.
        glow_positions = {
            0: (cx, 130),                   # Top
            3: (75, cy),                    # Left
            1: (TABLE_WIDTH - 75, cy),      # Right
            2: (cx, SCREEN_HEIGHT - 180),   # Bottom (human)
        }
        if pid not in glow_positions:
            return
        gx, gy = glow_positions[pid]

        # Create a 100x100 surface with a radial alpha gradient.
        glow_size = 100
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        center = glow_size // 2
        for r in range(center, 0, -2):
            alpha = int(60 * (1 - r / center))
            pygame.draw.circle(glow_surf, (76, 175, 80, alpha), (center, center), r)
        self.screen.blit(glow_surf, (gx - center, gy - center))

    def _play_whip_sound(self):
        """Play a short whip/crack sound effect for trumping."""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=1)
            # Generate a short percussive crack using raw bytes (no numpy needed).
            import array
            sample_rate = 22050
            duration = 0.12
            num_samples = int(sample_rate * duration)
            samples = array.array('h')  # signed short
            for i in range(num_samples):
                t = i / sample_rate
                # Decaying white noise burst.
                import random as _rnd
                noise = _rnd.randint(-32767, 32767)
                envelope = max(0.0, 1.0 - t * 12)  # Fast decay.
                val = int(noise * envelope * 0.5)
                samples.append(max(-32767, min(32767, val)))
            sound = pygame.mixer.Sound(buffer=samples.tobytes())
            sound.set_volume(0.4)
            sound.play()
        except Exception:
            pass

    def _spawn_confetti(self):
        """Spawn 80 confetti particles for victory celebration."""
        colors = [
            (255, 215, 0),    # Gold
            (76, 175, 80),    # Green
            (66, 165, 245),   # Blue
            (229, 57, 53),    # Red
            (255, 112, 67),   # Orange
        ]
        self._confetti_particles = []
        for _ in range(80):
            self._confetti_particles.append({
                "x": random.randint(0, SCREEN_WIDTH),
                "y": random.randint(-100, 0),
                "vx": random.uniform(-2, 2),
                "vy": random.uniform(1, 4),
                "color": random.choice(colors),
                "w": random.randint(4, 8),
                "h": random.randint(4, 10),
                "life": random.randint(180, 240),  # 3-4 seconds at 60fps
                "max_life": 240,
                "alpha": 255,
            })

    def _render_confetti(self):
        """Render confetti particles as small coloured rectangles."""
        for p in self._confetti_particles:
            if p["alpha"] <= 0:
                continue
            rect_surf = pygame.Surface((p["w"], p["h"]), pygame.SRCALPHA)
            r, g, b = p["color"]
            rect_surf.fill((r, g, b, p["alpha"]))
            self.screen.blit(rect_surf, (int(p["x"]), int(p["y"])))

    def _get_card_surface(self, rank: str, suit: str) -> pygame.Surface:
        """Get card surface at default size."""
        key = f"{rank}{suit}"
        if key not in self._card_cache:
            self._card_cache[key] = create_card_surface(rank, suit)
        return self._card_cache[key]

    def _get_card_surface_sized(self, rank: str, suit: str, w: int, h: int) -> pygame.Surface:
        """Get card surface at specified size (Feature 20)."""
        key = f"{rank}{suit}_{w}x{h}"
        if key not in self._card_cache:
            self._card_cache[key] = create_card_surface(rank, suit, w, h)
        return self._card_cache[key]
