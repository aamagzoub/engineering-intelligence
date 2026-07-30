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

from gui_pygame.constants import *
from gui_pygame.card_renderer import create_card_surface, create_card_back, create_shadow

from agents.rule_based.rule_based_agent import RuleBasedAgent
from environments.wist.environment import WistEnvironment
from environments.wist.actions import PlayCardAction
from environments.wist.round import Round
from environments.wist.rules import legal_cards, trick_winner, rank_value
from environments.wist.setup import create_standard_players
from environments.wist.tasmiya_engine import TasmiyaEngine, determine_first_shota_qabool, determine_trump_suit
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
                 start_scale: float = 1.2, end_scale: float = 1.0, delay: int = 0):
        self.surface = surface
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.total_frames = frames
        self.frame = 0
        self.done = False
        self.start_scale = start_scale
        self.end_scale = end_scale
        self.delay = delay  # frames to wait before starting

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

        # Animation state.
        self._trick_played: dict[int, tuple[str, str]] = {}
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

    def _shape_arabic(self, text: str) -> str:
        """Reshape Arabic text for proper RTL connected rendering."""
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            reshaped = arabic_reshaper.reshape(text)
            return get_display(reshaped)
        except ImportError:
            return text

    # ----------------------------------------------------------
    # Game lifecycle
    # ----------------------------------------------------------

    def start_game(self):
        """Start a new full game."""
        self.game_scores = [0, 0]
        self.shota_number = 0
        self._dak_count = 0
        self._game_log = []
        self._log_game_event("=== NEW GAME ===")
        self._show_quit_overlay = False

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
        # Rotate Qabool for new shotas (not for Dak re-deals).
        if self.shota_number == 1:
            self.qabool_id = determine_first_shota_qabool()
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
        self._trick_played = {}
        self._trump_revealed = False
        self._trick_winner_id = None
        self._trick_winner_timer = 0
        self._player_bids_display = {0: "", 1: "", 2: "", 3: ""}

        self.players = create_standard_players()
        self.round = Round(self.players)
        self.round.deal()

        # Feature 3: Card-based Dak detection with display.
        card_dak_detected = False
        while self.round.has_card_based_dak():
            card_dak_detected = True
            self.round = Round(self.players)
            self.round.deal()

        if card_dak_detected:
            self._message = "Card Dak detected! Re-dealing..."
            self._message_timer = 60
            self._log_game_event("Card Dak detected — re-dealt")

        self.agents = [RuleBasedAgent(), RuleBasedAgent(), None, RuleBasedAgent()]

        self._log_game_event(f"--- Shota {self.shota_number} ---")
        self._log_game_event(f"Qabool: {DISPLAY_NAMES[self.qabool_id]}")

        # Feature 4: Show Qabool rotation announcement.
        self._message = f"Qabool: {DISPLAY_NAMES[self.qabool_id]}"
        self._message_timer = 60

        # Feature 16: Deal animation.
        self._start_deal_animation()

        self.phase = "dealing"
        self._ai_timer = 35  # Wait for deal animation then go to bidding.

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
        self._ai_timer = 60

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
            self._ai_timer = 60

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
                # Feature 2: Dak ceremony.
                self._dak_count += 1
                if hasattr(self, '_game_stats'):
                    self._game_stats["daks"] = self._dak_count
                self._dak_shake_timer = 30
                self._message = "DAK! Re-dealing..."
                self._message_timer = 120
                self._bidding_done = True
                self._ai_timer = 120
                self._log_game_event(f"DAK #{self._dak_count}! All passed.")
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
        # Qabool advantage: bid >= trump_count + 2 (only when matching/outbidding someone).
        if is_qabool and someone_bid:
            min_bid = trump_count + 2  # Qabool extra card advantage.
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

        # Reset bid step.
        self._bid_step = "number"
        self._selected_bid = None
        self._selected_trump_idx = None

        if is_qabool:
            self._finalize_bidding()
        else:
            self._bid_index += 1
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

        if is_qabool:
            if self._bidding_engine.highest_bid is None:
                # Feature 2: Dak ceremony.
                self._dak_count += 1
                if hasattr(self, '_game_stats'):
                    self._game_stats["daks"] = self._dak_count
                self._dak_shake_timer = 30
                self._message = "DAK! Re-dealing..."
                self._message_timer = 120
                self._bidding_done = True
                self._ai_timer = 120
                self._log_game_event(f"DAK #{self._dak_count}!")
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
        self._message_timer = 90
        self._ai_timer = 90
        self._log_game_event(f"Shooter: {DISPLAY_NAMES[self.shooter_id]}, Bid: {self.bid_value}")

        self.phase = "playing"
        # Use frame-based timer instead of OS timer.
        self._play_idx = 99
        self._ai_timer = 50  # ~0.8s at 60fps, then _start_next_trick fires.

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
        self._ai_timer = 30
        print(f"[Trick {self.trick_number}] Leader: {DISPLAY_NAMES[leader]}, Order: {[DISPLAY_NAMES[p] for p in self._play_order]}")

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
        if seek_team == human_team:
            self._award_points("seek")
        elif seek_team is not None:
            self._award_points("seek_against")

        # Score breakdown in game log.
        result_str = "SUCCESS" if bid_met else "FAILED"
        self._log_game_event(f"Shota {self.shota_number}: Bid {self.bid_value} → {result_str}")
        self._log_game_event(f"  Tricks: T1={self.team_tricks[0]} T2={self.team_tricks[1]}")
        self._log_game_event(f"  Score: T1 +{shota_score_t1}, T2 +{shota_score_t2}")
        self._log_game_event(f"  Total: T1={self.game_scores[0]} T2={self.game_scores[1]}")

        if self.shota_number >= 5 or self.game_scores[0] >= 25 or self.game_scores[1] >= 25:
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

        if self.phase == "dealing":
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
                print(f"[Trick {self.trick_number}] No current trick — skipping {DISPLAY_NAMES[pid]}")
                self._play_idx += 1
                self._ai_timer = 5
                return
            if not self.players[HUMAN_ID].hand:
                print(f"[Trick {self.trick_number}] {DISPLAY_NAMES[pid]} has no cards — skipping")
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

                # Feature 1: Reveal trump on first card of first trick.
                if not self._trump_revealed:
                    self._trump_revealed = True
                    self._trump_flip_timer = 20  # Start flip animation.

                # Feature 17: Play animation.
                self._start_play_animation(pid, r, s)
            except Exception as e:
                print(f"[PyGame AI Error] {DISPLAY_NAMES.get(pid, pid)}: {e}")
                # Fallback: try to play any legal card.
                try:
                    hand = self.players[pid].hand
                    if hand and self.round.state.current_trick is not None:
                        leading = self.round.state.current_trick.leading_suit
                        playable = legal_cards(hand, leading, None)
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
        # End positions (centre trick slots).
        offsets = {0: (0, -70), 1: (90, 0), 2: (0, 50), 3: (-90, 0)}
        dx, dy = offsets.get(pid, (0, 0))
        end = (cx + dx - CARD_WIDTH // 2, cy + dy - CARD_HEIGHT // 2)
        start = start_positions.get(pid, (cx, cy))
        surf = self._get_card_surface(rank, suit)
        # Swoosh: start at 1.2x scale, animate down to 1.0x.
        anim = AnimatingCard(surf, start, end, frames=15, start_scale=1.2, end_scale=1.0)
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

        # Store winner info for phase 2.
        self._pending_trick_winner = winner
        self._pending_trick = trick

        # Feature 9: Gold highlight the winning card on the table.
        self._trick_winner_id = winner
        self._trick_winner_timer = 60

        # Pause with all 4 cards visible + gold highlight, then move to phase 2.
        self._ai_timer = 45  # ~0.75s pause at 60fps.
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
        self._message_timer = 50
        self._ai_timer = 40
        self._play_idx = 99
        self._log_game_event(f"Trick {self.trick_number}: {DISPLAY_NAMES[winner]} wins")

        # Clear the trick highlight after cards start moving.
        self._trick_played = {}

        # Frame-based delay — _ai_timer counts down, then update loop calls _start_next_trick.

    # ----------------------------------------------------------
    # Event handling
    # ----------------------------------------------------------

    def handle_event(self, event):
        """Handle PyGame events."""
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

        if self._show_quit_overlay:
            return  # Block all other input while overlay shown.

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._handle_click(event.pos)

    def _handle_click(self, pos):
        """Handle mouse click — bidding or card selection."""
        # Panel buttons (stats panel, bottom area).
        panel_x = TABLE_WIDTH
        pad = 10
        btn_w = STATS_PANEL_WIDTH - pad * 2
        btn_h = 30

        # Load AI Model button.
        btn_y = SCREEN_HEIGHT - 85
        load_btn = pygame.Rect(panel_x + pad, btn_y, btn_w, btn_h)
        if load_btn.collidepoint(pos):
            self._open_model_dialog()
            return

        # Restart button.
        restart_y = btn_y + btn_h + 8
        restart_btn = pygame.Rect(panel_x + pad, restart_y, btn_w, btn_h)
        if restart_btn.collidepoint(pos):
            self._restart_to_name = True
            self.phase = "idle"
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
                filetypes=[("Model files", "*.pt *.pth *.h5 *.onnx"), ("All files", "*.*")]
            )
            root.destroy()
            if path:
                self._ai_model_path = path
                self._message = f"Model loaded: {os.path.basename(path)}"
                self._message_timer = 90
                self._log_game_event(f"AI Model: {os.path.basename(path)}")
        except Exception:
            pass

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

        # Bid must be >= trump_count + 3 (or +2 for Qabool advantage when matching).
        if is_qabool and someone_bid:
            min_bid = trump_count + 2
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
        self._render_opponent(0, cx - 80, 75, horizontal=True)
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
        game_card_h = 240
        pygame.draw.rect(self.screen, (22, 40, 22),
                         (panel_x + 5, game_card_y, panel_w - 10, game_card_h), border_radius=8)
        pygame.draw.rect(self.screen, (45, 90, 45),
                         (panel_x + 5, game_card_y, panel_w - 10, game_card_h), width=1, border_radius=8)
        y += 8

        self.screen.blit(title_font.render("Game", True, TEXT_WHITE), (panel_x + pad + 4, y))
        y += 24

        # Score + pulse.
        self.screen.blit(label_font.render("Score", True, TEXT_LIGHT), (panel_x + pad + 4, y))
        score_text = f"{self.game_scores[0]} - {self.game_scores[1]}"
        if self._score_pulse_timer > 0:
            t = self._score_pulse_timer / 40.0
            glow = pygame.font.SysFont("Consolas", int(14 * (1.0 + 0.15 * t)), bold=True)\
                .render(score_text, True, TEXT_GOLD)
            glow.set_alpha(int(180 * t))
            self.screen.blit(glow, glow.get_rect(right=panel_x + panel_w - pad - 4, centery=y + 8))
        score_surf = value_font.render(score_text, True, TEXT_WHITE)
        self.screen.blit(score_surf, (panel_x + panel_w - pad - 4 - score_surf.get_width(), y))
        y += 18

        # Progress bars.
        bar_x = panel_x + pad + 4
        bar_w = inner_w - 8
        pygame.draw.rect(self.screen, (30, 50, 30), (bar_x, y, bar_w, 5), border_radius=3)
        t1 = min(1.0, self.game_scores[0] / 25.0)
        if t1 > 0:
            pygame.draw.rect(self.screen, TEAM1_BLUE, (bar_x, y, int(bar_w * t1), 5), border_radius=3)
        y += 8
        pygame.draw.rect(self.screen, (30, 50, 30), (bar_x, y, bar_w, 5), border_radius=3)
        t2 = min(1.0, self.game_scores[1] / 25.0)
        if t2 > 0:
            pygame.draw.rect(self.screen, TEAM2_ORANGE, (bar_x, y, int(bar_w * t2), 5), border_radius=3)
        y += 14

        # Info rows (always shown, stable positions).
        trump_sym = "?"
        trump_color = TEXT_LIGHT
        if self._trump_revealed and self.trump_suit:
            trump_sym = SUIT_SYMBOLS.get(self.trump_suit, "?")
            trump_color = RED_SUIT if self.trump_suit in (Suit.HEARTS, Suit.DIAMONDS) else TEXT_WHITE
        rows = [
            ("Shota", f"{self.shota_number}/5", TEXT_LIGHT),
            ("Trick", f"{self.trick_number}/13", TEXT_LIGHT),
            ("Qabool", DISPLAY_NAMES[self.qabool_id], TEXT_GOLD),
            ("Bid", str(self.bid_value) if self.bid_value > 0 else "-", HIGHLIGHT_GREEN),
            ("Shooter", DISPLAY_NAMES.get(self.shooter_id, "-") if self.bid_value > 0 else "-", HIGHLIGHT_GREEN),
            ("Trump", trump_sym, trump_color),
        ]
        for lbl, val, clr in rows:
            self._draw_stat_row(panel_x, pad + 4, panel_w, y, label_font, value_font, lbl, val, clr)
            y += 19

        # Team tricks + momentum.
        y += 4
        t1_arrow = " ↑" if self._team_streak[0] >= 3 else (" →" if self._team_streak[0] >= 2 else "")
        t2_arrow = " ↑" if self._team_streak[1] >= 3 else (" →" if self._team_streak[1] >= 2 else "")
        if "tricks" in self._stat_highlight_timers:
            a = int(100 * (self._stat_highlight_timers["tricks"] / 30.0))
            hl = pygame.Surface((inner_w - 8, 32), pygame.SRCALPHA)
            hl.fill((255, 213, 79, a))
            self.screen.blit(hl, (panel_x + pad + 4, y - 2))
        self._draw_stat_row(panel_x, pad + 4, panel_w, y, label_font, value_font,
                            "T1", f"{self.team_tricks[0]}{t1_arrow}", TEAM1_BLUE)
        y += 16
        self._draw_stat_row(panel_x, pad + 4, panel_w, y, label_font, value_font,
                            "T2", f"{self.team_tricks[1]}{t2_arrow}", TEAM2_ORANGE)

        # ========== STATS CARD (fixed position) ==========
        y = game_card_y + game_card_h + 8
        stats = getattr(self, '_game_stats', {})
        stats_card_h = 200
        pygame.draw.rect(self.screen, (22, 40, 22),
                         (panel_x + 5, y, panel_w - 10, stats_card_h), border_radius=8)
        pygame.draw.rect(self.screen, (45, 90, 45),
                         (panel_x + 5, y, panel_w - 10, stats_card_h), width=1, border_radius=8)
        y += 10

        self.screen.blit(title_font.render("Stats", True, TEXT_GREEN), (panel_x + pad + 4, y))
        y += 24

        # Player tricks with mini-bars.
        player_tricks = stats.get("player_tricks", {0: 0, 1: 0, 2: 0, 3: 0})
        team_colors = {0: TEAM1_BLUE, 1: TEAM2_ORANGE, 2: TEAM1_BLUE, 3: TEAM2_ORANGE}
        max_t = max(max(player_tricks.values()), 1)

        for pid in (2, 0, 1, 3):
            name = "You" if pid == HUMAN_ID else DISPLAY_NAMES.get(pid, f"P{pid}")
            color = team_colors[pid]
            count = player_tricks.get(pid, 0)
            self.screen.blit(label_font.render(name, True, color), (panel_x + pad + 4, y))
            c_surf = value_font.render(str(count), True, TEXT_WHITE)
            self.screen.blit(c_surf, (panel_x + panel_w - pad - 4 - c_surf.get_width(), y))
            y += 16
            bx = panel_x + pad + 4
            bw = inner_w - 8
            pygame.draw.rect(self.screen, (30, 50, 30), (bx, y, bw, 4), border_radius=2)
            ratio = count / max_t
            if ratio > 0:
                pygame.draw.rect(self.screen, color, (bx, y, int(bw * ratio), 4), border_radius=2)
            y += 9

        y += 6

        # Daks (with shake).
        daks = stats.get("daks", self._dak_count)
        dak_x_off = 0
        if self._dak_shake_timer > 0:
            intensity = int(4 * (self._dak_shake_timer / 30.0))
            dak_x_off = intensity * (1 if self._dak_shake_timer % 4 < 2 else -1)
        self.screen.blit(label_font.render("Daks", True, BUTTON_RED),
                         (panel_x + pad + 4 + dak_x_off, y))
        dak_v = value_font.render(str(daks), True, BUTTON_RED)
        self.screen.blit(dak_v, (panel_x + panel_w - pad - 4 - dak_v.get_width() + dak_x_off, y))
        y += 17

        # Best bid + bids met (always shown).
        highest_bid = stats.get("highest_bid", 0)
        bidder = stats.get("highest_bidder")
        bidder_name = DISPLAY_NAMES.get(bidder, "") if bidder is not None else ""
        bid_text = f"{highest_bid} ({bidder_name})" if highest_bid > 0 else "-"
        self._draw_stat_row(panel_x, pad + 4, panel_w, y, label_font, value_font,
                            "Best bid", bid_text, TEXT_LIGHT)
        y += 17

        bids_met_t1 = stats.get("bids_met_t1", 0)
        bids_met_t2 = stats.get("bids_met_t2", 0)
        shotas_played = stats.get("shotas_played", 0)
        met_text = f"T1:{bids_met_t1} T2:{bids_met_t2}" if shotas_played > 0 else "-"
        if "bids_met" in self._stat_highlight_timers:
            a = int(100 * (self._stat_highlight_timers["bids_met"] / 30.0))
            hl = pygame.Surface((inner_w - 8, 17), pygame.SRCALPHA)
            hl.fill((100, 255, 100, a))
            self.screen.blit(hl, (panel_x + pad + 4, y - 1))
        self._draw_stat_row(panel_x, pad + 4, panel_w, y, label_font, value_font,
                            "Bids met", met_text, TEXT_LIGHT)

        # ========== PROBABILITIES CARD ==========
        prob_card_y = game_card_y + game_card_h + 8 + stats_card_h + 8
        prob_card_h = 140
        pygame.draw.rect(self.screen, (22, 40, 22),
                         (panel_x + 5, prob_card_y, panel_w - 10, prob_card_h), border_radius=8)
        pygame.draw.rect(self.screen, (45, 90, 45),
                         (panel_x + 5, prob_card_y, panel_w - 10, prob_card_h), width=1, border_radius=8)

        py = prob_card_y + 10
        self.screen.blit(title_font.render("Live Odds", True, (180, 140, 255)), (panel_x + pad + 4, py))
        py += 24

        # Calculate probabilities.
        tricks_played = self.team_tricks[0] + self.team_tricks[1]
        tricks_left = 13 - tricks_played

        # Win probability: based on current game score + shota progress.
        # Simple heuristic: who's closer to 25 considering tricks in this shota.
        t1_total = self.game_scores[0]
        t2_total = self.game_scores[1]
        # Project current shota contribution.
        if tricks_played > 0:
            t1_rate = self.team_tricks[0] / tricks_played
            t2_rate = self.team_tricks[1] / tricks_played
        else:
            t1_rate = 0.5
            t2_rate = 0.5
        t1_projected = t1_total + self.team_tricks[0] + t1_rate * tricks_left
        t2_projected = t2_total + self.team_tricks[1] + t2_rate * tricks_left
        total_proj = t1_projected + t2_projected
        win_prob = (t1_projected / total_proj * 100) if total_proj > 0 else 50.0
        win_prob = max(5.0, min(95.0, win_prob))

        # Win probability bar.
        bar_x = panel_x + pad + 4
        bar_w = inner_w - 8
        win_label = "T1 Win Game" if win_prob >= 50 else "T2 Win Game"
        win_display_color = TEAM1_BLUE if win_prob >= 50 else TEAM2_ORANGE
        self.screen.blit(label_font.render(win_label, True, TEXT_LIGHT), (bar_x, py))
        pct_surf = value_font.render(f"{win_prob:.0f}%", True, win_display_color)
        self.screen.blit(pct_surf, (panel_x + panel_w - pad - 4 - pct_surf.get_width(), py))
        py += 17
        pygame.draw.rect(self.screen, (30, 50, 30), (bar_x, py, bar_w, 6), border_radius=3)
        win_fill = win_prob / 100.0
        # Color shifts: green when >60%, yellow 40-60%, orange <40%.
        if win_prob >= 60:
            bar_color = HIGHLIGHT_GREEN
        elif win_prob >= 40:
            bar_color = TEXT_GOLD
        else:
            bar_color = TEAM2_ORANGE
        pygame.draw.rect(self.screen, bar_color, (bar_x, py, int(bar_w * win_fill), 6), border_radius=3)
        py += 14

        # Seek probability.
        # Seek = one team wins all 13. Probability spikes as one team approaches 10+.
        t1_seek = 0.0
        t2_seek = 0.0
        if tricks_played > 0:
            if self.team_tricks[1] == 0 and self.team_tricks[0] >= 6:
                t1_seek = min(95, (self.team_tricks[0] / 13.0) ** 3 * 100)
            if self.team_tricks[0] == 0 and self.team_tricks[1] >= 6:
                t2_seek = min(95, (self.team_tricks[1] / 13.0) ** 3 * 100)
        seek_prob = max(t1_seek, t2_seek)
        seek_team_label = "T1 Seek" if t1_seek >= t2_seek else "T2 Seek"
        seek_color = TEAM1_BLUE if t1_seek >= t2_seek else TEAM2_ORANGE

        self.screen.blit(label_font.render(seek_team_label, True, TEXT_LIGHT), (bar_x, py))
        seek_text = f"{seek_prob:.0f}%" if seek_prob > 0 else "-"
        sk_surf = value_font.render(seek_text, True, seek_color if seek_prob > 0 else TEXT_LIGHT)
        self.screen.blit(sk_surf, (panel_x + panel_w - pad - 4 - sk_surf.get_width(), py))
        py += 17
        pygame.draw.rect(self.screen, (30, 50, 30), (bar_x, py, bar_w, 6), border_radius=3)
        if seek_prob > 0:
            pygame.draw.rect(self.screen, seek_color,
                             (bar_x, py, int(bar_w * seek_prob / 100.0), 6), border_radius=3)
        py += 14

        # Bid success chance.
        # Will the shooter's team make their bid?
        bid_prob = 50.0
        if self.bid_value > 0 and tricks_played > 0:
            shooter_team = 0 if self.shooter_id in (0, 2) else 1
            shooter_tricks = self.team_tricks[shooter_team]
            needed = self.bid_value - shooter_tricks
            if needed <= 0:
                bid_prob = 95.0
            elif needed > tricks_left:
                bid_prob = 5.0
            else:
                # Rate-based projection.
                rate = shooter_tricks / tricks_played if tricks_played > 0 else 0.5
                expected = shooter_tricks + rate * tricks_left
                if expected >= self.bid_value:
                    bid_prob = min(90, 50 + (expected - self.bid_value) * 15)
                else:
                    bid_prob = max(10, 50 - (self.bid_value - expected) * 15)
        elif self.bid_value == 0:
            bid_prob = 0  # No bid yet.

        self.screen.blit(label_font.render(
            f"Bid ({DISPLAY_NAMES.get(self.shooter_id, '?')})" if self.bid_value > 0 else "Bid",
            True, TEXT_LIGHT), (bar_x, py))
        bid_pct_text = f"{bid_prob:.0f}%" if self.bid_value > 0 else "-"
        bid_color = HIGHLIGHT_GREEN if bid_prob >= 60 else (TEXT_GOLD if bid_prob >= 40 else BUTTON_RED)
        bp_surf = value_font.render(bid_pct_text, True, bid_color if self.bid_value > 0 else TEXT_LIGHT)
        self.screen.blit(bp_surf, (panel_x + panel_w - pad - 4 - bp_surf.get_width(), py))
        py += 15
        if self.bid_value > 0:
            pygame.draw.rect(self.screen, (30, 50, 30), (bar_x, py, bar_w, 6), border_radius=3)
            pygame.draw.rect(self.screen, bid_color,
                             (bar_x, py, int(bar_w * bid_prob / 100.0), 6), border_radius=3)

        # ========== BUTTONS (fixed at bottom) ==========
        btn_y = SCREEN_HEIGHT - 85
        btn_w = panel_w - pad * 2
        btn_h = 30
        mx, my = pygame.mouse.get_pos()

        load_rect = pygame.Rect(panel_x + pad, btn_y, btn_w, btn_h)
        bg = (255, 230, 50) if load_rect.collidepoint(mx, my) else (255, 213, 79)
        pygame.draw.rect(self.screen, bg, load_rect, border_radius=5)
        pygame.draw.rect(self.screen, (200, 160, 0), load_rect, width=1, border_radius=5)
        load_text = btn_font.render("Load AI-Expert Model", True, (20, 20, 20))
        self.screen.blit(load_text, load_text.get_rect(center=load_rect.center))
        if self._ai_model_path:
            self.screen.blit(
                label_font.render(os.path.basename(self._ai_model_path)[:22], True, TEXT_GREEN),
                (panel_x + pad, btn_y - 16))

        restart_rect = pygame.Rect(panel_x + pad, btn_y + btn_h + 8, btn_w, btn_h)
        rbg = (50, 120, 50) if restart_rect.collidepoint(mx, my) else (40, 95, 40)
        pygame.draw.rect(self.screen, rbg, restart_rect, border_radius=5)
        pygame.draw.rect(self.screen, (30, 70, 30), restart_rect, width=1, border_radius=5)
        restart_text = btn_font.render("Restart Game", True, TEXT_LIGHT)
        self.screen.blit(restart_text, restart_text.get_rect(center=restart_rect.center))

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
            f"T1 ({DISPLAY_NAMES[2]} + {DISPLAY_NAMES[0]}): {self.game_scores[0]}  │  "
            f"T2 ({DISPLAY_NAMES[1]} + {DISPLAY_NAMES[3]}): {self.game_scores[1]}",
            True, TEXT_LIGHT)
        self.screen.blit(score, score.get_rect(centerx=cx, y=cy + 5))

        # Stats summary.
        stats = getattr(self, '_game_stats', {})
        stat_font = pygame.font.SysFont("Segoe UI", 14)
        stat_line = (f"Shotas: {stats.get('shotas_played', self.shota_number)}  │  "
                     f"Daks: {self._dak_count}  │  "
                     f"Seeks: {stats.get('seeks', 0)}  │  "
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

        # Result.
        your_team_won = self.team_tricks[0] > self.team_tricks[1]
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

        # Tricks & Score.
        info_font = pygame.font.SysFont("Segoe UI", 14)
        tricks_surf = info_font.render(
            f"Tricks — T1: {self.team_tricks[0]}  |  T2: {self.team_tricks[1]}", True, TEXT_LIGHT)
        self.screen.blit(tricks_surf, tricks_surf.get_rect(centerx=cx, y=info_y))
        info_y += 22

        score_font = pygame.font.SysFont("Segoe UI", 16, bold=True)
        score_surf = score_font.render(
            f"Total Score: T1 = {self.game_scores[0]}  |  T2 = {self.game_scores[1]}", True, TEXT_GOLD)
        self.screen.blit(score_surf, score_surf.get_rect(centerx=cx, y=info_y))
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

    def _render_player_labels(self, cx, cy):
        """Render player names, roles, and partnership indicators."""
        font = pygame.font.SysFont("Segoe UI", 13, bold=True)
        team_font = pygame.font.SysFont("Segoe UI", 9)

        cx_table = TABLE_WIDTH // 2
        qabool_radius = 5
        qabool_color = (220, 50, 50)  # Red circle for Sahib Al-Qabool.

        # Top player (pid 0 = Hima) — YOUR PARTNER.
        surf = font.render(f"{DISPLAY_NAMES[0]}", True, TEAM1_BLUE)
        name_rect = surf.get_rect(centerx=cx_table, y=57)
        self.screen.blit(surf, name_rect)
        partner_lbl = team_font.render("(Your Partner - Team 1)", True, (80, 140, 200))
        self.screen.blit(partner_lbl, partner_lbl.get_rect(centerx=cx_table, y=73))
        if 0 == self.qabool_id:
            pygame.draw.circle(self.screen, qabool_color, (name_rect.left - 10, name_rect.centery), qabool_radius)

        # Left player (pid 3 = Musaab) — OPPONENT.
        surf = font.render(f"{DISPLAY_NAMES[3]}", True, TEAM2_ORANGE)
        name_rect = surf.get_rect(centerx=45 + CARD_MINI_W // 2, y=cy - 80)
        self.screen.blit(surf, name_rect)
        opp_lbl = team_font.render("Team 2", True, (180, 100, 60))
        self.screen.blit(opp_lbl, opp_lbl.get_rect(centerx=45 + CARD_MINI_W // 2, y=cy - 66))
        if 3 == self.qabool_id:
            pygame.draw.circle(self.screen, qabool_color, (name_rect.left - 10, name_rect.centery), qabool_radius)

        # Right player (pid 1 = Gaafar) — OPPONENT.
        surf = font.render(f"{DISPLAY_NAMES[1]}", True, TEAM2_ORANGE)
        right_x = TABLE_WIDTH - 45 - CARD_MINI_W + CARD_MINI_W // 2
        name_rect = surf.get_rect(centerx=right_x, y=cy - 80)
        self.screen.blit(surf, name_rect)
        opp_lbl2 = team_font.render("Team 2", True, (180, 100, 60))
        self.screen.blit(opp_lbl2, opp_lbl2.get_rect(centerx=right_x, y=cy - 66))
        if 1 == self.qabool_id:
            pygame.draw.circle(self.screen, qabool_color, (name_rect.left - 10, name_rect.centery), qabool_radius)

        # Human (pid 2 = Omer) — YOUR area.
        surf = font.render(f"{DISPLAY_NAMES[HUMAN_ID]} (You) - Team 1", True, TEXT_GOLD)
        name_rect = surf.get_rect(centerx=cx_table, y=SCREEN_HEIGHT - CARD_HEIGHT - 70)
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

        # Top-left of table area.
        x = 35
        y = 55

        l1 = label_font.render("Team 1", True, TEAM1_BLUE)
        t1 = count_font.render(str(self.team_tricks[0]), True, TEAM1_BLUE)
        self.screen.blit(l1, (x, y))
        self.screen.blit(t1, (x + 2, y + 14))

        l2 = label_font.render("Team 2", True, TEAM2_ORANGE)
        t2 = count_font.render(str(self.team_tricks[1]), True, TEAM2_ORANGE)
        self.screen.blit(l2, (x + 70, y))
        self.screen.blit(t2, (x + 72, y + 14))

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

    def _render_centre_trick(self, cx, cy):
        """Render centre trick with placeholders (Feature 15) and labels (Feature 13)."""
        # Positions: P0=top, P1=right, P2=bottom, P3=left.
        offsets = {
            0: (0, -70),
            1: (90, 0),
            2: (0, 50),
            3: (-90, 0),
        }

        for pid in range(4):
            dx, dy = offsets[pid]
            slot_x = cx + dx - CARD_WIDTH // 2
            slot_y = cy + dy - CARD_HEIGHT // 2

            if pid in self._trick_played:
                # Render played card.
                r, s = self._trick_played[pid]
                card_surf = self._get_card_surface(r, s)
                rect = card_surf.get_rect(center=(cx + dx, cy + dy))

                # Shadow.
                shadow = pygame.Surface((CARD_WIDTH + 4, CARD_HEIGHT + 4), pygame.SRCALPHA)
                pygame.draw.rect(shadow, (0, 0, 0, 40), shadow.get_rect(), border_radius=6)
                self.screen.blit(shadow, (rect.x - 2, rect.y + 2))

                # Feature 9: Gold border on winning card + pulsing glow.
                if self._trick_winner_id == pid and self._trick_winner_timer > 0:
                    # Pulsing glow — alpha changes based on sine wave of timer.
                    pulse_alpha = int(120 + 80 * math.sin(self._trick_winner_timer * 0.3))
                    pulse_alpha = max(0, min(255, pulse_alpha))
                    glow_surf = pygame.Surface((CARD_WIDTH + 16, CARD_HEIGHT + 16), pygame.SRCALPHA)
                    pygame.draw.rect(glow_surf, (*HIGHLIGHT_GOLD, pulse_alpha),
                                     glow_surf.get_rect(), border_radius=10)
                    self.screen.blit(glow_surf, (rect.x - 8, rect.y - 8))
                    # Gold border.
                    glow = pygame.Surface((CARD_WIDTH + 8, CARD_HEIGHT + 8), pygame.SRCALPHA)
                    pygame.draw.rect(glow, (*HIGHLIGHT_GOLD, 200), glow.get_rect(),
                                     width=4, border_radius=8)
                    self.screen.blit(glow, (rect.x - 4, rect.y - 4))

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

            # Feature 13: Team labels next to each slot.
            label_text = DISPLAY_NAMES.get(pid, f"P{pid+1}")
            label_color = TEAM1_BLUE if pid in (0, 2) else TEAM2_ORANGE
            label_surf = self.fonts["small"].render(label_text, True, label_color)
            # Position label above/below/left/right of slot.
            if pid == 0:  # Top.
                self.screen.blit(label_surf, label_surf.get_rect(centerx=cx + dx, y=slot_y - 14))
            elif pid == 2:  # Bottom.
                self.screen.blit(label_surf, label_surf.get_rect(
                    centerx=cx + dx, y=slot_y + CARD_HEIGHT + 2))
            elif pid == 3:  # Left.
                self.screen.blit(label_surf, (slot_x - label_surf.get_width() - 4, slot_y + 40))
            else:  # Right (pid 1).
                self.screen.blit(label_surf, (slot_x + CARD_WIDTH + 4, slot_y + 40))

    def _render_human_hand(self):
        """Render the human player's hand as a fan arc with rotation."""
        if self.players is None:
            return

        hand = self.players[HUMAN_ID].hand
        if not hand:
            return

        # Sort.
        suit_order = {Suit.SPADES: 0, Suit.HEARTS: 1, Suit.CLUBS: 2, Suit.DIAMONDS: 3}
        sorted_hand = sorted(hand, key=lambda c: (suit_order[c.suit], -rank_value(c.rank)))

        # Get legal cards.
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

        # Feature 20: Responsive hand sizing — always use large cards for human.
        n = len(sorted_hand)
        card_w = CARD_LARGE_W
        card_h = CARD_LARGE_H

        # Fan arc parameters.
        # Total arc spread scales with number of cards (max ±15° = 30° total).
        max_spread = 30.0  # Total arc in degrees.
        fan_spread = min(max_spread, n * 2.5)  # Scale up gradually.
        half_spread = fan_spread / 2.0

        # Fan pivot point — below the screen (cards radiate from a point below).
        fan_cx = TABLE_WIDTH // 2
        fan_cy = SCREEN_HEIGHT + 300  # Virtual pivot well below screen.

        # Radius from pivot to card centres.
        fan_radius = 480

        # Base Y for the centre card (bottom of screen area).
        base_y = SCREEN_HEIGHT - card_h - 40

        mx, my = pygame.mouse.get_pos()

        # Feature 11: Suit spacing — find suit boundaries for slight extra angle gap.
        suit_gaps = set()
        for i in range(1, n):
            if sorted_hand[i].suit != sorted_hand[i-1].suit:
                suit_gaps.add(i)

        # Calculate angle for each card.
        # Distribute evenly across the arc, with small extra gap at suit boundaries.
        suit_gap_angle = 1.5  # Extra degrees per suit boundary.
        total_suit_gaps = len(suit_gaps)
        effective_spread = fan_spread - total_suit_gaps * suit_gap_angle

        # Store card rects for hit detection (used by _get_clicked_card).
        self._fan_card_data = []

        for i, card in enumerate(sorted_hand):
            # Calculate angle offset from centre.
            # Card 0 is leftmost (negative angle), card n-1 is rightmost (positive).
            if n == 1:
                angle_deg = 0.0
            else:
                # Count suit gaps before this card.
                gaps_before = sum(1 for g in suit_gaps if g <= i)
                # Base position without suit gaps.
                base_t = i / (n - 1)  # 0 to 1
                base_angle = -half_spread + base_t * fan_spread
                # Add cumulative suit gap offset.
                gap_offset = gaps_before * suit_gap_angle - (total_suit_gaps * suit_gap_angle / 2)
                angle_deg = base_angle + gap_offset

            angle_rad = math.radians(angle_deg)

            # Card position — arc from pivot.
            # X offset from centre based on angle.
            card_cx = fan_cx + fan_radius * math.sin(angle_rad)
            card_cy = fan_cy - fan_radius * math.cos(angle_rad)

            # Clamp Y so cards stay in the hand area.
            card_cy = max(base_y, card_cy - card_h // 2)

            is_legal = card in legal
            is_hovered = False

            # Approximate hover check (bounding box of rotated card).
            # More precise check done in _get_clicked_card.
            hover_margin = 10
            approx_rect = pygame.Rect(card_cx - card_w // 2 - hover_margin,
                                      card_cy - hover_margin,
                                      card_w + hover_margin * 2,
                                      card_h + hover_margin * 2)
            if approx_rect.collidepoint(mx, my) and is_legal:
                is_hovered = True

            # Get card surface.
            r, s = card_key(card)
            card_surf = self._get_card_surface_sized(r, s, card_w, card_h)

            # Rotate the card surface.
            rotated = pygame.transform.rotate(card_surf, -angle_deg)
            rot_rect = rotated.get_rect()

            # Position: centre the rotated surface at the card's position.
            draw_x = card_cx - rot_rect.width // 2
            draw_y = card_cy - (rot_rect.height - card_h) // 2

            # Hover lift: move card up along its angle direction.
            if is_hovered and is_legal:
                lift = 14
                draw_x -= lift * math.sin(angle_rad)
                draw_y -= lift * math.cos(angle_rad)

            if not is_legal:
                # Reduce opacity of the rotated card (respects card shape, no ugly square).
                rotated.set_alpha(90)
                self.screen.blit(rotated, (draw_x, draw_y))
                rotated.set_alpha(255)
            else:
                self.screen.blit(rotated, (draw_x, draw_y))

            # Store data for hit detection.
            self._fan_card_data.append({
                'card': card,
                'cx': card_cx,
                'cy': card_cy + card_h // 2,
                'w': card_w,
                'h': card_h,
                'angle_deg': angle_deg,
                'legal': is_legal,
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
        """Render face-down cards for an opponent."""
        if self.players is None:
            return
        count = len(self.players[pid].hand)

        for i in range(count):
            if horizontal:
                pos = (x + i * 12, y)
            else:
                pos = (x, y + i * 8)
            self.screen.blit(self._card_back_mini, pos)

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
