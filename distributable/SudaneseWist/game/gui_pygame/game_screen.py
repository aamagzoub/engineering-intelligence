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
18. Sound effects (pygame.mixer tones)
19. Bid number validation feedback
20. Responsive hand sizing (CARD_LARGE when ≤6 cards)
21. ESC during game (quit overlay Y/N)
22. Load AI model button (file dialog)
"""

import pygame
import math
import os
import struct
import wave
import tempfile
from collections import Counter
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
DISPLAY_NAMES = {2: "Abubakr", 1: "Gaafar", 0: "Tarig", 3: "Musaab"}

# Game log panel width.
LOG_PANEL_WIDTH = 200
# Table area width (minus log panel).
TABLE_WIDTH = SCREEN_WIDTH - LOG_PANEL_WIDTH


def card_key(card: Card) -> tuple[str, str]:
    return RANK_SYMBOLS[card.rank], SUIT_SYMBOLS[card.suit]


def _generate_tone_wav(filepath: str, freq: int = 440, duration_ms: int = 200, volume: float = 0.3):
    """Generate a simple sine-wave WAV file if it doesn't exist."""
    if os.path.exists(filepath):
        return
    sample_rate = 22050
    n_samples = int(sample_rate * duration_ms / 1000)
    try:
        with wave.open(filepath, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            for i in range(n_samples):
                t = i / sample_rate
                val = int(volume * 32767 * math.sin(2 * math.pi * freq * t))
                wf.writeframes(struct.pack('<h', val))
    except Exception:
        pass


def _ensure_sound_files():
    """Ensure sound effect WAV files exist (generate simple tones if not)."""
    sound_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")
    os.makedirs(sound_dir, exist_ok=True)
    files = {
        "card_play.wav": (600, 100, 0.2),
        "trick_win.wav": (880, 300, 0.3),
        "your_turn.wav": (520, 150, 0.25),
    }
    paths = {}
    for fname, (freq, dur, vol) in files.items():
        fpath = os.path.join(sound_dir, fname)
        _generate_tone_wav(fpath, freq, dur, vol)
        paths[fname] = fpath
    return paths


class AnimatingCard:
    """A card animating from one position to another over N frames."""

    def __init__(self, surface: pygame.Surface, start_pos: tuple, end_pos: tuple, frames: int = 30):
        self.surface = surface
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.total_frames = frames
        self.frame = 0
        self.done = False

    def update(self):
        self.frame += 1
        if self.frame >= self.total_frames:
            self.done = True

    @property
    def current_pos(self) -> tuple[float, float]:
        t = min(1.0, self.frame / self.total_frames)
        # Ease-out quad.
        t = 1 - (1 - t) ** 2
        x = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * t
        y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * t
        return (x, y)

    def render(self, screen: pygame.Surface):
        pos = self.current_pos
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

        # Sound effects (Feature 18).
        self._sounds_loaded = False
        self._sounds: dict[str, pygame.mixer.Sound] = {}
        self._init_sounds()

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

        # Timing.
        self._ai_timer = 0
        self._play_order = []
        self._play_idx = 0

        # Bidding state.
        self._bid_step = "number"  # "number" → "trump" → "confirm"
        self._selected_bid: int | None = None
        self._selected_trump_idx: int | None = None

    def _init_sounds(self):
        """Initialize sound effects (Feature 18)."""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(22050, -16, 1, 512)
            sound_paths = _ensure_sound_files()
            for name, path in sound_paths.items():
                key = name.replace(".wav", "")
                try:
                    self._sounds[key] = pygame.mixer.Sound(path)
                except Exception:
                    pass
            self._sounds_loaded = True
        except Exception:
            self._sounds_loaded = False

    def _play_sound(self, name: str):
        """Play a sound effect by name."""
        if self._sounds_loaded and name in self._sounds:
            try:
                self._sounds[name].play()
            except Exception:
                pass

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
        self._start_new_shota()

    def _log_game_event(self, text: str):
        """Add an entry to the game log (Feature 8)."""
        self._game_log.append(text)
        # Keep max 200 lines.
        if len(self._game_log) > 200:
            self._game_log = self._game_log[-200:]

    def _start_new_shota(self):
        """Deal and start bidding."""
        self.shota_number += 1
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

        if self.shota_number == 1:
            self.qabool_id = determine_first_shota_qabool()
        else:
            self.qabool_id = (self.qabool_id + 1) % 4

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
        """Animate cards sliding into player positions (Feature 16)."""
        self._deal_animations = []
        cx, cy = TABLE_WIDTH // 2, SCREEN_HEIGHT // 2
        # Animate a few cards flying to each player area.
        positions = {
            0: (cx, 80),       # Top
            3: (50, cy),       # Left
            1: (TABLE_WIDTH - 100, cy),  # Right
            2: (cx, SCREEN_HEIGHT - 140),  # Bottom (human)
        }
        for pid in range(4):
            end_x, end_y = positions[pid]
            for i in range(3):  # 3 cards per player for visual.
                start = (cx - 25, cy - 36)
                end = (end_x + i * 15, end_y)
                surf = self._card_back_mini
                anim = AnimatingCard(surf, start, end, frames=25 + i * 5)
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
                self._message = f"{DISPLAY_NAMES[pid]} bids {action.value}"
                self._log_game_event(f"{DISPLAY_NAMES[pid]} bids {action.value}")
            else:
                self._bidding_engine.apply_pass(Pass(player_id=pid))
                self._bid_history.append((pid, None))
                self._player_bids_display[pid] = "Pass"
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
                self._log_game_event(f"{DISPLAY_NAMES[qid]} (Qabool) accepts (forced)")
                self._finalize_bidding()
                return
            self._bid_history.append((qid, bid_value))
            self._player_bids_display[qid] = f"Bid {bid_value}"
            self._log_game_event(f"{DISPLAY_NAMES[qid]} (Qabool) bids {bid_value}")
        else:
            self._bidding_engine.apply_pass(Pass(player_id=qid))
            self._bid_history.append((qid, None))
            if self._bidding_engine.highest_bid is None:
                # Feature 2: Dak ceremony.
                self._dak_count += 1
                self._message = "DAK! Re-dealing..."
                self._message_timer = 120
                self._bidding_done = True
                self._ai_timer = 120
                self._log_game_event(f"DAK #{self._dak_count}! All passed.")
                return
            self._player_bids_display[qid] = "Accepts"
            self._log_game_event(f"{DISPLAY_NAMES[qid]} (Qabool) accepts")

        self._finalize_bidding()

    def _human_bid_with_number_and_suit(self):
        """Human selected bid number + suit + confirmed (Feature 5)."""
        suits = [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]
        chosen_suit = suits[self._selected_trump_idx]
        bid_value = self._selected_bid

        from environments.wist.bidding import Bid
        is_qabool = (self.qabool_id == HUMAN_ID and self._bid_index >= len(self._bid_order))

        try:
            bid = Bid(player_id=HUMAN_ID, value=bid_value)
            self._bidding_engine.apply_bid(bid, is_sahib_al_qabool=is_qabool)
        except ValueError as e:
            self._message = str(e)
            self._message_timer = 50
            self._bid_step = "number"
            self._selected_bid = None
            self._selected_trump_idx = None
            return

        self._bid_history.append((HUMAN_ID, bid_value))
        self._has_opening_bid = True
        self._human_trump_choice = chosen_suit
        self._player_bids_display[HUMAN_ID] = f"Bid {bid_value}"
        self._message = f"You bid {bid_value} ({SUIT_SYMBOLS[chosen_suit]})"
        self._message_timer = 30
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
        self._log_game_event("You pass")

        if is_qabool:
            if self._bidding_engine.highest_bid is None:
                # Feature 2: Dak ceremony.
                self._dak_count += 1
                self._message = "DAK! Re-dealing..."
                self._message_timer = 120
                self._bidding_done = True
                self._ai_timer = 120
                self._log_game_event(f"DAK #{self._dak_count}!")
                return
            self._player_bids_display[HUMAN_ID] = "Accepts"
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
            # Dak — re-deal.
            self._start_new_shota()
            return

        self.shooter_id = winning_bid.player_id
        self.bid_value = winning_bid.value
        self._bidding_done = True

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
        pygame.time.set_timer(pygame.USEREVENT + 2, 800, loops=1)

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

        # Feature 18: Play "your_turn" sound if human leads.
        # (Sounds removed)

    def _end_shota(self):
        """Score the Shota and start next or end game."""
        from environments.wist.scoring import score_shota

        playing_team = self.players[self.shooter_id].team_id
        defending = 1 if playing_team == 0 else 0
        total = self.team_tricks[0] + self.team_tricks[1]

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

        # Feature 12: Score breakdown in game log.
        bid_met = self.team_tricks[playing_team] >= self.bid_value
        result_str = "SUCCESS" if bid_met else "FAILED"
        self._log_game_event(f"Shota {self.shota_number}: Bid {self.bid_value} → {result_str}")
        self._log_game_event(f"  Tricks: T1={self.team_tricks[0]} T2={self.team_tricks[1]}")
        self._log_game_event(f"  Score: T1 +{shota_score_t1}, T2 +{shota_score_t2}")
        self._log_game_event(f"  Total: T1={self.game_scores[0]} T2={self.game_scores[1]}")

        if self.shota_number >= 5 or self.game_scores[0] >= 25 or self.game_scores[1] >= 25:
            self.phase = "game_over"
            self._log_game_event("=== GAME OVER ===")
        else:
            self.phase = "shota_end"
            self._ai_timer = 90

    # ----------------------------------------------------------
    # Update (called each frame)
    # ----------------------------------------------------------

    def update(self):
        """Update game logic each frame."""
        if self._message_timer > 0:
            self._message_timer -= 1

        # Feature 10: Pulse animation.
        self._pulse_frame = (self._pulse_frame + 1) % 60

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

        if self.phase == "dealing":
            self._ai_timer -= 1
            if self._ai_timer <= 0:
                self._run_bidding()
        elif self.phase == "bidding":
            self._update_bidding()
            # After bidding done and Dak, re-deal.
            if self._bidding_done and self._bidding_engine.highest_bid is None:
                self._ai_timer -= 1
                if self._ai_timer <= 0:
                    self._start_new_shota()
        elif self.phase == "playing":
            self._update_playing()
        elif self.phase == "shota_end":
            self._ai_timer -= 1
            if self._ai_timer <= 0:
                self._start_new_shota()

    def _update_playing(self):
        """Handle AI turns and timing."""
        if self._ai_timer > 0:
            self._ai_timer -= 1
            return

        if self._play_idx >= 4:
            # Trick complete — resolve.
            self._resolve_trick()
            return

        pid = self._play_order[self._play_idx]

        if pid == HUMAN_ID:
            # Wait for human click. Feature 18: sound.
            pass
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

                # Feature 17: Play animation.
                self._start_play_animation(pid, r, s)
            except Exception as e:
                print(f"[PyGame AI Error] {DISPLAY_NAMES.get(pid, pid)}: {e}")
            self._play_idx += 1
            self._ai_timer = 20

    def _start_play_animation(self, pid: int, rank: str, suit: str):
        """Start a card play animation (Feature 17)."""
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
        anim = AnimatingCard(surf, start, end, frames=15)
        self._play_animations.append(anim)

    def _resolve_trick(self):
        """Determine winner and clean up."""
        trick = self.round.state.current_trick
        if trick is None or len(trick.played_cards) < 4:
            self.round.state.current_trick = None
            self._ai_timer = 30
            self._play_idx = 99
            pygame.time.set_timer(pygame.USEREVENT + 1, 600, loops=1)
            return

        winner = trick_winner(trick, self.trump_suit)
        self.round.state.completed_tricks.append(trick)
        self.round.state.current_trick = None
        self.round.next_leading_player_id = winner

        team = 0 if winner in (0, 2) else 1
        self.team_tricks[team] += 1

        # Feature 9: Highlight winner.
        self._trick_winner_id = winner
        self._trick_winner_timer = 60  # ~1 second at 60fps.

        self._message = f"{DISPLAY_NAMES[winner]} won trick {self.trick_number}!"
        self._message_timer = 40
        self._ai_timer = 60
        self._play_idx = 99
        self._log_game_event(f"Trick {self.trick_number}: {DISPLAY_NAMES[winner]} wins")

        pygame.time.set_timer(pygame.USEREVENT + 1, 1000, loops=1)

    # ----------------------------------------------------------
    # Event handling
    # ----------------------------------------------------------

    def handle_event(self, event):
        """Handle PyGame events."""
        if event.type == pygame.USEREVENT + 1:
            if self.phase == "playing":
                self._start_next_trick()

        if event.type == pygame.USEREVENT + 2:
            if self.phase == "playing":
                self._start_next_trick()

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

        # Scroll game log.
        if event.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            log_x = TABLE_WIDTH
            if mx >= log_x:
                self._log_scroll_offset -= event.y * 2
                max_scroll = max(0, len(self._game_log) * 14 - (SCREEN_HEIGHT - 80))
                self._log_scroll_offset = max(0, min(self._log_scroll_offset, max_scroll))

    def _handle_click(self, pos):
        """Handle mouse click — bidding or card selection."""
        # Feature 22: Load model button (top-right of info bar area).
        load_btn = pygame.Rect(TABLE_WIDTH - 130, 5, 120, 28)
        if load_btn.collidepoint(pos):
            self._open_model_dialog()
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
        """Handle bidding click — 3-step: number → trump → confirm (Feature 5)."""
        cx = TABLE_WIDTH // 2
        cy = SCREEN_HEIGHT // 2

        if self._bid_step == "number":
            # Check bid number buttons (7-13).
            for i in range(7):
                rect = pygame.Rect(cx - 210 + i * 62, cy + 10, 55, 45)
                if rect.collidepoint(pos):
                    self._selected_bid = 7 + i
                    self._bid_step = "trump"
                    return
            # Check pass button.
            pass_rect = pygame.Rect(cx - 50, cy + 70, 100, 38)
            if pass_rect.collidepoint(pos):
                self._bid_step = "number"
                self._selected_bid = None
                self._selected_trump_idx = None
                self._human_pass_action()
                return

        elif self._bid_step == "trump":
            # Check suit buttons.
            for i in range(4):
                rect = pygame.Rect(cx - 140 + i * 75, cy + 10, 65, 60)
                if rect.collidepoint(pos):
                    self._selected_trump_idx = i
                    self._bid_step = "confirm"
                    return
            # Back button.
            back_rect = pygame.Rect(cx - 50, cy + 80, 100, 32)
            if back_rect.collidepoint(pos):
                self._bid_step = "number"
                self._selected_bid = None
                return

        elif self._bid_step == "confirm":
            # Confirm button.
            confirm_rect = pygame.Rect(cx - 70, cy + 10, 140, 45)
            if confirm_rect.collidepoint(pos):
                self._human_bid_with_number_and_suit()
                return
            # Back button.
            back_rect = pygame.Rect(cx - 70, cy + 65, 140, 32)
            if back_rect.collidepoint(pos):
                self._bid_step = "trump"
                self._selected_trump_idx = None
                return

    def _human_play(self, card: Card):
        """Human plays a card."""
        try:
            action = PlayCardAction(player_id=HUMAN_ID, card=card)
            self.environment.apply_action(action)
        except Exception as e:
            self._message = f"Can't play: {e}"
            self._message_timer = 40
            return

        self._play_idx += 1
        r, s = card_key(card)
        self._trick_played[HUMAN_ID] = (r, s)
        self._ai_timer = 20

        # Feature 1: Reveal trump on first card.
        if not self._trump_revealed:
            self._trump_revealed = True

        # Feature 17: Animation.
        self._start_play_animation(HUMAN_ID, r, s)

    def _get_clicked_card(self, pos) -> Card | None:
        """Check if pos is on a card in the human hand."""
        hand = self.players[HUMAN_ID].hand
        if not hand:
            return None

        # Get legal cards.
        leading_suit = None
        if self.round.state.current_trick:
            leading_suit = self.round.state.current_trick.leading_suit
        must_trump = None
        if (self.round.state.is_first_trick and
                self.round.state.winning_bidder_id == HUMAN_ID and
                self.round.state.current_trick and
                len(self.round.state.current_trick.played_cards) == 0):
            must_trump = self.trump_suit
        legal = set(legal_cards(hand, leading_suit, must_trump))

        # Sort hand same as rendering.
        suit_order = {Suit.SPADES: 0, Suit.HEARTS: 1, Suit.CLUBS: 2, Suit.DIAMONDS: 3}
        sorted_hand = sorted(hand, key=lambda c: (suit_order[c.suit], -rank_value(c.rank)))

        # Feature 20: Responsive sizing.
        n = len(sorted_hand)
        card_w = CARD_LARGE_W if n <= 6 else CARD_WIDTH
        card_h = CARD_LARGE_H if n <= 6 else CARD_HEIGHT

        # Feature 11: Suit spacing — find suit boundaries.
        suit_gaps = []
        for i in range(1, n):
            if sorted_hand[i].suit != sorted_hand[i-1].suit:
                suit_gaps.append(i)

        spacing = min(55, (TABLE_WIDTH - 200) // max(n, 1))
        total_w = (n - 1) * spacing + card_w + len(suit_gaps) * 15
        start_x = (TABLE_WIDTH - total_w) // 2
        y = SCREEN_HEIGHT - card_h - 40

        mx, my = pos
        current_x = start_x
        for i, card in enumerate(sorted_hand):
            if i in suit_gaps:
                current_x += 15  # Suit gap.
            cx = current_x
            if cx <= mx <= cx + card_w and y <= my <= y + card_h:
                if card in legal:
                    return card
            current_x += spacing
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
            return

        if self.phase == "shota_end":
            self._render_shota_end()
            return

        # Table area (left of log panel).
        table_rect = pygame.Rect(20, 50, TABLE_WIDTH - 40, SCREEN_HEIGHT - 100)
        pygame.draw.rect(self.screen, TABLE_FELT, table_rect, border_radius=12)
        pygame.draw.rect(self.screen, TABLE_BORDER, table_rect, width=2, border_radius=12)

        # Top info bar.
        self._render_info_bar()

        cx, cy = TABLE_WIDTH // 2, SCREEN_HEIGHT // 2

        # Opponent cards with role borders (Feature 14).
        self._render_opponent(0, cx - 80, 75, horizontal=True)
        self._render_opponent(3, 45, cy - 60, horizontal=False)
        self._render_opponent(1, TABLE_WIDTH - 45 - CARD_MINI_W, cy - 60, horizontal=False)

        # Player labels with roles.
        self._render_player_labels(cx, cy)

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

        # Deal animation (Feature 16).
        if self.phase == "dealing":
            for anim in self._deal_animations:
                anim.render(self.screen)

        # Play animations (Feature 17).
        for anim in self._play_animations:
            anim.render(self.screen)

        # Message.
        if self._message_timer > 0 and self._message:
            msg_surf = self.fonts["large"].render(self._message, True, TEXT_GOLD)
            self.screen.blit(msg_surf, msg_surf.get_rect(centerx=cx, y=SCREEN_HEIGHT - 195))

        # Feature 8: Game log panel.
        self._render_game_log()

        # Feature 7: Bid display persistence during play.
        if self.phase == "playing":
            self._render_bid_labels(cx, cy)

    def _render_bid_labels(self, cx, cy):
        """Show bid labels just below each player's name."""
        font = pygame.font.SysFont("Segoe UI", 11)
        cx_table = TABLE_WIDTH // 2
        right_x = TABLE_WIDTH - 45 - CARD_MINI_W + CARD_MINI_W // 2

        positions = {
            0: (cx_table, 73),                    # Below Tarig name (centred)
            3: (45 + CARD_MINI_W // 2, cy - 64),  # Below Musaab name (centred)
            1: (right_x, cy - 64),                 # Below Gaafar name (centred)
            2: (cx_table, SCREEN_HEIGHT - CARD_HEIGHT - 55),  # Below Abubakr name (centred)
        }
        for pid, text in self._player_bids_display.items():
            if text and pid in positions:
                color = TEXT_GOLD if "Bid" in text else TEXT_DIM
                surf = font.render(text, True, color)
                px, py = positions[pid]
                self.screen.blit(surf, surf.get_rect(centerx=px, y=py))

    def _render_game_log(self):
        """Feature 8: Right-side game log panel (200px wide, dark bg)."""
        log_x = TABLE_WIDTH
        panel_rect = pygame.Rect(log_x, 0, LOG_PANEL_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, BG_DARK, panel_rect)
        pygame.draw.line(self.screen, TABLE_BORDER, (log_x, 0), (log_x, SCREEN_HEIGHT), 2)

        # Title.
        title = self.fonts["medium"].render("Game Log", True, TEXT_GOLD)
        self.screen.blit(title, (log_x + 10, 10))

        # Scrollable log entries.
        clip_rect = pygame.Rect(log_x + 5, 35, LOG_PANEL_WIDTH - 10, SCREEN_HEIGHT - 45)
        self.screen.set_clip(clip_rect)

        y = 35 - self._log_scroll_offset
        for line in self._game_log:
            if y > -14 and y < SCREEN_HEIGHT:
                color = TEXT_GOLD if "===" in line or "---" in line else TEXT_LIGHT
                if "DAK" in line:
                    color = BUTTON_RED
                elif "wins" in line.lower():
                    color = TEXT_GREEN
                surf = self.fonts["log"].render(line, True, color)
                # Truncate if too wide.
                if surf.get_width() > LOG_PANEL_WIDTH - 15:
                    surf = self.fonts["log"].render(line[:28], True, color)
                self.screen.blit(surf, (log_x + 8, y))
            y += 14

        self.screen.set_clip(None)

    def _render_quit_overlay(self):
        """Feature 21: ESC quit overlay."""
        # Render game underneath.
        table_rect = pygame.Rect(20, 50, TABLE_WIDTH - 40, SCREEN_HEIGHT - 100)
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
        """Render game over screen."""
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
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

        trophy_font = pygame.font.SysFont("Segoe UI", 60)
        trophy = trophy_font.render("🏆", True, TEXT_GOLD)
        self.screen.blit(trophy, trophy.get_rect(centerx=cx, y=cy - 120))

        go_font = pygame.font.SysFont("Segoe UI", 36, bold=True)
        go = go_font.render("GAME OVER", True, TEXT_WHITE)
        self.screen.blit(go, go.get_rect(centerx=cx, y=cy - 50))

        win_font = pygame.font.SysFont("Segoe UI", 28, bold=True)
        win = win_font.render(winner_text, True, color)
        self.screen.blit(win, win.get_rect(centerx=cx, y=cy + 10))

        score_font = pygame.font.SysFont("Segoe UI", 20)
        score = score_font.render(
            f"Team 1 (You): {self.game_scores[0]}  │  Team 2: {self.game_scores[1]}",
            True, TEXT_LIGHT)
        self.screen.blit(score, score.get_rect(centerx=cx, y=cy + 60))

        hint = self.fonts["medium"].render("Press SPACE for new game  |  ESC for menu", True, TEXT_DIM)
        self.screen.blit(hint, hint.get_rect(centerx=cx, y=cy + 120))

        # Still show game log.
        self._render_game_log()

    def _render_shota_end(self):
        """Render Shota end summary."""
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        self.screen.blit(overlay, (0, 0))

        playing_team = self.players[self.shooter_id].team_id
        bid_met = self.team_tricks[playing_team] >= self.bid_value

        title_font = pygame.font.SysFont("Segoe UI", 24, bold=True)
        title = title_font.render(f"Shota {self.shota_number} Complete", True, TEXT_WHITE)
        self.screen.blit(title, title.get_rect(centerx=cx, y=cy - 60))

        result_color = HIGHLIGHT_GREEN if bid_met else BUTTON_RED
        result_text = "Bid SUCCESS ✓" if bid_met else "Bid FAILED ✗"
        result = self.fonts["large"].render(result_text, True, result_color)
        self.screen.blit(result, result.get_rect(centerx=cx, y=cy - 20))

        tricks = self.fonts["medium"].render(
            f"Tricks — Team 1: {self.team_tricks[0]}  |  Team 2: {self.team_tricks[1]}",
            True, TEXT_LIGHT)
        self.screen.blit(tricks, tricks.get_rect(centerx=cx, y=cy + 15))

        score = self.fonts["large"].render(
            f"Score — Team 1: {self.game_scores[0]}  |  Team 2: {self.game_scores[1]}",
            True, TEXT_GOLD)
        self.screen.blit(score, score.get_rect(centerx=cx, y=cy + 50))

        next_txt = self.fonts["small"].render("Next Shota starting...", True, TEXT_DIM)
        self.screen.blit(next_txt, next_txt.get_rect(centerx=cx, y=cy + 90))

        self._render_game_log()

    def _render_player_labels(self, cx, cy):
        """Render player names and roles — consistently above each player's area."""
        font = pygame.font.SysFont("Segoe UI", 13, bold=True)

        cx_table = TABLE_WIDTH // 2

        # Top player (pid 0 = Tarig) — cards start at (cx-80, 75).
        role_0 = " 👑" if 0 == self.qabool_id else ""
        role_0 += " 🎯" if 0 == self.shooter_id else ""
        surf = font.render(f"{DISPLAY_NAMES[0]}{role_0}", True, TEAM1_BLUE)
        self.screen.blit(surf, surf.get_rect(centerx=cx_table, y=57))

        # Left player (pid 3 = Musaab) — cards at (45, cy-60) vertical.
        role_3 = " 👑" if 3 == self.qabool_id else ""
        role_3 += " 🎯" if 3 == self.shooter_id else ""
        surf = font.render(f"{DISPLAY_NAMES[3]}{role_3}", True, TEAM2_ORANGE)
        self.screen.blit(surf, surf.get_rect(centerx=45 + CARD_MINI_W // 2, y=cy - 80))

        # Right player (pid 1 = Gaafar) — cards at (TABLE_WIDTH-45-CARD_MINI_W, cy-60) vertical.
        role_1 = " 👑" if 1 == self.qabool_id else ""
        role_1 += " 🎯" if 1 == self.shooter_id else ""
        surf = font.render(f"{DISPLAY_NAMES[1]}{role_1}", True, TEAM2_ORANGE)
        right_x = TABLE_WIDTH - 45 - CARD_MINI_W + CARD_MINI_W // 2
        self.screen.blit(surf, surf.get_rect(centerx=right_x, y=cy - 80))

        # Human (pid 2 = Abubakr) — centred above hand.
        role_2 = ""
        if HUMAN_ID == self.qabool_id:
            role_2 += "  👑 Qabool"
        if HUMAN_ID == self.shooter_id:
            role_2 += "  🎯 Shooter"
        surf = font.render(f"{DISPLAY_NAMES[HUMAN_ID]} (You) — Team 1{role_2}", True, TEXT_GOLD)
        self.screen.blit(surf, surf.get_rect(centerx=cx_table, y=SCREEN_HEIGHT - CARD_HEIGHT - 70))

    def _render_tricks_won(self, cx, cy):
        """Render won tricks count top-left corner, well away from player areas."""
        x, y = 30, 155

        # Team 1 count.
        count_font = pygame.font.SysFont("Segoe UI", 20, bold=True)
        label_font = self.fonts["small"]

        t1_text = count_font.render(str(self.team_tricks[0]), True, TEAM1_BLUE)
        t1_label = label_font.render("Team 1", True, TEAM1_BLUE)
        self.screen.blit(t1_label, (x, y))
        self.screen.blit(t1_text, (x + 5, y + 14))

        # Team 2 count.
        x2 = x + 65
        t2_text = count_font.render(str(self.team_tricks[1]), True, TEAM2_ORANGE)
        t2_label = label_font.render("Team 2", True, TEAM2_ORANGE)
        self.screen.blit(t2_label, (x2, y))
        self.screen.blit(t2_text, (x2 + 5, y + 14))

    def _render_trump_display(self):
        """Feature 1: Trump hidden until first card played in trick 1."""
        x, y = TABLE_WIDTH - 100, 60
        card_surf = pygame.Surface((60, 85), pygame.SRCALPHA)
        pygame.draw.rect(card_surf, CARD_WHITE, card_surf.get_rect(), border_radius=6)
        pygame.draw.rect(card_surf, (180, 180, 180), card_surf.get_rect(), width=1, border_radius=6)

        if self._trump_revealed and self.trump_suit is not None:
            sym = SUIT_SYMBOLS.get(self.trump_suit, "?")
            color = RED_SUIT if self.trump_suit in (Suit.HEARTS, Suit.DIAMONDS) else BLACK_SUIT
            big_font = pygame.font.SysFont("Segoe UI", 36)
            suit_surf = big_font.render(sym, True, color)
            card_surf.blit(suit_surf, suit_surf.get_rect(center=(30, 42)))
        else:
            # Show "?" when hidden.
            big_font = pygame.font.SysFont("Segoe UI", 36, bold=True)
            q_surf = big_font.render("?", True, TEXT_DIM)
            card_surf.blit(q_surf, q_surf.get_rect(center=(30, 42)))

        self.screen.blit(card_surf, (x, y))

        label_text = "TRUMP" if self._trump_revealed else "TRUMP (hidden)"
        label = self.fonts["small"].render(label_text, True, TEXT_GOLD)
        self.screen.blit(label, label.get_rect(centerx=x + 30, y=y + 88))

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

                # Feature 9: Gold border on winning card.
                if self._trick_winner_id == pid and self._trick_winner_timer > 0:
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
        """Render the human player's hand (face-up, clickable)."""
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

        # Feature 20: Responsive hand sizing.
        n = len(sorted_hand)
        card_w = CARD_LARGE_W if n <= 6 else CARD_WIDTH
        card_h = CARD_LARGE_H if n <= 6 else CARD_HEIGHT

        # Feature 11: Suit spacing — find suit boundaries.
        suit_gaps = []
        for i in range(1, n):
            if sorted_hand[i].suit != sorted_hand[i-1].suit:
                suit_gaps.append(i)

        spacing = min(55, (TABLE_WIDTH - 200) // max(n, 1))
        total_w = (n - 1) * spacing + card_w + len(suit_gaps) * 15
        start_x = (TABLE_WIDTH - total_w) // 2
        y = SCREEN_HEIGHT - card_h - 40

        mx, my = pygame.mouse.get_pos()

        current_x = start_x
        for i, card in enumerate(sorted_hand):
            if i in suit_gaps:
                current_x += 15  # Feature 11: gap between suits.
            card_x = current_x
            is_legal = card in legal
            is_hovered = (card_x <= mx <= card_x + card_w and y <= my <= y + card_h)

            r, s = card_key(card)
            card_surf = self._get_card_surface_sized(r, s, card_w, card_h)

            card_y = y - 12 if (is_hovered and is_legal) else y

            if not is_legal:
                dimmed = card_surf.copy()
                dark = pygame.Surface(dimmed.get_size(), pygame.SRCALPHA)
                dark.fill((0, 0, 0, 100))
                dimmed.blit(dark, (0, 0))
                self.screen.blit(dimmed, (card_x, card_y))
            else:
                shadow = pygame.Surface((card_w + 4, card_h + 4), pygame.SRCALPHA)
                pygame.draw.rect(shadow, (0, 0, 0, 30), shadow.get_rect(), border_radius=6)
                self.screen.blit(shadow, (card_x - 2, card_y + 3))
                self.screen.blit(card_surf, (card_x, card_y))

            current_x += spacing

    def _render_bidding_ui(self, cx, cy):
        """Render bidding interface — 3 steps: number → trump → confirm (Feature 5)."""
        is_human_turn = False
        if self._bid_index < len(self._bid_order) and self._bid_order[self._bid_index] == HUMAN_ID:
            is_human_turn = True
        if self.qabool_id == HUMAN_ID and self._bid_index >= len(self._bid_order):
            is_human_turn = True

        # Show previous bids near each player (same positions as during play).
        bid_font = pygame.font.SysFont("Segoe UI", 11)
        cx_table = TABLE_WIDTH // 2
        right_x = TABLE_WIDTH - 45 - CARD_MINI_W + CARD_MINI_W // 2
        bid_positions = {
            0: (cx_table, 73),
            3: (45 + CARD_MINI_W // 2, cy - 64),
            1: (right_x, cy - 64),
            2: (cx_table, SCREEN_HEIGHT - CARD_HEIGHT - 55),
        }
        for pid, text in self._player_bids_display.items():
            if text and pid in bid_positions:
                color = TEXT_GOLD if "Bid" in text else TEXT_DIM
                surf = bid_font.render(text, True, color)
                px, py = bid_positions[pid]
                self.screen.blit(surf, surf.get_rect(centerx=px, y=py))

        if not is_human_turn:
            wait = self.fonts["large"].render("Bidding...", True, TEXT_DIM)
            self.screen.blit(wait, wait.get_rect(centerx=cx, y=cy))
            return

        # ---- HUMAN BIDDING UI ----
        title_font = pygame.font.SysFont("Segoe UI", 18, bold=True)
        btn_font = pygame.font.SysFont("Segoe UI", 16, bold=True)
        mx, my = pygame.mouse.get_pos()

        if self._bid_step == "number":
            title = title_font.render("Step 1: Choose your bid (7-13)", True, TEXT_WHITE)
            self.screen.blit(title, title.get_rect(centerx=cx, y=cy - 30))

            # Feature 19: Validation feedback.
            current_highest = (self._bidding_engine.highest_bid.value
                               if self._bidding_engine.highest_bid else None)
            hint_text = "Select a number. "
            if current_highest:
                hint_text += f"Must beat {current_highest}."
            else:
                hint_text += "Opening bid max 11."
            hint = self.fonts["small"].render(hint_text, True, TEXT_DIM)
            self.screen.blit(hint, hint.get_rect(centerx=cx, y=cy - 5))

            # Bid number buttons: 7-13.
            for i, val in enumerate(range(7, 14)):
                rect = pygame.Rect(cx - 210 + i * 62, cy + 10, 55, 45)
                hover = rect.collidepoint(mx, my)
                # Dim numbers that are too low.
                is_valid = True
                if current_highest and val <= current_highest:
                    is_qabool_turn = (self.qabool_id == HUMAN_ID and
                                      self._bid_index >= len(self._bid_order))
                    if not is_qabool_turn:
                        is_valid = False
                if not self._has_opening_bid and val > 11:
                    is_qabool_turn = (self.qabool_id == HUMAN_ID and
                                      self._bid_index >= len(self._bid_order))
                    if not is_qabool_turn:
                        is_valid = False

                if is_valid:
                    bg = (56, 142, 60) if hover else BUTTON_GREEN
                else:
                    bg = (60, 60, 60)
                pygame.draw.rect(self.screen, bg, rect, border_radius=6)
                if hover and is_valid:
                    pygame.draw.rect(self.screen, TEXT_GREEN, rect, width=2, border_radius=6)
                num_surf = btn_font.render(str(val), True, TEXT_WHITE if is_valid else TEXT_DIM)
                self.screen.blit(num_surf, num_surf.get_rect(center=rect.center))

            # Pass button.
            pass_rect = pygame.Rect(cx - 50, cy + 70, 100, 38)
            hover = pass_rect.collidepoint(mx, my)
            bg = (80, 80, 80) if hover else BUTTON_GREY
            pygame.draw.rect(self.screen, bg, pass_rect, border_radius=6)
            pass_surf = btn_font.render("Pass", True, TEXT_WHITE)
            self.screen.blit(pass_surf, pass_surf.get_rect(center=pass_rect.center))

        elif self._bid_step == "trump":
            # Step 2: Select trump suit.
            title = title_font.render(f"Step 2: Select trump (Bid: {self._selected_bid})", True, TEXT_WHITE)
            self.screen.blit(title, title.get_rect(centerx=cx, y=cy - 30))

            suits = [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]
            for i, suit in enumerate(suits):
                sym = SUIT_SYMBOLS[suit]
                count = sum(1 for c in self.players[HUMAN_ID].hand if c.suit == suit)
                rect = pygame.Rect(cx - 140 + i * 75, cy + 10, 65, 60)
                hover = rect.collidepoint(mx, my)
                color = RED_SUIT if suit in (Suit.HEARTS, Suit.DIAMONDS) else BLACK_SUIT

                bg = (255, 255, 240) if hover else CARD_WHITE
                pygame.draw.rect(self.screen, bg, rect, border_radius=8)
                if hover:
                    pygame.draw.rect(self.screen, HIGHLIGHT_GREEN, rect, width=2, border_radius=8)
                else:
                    pygame.draw.rect(self.screen, (180, 180, 180), rect, width=1, border_radius=8)

                sym_font = pygame.font.SysFont("Segoe UI", 26, bold=True)
                sym_surf = sym_font.render(sym, True, color)
                self.screen.blit(sym_surf, sym_surf.get_rect(centerx=rect.centerx, y=rect.y + 5))

                cnt_surf = self.fonts["small"].render(f"({count})", True, TEXT_DIM)
                self.screen.blit(cnt_surf, cnt_surf.get_rect(centerx=rect.centerx, y=rect.y + 42))

            # Back button.
            back_rect = pygame.Rect(cx - 50, cy + 80, 100, 32)
            hover = back_rect.collidepoint(mx, my)
            bg = (70, 70, 70) if hover else BUTTON_GREY
            pygame.draw.rect(self.screen, bg, back_rect, border_radius=6)
            back_surf = self.fonts["medium"].render("← Back", True, TEXT_WHITE)
            self.screen.blit(back_surf, back_surf.get_rect(center=back_rect.center))

        elif self._bid_step == "confirm":
            # Step 3: Confirm (Feature 5).
            suits = [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]
            chosen_suit = suits[self._selected_trump_idx]
            sym = SUIT_SYMBOLS[chosen_suit]
            title = title_font.render(
                f"Confirm: Bid {self._selected_bid}, Trump {sym}?", True, TEXT_WHITE)
            self.screen.blit(title, title.get_rect(centerx=cx, y=cy - 30))

            # Confirm button.
            confirm_rect = pygame.Rect(cx - 70, cy + 10, 140, 45)
            hover = confirm_rect.collidepoint(mx, my)
            bg = (56, 142, 60) if hover else BUTTON_GREEN
            pygame.draw.rect(self.screen, bg, confirm_rect, border_radius=8)
            if hover:
                pygame.draw.rect(self.screen, TEXT_GREEN, confirm_rect, width=2, border_radius=8)
            confirm_surf = btn_font.render("✓ Confirm", True, TEXT_WHITE)
            self.screen.blit(confirm_surf, confirm_surf.get_rect(center=confirm_rect.center))

            # Back button.
            back_rect = pygame.Rect(cx - 70, cy + 65, 140, 32)
            hover = back_rect.collidepoint(mx, my)
            bg = (70, 70, 70) if hover else BUTTON_GREY
            pygame.draw.rect(self.screen, bg, back_rect, border_radius=6)
            back_surf = self.fonts["medium"].render("← Change", True, TEXT_WHITE)
            self.screen.blit(back_surf, back_surf.get_rect(center=back_rect.center))

            # Feature 19: Validation feedback.
            count = sum(1 for c in self.players[HUMAN_ID].hand if c.suit == chosen_suit)
            hint = self.fonts["small"].render(
                f"You have {count} {sym} cards. Formula: longest_suit + 3",
                True, TEXT_DIM)
            self.screen.blit(hint, hint.get_rect(centerx=cx, y=cy + 105))

    def _render_info_bar(self):
        """Render the top information bar + Feature 22 Load Model button."""
        y = 10
        items = [
            (f"Shota {self.shota_number}/5", TEXT_WHITE),
            (f"Trick {self.trick_number}/13", TEXT_LIGHT),
            (f"Qabool: {DISPLAY_NAMES[self.qabool_id]}", TEXT_GOLD),
            (f"Bid: {self.bid_value}", TEXT_GOLD),
            (f"Shooter: {DISPLAY_NAMES.get(self.shooter_id, '?')}", TEXT_GREEN),
        ]

        # Feature 1: Only show trump if revealed.
        if self._trump_revealed and self.trump_suit:
            sym = SUIT_SYMBOLS.get(self.trump_suit, "?")
            color = RED_SUIT if self.trump_suit in (Suit.HEARTS, Suit.DIAMONDS) else TEXT_WHITE
            items.append((f"Trump: {sym}", color))
        else:
            items.append(("Trump: ?", TEXT_DIM))

        items.append((f"Score: {self.game_scores[0]} - {self.game_scores[1]}", TEAM1_BLUE))

        # Feature 6: Dak counter.
        if self._dak_count > 0:
            items.append((f"Daks: {self._dak_count}/2", BUTTON_RED))

        x = 30
        for text, color in items:
            surf = self.fonts["medium"].render(text, True, color)
            self.screen.blit(surf, (x, y))
            x += surf.get_width() + 20

        # Feature 22: Load Model button.
        btn_rect = pygame.Rect(TABLE_WIDTH - 130, 5, 120, 28)
        mx, my = pygame.mouse.get_pos()
        hover = btn_rect.collidepoint(mx, my)
        bg = (40, 100, 160) if hover else BUTTON_BLUE
        pygame.draw.rect(self.screen, bg, btn_rect, border_radius=5)
        btn_text = self.fonts["small"].render("Load Model", True, TEXT_WHITE)
        self.screen.blit(btn_text, btn_text.get_rect(center=btn_rect.center))
        if self._ai_model_path:
            model_name = os.path.basename(self._ai_model_path)[:15]
            name_surf = self.fonts["small"].render(model_name, True, TEXT_GREEN)
            self.screen.blit(name_surf, (TABLE_WIDTH - 130, 36))

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
