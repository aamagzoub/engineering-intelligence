"""
Game screen — the main playing table for the PyGame version.

Handles rendering the table, players, cards, and user interaction.
Integrates with the game engine for actual gameplay.
"""

import pygame
from collections import Counter

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


def card_key(card: Card) -> tuple[str, str]:
    return RANK_SYMBOLS[card.rank], SUIT_SYMBOLS[card.suit]


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
        }

        # Card cache.
        self._card_cache: dict[str, pygame.Surface] = {}
        self._card_back = create_card_back(CARD_WIDTH, CARD_HEIGHT)
        self._card_back_mini = create_card_back(CARD_MINI_W, CARD_MINI_H)

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

        # Animation state.
        self._trick_played: dict[int, tuple[str, str]] = {}  # pid → (rank, suit)
        self._hover_card_idx = -1
        self._message = ""
        self._message_timer = 0

        # Timing.
        self._ai_timer = 0
        self._play_order = []
        self._play_idx = 0

    def start_game(self):
        """Start a new full game."""
        self.game_scores = [0, 0]
        self.shota_number = 0
        self._start_new_shota()

    def _start_new_shota(self):
        """Deal and start bidding."""
        self.shota_number += 1
        self.trick_number = 0
        self.team_tricks = [0, 0]
        self._trick_played = {}

        self.players = create_standard_players()
        self.round = Round(self.players)
        self.round.deal()

        while self.round.has_card_based_dak():
            self.round = Round(self.players)
            self.round.deal()

        self.agents = [RuleBasedAgent(), RuleBasedAgent(), None, RuleBasedAgent()]

        if self.shota_number == 1:
            self.qabool_id = determine_first_shota_qabool()
        else:
            self.qabool_id = (self.qabool_id + 1) % 4

        self.phase = "bidding"
        self._run_bidding()

    def _run_bidding(self):
        """Run bidding step by step — AI bids instantly, human gets UI."""
        from environments.wist.tasmiya_engine import tasmiya_order
        from environments.wist.bidding_engine import BiddingEngine
        from environments.wist.bidding import Bid, Pass
        from environments.wist.observation import BiddingObservation

        self._bidding_engine = BiddingEngine()
        self._bid_history = []
        self._bid_order = tasmiya_order(self.qabool_id)
        self._bid_index = 0
        self._has_opening_bid = False
        self._human_trump_choice = None
        self._bidding_done = False

        # Player bids display (persists).
        self._player_bids_display = {0: "", 1: "", 2: "", 3: ""}

        self.phase = "bidding"
        self._ai_timer = 30  # Small delay before first bid.

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

            from environments.wist.actions import BidAction, PassAction
            if isinstance(action, BidAction):
                bid = Bid(player_id=pid, value=action.value)
                self._bidding_engine.apply_bid(bid)
                self._bid_history.append((pid, action.value))
                self._has_opening_bid = True
                self._player_bids_display[pid] = f"Bid {action.value}"
                self._message = f"P{pid+1} bids {action.value}"
            else:
                self._bidding_engine.apply_pass(Pass(player_id=pid))
                self._bid_history.append((pid, None))
                self._player_bids_display[pid] = "Pass"
                self._message = f"P{pid+1} passes"

            self._message_timer = 25
            self._bid_index += 1
            self._ai_timer = 25

    def _bid_qabool(self):
        """Sahib Al-Qabool decides."""
        from environments.wist.observation import BiddingObservation
        from environments.wist.actions import BidAction, PassAction
        from environments.wist.bidding import Bid, Pass

        qid = self.qabool_id

        if qid == HUMAN_ID:
            # Human is Qabool — wait for their click.
            self._message = "YOU are Qabool! Select trump + confirm."
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

        if isinstance(action, BidAction):
            bid = Bid(player_id=qid, value=action.value)
            self._bidding_engine.apply_bid(bid, is_sahib_al_qabool=True)
            self._bid_history.append((qid, action.value))
            self._player_bids_display[qid] = f"Bid {action.value}"
        else:
            self._bidding_engine.apply_pass(Pass(player_id=qid))
            self._bid_history.append((qid, None))
            if self._bidding_engine.highest_bid is None:
                self._message = "All passed — Dak!"
                self._message_timer = 60
                self._bidding_done = True
                self._ai_timer = 60
                return
            self._player_bids_display[qid] = "Accepts"

        self._finalize_bidding()

    def _human_bid_action(self, suit_idx: int):
        """Human selects a trump suit and bids."""
        suits = [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]
        chosen_suit = suits[suit_idx]
        count = sum(1 for c in self.players[HUMAN_ID].hand if c.suit == chosen_suit)

        if count < 4:
            self._message = "Need 4+ cards in that suit!"
            self._message_timer = 40
            return

        bid_value = count + 3
        current_highest = (self._bidding_engine.highest_bid.value
                           if self._bidding_engine.highest_bid else None)

        # Validate.
        if not self._has_opening_bid and bid_value > 11:
            self._message = "Opening bid max 11!"
            self._message_timer = 40
            return
        if current_highest and bid_value <= current_highest and self.qabool_id != HUMAN_ID:
            self._message = f"Must beat {current_highest}!"
            self._message_timer = 40
            return

        from environments.wist.bidding import Bid
        is_qabool = (self.qabool_id == HUMAN_ID and self._bid_index >= len(self._bid_order))
        bid = Bid(player_id=HUMAN_ID, value=bid_value)
        self._bidding_engine.apply_bid(bid, is_sahib_al_qabool=is_qabool)
        self._bid_history.append((HUMAN_ID, bid_value))
        self._has_opening_bid = True
        self._human_trump_choice = chosen_suit
        self._player_bids_display[HUMAN_ID] = f"Bid {bid_value}"
        self._message = f"You bid {bid_value} ({SUIT_SYMBOLS[chosen_suit]})"
        self._message_timer = 30

        if is_qabool:
            self._finalize_bidding()
        else:
            self._bid_index += 1
            self._ai_timer = 25

    def _human_pass_action(self):
        """Human passes."""
        from environments.wist.bidding import Pass
        is_qabool = (self.qabool_id == HUMAN_ID and self._bid_index >= len(self._bid_order))

        self._bidding_engine.apply_pass(Pass(player_id=HUMAN_ID))
        self._bid_history.append((HUMAN_ID, None))
        self._player_bids_display[HUMAN_ID] = "Pass"

        if is_qabool:
            if self._bidding_engine.highest_bid is None:
                self._message = "Dak!"
                self._message_timer = 60
                self._bidding_done = True
                self._ai_timer = 60
                return
            self._player_bids_display[HUMAN_ID] = "Accepts"
            self._finalize_bidding()
        else:
            self._bid_index += 1
            self._ai_timer = 25

    def _finalize_bidding(self):
        """Bidding resolved — set up for play."""
        winning_bid = self._bidding_engine.highest_bid
        if winning_bid is None:
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

        self._message = f"P{self.shooter_id+1} shoots! Trump: {SUIT_SYMBOLS.get(self.trump_suit, '?')}"
        self._message_timer = 50
        self._ai_timer = 50

        self.phase = "playing"
        pygame.time.set_timer(pygame.USEREVENT + 2, 800, loops=1)

    # Bidding buttons (suit rects for click detection).
    def _get_bid_suit_rects(self) -> list[pygame.Rect]:
        """Get clickable rectangles for suit selection during bidding."""
        cx = SCREEN_WIDTH // 2
        y = SCREEN_HEIGHT // 2 + 40
        rects = []
        for i in range(4):
            rects.append(pygame.Rect(cx - 120 + i * 65, y, 55, 55))
        return rects

    def _get_pass_rect(self) -> pygame.Rect:
        """Get clickable pass button rect."""
        cx = SCREEN_WIDTH // 2
        return pygame.Rect(cx - 50, SCREEN_HEIGHT // 2 + 110, 100, 35)
        self.environment = WistEnvironment(self.round.state)

        self.phase = "playing"
        self._start_next_trick()

    def _start_next_trick(self):
        """Start a new trick."""
        if self.trick_number >= 13:
            self._end_shota()
            return

        self.trick_number += 1
        self._trick_played = {}

        leader = self.round.next_leading_player_id
        self.round.state.current_trick = Trick(leading_player_id=leader)
        self._play_order = [(leader + i) % 4 for i in range(4)]
        self._play_idx = 0
        self._ai_timer = 30  # Wait a bit before first play.

    def _end_shota(self):
        """Score the Shota and start next or end game."""
        from environments.wist.scoring import score_shota

        playing_team = self.players[self.shooter_id].team_id
        defending = 1 if playing_team == 0 else 0
        score_delta = score_shota(
            playing_team_id=playing_team, defending_team_id=defending,
            bid=self.bid_value,
            playing_team_tricks=self.team_tricks[playing_team],
            defending_team_tricks=self.team_tricks[defending])
        self.game_scores[0] += score_delta.get(0, 0)
        self.game_scores[1] += score_delta.get(1, 0)

        if self.shota_number >= 5 or self.game_scores[0] >= 25 or self.game_scores[1] >= 25:
            self.phase = "game_over"
        else:
            self.phase = "shota_end"
            self._ai_timer = 90  # Pause before next Shota.

    # ----------------------------------------------------------
    # Update (called each frame)
    # ----------------------------------------------------------

    def update(self):
        """Update game logic each frame."""
        if self._message_timer > 0:
            self._message_timer -= 1

        if self.phase == "bidding":
            self._update_bidding()
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
            # Wait for human click (handled in handle_click).
            pass
        else:
            # AI plays.
            obs = self.environment.observe(pid)
            action = self.agents[pid].act(obs)
            self.environment.apply_action(action)
            self._play_idx += 1

            r, s = card_key(action.card)
            self._trick_played[pid] = (r, s)
            self._ai_timer = 20  # Pause between AI cards.

    def _resolve_trick(self):
        """Determine winner and clean up."""
        trick = self.round.state.current_trick
        winner = trick_winner(trick, self.trump_suit)
        self.round.state.completed_tricks.append(trick)
        self.round.state.current_trick = None
        self.round.next_leading_player_id = winner

        team = 0 if winner in (0, 2) else 1
        self.team_tricks[team] += 1

        self._message = f"P{winner+1} won trick {self.trick_number}!"
        self._message_timer = 40
        self._ai_timer = 50
        self._play_idx = 99  # Prevent re-entry.

        # After delay, start next trick.
        pygame.time.set_timer(pygame.USEREVENT + 1, 800, loops=1)

    def handle_event(self, event):
        """Handle PyGame events."""
        if event.type == pygame.USEREVENT + 1:
            if self.phase == "playing":
                self._start_next_trick()

        if event.type == pygame.USEREVENT + 2:
            # Transition from bidding to first trick.
            if self.phase == "playing":
                self._start_next_trick()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and self.phase == "game_over":
                self.start_game()

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._handle_click(event.pos)

    def _handle_click(self, pos):
        """Handle mouse click — card selection or bidding."""
        # Bidding phase — suit selection.
        if self.phase == "bidding":
            # Check if it's human's turn.
            is_human_turn = False
            if self._bid_index < len(self._bid_order) and self._bid_order[self._bid_index] == HUMAN_ID:
                is_human_turn = True
            if self.qabool_id == HUMAN_ID and self._bid_index >= len(self._bid_order):
                is_human_turn = True

            if is_human_turn:
                # Check suit buttons.
                rects = self._get_bid_suit_rects()
                for i, rect in enumerate(rects):
                    if rect.collidepoint(pos):
                        self._human_bid_action(i)
                        return
                # Check pass button.
                if self._get_pass_rect().collidepoint(pos):
                    self._human_pass_action()
                    return

        # Playing phase — card selection.
        if self.phase == "playing" and self._play_idx < 4:
            pid = self._play_order[self._play_idx]
            if pid == HUMAN_ID:
                card = self._get_clicked_card(pos)
                if card:
                    self._human_play(card)

    def _human_play(self, card: Card):
        """Human plays a card."""
        action = PlayCardAction(player_id=HUMAN_ID, card=card)
        self.environment.apply_action(action)
        self._play_idx += 1

        r, s = card_key(card)
        self._trick_played[HUMAN_ID] = (r, s)
        self._ai_timer = 20

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

        # Card positions (must match render logic).
        n = len(sorted_hand)
        spacing = min(55, (SCREEN_WIDTH - 200) // max(n, 1))
        total_w = (n - 1) * spacing + CARD_WIDTH
        start_x = (SCREEN_WIDTH - total_w) // 2
        y = SCREEN_HEIGHT - CARD_HEIGHT - 40

        mx, my = pos
        for i, card in enumerate(sorted_hand):
            cx = start_x + i * spacing
            if cx <= mx <= cx + CARD_WIDTH and y <= my <= y + CARD_HEIGHT:
                if card in legal:
                    return card
        return None

    # ----------------------------------------------------------
    # Render
    # ----------------------------------------------------------

    def render(self):
        """Render the full game screen."""
        if self.phase == "game_over":
            self._render_game_over()
            return

        if self.phase == "shota_end":
            self._render_shota_end()
            return

        # Table.
        table_rect = pygame.Rect(20, 50, SCREEN_WIDTH - 40, SCREEN_HEIGHT - 100)
        pygame.draw.rect(self.screen, TABLE_FELT, table_rect, border_radius=12)
        pygame.draw.rect(self.screen, TABLE_BORDER, table_rect, width=2, border_radius=12)

        # Top info bar.
        self._render_info_bar()

        # Opponent cards (face-down) with labels.
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        self._render_opponent(0, cx - 80, 75, horizontal=True)   # Top
        self._render_opponent(3, 45, cy - 60, horizontal=False)  # Left
        self._render_opponent(1, SCREEN_WIDTH - 45 - CARD_MINI_W, cy - 60, horizontal=False)  # Right

        # Player labels with roles.
        self._render_player_labels(cx, cy)

        # Won tricks piles.
        self._render_tricks_won(cx, cy)

        # Trump display (top-right).
        self._render_trump_display()

        # Centre trick.
        self._render_centre_trick(cx, cy)

        # Human hand.
        self._render_human_hand()

        # "Your turn" indicator.
        if self.phase == "playing" and self._play_idx < len(self._play_order):
            if self._play_order[self._play_idx] == HUMAN_ID:
                turn_surf = self.fonts["large"].render("▶ Your Turn — Click a card!", True, TEXT_GREEN)
                self.screen.blit(turn_surf, turn_surf.get_rect(centerx=cx, y=SCREEN_HEIGHT - 175))

        # Bidding UI (suit buttons + pass).
        if self.phase == "bidding":
            self._render_bidding_ui(cx, cy)

        # Message.
        if self._message_timer > 0 and self._message:
            msg_surf = self.fonts["large"].render(self._message, True, TEXT_GOLD)
            self.screen.blit(msg_surf, msg_surf.get_rect(centerx=cx, y=SCREEN_HEIGHT - 195))

    def _render_game_over(self):
        """Render game over screen."""
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

        # Dark overlay.
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        # Winner text.
        if self.game_scores[0] > self.game_scores[1]:
            winner_text = "YOUR TEAM WINS!"
            color = HIGHLIGHT_GREEN
        elif self.game_scores[1] > self.game_scores[0]:
            winner_text = "TEAM 2 WINS!"
            color = TEAM2_ORANGE
        else:
            winner_text = "IT'S A DRAW!"
            color = TEXT_WHITE

        # Trophy.
        trophy_font = pygame.font.SysFont("Segoe UI", 60)
        trophy = trophy_font.render("🏆", True, TEXT_GOLD)
        self.screen.blit(trophy, trophy.get_rect(centerx=cx, y=cy - 120))

        # Game Over.
        go_font = pygame.font.SysFont("Segoe UI", 36, bold=True)
        go = go_font.render("GAME OVER", True, TEXT_WHITE)
        self.screen.blit(go, go.get_rect(centerx=cx, y=cy - 50))

        # Winner.
        win_font = pygame.font.SysFont("Segoe UI", 28, bold=True)
        win = win_font.render(winner_text, True, color)
        self.screen.blit(win, win.get_rect(centerx=cx, y=cy + 10))

        # Scores.
        score_font = pygame.font.SysFont("Segoe UI", 20)
        score = score_font.render(
            f"Team 1 (You): {self.game_scores[0]}  │  Team 2: {self.game_scores[1]}",
            True, TEXT_LIGHT)
        self.screen.blit(score, score.get_rect(centerx=cx, y=cy + 60))

        # Restart hint.
        hint = self.fonts["medium"].render("Press SPACE for new game  |  ESC for menu", True, TEXT_DIM)
        self.screen.blit(hint, hint.get_rect(centerx=cx, y=cy + 120))

    def _render_shota_end(self):
        """Render Shota end summary (brief pause between Shotas)."""
        cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

        # Semi-transparent overlay.
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        self.screen.blit(overlay, (0, 0))

        # Shota result.
        playing_team = self.players[self.shooter_id].team_id
        bid_met = self.team_tricks[playing_team] >= self.bid_value

        title_font = pygame.font.SysFont("Segoe UI", 24, bold=True)
        title = title_font.render(f"Shota {self.shota_number} Complete", True, TEXT_WHITE)
        self.screen.blit(title, title.get_rect(centerx=cx, y=cy - 60))

        result_color = HIGHLIGHT_GREEN if bid_met else (229, 57, 53)
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

    def _render_player_labels(self, cx, cy):
        """Render player names, teams, and role indicators."""
        # Player positions and info.
        players_info = [
            (0, cx, 65, "P3 (Partner)", "Team 1", True),    # Top
            (3, 40, cy + 55, "P4", "Team 2", False),        # Left
            (1, SCREEN_WIDTH - 90, cy + 55, "P2", "Team 2", False),  # Right
        ]

        for pid, x, y, name, team, is_top in players_info:
            # Role indicator.
            role = ""
            role_color = TEXT_DIM
            if pid == self.qabool_id:
                role = "👑 Qabool"
                role_color = TEXT_GOLD
            if pid == self.shooter_id:
                role = "🎯 Shooter" if not role else role + " | 🎯"
                role_color = TEXT_GREEN if pid != self.qabool_id else TEXT_GOLD

            # Team color.
            team_color = TEAM1_BLUE if "1" in team else TEAM2_ORANGE

            # Name.
            name_surf = self.fonts["small"].render(name, True, TEXT_WHITE)
            team_surf = self.fonts["small"].render(f"({team})", True, team_color)

            if is_top:
                self.screen.blit(name_surf, (x, y))
                self.screen.blit(team_surf, (x + name_surf.get_width() + 4, y))
            else:
                self.screen.blit(name_surf, (x, y))
                self.screen.blit(team_surf, (x, y + 14))

            # Role below.
            if role:
                role_surf = self.fonts["small"].render(role, True, role_color)
                self.screen.blit(role_surf, (x, y + (14 if is_top else 28)))

    def _render_tricks_won(self, cx, cy):
        """Render won tricks as small face-down piles near each team."""
        # Team 1 (left side of centre).
        t1_count = self.team_tricks[0]
        if t1_count > 0:
            x, y = cx - 180, cy - 20
            for i in range(t1_count):
                self.screen.blit(self._card_back_mini, (x + i * 4, y + i * 2))
            count_surf = self.fonts["small"].render(str(t1_count), True, TEAM1_BLUE)
            self.screen.blit(count_surf, (x + t1_count * 4 + CARD_MINI_W + 4, y + 20))

        # Team 2 (right side of centre).
        t2_count = self.team_tricks[1]
        if t2_count > 0:
            x, y = cx + 130, cy - 20
            for i in range(t2_count):
                self.screen.blit(self._card_back_mini, (x + i * 4, y + i * 2))
            count_surf = self.fonts["small"].render(str(t2_count), True, TEAM2_ORANGE)
            self.screen.blit(count_surf, (x + t2_count * 4 + CARD_MINI_W + 4, y + 20))

    def _render_trump_display(self):
        """Render trump suit card in top-right corner."""
        if self.trump_suit is None:
            return

        sym = SUIT_SYMBOLS.get(self.trump_suit, "?")
        color = RED_SUIT if self.trump_suit in (Suit.HEARTS, Suit.DIAMONDS) else BLACK_SUIT

        # Draw a card with the trump symbol.
        x, y = SCREEN_WIDTH - 100, 60
        card_surf = pygame.Surface((60, 85), pygame.SRCALPHA)
        pygame.draw.rect(card_surf, CARD_WHITE, card_surf.get_rect(), border_radius=6)
        pygame.draw.rect(card_surf, (180, 180, 180), card_surf.get_rect(), width=1, border_radius=6)

        # Big suit symbol.
        big_font = pygame.font.SysFont("Segoe UI", 36)
        suit_surf = big_font.render(sym, True, color)
        card_surf.blit(suit_surf, suit_surf.get_rect(center=(30, 42)))

        self.screen.blit(card_surf, (x, y))

        # "TRUMP" label below.
        label = self.fonts["small"].render("TRUMP", True, TEXT_GOLD)
        self.screen.blit(label, label.get_rect(centerx=x + 30, y=y + 88))

    def _render_bidding_ui(self, cx, cy):
        """Render bidding interface — suit buttons + pass for human."""
        # Check if it's human's turn.
        is_human_turn = False
        if self._bid_index < len(self._bid_order) and self._bid_order[self._bid_index] == HUMAN_ID:
            is_human_turn = True
        if self.qabool_id == HUMAN_ID and self._bid_index >= len(self._bid_order):
            is_human_turn = True

        # Show all bids so far.
        bid_y = cy - 80
        for pid, text in self._player_bids_display.items():
            if text:
                color = TEXT_GOLD if "Bid" in text else TEXT_DIM
                surf = self.fonts["medium"].render(f"P{pid+1}: {text}", True, color)
                self.screen.blit(surf, (cx - 100, bid_y))
                bid_y += 20

        if not is_human_turn:
            # Waiting for AI.
            wait = self.fonts["medium"].render("AI bidding...", True, TEXT_DIM)
            self.screen.blit(wait, wait.get_rect(centerx=cx, y=cy + 30))
            return

        # Title.
        title = self.fonts["large"].render("Select Trump Suit (click):", True, TEXT_WHITE)
        self.screen.blit(title, title.get_rect(centerx=cx, y=cy + 15))

        # Suit buttons.
        suits = [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]
        rects = self._get_bid_suit_rects()
        mx, my = pygame.mouse.get_pos()

        for i, (suit, rect) in enumerate(zip(suits, rects)):
            sym = SUIT_SYMBOLS[suit]
            count = sum(1 for c in self.players[HUMAN_ID].hand if c.suit == suit)
            color = RED_SUIT if suit in (Suit.HEARTS, Suit.DIAMONDS) else BLACK_SUIT

            # Button background.
            hover = rect.collidepoint(mx, my)
            bg = (255, 255, 240) if hover else CARD_WHITE
            pygame.draw.rect(self.screen, bg, rect, border_radius=8)
            pygame.draw.rect(self.screen, (150, 150, 150) if not hover else HIGHLIGHT_GREEN,
                             rect, width=2, border_radius=8)

            # Suit symbol + count.
            sym_font = pygame.font.SysFont("Segoe UI", 22, bold=True)
            sym_surf = sym_font.render(sym, True, color)
            self.screen.blit(sym_surf, sym_surf.get_rect(centerx=rect.centerx, y=rect.y + 5))

            count_surf = self.fonts["small"].render(f"({count})", True, TEXT_DIM)
            self.screen.blit(count_surf, count_surf.get_rect(centerx=rect.centerx, y=rect.y + 35))

        # Pass button.
        pass_rect = self._get_pass_rect()
        hover = pass_rect.collidepoint(mx, my)
        bg = (80, 80, 80) if hover else BUTTON_GREY
        pygame.draw.rect(self.screen, bg, pass_rect, border_radius=6)
        pass_surf = self.fonts["medium"].render("Pass", True, TEXT_WHITE)
        self.screen.blit(pass_surf, pass_surf.get_rect(center=pass_rect.center))

    def _render_info_bar(self):
        """Render the top information bar."""
        y = 10
        items = [
            (f"Shota {self.shota_number}/5", TEXT_WHITE),
            (f"Trick {self.trick_number}/13", TEXT_LIGHT),
            (f"Qabool: P{self.qabool_id+1}", TEXT_GOLD),
            (f"Bid: {self.bid_value}", TEXT_GOLD),
            (f"Shooter: P{self.shooter_id+1}", TEXT_GREEN),
        ]

        if self.trump_suit:
            sym = SUIT_SYMBOLS.get(self.trump_suit, "?")
            color = RED_SUIT if self.trump_suit in (Suit.HEARTS, Suit.DIAMONDS) else TEXT_WHITE
            items.append((f"Trump: {sym}", color))

        items.append((f"Score: {self.game_scores[0]} - {self.game_scores[1]}", TEAM1_BLUE))

        x = 30
        for text, color in items:
            surf = self.fonts["medium"].render(text, True, color)
            self.screen.blit(surf, (x, y))
            x += surf.get_width() + 30

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

        # Label.
        label = self.fonts["small"].render(f"P{pid+1}", True, TEXT_DIM)
        if horizontal:
            self.screen.blit(label, (x, y + CARD_MINI_H + 4))
        else:
            self.screen.blit(label, (x, y - 16))

    def _render_centre_trick(self, cx, cy):
        """Render played cards in the centre."""
        # Positions: P0=top, P1=right, P2=bottom, P3=left.
        offsets = {
            0: (0, -70),   # Top.
            1: (90, 0),    # Right.
            2: (0, 50),    # Bottom.
            3: (-90, 0),   # Left.
        }

        for pid, (r, s) in self._trick_played.items():
            dx, dy = offsets.get(pid, (0, 0))
            card_surf = self._get_card_surface(r, s)
            rect = card_surf.get_rect(center=(cx + dx, cy + dy))
            # Shadow.
            shadow = pygame.Surface((CARD_WIDTH + 4, CARD_HEIGHT + 4), pygame.SRCALPHA)
            pygame.draw.rect(shadow, (0, 0, 0, 40), shadow.get_rect(), border_radius=6)
            self.screen.blit(shadow, (rect.x - 2, rect.y + 2))
            self.screen.blit(card_surf, rect)

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

        n = len(sorted_hand)
        spacing = min(55, (SCREEN_WIDTH - 200) // max(n, 1))
        total_w = (n - 1) * spacing + CARD_WIDTH
        start_x = (SCREEN_WIDTH - total_w) // 2
        y = SCREEN_HEIGHT - CARD_HEIGHT - 40

        # Check hover.
        mx, my = pygame.mouse.get_pos()

        for i, card in enumerate(sorted_hand):
            cx = start_x + i * spacing
            is_legal = card in legal
            is_hovered = (cx <= mx <= cx + CARD_WIDTH and y <= my <= y + CARD_HEIGHT)

            r, s = card_key(card)
            card_surf = self._get_card_surface(r, s)

            # Highlight.
            card_y = y - 10 if (is_hovered and is_legal) else y

            if not is_legal:
                # Dim illegal cards.
                dimmed = card_surf.copy()
                dark = pygame.Surface(dimmed.get_size(), pygame.SRCALPHA)
                dark.fill((0, 0, 0, 100))
                dimmed.blit(dark, (0, 0))
                self.screen.blit(dimmed, (cx, card_y))
            else:
                # Shadow.
                shadow = pygame.Surface((CARD_WIDTH + 4, CARD_HEIGHT + 4), pygame.SRCALPHA)
                pygame.draw.rect(shadow, (0, 0, 0, 30), shadow.get_rect(), border_radius=6)
                self.screen.blit(shadow, (cx - 2, card_y + 3))

                if is_hovered:
                    # Green glow.
                    glow = pygame.Surface((CARD_WIDTH + 6, CARD_HEIGHT + 6), pygame.SRCALPHA)
                    pygame.draw.rect(glow, (76, 175, 80, 120), glow.get_rect(), border_radius=8)
                    self.screen.blit(glow, (cx - 3, card_y - 3))

                self.screen.blit(card_surf, (cx, card_y))

    def _get_card_surface(self, rank: str, suit: str) -> pygame.Surface:
        key = f"{rank}{suit}"
        if key not in self._card_cache:
            self._card_cache[key] = create_card_surface(rank, suit)
        return self._card_cache[key]
