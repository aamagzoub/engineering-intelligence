"""
Sudanese Hearts — Visual Game Watcher (PyGame)

Watch the Discovery AI agent play Hearts against random opponents.
See every trick played, cards passed, scoring, and learning progress.

Usage:
    python gui_hearts/main.py
    python gui_hearts/main.py --model agents/hearts_discovery/hearts_model.json
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
from gui_hearts.constants import *
from gui_hearts.card_renderer import create_card_surface, create_card_back

from agents.hearts_discovery.discovery_agent import DiscoveryAgent
from agents.hearts_discovery.random_hearts_agent import RandomHeartsAgent
from environments.hearts.player import HeartsPlayer
from environments.hearts.environment import HeartsEnvironment
from environments.hearts.observation import PassingObservation
from environments.hearts.actions import PassCardsAction, PlayCardAction
from environments.hearts.scoring import score_shota, count_penalties, QUEEN_OF_SPADES
from intelligence.core.cards.deck import Deck
from intelligence.core.cards.suit import Suit
from intelligence.core.cards.rank import Rank


# Rank values for sorting.
RANK_ORDER = {
    Rank.TWO: 2, Rank.THREE: 3, Rank.FOUR: 4, Rank.FIVE: 5,
    Rank.SIX: 6, Rank.SEVEN: 7, Rank.EIGHT: 8, Rank.NINE: 9,
    Rank.TEN: 10, Rank.JACK: 11, Rank.QUEEN: 12, Rank.KING: 13, Rank.ACE: 14,
}
SUIT_ORDER = {Suit.SPADES: 0, Suit.HEARTS: 1, Suit.CLUBS: 2, Suit.DIAMONDS: 3}


class HeartsWatcher:
    """PyGame app to watch Hearts games unfold visually."""

    def __init__(self, model_path: str | None = None):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()
        self.running = True

        # Fonts.
        self.fonts = {
            "title": pygame.font.SysFont("Segoe UI", 24, bold=True),
            "large": pygame.font.SysFont("Segoe UI", 16, bold=True),
            "medium": pygame.font.SysFont("Segoe UI", 13),
            "small": pygame.font.SysFont("Segoe UI", 11),
            "card": pygame.font.SysFont("Consolas", 12, bold=True),
        }

        # Agents.
        self.model_path = model_path

        # Game state.
        self.state = "idle"  # idle, passing, playing, scoring, game_over
        self.shota_num = 0
        self.dealer_id = 0
        self.total_scores = {i: 0 for i in range(4)}
        self.games_played = 0
        self.games_won_by_ai = 0

        # Visual state.
        self.players: list[HeartsPlayer] = []
        self.hands: dict[int, list] = {}
        self.current_trick_cards: list[tuple[int, object]] = []
        self.trick_num = 0
        self.last_winner = -1
        self.shota_scores: list[dict] = []
        self.event_log: list[str] = []
        self.speed = 1.0  # Playback speed multiplier.
        self.auto_play = True
        self.last_action_time = 0
        self.paused = False

        # Card surface cache.
        self._card_cache: dict[str, pygame.Surface] = {}
        self._card_back = create_card_back()

        # Setup agents (after event_log is initialized).
        self._setup_agents()

        # Start first game.
        self._start_new_game()

    def _setup_agents(self):
        """Create agents — Discovery + 3 Random."""
        if self.model_path:
            self.discovery = DiscoveryAgent(training=False)
            try:
                self.discovery.load(self.model_path)
                self._log(f"Loaded model: {self.discovery.episodes_trained} episodes trained")
            except FileNotFoundError:
                self._log("Model not found, using untrained agent")
                self.discovery = DiscoveryAgent(training=False)
        else:
            self.discovery = DiscoveryAgent(training=False)
            self._log("No model loaded — agent plays untrained")

        self.agents = [
            self.discovery,
            RandomHeartsAgent(),
            RandomHeartsAgent(),
            RandomHeartsAgent(),
        ]

    def _log(self, msg: str):
        """Add to event log."""
        self.event_log.append(msg)
        if len(self.event_log) > 50:
            self.event_log = self.event_log[-50:]

    def _get_card_surface(self, card) -> pygame.Surface:
        """Get or create card surface (cached)."""
        key = f"{card.rank.symbol}{card.suit.symbol}"
        if key not in self._card_cache:
            self._card_cache[key] = create_card_surface(
                card.rank.symbol, card.suit.symbol
            )
        return self._card_cache[key]

    def _start_new_game(self):
        """Start a new 5-shota game."""
        self.shota_num = 0
        self.dealer_id = 0
        self.total_scores = {i: 0 for i in range(4)}
        self.shota_scores = []
        self._log(f"{'='*40}")
        self._log(f"NEW GAME #{self.games_played + 1}")
        self._start_new_shota()

    def _start_new_shota(self):
        """Start a new shota within the current game."""
        self.shota_num += 1
        if self.shota_num > 5:
            self._end_game()
            return

        self._log(f"--- Shota {self.shota_num}/5 (Dealer: {PLAYER_NAMES[self.dealer_id]}) ---")

        # Reset players.
        self.players = [HeartsPlayer(player_id=i) for i in range(4)]

        # Deal.
        deck = Deck()
        deck.shuffle()
        for p in self.players:
            p.receive_cards(deck.deal(13))

        # Store hands for display.
        self.hands = {p.player_id: sorted(
            p.hand, key=lambda c: (SUIT_ORDER[c.suit], RANK_ORDER[c.rank])
        ) for p in self.players}

        self.state = "passing"
        self.current_trick_cards = []
        self.trick_num = 0
        self.last_action_time = pygame.time.get_ticks()

    def _do_passing(self):
        """Execute the passing phase."""
        cards_to_pass = {}
        for p in self.players:
            obs = PassingObservation(player_id=p.player_id, hand=list(p.hand))
            action = self.agents[p.player_id].act(obs)
            cards_to_pass[p.player_id] = action.cards
            passed = ", ".join(f"{c.rank.symbol}{c.suit.symbol}" for c in action.cards)
            receiver = PLAYER_NAMES[(p.player_id + 1) % 4]
            self._log(f"  {PLAYER_NAMES[p.player_id]} → {receiver}: {passed}")

        # Execute pass.
        for p in self.players:
            p.remove_cards(list(cards_to_pass[p.player_id]))
        for p in self.players:
            receiver_id = (p.player_id + 1) % 4
            self.players[receiver_id].receive_cards(list(cards_to_pass[p.player_id]))

        # Update hands for display.
        self.hands = {p.player_id: sorted(
            p.hand, key=lambda c: (SUIT_ORDER[c.suit], RANK_ORDER[c.rank])
        ) for p in self.players}

        # Move to playing.
        self.state = "playing"
        self.trick_num = 1
        first_leader = (self.dealer_id + 1) % 4
        self.env = HeartsEnvironment(self.players, first_leader)
        self.current_trick_cards = []
        self._log(f"  Cards passed. Playing begins.")
        self.last_action_time = pygame.time.get_ticks()

    def _do_one_trick(self):
        """Play one complete trick."""
        if self.env.is_shota_complete():
            self._do_scoring()
            return

        self.current_trick_cards = []
        for i in range(4):
            current_pid = self.env.current_player_id()
            obs = self.env.observe(current_pid)
            action = self.agents[current_pid].act(obs)
            self.current_trick_cards.append((current_pid, action.card))
            winner_id = self.env.apply_action(action)

        # Update hands.
        self.hands = {p.player_id: sorted(
            p.hand, key=lambda c: (SUIT_ORDER[c.suit], RANK_ORDER[c.rank])
        ) for p in self.players}

        # Check for penalty cards.
        trick_cards_only = [c for _, c in self.current_trick_cards]
        hearts_in_trick = sum(1 for c in trick_cards_only if c.suit == Suit.HEARTS)
        has_queen = QUEEN_OF_SPADES in trick_cards_only
        penalty_str = ""
        if hearts_in_trick > 0 or has_queen:
            pts = hearts_in_trick + (7 if has_queen else 0)
            penalty_str = f" [💔-{pts}]"

        plays_str = " | ".join(
            f"{PLAYER_NAMES[pid]}:{c.rank.symbol}{c.suit.symbol}"
            for pid, c in self.current_trick_cards
        )
        self._log(f"  T{self.trick_num}: {plays_str} → {PLAYER_NAMES[winner_id]}{penalty_str}")

        self.last_winner = winner_id
        self.trick_num += 1
        self.last_action_time = pygame.time.get_ticks()

    def _do_scoring(self):
        """Score the completed shota."""
        collected = {p.player_id: list(p.collected_cards) for p in self.players}
        tricks_won = {p.player_id: p.tricks_won for p in self.players}
        scores = score_shota(collected, tricks_won)

        self.shota_scores.append(scores)
        for pid, score in scores.items():
            self.total_scores[pid] += score

        # Log results.
        self._log(f"  Shota {self.shota_num} scores:")
        for pid in range(4):
            h_count = sum(1 for c in collected[pid] if c.suit == Suit.HEARTS)
            has_q = QUEEN_OF_SPADES in collected[pid]
            q_str = " +Q♠" if has_q else ""
            self._log(f"    {PLAYER_NAMES[pid]}: {scores[pid]:+d} ({tricks_won[pid]}T, {h_count}♥{q_str})")

        # Special scenario detection.
        zero_trick = [pid for pid in range(4) if tricks_won[pid] == 0]
        all_trick = [pid for pid in range(4) if tricks_won[pid] == 13]
        if all_trick:
            self._log(f"  🏆 ALL TRICKS: {PLAYER_NAMES[all_trick[0]]} (+18)")
        elif len(zero_trick) == 1:
            self._log(f"  🏆 FULL GALLON: {PLAYER_NAMES[zero_trick[0]]} (+20)")
        elif len(zero_trick) == 2:
            names = " & ".join(PLAYER_NAMES[pid] for pid in zero_trick)
            self._log(f"  🏆 HALF GALLON: {names} (+10)")

        self.state = "scoring"
        self.dealer_id = (self.dealer_id + 1) % 4
        self.last_action_time = pygame.time.get_ticks()

    def _end_game(self):
        """End the current game."""
        self.games_played += 1
        winner = max(self.total_scores, key=self.total_scores.get)
        loser = min(self.total_scores, key=self.total_scores.get)
        if winner == 0:
            self.games_won_by_ai += 1

        self._log(f"  GAME OVER! Winner: {PLAYER_NAMES[winner]} ({self.total_scores[winner]:+d})")
        self._log(f"  Loser: {PLAYER_NAMES[loser]} ({self.total_scores[loser]:+d})")
        self._log(f"  AI Win Rate: {self.games_won_by_ai}/{self.games_played}")
        self.state = "game_over"
        self.last_action_time = pygame.time.get_ticks()

    def run(self):
        """Main loop."""
        while self.running:
            self._handle_events()
            self._update()
            self._render()
            self.clock.tick(FPS)
        pygame.quit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_UP:
                    self.speed = min(10.0, self.speed * 1.5)
                    self._log(f"  Speed: {self.speed:.1f}x")
                elif event.key == pygame.K_DOWN:
                    self.speed = max(0.2, self.speed / 1.5)
                    self._log(f"  Speed: {self.speed:.1f}x")
                elif event.key == pygame.K_n:
                    # Skip to next game.
                    self._start_new_game()
                elif event.key == pygame.K_r:
                    # Reload model.
                    self._setup_agents()

    def _update(self):
        """Auto-advance the game state based on timing."""
        if self.paused:
            return

        now = pygame.time.get_ticks()
        delay = int(TRICK_DELAY_MS / self.speed)

        if now - self.last_action_time < delay:
            return

        if self.state == "passing":
            self._do_passing()
        elif self.state == "playing":
            if not self.env.is_shota_complete():
                self._do_one_trick()
            else:
                self._do_scoring()
        elif self.state == "scoring":
            # After scoring delay, start next shota.
            if now - self.last_action_time > int(SHOTA_DELAY_MS / self.speed):
                self._start_new_shota()
        elif self.state == "game_over":
            if now - self.last_action_time > int(SHOTA_DELAY_MS * 2 / self.speed):
                self._start_new_game()

    def _render(self):
        """Render the full interface."""
        self.screen.fill(BG_DARK)

        # Table area (center).
        table_rect = pygame.Rect(20, 60, SCREEN_WIDTH - 320, SCREEN_HEIGHT - 80)
        pygame.draw.rect(self.screen, TABLE_FELT, table_rect, border_radius=15)
        pygame.draw.rect(self.screen, TABLE_BORDER, table_rect, width=2, border_radius=15)

        # Title.
        title = self.fonts["title"].render(TITLE, True, TEXT_WHITE)
        self.screen.blit(title, (25, 15))

        # Game info bar.
        info = f"Shota {self.shota_num}/5  |  Trick {min(self.trick_num, 13)}/13  |  Speed: {self.speed:.1f}x  |  Games: {self.games_played} (AI won: {self.games_won_by_ai})"
        info_surf = self.fonts["medium"].render(info, True, TEXT_DIM)
        self.screen.blit(info_surf, (25, 45))

        # Controls hint.
        controls = "SPACE: pause  |  ↑↓: speed  |  N: new game  |  ESC: quit"
        ctrl_surf = self.fonts["small"].render(controls, True, TEXT_DIM)
        self.screen.blit(ctrl_surf, (SCREEN_WIDTH - 420, 45))

        if self.paused:
            pause_surf = self.fonts["large"].render("PAUSED", True, TEXT_GOLD)
            self.screen.blit(pause_surf, (SCREEN_WIDTH // 2 - 40, 15))

        # Render player hands and info.
        self._render_players(table_rect)

        # Render current trick in center.
        self._render_trick(table_rect)

        # Right panel — scores and log.
        self._render_side_panel()

        pygame.display.flip()

    def _render_players(self, table_rect):
        """Render each player's hand and info around the table."""
        # Player positions: 0=bottom, 1=left, 2=top, 3=right
        positions = {
            0: (table_rect.centerx, table_rect.bottom - 20, "bottom"),
            1: (table_rect.left + 30, table_rect.centery, "left"),
            2: (table_rect.centerx, table_rect.top + 20, "top"),
            3: (table_rect.right - 30, table_rect.centery, "right"),
        }

        for pid in range(4):
            x, y, pos = positions[pid]
            hand = self.hands.get(pid, [])
            color = PLAYER_COLORS[pid]

            # Player name and score.
            name = PLAYER_NAMES[pid]
            score_text = f"{self.total_scores[pid]:+d}"
            name_surf = self.fonts["large"].render(name, True, color)
            score_surf = self.fonts["medium"].render(score_text, True, TEXT_GOLD)

            # Penalty tracker — hearts collected + queen this shota.
            penalty_str = self._get_penalty_display(pid)

            if pos == "bottom":
                self.screen.blit(name_surf, (x - name_surf.get_width() // 2, y + 5))
                self.screen.blit(score_surf, (x - score_surf.get_width() // 2, y + 25))
                if penalty_str:
                    pen_surf = self.fonts["small"].render(penalty_str, True, TEXT_RED)
                    self.screen.blit(pen_surf, (x - pen_surf.get_width() // 2, y + 42))
                self._render_hand_horizontal(hand, x, y - CARD_HEIGHT - 10, face_up=True)
            elif pos == "top":
                self.screen.blit(name_surf, (x - name_surf.get_width() // 2, y - 15))
                self.screen.blit(score_surf, (x - score_surf.get_width() // 2, y + 5))
                if penalty_str:
                    pen_surf = self.fonts["small"].render(penalty_str, True, TEXT_RED)
                    self.screen.blit(pen_surf, (x - pen_surf.get_width() // 2, y + 22))
                self._render_hand_horizontal(hand, x, y + 40, face_up=True)
            elif pos == "left":
                self.screen.blit(name_surf, (x, y - 80))
                self.screen.blit(score_surf, (x, y - 60))
                if penalty_str:
                    pen_surf = self.fonts["small"].render(penalty_str, True, TEXT_RED)
                    self.screen.blit(pen_surf, (x, y - 42))
                self._render_hand_vertical(hand, x, y - 25, face_up=True)
            elif pos == "right":
                self.screen.blit(name_surf, (x - name_surf.get_width(), y - 80))
                self.screen.blit(score_surf, (x - score_surf.get_width(), y - 60))
                if penalty_str:
                    pen_surf = self.fonts["small"].render(penalty_str, True, TEXT_RED)
                    self.screen.blit(pen_surf, (x - pen_surf.get_width(), y - 42))
                self._render_hand_vertical(hand, x - CARD_MINI_W, y - 25, face_up=True)

    def _get_penalty_display(self, player_id: int) -> str:
        """Build a string showing hearts collected and queen status for a player."""
        if not self.players:
            return ""

        player = None
        for p in self.players:
            if p.player_id == player_id:
                player = p
                break

        if player is None or not player.collected_cards:
            return ""

        hearts_count = sum(1 for c in player.collected_cards if c.suit == Suit.HEARTS)
        has_queen = QUEEN_OF_SPADES in player.collected_cards
        tricks = player.tricks_won

        parts = []
        if hearts_count > 0:
            parts.append(f"{hearts_count}♥")
        if has_queen:
            parts.append("Q♠")
        if tricks > 0:
            parts.append(f"({tricks}T)")

        return " ".join(parts) if parts else ""

    def _render_hand_horizontal(self, hand, cx, y, face_up=True):
        """Render cards in a horizontal fan."""
        if not hand:
            return
        n = len(hand)
        overlap = min(CARD_WIDTH - 10, (SCREEN_WIDTH - 400) // max(n, 1))
        total_w = overlap * (n - 1) + CARD_WIDTH
        start_x = cx - total_w // 2

        for i, card in enumerate(hand):
            x = start_x + i * overlap
            if face_up:
                surf = self._get_card_surface(card)
            else:
                surf = self._card_back
            self.screen.blit(surf, (x, y))

    def _render_hand_vertical(self, hand, x, cy, face_up=True):
        """Render cards in a compact vertical stack."""
        if not hand:
            return
        n = len(hand)
        overlap = min(18, 200 // max(n, 1))
        total_h = overlap * (n - 1) + CARD_MINI_H
        start_y = cy

        for i, card in enumerate(hand):
            y = start_y + i * overlap
            if face_up:
                surf = self._get_card_surface(card)
                # Scale down for side players.
                surf = pygame.transform.smoothscale(surf, (CARD_MINI_W, CARD_MINI_H))
            else:
                surf = pygame.transform.smoothscale(self._card_back, (CARD_MINI_W, CARD_MINI_H))
            self.screen.blit(surf, (x, y))

    def _render_trick(self, table_rect):
        """Render the current trick's cards in the center of the table."""
        if not self.current_trick_cards:
            return

        cx, cy = table_rect.centerx, table_rect.centery
        # Positions for each player's card in the trick.
        offsets = {
            0: (0, 50),    # bottom player plays below center
            1: (-80, 0),   # left player plays left of center
            2: (0, -50),   # top player plays above center
            3: (80, 0),    # right player plays right of center
        }

        for pid, card in self.current_trick_cards:
            ox, oy = offsets[pid]
            x = cx + ox - CARD_WIDTH // 2
            y = cy + oy - CARD_HEIGHT // 2
            surf = self._get_card_surface(card)
            self.screen.blit(surf, (x, y))

            # Show player name below their card.
            name_surf = self.fonts["small"].render(PLAYER_NAMES[pid], True, PLAYER_COLORS[pid])
            self.screen.blit(name_surf, (x + CARD_WIDTH // 2 - name_surf.get_width() // 2, y + CARD_HEIGHT + 2))

        # Show trick winner.
        if self.last_winner >= 0 and len(self.current_trick_cards) == 4:
            winner_text = f"Winner: {PLAYER_NAMES[self.last_winner]}"
            winner_surf = self.fonts["large"].render(winner_text, True, TEXT_GOLD)
            self.screen.blit(winner_surf, (cx - winner_surf.get_width() // 2, cy + 90))

        # Render collected penalty cards on the table near each player.
        self._render_collected_cards(table_rect)

    def _render_collected_cards(self, table_rect):
        """
        Render the penalty cards (hearts + Q♠) each player has collected
        as small face-up card piles on the table near their position.
        """
        if not self.players:
            return

        # Positions for collected piles — in corners of the table near each player.
        pile_positions = {
            0: (table_rect.centerx + 150, table_rect.bottom - 90),   # bottom-right
            1: (table_rect.left + 80, table_rect.centery + 100),     # left-bottom
            2: (table_rect.centerx - 180, table_rect.top + 75),      # top-left
            3: (table_rect.right - 130, table_rect.centery - 120),   # right-top
        }

        mini_w, mini_h = 30, 42  # Tiny cards for the pile.

        for p in self.players:
            # Filter only penalty cards (hearts + Queen of Spades).
            penalty_cards = [
                c for c in p.collected_cards
                if c.suit == Suit.HEARTS or c == QUEEN_OF_SPADES
            ]
            if not penalty_cards:
                continue

            px, py = pile_positions[p.player_id]
            color = PLAYER_COLORS[p.player_id]

            # Sort: Queen of Spades first, then hearts by rank.
            penalty_cards.sort(key=lambda c: (
                0 if c == QUEEN_OF_SPADES else 1,
                -RANK_ORDER.get(c.rank, 0)
            ))

            # Draw overlapping mini cards.
            overlap_x = 14
            for i, card in enumerate(penalty_cards):
                cx = px + i * overlap_x
                cy = py

                # Draw mini card.
                card_rect = pygame.Rect(cx, cy, mini_w, mini_h)
                pygame.draw.rect(self.screen, CARD_WHITE, card_rect, border_radius=3)

                # Card suit color.
                if card.suit == Suit.HEARTS:
                    card_color = RED_SUIT
                else:
                    card_color = BLACK_SUIT  # Queen of Spades

                # Rank + suit text.
                card_font = pygame.font.SysFont("Consolas", 10, bold=True)
                text = f"{card.rank.symbol}{card.suit.symbol}"
                text_surf = card_font.render(text, True, card_color)
                self.screen.blit(text_surf, (cx + 2, cy + 2))

                # Border.
                border_color = (200, 50, 50) if card == QUEEN_OF_SPADES else (180, 180, 180)
                pygame.draw.rect(self.screen, border_color, card_rect, width=1, border_radius=3)

            # Label: penalty total.
            hearts_count = sum(1 for c in penalty_cards if c.suit == Suit.HEARTS)
            has_queen = QUEEN_OF_SPADES in penalty_cards
            total_penalty = hearts_count + (7 if has_queen else 0)
            label = f"-{total_penalty}"
            label_surf = self.fonts["small"].render(label, True, TEXT_RED)
            label_x = px + len(penalty_cards) * overlap_x + mini_w + 4
            self.screen.blit(label_surf, (label_x, py + mini_h // 2 - 6))

    def _render_side_panel(self):
        """Render the right-side panel with scores and event log."""
        panel_x = SCREEN_WIDTH - 290
        panel_rect = pygame.Rect(panel_x, 60, 280, SCREEN_HEIGHT - 80)
        pygame.draw.rect(self.screen, PANEL_DARK, panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, (50, 45, 70), panel_rect, width=1, border_radius=10)

        y = panel_rect.top + 10

        # Scoreboard.
        title = self.fonts["large"].render("Scoreboard", True, TEXT_WHITE)
        self.screen.blit(title, (panel_x + 10, y))
        y += 25

        sorted_pids = sorted(range(4), key=lambda p: self.total_scores[p], reverse=True)
        for rank, pid in enumerate(sorted_pids):
            color = PLAYER_COLORS[pid]
            marker = " 👑" if rank == 0 else ""
            name = f"{PLAYER_NAMES[pid]}"
            score = f"{self.total_scores[pid]:+d}{marker}"
            name_surf = self.fonts["medium"].render(name, True, color)
            score_surf = self.fonts["medium"].render(score, True, TEXT_GOLD)
            self.screen.blit(name_surf, (panel_x + 15, y))
            self.screen.blit(score_surf, (panel_x + 180, y))
            y += 20

        y += 10
        pygame.draw.line(self.screen, TEXT_DIM, (panel_x + 10, y), (panel_x + 270, y))
        y += 10

        # Shota history.
        history_title = self.fonts["large"].render("Shota History", True, TEXT_WHITE)
        self.screen.blit(history_title, (panel_x + 10, y))
        y += 22

        for i, scores in enumerate(self.shota_scores[-5:], 1):
            line_parts = []
            for pid in range(4):
                s = scores.get(pid, 0)
                line_parts.append(f"{s:+d}")
            line = f"  S{i}: " + " / ".join(line_parts)
            line_surf = self.fonts["small"].render(line, True, TEXT_LIGHT)
            self.screen.blit(line_surf, (panel_x + 10, y))
            y += 16

        y += 10
        pygame.draw.line(self.screen, TEXT_DIM, (panel_x + 10, y), (panel_x + 270, y))
        y += 10

        # Event log.
        log_title = self.fonts["large"].render("Event Log", True, TEXT_WHITE)
        self.screen.blit(log_title, (panel_x + 10, y))
        y += 22

        # Show last N events that fit.
        max_lines = (panel_rect.bottom - y - 10) // 15
        visible_log = self.event_log[-max_lines:]
        for line in visible_log:
            # Truncate long lines.
            display_line = line[:42]
            log_surf = self.fonts["small"].render(display_line, True, TEXT_DIM)
            self.screen.blit(log_surf, (panel_x + 10, y))
            y += 15


def main():
    parser = argparse.ArgumentParser(description="Sudanese Hearts — Visual Watcher")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model")
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        # Try default location.
        default = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "discovery", "hearts_model.json"
        )
        if os.path.exists(default):
            model_path = default

    app = HeartsWatcher(model_path=model_path)
    app.run()


if __name__ == "__main__":
    main()
