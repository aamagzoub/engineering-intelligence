"""
Sudanese Hearts — Visual Game Watcher (PyGame)

Watch the Discovery AI agent play Hearts against random opponents.
See every trick played, cards passed, scoring, and learning progress.

Usage:
    python gui_hearts_discovery/main.py
    python gui_hearts_discovery/main.py --model agents/hearts_discovery/hearts_model.json
"""

import sys
import os
import time
import argparse
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
from gui_hearts_discovery.constants import *
from gui_hearts_discovery.card_renderer import create_card_surface, create_card_back

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
        self._screen_w = SCREEN_WIDTH
        self._screen_h = SCREEN_HEIGHT
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
        self.state = "idle"  # idle, passing, playing, scoring, game_over, trick_pause
        self.shota_num = 0
        self.dealer_id = 0
        self.total_scores = {i: 0 for i in range(4)}
        self.games_played = 0
        self.games_won_by_ai = 0

        # Mode: "watch" (fast, auto) or "follow" (slow, manual continue).
        self.mode = "follow"  # Default to follow.

        # Mode display names.
        self._mode_labels = {
            "watch": "Watching",
            "follow": "Following",
        }
        self._mode_button_labels = {
            "watch": "See & Check",     # Button to switch TO follow
            "follow": "Watch Auto",     # Button to switch TO watch
        }

        # Learning progress tracking.
        self._win_history: list[bool] = []  # True/False for each game (AI won?)
        self._initial_win_rate: float | None = None  # Captured after first 5 games

        # Visual state.
        self.players: list[HeartsPlayer] = []
        self.hands: dict[int, list] = {}
        self.current_trick_cards: list[tuple[int, object]] = []
        self.trick_num = 0
        self.last_winner = -1
        self.shota_scores: list[dict] = []
        self.event_log: list[str] = []
        self.speed = 3.0  # Playback speed multiplier.
        self.auto_play = True
        self.last_action_time = 0
        self.paused = False

        # Card surface cache.
        self._card_cache: dict[str, pygame.Surface] = {}
        self._card_back = create_card_back()
        self._continue_btn_rect = None
        self._mode_btn_rect = None
        self._reset_btn_rect = None
        self._log_scroll_offset = 0
        self._log_auto_scroll = True  # Auto-scroll to bottom unless user scrolls up.
        self._pass_cards_selected = {}
        self._pass_cards_received = {}

        # Milestone detection — behavioral discoveries.
        self._milestones_achieved: set[str] = set()
        self._milestones_list: list[str] = []  # Ordered list of discovered behaviors.
        self._milestone_queue: list[tuple[str, str]] = []

        # Load previously discovered milestones.
        self._load_milestones()

        # Setup agents (after event_log is initialized).
        self._setup_agents()

        # Start first game.
        self._start_new_game()

    def _setup_agents(self):
        """Create agents — Discovery (training) + 3 Random."""
        if self.model_path:
            self.discovery = DiscoveryAgent(training=True)
            try:
                self.discovery.load(self.model_path)
                self._log(f"Loaded model: {self.discovery.episodes_trained} shotas learned")
            except FileNotFoundError:
                self._log("Model not found, starting fresh")
                self.discovery = DiscoveryAgent(training=True)
        else:
            self.discovery = DiscoveryAgent(training=True)
            self._log("No model file — agent learns from scratch")

        # Opponents share the same Q-tables (self-play) but don't record episodes.
        opp = DiscoveryAgent(training=False)
        opp.play_q = self.discovery.play_q
        opp.pass_q = self.discovery.pass_q
        opp.epsilon = self.discovery.epsilon
        self.agents = [
            self.discovery,
            opp,
            opp,
            opp,
        ]

    def _log(self, msg: str):
        """Add to event log."""
        self.event_log.append(msg)
        if len(self.event_log) > 200:
            self.event_log = self.event_log[-200:]

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
        self._shota_trick_history = []  # List of (ai_card, leading_suit, winner_id, trick_cards)
        self._pass_cards_selected = {}  # pid → list of cards they passed away
        self._pass_cards_received = {}  # pid → list of cards they received
        self.last_action_time = pygame.time.get_ticks()

    def _do_passing(self):
        """Follow mode: select and exchange cards, then show both passed and received."""
        cards_to_pass = {}
        for p in self.players:
            obs = PassingObservation(player_id=p.player_id, hand=list(p.hand))
            action = self.agents[p.player_id].act(obs)
            cards_to_pass[p.player_id] = list(action.cards)
            passed = ", ".join(f"{c.rank.symbol}{c.suit.symbol}" for c in action.cards)
            receiver = PLAYER_NAMES[(p.player_id + 1) % 4]
            self._log(f"  {PLAYER_NAMES[p.player_id]} → {receiver}: {passed}")

        self._pass_cards_selected = cards_to_pass

        # Execute exchange immediately.
        for p in self.players:
            p.remove_cards(list(cards_to_pass[p.player_id]))
        for pid in range(4):
            receiver_id = (pid + 1) % 4
            received = list(cards_to_pass[pid])
            self.players[receiver_id].receive_cards(received)
            self._pass_cards_received[receiver_id] = received

        # Update hands for display.
        self.hands = {p.player_id: sorted(
            p.hand, key=lambda c: (SUIT_ORDER[c.suit], RANK_ORDER[c.rank])
        ) for p in self.players}

        # Show both passed and received — one screen, click Continue to play.
        self.state = "pass_show"
        self.last_action_time = pygame.time.get_ticks()

    def _do_pass_exchange(self):
        """Not used anymore — kept for compatibility."""
        pass

    def _do_pass_finalize(self):
        """Start playing after pass display."""
        self.state = "playing"
        self.trick_num = 1
        first_leader = (self.dealer_id + 1) % 4
        self.env = HeartsEnvironment(self.players, first_leader)
        self.current_trick_cards = []
        self._log(f"  Cards exchanged. Playing begins.")
        self.last_action_time = pygame.time.get_ticks()

    def _do_start_playing(self):
        """Transition from passing phase to playing phase."""
        self._do_pass_finalize()

    def _do_passing_learn(self):
        """Learn mode passing — execute immediately, no overlay."""
        cards_to_pass = {}
        for p in self.players:
            obs = PassingObservation(player_id=p.player_id, hand=list(p.hand))
            action = self.agents[p.player_id].act(obs)
            cards_to_pass[p.player_id] = list(action.cards)

        for p in self.players:
            p.remove_cards(list(cards_to_pass[p.player_id]))
        for pid in range(4):
            receiver_id = (pid + 1) % 4
            self.players[receiver_id].receive_cards(list(cards_to_pass[pid]))

        self.hands = {p.player_id: sorted(
            p.hand, key=lambda c: (SUIT_ORDER[c.suit], RANK_ORDER[c.rank])
        ) for p in self.players}

        self.state = "playing"
        self.trick_num = 1
        first_leader = (self.dealer_id + 1) % 4
        self.env = HeartsEnvironment(self.players, first_leader)
        self.current_trick_cards = []
        self.last_action_time = pygame.time.get_ticks()

    def _do_one_trick_follow(self):
        """Follow mode: play one trick then pause for Continue."""
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

        # Record trick for milestone detection.
        ai_card = next((c for pid, c in self.current_trick_cards if pid == 0), None)
        leading_suit = self.current_trick_cards[0][1].suit if self.current_trick_cards else None
        self._shota_trick_history.append({
            "ai_card": ai_card,
            "leading_suit": leading_suit,
            "winner": winner_id,
            "cards": list(self.current_trick_cards),
        })

        self.hands = {p.player_id: sorted(
            p.hand, key=lambda c: (SUIT_ORDER[c.suit], RANK_ORDER[c.rank])
        ) for p in self.players}

        trick_cards_only = [c for _, c in self.current_trick_cards]
        hearts_in_trick = sum(1 for c in trick_cards_only if c.suit == Suit.HEARTS)
        has_queen = QUEEN_OF_SPADES in trick_cards_only
        penalty_str = ""
        if hearts_in_trick > 0 or has_queen:
            pts = hearts_in_trick + (7 if has_queen else 0)
            penalty_str = f" [-{pts}]"

        plays_str = " | ".join(
            f"{PLAYER_NAMES[pid]}:{c.rank.symbol}{c.suit.symbol}"
            for pid, c in self.current_trick_cards
        )
        self._log(f"  T{self.trick_num}: {plays_str} → {PLAYER_NAMES[winner_id]}{penalty_str}")

        self.last_winner = winner_id
        self.trick_num += 1
        # Pause here — wait for Continue click.
        self.state = "trick_pause"
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

        # Record trick for milestone detection.
        ai_card = next((c for pid, c in self.current_trick_cards if pid == 0), None)
        leading_suit = self.current_trick_cards[0][1].suit if self.current_trick_cards else None
        self._shota_trick_history.append({
            "ai_card": ai_card,
            "leading_suit": leading_suit,
            "winner": winner_id,
            "cards": list(self.current_trick_cards),
        })

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
            penalty_str = f" [-{pts}]"

        plays_str = " | ".join(
            f"{PLAYER_NAMES[pid]}:{c.rank.symbol}{c.suit.symbol}"
            for pid, c in self.current_trick_cards
        )
        self._log(f"  T{self.trick_num}: {plays_str} → {PLAYER_NAMES[winner_id]}{penalty_str}")

        self.last_winner = winner_id
        self.trick_num += 1
        self.last_action_time = pygame.time.get_ticks()

    def _do_scoring(self):
        """Score the completed shota and send reward to the AI."""
        collected = {p.player_id: list(p.collected_cards) for p in self.players}
        tricks_won = {p.player_id: p.tricks_won for p in self.players}
        scores = score_shota(collected, tricks_won)

        self.shota_scores.append(scores)
        for pid, score in scores.items():
            self.total_scores[pid] += score

        # LEARNING: send reward to the Discovery AI so it actually learns.
        ai_score = scores.get(0, 0)
        self.discovery.reward(float(ai_score))

        # Detect behavioral milestones.
        self._check_milestones(collected, tricks_won, scores)

        # Log results.
        self._log(f"  Shota {self.shota_num} scores:")
        for pid in range(4):
            h_count = sum(1 for c in collected[pid] if c.suit == Suit.HEARTS)
            has_q = QUEEN_OF_SPADES in collected[pid]
            q_str = " +Q" if has_q else ""
            self._log(f"    {PLAYER_NAMES[pid]}: {scores[pid]:+d} ({tricks_won[pid]}T, {h_count}h{q_str})")

        # Special scenario detection.
        zero_trick = [pid for pid in range(4) if tricks_won[pid] == 0]
        all_trick = [pid for pid in range(4) if tricks_won[pid] == 13]
        if all_trick:
            self._log(f"  * ALL TRICKS: {PLAYER_NAMES[all_trick[0]]} (+18)")
        elif len(zero_trick) == 1:
            self._log(f"  * FULL GALLON: {PLAYER_NAMES[zero_trick[0]]} (+20)")
        elif len(zero_trick) == 2:
            names = " & ".join(PLAYER_NAMES[pid] for pid in zero_trick)
            self._log(f"  * HALF GALLON: {names} (+10)")

        self.dealer_id = (self.dealer_id + 1) % 4
        self.last_action_time = pygame.time.get_ticks()

        # Show queued milestone or go to scoring state.
        self.state = "scoring"

    def _end_game(self):
        """End the current game."""
        self.games_played += 1
        winner = max(self.total_scores, key=self.total_scores.get)
        loser = min(self.total_scores, key=self.total_scores.get)
        ai_won = (winner == 0)
        if ai_won:
            self.games_won_by_ai += 1

        # Track for progress bar.
        self._win_history.append(ai_won)
        if self._initial_win_rate is None and len(self._win_history) >= 5:
            self._initial_win_rate = sum(self._win_history[:5]) / 5.0 * 100

        self._log(f"  GAME OVER! Winner: {PLAYER_NAMES[winner]} ({self.total_scores[winner]:+d})")
        self._log(f"  Loser: {PLAYER_NAMES[loser]} ({self.total_scores[loser]:+d})")
        self._log(f"  AI Win Rate: {self.games_won_by_ai}/{self.games_played}")
        self.state = "game_over"
        self.last_action_time = pygame.time.get_ticks()

        # Kick off background training between games (500 silent games).
        if not getattr(self, '_bg_training_active', False):
            self._run_background_training(500)

    def _reset_brain(self):
        """Reset the discovery agent's Q-tables — start learning from scratch."""
        self.discovery = DiscoveryAgent(training=True)
        self.agents[0] = self.discovery
        self.games_played = 0
        self.games_won_by_ai = 0
        self._win_history.clear()
        self._initial_win_rate = None
        self._milestones_achieved.clear()
        self._milestones_list.clear()
        self._milestone_queue.clear()
        self._save_model()
        self._log(f"  BRAIN RESET -- Q-tables cleared. Starting from zero.")

    def _check_milestones(self, collected, tricks_won, scores):
        """
        Detect behavioral milestones — moments the AI shows new learned behavior.
        Only triggers once per milestone. Pauses game for announcement.
        """
        ai_collected = collected.get(0, [])
        ai_tricks = tricks_won.get(0, 0)
        ai_score = scores.get(0, 0)
        ai_hearts = [c for c in ai_collected if c.suit == Suit.HEARTS]
        ai_has_queen = QUEEN_OF_SPADES in ai_collected
        passed = self._pass_cards_selected.get(0, [])
        history = getattr(self, '_shota_trick_history', [])

        # === VERY BASIC POSITIVES (first few games) ===

        # First safe take — won a trick without penalty cards.
        if ai_tricks > 0:
            # Check if AI took at least one trick with no penalties.
            safe_takes = [t for t in history
                          if t["winner"] == 0 and
                          not any(c.suit == Suit.HEARTS or c == QUEEN_OF_SPADES
                                  for _, c in t["cards"])]
            if safe_takes:
                self._trigger_milestone("first_safe_take",
                    "FIRST SAFE TAKE: The AI took a trick that had zero penalty cards. "
                    "It learned that taking tricks is fine when they carry no risk.")

        # First pass completed (if we got here, passing worked).
        if passed:
            self._trigger_milestone("first_pass",
                "FIRST PASS COMPLETED: The AI successfully navigated the passing phase. "
                "It selected cards and exchanged them with another player.")

        # Survived a shota — not the worst score.
        worst_score = min(scores.get(pid, 0) for pid in range(4))
        if ai_score > worst_score:
            self._trigger_milestone("survived_shota",
                "SURVIVED: The AI did not get the worst score this shota. "
                "It performed better than at least one opponent.")

        # Better than one opponent.
        beaten = sum(1 for pid in range(1, 4) if ai_score > scores.get(pid, 0))
        if beaten >= 1:
            self._trigger_milestone("better_than_one",
                "BETTER THAN ONE: Beat at least one opponent's score this shota. "
                "The AI is starting to outperform random play.")

        # Less penalty than last shota (improvement).
        if len(self.shota_scores) >= 2:
            prev_ai = self.shota_scores[-2].get(0, 0) if len(self.shota_scores) >= 2 else 0
            if ai_score > prev_ai and prev_ai < 0:
                self._trigger_milestone("less_penalty",
                    "IMPROVEMENT: Fewer penalties than the previous shota. "
                    "The AI is learning from its mistakes and adapting.")

        # First shota completed.
        self._trigger_milestone("first_shota",
            "FIRST SHOTA COMPLETED: Played all 13 tricks successfully. "
            "The AI navigated an entire round of the game.")

        # First game completed.
        if self.shota_num == 5:
            self._trigger_milestone("first_game",
                "FIRST GAME COMPLETED: Survived all 5 shotas. "
                "The AI endured a full game from start to finish.")

        # No queen this shota.
        if not ai_has_queen:
            self._trigger_milestone("no_queen",
                "NO QUEEN: Avoided the Queen of Spades this shota. "
                "The AI dodged the biggest single penalty in the game (-7).")

        # Won fewer tricks than average (13/4 = 3.25).
        if ai_tricks < 3:
            self._trigger_milestone("fewer_tricks",
                "LOW TRICK COUNT: Won fewer tricks than average. "
                "The AI is naturally trending toward avoidance — taking fewer tricks means fewer penalties.")

        # Positive score in a shota.
        if ai_score > 0:
            self._trigger_milestone("positive_score",
                "POSITIVE SCORE: Scored above zero in a shota. "
                "The AI earned a net positive outcome — bonuses outweighed penalties.")

        # Beat random baseline (won a game = better than 25% expected).
        if len(self._win_history) >= 4:
            recent_4 = self._win_history[-4:]
            if sum(recent_4) >= 2:
                self._trigger_milestone("beat_baseline",
                    "ABOVE RANDOM: Winning more than the 25% random baseline. "
                    "The AI's decisions are now measurably better than random card selection.")

        # === BASIC AWARENESS (early games) ===

        # Clean shota — 0 penalty cards.
        if not ai_hearts and not ai_has_queen and ai_tricks > 0:
            self._trigger_milestone("clean_shota",
                "CLEAN PLAY: Completed a shota without collecting any penalty cards. "
                "The AI learned that some cards are dangerous to take.")

        # Heart avoidance — AI led a non-heart when it had hearts in hand.
        for trick in history:
            cards = trick["cards"]
            if cards and cards[0][0] == 0:  # AI was the leader
                ai_card = trick["ai_card"]
                if ai_card and ai_card.suit != Suit.HEARTS:
                    # Check if AI had hearts it could have led.
                    # (We can't check hand at that moment, but if AI never leads hearts, that's the signal)
                    pass
        # Simpler: if AI led 0 hearts across all 13 tricks but collected 0 penalty.
        ai_leads = [t for t in history if t["cards"] and t["cards"][0][0] == 0]
        ai_led_hearts = [t for t in ai_leads if t["ai_card"] and t["ai_card"].suit == Suit.HEARTS]
        if len(ai_leads) >= 3 and len(ai_led_hearts) == 0 and not ai_hearts:
            self._trigger_milestone("heart_avoidance",
                "HEART AVOIDANCE: The AI avoided leading hearts even when it could. "
                "It discovered that leading hearts exposes you to taking penalty cards.")

        # Low card play — when following suit it can't win, plays lowest.
        low_plays = 0
        follow_plays = 0
        for trick in history:
            ai_card = trick["ai_card"]
            leading = trick["leading_suit"]
            cards = trick["cards"]
            if not ai_card or not leading:
                continue
            if ai_card.suit == leading and cards[0][0] != 0:
                # AI followed suit (wasn't leader).
                follow_plays += 1
                # Check if AI played the lowest of its options.
                suit_cards_played = [c for pid, c in cards if c.suit == leading]
                if suit_cards_played:
                    highest_before = max(RANK_ORDER.get(c.rank, 0)
                                        for pid, c in cards if pid != 0 and c.suit == leading)
                    if RANK_ORDER.get(ai_card.rank, 0) < highest_before:
                        low_plays += 1
        if follow_plays >= 5 and low_plays >= 4:
            self._trigger_milestone("low_card_play",
                "LOW CARD PLAY: The AI consistently plays low cards when it cannot win the trick. "
                "It learned to save high cards and minimize risk on losing tricks.")

        # === INTERMEDIATE STRATEGY (100-500 games) ===

        # Void creation — passed all cards of one suit.
        if passed:
            suit_counts = {}
            for c in passed:
                suit_counts[c.suit] = suit_counts.get(c.suit, 0) + 1
            if any(count >= 3 for count in suit_counts.values()):
                self._trigger_milestone("void_creation",
                    "VOID CREATION: The AI passed 3+ cards of the same suit. "
                    "It discovered that creating voids allows dumping penalty cards later.")

        # Passing strategy — passed 3+ hearts.
        if passed:
            passed_hearts = [c for c in passed if c.suit == Suit.HEARTS]
            if len(passed_hearts) >= 3:
                self._trigger_milestone("pass_hearts",
                    "PASSING STRATEGY: The AI passed 3+ hearts to another player. "
                    "It discovered that getting rid of penalty cards early is valuable.")

        # Queen dump — passed Q.
        if passed and QUEEN_OF_SPADES in passed:
            self._trigger_milestone("pass_queen",
                "QUEEN DUMP: The AI passed the Queen of Spades away. "
                "It discovered the Queen is extremely dangerous to hold (-7 points).")

        # Paint dump — played a heart on someone else's trick when void.
        for trick in history:
            ai_card = trick["ai_card"]
            leading = trick["leading_suit"]
            winner = trick["winner"]
            if ai_card and leading and ai_card.suit == Suit.HEARTS and ai_card.suit != leading and winner != 0:
                self._trigger_milestone("paint_dump",
                    "PAINT DUMP: The AI dumped a heart onto another player's trick. "
                    "It learned to offload penalty cards when void in the led suit.")
                break

        # Queen setup — played A or K of spades early (flush out Q without taking it).
        for trick in history[:6]:  # First 6 tricks = "early"
            ai_card = trick["ai_card"]
            if ai_card and ai_card.suit == Suit.SPADES and RANK_ORDER.get(ai_card.rank, 0) >= 13:
                if not ai_has_queen:  # And didn't end up with Q
                    self._trigger_milestone("queen_setup",
                        "QUEEN SETUP: The AI played high spades (A/K) early in the shota. "
                        "It learned to flush out the Queen of Spades without being stuck with it.")
                    break

        # Zero tricks — full avoidance.
        if ai_tricks == 0:
            self._trigger_milestone("zero_tricks",
                "FULL AVOIDANCE: The AI won zero tricks in a shota. "
                "It learned that avoiding ALL tricks completely eliminates penalty risk.")

        # === ADVANCED PLAY (500+ games) ===

        # Shooting attempt — collected ALL hearts + Q.
        if len(ai_hearts) == 13 and ai_has_queen:
            self._trigger_milestone("shooting",
                "SHOOT THE MOON: The AI collected ALL 13 hearts AND the Queen of Spades. "
                "It discovered the high-risk strategy of taking everything for a massive bonus.")

        # First win.
        if self.shota_num == 5:
            if all(self.total_scores[0] >= self.total_scores[p] for p in range(1, 4)):
                self._trigger_milestone("first_win",
                    "FIRST WIN: The AI won an entire 5-shota game. "
                    "Its overall strategy now consistently outperforms random play.")

        # Queen dodged.
        for pid in range(1, 4):
            if QUEEN_OF_SPADES in collected.get(pid, []):
                self._trigger_milestone("dodged_queen",
                    "QUEEN DODGED: Another player got stuck with the Queen of Spades. "
                    "The AI learned to manage spades carefully to avoid taking her.")
                break

        # Consistent wins — 40%+ over 10 games.
        if len(self._win_history) >= 10:
            recent = self._win_history[-10:]
            if sum(recent) >= 4:
                self._trigger_milestone("consistent_wins",
                    "CONSISTENT PERFORMANCE: Winning 40%+ of recent games (random baseline = 25%). "
                    "The agent has developed a sustained, reliable strategy.")

        # === META-LEARNING (1000+ games) ===

        # Late game dominance — won last 3 games.
        if len(self._win_history) >= 3 and all(self._win_history[-3:]):
            self._trigger_milestone("late_dominance",
                "LATE GAME DOMINANCE: Won 3 games in a row. "
                "The AI's strategy is now strong and consistent — not just lucky.")

        # Opponent exploitation — AI score 2x average opponent over last 5 shotas.
        if len(self.shota_scores) >= 5:
            recent_5 = self.shota_scores[-5:]
            ai_total = sum(s.get(0, 0) for s in recent_5)
            opp_avg = sum(s.get(pid, 0) for s in recent_5 for pid in range(1, 4)) / 3
            if ai_total > opp_avg * 2 and ai_total > 0:
                self._trigger_milestone("opponent_exploitation",
                    "OPPONENT EXPLOITATION: AI score is 2x the opponent average over 5 shotas. "
                    "It learned to consistently outmaneuver the competition.")

        # === CARD-PASS WISE ===

        # Strategic pass — passed high cards (A/K) of a short suit.
        if passed:
            high_non_hearts = [c for c in passed
                               if c.suit != Suit.HEARTS and RANK_ORDER.get(c.rank, 0) >= 13]
            if len(high_non_hearts) >= 2:
                self._trigger_milestone("strategic_pass",
                    "STRATEGIC PASS: Passed 2+ high cards (A/K) of non-heart suits. "
                    "The AI learned that holding high cards forces you to win tricks you don't want.")

        # Keep low hearts — passed high non-hearts but kept low hearts.
        if passed:
            passed_non_hearts_high = [c for c in passed
                                      if c.suit != Suit.HEARTS and RANK_ORDER.get(c.rank, 0) >= 10]
            kept_low_hearts = [c for c in ai_collected
                               if c.suit == Suit.HEARTS and RANK_ORDER.get(c.rank, 0) <= 5]
            if len(passed_non_hearts_high) >= 2 and len(ai_hearts) == 0:
                self._trigger_milestone("keep_low_hearts",
                    "SAFE HOLDING: Passed high non-hearts but avoided all penalties. "
                    "The AI discovered that low cards are safe — high cards are the real danger.")

        # === TRICK WISE ===

        # Safe lead — led a low card and the trick had no penalty cards.
        for trick in history:
            cards = trick["cards"]
            if cards and cards[0][0] == 0:  # AI led
                ai_card = trick["ai_card"]
                trick_cards = [c for _, c in cards]
                has_penalty = any(c.suit == Suit.HEARTS or c == QUEEN_OF_SPADES for c in trick_cards)
                if ai_card and RANK_ORDER.get(ai_card.rank, 0) <= 6 and not has_penalty and trick["winner"] == 0:
                    self._trigger_milestone("safe_lead",
                        "SAFE LEAD: Led a low card and won a trick with zero penalty cards. "
                        "The AI learned that leading low in safe suits lets you win tricks risk-free.")
                    break

        # Forced take — won a trick with 0 penalty cards intentionally.
        safe_wins = sum(1 for t in history
                        if t["winner"] == 0 and
                        not any(c.suit == Suit.HEARTS or c == QUEEN_OF_SPADES
                                for _, c in t["cards"]))
        if safe_wins >= 5:
            self._trigger_milestone("forced_take",
                "SELECTIVE WINNING: Won 5+ tricks that had zero penalty cards. "
                "The AI learned that winning tricks is fine — as long as they are clean.")

        # Heart break — AI played a heart when void in led suit (breaking hearts).
        for trick in history:
            ai_card = trick["ai_card"]
            leading = trick["leading_suit"]
            if ai_card and leading and ai_card.suit == Suit.HEARTS and leading != Suit.HEARTS:
                self._trigger_milestone("heart_break",
                    "HEARTS BROKEN: The AI played a heart when void in the led suit. "
                    "It discovered that breaking hearts is sometimes necessary to offload them.")
                break

        # High dump — dumped Q on someone when void in spades.
        for trick in history:
            ai_card = trick["ai_card"]
            leading = trick["leading_suit"]
            winner = trick["winner"]
            if (ai_card and ai_card == QUEEN_OF_SPADES and
                    leading and leading != Suit.SPADES and winner != 0):
                self._trigger_milestone("high_dump",
                    "QUEEN OFFLOAD: Dumped the Queen of Spades onto another player's trick. "
                    "The AI learned to exploit voids to get rid of the most dangerous card.")
                break

        # === SHOTA WISE ===

        # Minimal damage — only 1 penalty point in entire shota.
        total_penalty = len(ai_hearts) + (7 if ai_has_queen else 0)
        if total_penalty == 1 and ai_tricks > 0:
            self._trigger_milestone("minimal_damage",
                "MINIMAL DAMAGE: Only 1 penalty point in the entire shota. "
                "The AI played with surgical precision — nearly perfect avoidance.")

        # Best score in shota.
        if ai_score > 0 and all(ai_score >= scores.get(pid, 0) for pid in range(1, 4)):
            self._trigger_milestone("best_shota",
                "BEST IN SHOTA: The AI had the highest score among all 4 players this shota. "
                "Its strategy produced the best individual outcome.")

        # Gallon achieved — zero tricks bonus.
        if ai_tricks == 0 and ai_score >= 20:
            self._trigger_milestone("gallon",
                "FULL GALLON: Won zero tricks and earned the +20 bonus. "
                "The AI mastered the art of complete trick avoidance for maximum reward.")

        # === GAME WISE ===

        # Winning streak — won 5 in a row.
        if len(self._win_history) >= 5 and all(self._win_history[-5:]):
            self._trigger_milestone("winning_streak",
                "WINNING STREAK: Won 5 games in a row. "
                "The AI has achieved sustained dominance over its opponents.")

        # Mastery — 50%+ win rate over 20+ games.
        if len(self._win_history) >= 20:
            rate = sum(self._win_history[-20:]) / 20
            if rate >= 0.5:
                self._trigger_milestone("mastery",
                    "MASTERY: Win rate exceeds 50% over 20 games (random = 25%). "
                    "The AI has doubled the expected performance through pure learning.")

        # Comeback — won after being last at shota 3.
        if self.shota_num == 5 and len(self.shota_scores) >= 5:
            # Check if AI was last after 3 shotas.
            scores_after_3 = {pid: sum(s.get(pid, 0) for s in self.shota_scores[-5:-2])
                              for pid in range(4)}
            was_last = all(scores_after_3[0] <= scores_after_3[pid] for pid in range(1, 4))
            is_winner = all(self.total_scores[0] >= self.total_scores[pid] for pid in range(1, 4))
            if was_last and is_winner:
                self._trigger_milestone("comeback",
                    "COMEBACK: Was in last place after 3 shotas but won the game. "
                    "The AI adapted its strategy mid-game to recover from a bad position.")

        # Adaptation — win rate jumped 15%+ after a losing period.
        if len(self._win_history) >= 15:
            early = sum(self._win_history[-15:-5]) / 10
            late = sum(self._win_history[-5:]) / 5
            if late - early >= 0.15 and late >= 0.4:
                self._trigger_milestone("adaptation",
                    "ADAPTATION: Win rate jumped significantly after a weak period. "
                    "The AI adjusted its strategy based on accumulated experience.")

        # === PARTNER COOPERATION (understanding implicit teamwork) ===

        # Partner shield — AI avoided taking a trick that a partner also avoided.
        # In Hearts there are no formal teams, but avoiding helping opponents is cooperation.
        # Detect: AI played low to let someone else take a clean trick (non-penalty).
        for trick in history:
            cards = trick["cards"]
            ai_card = trick["ai_card"]
            winner = trick["winner"]
            if ai_card and winner != 0 and len(cards) == 4:
                trick_cards_list = [c for _, c in cards]
                has_penalty = any(c.suit == Suit.HEARTS or c == QUEEN_OF_SPADES
                                  for c in trick_cards_list)
                if has_penalty and RANK_ORDER.get(ai_card.rank, 0) <= 5:
                    self._trigger_milestone("penalty_deflection",
                        "PENALTY DEFLECTION: Played a low card to avoid winning a trick with penalties. "
                        "The AI learned to duck when danger cards are on the table.")
                    break

        # Poison pill — AI led a suit that forced an opponent to take the queen.
        for trick in history:
            cards = trick["cards"]
            winner = trick["winner"]
            if cards and cards[0][0] == 0 and winner != 0:  # AI led, opponent won
                trick_cards_list = [c for _, c in cards]
                if QUEEN_OF_SPADES in trick_cards_list:
                    self._trigger_milestone("poison_pill",
                        "POISON PILL: Led a suit that forced an opponent to eat the Queen of Spades. "
                        "The AI learned to set traps that push dangerous cards onto others.")
                    break

        # Sacrifice play — AI intentionally won a clean trick to maintain lead control.
        ai_wins_clean = [t for t in history if t["winner"] == 0 and
                         not any(c.suit == Suit.HEARTS or c == QUEEN_OF_SPADES
                                 for _, c in t["cards"])]
        ai_wins_after_clean = 0
        for i, t in enumerate(history):
            if t["winner"] == 0 and i > 0 and history[i-1]["winner"] == 0:
                ai_wins_after_clean += 1
        if ai_wins_after_clean >= 3 and len(ai_hearts) == 0:
            self._trigger_milestone("lead_control",
                "LEAD CONTROL: Won consecutive clean tricks to dictate the flow of play. "
                "The AI learned that controlling the lead lets you steer away from danger.")

        # === OPPONENT EXPLOITATION (punishing mistakes) ===

        # Spade flush — played high spades to flush out the Queen from opponents.
        spade_flushes = [t for t in history if t["ai_card"] and
                         t["ai_card"].suit == Suit.SPADES and
                         RANK_ORDER.get(t["ai_card"].rank, 0) >= 12 and
                         t["cards"][0][0] == 0]  # AI led high spade
        if len(spade_flushes) >= 2 and not ai_has_queen:
            self._trigger_milestone("spade_flush",
                "SPADE FLUSH: Led multiple high spades to force the Queen out. "
                "The AI learned to systematically smoke out the Queen of Spades from opponents' hands.")

        # Late game exploitation — won last 3 tricks cleanly when opponents had penalty cards.
        if len(history) >= 13:
            last_3 = history[-3:]
            ai_won_last_3_clean = all(
                t["winner"] == 0 and
                not any(c.suit == Suit.HEARTS or c == QUEEN_OF_SPADES for _, c in t["cards"])
                for t in last_3
            )
            if ai_won_last_3_clean:
                self._trigger_milestone("endgame_control",
                    "ENDGAME CONTROL: Won the last 3 tricks cleanly. "
                    "The AI learned to dominate the endgame when penalty cards have been exhausted.")

        # Opponent burdening — AI dumped multiple hearts on a single opponent.
        if len(history) >= 5:
            opp_hearts_taken = {1: 0, 2: 0, 3: 0}
            for t in history:
                if t["winner"] in opp_hearts_taken:
                    hearts_in = sum(1 for _, c in t["cards"] if c.suit == Suit.HEARTS)
                    opp_hearts_taken[t["winner"]] += hearts_in
            max_opp_hearts = max(opp_hearts_taken.values())
            if max_opp_hearts >= 6 and len(ai_hearts) <= 1:
                self._trigger_milestone("opponent_burden",
                    "OPPONENT BURDENING: Concentrated hearts onto a single opponent (6+ hearts). "
                    "The AI learned to target one player with penalty cards while staying clean.")

        # === GAME-WINNING STRATEGIES ===

        # Perfect shota — zero penalty AND positive bonus.
        if ai_score >= 20 and len(ai_hearts) == 0 and not ai_has_queen:
            self._trigger_milestone("perfect_shota",
                "PERFECT SHOTA: Zero penalties combined with a bonus (Gallon/Half-Gallon). "
                "The AI achieved the optimal outcome — maximum reward with zero risk.")

        # Comeback king — was losing by 10+ at shota 3, won the game.
        if self.shota_num == 5 and len(self.shota_scores) >= 5:
            scores_after_3 = {pid: sum(s.get(pid, 0) for s in self.shota_scores[-5:-2])
                              for pid in range(4)}
            deficit = max(scores_after_3[pid] for pid in range(1, 4)) - scores_after_3[0]
            is_winner = all(self.total_scores[0] >= self.total_scores[pid] for pid in range(1, 4))
            if deficit >= 10 and is_winner:
                self._trigger_milestone("comeback_king",
                    "COMEBACK KING: Overcame a 10+ point deficit after shota 3 to win the game. "
                    "The AI learned to shift into aggressive recovery mode when behind.")

        # Runaway leader — won a game by 15+ points over all opponents.
        if self.shota_num == 5:
            margin = min(self.total_scores[0] - self.total_scores[pid] for pid in range(1, 4))
            if margin >= 15:
                self._trigger_milestone("runaway_leader",
                    "RUNAWAY LEADER: Won a game with 15+ point margin over every opponent. "
                    "The AI achieved complete strategic superiority throughout the game.")

        # Marathon consistency — positive score in 4+ consecutive shotas.
        if len(self.shota_scores) >= 4:
            recent_4_scores = [s.get(0, 0) for s in self.shota_scores[-4:]]
            if all(s >= 0 for s in recent_4_scores):
                self._trigger_milestone("marathon_consistency",
                    "MARATHON CONSISTENCY: Non-negative score in 4+ consecutive shotas. "
                    "The AI maintains disciplined play across extended stretches without slipping.")

        # === ADVANCED PLAY PATTERNS ===

        # Tempo play — AI led low, then high in same suit across different tricks.
        suit_leads = {}  # suit -> list of ranks AI led
        for t in history:
            if t["cards"] and t["cards"][0][0] == 0 and t["ai_card"]:
                s = t["ai_card"].suit
                if s not in suit_leads:
                    suit_leads[s] = []
                suit_leads[s].append(RANK_ORDER.get(t["ai_card"].rank, 0))
        for s, ranks in suit_leads.items():
            if len(ranks) >= 2 and ranks[0] <= 6 and ranks[-1] >= 12:
                self._trigger_milestone("tempo_play",
                    "TEMPO PLAY: Led low then high in the same suit across tricks. "
                    "The AI learned to probe with low cards first, then strike with power cards.")
                break

        # Counting cards — AI played the exact card needed to win or avoid penalty.
        # Detected by: AI plays highest card below the current highest, avoiding taking the trick.
        duck_plays = 0
        for trick in history:
            cards = trick["cards"]
            ai_card = trick["ai_card"]
            leading = trick["leading_suit"]
            winner = trick["winner"]
            if (ai_card and leading and ai_card.suit == leading and
                    winner != 0 and len(cards) == 4):
                # AI played in suit but didn't win — was it a strategic duck?
                all_same_suit = [RANK_ORDER.get(c.rank, 0) for _, c in cards if c.suit == leading]
                ai_rank = RANK_ORDER.get(ai_card.rank, 0)
                if all_same_suit and ai_rank == sorted(all_same_suit)[-2]:
                    # AI played second-highest — a precise duck
                    duck_plays += 1
        if duck_plays >= 3:
            self._trigger_milestone("precise_ducking",
                "PRECISE DUCKING: Played the second-highest card multiple times to narrowly avoid winning. "
                "The AI learned card counting — knowing exactly how high it can play without taking the trick.")

        # Void exploitation — AI dumped penalty cards in 3+ different tricks using voids.
        void_dumps = 0
        for trick in history:
            ai_card = trick["ai_card"]
            leading = trick["leading_suit"]
            winner = trick["winner"]
            if (ai_card and leading and ai_card.suit != leading and
                    (ai_card.suit == Suit.HEARTS or ai_card == QUEEN_OF_SPADES) and
                    winner != 0):
                void_dumps += 1
        if void_dumps >= 3:
            self._trigger_milestone("systematic_void_exploit",
                "SYSTEMATIC VOID EXPLOITATION: Dumped penalty cards 3+ times via voids in a single shota. "
                "The AI mastered void creation and repeatedly exploits it to offload dangerous cards.")

        # Two-phase strategy — played defensively first half, aggressively second half.
        if len(history) >= 13:
            first_half = history[:6]
            second_half = history[7:]
            ai_wins_first = sum(1 for t in first_half if t["winner"] == 0)
            ai_wins_second = sum(1 for t in second_half if t["winner"] == 0)
            if ai_wins_first <= 1 and ai_wins_second >= 4 and len(ai_hearts) <= 1:
                self._trigger_milestone("two_phase_strategy",
                    "TWO-PHASE STRATEGY: Played passive early (0-1 tricks) then aggressive late (4+ tricks). "
                    "The AI learned to lay low while dangers circulate, then dominate the safe endgame.")

        # Suit control — AI led the same suit 3+ times and won most of those tricks.
        for s, ranks in suit_leads.items():
            if len(ranks) >= 3:
                suit_trick_wins = sum(1 for t in history
                                      if t["cards"] and t["cards"][0][0] == 0 and
                                      t["ai_card"] and t["ai_card"].suit == s and
                                      t["winner"] == 0)
                if suit_trick_wins >= 2:
                    self._trigger_milestone("suit_domination",
                        "SUIT DOMINATION: Led and won 3+ tricks in a single suit. "
                        "The AI learned to establish dominance in a suit where it holds length and power.")
                    break

        # Queen trap — AI held the Queen until an opponent led spades, then ducked under.
        queen_in_hand_start = any(c == QUEEN_OF_SPADES for c in (self._pass_cards_received.get(0, []) +
                                                                   [c for c in ai_collected]))
        if not ai_has_queen and queen_in_hand_start:
            # AI had the queen initially (received or dealt) but didn't end up with it
            for trick in history:
                cards = trick["cards"]
                leading = trick["leading_suit"]
                if leading == Suit.SPADES and trick["winner"] != 0:
                    ai_card = trick["ai_card"]
                    if ai_card and ai_card == QUEEN_OF_SPADES:
                        self._trigger_milestone("queen_trap",
                            "QUEEN TRAP: Held the Queen of Spades and played it when someone else led spades. "
                            "The AI learned to time the Queen dump perfectly — let others win the spade trick.")
                        break

        # === META-MASTERY (2000+ games) ===

        # Unbeatable streak — won 10 games in a row.
        if len(self._win_history) >= 10 and all(self._win_history[-10:]):
            self._trigger_milestone("unbeatable",
                "UNBEATABLE: Won 10 games in a row. "
                "The AI has achieved a level of play that opponents cannot counter.")

        # Grand mastery — 60%+ win rate over 50+ games.
        if len(self._win_history) >= 50:
            rate = sum(self._win_history[-50:]) / 50
            if rate >= 0.6:
                self._trigger_milestone("grand_mastery",
                    "GRAND MASTERY: Win rate exceeds 60% over 50 games (random = 25%). "
                    "The AI has more than doubled optimal random performance through deep learning.")

        # Score efficiency — average score per shota > +5 over 20 shotas.
        if len(self.shota_scores) >= 20:
            recent_20 = self.shota_scores[-20:]
            avg_score = sum(s.get(0, 0) for s in recent_20) / 20
            if avg_score >= 5:
                self._trigger_milestone("score_efficiency",
                    "SCORE EFFICIENCY: Averaging +5 per shota over 20 rounds. "
                    "The AI consistently extracts value from every shota — not just surviving, but thriving.")

        # Penalty minimizer — averaged less than 1 heart per shota over 10 shotas.
        if len(self.shota_scores) >= 10:
            # Track penalty accumulation across recent history
            if not hasattr(self, '_recent_hearts_taken'):
                self._recent_hearts_taken = []
            self._recent_hearts_taken.append(len(ai_hearts))
            if len(self._recent_hearts_taken) >= 10:
                avg_hearts = sum(self._recent_hearts_taken[-10:]) / 10
                if avg_hearts <= 1.0:
                    self._trigger_milestone("penalty_minimizer",
                        "PENALTY MINIMIZER: Averaging 1 or fewer hearts per shota over 10 rounds. "
                        "The AI has nearly eliminated penalty card exposure through precise play.")

        # Adaptive passing — different pass strategies in consecutive shotas.
        if len(self.shota_scores) >= 3:
            if not hasattr(self, '_pass_history'):
                self._pass_history = []
            self._pass_history.append(passed)
            if len(self._pass_history) >= 3:
                recent_passes = self._pass_history[-3:]
                suits_passed = [set(c.suit for c in p) for p in recent_passes if p]
                if len(suits_passed) >= 3 and len(set(frozenset(s) for s in suits_passed)) >= 3:
                    self._trigger_milestone("adaptive_passing",
                        "ADAPTIVE PASSING: Different suit patterns in 3 consecutive passes. "
                        "The AI adjusts its passing strategy based on each hand — not just following a formula.")

    def _trigger_milestone(self, key: str, message: str):
        """Record a discovered behavior with title and dynamic description.
        
        Descriptions are enriched with live game stats so they reflect
        actual performance rather than static text.
        """
        if key in self._milestones_achieved:
            return
        self._milestones_achieved.add(key)

        # Build dynamic context from current game state.
        stats = self._build_stats_context()

        # Split: title is before colon, base description is after.
        if ":" in message:
            title = message.split(":")[0].strip()
            base_desc = message.split(":", 1)[1].strip()
        else:
            title = key.upper()
            base_desc = message

        # Generate dynamic suffix with actual numbers.
        desc = self._enrich_description(key, base_desc, stats)

        self._milestones_list.append((title, desc))
        self._log(f"  ** DISCOVERED: {title} **")

    def _build_stats_context(self) -> dict:
        """Gather current performance stats for dynamic descriptions."""
        win_rate = (self.games_won_by_ai / max(self.games_played, 1)) * 100
        recent_win_rate = 0
        if len(self._win_history) >= 5:
            recent_win_rate = sum(self._win_history[-5:]) / 5 * 100

        q_states = len(self.discovery.play_q) + len(self.discovery.pass_q)
        episodes = self.discovery.episodes_trained
        epsilon = self.discovery.epsilon

        return {
            "games_played": self.games_played,
            "games_won": self.games_won_by_ai,
            "win_rate": win_rate,
            "recent_win_rate": recent_win_rate,
            "episodes": episodes,
            "q_states": q_states,
            "epsilon": epsilon,
            "shota_num": self.shota_num,
        }

    def _enrich_description(self, key: str, base_desc: str, stats: dict) -> str:
        """Add live stats to the description based on milestone type."""
        suffix_parts = []

        if stats["episodes"] > 0:
            suffix_parts.append(f"Shota #{stats['episodes']}")

        if stats["win_rate"] > 0:
            suffix_parts.append(f"win rate: {stats['win_rate']:.0f}%")

        suffix = f" [{', '.join(suffix_parts)}]" if suffix_parts else ""
        return f"{base_desc}{suffix}"

    def _show_next_milestone(self):
        """No longer used — milestones don't interrupt."""
        return False

    def _run_background_training(self, num_games: int = 500):
        """
        Run silent games in a background thread.
        Same contract: environment + legal moves + score signal only.
        No new knowledge — just more experience.
        """
        if getattr(self, '_bg_training_active', False):
            return  # Already running.
        self._bg_training_active = True
        self._bg_training_done = 0
        self._bg_training_target = num_games
        thread = threading.Thread(target=self._bg_train_worker, args=(num_games,), daemon=True)
        thread.start()
        self._log(f"  Background training: {num_games} silent games started...")

    def _bg_train_worker(self, num_games: int):
        """Worker thread: plays silent games and calls reward() after each shota."""
        agent = self.discovery  # Same agent instance — shared Q-tables.
        done = 0

        while done < num_games:
            # Setup.
            players = [HeartsPlayer(player_id=i) for i in range(4)]
            deck = Deck()
            deck.shuffle()
            for p in players:
                p.receive_cards(deck.deal(13))

            # Self-play: all use same Q-tables but only player 0 records episode.
            # Create non-training copies for opponents (share Q-tables, don't record).
            opp = DiscoveryAgent(training=False)
            opp.play_q = agent.play_q
            opp.pass_q = agent.pass_q
            opp.epsilon = agent.epsilon
            agents = [agent, opp, opp, opp]

            # Passing phase.
            cards_to_pass = {}
            for p in players:
                obs = PassingObservation(player_id=p.player_id, hand=list(p.hand))
                action = agents[p.player_id].act(obs)
                cards_to_pass[p.player_id] = list(action.cards)

            for p in players:
                p.remove_cards(cards_to_pass[p.player_id])
            for pid in range(4):
                receiver_id = (pid + 1) % 4
                players[receiver_id].receive_cards(cards_to_pass[pid])

            # Play 13 tricks.
            dealer_id = done % 4
            first_leader = (dealer_id + 1) % 4
            env = HeartsEnvironment(players, first_leader)

            while not env.is_shota_complete():
                for _ in range(4):
                    pid = env.current_player_id()
                    obs = env.observe(pid)
                    action = agents[pid].act(obs)
                    env.apply_action(action)

            # Score and reward.
            collected = {p.player_id: list(p.collected_cards) for p in players}
            tricks_won = {p.player_id: p.tricks_won for p in players}
            scores = score_shota(collected, tricks_won)
            agent.reward(float(scores.get(0, 0)))

            # Check milestones from background game.
            ai_collected = collected.get(0, [])
            ai_tricks = tricks_won.get(0, 0)
            ai_score = scores.get(0, 0)
            ai_hearts = [c for c in ai_collected if c.suit == Suit.HEARTS]
            ai_has_queen = QUEEN_OF_SPADES in ai_collected
            passed = cards_to_pass.get(0, [])

            if not ai_hearts and not ai_has_queen and ai_tricks > 0:
                self._trigger_milestone("clean_shota",
                    "CLEAN PLAY: Completed a shota without collecting any penalty cards.")
            if ai_tricks == 0:
                self._trigger_milestone("zero_tricks",
                    "FULL AVOIDANCE: Won zero tricks -- no penalties possible.")
            if ai_score > 0:
                self._trigger_milestone("positive_score",
                    "POSITIVE SCORE: Scored above zero in a shota.")
            if not ai_has_queen:
                self._trigger_milestone("no_queen",
                    "NO QUEEN: Avoided the Queen of Spades this shota.")
            if passed:
                passed_hearts = [c for c in passed if c.suit == Suit.HEARTS]
                if len(passed_hearts) >= 3:
                    self._trigger_milestone("pass_hearts",
                        "PASSING STRATEGY: Passed 3+ hearts to another player.")
                if QUEEN_OF_SPADES in passed:
                    self._trigger_milestone("pass_queen",
                        "QUEEN DUMP: Passed the Queen of Spades away.")

            # Epsilon decay — explore less over time.
            if agent.episodes_trained % 100 == 0 and agent.epsilon > 0.05:
                agent.epsilon *= 0.995

            done += 1
            self._bg_training_done = done

        self._bg_training_active = False

    def run(self):
        """Main loop."""
        while self.running:
            self._handle_events()
            self._update()
            self._render()
            self.clock.tick(FPS)
        # Auto-save model on exit.
        self._save_model()
        pygame.quit()

    def _save_model(self):
        """Save the discovery agent's learned Q-tables and milestones."""
        if self.discovery and self.discovery.episodes_trained > 0:
            save_path = self.model_path or "agents/hearts_discovery/hearts_model.json"
            try:
                self.discovery.save(save_path)
                print(f"Model saved: {self.discovery.episodes_trained} episodes")
            except Exception as e:
                print(f"Failed to save model: {e}")

        # Save milestones separately.
        import json
        milestones_path = "agents/hearts_discovery/milestones.json"
        try:
            data = {
                "achieved": list(self._milestones_achieved),
                "list": self._milestones_list,
            }
            with open(milestones_path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _load_milestones(self):
        """Load previously discovered milestones."""
        import json
        milestones_path = "agents/hearts_discovery/milestones.json"
        try:
            with open(milestones_path, "r") as f:
                data = json.load(f)
            self._milestones_achieved = set(data.get("achieved", []))
            self._milestones_list = [tuple(x) if isinstance(x, list) else x
                                     for x in data.get("list", [])]
        except (FileNotFoundError, Exception):
            pass

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
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_click(event.pos)
            elif event.type == pygame.MOUSEWHEEL:
                # Scroll the discoveries panel.
                self._disc_scroll_offset = getattr(self, '_disc_scroll_offset', 0)
                self._disc_scroll_offset += event.y * 1
                self._disc_scroll_offset = max(0, self._disc_scroll_offset)

    def _handle_click(self, pos):
        """Handle mouse clicks — Continue button, mode toggle, reset."""
        if hasattr(self, '_continue_btn_rect') and self._continue_btn_rect:
            if self._continue_btn_rect.collidepoint(pos):
                self._on_continue()
        if hasattr(self, '_mode_btn_rect') and self._mode_btn_rect:
            if self._mode_btn_rect.collidepoint(pos):
                self.mode = "follow" if self.mode == "watch" else "watch"
                self._log(f"  Mode: {self.mode.upper()}")
        if hasattr(self, '_reset_btn_rect') and self._reset_btn_rect:
            if self._reset_btn_rect.collidepoint(pos):
                self._reset_brain()

    def _on_continue(self):
        """Handle Continue button press."""
        if self.state == "pass_show":
            self._do_pass_finalize()
        elif self.state in ("pass_show_out", "pass_show_in"):
            self._do_pass_finalize()
        elif self.state == "trick_pause":
            self.state = "playing"
            self.last_action_time = pygame.time.get_ticks()
        elif self.state == "scoring":
            self._start_new_shota()
        elif self.state == "game_over":
            self._start_new_game()

    def _update(self):
        """Auto-advance the game state based on timing and mode."""
        if self.paused:
            return

        now = pygame.time.get_ticks()

        if self.mode == "watch":
            # Watch mode: fast auto-play (2x speed cap).
            effective_speed = max(self.speed, 2.0)
            delay = int(TRICK_DELAY_MS / effective_speed)
            if now - self.last_action_time < delay:
                return

            if self.state == "passing":
                self._do_passing_learn()
            elif self.state in ("pass_show", "pass_show_out", "pass_show_in"):
                # Auto-advance in watch mode.
                self._do_pass_finalize()
            elif self.state == "playing":
                if not self.env.is_shota_complete():
                    self._do_one_trick()
                else:
                    self._do_scoring()
            elif self.state == "trick_pause":
                # Auto-advance in watch mode.
                self.state = "playing"
                self.last_action_time = now
            elif self.state == "scoring":
                if now - self.last_action_time > int(SHOTA_DELAY_MS / self.speed):
                    self._start_new_shota()
            elif self.state == "game_over":
                if now - self.last_action_time > int(SHOTA_DELAY_MS * 2 / self.speed):
                    self._start_new_game()

        else:
            # Follow mode: slower, requires Continue at key moments.
            delay = int(TRICK_DELAY_MS * 1.2 / self.speed)
            if now - self.last_action_time < delay:
                return

            if self.state == "passing":
                self._do_passing()
            elif self.state == "playing":
                if not self.env.is_shota_complete():
                    self._do_one_trick_follow()
                else:
                    self._do_scoring()
            # pass_show, trick_pause, scoring, game_over → wait for Continue.

    def _render(self):
        """Render the full interface."""
        self.screen.fill(BG_DARK)

        # Table area (center).
        table_rect = pygame.Rect(20, 60, self._screen_w - 320, self._screen_h - 80)
        pygame.draw.rect(self.screen, TABLE_FELT, table_rect, border_radius=15)
        pygame.draw.rect(self.screen, TABLE_BORDER, table_rect, width=2, border_radius=15)

        # Title.
        title = self.fonts["title"].render(TITLE, True, TEXT_WHITE)
        self.screen.blit(title, (25, 15))

        # Game info — right-aligned above the panel, two lines, white.
        panel_x = self._screen_w - 290
        info_line1 = f"Shota {self.shota_num}/5  |  Trick {min(self.trick_num, 13)}/13  |  Speed: {self.speed:.1f}x"
        info_line2 = f"Games: {self.games_played}  |  AI won: {self.games_won_by_ai}  |  SPACE: pause  |  ESC: quit"
        info1_surf = self.fonts["small"].render(info_line1, True, TEXT_WHITE)
        info2_surf = self.fonts["small"].render(info_line2, True, TEXT_WHITE)
        self.screen.blit(info1_surf, (panel_x - info1_surf.get_width() - 10, 15))
        self.screen.blit(info2_surf, (panel_x - info2_surf.get_width() - 10, 32))

        if self.paused:
            pause_surf = self.fonts["large"].render("PAUSED", True, TEXT_GOLD)
            self.screen.blit(pause_surf, (self._screen_w // 2 - 40, 15))

        # Render player hands and info.
        self._render_players(table_rect)

        # Render current trick in center.
        self._render_trick(table_rect)

        # Render pass phase overlay (follow mode only).
        if self.mode == "follow" and self.state in ("pass_show", "pass_show_out", "pass_show_in"):
            self._render_pass_cards(table_rect)

        # Render Continue button only in Follow mode when needed.
        self._continue_btn_rect = None
        if self.mode == "follow" and self.state in ("pass_show", "pass_show_out", "pass_show_in", "trick_pause", "scoring", "game_over"):
            self._render_continue_button(table_rect)

        # Mode toggle button (top-right of table).
        self._render_mode_button(table_rect)

        # Reset brain button (below mode button).
        self._render_reset_button(table_rect)

        # Right panel — scores and log.
        self._render_side_panel()

        pygame.display.flip()


    def _render_players(self, table_rect):
        """Render all 4 players with clean, stable positioning."""
        cx, cy = table_rect.centerx, table_rect.centery

        for pid in range(4):
            hand = self.hands.get(pid, [])
            color = PLAYER_COLORS[pid]
            name_surf = self.fonts["medium"].render(PLAYER_NAMES[pid], True, color)
            score_surf = self.fonts["small"].render(
                f"{self.total_scores[pid]:+d}", True, TEXT_GOLD)

            if pid == 0:  # Bottom — Discovery AI
                hand_y = table_rect.bottom - CARD_HEIGHT - 55
                self._render_hand_h(hand, cx, hand_y)
                self.screen.blit(score_surf, (cx + 5, hand_y + CARD_HEIGHT + 6))

            elif pid == 2:  # Top — Random-2
                self.screen.blit(name_surf,
                    (cx - name_surf.get_width() // 2, table_rect.top + 10))
                self.screen.blit(score_surf,
                    (cx + name_surf.get_width() // 2 + 6, table_rect.top + 12))
                self._render_hand_h(hand, cx, table_rect.top + 30)

            elif pid == 1:  # Left — Random-1
                bx = table_rect.left + 15
                self.screen.blit(name_surf, (bx, cy - 155))
                self.screen.blit(score_surf, (bx, cy - 138))
                self._render_hand_v(hand, bx + 4, cy - 115)

            elif pid == 3:  # Right — Random-3
                bx = table_rect.right - CARD_MINI_W - 15
                self.screen.blit(name_surf,
                    (bx + CARD_MINI_W - name_surf.get_width(), cy - 155))
                self.screen.blit(score_surf,
                    (bx + CARD_MINI_W - score_surf.get_width(), cy - 138))
                self._render_hand_v(hand, bx, cy - 115)

    def _render_hand_h(self, hand, cx, y):
        """Horizontal hand — fixed 13-card left anchor so cards don't jump."""
        if not hand:
            return
        overlap = min(CARD_WIDTH - 8, (self._screen_w - 500) // 13)
        start_x = cx - (overlap * 12 + CARD_WIDTH) // 2
        for i, card in enumerate(hand):
            self.screen.blit(self._get_card_surface(card), (start_x + i * overlap, y))

    def _render_hand_v(self, hand, x, start_y):
        """Vertical hand with fixed 16px overlap."""
        if not hand:
            return
        for i, card in enumerate(hand):
            surf = pygame.transform.smoothscale(
                self._get_card_surface(card), (CARD_MINI_W, CARD_MINI_H))
            self.screen.blit(surf, (x, start_y + i * 16))

    def _render_trick(self, table_rect):
        """Render the 4 trick cards in the center."""
        if not self.current_trick_cards:
            return
        cx, cy = table_rect.centerx, table_rect.centery
        offsets = {0: (0, 44), 1: (-70, 0), 2: (0, -44), 3: (70, 0)}
        for pid, card in self.current_trick_cards:
            ox, oy = offsets[pid]
            self.screen.blit(self._get_card_surface(card),
                (cx + ox - CARD_WIDTH // 2, cy + oy - CARD_HEIGHT // 2))

        if self.last_winner >= 0 and len(self.current_trick_cards) == 4:
            t = self.fonts["small"].render(
                f"Taker: {PLAYER_NAMES[self.last_winner]}", True, TEXT_GOLD)
            self.screen.blit(t, (table_rect.left + 15, table_rect.top + 10))

        self._render_small_card_slots(table_rect)

    def _slot_positions(self, table_rect):
        """Single source of truth for pass/collected card slot positions."""
        cx, cy = table_rect.centerx, table_rect.centery
        sw, sh = CARD_SMALL_W, CARD_SMALL_H

        # Compute fixed hand left/right.
        overlap = min(CARD_WIDTH - 8, (self._screen_w - 500) // 13)
        full_w = overlap * 12 + CARD_WIDTH
        hand_left = cx - full_w // 2
        hand_right = cx + full_w // 2

        # P0 (bottom): above hand, left-aligned.
        p0_y = table_rect.bottom - CARD_HEIGHT - 55 - sh - 6
        # P2 (top): below hand, right-aligned.
        p2_y = table_rect.top + 30 + CARD_HEIGHT + 6
        # P1 (left): right of hand column.
        p1_x = table_rect.left + 15 + CARD_MINI_W + 10
        p1_y = cy - 115
        # P3 (right): left of hand column.
        p3_x = table_rect.right - CARD_MINI_W - 15 - sw - 10
        p3_y = cy - 115

        return {
            0: (hand_left, p0_y, "h"),
            1: (p1_x, p1_y, "v"),
            2: (hand_right, p2_y, "hr"),  # hr = horizontal right-aligned
            3: (p3_x, p3_y, "v"),
        }

    def _draw_small_cards(self, cards, x, y, direction):
        """Draw a list of small cards at (x,y) in direction h/hr/v."""
        sw, sh = CARD_SMALL_W, CARD_SMALL_H
        if not cards:
            return
        if direction == "h":
            for i, c in enumerate(cards):
                s = pygame.transform.smoothscale(self._get_card_surface(c), (sw, sh))
                self.screen.blit(s, (x + i * (sw + 2), y))
        elif direction == "hr":
            n = len(cards)
            sx = x - (n * (sw + 2) - 2)
            for i, c in enumerate(cards):
                s = pygame.transform.smoothscale(self._get_card_surface(c), (sw, sh))
                self.screen.blit(s, (sx + i * (sw + 2), y))
        elif direction == "v":
            for i, c in enumerate(cards):
                s = pygame.transform.smoothscale(self._get_card_surface(c), (sw, sh))
                self.screen.blit(s, (x, y + i * (sh + 2)))

    def _render_small_card_slots(self, table_rect):
        """Render collected penalty cards at their fixed slots."""
        if not self.players:
            return
        slots = self._slot_positions(table_rect)
        for p in self.players:
            cards = [c for c in p.collected_cards
                     if c.suit == Suit.HEARTS or c == QUEEN_OF_SPADES]
            if not cards:
                continue
            cards.sort(key=lambda c: (0 if c == QUEEN_OF_SPADES else 1,
                                      -RANK_ORDER.get(c.rank, 0)))
            x, y, d = slots[p.player_id]
            self._draw_small_cards(cards, x, y, d)

    def _render_pass_cards(self, table_rect):
        """Render pass cards at the same slots as collected cards."""
        slots = self._slot_positions(table_rect)
        data = self._pass_cards_selected if self.state == "pass_show_out" else self._pass_cards_received
        for pid in range(4):
            cards = data.get(pid, [])
            if cards:
                x, y, d = slots[pid]
                self._draw_small_cards(list(cards), x, y, d)

    def _get_penalty_display(self, player_id):
        """Unused now — kept for compatibility."""
        return ""

    def _render_milestone(self, table_rect):
        """Render milestone announcement — centered box with golden border."""
        cx, cy = table_rect.centerx, table_rect.centery

        # Dark overlay.
        overlay = pygame.Surface((table_rect.width, table_rect.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (table_rect.x, table_rect.y))

        # Message — wrap to lines.
        msg = self._milestone_announcement or ""
        msg_font = self.fonts["medium"]
        words = msg.split()
        lines = []
        current_line = ""
        max_w = 420
        for word in words:
            test = current_line + " " + word if current_line else word
            if msg_font.size(test)[0] < max_w:
                current_line = test
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        # Box dimensions.
        box_w = 480
        line_h = 22
        box_h = 60 + len(lines) * line_h + 20
        box_x = cx - box_w // 2
        box_y = cy - box_h // 2

        # Box background.
        box_rect = pygame.Rect(box_x, box_y, box_w, box_h)
        pygame.draw.rect(self.screen, (20, 20, 40), box_rect, border_radius=12)
        # Golden border.
        pygame.draw.rect(self.screen, TEXT_GOLD, box_rect, width=2, border_radius=12)
        # Inner glow line.
        inner = pygame.Rect(box_x + 3, box_y + 3, box_w - 6, box_h - 6)
        pygame.draw.rect(self.screen, (60, 50, 20), inner, width=1, border_radius=10)

        # Title.
        title_font = pygame.font.SysFont("Segoe UI", 18, bold=True)
        title_surf = title_font.render("DISCOVERY", True, TEXT_GOLD)
        self.screen.blit(title_surf, title_surf.get_rect(centerx=cx, y=box_y + 14))

        # Separator line.
        sep_y = box_y + 42
        pygame.draw.line(self.screen, (80, 70, 30), (box_x + 20, sep_y), (box_x + box_w - 20, sep_y))

        # Message lines.
        for i, line in enumerate(lines):
            line_surf = msg_font.render(line, True, TEXT_WHITE)
            self.screen.blit(line_surf, line_surf.get_rect(centerx=cx, y=sep_y + 12 + i * line_h))

    def _render_continue_button(self, table_rect):
        """Render a Continue button at the bottom of the table."""
        btn_w, btn_h = 110, 28
        btn_x = table_rect.centerx - btn_w // 2
        btn_y = table_rect.bottom - btn_h - 15
        btn_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
        self._continue_btn_rect = btn_rect

        mx, my = pygame.mouse.get_pos()
        hover = btn_rect.collidepoint(mx, my)
        bg = (90, 200, 90) if hover else (70, 170, 70)
        pygame.draw.rect(self.screen, bg, btn_rect, border_radius=6)
        pygame.draw.rect(self.screen, (180, 255, 180), btn_rect, width=1, border_radius=6)

        btn_font = self.fonts["medium"]
        btn_text = btn_font.render("Continue", True, (255, 255, 255))
        self.screen.blit(btn_text, btn_text.get_rect(center=btn_rect.center))

    def _render_mode_button(self, table_rect):
        """Render the Watch/Follow mode toggle button. Shows the OTHER mode to switch to."""
        btn_w, btn_h = 110, 28
        btn_x = table_rect.right - btn_w - 10
        btn_y = table_rect.top + 10
        btn_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
        self._mode_btn_rect = btn_rect

        mx, my = pygame.mouse.get_pos()
        hover = btn_rect.collidepoint(mx, my)

        # Button shows what you'd switch TO.
        if self.mode == "follow":
            # Currently following — offer Watch Auto.
            bg = (30, 100, 180) if not hover else (50, 130, 210)
            label = self._mode_button_labels["follow"]
        else:
            # Currently watching — offer See & Check.
            bg = (140, 80, 20) if not hover else (180, 110, 40)
            label = self._mode_button_labels["watch"]

        pygame.draw.rect(self.screen, bg, btn_rect, border_radius=6)
        if hover:
            pygame.draw.rect(self.screen, (200, 200, 200), btn_rect, width=1, border_radius=6)

        btn_font = self.fonts["small"]
        btn_text = btn_font.render(label, True, (255, 255, 255))
        self.screen.blit(btn_text, btn_text.get_rect(center=btn_rect.center))

        # Show current mode indicator below button.
        mode_label = self._mode_labels[self.mode]
        mode_color = (255, 180, 80) if self.mode == "follow" else (100, 180, 255)
        mode_surf = self.fonts["small"].render(mode_label, True, mode_color)
        self.screen.blit(mode_surf, mode_surf.get_rect(centerx=btn_rect.centerx, y=btn_rect.bottom + 3))

    def _render_reset_button(self, table_rect):
        """Render a Reset Brain button at top-left of table."""
        btn_w, btn_h = 110, 28
        btn_x = table_rect.left + 10
        btn_y = table_rect.top + 10
        btn_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
        self._reset_btn_rect = btn_rect

        mx, my = pygame.mouse.get_pos()
        hover = btn_rect.collidepoint(mx, my)
        bg = (140, 40, 40) if hover else (100, 30, 30)
        pygame.draw.rect(self.screen, bg, btn_rect, border_radius=6)
        if hover:
            pygame.draw.rect(self.screen, (200, 200, 200), btn_rect, width=1, border_radius=6)

        btn_font = self.fonts["small"]
        btn_text = btn_font.render("Reset Brain", True, (255, 255, 255))
        self.screen.blit(btn_text, btn_text.get_rect(center=btn_rect.center))

    def _render_progress_bar(self, x, y, width):
        """
        Render a horizontal learning progress bar showing model maturity.
        Based on episodes_trained from the loaded model (persists across sessions).
        0 = untrained, 100% = 10,000+ episodes (mature agent).
        """
        bar_h = 14
        bar_x = x + 10
        bar_w = width - 20

        # Label.
        label = self.fonts["small"].render("AI Learning Progress", True, TEXT_LIGHT)
        self.screen.blit(label, (bar_x, y))
        y += 16

        # Background bar (0-100%).
        bg_rect = pygame.Rect(bar_x, y, bar_w, bar_h)
        pygame.draw.rect(self.screen, (30, 30, 50), bg_rect, border_radius=4)

        # Progress based on total shotas the AI has learned from.
        # Scale: 0 = 0%, 1,000,000 = 100%.
        max_episodes = 1000000
        episodes = self.discovery.episodes_trained if self.discovery else 0
        progress_pct = min(100.0, episodes / max_episodes * 100)

        # Fill bar.
        fill_w = int(bar_w * progress_pct / 100)
        if fill_w > 0:
            if progress_pct < 25:
                bar_color = (180, 50, 50)
            elif progress_pct < 50:
                bar_color = (200, 160, 40)
            elif progress_pct < 75:
                bar_color = (100, 180, 60)
            else:
                bar_color = (50, 200, 100)
            fill_rect = pygame.Rect(bar_x, y, fill_w, bar_h)
            pygame.draw.rect(self.screen, bar_color, fill_rect, border_radius=4)

        # Border.
        pygame.draw.rect(self.screen, (80, 80, 100), bg_rect, width=1, border_radius=4)

        # Text: episodes + percentage.
        pct_text = f"{progress_pct:.0f}%"
        pct_surf = self.fonts["small"].render(pct_text, True, TEXT_WHITE)
        self.screen.blit(pct_surf, (bar_x + bar_w + 4, y))

        # Detail below bar.
        detail = f"{episodes:,} shotas learned"
        if self.games_played > 0:
            session_rate = self.games_won_by_ai / self.games_played * 100
            detail += f"  |  Session: {session_rate:.0f}% ({self.games_won_by_ai}/{self.games_played})"
        detail_surf = self.fonts["small"].render(detail, True, TEXT_DIM)
        self.screen.blit(detail_surf, (bar_x, y + bar_h + 2))

        # Explanation + background training status.
        if getattr(self, '_bg_training_active', False):
            done = getattr(self, '_bg_training_done', 0)
            target = getattr(self, '_bg_training_target', 500)
            explain = f"Training in background: {done}/{target} games..."
            explain_color = TEXT_GREEN
        else:
            explain = "1 shota = 13 tricks the AI learned from"
            explain_color = (60, 80, 110)
        explain_surf = self.fonts["small"].render(explain, True, explain_color)
        self.screen.blit(explain_surf, (bar_x, y + bar_h + 16))

    def _render_side_panel(self):
        """Render the right-side panel — two separate boxes: Scoreboard + Discoveries."""
        panel_x = self._screen_w - 290
        panel_w = 280

        # === Box 1: Scoreboard (top) ===
        score_h = 185
        score_rect = pygame.Rect(panel_x, 60, panel_w, score_h)
        pygame.draw.rect(self.screen, PANEL_DARK, score_rect, border_radius=10)
        pygame.draw.rect(self.screen, (50, 45, 70), score_rect, width=1, border_radius=10)

        y = score_rect.top + 10

        title = self.fonts["large"].render("Scoreboard", True, TEXT_WHITE)
        self.screen.blit(title, (panel_x + 10, y))
        y += 22

        # Table header: Player | S1 | S2 | S3 | S4 | S5 | Total
        col_name_w = 70
        col_shota_w = 30
        col_total_w = 38
        header_x = panel_x + 10

        # Draw header row.
        self.screen.blit(self.fonts["small"].render("Player", True, TEXT_DIM), (header_x, y))
        for s in range(5):
            sx = header_x + col_name_w + s * col_shota_w
            self.screen.blit(self.fonts["small"].render(f"S{s+1}", True, TEXT_DIM), (sx, y))
        total_x = header_x + col_name_w + 5 * col_shota_w
        self.screen.blit(self.fonts["small"].render("Total", True, TEXT_DIM), (total_x, y))
        y += 16

        # Draw a thin separator line.
        pygame.draw.line(self.screen, (80, 80, 100),
                         (panel_x + 10, y), (panel_x + panel_w - 10, y))
        y += 4

        # Player rows sorted by total score.
        sorted_pids = sorted(range(4), key=lambda p: self.total_scores[p], reverse=True)
        for pid in sorted_pids:
            color = PLAYER_COLORS[pid]
            name = PLAYER_NAMES[pid][:7]  # Truncate long names.
            self.screen.blit(self.fonts["small"].render(name, True, color), (header_x, y))

            # Per-shota scores.
            for s in range(5):
                sx = header_x + col_name_w + s * col_shota_w
                if s < len(self.shota_scores):
                    val = self.shota_scores[s].get(pid, 0)
                    sc_color = (100, 255, 100) if val > 0 else (255, 100, 100) if val < 0 else TEXT_LIGHT
                    self.screen.blit(self.fonts["small"].render(f"{val:+d}", True, sc_color), (sx, y))
                else:
                    self.screen.blit(self.fonts["small"].render("—", True, TEXT_DIM), (sx + 4, y))

            # Total.
            total_val = self.total_scores[pid]
            tc = (100, 255, 100) if total_val > 0 else (255, 100, 100) if total_val < 0 else TEXT_LIGHT
            self.screen.blit(self.fonts["small"].render(f"{total_val:+d}", True, tc), (total_x, y))
            y += 18

        y += 6
        # Learning progress bar inside scoreboard box.
        self._render_progress_bar(panel_x, y, 260)

        # === Box 2: Discoveries (bottom) ===
        disc_top = 60 + score_h + 8
        disc_h = self._screen_h - 80 - score_h - 8
        disc_rect = pygame.Rect(panel_x, disc_top, panel_w, disc_h)
        pygame.draw.rect(self.screen, PANEL_DARK, disc_rect, border_radius=10)
        pygame.draw.rect(self.screen, (70, 60, 20), disc_rect, width=1, border_radius=10)

        # Clip rendering to inside the box.
        self.screen.set_clip(pygame.Rect(panel_x + 5, disc_top + 5, panel_w - 10, disc_h - 10))

        y = disc_rect.top + 10
        disc_title = self.fonts["large"].render("Discoveries", True, TEXT_GOLD)
        self.screen.blit(disc_title, (panel_x + 10, y))
        y += 22

        if not self._milestones_list:
            self.screen.blit(self.fonts["medium"].render("None yet...", True, TEXT_DIM), (panel_x + 15, y))
            self.screen.set_clip(None)
        else:
            disc_scroll = getattr(self, '_disc_scroll_offset', 0)
            available_h = disc_rect.bottom - y - 5
            line_h = 30
            max_visible = available_h // line_h
            total = len(self._milestones_list)

            max_scroll = max(0, total - max_visible)
            disc_scroll = max(0, min(disc_scroll, max_scroll))
            self._disc_scroll_offset = disc_scroll

            if disc_scroll == 0:
                visible = self._milestones_list[-max_visible:]
                start_num = max(1, total - max_visible + 1)
            else:
                end_idx = total - disc_scroll
                start_idx = max(0, end_idx - max_visible)
                visible = self._milestones_list[start_idx:end_idx]
                start_num = start_idx + 1

            panel_text_w = panel_w - 30
            for i, (title_text, desc_text) in enumerate(visible):
                num = start_num + i
                is_latest = (disc_scroll == 0 and i == len(visible) - 1)

                # Don't render title if there's no room for at least one desc line too.
                if y + 18 + 15 > disc_rect.bottom - 15:
                    break

                title_color = (100, 255, 100) if is_latest else (255, 255, 255)
                title_line = f"{num}. {title_text}"
                title_surf = self.fonts["large"].render(title_line, True, title_color)
                self.screen.blit(title_surf, (panel_x + 10, y))
                y += 18

                desc_color = (200, 255, 200) if is_latest else (210, 210, 210)
                desc_font = self.fonts["medium"]
                words = desc_text.split()
                line = ""
                for w in words:
                    test = line + " " + w if line else w
                    if desc_font.size(test)[0] < panel_text_w:
                        line = test
                    else:
                        if y > disc_rect.bottom - 20:
                            break
                        self.screen.blit(desc_font.render(line, True, desc_color), (panel_x + 20, y))
                        y += 15
                        line = w
                if line and y <= disc_rect.bottom - 20:
                    self.screen.blit(desc_font.render(line, True, desc_color), (panel_x + 20, y))
                    y += 17

                if y > disc_rect.bottom - 15:
                    break

        # Reset clip.
        self.screen.set_clip(None)


def main():
    parser = argparse.ArgumentParser(description="Sudanese Hearts — Visual Watcher")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model")
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        # Try default location.
        default = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agents", "hearts_discovery", "hearts_model.json"
        )
        if os.path.exists(default):
            model_path = default

    app = HeartsWatcher(model_path=model_path)
    app.run()


if __name__ == "__main__":
    main()
