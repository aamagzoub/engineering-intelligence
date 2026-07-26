"""
Human vs AI tab — Player 3 (index 2) is a human who clicks cards to play.

Uses a coroutine-like approach: the game loop runs step by step
via root.after(), pausing whenever it's the human's turn and
waiting for a click before continuing.
"""

import tkinter as tk
from collections import Counter

from gui.colors import COLORS
from agents.rule_based.rule_based_agent import RuleBasedAgent
from environments.wist.environment import WistEnvironment
from environments.wist.actions import PlayCardAction, BidAction, PassAction
from environments.wist.observation import WistObservation, BiddingObservation
from environments.wist.round import Round
from environments.wist.round_state import RoundState
from environments.wist.rules import legal_cards, trick_winner, rank_value
from environments.wist.setup import create_standard_players
from environments.wist.tasmiya_engine import TasmiyaEngine, determine_trump_suit, tasmiya_order
from environments.wist.trick import Trick
from intelligence.core.cards.card import Card
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit


SUIT_SYMBOLS = {Suit.SPADES: "♠", Suit.HEARTS: "♥", Suit.CLUBS: "♣", Suit.DIAMONDS: "♦"}
RANK_SYMBOLS = {r: s for r, s in zip(Rank, ["2","3","4","5","6","7","8","9","10","J","Q","K","A"])}

HUMAN_PLAYER_ID = 2  # Player 3 = index 2 (Team 1).


def card_text(card: Card) -> str:
    return f"{RANK_SYMBOLS[card.rank]}{SUIT_SYMBOLS[card.suit]}"


class HumanTab:
    """Human vs AI interactive game tab."""

    def __init__(self, parent: tk.Frame, root: tk.Tk) -> None:
        self.parent = parent
        self.root = root
        self.game_running = False

        # Game state.
        self.players = None
        self.round = None
        self.environment = None
        self.agents = None  # AI agents (human slot is None).
        self.trump_suit = None
        self.trick_number = 0
        self.team_tricks = [0, 0]

        self._build()

    def _build(self) -> None:
        self.parent.configure(bg=COLORS["table_felt"])

        # Top info bar.
        info = tk.Frame(self.parent, bg=COLORS["header_bg"], height=40)
        info.pack(fill="x")
        info.pack_propagate(False)

        self._status = tk.Label(info, text="Press Start Game to play as Player 3",
                                font=("Segoe UI", 10, "bold"),
                                fg=COLORS["text_light"], bg=COLORS["header_bg"])
        self._status.pack(pady=8)

        # Game info row.
        info2 = tk.Frame(self.parent, bg=COLORS["header_bg"], height=30)
        info2.pack(fill="x")
        info2.pack_propagate(False)

        self._info = tk.Label(info2, text="",
                              font=("Consolas", 9), fg=COLORS["gold"],
                              bg=COLORS["header_bg"])
        self._info.pack(pady=4)

        # Centre trick area.
        centre = tk.Frame(self.parent, bg=COLORS["centre_bg"], height=140)
        centre.pack(fill="x", padx=20, pady=8)
        centre.pack_propagate(False)

        self._trick_label = tk.Label(centre, text="",
                                     font=("Consolas", 12, "bold"),
                                     fg=COLORS["text_white"], bg=COLORS["centre_bg"],
                                     justify="center")
        self._trick_label.pack(expand=True)

        # Your hand.
        hand_outer = tk.LabelFrame(self.parent, text=" Your Hand (Player 3 — Team 1) ",
                                    font=("Segoe UI", 10, "bold"),
                                    fg=COLORS["gold"], bg=COLORS["player_bg"],
                                    padx=10, pady=8)
        hand_outer.pack(fill="x", padx=20, pady=8)

        self._hand_frame = tk.Frame(hand_outer, bg=COLORS["player_bg"])
        self._hand_frame.pack(fill="x")

        # Bid frame (shown during bidding).
        self._bid_frame = tk.Frame(hand_outer, bg=COLORS["player_bg"])

        # Controls.
        ctrl = tk.Frame(self.parent, bg=COLORS["header_bg"], height=44)
        ctrl.pack(fill="x")
        ctrl.pack_propagate(False)

        btn_f = tk.Frame(ctrl, bg=COLORS["header_bg"])
        btn_f.pack(anchor="center", pady=8)

        tk.Button(btn_f, text="▶ Start Game", command=self._start_game,
                  font=("Segoe UI", 9, "bold"), fg="#fff", bg=COLORS["btn_green"],
                  bd=0, padx=16, pady=4, cursor="hand2").pack(side="left", padx=4)

        tk.Button(btn_f, text="⏹ Stop", command=self._stop_game,
                  font=("Segoe UI", 9, "bold"), fg="#fff", bg=COLORS["btn_red"],
                  bd=0, padx=16, pady=4, cursor="hand2").pack(side="left", padx=4)

    # ----------------------------------------------------------
    # Game flow
    # ----------------------------------------------------------

    def _start_game(self) -> None:
        if self.game_running:
            return

        self.game_running = True
        self.trick_number = 0
        self.team_tricks = [0, 0]

        self.players = create_standard_players()
        self.round = Round(self.players)
        self.round.deal()

        # Skip Dak hands.
        attempts = 0
        while self.round.has_card_based_dak() and attempts < 10:
            self.round = Round(self.players)
            self.round.deal()
            attempts += 1

        # AI agents for positions 0, 1, 3. Human at position 2.
        self.agents = [RuleBasedAgent(), RuleBasedAgent(), None, RuleBasedAgent()]

        self._status.config(text="Bidding phase...")
        self._run_bidding()

    def _stop_game(self) -> None:
        self.game_running = False
        self._status.config(text="Game stopped. Press Start to play again.")
        self._trick_label.config(text="")
        self._info.config(text="")
        for w in self._hand_frame.winfo_children():
            w.destroy()

    def _run_bidding(self) -> None:
        """Run bidding — AI bids, human gets to choose."""
        # For simplicity, run the AI bidding engine and let human just see the result.
        # Human is player 2 — we'll ask them for bid/pass.
        engine = TasmiyaEngine()

        # Use a temporary agent for the human's slot during tasmiya.
        temp_agents = [RuleBasedAgent(), RuleBasedAgent(), RuleBasedAgent(), RuleBasedAgent()]
        result = engine.run(players=self.players, agents=temp_agents, sahib_al_qabool_id=0)

        if result.is_dak:
            self._status.config(text="Dak! Re-dealing...")
            self.root.after(1000, self._start_game)
            return

        self.trump_suit = result.trump_suit
        self.round.state.trump_suit = result.trump_suit
        self.round.state.winning_bidder_id = result.winning_bidder_id
        self.round.next_leading_player_id = result.winning_bidder_id

        self.environment = WistEnvironment(self.round.state)

        trump_sym = SUIT_SYMBOLS.get(self.trump_suit, "?")
        self._info.config(
            text=f"Bid: {result.winning_bid_value} | "
                 f"Trump: {self.trump_suit.name} {trump_sym} | "
                 f"Shooter: Player {result.winning_bidder_id + 1}"
        )
        self._status.config(text="Playing! Watch the tricks and click your card when it's your turn.")

        # Show hand and start playing.
        self._show_hand()
        self.root.after(500, self._play_next_trick)

    def _play_next_trick(self) -> None:
        """Start the next trick."""
        if not self.game_running:
            return

        if self.trick_number >= 13:
            self._finish_game()
            return

        self.trick_number += 1
        leader_id = self.round.next_leading_player_id
        self.round.state.current_trick = Trick(leading_player_id=leader_id)

        self._play_order = [(leader_id + i) % 4 for i in range(4)]
        self._play_index = 0
        self._trick_cards_display = []

        self._trick_label.config(text=f"Trick {self.trick_number} — Player {leader_id + 1} leads")
        self._play_next_card()

    def _play_next_card(self) -> None:
        """Play the next card in the trick (AI or wait for human)."""
        if not self.game_running:
            return

        if self._play_index >= 4:
            # Trick complete.
            self.root.after(800, self._resolve_trick)
            return

        player_id = self._play_order[self._play_index]

        if player_id == HUMAN_PLAYER_ID:
            # Human's turn — show clickable legal cards.
            self._status.config(text=f"Trick {self.trick_number} — YOUR TURN! Click a card to play.")
            self._show_hand(clickable=True)
        else:
            # AI plays.
            obs = self.environment.observe(player_id)
            action = self.agents[player_id].act(obs)
            self.environment.apply_action(action)
            self._play_index += 1

            ct = card_text(action.card)
            self._trick_cards_display.append(f"P{player_id + 1}: {ct}")
            self._update_trick_display()

            self._show_hand()
            self.root.after(600, self._play_next_card)

    def _human_play_card(self, card: Card) -> None:
        """Called when human clicks a card."""
        if not self.game_running:
            return

        action = PlayCardAction(player_id=HUMAN_PLAYER_ID, card=card)
        self.environment.apply_action(action)
        self._play_index += 1

        ct = card_text(card)
        self._trick_cards_display.append(f"YOU: {ct}")
        self._update_trick_display()

        self._status.config(text=f"Trick {self.trick_number} — you played {ct}")
        self._show_hand()
        self.root.after(600, self._play_next_card)

    def _resolve_trick(self) -> None:
        """Determine trick winner and move on."""
        if not self.game_running:
            return

        trick = self.round.state.current_trick
        winner = trick_winner(trick, self.trump_suit)

        self.round.state.completed_tricks.append(trick)
        self.round.state.current_trick = None
        self.round.next_leading_player_id = winner

        winner_team = 0 if winner in (0, 2) else 1
        self.team_tricks[winner_team] += 1

        who = "YOU" if winner == HUMAN_PLAYER_ID else f"Player {winner + 1}"
        self._trick_label.config(
            text=f"Trick {self.trick_number} — {who} won!\n"
                 f"Team 1: {self.team_tricks[0]} | Team 2: {self.team_tricks[1]}"
        )

        self.root.after(1200, self._play_next_trick)

    def _finish_game(self) -> None:
        """Game over."""
        self.game_running = False
        winner = "Team 1 (yours)" if self.team_tricks[0] > self.team_tricks[1] else "Team 2"
        self._status.config(text="Game Over!")
        self._trick_label.config(
            text=f"🏆 GAME OVER 🏆\n\n"
                 f"{winner} wins!\n"
                 f"Team 1: {self.team_tricks[0]} | Team 2: {self.team_tricks[1]}"
        )

    # ----------------------------------------------------------
    # Hand display
    # ----------------------------------------------------------

    def _show_hand(self, clickable: bool = False) -> None:
        """Show the human player's current hand."""
        for w in self._hand_frame.winfo_children():
            w.destroy()

        hand = self.players[HUMAN_PLAYER_ID].hand
        if not hand:
            tk.Label(self._hand_frame, text="No cards left",
                     font=("Segoe UI", 9), fg=COLORS["text_dim"],
                     bg=COLORS["player_bg"]).pack()
            return

        # Get legal cards if clickable.
        legal = hand
        if clickable and self.round.state.current_trick:
            leading_suit = self.round.state.current_trick.leading_suit
            must_trump = None
            if (self.round.state.is_first_trick and
                    self.round.state.winning_bidder_id == HUMAN_PLAYER_ID and
                    len(self.round.state.current_trick.played_cards) == 0):
                must_trump = self.trump_suit
            legal = legal_cards(hand, leading_suit, must_trump)

        suit_order = [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]

        for suit in suit_order:
            suit_cards = sorted([c for c in hand if c.suit == suit],
                                key=lambda c: -rank_value(c.rank))
            if not suit_cards:
                continue

            row = tk.Frame(self._hand_frame, bg=COLORS["player_bg"])
            row.pack(fill="x", pady=2)

            fg = "#c62828" if suit in (Suit.HEARTS, Suit.DIAMONDS) else "#303030"

            for card in suit_cards:
                ct = card_text(card)
                is_legal = card in legal

                if clickable and is_legal:
                    btn = tk.Button(
                        row, text=ct, font=("Consolas", 11, "bold"),
                        fg=fg, bg=COLORS["card_bg"], relief="raised", bd=2,
                        padx=4, pady=2, width=3, cursor="hand2",
                        command=lambda c=card: self._human_play_card(c))
                    btn.pack(side="left", padx=2)
                else:
                    lbl = tk.Label(
                        row, text=ct, font=("Consolas", 10),
                        fg="#999999" if (clickable and not is_legal) else fg,
                        bg=COLORS["card_bg"] if not clickable else "#d0d0d0",
                        relief="solid", bd=1, padx=3, pady=1, width=3)
                    lbl.pack(side="left", padx=2)

    def _update_trick_display(self) -> None:
        """Update the centre trick area with cards played so far."""
        text = f"Trick {self.trick_number}\n" + "  |  ".join(self._trick_cards_display)
        self._trick_label.config(text=text)

    def set_status(self, text): pass
    def set_trick_display(self, text): pass
    def show_hand(self, *a, **kw): pass
    def show_bid_options(self, *a, **kw): pass
    def hide_bid_options(self): pass
