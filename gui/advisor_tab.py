"""
AI Advisor tab — Play alongside a real physical game.

You select the 13 cards in your hand, then as the game progresses
you tell the app what others played. The AI recommends what you
should play and you compare with your own judgment.
"""

import tkinter as tk
from tkinter import ttk

from gui.colors import COLORS
from intelligence.core.cards.card import Card
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit


ALL_SUITS = [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]
ALL_RANKS = list(Rank)

SUIT_SYMBOLS = {Suit.SPADES: "♠", Suit.HEARTS: "♥",
                Suit.CLUBS: "♣", Suit.DIAMONDS: "♦"}
RANK_SYMBOLS = {Rank.ACE: "A", Rank.KING: "K", Rank.QUEEN: "Q",
                Rank.JACK: "J", Rank.TEN: "10", Rank.NINE: "9",
                Rank.EIGHT: "8", Rank.SEVEN: "7", Rank.SIX: "6",
                Rank.FIVE: "5", Rank.FOUR: "4", Rank.THREE: "3",
                Rank.TWO: "2"}


class AdvisorTab:
    """Builds and manages the AI Advisor tab."""

    def __init__(self, parent: tk.Frame, root: tk.Tk) -> None:
        self.parent = parent
        self.root = root

        self.my_hand: list[Card] = []
        self.trick_cards: list[tuple[int, Card]] = []  # (player_id, card)
        self.trump_suit: Suit | None = None

        self._build()

    def _build(self) -> None:
        self.parent.configure(bg="#1a1a1a")

        # Top: instructions.
        top = tk.Frame(self.parent, bg="#252525", padx=12, pady=8)
        top.pack(fill="x", padx=12, pady=(12, 6))

        tk.Label(top, text="AI Advisor — Play Along a Real Game",
                 font=("Segoe UI", 12, "bold"), fg="#ffffff", bg="#252525"
                 ).pack(anchor="w")
        tk.Label(top, text="1. Select your 13 cards  2. Set trump  3. As each trick happens, pick what others played  4. Get AI recommendation",
                 font=("Segoe UI", 9), fg="#aaaaaa", bg="#252525"
                 ).pack(anchor="w", pady=(4, 0))

        # Main area: two columns.
        main = tk.Frame(self.parent, bg="#1a1a1a")
        main.pack(fill="both", expand=True, padx=12, pady=6)
        main.columnconfigure(0, weight=2)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # Left: card picker (full deck).
        left = tk.Frame(main, bg="#252525", bd=1, relief="groove", padx=8, pady=8)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        tk.Label(left, text="SELECT YOUR HAND (click cards to add/remove)",
                 font=("Segoe UI", 9, "bold"), fg="#888888", bg="#252525"
                 ).pack(anchor="w", pady=(0, 6))

        self._deck_frame = tk.Frame(left, bg="#252525")
        self._deck_frame.pack(fill="both", expand=True)

        self._card_buttons: dict[Card, tk.Button] = {}
        self._build_deck_grid()

        # Hand count label.
        self._hand_count_label = tk.Label(
            left, text="Hand: 0/13 cards selected",
            font=("Segoe UI", 9), fg=COLORS["gold"], bg="#252525")
        self._hand_count_label.pack(anchor="w", pady=(6, 0))

        # Right: controls + recommendation.
        right = tk.Frame(main, bg="#252525", bd=1, relief="groove", padx=12, pady=8)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        # Trump selector.
        trump_frame = tk.Frame(right, bg="#252525")
        trump_frame.pack(fill="x", pady=(0, 8))

        tk.Label(trump_frame, text="Trump:", font=("Segoe UI", 10),
                 fg="#aaaaaa", bg="#252525").pack(side="left")

        self._trump_var = tk.StringVar(value="—")
        for suit in ALL_SUITS:
            sym = SUIT_SYMBOLS[suit]
            fg = "#c62828" if suit in (Suit.HEARTS, Suit.DIAMONDS) else "#333333"
            tk.Button(trump_frame, text=sym, font=("Consolas", 14, "bold"),
                      fg=fg, bg=COLORS["card_bg"], bd=1, padx=6, pady=2,
                      cursor="hand2",
                      command=lambda s=suit: self._set_trump(s)
                      ).pack(side="left", padx=4)

        self._trump_label = tk.Label(right, text="Trump: not set",
                                     font=("Segoe UI", 9), fg=COLORS["gold"], bg="#252525")
        self._trump_label.pack(anchor="w")

        # Current trick input.
        tk.Label(right, text="CURRENT TRICK", font=("Segoe UI", 9, "bold"),
                 fg="#888888", bg="#252525").pack(anchor="w", pady=(12, 4))

        self._trick_display = tk.Label(
            right, text="No cards played yet",
            font=("Consolas", 10), fg="#cccccc", bg="#1a1a1a",
            justify="left", anchor="w", padx=8, pady=6, width=25)
        self._trick_display.pack(fill="x", pady=4)

        tk.Button(right, text="Clear Trick", command=self._clear_trick,
                  font=("Segoe UI", 8), fg="#fff", bg=COLORS["btn_grey"],
                  bd=0, padx=8, pady=2).pack(anchor="w", pady=4)

        # AI recommendation.
        tk.Label(right, text="AI RECOMMENDS", font=("Segoe UI", 9, "bold"),
                 fg="#888888", bg="#252525").pack(anchor="w", pady=(12, 4))

        self._recommendation_label = tk.Label(
            right, text="Set your hand and trump first",
            font=("Segoe UI", 12, "bold"), fg=COLORS["gold"], bg="#1a1a1a",
            padx=12, pady=10, anchor="center")
        self._recommendation_label.pack(fill="x", pady=4)

        tk.Button(right, text="🤖 Get Recommendation", command=self._get_recommendation,
                  font=("Segoe UI", 10, "bold"), fg="#fff", bg=COLORS["btn_green"],
                  bd=0, padx=12, pady=6, cursor="hand2").pack(fill="x", pady=(8, 4))

        tk.Button(right, text="↺ Reset All", command=self._reset_all,
                  font=("Segoe UI", 9), fg="#fff", bg=COLORS["btn_grey"],
                  bd=0, padx=8, pady=4).pack(anchor="w", pady=(8, 0))

    def _build_deck_grid(self) -> None:
        """Build a grid of all 52 cards as clickable buttons."""
        for suit in ALL_SUITS:
            row = tk.Frame(self._deck_frame, bg="#252525")
            row.pack(fill="x", pady=2)

            fg = "#c62828" if suit in (Suit.HEARTS, Suit.DIAMONDS) else "#303030"

            for rank in reversed(ALL_RANKS):
                card = Card(suit, rank)
                text = f"{RANK_SYMBOLS[rank]}{SUIT_SYMBOLS[suit]}"

                btn = tk.Button(
                    row, text=text, font=("Consolas", 9, "bold"),
                    fg=fg, bg=COLORS["card_bg"], relief="solid", bd=1,
                    padx=2, pady=1, width=3, cursor="hand2",
                    command=lambda c=card: self._toggle_card(c))
                btn.pack(side="left", padx=1)
                self._card_buttons[card] = btn

    def _toggle_card(self, card: Card) -> None:
        """Add or remove a card from the hand."""
        btn = self._card_buttons[card]

        if card in self.my_hand:
            self.my_hand.remove(card)
            fg = "#c62828" if card.suit in (Suit.HEARTS, Suit.DIAMONDS) else "#303030"
            btn.config(bg=COLORS["card_bg"], fg=fg, relief="solid")
        else:
            if len(self.my_hand) >= 13:
                return  # Can't add more than 13.
            self.my_hand.append(card)
            btn.config(bg="#4caf50", fg="#ffffff", relief="raised")

        self._hand_count_label.config(text=f"Hand: {len(self.my_hand)}/13 cards selected")

    def _set_trump(self, suit: Suit) -> None:
        self.trump_suit = suit
        self._trump_label.config(text=f"Trump: {suit.name} {SUIT_SYMBOLS[suit]}")

    def _clear_trick(self) -> None:
        self.trick_cards.clear()
        self._trick_display.config(text="No cards played yet")

    def _get_recommendation(self) -> None:
        """Ask the AI what card to play."""
        if len(self.my_hand) == 0:
            self._recommendation_label.config(text="Select your hand first!")
            return

        if self.trump_suit is None:
            self._recommendation_label.config(text="Set trump suit first!")
            return

        # Use the rule-based agent to get a recommendation.
        from agents.rule_based.rule_based_agent import RuleBasedAgent
        from environments.wist.observation import WistObservation
        from environments.wist.trick import Trick
        from environments.wist.actions import PlayCardAction

        # Build a trick from what's been played.
        trick = None
        if self.trick_cards:
            trick = Trick(leading_player_id=self.trick_cards[0][0])
            for pid, card in self.trick_cards:
                trick.play_card(pid, card)

        obs = WistObservation(
            player_id=2,  # Player 3 = index 2.
            hand=list(self.my_hand),
            current_trick=trick,
            trump_suit=self.trump_suit,
        )

        agent = RuleBasedAgent()
        action = agent.act(obs)

        if isinstance(action, PlayCardAction):
            card = action.card
            sym = f"{RANK_SYMBOLS[card.rank]}{SUIT_SYMBOLS[card.suit]}"
            self._recommendation_label.config(text=f"Play: {sym}")
        else:
            self._recommendation_label.config(text="No recommendation available")

    def _reset_all(self) -> None:
        """Reset everything."""
        self.my_hand.clear()
        self.trick_cards.clear()
        self.trump_suit = None

        for card, btn in self._card_buttons.items():
            fg = "#c62828" if card.suit in (Suit.HEARTS, Suit.DIAMONDS) else "#303030"
            btn.config(bg=COLORS["card_bg"], fg=fg, relief="solid")

        self._hand_count_label.config(text="Hand: 0/13 cards selected")
        self._trump_label.config(text="Trump: not set")
        self._trick_display.config(text="No cards played yet")
        self._recommendation_label.config(text="Set your hand and trump first")
