"""
AI Advisor tab — Play alongside a real physical game.

Flow:
1. Select your 13 cards from the deck
2. Set trump suit
3. For each trick: click cards that OTHER players played (in order)
4. Click "Get Recommendation" to see what the AI would play
5. Click the card YOU played to remove it from your hand
6. Start next trick

Supports both Rule-Based and Learning (loaded model) agents.
"""

import tkinter as tk
from tkinter import filedialog

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

def card_str(card: Card) -> str:
    return f"{RANK_SYMBOLS[card.rank]}{SUIT_SYMBOLS[card.suit]}"


class AdvisorTab:
    """AI Advisor — play alongside a real game."""

    def __init__(self, parent: tk.Frame, root: tk.Tk) -> None:
        self.parent = parent
        self.root = root

        self.my_hand: list[Card] = []
        self.trick_cards: list[tuple[int, Card]] = []  # (player_id, card)
        self.trump_suit: Suit | None = None
        self.trick_number = 0

        # Mode: "hand" = selecting hand, "trick" = marking trick cards.
        self.mode = "hand"

        # AI agent for recommendations.
        self._agent = None  # Will be created on first use.
        self._agent_type = "rule_based"

        self._build()

    def _build(self) -> None:
        self.parent.configure(bg="#1a1a1a")

        # Top: instructions + controls.
        top = tk.Frame(self.parent, bg="#252525", padx=12, pady=8)
        top.pack(fill="x", padx=12, pady=(8, 4))

        tk.Label(top, text="AI Advisor — Play Along a Real Game",
                 font=("Segoe UI", 12, "bold"), fg="#ffffff", bg="#252525"
                 ).pack(anchor="w")

        self._instruction_label = tk.Label(
            top, text="Step 1: Click cards to select your 13-card hand",
            font=("Segoe UI", 9), fg="#aaaaaa", bg="#252525")
        self._instruction_label.pack(anchor="w", pady=(4, 0))

        # Controls row.
        ctrl = tk.Frame(top, bg="#252525")
        ctrl.pack(fill="x", pady=(8, 0))

        # Trump selector.
        tk.Label(ctrl, text="Trump:", font=("Segoe UI", 9),
                 fg="#aaaaaa", bg="#252525").pack(side="left", padx=(0, 4))

        for suit in ALL_SUITS:
            sym = SUIT_SYMBOLS[suit]
            fg = "#c62828" if suit in (Suit.HEARTS, Suit.DIAMONDS) else "#333333"
            tk.Button(ctrl, text=sym, font=("Consolas", 12, "bold"),
                      fg=fg, bg=COLORS["card_bg"], bd=1, padx=4, pady=1,
                      cursor="hand2",
                      command=lambda s=suit: self._set_trump(s)
                      ).pack(side="left", padx=2)

        self._trump_label = tk.Label(ctrl, text="", font=("Segoe UI", 9, "bold"),
                                     fg=COLORS["gold"], bg="#252525")
        self._trump_label.pack(side="left", padx=8)

        # Agent selector.
        tk.Label(ctrl, text="│  Agent:", font=("Segoe UI", 9),
                 fg="#666666", bg="#252525").pack(side="left", padx=(12, 4))

        self._agent_var = tk.StringVar(value="Rule-Based")
        tk.OptionMenu(ctrl, self._agent_var, "Rule-Based", "Learning (load model)"
                      ).pack(side="left", padx=2)

        tk.Button(ctrl, text="📂 Load Model", command=self._load_model,
                  font=("Segoe UI", 8), fg="#fff", bg="#1e88e5",
                  bd=0, padx=8, pady=2, cursor="hand2").pack(side="left", padx=4)

        # Main area.
        main = tk.Frame(self.parent, bg="#1a1a1a")
        main.pack(fill="both", expand=True, padx=12, pady=4)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        # Left: deck grid.
        left = tk.Frame(main, bg="#252525", bd=1, relief="groove", padx=8, pady=6)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        self._mode_label = tk.Label(left, text="YOUR HAND — Click cards to add (0/13)",
                                    font=("Segoe UI", 9, "bold"),
                                    fg=COLORS["gold"], bg="#252525")
        self._mode_label.pack(anchor="w", pady=(0, 4))

        self._deck_frame = tk.Frame(left, bg="#252525")
        self._deck_frame.pack(fill="both", expand=True)

        self._card_buttons: dict[Card, tk.Button] = {}
        self._build_deck_grid()

        # Right: trick + recommendation.
        right = tk.Frame(main, bg="#252525", bd=1, relief="groove", padx=12, pady=8)
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        # Trick area.
        tk.Label(right, text="CURRENT TRICK", font=("Segoe UI", 10, "bold"),
                 fg="#ffffff", bg="#252525").pack(anchor="w", pady=(0, 6))

        self._trick_display = tk.Label(
            right, text="Select hand + trump first, then start trick",
            font=("Consolas", 10), fg="#cccccc", bg="#1a1a1a",
            justify="left", anchor="w", padx=10, pady=8, width=28, height=5)
        self._trick_display.pack(fill="x", pady=4)

        # Trick control buttons.
        trick_ctrl = tk.Frame(right, bg="#252525")
        trick_ctrl.pack(fill="x", pady=(4, 8))

        tk.Button(trick_ctrl, text="▶ Start Trick", command=self._start_trick_mode,
                  font=("Segoe UI", 9, "bold"), fg="#fff", bg=COLORS["btn_green"],
                  bd=0, padx=10, pady=3, cursor="hand2").pack(side="left", padx=3)

        tk.Button(trick_ctrl, text="✗ Clear Trick", command=self._clear_trick,
                  font=("Segoe UI", 9), fg="#fff", bg=COLORS["btn_grey"],
                  bd=0, padx=8, pady=3, cursor="hand2").pack(side="left", padx=3)

        tk.Button(trick_ctrl, text="✓ I Played", command=self._i_played_card,
                  font=("Segoe UI", 9, "bold"), fg="#fff", bg=COLORS["btn_orange"],
                  bd=0, padx=10, pady=3, cursor="hand2").pack(side="left", padx=3)

        # Recommendation.
        tk.Label(right, text="AI RECOMMENDS", font=("Segoe UI", 10, "bold"),
                 fg="#ffffff", bg="#252525").pack(anchor="w", pady=(12, 4))

        self._rec_label = tk.Label(
            right, text="—",
            font=("Segoe UI", 14, "bold"), fg=COLORS["gold"], bg="#1a1a1a",
            padx=12, pady=12, anchor="center")
        self._rec_label.pack(fill="x", pady=4)

        tk.Button(right, text="🤖 Get Recommendation", command=self._get_recommendation,
                  font=("Segoe UI", 10, "bold"), fg="#fff", bg=COLORS["btn_green"],
                  bd=0, padx=12, pady=6, cursor="hand2").pack(fill="x", pady=(8, 4))

        # Info.
        self._info_label = tk.Label(right, text="Trick: 0 | Hand: 0 cards",
                                    font=("Segoe UI", 8), fg="#888888", bg="#252525")
        self._info_label.pack(anchor="w", pady=(8, 0))

        tk.Button(right, text="↺ Reset All", command=self._reset_all,
                  font=("Segoe UI", 9), fg="#fff", bg=COLORS["btn_grey"],
                  bd=0, padx=8, pady=4).pack(anchor="w", pady=(8, 0))

    def _build_deck_grid(self) -> None:
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
                    command=lambda c=card: self._card_clicked(c))
                btn.pack(side="left", padx=1)
                self._card_buttons[card] = btn

    # ----------------------------------------------------------
    # Card click handling (mode-dependent)
    # ----------------------------------------------------------

    def _card_clicked(self, card: Card) -> None:
        if self.mode == "hand":
            self._toggle_hand_card(card)
        elif self.mode == "trick":
            self._add_trick_card(card)

    def _toggle_hand_card(self, card: Card) -> None:
        btn = self._card_buttons[card]
        if card in self.my_hand:
            self.my_hand.remove(card)
            fg = "#c62828" if card.suit in (Suit.HEARTS, Suit.DIAMONDS) else "#303030"
            btn.config(bg=COLORS["card_bg"], fg=fg, relief="solid")
        else:
            if len(self.my_hand) >= 13:
                return
            self.my_hand.append(card)
            btn.config(bg="#4caf50", fg="#ffffff", relief="raised")

        self._mode_label.config(text=f"YOUR HAND — Click cards to add ({len(self.my_hand)}/13)")
        self._update_info()

    def _add_trick_card(self, card: Card) -> None:
        """Add a card that another player played in the trick."""
        if len(self.trick_cards) >= 3:
            return  # Max 3 other players' cards.

        # Mark as played by opponent (player IDs: 0, 1, 3 for others).
        opponent_ids = [0, 1, 3]
        pid = opponent_ids[len(self.trick_cards)]

        self.trick_cards.append((pid, card))

        # Mark button as "used in trick" (blue).
        btn = self._card_buttons[card]
        btn.config(bg="#1e88e5", fg="#ffffff", relief="sunken")

        self._update_trick_display()

    # ----------------------------------------------------------
    # Mode switching
    # ----------------------------------------------------------

    def _start_trick_mode(self) -> None:
        """Switch to trick mode — clicks add cards to the trick."""
        if len(self.my_hand) == 0:
            self._instruction_label.config(text="⚠ Select your hand first!")
            return
        if self.trump_suit is None:
            self._instruction_label.config(text="⚠ Set trump first!")
            return

        self.mode = "trick"
        self.trick_cards.clear()
        self.trick_number += 1

        self._mode_label.config(text="TRICK MODE — Click cards that others played (up to 3)")
        self._instruction_label.config(
            text=f"Trick {self.trick_number}: Click the cards other players played, then Get Recommendation")
        self._update_trick_display()

    def _clear_trick(self) -> None:
        """Clear current trick cards and return buttons to normal."""
        for pid, card in self.trick_cards:
            btn = self._card_buttons[card]
            if card in self.my_hand:
                btn.config(bg="#4caf50", fg="#ffffff", relief="raised")
            else:
                fg = "#c62828" if card.suit in (Suit.HEARTS, Suit.DIAMONDS) else "#303030"
                btn.config(bg=COLORS["card_bg"], fg=fg, relief="solid")
        self.trick_cards.clear()
        self._update_trick_display()
        self._rec_label.config(text="—")

    def _i_played_card(self) -> None:
        """After you played a card, remove it from your hand.
        Click a card from your hand (green) to mark it as played."""
        self.mode = "played"
        self._mode_label.config(text="PLAYED — Click the card YOU played to remove it")
        self._instruction_label.config(text="Click the card you played from your hand")

        # Temporarily rebind hand cards to remove them.
        for card in list(self.my_hand):
            btn = self._card_buttons[card]
            btn.config(command=lambda c=card: self._remove_played_card(c))

    def _remove_played_card(self, card: Card) -> None:
        """Remove a card from hand (you played it)."""
        if card in self.my_hand:
            self.my_hand.remove(card)
            btn = self._card_buttons[card]
            btn.config(bg="#666666", fg="#999999", relief="flat",
                       command=lambda c=card: self._card_clicked(c))

        # Clear trick state for next trick.
        self._clear_trick()
        self.mode = "hand"
        self._mode_label.config(text=f"YOUR HAND — {len(self.my_hand)} cards remaining")
        self._instruction_label.config(text="Click 'Start Trick' for the next trick")
        self._update_info()

        # Rebind all buttons.
        for c in self.my_hand:
            self._card_buttons[c].config(command=lambda card=c: self._card_clicked(card))

    # ----------------------------------------------------------
    # Recommendation
    # ----------------------------------------------------------

    def _get_recommendation(self) -> None:
        if len(self.my_hand) == 0:
            self._rec_label.config(text="Select hand first!")
            return
        if self.trump_suit is None:
            self._rec_label.config(text="Set trump first!")
            return

        from environments.wist.observation import WistObservation
        from environments.wist.trick import Trick
        from environments.wist.actions import PlayCardAction

        # Build trick from recorded cards.
        trick = None
        if self.trick_cards:
            trick = Trick(leading_player_id=self.trick_cards[0][0])
            for pid, card in self.trick_cards:
                trick.play_card(pid, card)

        obs = WistObservation(
            player_id=2,
            hand=list(self.my_hand),
            current_trick=trick,
            trump_suit=self.trump_suit,
        )

        agent = self._get_agent()
        action = agent.act(obs)

        if isinstance(action, PlayCardAction):
            card = action.card
            sym = card_str(card)
            self._rec_label.config(text=f"▶ Play: {sym}")

            # Highlight the recommended card in green.
            if card in self.my_hand:
                self._card_buttons[card].config(bg="#ffd54f", fg="#000000")
        else:
            self._rec_label.config(text="No recommendation")

    def _get_agent(self):
        """Get or create the AI agent."""
        if self._agent is not None:
            return self._agent

        from agents.rule_based.rule_based_agent import RuleBasedAgent
        self._agent = RuleBasedAgent()
        return self._agent

    # ----------------------------------------------------------
    # Settings
    # ----------------------------------------------------------

    def _set_trump(self, suit: Suit) -> None:
        self.trump_suit = suit
        sym = SUIT_SYMBOLS[suit]
        self._trump_label.config(text=f"→ {suit.name} {sym}")

    def _load_model(self) -> None:
        from agents.learning.learning_agent import LearningAgent
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")],
            title="Load Learning Agent Model",
        )
        if path:
            self._agent = LearningAgent.load(path, training=False)
            self._agent_var.set("Learning (loaded)")
            self._instruction_label.config(text=f"✓ Model loaded: {self._agent.q_table_size} entries")

    def _update_trick_display(self) -> None:
        if not self.trick_cards:
            self._trick_display.config(text=f"Trick {self.trick_number}\nWaiting for cards...")
            return

        lines = [f"Trick {self.trick_number}"]
        for pid, card in self.trick_cards:
            lines.append(f"  P{pid+1}: {card_str(card)}")
        lines.append(f"  → Your turn ({len(self.trick_cards)}/3 played)")
        self._trick_display.config(text="\n".join(lines))

    def _update_info(self) -> None:
        self._info_label.config(
            text=f"Trick: {self.trick_number} | Hand: {len(self.my_hand)} cards")

    def _reset_all(self) -> None:
        self.my_hand.clear()
        self.trick_cards.clear()
        self.trump_suit = None
        self.trick_number = 0
        self.mode = "hand"
        self._agent = None

        for card, btn in self._card_buttons.items():
            fg = "#c62828" if card.suit in (Suit.HEARTS, Suit.DIAMONDS) else "#303030"
            btn.config(bg=COLORS["card_bg"], fg=fg, relief="solid",
                       command=lambda c=card: self._card_clicked(c))

        self._mode_label.config(text="YOUR HAND — Click cards to add (0/13)")
        self._trump_label.config(text="")
        self._trick_display.config(text="Select hand + trump first")
        self._rec_label.config(text="—")
        self._instruction_label.config(text="Step 1: Click cards to select your 13-card hand")
        self._update_info()
