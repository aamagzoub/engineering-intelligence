"""
Human vs AI tab — Player 3 is a human who clicks cards to play.

The human player sees their hand as clickable buttons and makes
bidding/play decisions through the GUI. Other 3 players are AI.
"""

import tkinter as tk
from tkinter import ttk

from gui.colors import COLORS
from intelligence.core.cards.card import Card
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit


class HumanTab:
    """Builds and manages the Human vs AI tab."""

    def __init__(self, parent: tk.Frame, root: tk.Tk) -> None:
        self.parent = parent
        self.root = root
        self.game_running = False

        # Callback set by the app when human makes a decision.
        self.on_card_selected = None
        self.on_bid_selected = None

        self._build()

    def _build(self) -> None:
        self.parent.configure(bg=COLORS["table_felt"])

        # Top: info bar.
        info = tk.Frame(self.parent, bg=COLORS["header_bg"], height=50)
        info.pack(fill="x", padx=0, pady=0)
        info.pack_propagate(False)

        self._status_label = tk.Label(
            info, text="Press 'Start Game' to begin playing as Player 3",
            font=("Segoe UI", 11), fg=COLORS["text_light"], bg=COLORS["header_bg"])
        self._status_label.pack(pady=12)

        # Middle: game area (simplified table showing your hand + centre trick).
        game_area = tk.Frame(self.parent, bg=COLORS["table_felt"])
        game_area.pack(fill="both", expand=True, padx=12, pady=8)
        game_area.columnconfigure(0, weight=1)
        game_area.rowconfigure(0, weight=1)
        game_area.rowconfigure(1, weight=1)

        # Centre: trick display.
        centre = tk.Frame(game_area, bg=COLORS["centre_bg"], bd=2, relief="sunken",
                          width=400, height=150)
        centre.grid(row=0, column=0, pady=(0, 8), sticky="nsew")
        centre.pack_propagate(False)

        self._trick_label = tk.Label(
            centre, text="Waiting to start...",
            font=("Segoe UI", 12, "bold"), fg=COLORS["text_light"],
            bg=COLORS["centre_bg"], justify="center")
        self._trick_label.pack(expand=True)

        # Bottom: your hand (clickable cards).
        hand_frame = tk.LabelFrame(game_area, text="Your Hand (Player 3)",
                                   font=("Segoe UI", 10, "bold"),
                                   fg=COLORS["gold"], bg=COLORS["player_bg"],
                                   bd=2, padx=10, pady=8)
        hand_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        self._hand_frame = tk.Frame(hand_frame, bg=COLORS["player_bg"])
        self._hand_frame.pack(fill="x")

        # Bid controls (hidden until bidding phase).
        self._bid_frame = tk.Frame(hand_frame, bg=COLORS["player_bg"])
        self._bid_label = tk.Label(self._bid_frame, text="",
                                   font=("Segoe UI", 10), fg=COLORS["text_light"],
                                   bg=COLORS["player_bg"])
        self._bid_label.pack(side="left", padx=(0, 12))

        # Bottom controls.
        ctrl = tk.Frame(self.parent, bg=COLORS["header_bg"], height=44)
        ctrl.pack(fill="x", padx=0, pady=0)
        ctrl.pack_propagate(False)

        btn_frame = tk.Frame(ctrl, bg=COLORS["header_bg"])
        btn_frame.pack(anchor="center", pady=8)

        tk.Button(btn_frame, text="▶ Start Game", command=self._start_game,
                  font=("Segoe UI", 9, "bold"), fg="#fff", bg=COLORS["btn_green"],
                  bd=0, padx=16, pady=4, cursor="hand2").pack(side="left", padx=4)

        tk.Button(btn_frame, text="⏹ Stop", command=self._stop_game,
                  font=("Segoe UI", 9, "bold"), fg="#fff", bg=COLORS["btn_red"],
                  bd=0, padx=16, pady=4, cursor="hand2").pack(side="left", padx=4)

        self._info_label = tk.Label(btn_frame, text="",
                                    font=("Segoe UI", 9), fg=COLORS["text_muted"],
                                    bg=COLORS["header_bg"])
        self._info_label.pack(side="left", padx=16)

    def set_status(self, text: str) -> None:
        self._status_label.config(text=text)

    def set_trick_display(self, text: str) -> None:
        self._trick_label.config(text=text)

    def show_hand(self, cards: list[Card], on_click) -> None:
        """Display clickable card buttons for the human player."""
        for w in self._hand_frame.winfo_children():
            w.destroy()

        suit_order = [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]
        suit_symbols = {Suit.SPADES: "♠", Suit.HEARTS: "♥",
                        Suit.CLUBS: "♣", Suit.DIAMONDS: "♦"}
        rank_symbols = {Rank.ACE: "A", Rank.KING: "K", Rank.QUEEN: "Q",
                        Rank.JACK: "J", Rank.TEN: "10", Rank.NINE: "9",
                        Rank.EIGHT: "8", Rank.SEVEN: "7", Rank.SIX: "6",
                        Rank.FIVE: "5", Rank.FOUR: "4", Rank.THREE: "3",
                        Rank.TWO: "2"}

        for suit in suit_order:
            suit_cards = sorted([c for c in cards if c.suit == suit],
                                key=lambda c: -list(Rank).index(c.rank))
            if not suit_cards:
                continue

            row = tk.Frame(self._hand_frame, bg=COLORS["player_bg"])
            row.pack(fill="x", pady=2)

            fg = "#c62828" if suit in (Suit.HEARTS, Suit.DIAMONDS) else "#303030"

            for card in suit_cards:
                text = f"{rank_symbols[card.rank]}{suit_symbols[card.suit]}"
                btn = tk.Button(
                    row, text=text, font=("Consolas", 11, "bold"),
                    fg=fg, bg=COLORS["card_bg"], relief="raised", bd=2,
                    padx=4, pady=2, width=3, cursor="hand2",
                    command=lambda c=card: on_click(c))
                btn.pack(side="left", padx=2)

    def show_bid_options(self, min_bid: int, max_bid: int, can_pass: bool, on_bid) -> None:
        """Show bidding buttons."""
        self._bid_frame.pack(fill="x", pady=(8, 0))
        for w in self._bid_frame.winfo_children():
            if w != self._bid_label:
                w.destroy()

        self._bid_label.config(text="Your bid:")

        if can_pass:
            tk.Button(self._bid_frame, text="Pass", font=("Segoe UI", 9, "bold"),
                      fg="#fff", bg=COLORS["btn_grey"], bd=0, padx=10, pady=3,
                      cursor="hand2", command=lambda: on_bid(None)
                      ).pack(side="left", padx=3)

        for val in range(min_bid, min(max_bid + 1, 14)):
            tk.Button(self._bid_frame, text=str(val), font=("Segoe UI", 9, "bold"),
                      fg="#fff", bg=COLORS["btn_blue"], bd=0, padx=8, pady=3,
                      cursor="hand2", command=lambda v=val: on_bid(v)
                      ).pack(side="left", padx=2)

    def hide_bid_options(self) -> None:
        self._bid_frame.pack_forget()

    def _start_game(self) -> None:
        self.set_status("Feature coming soon — Human vs AI game mode.")
        self._info_label.config(text="Interactive play not yet wired to game engine.")

    def _stop_game(self) -> None:
        self.game_running = False
        self.set_status("Game stopped.")
