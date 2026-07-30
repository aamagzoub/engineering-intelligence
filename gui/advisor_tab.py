"""
Play for AI tab — You are the AI's eyes and hands at a physical table.

Flow:
1. Select AI's 13 cards from the deck
2. Tell it who is Qabool
3. AI decides its bid (you announce it at the table)
4. Input what others bid → app resolves bidding
5. Tell it the trump suit + who won the bid
6. Tricks: AI tells you what to play, you input what others play
"""

import tkinter as tk
from tkinter import filedialog
from collections import Counter

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
    """Play for AI — you execute AI decisions at a physical table."""

    def __init__(self, parent: tk.Frame, root: tk.Tk) -> None:
        self.parent = parent
        self.root = root

        # Game state.
        self.ai_hand: list[Card] = []
        self.trump_suit: Suit | None = None
        self.qabool_id: int = 0  # 0=You(AI), 1=Right, 2=Partner, 3=Left
        self.bid_value: int = 0
        self.bid_winner_id: int = 0
        self.trick_number: int = 0
        self.trick_cards: list[tuple[int, Card]] = []
        self.leader_id: int = 0
        self.team_tricks = [0, 0]  # Team 0 (AI+Partner) vs Team 1

        # Phase: "hand", "bidding", "setup", "playing", "done"
        self.phase = "hand"

        # Track which players are known to be void in which suits.
        # {pid: set of suits they've shown void in}
        self._known_voids: dict[int, set] = {0: set(), 1: set(), 2: set(), 3: set()}
        # Pending off-suit confirmation (player tried off-suit, needs confirm).
        self._pending_offsuit: tuple | None = None

        # AI agent.
        self._agent = None

        self._build()

    def _build(self) -> None:
        self.parent.configure(bg="#1a1a1a")

        # Top: phase instruction (big, clear).
        top = tk.Frame(self.parent, bg="#252525", padx=12, pady=10)
        top.pack(fill="x", padx=12, pady=(8, 4))

        tk.Label(top, text="Play for AI",
                 font=("Segoe UI", 14, "bold"), fg="#ffffff", bg="#252525"
                 ).pack(anchor="w")

        self._instruction = tk.Label(
            top, text="Step 1: Click your 13 cards from the deck below",
            font=("Segoe UI", 11), fg="#ffd54f", bg="#252525")
        self._instruction.pack(anchor="w", pady=(4, 0))

        # AI command display (what to do at the table).
        self._command_frame = tk.Frame(top, bg="#1a1a1a", padx=12, pady=8)
        self._command_frame.pack(fill="x", pady=(8, 0))

        self._command_label = tk.Label(
            self._command_frame, text="Select your hand first",
            font=("Consolas", 16, "bold"), fg="#4caf50", bg="#1a1a1a",
            anchor="center")
        self._command_label.pack(fill="x")

        # Main area: deck grid + controls.
        main = tk.Frame(self.parent, bg="#1a1a1a")
        main.pack(fill="both", expand=True, padx=12, pady=4)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        # Left: deck grid.
        left = tk.Frame(main, bg="#252525", bd=1, relief="groove", padx=8, pady=6)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        self._deck_label = tk.Label(left, text="AI HAND — Click 13 cards (0/13)",
                                    font=("Segoe UI", 10, "bold"),
                                    fg=COLORS["gold"], bg="#252525")
        self._deck_label.pack(anchor="w", pady=(0, 4))

        self._deck_frame = tk.Frame(left, bg="#252525")
        self._deck_frame.pack(fill="both", expand=True)

        self._card_buttons: dict[Card, tk.Button] = {}
        self._build_deck_grid()

        # Trump indicator (bottom-left, visible during play).
        self._trump_display = tk.Label(left, text="",
                                       font=("Segoe UI", 14, "bold"),
                                       fg=COLORS["gold"], bg="#252525")
        self._trump_display.pack(anchor="w", pady=(8, 0))

        # Right: controls panel (changes per phase, fixed size to prevent jumps).
        self._right = tk.Frame(main, bg="#252525", bd=1, relief="groove", padx=12, pady=8,
                               width=320, height=500)
        self._right.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self._right.pack_propagate(False)

        self._build_right_panel()

    def _build_deck_grid(self) -> None:
        for suit in ALL_SUITS:
            row = tk.Frame(self._deck_frame, bg="#252525")
            row.pack(fill="x", pady=6)
            fg = "#c62828" if suit in (Suit.HEARTS, Suit.DIAMONDS) else "#303030"
            for rank in reversed(ALL_RANKS):
                card = Card(suit, rank)
                text = f"{RANK_SYMBOLS[rank]}{SUIT_SYMBOLS[suit]}"
                btn = tk.Button(
                    row, text=text, font=("Consolas", 13, "bold"),
                    fg=fg, bg=COLORS["card_bg"], relief="solid", bd=1,
                    padx=5, pady=4, width=4, cursor="hand2",
                    command=lambda c=card: self._card_clicked(c))
                btn.pack(side="left", padx=2)
                self._card_buttons[card] = btn

    def _build_right_panel(self) -> None:
        """Build the right panel controls (setup phase)."""
        for w in self._right.winfo_children():
            w.destroy()

        # Agent selector.
        tk.Label(self._right, text="AI Agent", font=("Segoe UI", 10, "bold"),
                 fg="#ffffff", bg="#252525").pack(anchor="w", pady=(0, 4))

        agent_row = tk.Frame(self._right, bg="#252525")
        agent_row.pack(fill="x", pady=(0, 8))
        self._agent_var = tk.StringVar(value="Rule-Based")
        tk.OptionMenu(agent_row, self._agent_var, "Rule-Based", "Learning"
                      ).pack(side="left", padx=(0, 8))
        tk.Button(agent_row, text="📂 Load Model", command=self._load_model,
                  font=("Segoe UI", 9), fg="#fff", bg="#1e88e5",
                  bd=0, padx=8, pady=3, cursor="hand2").pack(side="left")

        # Qabool selector.
        tk.Label(self._right, text="Who is Sahib Al-Qabool?",
                 font=("Segoe UI", 10, "bold"), fg="#ffffff", bg="#252525"
                 ).pack(anchor="w", pady=(12, 4))

        self._qabool_var = tk.StringVar(value="AI (You)")
        options = ["AI (You)", "P2 (Left)", "P3 (Opposite)", "P4 (Right)"]
        tk.OptionMenu(self._right, self._qabool_var, *options).pack(anchor="w")

        # Start button.
        btn_row2 = tk.Frame(self._right, bg="#252525")
        btn_row2.pack(anchor="w", pady=(20, 0))
        tk.Button(btn_row2, text="▶  Start Bidding", command=self._start_bidding,
                  font=("Segoe UI", 11, "bold"), fg="#fff", bg=COLORS["btn_green"],
                  bd=0, padx=16, pady=8, cursor="hand2").pack(side="left", padx=(0, 8))
        tk.Button(btn_row2, text="🎲 Random Hand", command=self._random_hand,
                  font=("Segoe UI", 9, "bold"), fg="#fff", bg="#7b1fa2",
                  bd=0, padx=10, pady=8, cursor="hand2").pack(side="left")

        # Info.
        self._info_label = tk.Label(self._right, text="",
                                    font=("Segoe UI", 9), fg="#888888", bg="#252525")
        self._info_label.pack(anchor="w", pady=(12, 0))

        # Reset.
        tk.Button(self._right, text="↺ Reset All", command=self._reset_all,
                  font=("Segoe UI", 9), fg="#fff", bg=COLORS["btn_grey"],
                  bd=0, padx=8, pady=4, cursor="hand2").pack(anchor="w", pady=(12, 0))

    # ----------------------------------------------------------
    # Card selection
    # ----------------------------------------------------------

    def _card_clicked(self, card: Card) -> None:
        if self.phase == "hand":
            self._toggle_hand_card(card)
        elif self.phase == "playing":
            self._opponent_plays(card)

    def _toggle_hand_card(self, card: Card) -> None:
        btn = self._card_buttons[card]
        if card in self.ai_hand:
            self.ai_hand.remove(card)
            fg = "#c62828" if card.suit in (Suit.HEARTS, Suit.DIAMONDS) else "#303030"
            btn.config(bg=COLORS["card_bg"], fg=fg, relief="solid")
        else:
            if len(self.ai_hand) >= 13:
                return
            self.ai_hand.append(card)
            btn.config(bg="#4caf50", fg="#ffffff", relief="raised")
        self._deck_label.config(text=f"AI HAND — Click 13 cards ({len(self.ai_hand)}/13)")


    # ----------------------------------------------------------
    # Bidding phase
    # ----------------------------------------------------------

    def _start_bidding(self) -> None:
        if len(self.ai_hand) != 13:
            self._command_label.config(text="⚠ Select exactly 13 cards first!")
            return

        # Check for card-based Dak before bidding.
        from collections import Counter as _Counter
        suit_counts = _Counter(c.suit for c in self.ai_hand)
        has_picture = any(c.rank in (Rank.ACE, Rank.KING, Rank.QUEEN, Rank.JACK)
                          for c in self.ai_hand)
        max_suit_count = max(suit_counts.values()) if suit_counts else 0

        if not has_picture or max_suit_count >= 8:
            reason = "No picture cards" if not has_picture else f"8+ cards in one suit ({max_suit_count})"
            self._command_label.config(
                text=f"📢 Declare CARD DAK! ({reason})", fg="#ff5252")
            self._instruction.config(text="You must declare Card-Based Dak. Show proof and re-deal.")

            # Show declare button.
            for w in self._right.winfo_children():
                w.destroy()
            tk.Label(self._right, text=f"Card-Based Dak Detected",
                     font=("Segoe UI", 12, "bold"), fg="#ff5252", bg="#252525"
                     ).pack(anchor="w", pady=(8, 4))
            tk.Label(self._right, text=f"Reason: {reason}",
                     font=("Segoe UI", 10), fg="#aaaaaa", bg="#252525"
                     ).pack(anchor="w", pady=(0, 12))
            tk.Button(self._right, text="Declare Dak (Re-deal)",
                      command=self._declare_dak,
                      font=("Segoe UI", 11, "bold"), fg="#fff", bg=COLORS["btn_red"],
                      bd=0, padx=16, pady=8, cursor="hand2").pack(anchor="w")
            return

        qabool_map = {"AI (You)": 0, "P2 (Left)": 1, "P3 (Opposite)": 2, "P4 (Right)": 3}
        self.qabool_id = qabool_map.get(self._qabool_var.get(), 0)
        self.phase = "bidding"

        # Bidding order: clockwise from left of Qabool.
        order = [(self.qabool_id + 1) % 4,
                 (self.qabool_id + 2) % 4,
                 (self.qabool_id + 3) % 4]
        self._bid_order = order
        self._bid_history: list[tuple[int, int | None]] = []  # (pid, value or None=pass)
        self._highest_bid: int | None = None
        self._highest_bidder: int | None = None
        self._bid_step = 0  # Which player in order we're at.
        self._is_opening_bid = True

        self._instruction.config(text="Bidding — enter each player's bid in order")
        self._build_bidding_step_panel()

    def _build_bidding_step_panel(self) -> None:
        """Show the current bidding step — who's bidding now."""
        for w in self._right.winfo_children():
            w.destroy()

        player_names = {0: "AI", 1: "P2", 2: "P3", 3: "P4"}

        # Mini-table showing player positions and their bid status.
        table_frame = tk.Frame(self._right, bg="#1a1a1a", padx=6, pady=6)
        table_frame.pack(fill="x", pady=(0, 8))
        table_frame.columnconfigure(0, weight=1)
        table_frame.columnconfigure(1, weight=1)
        table_frame.columnconfigure(2, weight=1)

        # Determine current bidder.
        current_pid = None
        if self._bid_step < 3:
            current_pid = self._bid_order[self._bid_step]
        else:
            current_pid = self.qabool_id

        slot_positions = {2: (0, 1), 1: (1, 0), 3: (1, 2), 0: (2, 1)}
        for pid, (row, col) in slot_positions.items():
            # Determine what to show in the slot.
            bid_text = ""
            for h_pid, h_val in self._bid_history:
                if h_pid == pid:
                    bid_text = "Pass" if h_val is None else f"Bid {h_val}"
            if pid == self.qabool_id:
                name = f"● {player_names[pid]}"
            else:
                name = player_names[pid]

            display = f"{name}\n{bid_text}" if bid_text else name

            # Highlight active player.
            if pid == current_pid:
                bg, fg = "#3a5a3a", "#ffd54f"
            elif bid_text:
                bg, fg = "#2a3a2a", "#aaaaaa"
            else:
                bg, fg = "#2a3a2a", "#666666"

            # Red highlight for Qabool.
            border_color = "#cc3333" if pid == self.qabool_id else "#444444"

            slot = tk.Label(table_frame, text=display,
                            font=("Segoe UI", 9, "bold"), fg=fg, bg=bg,
                            width=9, height=2, relief="solid", bd=2,
                            highlightbackground=border_color, highlightthickness=2)
            slot.grid(row=row, column=col, padx=3, pady=3)

        # Full player names reference.
        player_names_full = {0: "AI (You)", 1: "P2 (Left)", 2: "P3 (Opposite)", 3: "P4 (Right)"}

        # Show bid history so far.
        if self._bid_history:
            history_frame = tk.Frame(self._right, bg="#252525")
            history_frame.pack(fill="x", pady=(0, 8))
            tk.Label(history_frame, text="Bids so far:", font=("Segoe UI", 9),
                     fg="#888888", bg="#252525").pack(anchor="w")
            for pid, val in self._bid_history:
                text = f"  {player_names[pid]}: {'Pass' if val is None else f'Bid {val}'}"
                fg = "#888888" if val is None else COLORS["gold"]
                tk.Label(history_frame, text=text, font=("Consolas", 10),
                         fg=fg, bg="#252525").pack(anchor="w")

        # Are we still in the 3-player round or at Qabool's decision?
        if self._bid_step < 3:
            current_pid = self._bid_order[self._bid_step]
            is_ai_turn = (current_pid == 0)

            tk.Label(self._right, text=f"Now: {player_names[current_pid]}",
                     font=("Segoe UI", 12, "bold"), fg="#ffffff", bg="#252525"
                     ).pack(anchor="w", pady=(4, 8))

            if is_ai_turn:
                # AI decides — ask the agent.
                self._ai_bid_decision()
            else:
                # Other player — ask user to input their bid/pass.
                tk.Label(self._right, text="What did they say?",
                         font=("Segoe UI", 10), fg="#aaaaaa", bg="#252525").pack(anchor="w")

                btn_row = tk.Frame(self._right, bg="#252525")
                btn_row.pack(fill="x", pady=(6, 4))

                tk.Button(btn_row, text="Pass", command=lambda: self._record_bid(current_pid, None),
                          font=("Segoe UI", 10, "bold"), fg="#fff", bg=COLORS["btn_grey"],
                          bd=0, padx=12, pady=5, cursor="hand2").pack(side="left", padx=3)

                # Bid value buttons.
                min_val = (self._highest_bid + 1) if self._highest_bid else 7
                for v in range(min_val, 14):
                    tk.Button(btn_row, text=str(v),
                              command=lambda val=v: self._record_bid(current_pid, val),
                              font=("Segoe UI", 10, "bold"), fg="#fff", bg=COLORS["btn_green"],
                              bd=0, padx=8, pady=5, cursor="hand2").pack(side="left", padx=2)

        else:
            # Qabool's decision.
            tk.Label(self._right, text=f"Qabool: {player_names[self.qabool_id]}",
                     font=("Segoe UI", 12, "bold"), fg=COLORS["gold"], bg="#252525"
                     ).pack(anchor="w", pady=(4, 4))

            is_ai_qabool = (self.qabool_id == 0)

            if self._highest_bid is None:
                # Everyone passed — Qabool must decide: bid or Dak.
                if is_ai_qabool:
                    self._ai_qabool_decision()
                else:
                    tk.Label(self._right, text="All passed. Qabool decides:",
                             font=("Segoe UI", 10), fg="#aaaaaa", bg="#252525").pack(anchor="w")
                    btn_row = tk.Frame(self._right, bg="#252525")
                    btn_row.pack(fill="x", pady=6)
                    tk.Button(btn_row, text="Dak!", command=self._declare_dak,
                              font=("Segoe UI", 10, "bold"), fg="#fff", bg=COLORS["btn_red"],
                              bd=0, padx=12, pady=5, cursor="hand2").pack(side="left", padx=3)
                    for v in range(7, 14):
                        tk.Button(btn_row, text=str(v),
                                  command=lambda val=v: self._qabool_bids(val),
                                  font=("Segoe UI", 10, "bold"), fg="#fff", bg=COLORS["btn_green"],
                                  bd=0, padx=6, pady=5, cursor="hand2").pack(side="left", padx=2)
            else:
                # Someone bid — Qabool: accept or match/outbid.
                if is_ai_qabool:
                    self._ai_qabool_decision()
                else:
                    tk.Label(self._right,
                             text=f"Highest bid: {self._highest_bid} by {player_names[self._highest_bidder]}",
                             font=("Segoe UI", 10), fg="#aaaaaa", bg="#252525").pack(anchor="w")
                    tk.Label(self._right, text="Qabool decides:",
                             font=("Segoe UI", 10), fg="#aaaaaa", bg="#252525").pack(anchor="w")
                    btn_row = tk.Frame(self._right, bg="#252525")
                    btn_row.pack(fill="x", pady=6)
                    tk.Button(btn_row, text="Accept",
                              command=lambda: self._qabool_accepts(),
                              font=("Segoe UI", 10, "bold"), fg="#fff", bg=COLORS["btn_orange"],
                              bd=0, padx=12, pady=5, cursor="hand2").pack(side="left", padx=3)
                    for v in range(self._highest_bid, 14):
                        tk.Button(btn_row, text=str(v),
                                  command=lambda val=v: self._qabool_bids(val),
                                  font=("Segoe UI", 10, "bold"), fg="#fff", bg=COLORS["btn_green"],
                                  bd=0, padx=6, pady=5, cursor="hand2").pack(side="left", padx=2)

    def _record_bid(self, pid: int, value: int | None) -> None:
        """Record a player's bid or pass and advance."""
        self._bid_history.append((pid, value))
        if value is not None:
            if self._highest_bid is None or value > self._highest_bid:
                self._highest_bid = value
                self._highest_bidder = pid
            self._is_opening_bid = False
            # Bid of 13 stops bidding immediately.
            if value == 13:
                self._bid_step = 3  # Jump to Qabool.
                self._build_bidding_step_panel()
                return
        self._bid_step += 1
        self._build_bidding_step_panel()

    def _ai_bid_decision(self) -> None:
        """AI decides its bid."""
        from environments.wist.observation import BiddingObservation
        from environments.wist.actions import BidAction, PassAction

        agent = self._get_agent()
        obs = BiddingObservation(
            player_id=0,
            hand=list(self.ai_hand),
            previous_bids=list(self._bid_history),
            current_highest_bid=self._highest_bid,
            is_sahib_al_qabool=False,
            is_opening_bid=self._is_opening_bid,
        )
        action = agent.act(obs)

        if isinstance(action, BidAction):
            bid_val = action.value
            self._command_label.config(text=f"📢 Say: \"BID {bid_val}\"", fg="#4caf50")
            tk.Label(self._right, text=f"AI bids: {bid_val}",
                     font=("Segoe UI", 14, "bold"), fg="#4caf50", bg="#252525"
                     ).pack(anchor="w", pady=8)
        else:
            bid_val = None
            self._command_label.config(text=f"📢 Say: \"PASS\"", fg="#ff9800")
            tk.Label(self._right, text="AI passes",
                     font=("Segoe UI", 14, "bold"), fg="#ff9800", bg="#252525"
                     ).pack(anchor="w", pady=8)

        tk.Button(self._right, text="→ Continue", command=lambda: self._record_bid(0, bid_val),
                  font=("Segoe UI", 10, "bold"), fg="#fff", bg=COLORS["btn_green"],
                  bd=0, padx=12, pady=5, cursor="hand2").pack(anchor="w", pady=4)

    def _ai_qabool_decision(self) -> None:
        """AI is Qabool — decide accept/match/dak."""
        from environments.wist.observation import BiddingObservation
        from environments.wist.actions import BidAction, PassAction

        agent = self._get_agent()
        obs = BiddingObservation(
            player_id=0,
            hand=list(self.ai_hand),
            previous_bids=list(self._bid_history),
            current_highest_bid=self._highest_bid,
            is_sahib_al_qabool=True,
            is_opening_bid=self._is_opening_bid,
        )
        action = agent.act(obs)

        if isinstance(action, PassAction):
            if self._highest_bid is None:
                # Dak.
                self._command_label.config(text="📢 Say: \"DAK!\"", fg="#ff5252")
                tk.Label(self._right, text="AI declares DAK!",
                         font=("Segoe UI", 14, "bold"), fg="#ff5252", bg="#252525"
                         ).pack(anchor="w", pady=8)
                tk.Button(self._right, text="OK (Re-deal)", command=self._declare_dak,
                          font=("Segoe UI", 10), fg="#fff", bg=COLORS["btn_red"],
                          bd=0, padx=12, pady=5, cursor="hand2").pack(anchor="w")
            else:
                # Accept.
                self._command_label.config(text="📢 AI ACCEPTS the bid", fg="#ffd54f")
                tk.Label(self._right, text="AI accepts",
                         font=("Segoe UI", 14, "bold"), fg="#ffd54f", bg="#252525"
                         ).pack(anchor="w", pady=8)
                tk.Button(self._right, text="→ Continue", command=self._qabool_accepts,
                          font=("Segoe UI", 10, "bold"), fg="#fff", bg=COLORS["btn_green"],
                          bd=0, padx=12, pady=5, cursor="hand2").pack(anchor="w")
        else:
            bid_val = action.value
            self._command_label.config(text=f"📢 Say: \"I BID {bid_val}\"", fg="#4caf50")
            tk.Label(self._right, text=f"AI bids: {bid_val} (as Qabool)",
                     font=("Segoe UI", 14, "bold"), fg="#4caf50", bg="#252525"
                     ).pack(anchor="w", pady=8)
            tk.Button(self._right, text="→ Continue",
                      command=lambda: self._qabool_bids(bid_val),
                      font=("Segoe UI", 10, "bold"), fg="#fff", bg=COLORS["btn_green"],
                      bd=0, padx=12, pady=5, cursor="hand2").pack(anchor="w")

    def _qabool_accepts(self) -> None:
        """Qabool accepts the highest bid — that bidder's team plays."""
        self.bid_winner_id = self._highest_bidder
        self.bid_value = self._highest_bid
        self._finalize_bidding()

    def _qabool_bids(self, value: int) -> None:
        """Qabool matches or outbids — Qabool's team plays."""
        self.bid_winner_id = self.qabool_id
        self.bid_value = value
        self._highest_bid = value
        self._highest_bidder = self.qabool_id
        self._finalize_bidding()

    def _declare_dak(self) -> None:
        """Dak declared — re-deal: clear hand and go back to card selection."""
        self.ai_hand.clear()
        self.trick_cards.clear()
        self.trump_suit = None
        self.trick_number = 0
        self.phase = "hand"
        self.team_tricks = [0, 0]

        # Reset card buttons.
        for card, btn in self._card_buttons.items():
            fg = "#c62828" if card.suit in (Suit.HEARTS, Suit.DIAMONDS) else "#303030"
            btn.config(bg=COLORS["card_bg"], fg=fg, relief="solid",
                       command=lambda c=card: self._card_clicked(c))

        self._deck_label.config(text="AI HAND — Click 13 cards (0/13)")
        self._command_label.config(text="DAK! Select your new 13 cards after re-deal.", fg="#ff5252")
        self._instruction.config(text="Dak declared — cards re-dealt. Select your new hand.")
        self._build_right_panel()

    def _finalize_bidding(self) -> None:
        """Bidding resolved — determine trump and start play."""
        player_names = {0: "AI (You)", 1: "P2 (Left)", 2: "P3 (Opposite)", 3: "P4 (Right)"}

        # If AI won the bid, AI picks trump (longest suit).
        if self.bid_winner_id == 0:
            from environments.wist.tasmiya_engine import determine_trump_suit
            self.trump_suit = determine_trump_suit(self.ai_hand)
            sym = SUIT_SYMBOLS[self.trump_suit]
            self._command_label.config(
                text=f"🃏 Play your first card from {sym} (trump)", fg="#4caf50")
            self._instruction.config(
                text=f"You won bid {self.bid_value}. Trump: {sym}. Play starts!")
            self._show_trump_and_start()
        else:
            # Someone else won — ask user what trump was revealed.
            self._instruction.config(
                text=f"{player_names[self.bid_winner_id]} won bid {self.bid_value}. What is trump?")
            self._command_label.config(text="⏳ Waiting: what suit did they lead?", fg="#ff9800")
            self._build_trump_selection_panel()

    def _build_trump_selection_panel(self) -> None:
        """Ask user to select the revealed trump suit."""
        for w in self._right.winfo_children():
            w.destroy()

        tk.Label(self._right, text="What is the trump suit?",
                 font=("Segoe UI", 12, "bold"), fg="#ffffff", bg="#252525"
                 ).pack(anchor="w", pady=(0, 8))
        tk.Label(self._right, text="(Revealed by the first card played)",
                 font=("Segoe UI", 9), fg="#888888", bg="#252525").pack(anchor="w", pady=(0, 8))

        trump_row = tk.Frame(self._right, bg="#252525")
        trump_row.pack(anchor="w", pady=8)
        for suit in ALL_SUITS:
            sym = SUIT_SYMBOLS[suit]
            fg = "#c62828" if suit in (Suit.HEARTS, Suit.DIAMONDS) else "#ffffff"
            tk.Button(trump_row, text=sym, font=("Segoe UI", 20, "bold"),
                      fg=fg, bg="#333333", bd=1, padx=12, pady=6, cursor="hand2",
                      command=lambda s=suit: self._set_trump_and_start(s)
                      ).pack(side="left", padx=6)

    def _set_trump_and_start(self, suit: Suit) -> None:
        self.trump_suit = suit
        self._show_trump_and_start()

    def _show_trump_and_start(self) -> None:
        """Trump is set — show it on the left panel and start playing."""
        sym = SUIT_SYMBOLS[self.trump_suit]
        fg = "#c62828" if self.trump_suit in (Suit.HEARTS, Suit.DIAMONDS) else "#ffffff"
        self._trump_display.config(text=f"Trump: {sym}  |  Bid: {self.bid_value}", fg=fg)

        self.phase = "playing"
        self.trick_number = 0
        self.leader_id = self.bid_winner_id
        self.team_tricks = [0, 0]
        self._instruction.config(text=f"Playing — Trump: {sym} | Bid: {self.bid_value}")
        self._build_playing_panel()
        self._start_next_trick()

    # ----------------------------------------------------------
    # Playing phase
    # ----------------------------------------------------------

    def _start_playing(self) -> None:
        """Called from _show_trump_and_start — phase is already 'playing'."""
        pass  # Playing is started from _show_trump_and_start.

    def _build_playing_panel(self) -> None:
        """Build the right panel with a visual mini-table for trick play."""
        for w in self._right.winfo_children():
            w.destroy()

        player_names = {0: "AI", 1: "P2", 2: "P3", 3: "P4"}

        tk.Label(self._right, text=f"Trick {self.trick_number}/13",
                 font=("Segoe UI", 11, "bold"), fg="#ffffff", bg="#252525"
                 ).pack(anchor="w")

        self._trick_info = tk.Label(self._right, text="",
                                    font=("Segoe UI", 9), fg="#aaaaaa", bg="#252525")
        self._trick_info.pack(anchor="w", pady=(2, 6))

        # Mini-table: 4 card slots arranged as a cross.
        table_frame = tk.Frame(self._right, bg="#1a1a1a", padx=8, pady=8)
        table_frame.pack(fill="x", pady=4)
        # Layout: 3 rows x 3 cols. P3=top, P4=left, P2=right, AI=bottom.
        table_frame.columnconfigure(0, weight=1)
        table_frame.columnconfigure(1, weight=1)
        table_frame.columnconfigure(2, weight=1)

        # Card slot widgets (label-based, will show card text or player name).
        self._trick_slots = {}
        slot_positions = {2: (0, 1), 1: (1, 0), 3: (1, 2), 0: (2, 1)}  # P3=top, P2=left, P4=right, AI=bottom

        for pid, (row, col) in slot_positions.items():
            slot = tk.Label(table_frame, text=player_names[pid],
                            font=("Segoe UI", 9, "bold"),
                            fg="#555555", bg="#2a3a2a",
                            width=9, height=2, relief="solid", bd=2,
                            highlightbackground="#444444", highlightthickness=2)
            slot.grid(row=row, column=col, padx=3, pady=3)
            self._trick_slots[pid] = slot

        # Score display.
        self._score_label = tk.Label(self._right, text="Tricks — T1: 0 | T2: 0",
                                     font=("Segoe UI", 10, "bold"),
                                     fg=COLORS["gold"], bg="#252525")
        self._score_label.pack(anchor="w", pady=(12, 0))

        # Reset.
        tk.Button(self._right, text="↺ Reset All", command=self._reset_all,
                  font=("Segoe UI", 9), fg="#fff", bg=COLORS["btn_grey"],
                  bd=0, padx=8, pady=4, cursor="hand2").pack(anchor="w", pady=(12, 0))

    def _start_next_trick(self) -> None:
        """Start a new trick."""
        self.trick_number += 1
        self.trick_cards = []

        if self.trick_number > 13:
            self._end_shota()
            return

        player_names = {0: "AI", 1: "P2", 2: "P3", 3: "P4"}
        self._trick_info.config(
            text=f"Trick {self.trick_number}/13 — Leader: {player_names[self.leader_id]}")

        # Reset slot visuals.
        for pid, slot in self._trick_slots.items():
            slot.config(text=player_names[pid], fg="#555555", bg="#2a3a2a",
                        font=("Segoe UI", 9, "bold"),
                        highlightbackground="#444444")

        # Reset card grid highlights.
        self._reset_card_highlights()

        # Determine play order.
        self._play_order = [(self.leader_id + i) % 4 for i in range(4)]
        self._play_idx = 0
        self._advance_play()

    def _advance_play(self) -> None:
        """Advance to the next player in the trick."""
        if self._play_idx >= 4:
            self._resolve_trick()
            return

        pid = self._play_order[self._play_idx]
        player_names = {0: "AI", 1: "P2", 2: "P3", 3: "P4"}

        # Highlight the active player slot.
        for p, slot in self._trick_slots.items():
            if p == pid and p not in [pc[0] for pc in self.trick_cards]:
                slot.config(bg="#3a5a3a", fg="#ffd54f")  # Active highlight.

        if pid == 0:
            self._ai_play()
        else:
            self._deck_label.config(text=f"Click the card {player_names[pid]} played")
            self._command_label.config(
                text=f"⏳ {player_names[pid]}'s turn — click their card", fg="#ff9800")

            # Visual hint: if a suit was led and this player isn't known void,
            # highlight only the led-suit cards as valid choices.
            self._highlight_valid_cards(pid)

    def _ai_play(self) -> None:
        """AI decides which card to play."""
        from environments.wist.observation import WistObservation
        from environments.wist.trick import Trick
        from environments.wist.actions import PlayCardAction

        trick = None
        if self.trick_cards:
            trick = Trick(leading_player_id=self._play_order[0])
            for pid, card in self.trick_cards:
                trick.play_card(pid, card)

        obs = WistObservation(
            player_id=0,
            hand=list(self.ai_hand),
            current_trick=trick,
            trump_suit=self.trump_suit,
            team_scores={0: self.team_tricks[0], 1: self.team_tricks[1]},
            must_lead_trump=(self.trick_number == 1 and self.bid_winner_id == 0
                             and len(self.trick_cards) == 0),
        )

        agent = self._get_agent()
        action = agent.act(obs)

        if isinstance(action, PlayCardAction):
            card = action.card
            sym = card_str(card)
            self._command_label.config(text=f"🃏 PLAY: {sym}", fg="#4caf50")

            # Remove from AI hand and mark on grid.
            if card in self.ai_hand:
                self.ai_hand.remove(card)
                btn = self._card_buttons[card]
                btn.config(bg="#666666", fg="#999999", relief="flat")

            self.trick_cards.append((0, card))
            # Make the slot look like an actual card (white bg, colored suit text).
            card_fg = "#c62828" if card.suit in (Suit.HEARTS, Suit.DIAMONDS) else "#1a1a1a"
            self._trick_slots[0].config(text=sym, fg=card_fg, bg="#ffffff")
            self._play_idx += 1
            self._deck_label.config(text=f"AI HAND — {len(self.ai_hand)} cards left")

            self.root.after(500, self._advance_play)

    def _opponent_plays(self, card: Card) -> None:
        """An opponent played a card (user clicked it)."""
        if self.phase != "playing":
            return
        if self._play_idx >= 4:
            return

        pid = self._play_order[self._play_idx]
        if pid == 0:
            return

        # Block clicking AI's own cards for an opponent.
        if card in self.ai_hand:
            self._command_label.config(text="⚠ That card is in AI's hand!", fg="#ff5252")
            return

        # Block clicking already-played cards.
        already_played = [c for _, c in self.trick_cards]
        if card in already_played:
            return

        # Validation: bid winner's first card must be trump.
        if self.trick_number == 1 and pid == self.bid_winner_id and len(self.trick_cards) == 0:
            if card.suit != self.trump_suit:
                sym = SUIT_SYMBOLS[self.trump_suit]
                self._command_label.config(
                    text=f"⚠ First card must be trump ({sym})!", fg="#ff5252")
                return

        # Validation: must follow led suit unless proven void.
        if self.trick_cards:
            leading_suit = self.trick_cards[0][1].suit
            if card.suit != leading_suit:
                # Check if this player is known to be void in the led suit.
                if leading_suit not in self._known_voids.get(pid, set()):
                    # Not known void — check if there are still cards of that suit out.
                    suit_in_ai_hand = sum(1 for c in self.ai_hand if c.suit == leading_suit)
                    suit_played = sum(1 for c, btn in self._card_buttons.items()
                                      if c.suit == leading_suit and
                                      btn.cget("bg") in ("#666666", "#1e88e5"))
                    suit_remaining = 13 - suit_in_ai_hand - suit_played
                    sym = SUIT_SYMBOLS[leading_suit]

                    if suit_remaining > 0:
                        # Block — they might have it. Show confirm button.
                        if self._pending_offsuit == (pid, card):
                            # Already confirmed — allow it and mark void.
                            self._known_voids.setdefault(pid, set()).add(leading_suit)
                            self._pending_offsuit = None
                        else:
                            self._pending_offsuit = (pid, card)
                            player_names = {0: "AI", 1: "P2", 2: "P3", 3: "P4"}
                            self._command_label.config(
                                text=f"⚠ \"{player_names[pid]}, you must play {sym}! There are {suit_remaining} {sym} cards still out!\" — Click again if they insist",
                                fg="#ff5252")
                            return
                    else:
                        # All cards of that suit accounted for — they're void, allow.
                        self._known_voids.setdefault(pid, set()).add(leading_suit)
                else:
                    # Known void — allow silently.
                    pass
            else:
                # Following suit — clear any pending.
                self._pending_offsuit = None

        # Mark the card as played on grid.
        btn = self._card_buttons[card]
        btn.config(bg="#1e88e5", fg="#ffffff", relief="sunken")

        # Update the slot on the mini-table — make it look like a card.
        sym = card_str(card)
        card_fg = "#c62828" if card.suit in (Suit.HEARTS, Suit.DIAMONDS) else "#1a1a1a"
        self._trick_slots[pid].config(text=sym, fg=card_fg, bg="#ffffff")

        self.trick_cards.append((pid, card))
        self._play_idx += 1
        self._reset_card_highlights()
        self._advance_play()

    def _resolve_trick(self) -> None:
        """Determine trick winner and highlight."""
        from environments.wist.trick import Trick
        from environments.wist.rules import trick_winner

        trick = Trick(leading_player_id=self._play_order[0])
        for pid, card in self.trick_cards:
            trick.play_card(pid, card)

        winner = trick_winner(trick, self.trump_suit)
        winner_team = 0 if winner in (0, 2) else 1
        self.team_tricks[winner_team] += 1

        # Grey out opponent cards on the deck grid.
        for pid, card in self.trick_cards:
            if pid != 0:
                btn = self._card_buttons[card]
                btn.config(bg="#666666", fg="#999999", relief="flat")

        # Highlight winner slot gold.
        player_names = {0: "AI", 1: "P2", 2: "P3", 3: "P4"}
        self._trick_slots[winner].config(bg="#5a4a00", fg="#ffd54f")
        self._command_label.config(
            text=f"🏆 {player_names[winner]} wins the trick!", fg="#ffd54f")
        self._score_label.config(
            text=f"Tricks — T1: {self.team_tricks[0]} | T2: {self.team_tricks[1]}")

        self.leader_id = winner
        self._deck_label.config(text=f"AI HAND — {len(self.ai_hand)} cards left")

        # Auto-advance to next trick after a brief pause.
        self.root.after(1200, self._start_next_trick)

    def _update_trick_display(self) -> None:
        """No-op — display is now handled by slot widgets."""
        pass

    def _highlight_valid_cards(self, pid: int) -> None:
        """Visually highlight which cards are valid for this player to play.

        If a suit was led and the player isn't known void in it,
        dim all non-led-suit cards (except already-used ones).
        If the player IS known void or is leading, show all available.
        """
        # Determine the led suit (if any cards have been played this trick).
        leading_suit = None
        if self.trick_cards:
            leading_suit = self.trick_cards[0][1].suit

        # If this player is leading (no cards played yet), or known void, allow all.
        is_void = leading_suit in self._known_voids.get(pid, set()) if leading_suit else True

        for card, btn in self._card_buttons.items():
            # Skip cards already played (greyed out).
            current_bg = btn.cget("bg")
            if current_bg in ("#666666", "#1e88e5"):
                continue
            # Skip AI's own cards (green).
            if card in self.ai_hand:
                continue

            if leading_suit and not is_void:
                # Must follow suit — highlight led-suit cards, dim others.
                if card.suit == leading_suit:
                    # Valid card — bright border.
                    fg = "#c62828" if card.suit in (Suit.HEARTS, Suit.DIAMONDS) else "#303030"
                    btn.config(bg="#e8f5e9", fg=fg, relief="solid")
                else:
                    # Invalid (unless they're void) — dim it.
                    btn.config(bg="#3a3a3a", fg="#555555", relief="flat")
            else:
                # Can play anything — reset to normal.
                fg = "#c62828" if card.suit in (Suit.HEARTS, Suit.DIAMONDS) else "#303030"
                btn.config(bg=COLORS["card_bg"], fg=fg, relief="solid")

    def _reset_card_highlights(self) -> None:
        """Reset all non-used card buttons back to normal appearance."""
        for card, btn in self._card_buttons.items():
            current_bg = btn.cget("bg")
            if current_bg in ("#666666", "#1e88e5"):
                continue  # Already played.
            if card in self.ai_hand:
                continue  # AI's cards stay green.
            fg = "#c62828" if card.suit in (Suit.HEARTS, Suit.DIAMONDS) else "#303030"
            btn.config(bg=COLORS["card_bg"], fg=fg, relief="solid")

    def _end_shota(self) -> None:
        """Show end-of-shota summary with option to continue."""
        self.phase = "shota_end"
        t1 = self.team_tricks[0]
        t2 = self.team_tricks[1]
        winner = "Your Team (AI+P3)" if t1 > t2 else "Opponents (P2+P4)"
        self._command_label.config(
            text=f"Shota Complete — {winner} wins! ({t1}-{t2})", fg="#ffd54f")
        self._instruction.config(text="Shota done. Start next shota or reset.")
        self._next_trick_btn.config(state="disabled")

        # Replace next trick button with next shota button.
        for w in self._right.winfo_children():
            w.destroy()

        tk.Label(self._right, text="SHOTA COMPLETE",
                 font=("Segoe UI", 14, "bold"), fg=COLORS["gold"], bg="#252525"
                 ).pack(anchor="w", pady=(8, 4))
        tk.Label(self._right, text=f"{winner} — Tricks: {t1} vs {t2}",
                 font=("Segoe UI", 11), fg="#ffffff", bg="#252525"
                 ).pack(anchor="w", pady=(0, 8))
        tk.Label(self._right, text=f"Bid: {self.bid_value} by P{self.bid_winner_id + 1}",
                 font=("Segoe UI", 10), fg="#aaaaaa", bg="#252525"
                 ).pack(anchor="w", pady=(0, 12))

        tk.Button(self._right, text="▶  Next Shota", command=self._next_shota,
                  font=("Segoe UI", 11, "bold"), fg="#fff", bg=COLORS["btn_green"],
                  bd=0, padx=16, pady=8, cursor="hand2").pack(anchor="w", pady=(4, 0))
        tk.Button(self._right, text="↺ Reset All", command=self._reset_all,
                  font=("Segoe UI", 9), fg="#fff", bg=COLORS["btn_grey"],
                  bd=0, padx=8, pady=4, cursor="hand2").pack(anchor="w", pady=(12, 0))

    def _next_shota(self) -> None:
        """Start a new shota — clear hand, keep scores, rotate Qabool."""
        self.ai_hand.clear()
        self.trick_cards.clear()
        self.trump_suit = None
        self.trick_number = 0
        self.team_tricks = [0, 0]
        self.phase = "hand"
        self._known_voids = {0: set(), 1: set(), 2: set(), 3: set()}
        self._pending_offsuit = None

        # Rotate Qabool: P1→P2→P3→P4→P1.
        self.qabool_id = (self.qabool_id + 1) % 4

        # Reset card buttons.
        for card, btn in self._card_buttons.items():
            fg = "#c62828" if card.suit in (Suit.HEARTS, Suit.DIAMONDS) else "#303030"
            btn.config(bg=COLORS["card_bg"], fg=fg, relief="solid",
                       command=lambda c=card: self._card_clicked(c))

        player_names = {0: "AI", 1: "P2", 2: "P3", 3: "P4"}
        self._deck_label.config(text="AI HAND — Click 13 cards (0/13)")
        self._trump_display.config(text="")
        self._command_label.config(
            text=f"New Shota — Qabool: {player_names[self.qabool_id]}", fg="#ffd54f")
        self._instruction.config(text="Select your new 13 cards for this shota")
        self._build_right_panel()

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _get_agent(self):
        if self._agent is not None:
            return self._agent
        if self._agent_var.get() == "Learning":
            from agents.learning.learning_agent import LearningAgent
            self._agent = LearningAgent(training=False)
        else:
            from agents.rule_based.rule_based_agent import RuleBasedAgent
            self._agent = RuleBasedAgent()
        return self._agent

    def _random_hand(self) -> None:
        """Populate 13 random cards for quick testing."""
        import random as _rnd

        # Clear current hand.
        for card in list(self.ai_hand):
            btn = self._card_buttons[card]
            fg = "#c62828" if card.suit in (Suit.HEARTS, Suit.DIAMONDS) else "#303030"
            btn.config(bg=COLORS["card_bg"], fg=fg, relief="solid")
        self.ai_hand.clear()

        # Pick 13 random cards.
        all_cards = [Card(s, r) for s in ALL_SUITS for r in ALL_RANKS]
        hand = _rnd.sample(all_cards, 13)
        for card in hand:
            self.ai_hand.append(card)
            btn = self._card_buttons[card]
            btn.config(bg="#4caf50", fg="#ffffff", relief="raised")

        self._deck_label.config(text=f"AI HAND — Random 13 cards (13/13)")

    def _load_model(self) -> None:
        from agents.learning.learning_agent import LearningAgent
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")], title="Load AI Model")
        if path:
            self._agent = LearningAgent.load(path, training=False)
            self._agent_var.set("Learning")
            self._command_label.config(
                text=f"✓ Model loaded ({self._agent.q_table_size} entries)", fg="#66bb6a")

    def _reset_all(self) -> None:
        self.ai_hand.clear()
        self.trick_cards.clear()
        self.trump_suit = None
        self.trick_number = 0
        self.phase = "hand"
        self.team_tricks = [0, 0]
        self._known_voids = {0: set(), 1: set(), 2: set(), 3: set()}
        self._pending_offsuit = None
        self._agent = None

        for card, btn in self._card_buttons.items():
            fg = "#c62828" if card.suit in (Suit.HEARTS, Suit.DIAMONDS) else "#303030"
            btn.config(bg=COLORS["card_bg"], fg=fg, relief="solid",
                       command=lambda c=card: self._card_clicked(c))

        self._deck_label.config(text="AI HAND — Click 13 cards (0/13)")
        self._trump_display.config(text="")
        self._command_label.config(text="Select your hand first", fg="#4caf50")
        self._instruction.config(text="Step 1: Click your 13 cards from the deck below")
        self._build_right_panel()
