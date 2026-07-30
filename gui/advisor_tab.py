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

        self._qabool_var = tk.StringVar(value="P1 (You)")
        options = ["P1 (You)", "P2 (Left)", "P3 (Opposite)", "P4 (Right)"]
        tk.OptionMenu(self._right, self._qabool_var, *options).pack(anchor="w")

        # Start button.
        tk.Button(self._right, text="▶  Start Bidding", command=self._start_bidding,
                  font=("Segoe UI", 11, "bold"), fg="#fff", bg=COLORS["btn_green"],
                  bd=0, padx=16, pady=8, cursor="hand2").pack(anchor="w", pady=(20, 0))

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

        qabool_map = {"P1 (You)": 0, "P2 (Left)": 1, "P3 (Opposite)": 2, "P4 (Right)": 3}
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

        player_names = {0: "P1 (You)", 1: "P2 (Left)", 2: "P3 (Opposite)", 3: "P4 (Right)"}

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
        player_names = {0: "P1 (You)", 1: "P2 (Left)", 2: "P3 (Opposite)", 3: "P4 (Right)"}

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
        """Trump is set — start playing."""
        sym = SUIT_SYMBOLS[self.trump_suit]
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
        """Build the right panel for trick play."""
        for w in self._right.winfo_children():
            w.destroy()

        tk.Label(self._right, text="TRICK", font=("Segoe UI", 11, "bold"),
                 fg="#ffffff", bg="#252525").pack(anchor="w")

        self._trick_info = tk.Label(self._right, text="",
                                    font=("Segoe UI", 10), fg="#aaaaaa", bg="#252525")
        self._trick_info.pack(anchor="w", pady=(4, 8))

        # Cards played this trick.
        self._trick_display = tk.Label(self._right, text="",
                                       font=("Consolas", 12), fg="#ffffff", bg="#1a1a1a",
                                       padx=8, pady=8, anchor="w", justify="left")
        self._trick_display.pack(fill="x", pady=4)

        # Buttons.
        btn_row = tk.Frame(self._right, bg="#252525")
        btn_row.pack(fill="x", pady=(8, 0))

        self._next_trick_btn = tk.Button(
            btn_row, text="→ Next Trick", command=self._start_next_trick,
            font=("Segoe UI", 10, "bold"), fg="#fff", bg=COLORS["btn_green"],
            bd=0, padx=12, pady=5, cursor="hand2")
        self._next_trick_btn.pack(side="left", padx=3)
        self._next_trick_btn.config(state="disabled")

        # Score display.
        self._score_label = tk.Label(self._right, text="Score: T1: 0 | T2: 0",
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
        self._next_trick_btn.config(state="disabled")

        if self.trick_number > 13:
            self._end_game()
            return

        self._trick_info.config(text=f"Trick {self.trick_number}/13 — Leader: P{self.leader_id + 1}")

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
        self._update_trick_display()

        if pid == 0:
            # AI's turn — ask AI what to play.
            self._ai_play()
        else:
            # Other player's turn — wait for user to click their card.
            player_names = {0: "P1 (You)", 1: "P2 (Left)", 2: "P3 (Opposite)", 3: "P4 (Right)"}
            self._deck_label.config(text=f"Click the card P{pid+1} played")
            self._command_label.config(
                text=f"⏳ Waiting: What did {player_names[pid]} play?", fg="#ff9800")

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
            self._play_idx += 1
            self._deck_label.config(text=f"AI HAND — {len(self.ai_hand)} cards left")
            self._update_trick_display()

            # Auto-advance after short delay.
            self.root.after(500, self._advance_play)

    def _opponent_plays(self, card: Card) -> None:
        """An opponent played a card (user clicked it)."""
        if self.phase != "playing":
            return
        if self._play_idx >= 4:
            return

        pid = self._play_order[self._play_idx]
        if pid == 0:
            return  # AI plays itself.

        # Mark the card as played.
        btn = self._card_buttons[card]
        btn.config(bg="#1e88e5", fg="#ffffff", relief="sunken")

        self.trick_cards.append((pid, card))
        self._play_idx += 1
        self._update_trick_display()
        self._advance_play()

    def _resolve_trick(self) -> None:
        """Determine trick winner."""
        from environments.wist.trick import Trick
        from environments.wist.rules import trick_winner

        trick = Trick(leading_player_id=self._play_order[0])
        for pid, card in self.trick_cards:
            trick.play_card(pid, card)

        winner = trick_winner(trick, self.trump_suit)
        winner_team = 0 if winner in (0, 2) else 1
        self.team_tricks[winner_team] += 1

        # Grey out all opponent cards played this trick.
        for pid, card in self.trick_cards:
            if pid != 0:  # Not the AI's card (already greyed).
                btn = self._card_buttons[card]
                btn.config(bg="#666666", fg="#999999", relief="flat")

        player_names = {0: "P1 (You)", 1: "P2 (Left)", 2: "P3 (Opposite)", 3: "P4 (Right)"}
        self._command_label.config(
            text=f"🏆 Winner: {player_names[winner]}", fg="#ffd54f")
        self._score_label.config(
            text=f"Score: T1: {self.team_tricks[0]} | T2: {self.team_tricks[1]}")

        self.leader_id = winner
        self._next_trick_btn.config(state="normal")
        self._deck_label.config(text=f"AI HAND — {len(self.ai_hand)} cards left")

    def _update_trick_display(self) -> None:
        """Show cards played in current trick with leader indicator."""
        player_names = {0: "P1", 1: "P2", 2: "P3", 3: "P4"}
        lines = []
        leader = self._play_order[0] if hasattr(self, '_play_order') and self._play_order else None
        for pid, card in self.trick_cards:
            prefix = "→ " if pid == leader else "  "
            lines.append(f"{prefix}{player_names[pid]}: {card_str(card)}")
        if not lines:
            leader_name = player_names.get(self.leader_id, "?")
            lines.append(f"  Leader: {leader_name} (waiting...)")
        self._trick_display.config(text="\n".join(lines))

    def _end_game(self) -> None:
        """Show end-of-shota summary."""
        self.phase = "done"
        t1 = self.team_tricks[0]
        t2 = self.team_tricks[1]
        winner = "Your Team (T1)" if t1 > t2 else "Opponent Team (T2)"
        self._command_label.config(
            text=f"GAME OVER — {winner} wins! ({t1}-{t2})", fg="#ffd54f")
        self._instruction.config(text="Shota complete. Click Reset to play again.")
        self._next_trick_btn.config(state="disabled")

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
        self._agent = None

        for card, btn in self._card_buttons.items():
            fg = "#c62828" if card.suit in (Suit.HEARTS, Suit.DIAMONDS) else "#303030"
            btn.config(bg=COLORS["card_bg"], fg=fg, relief="solid",
                       command=lambda c=card: self._card_clicked(c))

        self._deck_label.config(text="AI HAND — Click 13 cards (0/13)")
        self._command_label.config(text="Select your hand first", fg="#4caf50")
        self._instruction.config(text="Step 1: Click your 13 cards from the deck below")
        self._build_right_panel()
