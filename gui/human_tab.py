"""
Human vs AI tab — Same table layout as Game Table, but:
- Player 3 (you) has clickable face-up cards
- Other players' cards are face-down
- Shows Qabool, trump, bid, trick info
- You click cards to play when it's your turn
"""

import tkinter as tk
from collections import Counter

from gui.colors import COLORS
from gui.card_widget import (
    draw_card, draw_card_back, parse_card_text,
    CARD_MINI_WIDTH, CARD_MINI_HEIGHT, CARD_LARGE_WIDTH, CARD_LARGE_HEIGHT,
    CARD_WIDTH, CARD_HEIGHT, CARD_HIGHLIGHT
)
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
HUMAN_ID = 2  # Player 3, Team 1.


def card_str(card: Card) -> str:
    return f"{RANK_SYMBOLS[card.rank]}{SUIT_SYMBOLS[card.suit]}"


class HumanTab:
    """Human vs AI — full table layout with face-down opponents."""

    def __init__(self, parent: tk.Frame, root: tk.Tk) -> None:
        self.parent = parent
        self.root = root
        self.game_running = False

        # Game state.
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

        self._build()

    def _build(self) -> None:
        self.parent.configure(bg=COLORS["table_border"])

        # Top info bar (mirrors Game Table top bar).
        info_bar = tk.Frame(self.parent, bg=COLORS["header_bg"], height=50)
        info_bar.pack(fill="x")
        info_bar.pack_propagate(False)

        self._info_labels = {}
        info_items = [
            ("Shota", "shota"), ("Deal", "deal"), ("Qabool", "qabool"),
            ("Bid", "bid"), ("Shooter", "shooter"), ("Trump", "trump"),
            ("Trick", "trick"), ("T1 Won", "t1_won"), ("T2 Won", "t2_won"),
        ]

        row = tk.Frame(info_bar, bg=COLORS["header_bg"])
        row.pack(anchor="center", pady=8)

        for label, key in info_items:
            f = tk.Frame(row, bg=COLORS["header_bg"])
            f.pack(side="left", padx=8)
            tk.Label(f, text=label, font=("Segoe UI", 7), fg="#666666",
                     bg=COLORS["header_bg"]).pack()
            val = tk.Label(f, text="—", font=("Segoe UI", 10, "bold"),
                           fg=COLORS["gold"], bg=COLORS["header_bg"], width=7, anchor="center")
            val.pack()
            self._info_labels[key] = val

        # Table area.
        table = tk.Frame(self.parent, bg=COLORS["table_felt"], bd=3, relief="ridge")
        table.pack(fill="both", expand=True, padx=8, pady=4)
        table.columnconfigure(0, weight=1, minsize=160)
        table.columnconfigure(1, weight=0, minsize=280)
        table.columnconfigure(2, weight=1, minsize=160)
        table.rowconfigure(0, weight=0, minsize=70)
        table.rowconfigure(1, weight=1, minsize=190)
        table.rowconfigure(2, weight=0, minsize=80)

        # Player areas (P1=top, P4=left, P2=right, P3=bottom=YOU).
        self._player_frames = {}
        self._player_canvases = {}
        self._player_status = {}

        self._create_opponent_area(table, 0, "Player 1 (AI)", "Team 1", row=0, col=1)
        self._create_opponent_area(table, 3, "Player 4 (AI)", "Team 2", row=1, col=0)
        self._create_opponent_area(table, 1, "Player 2 (AI)", "Team 2", row=1, col=2)

        # Player 3 = YOU (bottom).
        self._create_human_area(table)

        # Centre trick canvas — fixed size, centred, subtle border.
        centre_frame = tk.Frame(table, bg="#0d2e0d", bd=2, relief="groove")
        centre_frame.grid(row=1, column=1, padx=4, pady=4)
        self._centre_canvas = tk.Canvas(centre_frame, bg=COLORS["centre_bg"],
                                        width=270, height=190, highlightthickness=0)
        self._centre_canvas.pack(padx=2, pady=2)

        # Status bar (between table and controls — clearly visible).
        status_bar = tk.Frame(self.parent, bg="#1a3a1a", height=30)
        status_bar.pack(fill="x", padx=8, pady=(2, 0))
        status_bar.pack_propagate(False)

        self._status_label = tk.Label(status_bar, text="Press Start Game",
                                      font=("Segoe UI", 10, "bold"),
                                      fg=COLORS["gold"], bg="#1a3a1a")
        self._status_label.pack(expand=True)

        # Controls.
        ctrl = tk.Frame(self.parent, bg=COLORS["header_bg"], height=40)
        ctrl.pack(fill="x")
        ctrl.pack_propagate(False)

        bf = tk.Frame(ctrl, bg=COLORS["header_bg"])
        bf.pack(anchor="center", pady=6)

        tk.Button(bf, text="▶ Start Game", command=self._start_game,
                  font=("Segoe UI", 9, "bold"), fg="#fff", bg=COLORS["btn_green"],
                  bd=0, padx=14, pady=3, cursor="hand2").pack(side="left", padx=4)
        tk.Button(bf, text="⏹ Stop", command=self._stop_game,
                  font=("Segoe UI", 9, "bold"), fg="#fff", bg=COLORS["btn_red"],
                  bd=0, padx=14, pady=3, cursor="hand2").pack(side="left", padx=4)

    def _create_opponent_area(self, parent, pid, name, team, row, col):
        """Create an opponent player area — no dark box, just on the felt."""
        # Use sticky to centre: top/bottom=ew (horizontal centre), left/right=ns (vertical centre).
        sticky = "ew" if row in (0, 2) else "ns"

        frame = tk.Frame(parent, bg=COLORS["table_felt"], padx=4, pady=3)
        frame.grid(row=row, column=col, sticky=sticky, padx=3, pady=3)

        team_color = COLORS["score_team1"] if "1" in team else COLORS["score_team2"]
        header = tk.Frame(frame, bg=COLORS["table_felt"])
        header.pack(fill="x", pady=(0, 2))
        tk.Label(header, text=name, font=("Segoe UI", 9, "bold"),
                 fg="#ffffff", bg=COLORS["table_felt"]).pack(side="left")
        tk.Label(header, text=f"  ({team})", font=("Segoe UI", 8),
                 fg=team_color, bg=COLORS["table_felt"]).pack(side="left")

        status = tk.Label(frame, text="", font=("Segoe UI", 8),
                          fg=COLORS["text_muted"], bg=COLORS["table_felt"], anchor="w")
        status.pack(fill="x", pady=(0, 2))
        self._player_status[pid] = status

        # Side players (P2, P4) need more vertical space to centre with the middle.
        canvas_height = CARD_MINI_HEIGHT + 6 if row in (0, 2) else CARD_MINI_HEIGHT + 30

        canvas = tk.Canvas(frame, bg=COLORS["table_felt"], height=canvas_height,
                           highlightthickness=0)
        canvas.pack(fill="x", pady=2, expand=True)
        self._player_canvases[pid] = canvas
        self._player_frames[pid] = frame

    def _create_human_area(self, parent):
        """Create the human player area (bottom, face-up, clickable)."""
        frame = tk.Frame(parent, bg=COLORS["table_felt"], padx=6, pady=4)
        frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 4))

        header = tk.Frame(frame, bg=COLORS["table_felt"])
        header.pack(fill="x", pady=(0, 4))
        tk.Label(header, text="🧑 YOU — Player 3",
                 font=("Segoe UI", 10, "bold"),
                 fg=COLORS["gold"], bg=COLORS["table_felt"]).pack(side="left")
        tk.Label(header, text="  (Team 1)",
                 font=("Segoe UI", 9),
                 fg=COLORS["score_team1"], bg=COLORS["table_felt"]).pack(side="left")

        self._human_canvas = tk.Canvas(frame, bg=COLORS["table_felt"],
                                       height=CARD_HEIGHT + 8, highlightthickness=0)
        self._human_canvas.pack(fill="x")

        # Reserve space for bid buttons so layout doesn't jump.
        self._bid_placeholder = tk.Frame(frame, bg=COLORS["table_felt"], height=60)
        self._bid_placeholder.pack(fill="x")
        self._bid_placeholder.pack_propagate(False)
        self._player_frames[HUMAN_ID] = frame

    # ----------------------------------------------------------
    # Display helpers
    # ----------------------------------------------------------

    def _show_opponent_cards(self, pid):
        """Show face-down cards for an opponent, centred appropriately."""
        canvas = self._player_canvases[pid]
        canvas.delete("all")
        if self.players is None:
            return
        count = len(self.players[pid].hand)
        if count == 0:
            return

        # P1 (top) and P3 would be horizontal — but P3 is human.
        # P2 (right) and P4 (left) are side players — show cards horizontally but centred.
        spacing = min(14, max(8, 150 // max(count, 1)))
        total_width = (count - 1) * spacing + CARD_MINI_WIDTH
        canvas_w = canvas.winfo_width() or 160
        canvas_h = canvas.winfo_height() or (CARD_MINI_HEIGHT + 6)
        start_x = max(2, (canvas_w - total_width) // 2)
        start_y = max(2, (canvas_h - CARD_MINI_HEIGHT) // 2)

        for i in range(count):
            draw_card_back(canvas, start_x + i * spacing, start_y,
                           width=CARD_MINI_WIDTH, height=CARD_MINI_HEIGHT)

    def _show_human_hand(self, clickable=False):
        """Show the human player's hand as face-up cards."""
        canvas = self._human_canvas
        canvas.delete("all")

        if self.players is None:
            return

        hand = self.players[HUMAN_ID].hand
        if not hand:
            canvas.create_text(100, 30, text="No cards left", fill="#888888",
                               font=("Segoe UI", 9))
            return

        # Get legal cards.
        legal = set(hand)
        if clickable and self.round and self.round.state.current_trick:
            leading_suit = self.round.state.current_trick.leading_suit
            must_trump = None
            if (self.round.state.is_first_trick and
                    self.round.state.winning_bidder_id == HUMAN_ID and
                    len(self.round.state.current_trick.played_cards) == 0):
                must_trump = self.trump_suit
            legal = set(legal_cards(hand, leading_suit, must_trump))

        # Sort by suit then rank.
        suit_order = {Suit.SPADES: 0, Suit.HEARTS: 1, Suit.CLUBS: 2, Suit.DIAMONDS: 3}
        sorted_hand = sorted(hand, key=lambda c: (suit_order[c.suit], -rank_value(c.rank)))

        spacing = min(CARD_WIDTH + 4, max(28, (canvas.winfo_width() or 600) // max(len(sorted_hand) + 3, 1)))
        suit_gap = 12  # Extra gap between suits.

        # Calculate total width to centre the hand.
        total_width = 0
        prev_s = None
        for card in sorted_hand:
            if prev_s is not None and card.suit != prev_s:
                total_width += suit_gap
            total_width += spacing
            prev_s = card.suit
        total_width = total_width - spacing + CARD_WIDTH  # Last card full width.

        canvas_w = canvas.winfo_width() or 600
        current_x = max(6, (canvas_w - total_width) // 2)
        prev_suit = None

        for i, card in enumerate(sorted_hand):
            # Add spacer between different suits.
            if prev_suit is not None and card.suit != prev_suit:
                current_x += suit_gap
            prev_suit = card.suit

            x = current_x
            y = 4
            ct = card_str(card)
            rank, suit = parse_card_text(ct)

            is_legal = card in legal
            highlight = CARD_HIGHLIGHT if (clickable and is_legal) else None
            faded = clickable and not is_legal

            tag = f"hcard_{i}"
            draw_card(canvas, x, y, rank, suit,
                      width=CARD_WIDTH, height=CARD_HEIGHT,
                      highlight=highlight, faded=faded, tag=tag)

            if clickable and is_legal:
                canvas.tag_bind(tag, "<Button-1>", lambda e, c=card: self._human_play(c))

            current_x += spacing

    def _draw_centre_trick(self):
        """Draw played cards in the centre — fixed positions, no jumping."""
        canvas = self._centre_canvas
        canvas.delete("all")

        # Fixed canvas size.
        w = 270
        h = 190
        cw, ch = CARD_LARGE_WIDTH, CARD_LARGE_HEIGHT

        # Fixed positions for each player's card.
        positions = {
            0: (w // 2 - cw // 2, 4),                   # P1 top centre
            1: (w - cw - 10, h // 2 - ch // 2),        # P2 right
            2: (w // 2 - cw // 2, h - ch - 4),         # P3 bottom centre
            3: (10, h // 2 - ch // 2),                  # P4 left
        }

        for pid, (x, y) in positions.items():
            if hasattr(self, "_trick_played") and pid in self._trick_played:
                ct = self._trick_played[pid]
                rank, suit = parse_card_text(ct)
                draw_card(canvas, x, y, rank, suit, width=cw, height=ch)
            else:
                canvas.create_rectangle(x, y, x + cw, y + ch,
                                        fill="#2a4a2a", outline="#3a5a3a", dash=(3, 3))
                canvas.create_text(x + cw // 2, y + ch // 2,
                                   text=f"P{pid+1}", fill="#5a8a5a", font=("Segoe UI", 8))

    def _update_info(self, **kw):
        for key, val in kw.items():
            if key in self._info_labels:
                self._info_labels[key].config(text=str(val) if val else "—")

    def _set_status(self, text):
        self._status_label.config(text=text)

    # ----------------------------------------------------------
    # Game flow
    # ----------------------------------------------------------

    def _start_game(self):
        if self.game_running:
            return

        # Reset everything visually.
        self._reset_table()

        self.game_running = True
        self.trick_number = 0
        self.team_tricks = [0, 0]
        self._trick_played = {}
        self._human_chosen_trump = None
        self._human_trump_choice = None

        self.players = create_standard_players()
        self.round = Round(self.players)
        self.round.deal()

        # Redeal on Dak.
        attempts = 0
        while self.round.has_card_based_dak() and attempts < 10:
            self.round = Round(self.players)
            self.round.deal()
            attempts += 1

        self.agents = [RuleBasedAgent(), RuleBasedAgent(), None, RuleBasedAgent()]
        self.qabool_id = determine_first_shota_qabool()

        self._update_info(shota="1", deal="1", qabool=f"P{self.qabool_id+1}")
        self._show_all_hands()
        self._set_status("Bidding...")

        self.root.after(500, self._run_bidding)

    def _stop_game(self):
        self.game_running = False
        self._set_status("Stopped")

    def _show_all_hands(self):
        """Show all opponent hands face-down, human hand face-up."""
        for pid in [0, 1, 3]:
            self._show_opponent_cards(pid)
        self._show_human_hand(clickable=False)

    def _run_bidding(self):
        """Run bidding step by step — show each bid, let human participate."""
        if not self.game_running:
            return

        from environments.wist.tasmiya_engine import tasmiya_order
        from environments.wist.observation import BiddingObservation
        from environments.wist.bidding_engine import BiddingEngine
        from environments.wist.bidding import Bid, Pass, validate_opening_bid, validate_regular_bid

        self._bidding_engine = BiddingEngine()
        self._bid_history = []
        self._bid_order = tasmiya_order(self.qabool_id)
        self._bid_index = 0
        self._has_opening_bid = False

        self._set_status(f"Al-Tasmiya — Sahib Al-Qabool: P{self.qabool_id+1}")
        self.root.after(800, self._bid_next_player)

    def _bid_next_player(self):
        """Process the next player's bid."""
        if not self.game_running:
            return

        if self._bid_index >= len(self._bid_order):
            # All 3 players bid — now Qabool decides.
            self._bid_qabool_turn()
            return

        pid = self._bid_order[self._bid_index]

        if pid == HUMAN_ID:
            # Human's turn to bid.
            self._show_human_bid_options()
        else:
            # AI bids.
            from environments.wist.observation import BiddingObservation
            obs = BiddingObservation(
                player_id=pid,
                hand=list(self.players[pid].hand),
                previous_bids=list(self._bid_history),
                current_highest_bid=(
                    self._bidding_engine.highest_bid.value
                    if self._bidding_engine.highest_bid else None),
                is_sahib_al_qabool=False,
                is_opening_bid=(not self._has_opening_bid),
            )
            action = self.agents[pid].act(obs)

            from environments.wist.actions import BidAction, PassAction
            from environments.wist.bidding import Bid, Pass

            if isinstance(action, BidAction):
                bid = Bid(player_id=pid, value=action.value)
                self._bidding_engine.apply_bid(bid)
                self._bid_history.append((pid, action.value))
                self._has_opening_bid = True
                self._player_status[pid].config(
                    text=f"Bid: {action.value}", fg=COLORS["gold"])
                self._set_status(f"Player {pid+1} bids {action.value}")
            else:
                from environments.wist.bidding import Pass
                self._bidding_engine.apply_pass(Pass(player_id=pid))
                self._bid_history.append((pid, None))
                self._player_status[pid].config(text="Pass", fg=COLORS["text_muted"])
                self._set_status(f"Player {pid+1} passes")

            self._bid_index += 1
            self.root.after(800, self._bid_next_player)

    def _show_human_bid_options(self):
        """Show trump + bid selection with proper constraints and confirm."""
        self._set_status("YOUR TURN! Pick trump suit → bid is auto-calculated → Confirm or Pass.")

        # Keep the hand visible!
        self._show_human_hand(clickable=False)

        # Show bid controls inside the reserved placeholder (no jumping).
        if hasattr(self, "_bid_btn_frame") and self._bid_btn_frame:
            self._bid_btn_frame.destroy()

        self._bid_btn_frame = tk.Frame(self._bid_placeholder, bg=COLORS["table_felt"])
        self._bid_btn_frame.pack(fill="both", expand=True)

        self._human_trump_choice = None
        self._human_bid_value = None

        # Row 1: Trump selection (centred).
        trump_row = tk.Frame(self._bid_btn_frame, bg=COLORS["table_felt"])
        trump_row.pack(pady=(4, 2), fill="x")
        # Inner container to centre contents.
        trump_inner = tk.Frame(trump_row, bg=COLORS["table_felt"])
        trump_inner.pack(expand=True)

        tk.Label(trump_inner, text="Trump:",
                 font=("Segoe UI", 9, "bold"), fg=COLORS["gold"],
                 bg=COLORS["table_felt"]).pack(side="left", padx=(4, 8))

        self._trump_buttons = {}
        for suit in [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]:
            sym = SUIT_SYMBOLS[suit]
            count = sum(1 for c in self.players[HUMAN_ID].hand if c.suit == suit)
            fg = "#c62828" if suit in (Suit.HEARTS, Suit.DIAMONDS) else "#1a1a1a"
            btn = tk.Button(trump_inner, text=f"{sym}({count})",
                            font=("Consolas", 11, "bold"), fg=fg,
                            bg=COLORS["card_bg"], bd=1, padx=5, pady=1,
                            cursor="hand2",
                            command=lambda s=suit, c=count: self._select_trump_with_bid(s, c))
            btn.pack(side="left", padx=3)
            self._trump_buttons[suit] = btn

        # Row 2: Result + Confirm / Pass / Dak (centred).
        action_row = tk.Frame(self._bid_btn_frame, bg=COLORS["table_felt"])
        action_row.pack(pady=(2, 4), fill="x")
        action_inner = tk.Frame(action_row, bg=COLORS["table_felt"])
        action_inner.pack(expand=True)

        self._bid_result_label = tk.Label(action_inner, text="Select a trump suit above",
                                          font=("Segoe UI", 9),
                                          fg=COLORS["text_muted"], bg=COLORS["table_felt"])
        self._bid_result_label.pack(side="left", padx=(4, 12))

        self._confirm_btn = tk.Button(action_inner, text="✓ Confirm Bid",
                                      font=("Segoe UI", 9, "bold"), fg="#fff",
                                      bg=COLORS["btn_green"], bd=0, padx=10, pady=3,
                                      cursor="hand2", state="disabled",
                                      command=self._confirm_human_bid)
        self._confirm_btn.pack(side="left", padx=4)

        tk.Button(action_inner, text="Pass",
                  font=("Segoe UI", 9, "bold"), fg="#fff",
                  bg=COLORS["btn_grey"], bd=0, padx=10, pady=3,
                  cursor="hand2", command=lambda: self._human_bid(None)
                  ).pack(side="left", padx=4)

        tk.Button(action_inner, text="Dak",
                  font=("Segoe UI", 9, "bold"), fg="#fff",
                  bg=COLORS["btn_red"], bd=0, padx=10, pady=3,
                  cursor="hand2", command=lambda: self._human_bid(None)
                  ).pack(side="left", padx=4)

    def _select_trump_with_bid(self, suit, count):
        """Human selects trump — auto-calculates the bid."""
        self._human_trump_choice = suit
        bid_value = count + 3
        bid_value = max(7, min(bid_value, 13))

        # Check if this bid is valid.
        current_highest = (self._bidding_engine.highest_bid.value
                           if self._bidding_engine.highest_bid else None)

        valid = True
        reason = ""
        if count < 4:
            valid = False
            reason = f"Need 4+ cards (you have {count})"
        elif not self._has_opening_bid and bid_value > 11:
            valid = False
            reason = "Opening bid cannot exceed 11"
        elif current_highest and bid_value <= current_highest:
            valid = False
            reason = f"Must beat current bid ({current_highest})"

        # Highlight selected trump button.
        for s, btn in self._trump_buttons.items():
            if s == suit:
                btn.config(relief="sunken", bg="#a5d6a7")
            else:
                btn.config(relief="raised", bg=COLORS["card_bg"])

        sym = SUIT_SYMBOLS[suit]
        if valid:
            self._human_bid_value = bid_value
            self._bid_result_label.config(
                text=f"→ {suit.name} {sym} | Bid: {bid_value}",
                fg=COLORS["gold"])
            self._confirm_btn.config(state="normal")
        else:
            self._human_bid_value = None
            self._bid_result_label.config(
                text=f"→ {suit.name} {sym} | ✗ {reason}",
                fg="#ff6666")
            self._confirm_btn.config(state="disabled")

    def _confirm_human_bid(self):
        """Human confirms their bid + trump selection."""
        if self._human_trump_choice is None or self._human_bid_value is None:
            return
        self._human_chosen_trump = self._human_trump_choice
        self._human_bid(self._human_bid_value)

    def _select_trump(self, suit):
        """Legacy — replaced by _select_trump_with_bid."""
        pass

    def _human_bid_with_trump(self, value):
        """Legacy — replaced by _confirm_human_bid."""
        pass

    def _human_bid(self, value):
        """Handle human's bid decision."""
        from environments.wist.bidding import Bid, Pass

        # Remove bid buttons.
        if hasattr(self, "_bid_btn_frame"):
            self._bid_btn_frame.destroy()

        if value is not None:
            bid = Bid(player_id=HUMAN_ID, value=value)
            self._bidding_engine.apply_bid(bid)
            self._bid_history.append((HUMAN_ID, value))
            self._has_opening_bid = True
            self._set_status(f"You bid {value}")
        else:
            self._bidding_engine.apply_pass(Pass(player_id=HUMAN_ID))
            self._bid_history.append((HUMAN_ID, None))
            self._set_status("You pass")

        self._bid_index += 1
        self._show_human_hand(clickable=False)
        self.root.after(600, self._bid_next_player)

    def _bid_qabool_turn(self):
        """Sahib Al-Qabool makes the final decision."""
        from environments.wist.observation import BiddingObservation
        from environments.wist.actions import BidAction, PassAction
        from environments.wist.bidding import Bid, Pass, validate_sahib_al_qabool_bid

        qid = self.qabool_id

        if qid == HUMAN_ID:
            # Human is Qabool — show accept/match options.
            self._show_human_qabool_options()
            return

        # AI Qabool decides.
        obs = BiddingObservation(
            player_id=qid,
            hand=list(self.players[qid].hand),
            previous_bids=list(self._bid_history),
            current_highest_bid=(
                self._bidding_engine.highest_bid.value
                if self._bidding_engine.highest_bid else None),
            is_sahib_al_qabool=True,
            is_opening_bid=(not self._has_opening_bid),
        )
        action = self.agents[qid].act(obs)

        if isinstance(action, BidAction):
            bid = Bid(player_id=qid, value=action.value)
            self._bidding_engine.apply_bid(bid, is_sahib_al_qabool=True)
            self._bid_history.append((qid, action.value))
            self._player_status[qid].config(
                text=f"Bid: {action.value} (Qabool)", fg=COLORS["gold"])
            self._set_status(f"Sahib Al-Qabool (P{qid+1}) bids {action.value}")
        else:
            self._bidding_engine.apply_pass(Pass(player_id=qid))
            self._bid_history.append((qid, None))
            all_passed = self._bidding_engine.highest_bid is None
            if all_passed:
                self._set_status("All passed — Dak!")
                self.game_running = False
                self.root.after(1000, self._start_game)
                return
            self._player_status[qid].config(text="Accepts", fg=COLORS["text_muted"])
            self._set_status(f"Sahib Al-Qabool accepts")

        self.root.after(800, self._finalize_bidding)

    def _show_human_qabool_options(self):
        """Show Qabool options with trump selection + confirm."""
        current = self._bidding_engine.highest_bid
        if current:
            self._set_status(f"You are Qabool! Current bid: {current.value}. Pick trump + bid, or Accept.")
        else:
            self._set_status("You are Qabool! Everyone passed. Pick trump + bid, or Dak.")

        self._show_human_hand(clickable=False)

        if hasattr(self, "_bid_btn_frame"):
            self._bid_btn_frame.destroy()

        self._bid_btn_frame = tk.Frame(self._bid_placeholder, bg=COLORS["table_felt"])
        self._bid_btn_frame.pack(fill="both", expand=True)

        self._human_trump_choice = None
        self._human_bid_value = None

        # Row 1: Trump selection with card counts.
        trump_row = tk.Frame(self._bid_btn_frame, bg=COLORS["table_felt"])
        trump_row.pack(fill="x", pady=(4, 2))
        trump_inner = tk.Frame(trump_row, bg=COLORS["table_felt"])
        trump_inner.pack(expand=True)

        tk.Label(trump_inner, text="Trump:",
                 font=("Segoe UI", 9, "bold"), fg=COLORS["gold"],
                 bg=COLORS["table_felt"]).pack(side="left", padx=(4, 8))

        self._trump_buttons = {}
        for suit in [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]:
            sym = SUIT_SYMBOLS[suit]
            count = sum(1 for c in self.players[HUMAN_ID].hand if c.suit == suit)
            fg = "#c62828" if suit in (Suit.HEARTS, Suit.DIAMONDS) else "#1a1a1a"
            btn = tk.Button(trump_inner, text=f"{sym}({count})",
                            font=("Consolas", 11, "bold"), fg=fg,
                            bg=COLORS["card_bg"], bd=1, padx=5, pady=1,
                            cursor="hand2",
                            command=lambda s=suit, c=count: self._select_qabool_trump(s, c))
            btn.pack(side="left", padx=3)
            self._trump_buttons[suit] = btn

        # Row 2: Result + Confirm/Accept/Dak.
        action_row = tk.Frame(self._bid_btn_frame, bg=COLORS["table_felt"])
        action_row.pack(fill="x", pady=(2, 4))
        action_inner = tk.Frame(action_row, bg=COLORS["table_felt"])
        action_inner.pack(expand=True)

        self._bid_result_label = tk.Label(action_inner, text="Select a trump suit above",
                                          font=("Segoe UI", 9),
                                          fg=COLORS["text_muted"], bg=COLORS["table_felt"])
        self._bid_result_label.pack(side="left", padx=(4, 12))

        self._confirm_btn = tk.Button(action_inner, text="✓ Confirm Bid",
                                      font=("Segoe UI", 9, "bold"), fg="#fff",
                                      bg=COLORS["btn_green"], bd=0, padx=10, pady=3,
                                      cursor="hand2", state="disabled",
                                      command=self._confirm_human_qabool_bid)
        self._confirm_btn.pack(side="left", padx=4)

        if current:
            tk.Button(action_inner, text="Accept",
                      font=("Segoe UI", 9, "bold"), fg="#fff",
                      bg=COLORS["btn_grey"], bd=0, padx=10, pady=3,
                      cursor="hand2", command=lambda: self._human_bid(None)
                      ).pack(side="left", padx=4)
        else:
            tk.Button(action_inner, text="Dak",
                      font=("Segoe UI", 9, "bold"), fg="#fff",
                      bg=COLORS["btn_red"], bd=0, padx=10, pady=3,
                      cursor="hand2", command=lambda: self._human_bid(None)
                      ).pack(side="left", padx=4)

    def _select_qabool_trump(self, suit, count):
        """Qabool selects trump — with advantage (count+2, min 7)."""
        self._human_trump_choice = suit
        # Qabool advantage: can bid one lower.
        bid_value = max(7, count + 2)
        bid_value = min(bid_value, 13)

        for s, btn in self._trump_buttons.items():
            if s == suit:
                btn.config(relief="sunken", bg="#a5d6a7")
            else:
                btn.config(relief="raised", bg=COLORS["card_bg"])

        sym = SUIT_SYMBOLS[suit]
        current_highest = (self._bidding_engine.highest_bid.value
                           if self._bidding_engine.highest_bid else None)

        # Qabool can match (doesn't need to go higher).
        valid = count >= 4
        if valid:
            if current_highest:
                bid_value = max(bid_value, current_highest)  # At least match.
            self._human_bid_value = bid_value
            self._bid_result_label.config(
                text=f"→ {suit.name} {sym} | Bid: {bid_value} (advantage)",
                fg=COLORS["gold"])
            self._confirm_btn.config(state="normal")
        else:
            self._human_bid_value = None
            self._bid_result_label.config(
                text=f"→ {suit.name} {sym} | ✗ Need 4+ cards",
                fg="#ff6666")
            self._confirm_btn.config(state="disabled")

    def _confirm_human_qabool_bid(self):
        """Confirm Qabool bid."""
        if self._human_trump_choice is None or self._human_bid_value is None:
            return
        self._human_chosen_trump = self._human_trump_choice
        self._human_qabool_bid(self._human_bid_value)

    def _human_qabool_bid_with_trump(self, value):
        """Legacy — replaced."""
        pass

    def _human_qabool_bid(self, value):
        """Human Qabool bids."""
        from environments.wist.bidding import Bid

        # Remove bid buttons.
        if hasattr(self, "_bid_btn_frame"):
            self._bid_btn_frame.destroy()

        bid = Bid(player_id=HUMAN_ID, value=value)
        self._bidding_engine.apply_bid(bid, is_sahib_al_qabool=True)
        self._bid_history.append((HUMAN_ID, value))
        self._set_status(f"You bid {value} as Qabool!")
        self._show_human_hand(clickable=False)
        self.root.after(600, self._finalize_bidding)

    def _finalize_bidding(self):
        """Finalize bidding and start play."""
        winning_bid = self._bidding_engine.highest_bid
        if winning_bid is None:
            self._set_status("Dak!")
            self.game_running = False
            self.root.after(1000, self._start_game)
            return

        self.shooter_id = winning_bid.player_id
        self.bid_value = winning_bid.value

        # Use human's chosen trump if human won the bid.
        if self.shooter_id == HUMAN_ID and hasattr(self, "_human_chosen_trump") and self._human_chosen_trump:
            self.trump_suit = self._human_chosen_trump
        else:
            self.trump_suit = determine_trump_suit(self.players[self.shooter_id].hand)

        self.round.state.trump_suit = self.trump_suit
        self.round.state.winning_bidder_id = self.shooter_id
        self.round.next_leading_player_id = self.shooter_id
        self.environment = WistEnvironment(self.round.state)

        trump_sym = SUIT_SYMBOLS.get(self.trump_suit, "?")
        self._update_info(
            bid=str(self.bid_value),
            trump=f"{self.trump_suit.name} {trump_sym}",
            shooter=f"P{self.shooter_id+1}",
        )

        for pid in [0, 1, 3]:
            s = self._player_status[pid]
            if pid == self.qabool_id:
                s.config(text="👑 Qabool", fg=COLORS["gold"])
            elif pid == self.shooter_id:
                s.config(text="🎯 Shooter", fg="#66bb6a")
            else:
                s.config(text="")

        self._set_status(f"Bidding done! P{self.shooter_id+1} plays. Trump revealed on first card.")
        self._show_all_hands()
        self.root.after(1000, self._play_next_trick)

    def _play_next_trick(self):
        if not self.game_running:
            return
        if self.trick_number >= 13:
            self._finish_game()
            return

        self.trick_number += 1
        self._trick_played = {}
        self._draw_centre_trick()
        self._update_info(trick=f"{self.trick_number}/13")

        leader = self.round.next_leading_player_id
        self.round.state.current_trick = Trick(leading_player_id=leader)
        self._play_order = [(leader + i) % 4 for i in range(4)]
        self._play_idx = 0

        self._set_status(f"Trick {self.trick_number} — P{leader+1} leads")
        self.root.after(400, self._play_next_card)

    def _play_next_card(self):
        if not self.game_running or self._play_idx >= 4:
            self.root.after(600, self._resolve_trick)
            return

        pid = self._play_order[self._play_idx]

        if pid == HUMAN_ID:
            self._set_status(f"Trick {self.trick_number} — YOUR TURN! Click a card.")
            self._show_human_hand(clickable=True)
        else:
            obs = self.environment.observe(pid)
            action = self.agents[pid].act(obs)
            self.environment.apply_action(action)
            self._play_idx += 1

            ct = card_str(action.card)
            self._trick_played[pid] = ct
            self._draw_centre_trick()
            self._show_opponent_cards(pid)

            self._set_status(f"Trick {self.trick_number} — P{pid+1} played {ct}")
            self.root.after(600, self._play_next_card)

    def _human_play(self, card: Card):
        if not self.game_running:
            return
        action = PlayCardAction(player_id=HUMAN_ID, card=card)
        self.environment.apply_action(action)
        self._play_idx += 1

        ct = card_str(card)
        self._trick_played[HUMAN_ID] = ct
        self._draw_centre_trick()
        self._show_human_hand(clickable=False)

        self._set_status(f"Trick {self.trick_number} — you played {ct}")
        self.root.after(600, self._play_next_card)

    def _resolve_trick(self):
        if not self.game_running:
            return
        trick = self.round.state.current_trick
        winner = trick_winner(trick, self.trump_suit)
        self.round.state.completed_tricks.append(trick)
        self.round.state.current_trick = None
        self.round.next_leading_player_id = winner

        team = 0 if winner in (0, 2) else 1
        self.team_tricks[team] += 1

        # Update trick counts in info bar.
        self._update_info(t1_won=str(self.team_tricks[0]), t2_won=str(self.team_tricks[1]))

        who = "YOU" if winner == HUMAN_ID else f"P{winner+1}"
        self._set_status(f"Trick {self.trick_number} — {who} won! (T1:{self.team_tricks[0]} T2:{self.team_tricks[1]})")

        # Update opponent status.
        for pid in [0, 1, 3]:
            if pid == winner:
                self._player_status[pid].config(text="🏆 Won", fg="#ffd54f")
            else:
                lbl = self._player_status[pid]
                if pid == self.qabool_id:
                    lbl.config(text="👑 Qabool", fg=COLORS["gold"])
                elif pid == self.shooter_id:
                    lbl.config(text="🎯 Shooter", fg="#66bb6a")
                else:
                    lbl.config(text="")

        self._show_all_hands()
        self.root.after(1200, self._play_next_trick)

    def _finish_game(self):
        self.game_running = False
        winner = "YOUR TEAM (Team 1)" if self.team_tricks[0] > self.team_tricks[1] else "Team 2"

        # Clear the centre and show winner announcement.
        canvas = self._centre_canvas
        canvas.delete("all")
        w = 270
        h = 190
        canvas.create_text(w // 2, h // 2 - 20,
                           text="🏆  GAME OVER  🏆",
                           fill=COLORS["gold"], font=("Segoe UI", 14, "bold"))
        canvas.create_text(w // 2, h // 2 + 10,
                           text=f"{winner} wins!",
                           fill="#ffffff", font=("Segoe UI", 12, "bold"))
        canvas.create_text(w // 2, h // 2 + 40,
                           text=f"Team 1: {self.team_tricks[0]}  │  Team 2: {self.team_tricks[1]}",
                           fill="#aaaaaa", font=("Segoe UI", 10))

        self._set_status("Game over! Press Start Game to play again.")

    def _reset_table(self):
        """Reset all visual elements for a fresh game."""
        self._centre_canvas.delete("all")
        for pid in [0, 1, 3]:
            if pid in self._player_canvases:
                self._player_canvases[pid].delete("all")
            if pid in self._player_status:
                self._player_status[pid].config(text="")
        self._human_canvas.delete("all")
        # Clear bid placeholder.
        if hasattr(self, "_bid_btn_frame") and self._bid_btn_frame:
            try:
                self._bid_btn_frame.destroy()
            except tk.TclError:
                pass
        for key in self._info_labels:
            self._info_labels[key].config(text="—")

    # Stub methods for interface compat.
    def set_status(self, *a): pass
    def set_trick_display(self, *a): pass
    def show_hand(self, *a, **kw): pass
    def show_bid_options(self, *a, **kw): pass
    def hide_bid_options(self): pass
