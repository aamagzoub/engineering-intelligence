"""
Human vs AI tab — Full interactive game with:
- Face-down opponent cards, face-up human hand
- Step-by-step bidding with trump selection
- Visual turn indicators, trick winner highlights
- Trump hidden until first card played
- Score history, game log, Dak ceremony
- Multi-shota game flow (5 Shotas)
"""

import tkinter as tk

from gui_wist_lab.colors import COLORS
from gui_wist_lab.card_widget import (
    draw_card, draw_card_back, parse_card_text,
    CARD_MINI_WIDTH, CARD_MINI_HEIGHT, CARD_LARGE_WIDTH, CARD_LARGE_HEIGHT,
    CARD_WIDTH, CARD_HEIGHT, CARD_HIGHLIGHT
)
from agents.wist_rule_based.rule_based_agent import RuleBasedAgent
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
HUMAN_ID = 2  # Internal index 2 = Team 1 (you).
DISPLAY_NAMES = {2: "P1 (You)", 1: "P2", 0: "P3", 3: "P4"}


def card_str(card: Card) -> str:
    return f"{RANK_SYMBOLS[card.rank]}{SUIT_SYMBOLS[card.suit]}"


class HumanTab:
    """Human vs AI — full table layout with all game features."""

    def __init__(self, parent: tk.Frame, root: tk.Tk) -> None:
        self.parent = parent
        self.root = root
        self.game_running = False
        self._loaded_learning_agent = None
        self._bid_btn_frame = None
        self._confirm_frame = None
        self._trick_played = {}

        # Player evaluation system.
        from gui_wist_lab.player_evaluator import PlayerEvaluator
        self._evaluator = PlayerEvaluator()

        # Game state.
        self.players = None
        self.round = None
        self.environment = None
        self.agents = None
        self.trump_suit = None
        self.trump_revealed = False
        self.qabool_id = 0
        self.shooter_id = 0
        self.bid_value = 0
        self.trick_number = 0
        self.team_tricks = [0, 0]
        self._active_player_id = None
        self._turn_indicator_active = False
        self._shota_finishing = False

        # Game-level (across 5 Shotas).
        self.game_scores = [0, 0]
        self.shota_number = 0
        self.playing_team_id = None
        self._shota_history = []
        self._game_log = []

        self._build()

    def _build(self) -> None:
        self.parent.configure(bg=COLORS["table_border"])

        # Top info bar — FIXED height.
        info_bar = tk.Frame(self.parent, bg=COLORS["header_bg"], height=50)
        info_bar.pack(fill="x")
        info_bar.pack_propagate(False)

        self._info_labels = {}
        info_items = [
            ("Shota", "shota"), ("Qabool", "qabool"),
            ("Bid", "bid"), ("Shooter", "shooter"), ("Trump", "trump"),
            ("Trick", "trick"), ("T1 Won", "t1_won"), ("T2 Won", "t2_won"),
            ("Score", "score"),
        ]

        row = tk.Frame(info_bar, bg=COLORS["header_bg"])
        row.pack(anchor="center", pady=8)

        for label, key in info_items:
            f = tk.Frame(row, bg=COLORS["header_bg"])
            f.pack(side="left", padx=6)
            tk.Label(f, text=label, font=("Segoe UI", 7), fg="#666666",
                     bg=COLORS["header_bg"]).pack()
            val = tk.Label(f, text="—", font=("Segoe UI", 10, "bold"),
                           fg=COLORS["gold"], bg=COLORS["header_bg"],
                           width=8, anchor="center")
            val.pack()
            self._info_labels[key] = val

        # Main content: table (left) + log (right) — use pack with fixed log width.
        content = tk.Frame(self.parent, bg=COLORS["table_felt"])
        content.pack(fill="both", expand=True, padx=4, pady=2)

        # Game log panel (right, fixed width).
        log_frame = tk.Frame(content, bg="#0d1b0d", bd=1, relief="sunken", width=180)
        log_frame.pack(side="right", fill="y", padx=(2, 4), pady=4)
        log_frame.pack_propagate(False)

        tk.Label(log_frame, text="Game Log", font=("Segoe UI", 9, "bold"),
                 fg=COLORS["gold"], bg="#0d1b0d").pack(pady=(4, 2))
        self._log_text = tk.Text(log_frame, bg="#0a150a", fg="#81c784",
                                 font=("Consolas", 8), width=20, height=30,
                                 relief="flat", state="disabled",
                                 wrap="word", highlightthickness=0)
        self._log_text.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        # Table area (left, fills remaining space).
        table = tk.Frame(content, bg=COLORS["table_felt"], bd=3, relief="ridge")
        table.pack(side="left", fill="both", expand=True, padx=(4, 2), pady=4)
        table.grid_propagate(False)
        table.columnconfigure(0, weight=1, uniform="col")
        table.columnconfigure(1, weight=2, uniform="col")
        table.columnconfigure(2, weight=1, uniform="col")
        table.rowconfigure(0, weight=1, uniform="row")
        table.rowconfigure(1, weight=2, uniform="row")
        table.rowconfigure(2, weight=2, uniform="row")

        # Player areas.
        self._player_frames = {}
        self._player_canvases = {}
        self._player_status = {}
        self._player_bid_labels = {}

        self._create_opponent_area(table, 0, "P3 (AI)", "Team 1", row=0, col=1)
        self._create_opponent_area(table, 3, "P4 (AI)", "Team 2", row=1, col=0)
        self._create_opponent_area(table, 1, "P2 (AI)", "Team 2", row=1, col=2)
        self._create_human_area(table)

        # Centre trick canvas — fixed size.
        centre_frame = tk.Frame(table, bg="#0d2e0d", bd=2, relief="groove",
                                width=284, height=204)
        centre_frame.grid(row=1, column=1, padx=4, pady=4)
        centre_frame.grid_propagate(False)
        centre_frame.pack_propagate(False)
        self._centre_canvas = tk.Canvas(centre_frame, bg=COLORS["centre_bg"],
                                        width=280, height=200, highlightthickness=0)
        self._centre_canvas.pack(padx=2, pady=2)

        # Trump display (top-right of table).
        self._trump_display_label = tk.Label(
            table, text="", font=("Segoe UI", 22, "bold"),
            fg=COLORS["table_felt"], bg=COLORS["table_felt"])
        self._trump_display_label.place(relx=0.96, rely=0.04, anchor="ne")

        # Status bar — FIXED height.
        status_bar = tk.Frame(self.parent, bg="#1a3a1a", height=30)
        status_bar.pack(fill="x", padx=8, pady=(2, 0))
        status_bar.pack_propagate(False)
        self._status_label = tk.Label(status_bar, text="Press Start Game",
                                      font=("Segoe UI", 10, "bold"),
                                      fg=COLORS["gold"], bg="#1a3a1a")
        self._status_label.pack(expand=True)

        self._turn_indicator_active = False

        # Controls — FIXED height.
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
        tk.Button(bf, text="📂 Load AI Model", command=self._load_ai_model,
                  font=("Segoe UI", 9), fg="#fff", bg="#1e88e5",
                  bd=0, padx=10, pady=3, cursor="hand2").pack(side="left", padx=4)
        self._ai_model_label = tk.Label(bf, text="AI: Rule-Based",
                                        font=("Segoe UI", 8), fg="#aaaaaa",
                                        bg=COLORS["header_bg"])
        self._ai_model_label.pack(side="left", padx=8)

    def _create_opponent_area(self, parent, pid, name, team, row, col):
        """Create opponent player area — fixed size to prevent jumping."""
        sticky = "nsew"
        frame = tk.Frame(parent, bg=COLORS["table_felt"], padx=4, pady=3)
        frame.grid(row=row, column=col, sticky=sticky, padx=3, pady=3)
        frame.grid_propagate(False) if row == 1 else None

        team_color = COLORS["score_team1"] if "1" in team else COLORS["score_team2"]
        header = tk.Frame(frame, bg=COLORS["table_felt"])
        header.pack(fill="x", pady=(0, 2))
        tk.Label(header, text=name, font=("Segoe UI", 9, "bold"),
                 fg="#ffffff", bg=COLORS["table_felt"]).pack(side="left")
        tk.Label(header, text=f"  ({team})", font=("Segoe UI", 8),
                 fg=team_color, bg=COLORS["table_felt"]).pack(side="left")

        status = tk.Label(frame, text="", font=("Segoe UI", 8),
                          fg=COLORS["text_muted"], bg=COLORS["table_felt"],
                          anchor="w", width=20)
        status.pack(fill="x", pady=(0, 1))
        self._player_status[pid] = status

        bid_lbl = tk.Label(frame, text="", font=("Consolas", 9, "bold"),
                           fg=COLORS["text_dim"], bg=COLORS["table_felt"],
                           anchor="w", width=12)
        bid_lbl.pack(fill="x", pady=(0, 2))
        self._player_bid_labels[pid] = bid_lbl

        canvas_height = CARD_MINI_HEIGHT + 6
        canvas = tk.Canvas(frame, bg=COLORS["table_felt"], height=canvas_height,
                           highlightthickness=0, width=140)
        canvas.pack(fill="x", pady=2)
        self._player_canvases[pid] = canvas
        self._player_frames[pid] = frame

    def _create_human_area(self, parent):
        """Create the human player area (bottom, face-up, clickable) — fixed size."""
        frame = tk.Frame(parent, bg=COLORS["table_felt"], padx=6, pady=4, height=165)
        frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 4))
        frame.pack_propagate(False)

        header = tk.Frame(frame, bg=COLORS["table_felt"])
        header.pack(fill="x", pady=(0, 2))
        tk.Label(header, text="🧑 YOU — Player 1",
                 font=("Segoe UI", 10, "bold"),
                 fg=COLORS["gold"], bg=COLORS["table_felt"]).pack(side="left")
        tk.Label(header, text="  (Team 1)", font=("Segoe UI", 9),
                 fg=COLORS["score_team1"], bg=COLORS["table_felt"]).pack(side="left")

        self._human_bid_label = tk.Label(header, text="", font=("Consolas", 9, "bold"),
                                         fg=COLORS["text_dim"], bg=COLORS["table_felt"])
        self._human_bid_label.pack(side="right", padx=8)

        self._human_canvas = tk.Canvas(frame, bg=COLORS["table_felt"],
                                       height=CARD_HEIGHT + 8, highlightthickness=0)
        self._human_canvas.pack(fill="x")

        # Bid buttons placeholder — FIXED height.
        self._bid_placeholder = tk.Frame(frame, bg=COLORS["table_felt"], height=55)
        self._bid_placeholder.pack(fill="x")
        self._bid_placeholder.pack_propagate(False)
        self._player_frames[HUMAN_ID] = frame

    # ----------------------------------------------------------
    # Game log
    # ----------------------------------------------------------

    def _log(self, msg: str):
        """Add an entry to the game log panel — crash-safe."""
        self._game_log.append(msg)
        try:
            self._log_text.config(state="normal")
            self._log_text.insert("end", msg + "\n")
            self._log_text.see("end")
            self._log_text.config(state="disabled")
        except (tk.TclError, Exception):
            pass

    def _clear_log(self):
        self._game_log.clear()
        try:
            self._log_text.config(state="normal")
            self._log_text.delete("1.0", "end")
            self._log_text.config(state="disabled")
        except (tk.TclError, Exception):
            pass

    # ----------------------------------------------------------
    # Display helpers
    # ----------------------------------------------------------

    def _show_opponent_cards(self, pid):
        """Show face-down cards for an opponent."""
        canvas = self._player_canvases[pid]
        canvas.delete("all")
        if self.players is None:
            return
        count = len(self.players[pid].hand)
        if count == 0:
            return
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
        """Show human hand face-up. Cards scale with count."""
        canvas = self._human_canvas
        canvas.delete("all")
        if self.players is None:
            return
        hand = self.players[HUMAN_ID].hand
        if not hand:
            canvas.create_text(100, 30, text="No cards left", fill="#888888",
                               font=("Segoe UI", 9))
            return

        # Legal cards for highlighting.
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

        # Adaptive card size based on cards remaining.
        card_count = len(sorted_hand)
        card_w = CARD_WIDTH if card_count <= 10 else CARD_MINI_WIDTH + 4
        card_h = CARD_HEIGHT if card_count <= 10 else CARD_MINI_HEIGHT + 4

        spacing = min(card_w + 4, max(28, (canvas.winfo_width() or 600) // max(card_count + 3, 1)))
        suit_gap = 12

        # Calculate total width to centre.
        total_width = 0
        prev_s = None
        for card in sorted_hand:
            if prev_s is not None and card.suit != prev_s:
                total_width += suit_gap
            total_width += spacing
            prev_s = card.suit
        total_width = total_width - spacing + card_w

        canvas_w = canvas.winfo_width() or 600
        current_x = max(6, (canvas_w - total_width) // 2)
        prev_suit = None

        for i, card in enumerate(sorted_hand):
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
                      width=card_w, height=card_h,
                      highlight=highlight, faded=faded, tag=tag)

            if clickable and is_legal:
                canvas.tag_bind(tag, "<Button-1>", lambda e, c=card: self._human_play(c))
                # Hover effect: move card up.
                canvas.tag_bind(tag, "<Enter>", lambda e, t=tag: self._card_hover(t, True))
                canvas.tag_bind(tag, "<Leave>", lambda e, t=tag: self._card_hover(t, False))

            current_x += spacing

    def _card_hover(self, tag, entering):
        """Hover effect — lift card up slightly. Safe against destroyed canvas."""
        if not self.game_running:
            return
        try:
            canvas = self._human_canvas
            if not canvas.winfo_exists():
                return
            if entering:
                canvas.move(tag, 0, -3)
            else:
                canvas.move(tag, 0, 3)
        except (tk.TclError, Exception):
            pass

    def _draw_centre_trick(self, winner_id=None):
        """Draw played cards in the centre with player labels and winner/whip highlight."""
        canvas = self._centre_canvas
        canvas.delete("all")
        w, h = 280, 200
        cw, ch = CARD_LARGE_WIDTH, CARD_LARGE_HEIGHT

        # Fixed positions for each player's card.
        positions = {
            0: (w // 2 - cw // 2, 4),              # P3 top
            1: (w - cw - 12, h // 2 - ch // 2),    # P2 right
            2: (w // 2 - cw // 2, h - ch - 4),     # P1 (you) bottom
            3: (12, h // 2 - ch // 2),              # P4 left
        }

        # Player labels.
        label_pos = {
            0: (w // 2, 4 + ch + 8),
            1: (w - cw // 2 - 12, h // 2 + ch // 2 + 10),
            2: (w // 2, h - ch - 12),
            3: (cw // 2 + 12, h // 2 + ch // 2 + 10),
        }

        # Determine whipping/over-trumping for red highlight.
        # Rule: red highlight ONLY when led suit is NOT trump and a card is trump.
        # Over-trump: a higher trump after a previous trump, still only in non-trump led tricks.
        trump_sym = SUIT_SYMBOLS.get(self.trump_suit, "") if self.trump_suit else ""
        play_order = self.round.state.current_trick.play_order if (
            self.round and self.round.state.current_trick and
            hasattr(self.round.state.current_trick, 'play_order')
        ) else []

        # Determine the led suit from the first card played in this trick.
        led_suit_sym = ""
        first_pid = play_order[0] if play_order else None
        if first_pid is not None and first_pid in self._trick_played:
            _, led_suit_sym = parse_card_text(self._trick_played[first_pid])
        elif self._trick_played:
            # Fallback: first entry in trick_played dict.
            first_card = next(iter(self._trick_played.values()), "")
            _, led_suit_sym = parse_card_text(first_card)

        led_is_trump = (led_suit_sym == trump_sym) if trump_sym else False

        # Track highest trump seen so far for over-trump detection.
        whip_cards = set()  # PIDs that are whipping or over-trumping.
        if not led_is_trump and trump_sym:
            highest_trump_rank = -1
            rank_values = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
                           "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
            for pid in (play_order if play_order else sorted(self._trick_played.keys())):
                if pid not in self._trick_played:
                    continue
                ct = self._trick_played[pid]
                rank, suit_sym = parse_card_text(ct)
                if suit_sym == trump_sym and pid != first_pid:
                    # This is a trump played in a non-trump trick = whip or over-trump.
                    rv = rank_values.get(rank, 0)
                    if highest_trump_rank < 0:
                        # First trump in this trick = whipping.
                        whip_cards.add(pid)
                    elif rv > highest_trump_rank:
                        # Higher trump than previous = over-trumping.
                        whip_cards.add(pid)
                    highest_trump_rank = max(highest_trump_rank, rv)

        for pid, (x, y) in positions.items():
            if pid in self._trick_played:
                ct = self._trick_played[pid]
                rank, suit = parse_card_text(ct)
                # Highlight: gold for winner, red for whip/over-trump, none otherwise.
                if pid == winner_id:
                    hl = "#ffd54f"
                elif pid in whip_cards:
                    hl = "#e53935"  # Red for whipping / over-trumping.
                else:
                    hl = None
                draw_card(canvas, x, y, rank, suit, width=cw, height=ch,
                          highlight=hl)
            else:
                canvas.create_rectangle(x, y, x + cw, y + ch,
                                        fill="#2a4a2a", outline="#3a5a3a", dash=(3, 3))

            # Player label near each slot.
            lx, ly = label_pos[pid]
            pname = DISPLAY_NAMES.get(pid, f"P{pid+1}")
            color = "#ffd54f" if pid == winner_id else ("#e53935" if pid in whip_cards else "#5a8a5a")
            canvas.create_text(lx, ly, text=pname, fill=color,
                               font=("Segoe UI", 7))

    def _update_info(self, **kw):
        for key, val in kw.items():
            if key in self._info_labels:
                self._info_labels[key].config(text=str(val) if val else "—")

    def _set_status(self, text):
        self._status_label.config(text=text)

    def _show_all_hands(self):
        for pid in [0, 1, 3]:
            self._show_opponent_cards(pid)
        self._show_human_hand(clickable=False)

    # ----------------------------------------------------------
    # Turn indicator
    # ----------------------------------------------------------

    def _highlight_active_player(self, pid):
        """Highlight the active player's area."""
        self._active_player_id = pid
        try:
            # Reset all opponent borders to role-based.
            for p in [0, 1, 3]:
                frame = self._player_frames.get(p)
                if frame is None:
                    continue
                if p == self.qabool_id and p == self.shooter_id:
                    frame.config(highlightbackground=COLORS["gold"], highlightthickness=3)
                elif p == self.qabool_id:
                    frame.config(highlightbackground=COLORS["gold"], highlightthickness=3)
                elif p == self.shooter_id:
                    frame.config(highlightbackground="#66ff66", highlightthickness=2)
                else:
                    frame.config(highlightthickness=0)

            # Human frame.
            hf = self._player_frames.get(HUMAN_ID)
            if hf is None:
                return
            if HUMAN_ID == self.qabool_id:
                hf.config(highlightbackground=COLORS["gold"], highlightthickness=3)
            elif HUMAN_ID == self.shooter_id:
                hf.config(highlightbackground="#66ff66", highlightthickness=2)
            else:
                hf.config(highlightthickness=0)

            # Now highlight the currently active one.
            if pid == HUMAN_ID:
                hf.config(highlightbackground="#ffffff", highlightthickness=3)
                self._turn_indicator_active = True
                self._pulse_turn()
            else:
                self._turn_indicator_active = False
                if pid in self._player_frames:
                    self._player_frames[pid].config(
                        highlightbackground="#aaaaaa", highlightthickness=2)
        except (tk.TclError, Exception):
            pass

    def _pulse_turn(self):
        """Pulse the human's border — safe against destroyed widgets."""
        if not self._turn_indicator_active or not self.game_running:
            return
        try:
            hf = self._player_frames.get(HUMAN_ID)
            if hf is None or not hf.winfo_exists():
                return
            current = hf.cget("highlightbackground")
            next_color = "#ffffff" if current != "#ffffff" else COLORS["gold"]
            hf.config(highlightbackground=next_color)
            self.root.after(600, self._pulse_turn)
        except (tk.TclError, Exception):
            self._turn_indicator_active = False

    def _clear_active_highlight(self):
        """Remove active turn highlight — stop pulsing."""
        self._turn_indicator_active = False
        self._active_player_id = None

    # ----------------------------------------------------------
    # Trump display
    # ----------------------------------------------------------

    def _show_trump(self):
        """Reveal trump symbol in top-right corner."""
        trump_sym = SUIT_SYMBOLS.get(self.trump_suit, "?")
        fg = "#c62828" if self.trump_suit in (Suit.HEARTS, Suit.DIAMONDS) else "#ffffff"
        self._trump_display_label.config(text=trump_sym, fg=fg, bg=COLORS["table_felt"])
        self.trump_revealed = True
        self._update_info(trump=f"{self.trump_suit.name} {trump_sym}")
        self._log(f"♚ Trump revealed: {self.trump_suit.name} {trump_sym}")

    def _hide_trump(self):
        """Hide trump (shown as ? until first card played)."""
        self._trump_display_label.config(text="?", fg="#5a8a5a", bg=COLORS["table_felt"])
        self.trump_revealed = False
        self._update_info(trump="? (hidden)")

    # ----------------------------------------------------------
    # Game flow
    # ----------------------------------------------------------

    def _start_game(self):
        if self.game_running:
            return
        self._reset_table()
        self._clear_log()
        self.game_running = True
        self.trick_number = 0
        self.team_tricks = [0, 0]
        self._trick_played = {}
        self._human_chosen_trump = None
        self._human_trump_choice = None
        self.game_scores = [0, 0]
        self.shota_number = 0
        self.playing_team_id = None
        self._shota_history = []
        self._log("━━━ GAME STARTED ━━━")
        self._start_new_shota()

    def _stop_game(self):
        self.game_running = False
        self._turn_indicator_active = False
        self._set_status("Stopped")
        self._log("⏹ Game stopped.")

    def _load_ai_model(self):
        """Load a trained learning model JSON for the AI opponents."""
        from tkinter import filedialog
        from agents.wist_learning.learning_agent import LearningAgent
        import os
        import sys

        # Try multiple strategies to find the agents/wist_discovery folder.
        default_dir = ""
        candidates = [
            # Hardcoded project path.
            r"C:\Users\emagabu\workspace\Telecom-Native-Intelligence\agents\wist_discovery",
            # Relative to exe location.
            os.path.join(os.path.dirname(sys.executable), "agents", "wist_discovery"),
            # Relative to working directory.
            os.path.join(os.getcwd(), "agents", "wist_discovery"),
            # Relative to this file (works in dev, not in exe).
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "agents", "wist_discovery"),
        ]
        for candidate in candidates:
            if os.path.isdir(candidate):
                default_dir = candidate
                break

        path = filedialog.askopenfilename(
            initialdir=default_dir if default_dir else None,
            filetypes=[("JSON files", "*.json")],
            title="Load Learning Agent for AI opponents",
        )
        if path:
            try:
                self._loaded_learning_agent = LearningAgent.load(path, training=False)
                self._ai_model_label.config(
                    text=f"AI: Learning ({self._loaded_learning_agent.q_table_size} entries)",
                    fg="#66bb6a")
            except Exception as e:
                self._ai_model_label.config(text=f"Error: {e}", fg="#ff6666")

    def _get_ai_agent(self):
        """Return AI agent — uses loaded model if available, else Rule-Based."""
        if self._loaded_learning_agent:
            return self._loaded_learning_agent
        return RuleBasedAgent()

    def _start_new_shota(self):
        """Start a new Shota (deal, bid, play 13 tricks)."""
        self.shota_number += 1
        self.trick_number = 0
        self.team_tricks = [0, 0]
        self._trick_played = {}
        self._human_chosen_trump = None
        self._human_trump_choice = None
        self._shota_finishing = False
        self.trump_revealed = False

        self.players = create_standard_players()
        self.round = Round(self.players)
        self.round.deal()

        # Handle card-based Dak with visual ceremony.
        attempts = 0
        while self.round.has_card_based_dak() and attempts < 10:
            if attempts == 0:
                self._log(f"🃏 Card-based Dak detected! Re-dealing...")
                self._set_status("Card-based Dak! Re-dealing...")
            self.round = Round(self.players)
            self.round.deal()
            attempts += 1

        if attempts > 0:
            self._log(f"   Re-dealt {attempts} time(s).")

        self.agents = [self._get_ai_agent(), self._get_ai_agent(), None, self._get_ai_agent()]

        # Determine Qabool.
        if self.shota_number == 1:
            self.qabool_id = determine_first_shota_qabool()
            self._log(f"━━━ SHOTA {self.shota_number}/5 ━━━")
            self._log(f"👑 First Qabool draw → P{self.qabool_id+1} ({DISPLAY_NAMES[self.qabool_id]})")
        else:
            self.qabool_id = (self.qabool_id + 1) % 4  # Counter-clockwise.
            self._log(f"━━━ SHOTA {self.shota_number}/5 ━━━")
            self._log(f"👑 Qabool rotates → P{self.qabool_id+1} ({DISPLAY_NAMES[self.qabool_id]})")

        self._update_info(shota=f"{self.shota_number}/5",
                          qabool=DISPLAY_NAMES[self.qabool_id],
                          t1_won="0", t2_won="0",
                          score=f"{self.game_scores[0]}–{self.game_scores[1]}")
        self._hide_trump()

        # Highlight Qabool.
        for pid in [0, 1, 3]:
            if pid == self.qabool_id:
                self._player_status[pid].config(text="👑 Sahib Al-Qabool", fg=COLORS["gold"])
                self._player_frames[pid].config(highlightbackground=COLORS["gold"],
                                                highlightthickness=3)
            else:
                self._player_status[pid].config(text="")
                self._player_frames[pid].config(highlightthickness=0)

        hf = self._player_frames[HUMAN_ID]
        if self.qabool_id == HUMAN_ID:
            hf.config(highlightbackground=COLORS["gold"], highlightthickness=3)
        else:
            hf.config(highlightthickness=0)

        self._show_all_hands()
        self._centre_canvas.delete("all")

        if self.qabool_id == HUMAN_ID:
            self._set_status(f"Shota {self.shota_number} — YOU are Sahib Al-Qabool! 👑")
        else:
            self._set_status(f"Shota {self.shota_number} — {DISPLAY_NAMES[self.qabool_id]} is Qabool. Bidding...")

        self._schedule(800, self._run_bidding)

    # ----------------------------------------------------------
    # Bidding
    # ----------------------------------------------------------

    def _run_bidding(self):
        if not self.game_running:
            return
        from environments.wist.tasmiya_engine import tasmiya_order
        from environments.wist.bidding_engine import BiddingEngine

        self._bidding_engine = BiddingEngine()
        self._bid_history = []
        self._bid_order = tasmiya_order(self.qabool_id)
        self._bid_index = 0
        self._has_opening_bid = False

        self._set_status(f"Al-Tasmiya — Sahib Al-Qabool: {DISPLAY_NAMES[self.qabool_id]}")
        self._log("─── BIDDING ───")
        self._schedule(800, self._bid_next_player)

    def _bid_next_player(self):
        if not self.game_running:
            return
        if self._bid_index >= len(self._bid_order):
            self._bid_qabool_turn()
            return

        pid = self._bid_order[self._bid_index]

        if pid == HUMAN_ID:
            self._show_human_bid_options()
        else:
            from environments.wist.observation import BiddingObservation
            from environments.wist.actions import BidAction, PassAction
            from environments.wist.bidding import Bid, Pass

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

            if isinstance(action, BidAction):
                bid = Bid(player_id=pid, value=action.value)
                self._bidding_engine.apply_bid(bid)
                self._bid_history.append((pid, action.value))
                self._has_opening_bid = True
                self._player_status[pid].config(text=f"Bid: {action.value}", fg=COLORS["gold"])
                self._player_bid_labels[pid].config(text=f"Bid: {action.value}", fg=COLORS["gold"])
                self._set_status(f"{DISPLAY_NAMES[pid]} bids {action.value}")
                self._log(f"  {DISPLAY_NAMES[pid]}: Bid {action.value}")
            else:
                self._bidding_engine.apply_pass(Pass(player_id=pid))
                self._bid_history.append((pid, None))
                self._player_status[pid].config(text="Pass", fg=COLORS["text_muted"])
                self._player_bid_labels[pid].config(text="Pass", fg=COLORS["text_dim"])
                self._set_status(f"{DISPLAY_NAMES[pid]} passes")
                self._log(f"  {DISPLAY_NAMES[pid]}: Pass")

            self._bid_index += 1
            self._schedule(800, self._bid_next_player)

    def _show_human_bid_options(self):
        """Step-by-step: bid number → trump suit → confirm."""
        self._show_human_hand(clickable=False)
        self._safe_destroy(self._bid_btn_frame)

        self._bid_btn_frame = tk.Frame(self._bid_placeholder, bg=COLORS["table_felt"])
        self._bid_btn_frame.pack(fill="both", expand=True)
        self._human_trump_choice = None
        self._human_bid_value = None

        current_highest = (self._bidding_engine.highest_bid.value
                           if self._bidding_engine.highest_bid else None)
        is_qabool = (self.qabool_id == HUMAN_ID and self._bid_index >= len(self._bid_order))
        can_dak = is_qabool and current_highest is None

        self._set_status("YOUR TURN! Select bid → trump → confirm.")
        self._highlight_active_player(HUMAN_ID)

        # Row 1: Bid numbers 7-13 + Pass/Dak.
        row1 = tk.Frame(self._bid_btn_frame, bg=COLORS["table_felt"])
        row1.pack(fill="x", pady=(4, 2))
        inner1 = tk.Frame(row1, bg=COLORS["table_felt"])
        inner1.pack(expand=True)

        tk.Label(inner1, text="Bid:", font=("Segoe UI", 10, "bold"),
                 fg=COLORS["gold"], bg=COLORS["table_felt"]).pack(side="left", padx=(0, 8))

        for val in range(7, 14):
            btn = tk.Button(inner1, text=str(val),
                            font=("Segoe UI", 10, "bold"), fg="#fff",
                            bg=COLORS["btn_green"], bd=0, padx=8, pady=3,
                            cursor="hand2",
                            command=lambda v=val: self._bid_number_selected(v))
            btn.pack(side="left", padx=2)

        pass_text = "Accept" if (is_qabool and current_highest) else "Pass"
        tk.Button(inner1, text=pass_text, font=("Segoe UI", 10, "bold"),
                  fg="#fff", bg=COLORS["btn_grey"], bd=0, padx=10, pady=3,
                  cursor="hand2", command=lambda: self._human_bid(None)
                  ).pack(side="left", padx=(12, 2))

        if can_dak:
            tk.Button(inner1, text="Dak", font=("Segoe UI", 10, "bold"),
                      fg="#fff", bg=COLORS["btn_red"], bd=0, padx=10, pady=3,
                      cursor="hand2", command=self._human_dak
                      ).pack(side="left", padx=2)

        # Row 2: Trump + confirm (shown after bid selected).
        self._trump_row = tk.Frame(self._bid_btn_frame, bg=COLORS["table_felt"])

    def _bid_number_selected(self, value):
        """Human selected a bid number — show trump selection."""
        self._human_bid_value = value
        self._set_status(f"Bid {value} selected. Choose your trump suit →")

        self._trump_row.pack(fill="x", pady=(4, 2))
        for w in self._trump_row.winfo_children():
            w.destroy()

        inner = tk.Frame(self._trump_row, bg=COLORS["table_felt"])
        inner.pack(expand=True)

        tk.Label(inner, text=f"Bid {value} — Trump:",
                 font=("Segoe UI", 10, "bold"), fg=COLORS["gold"],
                 bg=COLORS["table_felt"]).pack(side="left", padx=(0, 8))

        for suit in [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]:
            sym = SUIT_SYMBOLS[suit]
            count = sum(1 for c in self.players[HUMAN_ID].hand if c.suit == suit)
            fg = "#c62828" if suit in (Suit.HEARTS, Suit.DIAMONDS) else "#1a1a1a"
            btn = tk.Button(inner, text=f"{sym}({count})",
                            font=("Consolas", 11, "bold"), fg=fg,
                            bg=COLORS["card_bg"], bd=1, padx=5, pady=2,
                            cursor="hand2",
                            command=lambda s=suit: self._trump_selected(s))
            btn.pack(side="left", padx=3)

    def _trump_selected(self, suit):
        """Trump selected — show confirm."""
        self._human_trump_choice = suit
        sym = SUIT_SYMBOLS[suit]
        self._set_status(f"Bid {self._human_bid_value}, Trump: {suit.name} {sym} — Confirm?")

        # Replace any existing confirm button.
        self._safe_destroy(self._confirm_frame)
        self._confirm_frame = tk.Frame(self._trump_row, bg=COLORS["table_felt"])
        self._confirm_frame.pack(side="left", padx=(12, 0))

        tk.Button(self._confirm_frame, text="✓ Confirm",
                  font=("Segoe UI", 10, "bold"), fg="#fff",
                  bg=COLORS["btn_green"], bd=0, padx=12, pady=3,
                  cursor="hand2", command=self._confirm_human_bid).pack()

    def _confirm_human_bid(self):
        if self._human_bid_value is None or self._human_trump_choice is None:
            return
        self._human_chosen_trump = self._human_trump_choice
        self._human_bid(self._human_bid_value)

    def _human_dak(self):
        """Handle Dak — all passed and qabool declares redeal of current shota."""
        self._safe_destroy(self._bid_btn_frame)
        self._bid_btn_frame = None
        self._clear_active_highlight()
        self._show_human_hand(clickable=False)

        self._set_status("Dak! Re-dealing this shota...")
        self._log("  ⚡ DAK! All passed — re-dealing.")
        # Don't reset game_running or game scores — just restart the current shota.
        self.shota_number -= 1  # Will be incremented again by _start_new_shota.
        self._schedule(1500, self._start_new_shota)

    def _human_bid(self, value):
        """Handle human's bid decision."""
        from environments.wist.bidding import Bid, Pass

        self._safe_destroy(self._bid_btn_frame)
        self._bid_btn_frame = None
        self._clear_active_highlight()

        if value is not None:
            is_qabool_bidding = (self.qabool_id == HUMAN_ID and self._bid_index >= len(self._bid_order))
            bid = Bid(player_id=HUMAN_ID, value=value)
            self._bidding_engine.apply_bid(bid, is_sahib_al_qabool=is_qabool_bidding)
            self._bid_history.append((HUMAN_ID, value))
            self._has_opening_bid = True
            self._set_status(f"You bid {value}")
            self._human_bid_label.config(text=f"Bid: {value}", fg=COLORS["gold"])
            self._log(f"  {DISPLAY_NAMES[HUMAN_ID]}: Bid {value}")
        else:
            self._bidding_engine.apply_pass(Pass(player_id=HUMAN_ID))
            self._bid_history.append((HUMAN_ID, None))
            self._set_status("You pass")
            self._human_bid_label.config(text="Pass", fg=COLORS["text_dim"])
            self._log(f"  {DISPLAY_NAMES[HUMAN_ID]}: Pass")

        self._show_human_hand(clickable=False)

        if self._bid_index >= len(self._bid_order):
            # Qabool just decided.
            all_passed = self._bidding_engine.highest_bid is None
            if all_passed and value is None:
                self._set_status("Dak! Re-dealing this shota...")
                self._log("  ⚡ DAK! All passed.")
                self.shota_number -= 1
                self._schedule(1500, self._start_new_shota)
                return
            self._schedule(600, self._finalize_bidding)
        else:
            self._bid_index += 1
            self._schedule(600, self._bid_next_player)

    def _bid_qabool_turn(self):
        """Sahib Al-Qabool makes the final decision."""
        from environments.wist.observation import BiddingObservation
        from environments.wist.actions import BidAction, PassAction
        from environments.wist.bidding import Bid, Pass

        qid = self.qabool_id
        if qid == HUMAN_ID:
            self._show_human_bid_options()
            return

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
            self._player_status[qid].config(text=f"Bid: {action.value} (Qabool)", fg=COLORS["gold"])
            self._player_bid_labels[qid].config(text=f"Bid: {action.value}", fg=COLORS["gold"])
            self._log(f"  {DISPLAY_NAMES[qid]} (Qabool): Bid {action.value}")
        else:
            self._bidding_engine.apply_pass(Pass(player_id=qid))
            self._bid_history.append((qid, None))
            all_passed = self._bidding_engine.highest_bid is None
            if all_passed:
                self._set_status("All passed — Dak! Re-dealing...")
                self._log("  ⚡ DAK! All passed — re-dealing.")
                self.shota_number -= 1
                self._schedule(1500, self._start_new_shota)
                return
            self._player_status[qid].config(text="Accepts", fg=COLORS["text_muted"])
            self._player_bid_labels[qid].config(text="Accepts", fg=COLORS["text_dim"])
            self._log(f"  {DISPLAY_NAMES[qid]} (Qabool): Accepts")

        self._schedule(800, self._finalize_bidding)

    def _finalize_bidding(self):
        """Finalize bidding and start play."""
        winning_bid = self._bidding_engine.highest_bid
        if winning_bid is None:
            self._set_status("Dak! Re-dealing...")
            self._log("  ⚡ DAK!")
            self.shota_number -= 1
            self._schedule(1500, self._start_new_shota)
            return

        self.shooter_id = winning_bid.player_id
        self.bid_value = winning_bid.value
        self.playing_team_id = self.players[self.shooter_id].team_id

        # Trump: use human's choice if human won, else auto-determine.
        if self.shooter_id == HUMAN_ID and self._human_chosen_trump:
            self.trump_suit = self._human_chosen_trump
        else:
            self.trump_suit = determine_trump_suit(self.players[self.shooter_id].hand)

        self.round.state.trump_suit = self.trump_suit
        self.round.state.winning_bidder_id = self.shooter_id
        self.round.next_leading_player_id = self.shooter_id
        self.environment = WistEnvironment(self.round.state)

        # Update info bar — trump HIDDEN until first card played.
        self._update_info(bid=str(self.bid_value),
                          shooter=DISPLAY_NAMES[self.shooter_id])
        self._hide_trump()

        self._log(f"─── BIDDING RESULT ───")
        self._log(f"  Shooter: {DISPLAY_NAMES[self.shooter_id]} | Bid: {self.bid_value}")
        self._log(f"  Trump: {self.trump_suit.name} (hidden until first card)")

        # Highlight roles persistently.
        for pid in [0, 1, 3]:
            s = self._player_status[pid]
            if pid == self.shooter_id and pid == self.qabool_id:
                s.config(text="👑 Qabool | 🎯 SHOOTER", fg="#66ff66")
                self._player_frames[pid].config(highlightbackground=COLORS["gold"],
                                                highlightthickness=3, bd=2, relief="solid")
            elif pid == self.shooter_id:
                s.config(text="🎯 SHOOTER", fg="#66ff66")
                self._player_frames[pid].config(highlightbackground="#66ff66",
                                                highlightthickness=2)
            elif pid == self.qabool_id:
                s.config(text="👑 Qabool", fg=COLORS["gold"])
                self._player_frames[pid].config(highlightbackground=COLORS["gold"],
                                                highlightthickness=3)
            else:
                s.config(text="")
                self._player_frames[pid].config(highlightthickness=0)

        # Human frame highlight.
        hf = self._player_frames[HUMAN_ID]
        if self.shooter_id == HUMAN_ID and self.qabool_id == HUMAN_ID:
            hf.config(highlightbackground=COLORS["gold"], highlightthickness=3, bd=2, relief="solid")
        elif self.shooter_id == HUMAN_ID:
            hf.config(highlightbackground="#66ff66", highlightthickness=2)
        elif self.qabool_id == HUMAN_ID:
            hf.config(highlightbackground=COLORS["gold"], highlightthickness=3)
        else:
            hf.config(highlightthickness=0)

        self._set_status(f"Bidding done! {DISPLAY_NAMES[self.shooter_id]} shoots. Trump hidden.")
        self._show_all_hands()
        self._log("─── TRICKS ───")
        self._schedule(1000, self._play_next_trick)

    # ----------------------------------------------------------
    # Trick play
    # ----------------------------------------------------------

    def _play_next_trick(self):
        if not self.game_running:
            return
        if self.trick_number >= 13:
            if self._shota_finishing:
                return
            self._shota_finishing = True
            self._finish_shota()
            return

        self.trick_number += 1
        self._trick_played = {}
        self._draw_centre_trick()
        self._update_info(trick=f"{self.trick_number}/13")

        leader = self.round.next_leading_player_id
        self.round.state.current_trick = Trick(leading_player_id=leader)
        self._play_order = [(leader + i) % 4 for i in range(4)]
        self._play_idx = 0

        self._set_status(f"Trick {self.trick_number} — {DISPLAY_NAMES[leader]} leads")
        self._schedule(400, self._play_next_card)

    def _play_next_card(self):
        if not self.game_running:
            return
        if self._play_idx >= 4:
            self._schedule(600, self._resolve_trick)
            return

        pid = self._play_order[self._play_idx]
        self._highlight_active_player(pid)

        if pid == HUMAN_ID:
            self._set_status(f"Trick {self.trick_number} — YOUR TURN! Click a card.")
            self._show_human_hand(clickable=True)
        else:
            try:
                obs = self.environment.observe(pid)
                action = self.agents[pid].act(obs)
                self.environment.apply_action(action)

                self._play_idx += 1
                ct = card_str(action.card)
                self._trick_played[pid] = ct
                self._draw_centre_trick()
                self._show_opponent_cards(pid)

                # Reveal trump on first card of first trick.
                if not self.trump_revealed and self.trick_number == 1:
                    self._show_trump()

                self._set_status(f"Trick {self.trick_number} — {DISPLAY_NAMES[pid]} played {ct}")
            except Exception as e:
                # AI failed — skip this player with a dummy play if possible.
                self._log(f"  ⚠ AI error P{pid+1}: {e}")
                self._play_idx += 1

            self._schedule(600, self._play_next_card)

    def _human_play(self, card: Card):
        if not self.game_running:
            return
        if self.round.state.current_trick is None:
            return
        if card not in self.players[HUMAN_ID].hand:
            return
        # Disable immediately to prevent double-click.
        self._show_human_hand(clickable=False)
        self._clear_active_highlight()

        try:
            action = PlayCardAction(player_id=HUMAN_ID, card=card)
            self.environment.apply_action(action)
        except Exception as e:
            self._log(f"  ⚠ Play error: {e}")
            # Re-enable hand so user can try again.
            self._show_human_hand(clickable=True)
            return

        self._play_idx += 1
        ct = card_str(card)
        self._trick_played[HUMAN_ID] = ct
        self._draw_centre_trick()
        self._show_human_hand(clickable=False)

        # --- Record decision for evaluation ---
        try:
            ai_agent = self._get_ai_agent()
            if ai_agent and hasattr(ai_agent, 'act'):
                # Ask AI what it would play in this situation.
                from environments.wist.observations import WistObservation
                obs = WistObservation(
                    player_id=HUMAN_ID,
                    hand=self.players[HUMAN_ID].hand,
                    trump_suit=self.trump_suit,
                    current_trick=self.round.state.current_trick,
                    team_scores={0: self.team_tricks[0], 1: self.team_tricks[1]},
                )
                ai_action = ai_agent.act(obs)
                ai_card_str = card_str(ai_action.card) if hasattr(ai_action, 'card') else ""
                was_trump = (self.trump_suit and card.suit == self.trump_suit)
                # Check if this creates a void.
                from collections import Counter
                suit_counts = Counter(c.suit for c in self.players[HUMAN_ID].hand)
                created_void = (suit_counts.get(card.suit, 0) == 0)
                position = self._play_idx - 1
                self._evaluator.record_trick_decision(
                    shota=self.shota_number,
                    trick=self.trick_number,
                    player_card=ct,
                    ai_card=ai_card_str,
                    position=position,
                    was_trump=bool(was_trump),
                    created_void=created_void,
                    trick_won_by_team=False,  # Updated after trick resolves.
                    context=f"Trick {self.trick_number}, pos {position}",
                )
        except Exception:
            pass  # Don't let evaluation errors break gameplay.

        # Reveal trump on first card of first trick.
        if not self.trump_revealed and self.trick_number == 1:
            self._show_trump()

        self._set_status(f"Trick {self.trick_number} — you played {ct}")
        self._schedule(600, self._play_next_card)

    def _resolve_trick(self):
        if not self.game_running:
            return
        trick = self.round.state.current_trick
        if trick is None:
            self._schedule(400, self._play_next_trick)
            return

        # If trick is incomplete (AI error skipped), just move on.
        if len(trick.played_cards) < 4:
            self._log(f"  ⚠ Trick {self.trick_number} incomplete ({len(trick.played_cards)} cards)")
            self.round.state.current_trick = None
            self._schedule(400, self._play_next_trick)
            return

        winner = trick_winner(trick, self.trump_suit)
        self.round.state.completed_tricks.append(trick)
        self.round.state.current_trick = None
        self.round.next_leading_player_id = winner

        team = 0 if winner in (0, 2) else 1
        self.team_tricks[team] += 1
        self._update_info(t1_won=str(self.team_tricks[0]), t2_won=str(self.team_tricks[1]))

        who = DISPLAY_NAMES[winner]
        self._set_status(f"Trick {self.trick_number} — {who} won! (T1:{self.team_tricks[0]} T2:{self.team_tricks[1]})")
        self._log(f"  T{self.trick_number}: {who} won | {self.team_tricks[0]}–{self.team_tricks[1]}")

        # Show winner highlight in centre.
        self._draw_centre_trick(winner_id=winner)
        self._clear_active_highlight()

        # Keep Qabool + Shooter roles visible.
        for pid in [0, 1, 3]:
            roles = []
            if pid == self.qabool_id:
                roles.append("👑 Qabool")
            if pid == self.shooter_id:
                roles.append("🎯 Shooter")
            if pid == winner:
                roles.append("🏆 Won")
            lbl = self._player_status[pid]
            if roles:
                lbl.config(text=" | ".join(roles),
                           fg="#ffd54f" if pid == winner else
                           (COLORS["gold"] if pid == self.qabool_id else "#66bb6a"))
            else:
                lbl.config(text="")

        self._show_all_hands()
        # Pause to show winner highlight, then continue.
        self._schedule(1400, self._play_next_trick)

    # ----------------------------------------------------------
    # End of Shota / Game
    # ----------------------------------------------------------

    def _finish_shota(self):
        """End of a Shota — score and transition."""
        from environments.wist.scoring import score_shota

        # Score.
        try:
            if self.playing_team_id is not None:
                defending = 1 if self.playing_team_id == 0 else 0
                total = self.team_tricks[0] + self.team_tricks[1]
                if total == 13:
                    score_delta = score_shota(
                        playing_team_id=self.playing_team_id,
                        defending_team_id=defending,
                        bid=self.bid_value,
                        playing_team_tricks=self.team_tricks[self.playing_team_id],
                        defending_team_tricks=self.team_tricks[defending],
                    )
                    self.game_scores[0] += score_delta.get(0, 0)
                    self.game_scores[1] += score_delta.get(1, 0)
                else:
                    self.game_scores[0] += self.team_tricks[0]
                    self.game_scores[1] += self.team_tricks[1]
        except Exception:
            pass

        # Record shota history.
        bid_met = False
        if self.playing_team_id is not None and self.bid_value:
            bid_met = self.team_tricks[self.playing_team_id] >= self.bid_value
        self._shota_history.append({
            "shota": self.shota_number, "bid": self.bid_value,
            "shooter": DISPLAY_NAMES[self.shooter_id],
            "t1": self.team_tricks[0], "t2": self.team_tricks[1],
            "bid_met": bid_met,
        })

        self._update_info(score=f"{self.game_scores[0]}–{self.game_scores[1]}")
        self._log(f"─── SHOTA {self.shota_number} RESULT ───")
        self._log(f"  {'✓ Bid MET' if bid_met else '✗ Bid FAILED'}")
        self._log(f"  Tricks: T1={self.team_tricks[0]} T2={self.team_tricks[1]}")
        self._log(f"  Score: {self.game_scores[0]}–{self.game_scores[1]}")

        # Check if game is over.
        game_over = (self.shota_number >= 5 or
                     self.game_scores[0] >= 25 or self.game_scores[1] >= 25)

        canvas = self._centre_canvas
        canvas.delete("all")
        w, h = 280, 200

        try:
            if game_over:
                if self.game_scores[0] > self.game_scores[1]:
                    winner_text = "YOUR TEAM Wins! 🏆"
                elif self.game_scores[1] > self.game_scores[0]:
                    winner_text = "Team 2 Wins!"
                else:
                    winner_text = "Draw!"

                canvas.create_text(w // 2, h // 2 - 30, text="🏆 GAME OVER 🏆",
                                   fill=COLORS["gold"], font=("Segoe UI", 14, "bold"))
                canvas.create_text(w // 2, h // 2, text=winner_text,
                                   fill="#ffffff", font=("Segoe UI", 12, "bold"))
                canvas.create_text(w // 2, h // 2 + 30,
                                   text=f"Final: T1={self.game_scores[0]} │ T2={self.game_scores[1]}",
                                   fill="#aaaaaa", font=("Segoe UI", 10))
                self._set_status("Game over! Press Start Game to play again.")
                self._log("━━━ GAME OVER ━━━")
                self._log(f"  Winner: {'Team 1 (YOU)' if self.game_scores[0] > self.game_scores[1] else 'Team 2'}")
                self.game_running = False

                # --- Player Evaluation ---
                try:
                    won = self.game_scores[0] > self.game_scores[1]
                    analysis = self._evaluator.finish_game(
                        won=won,
                        score_team1=self.game_scores[0],
                        score_team2=self.game_scores[1],
                        shotas_played=self.shota_number,
                    )
                    self._log("")
                    self._log("━━━ YOUR PERFORMANCE ━━━")
                    self._log(f"  Skill Rating: {analysis['elo']} ({'+' if analysis['elo_change'] > 0 else ''}{analysis['elo_change']:.0f})")
                    self._log(f"  Games Played: {analysis['games_played']}")
                    self._log(f"  Bid Accuracy: {analysis['bid_accuracy']}%")
                    self._log(f"  AI Agreement: {analysis['decision_agreement']}%")
                    self._log(f"  Trump Use: {analysis['trump_efficiency']}%")
                    self._log(f"  Void Play: {analysis['void_exploitation']}%")
                    self._log(f"  Partnership: {analysis['partnership_score']}%")
                    self._log(f"  Defense: {analysis['defense_score']}%")
                    if analysis.get("strengths"):
                        self._log("  Strengths:")
                        for s in analysis["strengths"]:
                            self._log(f"    + {s}")
                    if analysis.get("weaknesses"):
                        self._log("  Areas to improve:")
                        for w in analysis["weaknesses"]:
                            self._log(f"    - {w}")
                    if analysis.get("turning_points"):
                        self._log("  Key moments:")
                        for tp in analysis["turning_points"]:
                            icon = "★" if tp["type"] == "brilliance" else "✗"
                            self._log(f"    {icon} S{tp['shota']}T{tp['trick']}: played {tp['played']} (AI: {tp['ai_choice']})")
                    if analysis.get("improvement", {}).get("details"):
                        self._log("  Trend:")
                        for d in analysis["improvement"]["details"]:
                            self._log(f"    → {d}")
                except Exception:
                    pass  # Don't break game over screen.
            else:
                result_text = "✓ Bid MET" if bid_met else "✗ Bid FAILED"
                canvas.create_text(w // 2, h // 2 - 30,
                                   text=f"Shota {self.shota_number} Complete",
                                   fill="#ffffff", font=("Segoe UI", 12, "bold"))
                canvas.create_text(w // 2, h // 2,
                                   text=f"{result_text} | T1:{self.team_tricks[0]} – T2:{self.team_tricks[1]}",
                                   fill=COLORS["gold"] if bid_met else "#ff6666",
                                   font=("Segoe UI", 10))
                canvas.create_text(w // 2, h // 2 + 30,
                                   text=f"Score: {self.game_scores[0]} – {self.game_scores[1]}",
                                   fill="#aaaaaa", font=("Segoe UI", 10))
                self._set_status(f"Shota {self.shota_number} done. Next starting...")
                self._schedule(2500, self._start_new_shota)
        except Exception:
            self._set_status(f"Shota done. Moving on...")
            if not game_over:
                self._schedule(2000, self._start_new_shota)

    def _reset_table(self):
        """Reset all visual elements for a fresh game."""
        self._centre_canvas.delete("all")
        for pid in [0, 1, 3]:
            if pid in self._player_canvases:
                self._player_canvases[pid].delete("all")
            if pid in self._player_status:
                self._player_status[pid].config(text="")
            if pid in self._player_bid_labels:
                self._player_bid_labels[pid].config(text="")
            if pid in self._player_frames:
                self._player_frames[pid].config(highlightthickness=0)
        self._human_canvas.delete("all")
        self._human_bid_label.config(text="")
        self._player_frames[HUMAN_ID].config(highlightthickness=0)
        self._safe_destroy(self._bid_btn_frame)
        self._bid_btn_frame = None
        for key in self._info_labels:
            self._info_labels[key].config(text="—")
        self._trump_display_label.config(text="", fg=COLORS["table_felt"])
        self._turn_indicator_active = False

    # ----------------------------------------------------------
    # Safe scheduler — wraps all after() callbacks to prevent silent hangs
    # ----------------------------------------------------------

    def _schedule(self, delay_ms, callback):
        """Schedule a callback with error protection. If it throws, log and continue."""
        def safe_wrapper():
            try:
                callback()
            except Exception as e:
                print(f"[HumanTab ERROR] {callback.__name__}: {e}")
                self._log(f"⚠ Error: {e}")
        self.root.after(delay_ms, safe_wrapper)

    # ----------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------

    def _safe_destroy(self, widget):
        """Safely destroy a widget — no crash if already destroyed or None."""
        if widget is None:
            return
        try:
            widget.destroy()
        except (tk.TclError, Exception):
            pass

    # Stub methods for interface compat.
    def set_status(self, *a): pass
    def set_trick_display(self, *a): pass
    def show_hand(self, *a, **kw): pass
    def show_bid_options(self, *a, **kw): pass
    def hide_bid_options(self): pass
