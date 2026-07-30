"""
Sudanese Wist AI Laboratory — GUI

A visual table for observing AI agents play Sudanese Wist.
Everything visible at a glance — no CLI output needed.
"""

import tkinter as tk
from tkinter import ttk
import threading

from gui.controller import SimulationController
from gui.stats import GameStats
from gui.colors import COLORS


# Colors imported from gui.colors.


# ---------------------------------------------------------------
# App
# ---------------------------------------------------------------


class WistAILabApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Sudanese Wist — AI Laboratory")
        # Fixed size, centred on screen.
        win_w, win_h = 1280, 800
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2 - 30
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.root.resizable(False, False)
        self.root.configure(bg=COLORS["table_border"])
        self.root.minsize(1050, 720)

        self.controller = SimulationController(self)

        # Widget references
        self.status_label = None
        self.shota_info_labels = {}
        self.current_trick_label = None
        self.trick_counter_label = None
        self.shota_counter_label = None
        self.deal_counter_label = None
        self.score_team1_label = None
        self.score_team2_label = None

        self.player_name_labels = [None, None, None, None]
        self.player_role_labels = [None, None, None, None]
        self.player_bid_labels = [None, None, None, None]
        self.player_card_frames = [None, None, None, None]
        self.player_frames = [None, None, None, None]
        self.player_tricks_labels = [None, None, None, None]

        self.played_card_labels = {}

        # Stats tracking.
        self.stats = GameStats()

        self._build_layout()

    # ==========================================================
    # LAYOUT
    # ==========================================================

    def _build_layout(self) -> None:
        # Top bar: title + shota info
        self._build_top_bar()

        # Tabbed notebook: Game Table | Stats & Lab
        style = ttk.Style()
        style.configure("Lab.TNotebook", background=COLORS["header_bg"])
        style.configure("Lab.TNotebook.Tab", font=("Segoe UI", 10, "bold"),
                        padding=[12, 4])

        self.notebook = ttk.Notebook(self.root, style="Lab.TNotebook")
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(2, 4))

        # Tab 1: Stats & Lab
        stats_tab = tk.Frame(self.notebook, bg=COLORS["header_bg"])
        self.notebook.add(stats_tab, text="  📊 Stats & Lab  ")

        from gui.stats_tab import StatsTab
        self._stats_tab = StatsTab(stats_tab, self.root, self.stats)

        # Tab 2: Play for AI
        advisor_tab_frame = tk.Frame(self.notebook, bg="#1a1a1a")
        self.notebook.add(advisor_tab_frame, text="  🤖 Play for AI  ")

        from gui.advisor_tab import AdvisorTab
        self._advisor_tab = AdvisorTab(advisor_tab_frame, self.root)

        # Tab 3: Game Table (AI vs AI)
        game_tab = tk.Frame(self.notebook, bg=COLORS["table_border"])
        self.notebook.add(game_tab, text="  🃏 Game Table  ")
        self._build_game_tab(game_tab)

        # Bottom: controls (visible on all tabs)
        self._build_controls()

    def _build_game_tab(self, parent) -> None:
        """Build the game table inside the given tab frame."""

        # Agent selector row at the top of the game tab.
        selector_frame = tk.Frame(parent, bg=COLORS["header_bg"], height=32)
        selector_frame.pack(fill="x", padx=4, pady=(4, 0))

        tk.Label(selector_frame, text="Agents:", font=("Segoe UI", 9),
                 fg=COLORS["text_dim"], bg=COLORS["header_bg"]).pack(side="left", padx=(8, 4))

        self._game_agent_vars = []
        player_labels = ["P1", "P2", "P3", "P4"]
        for i, plbl in enumerate(player_labels):
            tk.Label(selector_frame, text=plbl, font=("Segoe UI", 8),
                     fg=COLORS["text_muted"], bg=COLORS["header_bg"]).pack(side="left", padx=(8, 2))
            var = tk.StringVar(value="Rule-Based")
            menu = tk.OptionMenu(selector_frame, var,
                                 "Rule-Based", "Learning")
            menu.config(font=("Segoe UI", 8), width=8)
            menu.pack(side="left", padx=2)
            self._game_agent_vars.append(var)

        tk.Button(selector_frame, text="📂 Load Model", command=self._load_model_for_game,
                  font=("Segoe UI", 8), fg="#fff", bg="#1e88e5",
                  bd=0, padx=8, pady=2, cursor="hand2").pack(side="right", padx=8)

        # Main table area
        table = tk.Frame(parent, bg=COLORS["table_felt"], bd=4, relief="ridge",
                         highlightbackground=COLORS["table_border"], highlightthickness=3)
        table.pack(fill="both", expand=True, padx=4, pady=4)
        table.columnconfigure(0, weight=1)
        table.columnconfigure(1, weight=2)
        table.columnconfigure(2, weight=1)
        table.rowconfigure(0, weight=1)
        table.rowconfigure(1, weight=2)
        table.rowconfigure(2, weight=1)

        # Players around the table
        p1 = self._create_player_area(table, 0, "Player 1", "Team 1")
        p1.grid(row=0, column=1, sticky="ew", padx=6, pady=4)

        p4 = self._create_player_area(table, 3, "Player 4", "Team 2")
        p4.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        p2 = self._create_player_area(table, 1, "Player 2", "Team 2")
        p2.grid(row=1, column=2, sticky="nsew", padx=4, pady=4)

        p3 = self._create_player_area(table, 2, "Player 3", "Team 1")
        p3.grid(row=2, column=1, sticky="ew", padx=6, pady=4)

        # Centre: played cards + trick info
        self._build_centre(table)

    # ----------------------------------------------------------
    # Top bar
    # ----------------------------------------------------------

    def _build_top_bar(self) -> None:
        bar = tk.Frame(self.root, bg=COLORS["header_bg"], height=70)
        bar.pack(fill="x", padx=0, pady=0)
        bar.pack_propagate(False)

        # Left: title
        left = tk.Frame(bar, bg=COLORS["header_bg"])
        left.pack(side="left", padx=12, fill="y")

        tk.Label(left, text="Sudanese Wist", font=("Segoe UI", 14, "bold"),
                 fg=COLORS["text_white"], bg=COLORS["header_bg"]).pack(anchor="w", pady=(12, 0))
        tk.Label(left, text="AI Laboratory",
                 font=("Segoe UI", 9), fg=COLORS["text_dim"],
                 bg=COLORS["header_bg"]).pack(anchor="w")

        # Right: all game info
        right = tk.Frame(bar, bg=COLORS["header_bg"])
        right.pack(side="right", padx=12, fill="y")

        info_row = tk.Frame(right, bg=COLORS["header_bg"])
        info_row.pack(anchor="e", pady=(10, 0))

        def add_info_item(parent, label_text, width=6, fg=COLORS["text_light"], default="—"):
            item = tk.Frame(parent, bg=COLORS["header_bg"])
            item.pack(side="left", padx=6)
            tk.Label(item, text=label_text, font=("Segoe UI", 8),
                     fg=COLORS["text_dim"], bg=COLORS["header_bg"]).pack()
            val = tk.Label(item, text=default, font=("Segoe UI", 11, "bold"),
                           fg=fg, bg=COLORS["header_bg"], width=width, anchor="center")
            val.pack()
            return val

        # Shota | Deal | Qabool │ Bid | Shooter | Trump │ Trick | Winner │ Team 1 | Team 2
        self.shota_counter_label = add_info_item(info_row, "Shota", width=5, default="1 / 5")
        self.deal_counter_label = add_info_item(info_row, "Deal", width=3, default="1")
        self.shota_info_labels["qabool"] = add_info_item(info_row, "Qabool", width=8, fg=COLORS["gold"])

        tk.Label(info_row, text="│", font=("Segoe UI", 12),
                 fg=COLORS["text_dim"], bg=COLORS["header_bg"]).pack(side="left", padx=2)

        self.shota_info_labels["bid"] = add_info_item(info_row, "Bid", width=4, fg=COLORS["gold"])
        self.shota_info_labels["first_shooter"] = add_info_item(info_row, "Shooter", width=8, fg=COLORS["gold"])
        self.shota_info_labels["trump"] = add_info_item(info_row, "Trump", width=8, fg=COLORS["gold"])

        tk.Label(info_row, text="│", font=("Segoe UI", 12),
                 fg=COLORS["text_dim"], bg=COLORS["header_bg"]).pack(side="left", padx=2)

        self.trick_counter_label = add_info_item(info_row, "Trick", width=5, default="— / 13")
        self.shota_info_labels["winner"] = add_info_item(info_row, "Winner", width=7, fg=COLORS["gold"])

        tk.Label(info_row, text="│", font=("Segoe UI", 12),
                 fg=COLORS["text_dim"], bg=COLORS["header_bg"]).pack(side="left", padx=2)

        self.score_team1_label = add_info_item(info_row, "Team 1", width=4, fg=COLORS["score_team1"], default="0")
        self.score_team2_label = add_info_item(info_row, "Team 2", width=4, fg=COLORS["score_team2"], default="0")

    # ----------------------------------------------------------
    # Centre trick area
    # ----------------------------------------------------------

    def _build_centre(self, parent) -> None:
        from gui.card_widget import draw_card, parse_card_text, CARD_LARGE_WIDTH, CARD_LARGE_HEIGHT

        frame = tk.Frame(parent, bg=COLORS["centre_bg"], bd=2, relief="sunken",
                         width=320, height=240)
        frame.grid(row=1, column=1, sticky="nsew", padx=8, pady=8)
        frame.grid_propagate(False)

        # Use a single Canvas for the centre trick area.
        self._centre_canvas = tk.Canvas(frame, bg=COLORS["centre_bg"],
                                        highlightthickness=0)
        self._centre_canvas.pack(fill="both", expand=True)

        # Trick status label (over the canvas).
        self.current_trick_label = tk.Label(
            frame, text="Waiting to start",
            font=("Segoe UI", 9, "bold"),
            fg=COLORS["text_light"], bg=COLORS["centre_bg"],
            justify="center",
        )
        self.current_trick_label.place(relx=0.5, rely=0.5, anchor="center")

        # Store card data for redrawing.
        self._centre_cards: dict[int, str] = {}  # player_id → card_text

    # ----------------------------------------------------------
    # Player area
    # ----------------------------------------------------------

    def _create_player_area(self, parent, player_index: int, name: str, team: str):
        # Players 1 and 3 (top/bottom) need more width for horizontal cards.
        # Players 2 and 4 (left/right) need more height for 4 suit rows.
        if player_index in (1, 3):  # side players
            frame = tk.Frame(parent, bg=COLORS["player_bg"], bd=1, relief="groove",
                             padx=6, pady=4, width=200, height=160)
        else:  # top/bottom players
            frame = tk.Frame(parent, bg=COLORS["player_bg"], bd=1, relief="groove",
                             padx=6, pady=4, width=350, height=130)
        frame.pack_propagate(False)  # Fixed size — no jumping.

        # Row 1: name + team + tricks won
        top_row = tk.Frame(frame, bg=COLORS["player_bg"])
        top_row.pack(fill="x")

        name_lbl = tk.Label(top_row, text=name, font=("Segoe UI", 10, "bold"),
                            fg=COLORS["text_white"], bg=COLORS["player_bg"])
        name_lbl.pack(side="left")

        team_color = COLORS["score_team1"] if "1" in team else COLORS["score_team2"]
        tk.Label(top_row, text=f"  {team}", font=("Segoe UI", 8),
                 fg=team_color, bg=COLORS["player_bg"]).pack(side="left")

        # Tricks won (face-down pile)
        tricks_lbl = tk.Label(top_row, text="", font=("Segoe UI", 9),
                              fg=COLORS["gold"], bg=COLORS["player_bg"])
        tricks_lbl.pack(side="right")
        self.player_tricks_labels[player_index] = tricks_lbl

        # Row 2: role + bid
        info_row = tk.Frame(frame, bg=COLORS["player_bg"])
        info_row.pack(fill="x", pady=(2, 2))

        role_lbl = tk.Label(info_row, text="", font=("Segoe UI", 8),
                            fg=COLORS["text_muted"], bg=COLORS["player_bg"])
        role_lbl.pack(side="left")
        self.player_role_labels[player_index] = role_lbl

        bid_lbl = tk.Label(info_row, text="", font=("Consolas", 9, "bold"),
                           fg=COLORS["text_dim"], bg=COLORS["player_bg"])
        bid_lbl.pack(side="right")
        self.player_bid_labels[player_index] = bid_lbl

        # Row 3: cards (grouped by suit)
        card_frame = tk.Frame(frame, bg=COLORS["player_bg"])
        card_frame.pack(fill="x", pady=(2, 2))

        self.player_name_labels[player_index] = name_lbl
        self.player_card_frames[player_index] = card_frame
        self.player_frames[player_index] = frame

        return frame

    # ----------------------------------------------------------
    # Stats tab is in gui/stats_tab.py (StatsTab class)
    # ----------------------------------------------------------

    def _update_stats_display(self) -> None:
        """Delegate to stats tab."""
        if hasattr(self, "_stats_tab"):
            self._stats_tab.update_display()

    # ----------------------------------------------------------
    # Controls
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # Controls
    # ----------------------------------------------------------

    def _build_controls(self) -> None:
        bar = tk.Frame(self.root, bg=COLORS["header_bg"], height=46)
        bar.pack(fill="x", padx=0, pady=0)
        bar.pack_propagate(False)

        btn_frame = tk.Frame(bar, bg=COLORS["header_bg"])
        btn_frame.pack(anchor="center", pady=8)

        def add_btn(text, cmd, bg, fg="#fff"):
            btn = tk.Button(
                btn_frame, text=text, command=cmd,
                font=("Segoe UI", 9, "bold"),
                fg=fg, bg=bg, activeforeground=fg, activebackground=bg,
                bd=0, padx=12, pady=4, cursor="hand2", relief="flat",
            )
            btn.pack(side="left", padx=3)

        def add_sep():
            tk.Label(btn_frame, text="│", font=("Segoe UI", 12),
                     fg=COLORS["text_dim"], bg=COLORS["header_bg"]).pack(side="left", padx=6)

        # Mode buttons
        add_btn("▶▶ Auto Game", self.controller.cmd_auto_game, COLORS["btn_green"])
        add_btn("▶  Auto Shota", self.controller.cmd_auto_shota, COLORS["btn_green"])
        add_btn("🔧 Manual Steps", self.controller.cmd_manual, COLORS["btn_orange"])

        add_sep()

        # Control buttons
        add_btn("⏵ Continue", self.controller.cmd_continue, COLORS["btn_green"])
        add_btn("→ Next", self.controller.cmd_next, COLORS["btn_orange"])
        add_btn("⏸ Pause", self.controller.cmd_pause, COLORS["btn_green"])
        add_btn("⏹ Stop", self.controller.cmd_stop, COLORS["btn_green"])

        add_sep()

        # Reset
        add_btn("↺ Reset", self.controller.cmd_reset, COLORS["btn_grey"])

    # ==========================================================
    # PUBLIC API (called by controller)
    # ==========================================================

    def _load_model_for_game(self) -> None:
        """Load a trained model for use in the Game Table."""
        from tkinter import filedialog
        from agents.learning.learning_agent import LearningAgent
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")],
            title="Load Learning Agent for Game Table",
        )
        if path:
            agent = LearningAgent.load(path, training=False)
            self.controller._learning_agent_cache = agent
            self.set_current_trick(f"Model loaded ({agent.q_table_size} entries)")

    def _apply_agent_selection(self) -> None:
        """Apply the agent type dropdowns to the controller."""
        if hasattr(self, "_game_agent_vars"):
            type_map = {"Rule-Based": "rule_based", "Random": "random", "Learning": "learning"}
            for i, var in enumerate(self._game_agent_vars):
                self.controller.agent_types[i] = type_map.get(var.get(), "rule_based")

    def set_status(self, message: str) -> None:
        """No-op — status removed from UI. Info shown on table only."""
        pass

    def set_shota_info(self, trump: str, qabool: str = "-",
                       bid: str = "-", first_shooter: str = "-") -> None:
        mapping = {
            "trump": trump,
            "bid": bid,
            "qabool": qabool,
            "first_shooter": first_shooter,
        }
        for key, val in mapping.items():
            if key in self.shota_info_labels:
                display = val if val != "-" else "—"
                self.shota_info_labels[key].config(text=display)

                # Color the trump label by suit.
                if key == "trump" and val != "-":
                    if "♥" in val or "♦" in val:
                        self.shota_info_labels[key].config(fg="#ff5252")
                    elif "♠" in val or "♣" in val:
                        self.shota_info_labels[key].config(fg="#ffffff")
                    else:
                        self.shota_info_labels[key].config(fg=COLORS["gold"])
                elif key == "trump":
                    self.shota_info_labels[key].config(fg=COLORS["gold"])

    def set_game_score(self, team_1: int, team_2: int, shotas: int) -> None:
        """Update the game score display."""
        if hasattr(self, "score_team1_label") and self.score_team1_label:
            self.score_team1_label.config(text=str(team_1))
        if hasattr(self, "score_team2_label") and self.score_team2_label:
            self.score_team2_label.config(text=str(team_2))
        if hasattr(self, "shota_counter_label") and self.shota_counter_label:
            self.shota_counter_label.config(text=f"{shotas} / 5")

    def set_deal_number(self, deal: int) -> None:
        """Update the deal counter (1st, 2nd, 3rd within a shota)."""
        if hasattr(self, "deal_counter_label") and self.deal_counter_label:
            self.deal_counter_label.config(text=str(deal))

    def set_current_trick(self, message: str) -> None:
        if self.current_trick_label:
            self.current_trick_label.config(text=message)
        # Update trick counter
        if self.trick_counter_label:
            import re
            m = re.search(r"trick\s*(\d+)", message, re.IGNORECASE)
            if m:
                self.trick_counter_label.config(text=f"{m.group(1)} / 13")

    def show_game_over(self, winner_team: int, score_1: int, score_2: int) -> None:
        """Show a decorative game-over display in the centre."""
        if self.current_trick_label:
            if winner_team > 0:
                winner_text = f"Team {winner_team} Wins!"
            else:
                winner_text = "It's a Draw!"

            self.current_trick_label.config(
                text=(
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"🏆  GAME  OVER  🏆\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"{winner_text}\n\n"
                    f"Final Score\n"
                    f"Team 1: {score_1}  │  Team 2: {score_2}\n"
                    f"━━━━━━━━━━━━━━━━━━━━"
                ),
                font=("Segoe UI", 12, "bold"),
                fg=COLORS["gold"],
            )
        # Clear played cards
        self.clear_played_cards()

    def set_player_status(self, player_index: int, message: str,
                          is_qabool: bool = False, is_first_shooter: bool = False) -> None:
        if not (0 <= player_index < 4):
            return

        role_lbl = self.player_role_labels[player_index]
        name_lbl = self.player_name_labels[player_index]
        frame = self.player_frames[player_index]

        # Build role text — Qabool is ALWAYS shown.
        roles = []
        if is_qabool:
            roles.append("👑 Qabool")
        if is_first_shooter:
            roles.append("🎯 Shooter")
        if "won" in message.lower():
            roles.append("🏆 Won")

        role_text = " | ".join(roles) if roles else message

        if role_lbl:
            role_lbl.config(text=role_text)
            if "won" in message.lower():
                role_lbl.config(fg="#ffd54f")
            elif is_first_shooter:
                role_lbl.config(fg="#66bb6a")
            elif is_qabool:
                role_lbl.config(fg=COLORS["gold"])
            else:
                role_lbl.config(fg=COLORS["text_muted"])

        # Highlight frame based on role — Qabool = outer gold border, Shooter = inner green border.
        if frame:
            if "won" in message.lower():
                frame.config(bg=COLORS["player_winner"], highlightthickness=0, bd=0)
                self._recolor_frame(frame, COLORS["player_winner"])
            elif is_first_shooter and is_qabool:
                # Both! Outer gold (highlightbackground) + inner green (bd).
                frame.config(bg="#1a4a1a",
                             highlightbackground=COLORS["gold"], highlightthickness=3,
                             bd=2, relief="solid")
                self._recolor_frame(frame, "#1a4a1a")
            elif is_first_shooter:
                # Green border only.
                frame.config(bg="#1a4a1a",
                             highlightbackground="#66bb6a", highlightthickness=2,
                             bd=0)
                self._recolor_frame(frame, "#1a4a1a")
            elif is_qabool:
                # Gold border only.
                frame.config(bg=COLORS["player_active"],
                             highlightbackground=COLORS["gold"], highlightthickness=3,
                             bd=0)
                self._recolor_frame(frame, COLORS["player_active"])
            else:
                frame.config(bg=COLORS["player_bg"], highlightthickness=0, bd=0)
                self._recolor_frame(frame, COLORS["player_bg"])

    def set_player_bid(self, player_index: int, bid: str) -> None:
        if 0 <= player_index < 4:
            lbl = self.player_bid_labels[player_index]
            if lbl:
                if not bid or bid == "-":
                    lbl.config(text="", fg=COLORS["text_dim"])
                elif bid.lower() == "pass":
                    lbl.config(text="Pass", fg=COLORS["text_dim"])
                else:
                    lbl.config(text=f"Bid: {bid}", fg=COLORS["gold"])

    def set_player_hand(self, player_index: int, cards) -> None:
        """Display cards as drawn mini-cards grouped by suit."""
        from gui.card_widget import draw_card, parse_card_text, CARD_MINI_WIDTH, CARD_MINI_HEIGHT

        if isinstance(cards, str):
            cards = [cards]

        if not (0 <= player_index < 4):
            return

        frame = self.player_card_frames[player_index]
        if frame is None:
            return

        for w in frame.winfo_children():
            w.destroy()

        if not cards:
            tk.Label(frame, text="No cards", font=("Segoe UI", 8),
                     fg=COLORS["text_dim"], bg=frame.cget("bg")).pack(anchor="w")
            return

        # Group by suit: ♠ ♥ ♣ ♦
        suit_groups = {"♠": [], "♥": [], "♣": [], "♦": []}
        for ct in cards:
            for sym in suit_groups:
                if sym in ct:
                    suit_groups[sym].append(ct)
                    break

        # Draw each suit row as a canvas.
        for sym in ["♠", "♥", "♣", "♦"]:
            group = suit_groups.get(sym, [])
            if not group:
                continue

            # Canvas for this suit row.
            row_canvas = tk.Canvas(frame, bg=frame.cget("bg"),
                                   height=CARD_MINI_HEIGHT + 4,
                                   highlightthickness=0)
            row_canvas.pack(fill="x", pady=1)

            spacing = min(CARD_MINI_WIDTH + 2, 26)
            for i, ct in enumerate(group):
                rank, suit = parse_card_text(ct)
                draw_card(row_canvas, 2 + i * spacing, 2, rank, suit,
                          width=CARD_MINI_WIDTH, height=CARD_MINI_HEIGHT)

    def set_played_cards(self, played_cards) -> None:
        from gui.card_widget import draw_card, parse_card_text, CARD_LARGE_WIDTH, CARD_LARGE_HEIGHT

        self._centre_cards = {}
        for player_id, card_text in played_cards:
            self._centre_cards[player_id] = card_text

        self._redraw_centre_cards()

    def clear_played_cards(self) -> None:
        self._centre_cards = {}
        if hasattr(self, "_centre_canvas"):
            self._centre_canvas.delete("all")

    def _redraw_centre_cards(self) -> None:
        """Redraw the centre trick area with card graphics."""
        from gui.card_widget import draw_card, draw_card_back, parse_card_text, CARD_LARGE_WIDTH, CARD_LARGE_HEIGHT

        canvas = self._centre_canvas
        canvas.delete("all")

        w = canvas.winfo_width() or 300
        h = canvas.winfo_height() or 220

        cw = CARD_LARGE_WIDTH
        ch = CARD_LARGE_HEIGHT

        # Card positions: P1=top, P2=right, P3=bottom, P4=left.
        positions = {
            0: (w // 2 - cw // 2, 8),                    # top centre
            1: (w - cw - 12, h // 2 - ch // 2),         # right
            2: (w // 2 - cw // 2, h - ch - 8),          # bottom centre
            3: (12, h // 2 - ch // 2),                   # left
        }

        for player_id, (x, y) in positions.items():
            if player_id in self._centre_cards:
                ct = self._centre_cards[player_id]
                rank, suit = parse_card_text(ct)
                draw_card(canvas, x, y, rank, suit,
                          width=cw, height=ch)
                # Player label above/below card.
                canvas.create_text(x + cw // 2, y - 6 if player_id == 2 else y + ch + 10,
                                   text=f"P{player_id + 1}", fill="#aaaaaa",
                                   font=("Segoe UI", 7))
            else:
                # Empty slot — show placeholder.
                canvas.create_rectangle(x, y, x + cw, y + ch,
                                        fill="#2a4a2a", outline="#3a5a3a", dash=(3, 3))
                canvas.create_text(x + cw // 2, y + ch // 2,
                                   text=f"P{player_id + 1}", fill="#5a8a5a",
                                   font=("Segoe UI", 9))

    def set_tricks_won(self, player_tricks: list[int]) -> None:
        """Show face-down won trick piles as stacked rectangles."""
        for idx, count in enumerate(player_tricks):
            lbl = self.player_tricks_labels[idx]
            if lbl:
                if count > 0:
                    # Show as small stacked cards: 🂠 repeated
                    pile = "🂠" * min(count, 13)
                    lbl.config(text=f"{pile} ({count})")
                else:
                    lbl.config(text="")

    def reset_player_statuses(self) -> None:
        for i in range(4):
            self.set_player_status(i, "Waiting")
            self.set_player_bid(i, "-")
            self.set_player_hand(i, [])
            if self.player_tricks_labels[i]:
                self.player_tricks_labels[i].config(text="")
        self.clear_played_cards()

    def log(self, message: str) -> None:
        """No-op: we removed the log panel. All info is visual."""
        pass

    def clear_log(self) -> None:
        """No-op."""
        pass

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _recolor_frame(self, frame, bg: str) -> None:
        for child in frame.winfo_children():
            try:
                if isinstance(child, tk.Frame):
                    child.config(bg=bg)
                    self._recolor_frame(child, bg)
                elif isinstance(child, tk.Label):
                    # Don't recolor card labels (white bg)
                    if child.cget("bg") != COLORS["card_bg"]:
                        child.config(bg=bg)
            except tk.TclError:
                pass

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    app = WistAILabApp()
    app.run()
