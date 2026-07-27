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
        self.root.state("zoomed")  # Maximize window.
        self.root.resizable(False, False)  # Prevent resizing.
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

        # Tab 1: Human vs AI
        human_tab_frame = tk.Frame(self.notebook, bg=COLORS["table_border"])
        self.notebook.add(human_tab_frame, text="  🧑 Human vs AI  ")

        from gui.human_tab import HumanTab
        self._human_tab = HumanTab(human_tab_frame, self.root)

        # Tab 2: AI Advisor
        advisor_tab_frame = tk.Frame(self.notebook, bg="#1a1a1a")
        self.notebook.add(advisor_tab_frame, text="  🔬 AI Advisor  ")

        from gui.advisor_tab import AdvisorTab
        self._advisor_tab = AdvisorTab(advisor_tab_frame, self.root)

        # Tab 3: Game Table
        game_tab = tk.Frame(self.notebook, bg=COLORS["table_border"])
        self.notebook.add(game_tab, text="  🃏 Game Table  ")
        self._build_game_tab(game_tab)

        # Tab 4: Stats & Lab
        stats_tab = tk.Frame(self.notebook, bg=COLORS["header_bg"])
        self.notebook.add(stats_tab, text="  📊 Stats & Lab  ")
        self._build_stats_tab(stats_tab)

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
                                 "Rule-Based", "Random", "Learning")
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
    # Stats & Lab tab
    # ----------------------------------------------------------

    def _build_stats_tab(self, parent) -> None:
        """Build the statistics and laboratory dashboard tab."""
        # Use a single scrollable frame with clear sections.
        parent.configure(bg="#1a1a1a")

        # ---- Top: Batch Controls ----
        controls = tk.Frame(parent, bg="#252525", bd=0, padx=16, pady=12)
        controls.pack(fill="x", padx=12, pady=(12, 6))

        tk.Label(controls, text="Run Experiment", font=("Segoe UI", 12, "bold"),
                 fg="#ffffff", bg="#252525").grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 10))

        # Games count
        tk.Label(controls, text="Games:", font=("Segoe UI", 10),
                 fg="#aaaaaa", bg="#252525").grid(row=1, column=0, padx=(0, 6))
        self._batch_count_var = tk.StringVar(value="10000")
        tk.Entry(controls, textvariable=self._batch_count_var, font=("Consolas", 11),
                 width=6, bg="#333333", fg="#ffffff", insertbackground="#fff",
                 bd=1, relief="solid").grid(row=1, column=1, padx=(0, 16))

        # Team 1 agent
        tk.Label(controls, text="Team 1:", font=("Segoe UI", 10),
                 fg=COLORS["score_team1"], bg="#252525").grid(row=1, column=2, padx=(0, 4))
        self._agent_t1_var = tk.StringVar(value="Learning")
        tk.OptionMenu(controls, self._agent_t1_var,
                      "Rule-Based", "Random", "Learning").grid(row=1, column=3, padx=(0, 16))

        # Team 2 agent
        tk.Label(controls, text="Team 2:", font=("Segoe UI", 10),
                 fg=COLORS["score_team2"], bg="#252525").grid(row=1, column=4, padx=(0, 4))
        self._agent_t2_var = tk.StringVar(value="Random")
        tk.OptionMenu(controls, self._agent_t2_var,
                      "Rule-Based", "Random", "Learning").grid(row=1, column=5, padx=(0, 16))

        # Buttons
        tk.Button(controls, text="▶  Run", command=self._run_batch,
                  font=("Segoe UI", 10, "bold"), fg="#fff", bg=COLORS["btn_green"],
                  bd=0, padx=16, pady=6, cursor="hand2").grid(row=1, column=6, padx=4)
        tk.Button(controls, text="↺  Reset", command=self._reset_stats,
                  font=("Segoe UI", 10, "bold"), fg="#fff", bg=COLORS["btn_grey"],
                  bd=0, padx=16, pady=6, cursor="hand2").grid(row=1, column=7, padx=4)

        # Progress
        self._batch_progress_label = tk.Label(controls, text="Ready",
                                              font=("Segoe UI", 9), fg="#888888", bg="#252525")
        self._batch_progress_label.grid(row=2, column=0, columnspan=8, sticky="w", pady=(8, 0))

        # ---- Middle: Results ----
        results_frame = tk.Frame(parent, bg="#1a1a1a")
        results_frame.pack(fill="both", expand=True, padx=12, pady=6)
        results_frame.columnconfigure(0, weight=1)
        results_frame.columnconfigure(1, weight=1)
        results_frame.rowconfigure(0, weight=1)

        # Left: Stats numbers
        stats_box = tk.Frame(results_frame, bg="#252525", bd=0, padx=16, pady=12)
        stats_box.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        tk.Label(stats_box, text="Results", font=("Segoe UI", 11, "bold"),
                 fg="#ffffff", bg="#252525").pack(anchor="w", pady=(0, 10))

        self._stat_labels = {}

        stats_sections = [
            ("GAMES", [
                ("Games Played", "games_played"),
                ("Team 1 Wins", "team_1_wins"),
                ("Team 2 Wins", "team_2_wins"),
                ("Draws", "draws"),
            ]),
            ("PERFORMANCE", [
                ("Team 1 Win Rate", "team_1_win_rate"),
                ("Team 2 Win Rate", "team_2_win_rate"),
                ("Avg Tricks (T1)", "avg_tricks_t1"),
                ("Avg Tricks (T2)", "avg_tricks_t2"),
            ]),
            ("BIDDING", [
                ("Bid Success Rate", "bid_success_rate"),
                ("Avg Bid Value", "avg_bid"),
                ("Dak Rate", "dak_rate"),
                ("Seeks", "seek_count"),
            ]),
        ]

        for section_title, items in stats_sections:
            tk.Label(stats_box, text=section_title, font=("Segoe UI", 8, "bold"),
                     fg="#666666", bg="#252525").pack(anchor="w", pady=(10, 4))

            for label_text, key in items:
                row = tk.Frame(stats_box, bg="#252525")
                row.pack(fill="x", pady=1)

                tk.Label(row, text=label_text, font=("Segoe UI", 10),
                         fg="#aaaaaa", bg="#252525", anchor="w").pack(side="left")
                val = tk.Label(row, text="—", font=("Consolas", 11, "bold"),
                               fg="#ffffff", bg="#252525", anchor="e")
                val.pack(side="right")
                self._stat_labels[key] = val

        # Right: Chart + Learning controls
        chart_box = tk.Frame(results_frame, bg="#252525", bd=0, padx=16, pady=12)
        chart_box.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        tk.Label(chart_box, text="Learning Progress", font=("Segoe UI", 11, "bold"),
                 fg="#ffffff", bg="#252525").pack(anchor="w", pady=(0, 6))

        # Progress line chart.
        self._chart_canvas = tk.Canvas(chart_box, bg="#1a1a1a", height=180,
                                       highlightthickness=0)
        self._chart_canvas.pack(fill="both", expand=True, pady=4)

        # Learning agent info.
        self._learning_info_label = tk.Label(
            chart_box, text="No learning agent active",
            font=("Segoe UI", 9), fg="#888888", bg="#252525")
        self._learning_info_label.pack(anchor="w", pady=(4, 6))

        # Model controls.
        model_row = tk.Frame(chart_box, bg="#252525")
        model_row.pack(fill="x", pady=(4, 0))

        tk.Button(model_row, text="💾 Save Model", command=self._save_model,
                  font=("Segoe UI", 9, "bold"), fg="#fff", bg="#1e88e5",
                  bd=0, padx=10, pady=4, cursor="hand2").pack(side="left", padx=3)
        tk.Button(model_row, text="📂 Load Model", command=self._load_model,
                  font=("Segoe UI", 9, "bold"), fg="#fff", bg="#1e88e5",
                  bd=0, padx=10, pady=4, cursor="hand2").pack(side="left", padx=3)
        tk.Button(model_row, text="🧹 Reset Brain", command=self._reset_brain,
                  font=("Segoe UI", 9, "bold"), fg="#fff", bg=COLORS["btn_red"],
                  bd=0, padx=10, pady=4, cursor="hand2").pack(side="left", padx=3)

        # Learning progress data.
        self._learning_win_history: list[float] = []
        self._learning_agent_instance = None

    def _update_stats_display(self) -> None:
        """Refresh the stats panel with current values."""
        s = self.stats
        updates = {
            "games_played": str(s.games_played),
            "team_1_wins": str(s.team_1_wins),
            "team_2_wins": str(s.team_2_wins),
            "draws": str(s.draws),
            "team_1_win_rate": f"{s.team_1_win_rate:.1f}%",
            "team_2_win_rate": f"{s.team_2_win_rate:.1f}%",
            "avg_tricks_t1": f"{s.avg_tricks_team_1:.1f}",
            "avg_tricks_t2": f"{s.avg_tricks_team_2:.1f}",
            "bid_success_rate": f"{s.bid_success_rate:.1f}%",
            "avg_bid": f"{s.avg_bid:.1f}",
            "dak_rate": f"{s.dak_rate:.1f}%",
            "seek_count": str(s.seek_count),
        }

        for key, value in updates.items():
            if key in self._stat_labels:
                self._stat_labels[key].config(text=value)

        # Update chart.
        self._draw_win_chart()

    def _draw_win_chart(self) -> None:
        """Draw learning progress with auto-scaled Y axis to show improvement clearly."""
        canvas = self._chart_canvas
        canvas.delete("all")

        w = canvas.winfo_width() or 400
        h = canvas.winfo_height() or 180

        history = self._learning_win_history

        if not history:
            if self.stats.games_played == 0:
                canvas.create_text(w // 2, h // 2, text="Click Run to start training",
                                   fill="#555555", font=("Segoe UI", 11))
            else:
                self._draw_bar_chart(canvas, w, h)
            return

        # Chart area.
        left = 50
        right = w - 20
        top = 30
        bottom = h - 25
        chart_w = right - left
        chart_h = bottom - top

        # Smooth the data.
        window = max(3, len(history) // 20)
        smoothed = []
        for i in range(len(history)):
            start = max(0, i - window + 1)
            avg = sum(history[start:i + 1]) / (i - start + 1)
            smoothed.append(avg)

        # Auto-scale Y axis to show the improvement range clearly.
        data_min = min(smoothed)
        data_max = max(smoothed)
        margin = max((data_max - data_min) * 0.3, 3)
        y_low = max(0, data_min - margin)
        y_high = min(100, data_max + margin)
        y_span = max(y_high - y_low, 1)

        # Background.
        canvas.create_rectangle(left, top, right, bottom, fill="#111111", outline="#333333")

        # Grid lines.
        num_lines = 4
        for i in range(num_lines + 1):
            frac = i / num_lines
            y = bottom - int(frac * chart_h)
            val = y_low + frac * y_span
            canvas.create_line(left, y, right, y, fill="#262626", width=1)
            canvas.create_text(left - 6, y, text=f"{val:.0f}%", anchor="e",
                               fill="#666666", font=("Consolas", 8))

        # 50% line if visible.
        if y_low < 50 < y_high:
            y50 = bottom - int((50 - y_low) / y_span * chart_h)
            canvas.create_line(left, y50, right, y50, fill="#444444", width=1, dash=(6, 3))

        # Build points.
        points = []
        for i, rate in enumerate(smoothed):
            x = left + int(i / max(1, len(smoothed) - 1) * chart_w)
            y = bottom - int((rate - y_low) / y_span * chart_h)
            y = max(top, min(bottom, y))
            points.append((x, y))

        if len(points) < 2:
            return

        # Filled area.
        fill_points = [(left, bottom)] + points + [(right, bottom)]
        canvas.create_polygon(fill_points, fill="#0d2e0d", outline="")

        # Main line.
        for i in range(1, len(points)):
            canvas.create_line(points[i-1][0], points[i-1][1],
                               points[i][0], points[i][1],
                               fill="#4caf50", width=2)

        # Start marker.
        sx, sy = points[0]
        canvas.create_oval(sx - 4, sy - 4, sx + 4, sy + 4, fill="#ff9800", outline="#fff", width=1)
        canvas.create_text(sx + 10, sy, text=f"{smoothed[0]:.1f}%", anchor="w",
                           fill="#ff9800", font=("Segoe UI", 9, "bold"))

        # End marker.
        ex, ey = points[-1]
        end_color = "#66ff66" if smoothed[-1] > smoothed[0] else "#ff6666"
        canvas.create_oval(ex - 5, ey - 5, ex + 5, ey + 5, fill=end_color, outline="#fff", width=1)
        canvas.create_text(ex - 10, ey - 14, text=f"{smoothed[-1]:.1f}%", anchor="e",
                           fill=end_color, font=("Segoe UI", 10, "bold"))

        # Top: improvement summary.
        improvement = smoothed[-1] - smoothed[0]
        sign = "+" if improvement >= 0 else ""
        imp_color = "#66ff66" if improvement >= 0 else "#ff6666"
        canvas.create_text(w // 2, 12,
                           text=f"Start: {smoothed[0]:.1f}% → Now: {smoothed[-1]:.1f}%  ({sign}{improvement:.1f}%)   •  {len(history)*10} games",
                           fill=imp_color, font=("Segoe UI", 9, "bold"))

    def _draw_bar_chart(self, canvas, w, h) -> None:
        """Fallback bar chart."""
        bar_width = 70
        gap = 50
        start_x = (w - (2 * bar_width + gap)) // 2
        bar_bottom = h - 30
        max_height = h - 60

        t1_rate = self.stats.team_1_win_rate / 100
        t1_h = max(4, int(t1_rate * max_height))
        canvas.create_rectangle(start_x, bar_bottom - t1_h,
                                start_x + bar_width, bar_bottom,
                                fill=COLORS["score_team1"], outline="")
        canvas.create_text(start_x + bar_width // 2, bar_bottom + 14,
                           text=f"T1: {self.stats.team_1_win_rate:.0f}%",
                           fill=COLORS["score_team1"], font=("Segoe UI", 10, "bold"))

        x2 = start_x + bar_width + gap
        t2_rate = self.stats.team_2_win_rate / 100
        t2_h = max(4, int(t2_rate * max_height))
        canvas.create_rectangle(x2, bar_bottom - t2_h,
                                x2 + bar_width, bar_bottom,
                                fill=COLORS["score_team2"], outline="")
        canvas.create_text(x2 + bar_width // 2, bar_bottom + 14,
                           text=f"T2: {self.stats.team_2_win_rate:.0f}%",
                           fill=COLORS["score_team2"], font=("Segoe UI", 10, "bold"))

    def _run_batch(self) -> None:
        """Run a batch of games in a background thread."""
        try:
            count = int(self._batch_count_var.get())
        except ValueError:
            count = 100

        self._batch_progress_label.config(text=f"Running {count} games...")
        self.root.update()

        # Run in background thread.
        thread = threading.Thread(target=self._batch_worker, args=(count,), daemon=True)
        thread.start()

    def _batch_worker(self, count: int) -> None:
        """Background worker that runs batch games with learning support."""
        from agents.random.random_agent import RandomAgent
        from agents.rule_based.rule_based_agent import RuleBasedAgent
        from agents.learning.learning_agent import LearningAgent
        from environments.wist.environment import WistEnvironment
        from environments.wist.round import Round
        from environments.wist.rules import trick_winner
        from environments.wist.setup import create_standard_players
        from environments.wist.tasmiya_engine import TasmiyaEngine
        from environments.wist.trick import Trick
        from environments.wist.scoring import score_shota, detect_seek

        t1_agent_name = self._agent_t1_var.get()
        t2_agent_name = self._agent_t2_var.get()

        # For learning agents, reuse the same instance (persists learning).
        if self._learning_agent_instance is None:
            self._learning_agent_instance = LearningAgent(training=True)

        # Update info immediately.
        self.root.after(0, lambda: self._learning_info_label.config(
            text=f"Training... Q-table: {self._learning_agent_instance.q_table_size} entries"))

        def make_agent(name):
            if name == "Rule-Based":
                return RuleBasedAgent()
            elif name == "Learning":
                return self._learning_agent_instance
            return RandomAgent()

        tasmiya_engine = TasmiyaEngine()
        window_wins = 0
        window_total = 0
        games_completed = 0

        i = 0
        while games_completed < count:
            i += 1
            players = create_standard_players()

            t1_agent = make_agent(t1_agent_name)
            t2_agent = make_agent(t2_agent_name)
            agents = [t1_agent, t2_agent, t1_agent, t2_agent]

            round_ = Round(players)
            round_.deal()

            if round_.has_card_based_dak():
                continue

            # Bidding.
            result = tasmiya_engine.run(
                players=players, agents=agents, sahib_al_qabool_id=0)

            if result.is_dak:
                self.stats.record_dak()
                if isinstance(t1_agent, LearningAgent):
                    t1_agent.reset_episode()
                if isinstance(t2_agent, LearningAgent):
                    t2_agent.reset_episode()
                continue

            # Play 13 tricks (manually, to give per-trick rewards).
            round_.state.trump_suit = result.trump_suit
            round_.state.winning_bidder_id = result.winning_bidder_id
            round_.next_leading_player_id = result.winning_bidder_id

            environment = WistEnvironment(round_.state)
            team_tricks = {0: 0, 1: 0}

            for _ in range(13):
                leader_id = round_.next_leading_player_id
                round_.state.current_trick = Trick(leading_player_id=leader_id)
                play_order = [(leader_id + j) % 4 for j in range(4)]

                for pid in play_order:
                    obs = environment.observe(pid)
                    action = agents[pid].act(obs)
                    environment.apply_action(action)

                trick = round_.state.current_trick
                winner = trick_winner(trick, round_.state.trump_suit)
                round_.state.completed_tricks.append(trick)
                round_.state.current_trick = None
                round_.next_leading_player_id = winner

                winner_team = players[winner].team_id
                team_tricks[winner_team] += 1

                # Reward learning agents.
                if isinstance(t1_agent, LearningAgent):
                    t1_agent.reward_trick(won=(winner_team == 0))
                if isinstance(t2_agent, LearningAgent) and t2_agent is not t1_agent:
                    t2_agent.reward_trick(won=(winner_team == 1))

            # Shota-level rewards.
            playing_team = result.playing_team_id
            bid = result.winning_bid_value
            t1_tricks = team_tricks[0]
            t2_tricks = team_tricks[1]

            bid_met_t1 = (t1_tricks >= bid) if playing_team == 0 else False
            bid_met_t2 = (t2_tricks >= bid) if playing_team == 1 else False

            if isinstance(t1_agent, LearningAgent):
                t1_agent.reward_shota(team_won_shota=(t1_tricks > t2_tricks), bid_met=bid_met_t1)
                t1_agent.decay_epsilon()
            if isinstance(t2_agent, LearningAgent) and t2_agent is not t1_agent:
                t2_agent.reward_shota(team_won_shota=(t2_tricks > t1_tricks), bid_met=bid_met_t2)
                t2_agent.decay_epsilon()

            # Record stats.
            winner_team = 0 if t1_tricks > t2_tricks else (1 if t2_tricks > t1_tricks else None)
            self.stats.record_game(winner_team=winner_team, score_1=t1_tricks, score_2=t2_tricks)

            bid_met = team_tricks.get(playing_team, 0) >= bid
            self.stats.record_shota(
                team_1_tricks=t1_tricks, team_2_tricks=t2_tricks,
                bid=bid, playing_team_id=playing_team, bid_met=bid_met)

            # Track learning progress window.
            window_total += 1
            if winner_team == 0:
                window_wins += 1

            if window_total >= 10:
                rate = window_wins / window_total * 100
                self._learning_win_history.append(rate)
                window_wins = 0
                window_total = 0

            games_completed += 1

            # Update progress every 50 games (for responsive UI).
            if games_completed % 50 == 0 or games_completed == count:
                self.root.after(0, self._batch_progress_update, games_completed, count)

        # Update learning info.
        if isinstance(t1_agent, LearningAgent):
            size = t1_agent.q_table_size
            eps = t1_agent.epsilon
            self.root.after(0, lambda: self._learning_info_label.config(
                text=f"Q-table: {size} entries | ε: {eps:.3f}"))

        self.root.after(0, self._batch_done, count)

    def _batch_progress_update(self, current: int, total: int) -> None:
        self._batch_progress_label.config(text=f"Progress: {current}/{total}")
        self._update_stats_display()
        # Update learning info.
        if self._learning_agent_instance:
            size = self._learning_agent_instance.q_table_size
            eps = self._learning_agent_instance.epsilon
            self._learning_info_label.config(
                text=f"Q-table: {size} entries | ε: {eps:.3f} | Games: {current}"
            )

    def _batch_done(self, count: int) -> None:
        self._batch_progress_label.config(
            text=f"✓ DONE — {count} games completed."
        )
        self._update_stats_display()
        if self._learning_agent_instance:
            size = self._learning_agent_instance.q_table_size
            eps = self._learning_agent_instance.epsilon
            self._learning_info_label.config(
                text=f"✓ Done | Q-table: {size} entries | ε: {eps:.3f}"
            )

    def _reset_stats(self) -> None:
        """Reset all collected statistics."""
        self.stats.reset()
        self._learning_win_history.clear()
        self._update_stats_display()
        self._batch_progress_label.config(text="Stats reset.")

    def _save_model(self) -> None:
        """Save the learning agent's brain to a file."""
        if self._learning_agent_instance is None:
            self._learning_info_label.config(text="No learning agent to save. Run a batch first.")
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Save Learning Agent Model",
        )
        if path:
            self._learning_agent_instance.save(path)
            self._learning_info_label.config(text=f"✓ Model saved: {path}")

    def _load_model(self) -> None:
        """Load a previously trained learning agent."""
        from tkinter import filedialog
        from agents.learning.learning_agent import LearningAgent
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")],
            title="Load Learning Agent Model",
        )
        if path:
            self._learning_agent_instance = LearningAgent.load(path, training=True)
            size = self._learning_agent_instance.q_table_size
            self._learning_info_label.config(text=f"✓ Model loaded: {size} entries")

    def _reset_brain(self) -> None:
        """Reset the learning agent to a blank state."""
        from agents.learning.learning_agent import LearningAgent
        self._learning_agent_instance = LearningAgent(training=True)
        self._learning_win_history.clear()
        self._learning_info_label.config(text="Brain reset — starting fresh.")
        self._draw_win_chart()

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
            size = agent.q_table_size
            # Show confirmation.
            self.set_current_trick(f"Model loaded ({size} entries)\nReady to play.")

    def _apply_agent_selection(self) -> None:
        """Apply the agent type dropdowns to the controller."""
        if hasattr(self, "_game_agent_vars"):
            type_map = {"Rule-Based": "rule_based", "Random": "random", "Learning": "learning"}
            for i, var in enumerate(self._game_agent_vars):
                self.controller.agent_types[i] = type_map.get(var.get(), "rule_based")

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

        # Build role text
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
            elif is_qabool:
                role_lbl.config(fg=COLORS["gold"])
            elif is_first_shooter:
                role_lbl.config(fg="#66bb6a")
            else:
                role_lbl.config(fg=COLORS["text_muted"])

        # Highlight frame on win
        if frame:
            if "won" in message.lower():
                frame.config(bg=COLORS["player_winner"])
                self._recolor_frame(frame, COLORS["player_winner"])
            elif is_qabool:
                frame.config(bg=COLORS["player_active"])
                self._recolor_frame(frame, COLORS["player_active"])
            else:
                frame.config(bg=COLORS["player_bg"])
                self._recolor_frame(frame, COLORS["player_bg"])

    def set_player_bid(self, player_index: int, bid: str) -> None:
        if 0 <= player_index < 4:
            lbl = self.player_bid_labels[player_index]
            if lbl:
                if bid and bid != "-":
                    lbl.config(text=f"Bid: {bid}", fg=COLORS["gold"])
                else:
                    lbl.config(text="", fg=COLORS["text_dim"])

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
