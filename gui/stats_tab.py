"""
Stats & Lab tab — batch runner, learning progress chart, model management.

Extracted from app.py for maintainability.
"""

import tkinter as tk
import threading

from gui.colors import COLORS
from gui.stats import GameStats


class StatsTab:
    """Stats & Lab dashboard tab."""

    def __init__(self, parent: tk.Frame, root: tk.Tk, stats: GameStats) -> None:
        self.parent = parent
        self.root = root
        self.stats = stats

        self._learning_win_history: list[float] = []
        self._learning_agent_instance = None

        self._build()

    def _build(self) -> None:
        self.parent.configure(bg="#1a1a1a")

        # ---- Top: Batch Controls ----
        controls = tk.Frame(self.parent, bg="#252525", bd=0, padx=16, pady=12)
        controls.pack(fill="x", padx=12, pady=(12, 6))

        tk.Label(controls, text="Run Experiment", font=("Segoe UI", 12, "bold"),
                 fg="#ffffff", bg="#252525").grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 10))

        tk.Label(controls, text="Games:", font=("Segoe UI", 10),
                 fg="#aaaaaa", bg="#252525").grid(row=1, column=0, padx=(0, 6))
        self._batch_count_var = tk.StringVar(value="10000")
        tk.Entry(controls, textvariable=self._batch_count_var, font=("Consolas", 11),
                 width=6, bg="#333333", fg="#ffffff", insertbackground="#fff",
                 bd=1, relief="solid").grid(row=1, column=1, padx=(0, 16))

        tk.Label(controls, text="Team 1:", font=("Segoe UI", 10),
                 fg=COLORS["score_team1"], bg="#252525").grid(row=1, column=2, padx=(0, 4))
        self._agent_t1_var = tk.StringVar(value="Learning")
        tk.OptionMenu(controls, self._agent_t1_var,
                      "Rule-Based", "Random", "Learning").grid(row=1, column=3, padx=(0, 16))

        tk.Label(controls, text="Team 2:", font=("Segoe UI", 10),
                 fg=COLORS["score_team2"], bg="#252525").grid(row=1, column=4, padx=(0, 4))
        self._agent_t2_var = tk.StringVar(value="Random")
        tk.OptionMenu(controls, self._agent_t2_var,
                      "Rule-Based", "Random", "Learning").grid(row=1, column=5, padx=(0, 16))

        tk.Button(controls, text="▶  Run", command=self._run_batch,
                  font=("Segoe UI", 10, "bold"), fg="#fff", bg=COLORS["btn_green"],
                  bd=0, padx=16, pady=6, cursor="hand2").grid(row=1, column=6, padx=4)
        tk.Button(controls, text="↺  Reset", command=self._reset_stats,
                  font=("Segoe UI", 10, "bold"), fg="#fff", bg=COLORS["btn_grey"],
                  bd=0, padx=16, pady=6, cursor="hand2").grid(row=1, column=7, padx=4)

        self._batch_progress_label = tk.Label(controls, text="Ready",
                                              font=("Segoe UI", 9), fg="#888888", bg="#252525")
        self._batch_progress_label.grid(row=2, column=0, columnspan=8, sticky="w", pady=(8, 0))

        # ---- Middle: Results ----
        results_frame = tk.Frame(self.parent, bg="#1a1a1a")
        results_frame.pack(fill="both", expand=True, padx=12, pady=6)
        results_frame.columnconfigure(0, weight=1)
        results_frame.columnconfigure(1, weight=1)
        results_frame.rowconfigure(0, weight=1)

        # Left: Stats numbers.
        stats_box = tk.Frame(results_frame, bg="#252525", bd=0, padx=16, pady=12)
        stats_box.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        tk.Label(stats_box, text="Results", font=("Segoe UI", 11, "bold"),
                 fg="#ffffff", bg="#252525").pack(anchor="w", pady=(0, 10))

        self._stat_labels = {}
        stats_sections = [
            ("GAMES", [
                ("Games Played", "games_played"), ("Team 1 Wins", "team_1_wins"),
                ("Team 2 Wins", "team_2_wins"), ("Draws", "draws"),
            ]),
            ("PERFORMANCE", [
                ("Team 1 Win Rate", "team_1_win_rate"), ("Team 2 Win Rate", "team_2_win_rate"),
                ("Avg Tricks (T1)", "avg_tricks_t1"), ("Avg Tricks (T2)", "avg_tricks_t2"),
            ]),
            ("BIDDING", [
                ("Bid Success Rate", "bid_success_rate"), ("Avg Bid Value", "avg_bid"),
                ("Dak Rate", "dak_rate"), ("Seeks", "seek_count"),
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

        # Right: Chart + controls.
        chart_box = tk.Frame(results_frame, bg="#252525", bd=0, padx=16, pady=12)
        chart_box.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        tk.Label(chart_box, text="Learning Progress", font=("Segoe UI", 11, "bold"),
                 fg="#ffffff", bg="#252525").pack(anchor="w", pady=(0, 6))

        self._chart_canvas = tk.Canvas(chart_box, bg="#1a1a1a", height=180,
                                       highlightthickness=0)
        self._chart_canvas.pack(fill="both", expand=True, pady=4)

        self._learning_info_label = tk.Label(
            chart_box, text="No learning agent active",
            font=("Segoe UI", 9), fg="#888888", bg="#252525")
        self._learning_info_label.pack(anchor="w", pady=(4, 6))

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

    # ----------------------------------------------------------
    # Stats display
    # ----------------------------------------------------------

    def update_display(self) -> None:
        s = self.stats
        updates = {
            "games_played": str(s.games_played), "team_1_wins": str(s.team_1_wins),
            "team_2_wins": str(s.team_2_wins), "draws": str(s.draws),
            "team_1_win_rate": f"{s.team_1_win_rate:.1f}%",
            "team_2_win_rate": f"{s.team_2_win_rate:.1f}%",
            "avg_tricks_t1": f"{s.avg_tricks_team_1:.1f}",
            "avg_tricks_t2": f"{s.avg_tricks_team_2:.1f}",
            "bid_success_rate": f"{s.bid_success_rate:.1f}%",
            "avg_bid": f"{s.avg_bid:.1f}", "dak_rate": f"{s.dak_rate:.1f}%",
            "seek_count": str(s.seek_count),
        }
        for key, value in updates.items():
            if key in self._stat_labels:
                self._stat_labels[key].config(text=value)
        self._draw_chart()

    # ----------------------------------------------------------
    # Chart
    # ----------------------------------------------------------

    def _draw_chart(self) -> None:
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

        left, right, top, bottom = 50, w - 20, 30, h - 25
        chart_w, chart_h = right - left, bottom - top

        # Smooth.
        window = max(3, len(history) // 20)
        smoothed = []
        for i in range(len(history)):
            start = max(0, i - window + 1)
            smoothed.append(sum(history[start:i + 1]) / (i - start + 1))

        data_min, data_max = min(smoothed), max(smoothed)
        margin = max((data_max - data_min) * 0.3, 3)
        y_low = max(0, data_min - margin)
        y_high = min(100, data_max + margin)
        y_span = max(y_high - y_low, 1)

        canvas.create_rectangle(left, top, right, bottom, fill="#111111", outline="#333333")

        for i in range(5):
            frac = i / 4
            y = bottom - int(frac * chart_h)
            val = y_low + frac * y_span
            canvas.create_line(left, y, right, y, fill="#262626")
            canvas.create_text(left - 6, y, text=f"{val:.0f}%", anchor="e",
                               fill="#666666", font=("Consolas", 8))

        if y_low < 50 < y_high:
            y50 = bottom - int((50 - y_low) / y_span * chart_h)
            canvas.create_line(left, y50, right, y50, fill="#444444", dash=(6, 3))

        points = []
        for i, rate in enumerate(smoothed):
            x = left + int(i / max(1, len(smoothed) - 1) * chart_w)
            y = max(top, min(bottom, bottom - int((rate - y_low) / y_span * chart_h)))
            points.append((x, y))

        if len(points) < 2:
            return

        canvas.create_polygon([(left, bottom)] + points + [(right, bottom)],
                              fill="#0d2e0d", outline="")
        for i in range(1, len(points)):
            canvas.create_line(points[i-1][0], points[i-1][1],
                               points[i][0], points[i][1], fill="#4caf50", width=2)

        sx, sy = points[0]
        canvas.create_oval(sx-4, sy-4, sx+4, sy+4, fill="#ff9800", outline="#fff", width=1)
        canvas.create_text(sx+10, sy, text=f"{smoothed[0]:.1f}%", anchor="w",
                           fill="#ff9800", font=("Segoe UI", 9, "bold"))

        ex, ey = points[-1]
        ec = "#66ff66" if smoothed[-1] > smoothed[0] else "#ff6666"
        canvas.create_oval(ex-5, ey-5, ex+5, ey+5, fill=ec, outline="#fff", width=1)
        canvas.create_text(ex-10, ey-14, text=f"{smoothed[-1]:.1f}%", anchor="e",
                           fill=ec, font=("Segoe UI", 10, "bold"))

        imp = smoothed[-1] - smoothed[0]
        sign = "+" if imp >= 0 else ""
        canvas.create_text(w // 2, 12,
                           text=f"Start: {smoothed[0]:.1f}% → Now: {smoothed[-1]:.1f}%  ({sign}{imp:.1f}%)   •  {len(history)*10} games",
                           fill=ec, font=("Segoe UI", 9, "bold"))

    def _draw_bar_chart(self, canvas, w, h) -> None:
        bw, gap = 70, 50
        sx = (w - (2 * bw + gap)) // 2
        bb, mh = h - 30, h - 60
        t1h = max(4, int(self.stats.team_1_win_rate / 100 * mh))
        canvas.create_rectangle(sx, bb - t1h, sx + bw, bb, fill=COLORS["score_team1"], outline="")
        canvas.create_text(sx + bw // 2, bb + 14, text=f"T1: {self.stats.team_1_win_rate:.0f}%",
                           fill=COLORS["score_team1"], font=("Segoe UI", 10, "bold"))
        x2 = sx + bw + gap
        t2h = max(4, int(self.stats.team_2_win_rate / 100 * mh))
        canvas.create_rectangle(x2, bb - t2h, x2 + bw, bb, fill=COLORS["score_team2"], outline="")
        canvas.create_text(x2 + bw // 2, bb + 14, text=f"T2: {self.stats.team_2_win_rate:.0f}%",
                           fill=COLORS["score_team2"], font=("Segoe UI", 10, "bold"))

    # ----------------------------------------------------------
    # Batch runner
    # ----------------------------------------------------------

    def _run_batch(self) -> None:
        try:
            count = int(self._batch_count_var.get())
        except ValueError:
            count = 100
        self._batch_progress_label.config(text=f"Running {count} games...")
        self.root.update()
        thread = threading.Thread(target=self._batch_worker, args=(count,), daemon=True)
        thread.start()

    def _batch_worker(self, count: int) -> None:
        from agents.random.random_agent import RandomAgent
        from agents.rule_based.rule_based_agent import RuleBasedAgent
        from agents.learning.learning_agent import LearningAgent
        from environments.wist.environment import WistEnvironment
        from environments.wist.round import Round
        from environments.wist.rules import trick_winner
        from environments.wist.setup import create_standard_players
        from environments.wist.tasmiya_engine import TasmiyaEngine
        from environments.wist.trick import Trick

        t1_name = self._agent_t1_var.get()
        t2_name = self._agent_t2_var.get()

        if self._learning_agent_instance is None:
            self._learning_agent_instance = LearningAgent(training=True)

        self.root.after(0, lambda: self._learning_info_label.config(
            text=f"Training... Q-table: {self._learning_agent_instance.q_table_size} entries"))

        def make_agent(name):
            if name == "Rule-Based": return RuleBasedAgent()
            elif name == "Learning": return self._learning_agent_instance
            return RandomAgent()

        tasmiya = TasmiyaEngine()
        w_wins, w_total, done = 0, 0, 0

        while done < count:
            players = create_standard_players()
            t1, t2 = make_agent(t1_name), make_agent(t2_name)
            agents = [t1, t2, t1, t2]
            r = Round(players); r.deal()
            if r.has_card_based_dak(): continue

            res = tasmiya.run(players=players, agents=agents, sahib_al_qabool_id=0)
            if res.is_dak:
                self.stats.record_dak()
                if isinstance(t1, LearningAgent): t1.reset_episode()
                if isinstance(t2, LearningAgent) and t2 is not t1: t2.reset_episode()
                continue

            r.state.trump_suit = res.trump_suit
            r.state.winning_bidder_id = res.winning_bidder_id
            r.next_leading_player_id = res.winning_bidder_id
            env = WistEnvironment(r.state)
            tt = {0: 0, 1: 0}

            for _ in range(13):
                lid = r.next_leading_player_id
                r.state.current_trick = Trick(leading_player_id=lid)
                for pid in [(lid + j) % 4 for j in range(4)]:
                    obs = env.observe(pid)
                    action = agents[pid].act(obs)
                    env.apply_action(action)
                trick = r.state.current_trick
                winner = trick_winner(trick, r.state.trump_suit)
                r.state.completed_tricks.append(trick)
                r.state.current_trick = None
                r.next_leading_player_id = winner
                tt[players[winner].team_id] += 1
                if isinstance(t1, LearningAgent): t1.reward_trick(won=(players[winner].team_id == 0))
                if isinstance(t2, LearningAgent) and t2 is not t1: t2.reward_trick(won=(players[winner].team_id == 1))

            bid_met_t1 = (tt[0] >= res.winning_bid_value) if res.playing_team_id == 0 else False
            bid_met_t2 = (tt[1] >= res.winning_bid_value) if res.playing_team_id == 1 else False
            if isinstance(t1, LearningAgent): t1.reward_shota(team_won_shota=(tt[0] > tt[1]), bid_met=bid_met_t1); t1.decay_epsilon()
            if isinstance(t2, LearningAgent) and t2 is not t1: t2.reward_shota(team_won_shota=(tt[1] > tt[0]), bid_met=bid_met_t2); t2.decay_epsilon()

            wt = 0 if tt[0] > tt[1] else (1 if tt[1] > tt[0] else None)
            self.stats.record_game(winner_team=wt, score_1=tt[0], score_2=tt[1])
            self.stats.record_shota(team_1_tricks=tt[0], team_2_tricks=tt[1],
                                    bid=res.winning_bid_value, playing_team_id=res.playing_team_id,
                                    bid_met=tt.get(res.playing_team_id, 0) >= res.winning_bid_value)

            w_total += 1
            if wt == 0: w_wins += 1
            if w_total >= 10:
                self._learning_win_history.append(w_wins / w_total * 100)
                w_wins, w_total = 0, 0

            done += 1
            if done % 50 == 0 or done == count:
                self.root.after(0, self._progress_update, done, count)

        if isinstance(t1, LearningAgent):
            size, eps = t1.q_table_size, t1.epsilon
            self.root.after(0, lambda: self._learning_info_label.config(
                text=f"✓ Done | Q-table: {size} entries | ε: {eps:.3f}"))
        self.root.after(0, self._done, count)

    def _progress_update(self, current, total):
        self._batch_progress_label.config(text=f"Progress: {current}/{total}")
        self.update_display()
        if self._learning_agent_instance:
            s, e = self._learning_agent_instance.q_table_size, self._learning_agent_instance.epsilon
            self._learning_info_label.config(text=f"Q-table: {s} | ε: {e:.3f} | Games: {current}")

    def _done(self, count):
        self._batch_progress_label.config(text=f"✓ DONE — {count} games completed.")
        self.update_display()

    def _reset_stats(self):
        self.stats.reset()
        self._learning_win_history.clear()
        self.update_display()
        self._batch_progress_label.config(text="Stats reset.")

    def _save_model(self):
        if self._learning_agent_instance is None:
            self._learning_info_label.config(text="No agent to save.")
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            self._learning_agent_instance.save(path)
            self._learning_info_label.config(text=f"✓ Saved: {path}")

    def _load_model(self):
        from tkinter import filedialog
        from agents.learning.learning_agent import LearningAgent
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            self._learning_agent_instance = LearningAgent.load(path, training=True)
            self._learning_info_label.config(text=f"✓ Loaded: {self._learning_agent_instance.q_table_size} entries")

    def _reset_brain(self):
        from agents.learning.learning_agent import LearningAgent
        self._learning_agent_instance = LearningAgent(training=True)
        self._learning_win_history.clear()
        self._learning_info_label.config(text="Brain reset.")
        self._draw_chart()
