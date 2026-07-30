"""
Sudanese Wist AI Laboratory — GUI

Two tabs:
- Stats & Lab: batch training, model management
- Play for AI: proxy at physical table
"""

import tkinter as tk
from tkinter import ttk

from gui.colors import COLORS
from gui.stats import GameStats


class WistAILabApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Sudanese Wist — AI Laboratory")

        # Window size.
        win_w, win_h = 1100, 700
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2 - 30
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.root.resizable(True, True)
        self.root.configure(bg="#1a1a1a")
        self.root.minsize(900, 600)

        # Stats tracking.
        self.stats = GameStats()

        self._build_layout()

    def _build_layout(self) -> None:
        # Title bar.
        title_bar = tk.Frame(self.root, bg="#1a1a1a", height=50)
        title_bar.pack(fill="x")
        title_bar.pack_propagate(False)

        tk.Label(title_bar, text="Sudanese Wist — AI Laboratory",
                 font=("Segoe UI", 14, "bold"),
                 fg="#ffffff", bg="#1a1a1a").pack(side="left", padx=16, pady=12)

        # Notebook tabs.
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Lab.TNotebook", background="#1a1a1a", borderwidth=0)
        style.configure("Lab.TNotebook.Tab",
                        font=("Segoe UI", 11, "bold"),
                        padding=[16, 8],
                        background="#2a2a2a",
                        foreground="#aaaaaa")
        style.map("Lab.TNotebook.Tab",
                  background=[("selected", "#333333")],
                  foreground=[("selected", "#ffffff")])

        self.notebook = ttk.Notebook(self.root, style="Lab.TNotebook")
        self.notebook.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Tab 1: Stats & Lab
        stats_frame = tk.Frame(self.notebook, bg="#1a1a1a")
        self.notebook.add(stats_frame, text="  Stats & Lab  ")

        from gui.stats_tab import StatsTab
        self._stats_tab = StatsTab(stats_frame, self.root, self.stats)

        # Tab 2: Play for AI
        play_frame = tk.Frame(self.notebook, bg="#1a1a1a")
        self.notebook.add(play_frame, text="  Play for AI  ")

        from gui.advisor_tab import AdvisorTab
        self._advisor_tab = AdvisorTab(play_frame, self.root)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    app = WistAILabApp()
    app.run()
