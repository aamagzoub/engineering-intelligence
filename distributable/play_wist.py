"""
Sudanese Wist — Play Against AI

Standalone launcher for the Human vs AI game mode.
Run this file directly or package into .exe with PyInstaller:

    pyinstaller --onefile --windowed play_wist.py

"""

import sys
import os

# Add the project root to the path so imports work.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import tkinter as tk
from gui.human_tab import HumanTab
from gui.colors import COLORS


def main():
    root = tk.Tk()
    root.title("Sudanese Wist — Play Against AI")
    root.state("zoomed")
    root.resizable(False, False)
    root.configure(bg=COLORS["table_border"])

    # Single frame for the game.
    frame = tk.Frame(root, bg=COLORS["table_border"])
    frame.pack(fill="both", expand=True)

    # Create the Human vs AI game.
    game = HumanTab(frame, root)

    root.mainloop()


if __name__ == "__main__":
    main()
