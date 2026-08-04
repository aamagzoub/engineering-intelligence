"""
Sudanese Wist — Entry Points

Usage:
    python run_wist.py play                     (PyGame interactive game)
    python run_wist.py lab                      (Tkinter AI laboratory)
    python run_wist.py train                    (CLI curriculum training)
    python run_wist.py train --episodes 5000    (custom episode count)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def play():
    """Launch the PyGame interactive game."""
    from gui_wist.main import main
    main()


def lab():
    """Launch the Tkinter AI laboratory."""
    from gui_wist_lab.app import main
    main()


def train(episodes: int = 15000):
    """Run CLI curriculum training."""
    from agents.wist_learning.trainer import train_curriculum

    def progress(ep, wins, losses, rate, eps):
        total = wins + losses
        if total > 0 and ep % 500 == 0:
            print(f"EP {ep:5d} | WR={rate:.1f}% | ε={eps:.4f} | "
                  f"Overall={wins}/{total} ({wins/total*100:.1f}%)")

    print("=" * 60)
    print("Sudanese Wist — Curriculum Training")
    print("=" * 60)

    agent, results = train_curriculum(
        save_path="agents/wist_learning/wist_model.json",
        on_progress=progress,
    )

    total = results.wins + results.losses
    print(f"\nDone: {results.wins}/{total} wins ({results.wins/total*100:.1f}%)")
    print(f"Q-table: {agent.q_table_size} | Model saved.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sudanese Wist")
    parser.add_argument("command", choices=["play", "lab", "train"],
                        help="play=PyGame, lab=Tkinter, train=CLI training")
    parser.add_argument("--episodes", type=int, default=15000)
    args = parser.parse_args()

    if args.command == "play":
        play()
    elif args.command == "lab":
        lab()
    elif args.command == "train":
        train(args.episodes)
