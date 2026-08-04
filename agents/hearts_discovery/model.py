"""
Model persistence for the discovery agent.

Saves and loads Q-tables as JSON for training continuity.
"""

import json
from collections import defaultdict
from pathlib import Path


def save_model(
    play_q: dict[str, dict[str, float]],
    pass_q: dict[str, dict[str, float]],
    metadata: dict,
    path: str,
) -> None:
    """
    Save Q-tables and metadata to a JSON file.

    Args:
        play_q: Q-table for trick play decisions
        pass_q: Q-table for card passing decisions
        metadata: training metadata (episodes, epsilon, etc.)
        path: file path to save to
    """
    data = {
        "play_q": {state: dict(actions) for state, actions in play_q.items()},
        "pass_q": {state: dict(actions) for state, actions in pass_q.items()},
        "metadata": metadata,
    }

    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_model(path: str) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict,
]:
    """
    Load Q-tables and metadata from a JSON file.

    Returns:
        (play_q, pass_q, metadata)
    """
    with open(path, "r") as f:
        data = json.load(f)

    play_q: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for state, actions in data.get("play_q", {}).items():
        for action, value in actions.items():
            play_q[state][action] = value

    pass_q: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for state, actions in data.get("pass_q", {}).items():
        for action, value in actions.items():
            pass_q[state][action] = value

    metadata = data.get("metadata", {})

    return play_q, pass_q, metadata
