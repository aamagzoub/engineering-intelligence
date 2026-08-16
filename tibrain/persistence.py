"""Persistence module for saving and loading TIBRAIN agent state.

Provides save/load functions that serialize all agent state (Q-tables,
neural network weights, replay buffer contents, reward normalizer
statistics, and discovery data) to/from JSON files.

Supports incremental saves where only changed components are written
when a changed_components set is provided.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tibrain.agent import Agent


def save(
    agent: "Agent",
    path: Path,
    changed_components: set[str] | None = None,
) -> None:
    """Serialize all agent state to a JSON file.

    If changed_components is provided and the file already exists,
    performs an incremental save by only updating those keys in the
    existing file. Otherwise, writes a full snapshot.

    Args:
        agent: The TIBRAIN Agent whose state should be persisted.
        path: File path for the JSON output.
        changed_components: Optional set of component keys to update
            incrementally (e.g. {"q1", "q2", "evaluator"}).
    """
    data = _agent_to_dict(agent)

    if changed_components and path.exists():
        existing = json.loads(path.read_text())
        for key in changed_components:
            if key in data:
                existing[key] = data[key]
        data = existing

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load(path: Path) -> dict:
    """Deserialize agent state from a JSON file.

    Returns an empty dictionary if the file does not exist, without
    raising an exception.

    Args:
        path: File path to read JSON from.

    Returns:
        Dictionary containing the serialized agent state, or empty dict
        if the path does not exist.
    """
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _agent_to_dict(agent: "Agent") -> dict[str, Any]:
    """Convert agent internal state to a JSON-serializable dictionary.

    Extracts and serializes:
    - Q-tables (q1, q2) via their to_dict() methods
    - Policy state (epsilon, epsilon_min)
    - Replay buffer entries
    - Evaluator neural network weights (if present)
    - Reward normalizer statistics (if present)
    - Discovery engine data (if present)

    Args:
        agent: The TIBRAIN Agent to serialize.

    Returns:
        Dictionary with all serializable agent components.
    """
    data: dict[str, Any] = {}

    # Q-tables from the Q-learning engine
    data["q1"] = agent.q_engine.q1.to_dict()
    data["q2"] = agent.q_engine.q2.to_dict()

    # Policy state
    data["policy"] = {
        "epsilon": agent.policy.epsilon,
        "epsilon_min": agent.policy.epsilon_min,
    }

    # Replay buffer contents
    if agent.replay_buffer is not None:
        data["replay_buffer"] = [list(entry) for entry in agent.replay_buffer._buffer]

    # Neural network evaluator weights
    if hasattr(agent, "_evaluator") and agent._evaluator is not None:
        data["evaluator"] = agent._evaluator.to_dict()

    # Reward normalizer statistics
    if hasattr(agent, "_reward_normalizer") and agent._reward_normalizer is not None:
        data["reward_normalizer"] = agent._reward_normalizer.to_dict()

    # Discovery engine data
    if hasattr(agent, "_discovery_engine") and agent._discovery_engine is not None:
        data["discovery"] = agent._discovery_engine.to_dict()

    return data
