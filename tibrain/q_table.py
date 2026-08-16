"""Q-Table data structure for storing state-action value estimates."""

from __future__ import annotations


class QTable:
    """Stores Q-values indexed by (state_key, action_key) pairs."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, float]] = {}

    def get(self, state: str, action: str) -> float:
        """Return Q-value, defaulting to 0.0 for unvisited pairs."""
        return self._data.get(state, {}).get(action, 0.0)

    def set(self, state: str, action: str, value: float) -> None:
        """Set Q-value for a state-action pair."""
        if state not in self._data:
            self._data[state] = {}
        self._data[state][action] = value

    def get_best_action(self, state: str, actions: list[str]) -> str:
        """Return the action with highest Q-value among given actions."""
        best_action = actions[0]
        best_value = self.get(state, actions[0])
        for a in actions[1:]:
            v = self.get(state, a)
            if v > best_value:
                best_value = v
                best_action = a
        return best_action

    @property
    def size(self) -> int:
        """Number of stored state-action pairs."""
        return sum(len(actions) for actions in self._data.values())

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {state: dict(actions) for state, actions in self._data.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "QTable":
        """Deserialize from a plain dictionary."""
        table = cls()
        table._data = {state: dict(actions) for state, actions in data.items()}
        return table
