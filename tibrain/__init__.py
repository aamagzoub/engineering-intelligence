"""TIBRAIN: Domain-neutral reinforcement learning library.

Provides generic Q-learning, neural network function approximation,
Monte Carlo Tree Search, and pattern discovery through protocol-based interfaces.
"""

from typing import Hashable, Protocol, runtime_checkable

# Core type aliases — domain provides concrete types
State = Hashable
Action = Hashable


@runtime_checkable
class Environment(Protocol):
    """Protocol for any domain environment that TIBRAIN can learn in."""

    def reset(self) -> State:
        """Reset the environment and return the initial state."""
        ...

    def observe(self) -> State:
        """Return the current observable state."""
        ...

    def get_legal_actions(self, state: State) -> list[Action]:
        """Return legal actions from the given state."""
        ...

    def step(self, action: Action) -> tuple[State, float, dict]:
        """Execute action, return (next_state, reward, info)."""
        ...


@runtime_checkable
class StateEncoder(Protocol):
    """Converts domain State to a hashable string key for Q-tables."""

    def __call__(self, state: State) -> str: ...


@runtime_checkable
class ActionEncoder(Protocol):
    """Converts domain Action to a hashable string key for Q-tables."""

    def __call__(self, action: Action) -> str: ...


__all__ = [
    "State",
    "Action",
    "Environment",
    "StateEncoder",
    "ActionEncoder",
]
