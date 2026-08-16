"""Prioritized experience replay buffer for TIBRAIN."""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Prioritized experience replay buffer.

    Stores (state, action, reward, next_state, td_error) tuples in a ring buffer
    and samples proportionally to |td_error| + epsilon for prioritized replay.
    """

    def __init__(self, capacity: int = 10000, priority_epsilon: float = 0.01) -> None:
        self._capacity = capacity
        self._priority_epsilon = priority_epsilon
        self._buffer: list[tuple[str, str, float, str, float]] = []
        self._pos: int = 0

    def add(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        td_error: float,
    ) -> None:
        """Add an experience tuple to the buffer.

        Uses ring buffer semantics: appends until capacity is reached,
        then overwrites the oldest entry.
        """
        entry = (state, action, reward, next_state, td_error)
        if len(self._buffer) < self._capacity:
            self._buffer.append(entry)
        else:
            self._buffer[self._pos] = entry
        self._pos = (self._pos + 1) % self._capacity

    def sample(self, batch_size: int) -> list[tuple[str, str, float, str]]:
        """Sample experiences with probability proportional to |td_error| + epsilon.

        Returns (state, action, reward, next_state) tuples without the td_error.
        When the buffer contains fewer entries than batch_size, returns all entries.
        """
        if not self._buffer:
            return []

        actual_size = min(batch_size, len(self._buffer))

        # Compute priorities: |td_error| + epsilon
        priorities = np.array(
            [abs(entry[4]) + self._priority_epsilon for entry in self._buffer],
            dtype=np.float64,
        )
        probs = priorities / priorities.sum()

        indices = np.random.choice(
            len(self._buffer), size=actual_size, replace=False, p=probs
        )

        # Return tuples stripped of td_error
        return [
            (self._buffer[i][0], self._buffer[i][1],
             self._buffer[i][2], self._buffer[i][3])
            for i in indices
        ]

    def __len__(self) -> int:
        """Return the number of entries currently in the buffer."""
        return len(self._buffer)

    @property
    def capacity(self) -> int:
        """Return the maximum capacity of the buffer."""
        return self._capacity
