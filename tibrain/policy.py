"""Exploration policies: epsilon-greedy with UCB bonus and adaptive decay."""

from __future__ import annotations

import math
import random


class Policy:
    """Exploration policies: epsilon-greedy with UCB bonus and adaptive decay.

    Implements epsilon-greedy action selection where, with probability epsilon,
    a random action is chosen (exploration), and otherwise the action with the
    highest Q-value plus UCB bonus is chosen (exploitation with exploration bonus).

    Attributes:
        epsilon: Current exploration probability.
        epsilon_min: Minimum epsilon floor below which decay has no effect.
    """

    def __init__(
        self,
        epsilon: float = 0.3,
        epsilon_min: float = 0.01,
    ) -> None:
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self._visit_counts: dict[str, int] = {}
        self._total_visits: int = 0

    def select(
        self,
        q_values: dict[str, float],
        action_keys: list[str],
    ) -> str:
        """Epsilon-greedy selection with UCB bonus on the greedy branch.

        With probability epsilon, selects a uniformly random action.
        Otherwise, selects the action maximizing Q-value + UCB bonus.

        Args:
            q_values: Mapping of action keys to their Q-values.
            action_keys: List of available action keys.

        Returns:
            The selected action key.

        Raises:
            ValueError: If action_keys is empty.
        """
        if not action_keys:
            raise ValueError("No actions to select from")

        # Epsilon-greedy: random with probability epsilon
        if random.random() < self.epsilon:
            chosen = random.choice(action_keys)
        else:
            chosen = self._select_with_ucb(q_values, action_keys)

        # Record visit
        self._visit_counts[chosen] = self._visit_counts.get(chosen, 0) + 1
        self._total_visits += 1
        return chosen

    def select_greedy(
        self,
        q_values: dict[str, float],
        action_keys: list[str],
    ) -> str:
        """Pure greedy selection — picks the action with the highest Q-value.

        No exploration is performed. Ties are broken arbitrarily by max().

        Args:
            q_values: Mapping of action keys to their Q-values.
            action_keys: List of available action keys.

        Returns:
            The action key with the highest Q-value.

        Raises:
            ValueError: If action_keys is empty.
        """
        if not action_keys:
            raise ValueError("No actions to select from")
        return max(action_keys, key=lambda a: q_values.get(a, 0.0))

    def decay(self, factor: float) -> None:
        """Multiply epsilon by factor, respecting the minimum bound.

        After decay, epsilon = max(epsilon_min, epsilon * factor).

        Args:
            factor: Multiplicative decay factor (typically < 1.0).
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * factor)

    def _select_with_ucb(
        self,
        q_values: dict[str, float],
        action_keys: list[str],
    ) -> str:
        """Select action with UCB-inspired exploration bonus.

        UCB bonus = 0.15 * sqrt(log(total_visits + 1) / visits).
        Unvisited actions get instant priority (selected immediately).
        When total_visits is 0, falls back to uniform random selection.

        Args:
            q_values: Mapping of action keys to their Q-values.
            action_keys: List of available action keys.

        Returns:
            The action key with the highest Q + UCB score.
        """
        if self._total_visits == 0:
            return random.choice(action_keys)

        best_action = action_keys[0]
        best_score = float("-inf")

        for a in action_keys:
            q = q_values.get(a, 0.0)
            visits = self._visit_counts.get(a, 0)
            if visits == 0:
                # Unvisited actions get maximum priority
                return a
            ucb_bonus = math.sqrt(math.log(self._total_visits + 1) / visits)
            score = q + 0.15 * ucb_bonus
            if score > best_score:
                best_score = score
                best_action = a

        return best_action
