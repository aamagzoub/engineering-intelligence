"""Double Q-learning engine with TD(λ) eligibility traces."""

from __future__ import annotations

import random

from tibrain.q_table import QTable


class QLearningEngine:
    """Double Q-learning with TD(λ) eligibility traces."""

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        lambda_trace: float = 0.7,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_trace = lambda_trace

        self.q1 = QTable()
        self.q2 = QTable()
        self._traces: dict[tuple[str, str], float] = {}

    def get_values(self, state_key: str, action_keys: list[str]) -> dict[str, float]:
        """Get combined Q-values (average of Q1 and Q2) for given actions."""
        return {
            a: (self.q1.get(state_key, a) + self.q2.get(state_key, a)) / 2.0
            for a in action_keys
        }

    def td_update(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        next_actions: list[str],
    ) -> float:
        """
        Perform a Double Q-learning TD(λ) update.

        Randomly chooses which table (Q1 or Q2) to update each step.
        Uses the other table for evaluation to reduce overestimation bias.
        Propagates the TD error through all active eligibility traces.

        Returns the TD error for prioritized replay.
        """
        # Randomly choose which table to update (Double Q-learning)
        if random.random() < 0.5:
            q_update, q_eval = self.q1, self.q2
        else:
            q_update, q_eval = self.q2, self.q1

        # Compute max next Q using Double Q logic:
        # Select best action from q_update, evaluate with q_eval
        if next_actions:
            best_next_action = q_update.get_best_action(next_state, next_actions)
            max_next_q = q_eval.get(next_state, best_next_action)
        else:
            max_next_q = 0.0

        # TD error
        current_q = q_update.get(state, action)
        td_error = reward + self.gamma * max_next_q - current_q

        # Update eligibility trace for current (state, action) — accumulating traces
        sa = (state, action)
        self._traces[sa] = self._traces.get(sa, 0.0) + 1.0

        # Propagate update through all active traces
        to_remove: list[tuple[str, str]] = []
        for (s, a), trace in self._traces.items():
            old_val = q_update.get(s, a)
            q_update.set(s, a, old_val + self.alpha * td_error * trace)
            # Decay trace by gamma * lambda_trace
            new_trace = self.gamma * self.lambda_trace * trace
            self._traces[(s, a)] = new_trace
            if new_trace < 0.01:
                to_remove.append((s, a))

        # Remove traces below threshold
        for key in to_remove:
            del self._traces[key]

        return td_error

    def reset_episode(self) -> None:
        """Clear all eligibility traces for a new episode."""
        self._traces.clear()
