"""Generic Monte Carlo Tree Search engine.

Domain-agnostic look-ahead: simulates random future actions to determine
which action leads to the best outcome. Uses only a caller-provided
simulate_fn that encapsulates the domain's transition logic.
"""

from __future__ import annotations

import random
from typing import Callable

from tibrain import Action, State

# Type alias for the simulation function provided by the domain.
# Takes (state, action) and returns (next_state, reward, done, legal_actions).
SimulateFn = Callable[
    [State, Action],
    tuple[State, float, bool, list[Action]],
]


class MCTSEngine:
    """Generic Monte Carlo Tree Search engine.

    Uses flat rollouts with a random policy to estimate action values.
    The domain provides a simulate_fn that handles state transitions.
    """

    def __init__(
        self,
        simulate_fn: SimulateFn,
        num_simulations: int = 100,
    ) -> None:
        """Initialize the MCTS engine.

        Args:
            simulate_fn: Callable taking (state, action) and returning
                (next_state, reward, done, legal_actions).
            num_simulations: Default number of simulations per action.
        """
        self._simulate = simulate_fn
        self.num_simulations = num_simulations

    def choose_action(
        self,
        state: State,
        legal_actions: list[Action],
        num_simulations: int | None = None,
    ) -> Action:
        """Return the action with highest average simulated reward.

        If only one legal action is available, returns it immediately
        without performing any simulations.

        Args:
            state: Current state.
            legal_actions: Available actions from this state.
            num_simulations: Override default simulation count.

        Returns:
            The best action based on average rollout reward.
        """
        if len(legal_actions) == 1:
            return legal_actions[0]

        scores = self._score_actions(state, legal_actions, num_simulations)
        return max(legal_actions, key=lambda a: scores.get(a, 0.0))

    def evaluate_actions(
        self,
        state: State,
        legal_actions: list[Action],
        num_simulations: int | None = None,
    ) -> dict[Action, float]:
        """Return normalized scores [0, 1] per action.

        Normalization maps the best action to 1.0 and the worst to 0.0.
        If all actions have the same score, all get 0.0 (spread is 1.0).

        If only one legal action or fewer, returns an empty dict.

        Args:
            state: Current state.
            legal_actions: Available actions from this state.
            num_simulations: Override default simulation count.

        Returns:
            Dictionary mapping each action to a normalized score in [0, 1].
        """
        if len(legal_actions) <= 1:
            return {}

        scores = self._score_actions(state, legal_actions, num_simulations)
        values = list(scores.values())
        min_v, max_v = min(values), max(values)
        spread = max_v - min_v if max_v > min_v else 1.0

        return {a: (scores[a] - min_v) / spread for a in legal_actions}

    def _score_actions(
        self,
        state: State,
        legal_actions: list[Action],
        num_simulations: int | None,
    ) -> dict[Action, float]:
        """Run simulations for each action, return average rewards."""
        n_sims = num_simulations or self.num_simulations
        scores: dict[Action, float] = {}

        for action in legal_actions:
            total_reward = 0.0
            for _ in range(n_sims):
                total_reward += self._rollout(state, action)
            scores[action] = total_reward / n_sims

        return scores

    def _rollout(self, state: State, action: Action) -> float:
        """Execute a single simulation rollout from (state, action).

        Takes the given action from the given state, then continues
        with a random policy until a terminal state is reached or
        no legal actions remain.

        Returns:
            Cumulative reward collected during the rollout.
        """
        cumulative_reward = 0.0
        next_state, reward, done, next_actions = self._simulate(state, action)
        cumulative_reward += reward

        # Continue rollout with random actions until terminal
        current_state = next_state
        current_actions = next_actions

        while not done and current_actions:
            a = random.choice(current_actions)
            current_state, reward, done, current_actions = self._simulate(
                current_state, a
            )
            cumulative_reward += reward

        return cumulative_reward
