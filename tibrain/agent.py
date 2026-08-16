"""Generic RL agent that works with any domain via encoders."""

from __future__ import annotations

from tibrain import Action, State, StateEncoder, ActionEncoder
from tibrain.neural_net import Evaluator
from tibrain.policy import Policy
from tibrain.q_learning import QLearningEngine
from tibrain.replay_buffer import ReplayBuffer


class Agent:
    """Generic RL agent that works with any domain via encoders.

    The Agent orchestrates the core RL components (Q-learning engine, policy,
    replay buffer, optional neural evaluator) and translates between domain
    objects and internal string keys using configurable encoder callables.

    When training=True, the agent explores (epsilon-greedy with UCB) and
    updates Q-values on each learn() call. When training=False, the agent
    selects greedily and skips all learning updates.
    """

    def __init__(
        self,
        state_encoder: StateEncoder | None = None,
        action_encoder: ActionEncoder | None = None,
        *,
        alpha: float = 0.1,
        gamma: float = 0.95,
        lambda_trace: float = 0.7,
        epsilon: float = 0.3,
        epsilon_min: float = 0.01,
        training: bool = True,
        use_neural: bool = False,
        neural_config: dict | None = None,
        replay_capacity: int = 10000,
    ) -> None:
        self.state_encoder: StateEncoder = state_encoder or str
        self.action_encoder: ActionEncoder = action_encoder or str
        self.training = training

        self.q_engine = QLearningEngine(
            alpha=alpha, gamma=gamma, lambda_trace=lambda_trace
        )
        self.policy = Policy(epsilon=epsilon, epsilon_min=epsilon_min)
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

        self._evaluator: Evaluator | None = None
        if use_neural and neural_config:
            self._evaluator = Evaluator(**neural_config)

    def choose_action(self, state: State, legal_actions: list[Action]) -> Action:
        """Select action according to current policy and Q-values.

        In training mode, uses epsilon-greedy with UCB exploration bonus.
        In evaluation mode (training=False), selects the action with the
        highest Q-value (pure greedy).

        Args:
            state: The current domain state (any hashable object).
            legal_actions: List of legal actions available in this state.

        Returns:
            The selected action from legal_actions.

        Raises:
            ValueError: If legal_actions is empty.
        """
        if not legal_actions:
            raise ValueError("No legal actions available")

        state_key = self.state_encoder(state)
        action_keys = [self.action_encoder(a) for a in legal_actions]

        if self.training:
            chosen_key = self.policy.select(
                self.q_engine.get_values(state_key, action_keys),
                action_keys,
            )
        else:
            # Greedy: pick highest Q-value action
            chosen_key = self.policy.select_greedy(
                self.q_engine.get_values(state_key, action_keys),
                action_keys,
            )

        idx = action_keys.index(chosen_key)
        return legal_actions[idx]

    def learn(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        next_legal_actions: list[Action],
    ) -> None:
        """Update value estimates from a transition.

        Encodes domain objects to string keys, performs a TD(λ) update
        via the Q-learning engine, and stores the experience in the
        replay buffer for future sampling.

        Does nothing when training=False.

        Args:
            state: The state before the transition.
            action: The action taken.
            reward: The scalar reward received.
            next_state: The state after the transition.
            next_legal_actions: Legal actions available in next_state.
        """
        if not self.training:
            return

        state_key = self.state_encoder(state)
        action_key = self.action_encoder(action)
        next_state_key = self.state_encoder(next_state)
        next_action_keys = [self.action_encoder(a) for a in next_legal_actions]

        td_error = self.q_engine.td_update(
            state_key, action_key, reward, next_state_key, next_action_keys
        )

        self.replay_buffer.add(
            state_key, action_key, reward, next_state_key, td_error
        )

    def reset_episode(self) -> None:
        """Clear eligibility traces for a new episode."""
        self.q_engine.reset_episode()
