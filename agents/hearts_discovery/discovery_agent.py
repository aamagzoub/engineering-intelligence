"""
Discovery-based learning agent.

This agent has ZERO domain knowledge about Hearts. It does not know:
- That hearts are bad
- That Queen of Spades is special
- What a Gallon is
- Any strategy whatsoever

It receives ONLY:
- Observation: hand + legal moves + visible trick state
- Reward: numeric score at the end of each shota

It must learn everything from the reward signal alone.

Architecture:
- Q-learning with episode memory
- Separate Q-tables for passing and trick play
- Domain-agnostic state encoding
- Epsilon-greedy exploration with decay
"""

import random
from collections import defaultdict

from agents.hearts_discovery.model import load_model, save_model
from agents.hearts_discovery.state_encoder import (
    encode_passing_action,
    encode_passing_state,
    encode_play_action,
    encode_play_state,
)
from environments.hearts.actions import PassCardsAction, PlayCardAction
from environments.hearts.observation import HeartsObservation, PassingObservation
from intelligence.core.action import Action
from intelligence.core.agent import Agent
from intelligence.core.observation import Observation


class DiscoveryAgent(Agent):
    """
    Discovery-based learning agent — learns entirely from reward signals.

    No domain knowledge. No hard-coded strategy. No cheating.
    """

    def __init__(
        self,
        epsilon: float = 0.4,
        alpha: float = 0.2,
        gamma: float = 0.97,
        training: bool = True,
    ) -> None:
        # Q-tables.
        self.play_q: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.pass_q: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        # Hyperparameters.
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.training = training

        # Episode memory — stores (state, action) pairs for credit assignment.
        self._play_episode: list[tuple[str, str]] = []
        self._pass_episode: list[tuple[str, str]] = []

        # Training stats.
        self.episodes_trained: int = 0
        self.total_updates: int = 0

    def act(self, observation: Observation) -> Action:
        """Choose an action based on the observation type."""
        if isinstance(observation, PassingObservation):
            return self._act_passing(observation)
        if isinstance(observation, HeartsObservation):
            return self._act_play(observation)
        raise TypeError(
            f"DiscoveryAgent does not support {type(observation).__name__}."
        )

    # ----------------------------------------------------------
    # Card passing
    # ----------------------------------------------------------

    def _act_passing(self, obs: PassingObservation) -> PassCardsAction:
        """
        Choose 4 cards to pass.

        Strategy is learned — initially random, improves with training.
        """
        hand = obs.hand

        if self.training and random.random() < self.epsilon:
            # Explore: pass 4 random cards.
            cards_to_pass = tuple(random.sample(hand, 4))
        else:
            # Exploit: use learned passing preferences.
            cards_to_pass = self._best_pass(obs)

        # Record for learning.
        if self.training:
            state = encode_passing_state(obs)
            action_key = encode_passing_action(cards_to_pass, hand)
            self._pass_episode.append((state, action_key))

        return PassCardsAction(
            player_id=obs.player_id,
            cards=cards_to_pass,
        )

    def _best_pass(self, obs: PassingObservation) -> tuple:
        """Choose the best 4 cards to pass based on learned Q-values."""
        state = encode_passing_state(obs)
        q_values = self.pass_q[state]
        hand = obs.hand

        if not q_values:
            # No learned values yet — pass random.
            return tuple(random.sample(hand, 4))

        # Try multiple random candidate passes and pick the one with best Q.
        best_pass = None
        best_q = float("-inf")

        # Sample candidate passes (exhaustive is too expensive for 13-choose-4).
        for _ in range(20):
            candidate = tuple(random.sample(hand, 4))
            action_key = encode_passing_action(candidate, hand)
            q = q_values.get(action_key, 0.0)
            if q > best_q:
                best_q = q
                best_pass = candidate

        return best_pass if best_pass else tuple(random.sample(hand, 4))

    # ----------------------------------------------------------
    # Trick play
    # ----------------------------------------------------------

    def _act_play(self, obs: HeartsObservation) -> PlayCardAction:
        """
        Choose a card to play.

        Uses epsilon-greedy: explore randomly or pick best learned action.
        """
        legal = obs.legal_cards

        if len(legal) == 1:
            card = legal[0]
        elif self.training and random.random() < self.epsilon:
            # Explore: play random legal card.
            card = random.choice(legal)
        else:
            # Exploit: pick card with best learned Q-value.
            card = self._best_card(obs)

        # Record for learning.
        if self.training:
            state = encode_play_state(obs)
            action_key = encode_play_action(card, obs)
            self._play_episode.append((state, action_key))

        return PlayCardAction(player_id=obs.player_id, card=card)

    def _best_card(self, obs: HeartsObservation) -> object:
        """Pick the legal card whose action encoding has the best Q-value."""
        state = encode_play_state(obs)
        q_values = self.play_q[state]
        legal = obs.legal_cards

        best_card = legal[0]
        best_q = float("-inf")

        for card in legal:
            action_key = encode_play_action(card, obs)
            q = q_values.get(action_key, 0.0)
            if q > best_q:
                best_q = q
                best_card = card

        return best_card

    # ----------------------------------------------------------
    # Learning — reward signal at end of shota
    # ----------------------------------------------------------

    def reward(self, score: float) -> None:
        """
        Receive the shota score and propagate credit to all actions taken.

        This is the ONLY learning signal. The agent must figure out which
        of its ~17 actions (4 pass + 13 plays) contributed to the score.

        Uses Monte Carlo-style update: assign discounted reward backwards
        through the episode.
        """
        if not self.training:
            return

        # Adaptive learning rate — learn fast early, stabilize later.
        effective_alpha = max(0.05, self.alpha * (1.0 / (1.0 + self.episodes_trained / 1000)))

        # Update play Q-table — later actions get more credit.
        reward = score
        for state, action in reversed(self._play_episode):
            current_q = self.play_q[state][action]
            self.play_q[state][action] += effective_alpha * (reward - current_q)
            reward *= self.gamma  # Discount earlier actions.
            self.total_updates += 1

        # Update pass Q-table — passing gets discounted credit too.
        pass_reward = score * (self.gamma ** len(self._play_episode))
        for state, action in reversed(self._pass_episode):
            current_q = self.pass_q[state][action]
            self.pass_q[state][action] += effective_alpha * (pass_reward - current_q)
            self.total_updates += 1

        # Clear episode memory.
        self._play_episode.clear()
        self._pass_episode.clear()
        self.episodes_trained += 1

    # ----------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------

    def save(self, path: str) -> None:
        """Save learned model to disk."""
        metadata = {
            "episodes_trained": self.episodes_trained,
            "total_updates": self.total_updates,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "play_states": len(self.play_q),
            "pass_states": len(self.pass_q),
        }
        save_model(self.play_q, self.pass_q, metadata, path)

    def load(self, path: str) -> None:
        """Load a previously trained model."""
        play_q, pass_q, metadata = load_model(path)
        self.play_q = defaultdict(lambda: defaultdict(float), play_q)
        self.pass_q = defaultdict(lambda: defaultdict(float), pass_q)
        self.episodes_trained = metadata.get("episodes_trained", 0)
        self.total_updates = metadata.get("total_updates", 0)
