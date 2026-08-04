"""
Random Hearts agent — baseline for comparison.

Plays completely randomly: picks random legal cards and passes random cards.
Used to measure if the discovery agent actually learns anything.
"""

import random

from environments.hearts.actions import PassCardsAction, PlayCardAction
from environments.hearts.observation import HeartsObservation, PassingObservation
from intelligence.core.action import Action
from intelligence.core.agent import Agent
from intelligence.core.observation import Observation


class RandomHeartsAgent(Agent):
    """Plays Hearts randomly — baseline agent."""

    def act(self, observation: Observation) -> Action:
        if isinstance(observation, PassingObservation):
            return self._act_passing(observation)
        if isinstance(observation, HeartsObservation):
            return self._act_play(observation)
        raise TypeError(
            f"RandomHeartsAgent does not support {type(observation).__name__}."
        )

    def _act_passing(self, obs: PassingObservation) -> PassCardsAction:
        """Pass 4 random cards."""
        cards = tuple(random.sample(obs.hand, 4))
        return PassCardsAction(player_id=obs.player_id, cards=cards)

    def _act_play(self, obs: HeartsObservation) -> PlayCardAction:
        """Play a random legal card."""
        card = random.choice(obs.legal_cards)
        return PlayCardAction(player_id=obs.player_id, card=card)
