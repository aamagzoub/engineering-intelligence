import random

from environments.wist.actions import PlayCardAction
from environments.wist.observation import WistObservation
from environments.wist.rules import legal_cards
from intelligence.core.action import Action
from intelligence.core.agent import Agent
from intelligence.core.observation import Observation


class RandomAgent(Agent):
    """
    A simple Wist agent that randomly chooses from legal cards.
    """

    def act(self, observation: Observation) -> Action:
        if not isinstance(observation, WistObservation):
            raise TypeError("RandomAgent currently only supports WistObservation.")

        if not observation.hand:
            raise ValueError("RandomAgent cannot act with an empty hand.")

        leading_suit = None

        if observation.current_trick is not None:
            leading_suit = observation.current_trick.leading_suit

        card = random.choice(
            legal_cards(
                hand=observation.hand,
                leading_suit=leading_suit,
            )
        )

        return PlayCardAction(
            player_id=observation.player_id,
            card=card,
        )