from dataclasses import dataclass

from intelligence.core.action import Action
from intelligence.core.cards.card import Card


@dataclass(frozen=True)
class PlayCardAction(Action):
    """
    Action representing a player choosing to play one card.
    """

    player_id: int
    card: Card


@dataclass(frozen=True)
class BidAction(Action):
    """
    Action representing a player placing a bid during Al-Tasmiya.
    """

    player_id: int
    value: int


@dataclass(frozen=True)
class PassAction(Action):
    """
    Action representing a player passing during Al-Tasmiya.
    """

    player_id: int