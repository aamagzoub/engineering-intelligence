from dataclasses import dataclass

from intelligence.core.action import Action
from intelligence.core.cards.card import Card


@dataclass(frozen=True)
class PlayCardAction(Action):
    """Action representing a player choosing to play one card."""

    player_id: int
    card: Card


@dataclass(frozen=True)
class PassCardsAction(Action):
    """Action representing a player passing 4 cards to the left."""

    player_id: int
    cards: tuple[Card, Card, Card, Card]
