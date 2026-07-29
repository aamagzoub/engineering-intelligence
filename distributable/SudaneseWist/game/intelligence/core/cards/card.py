from dataclasses import dataclass

from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit


@dataclass(frozen=True, slots=True)
class Card:
    """
    Represents one playing card.

    The card has no knowledge of any game rules.
    """

    suit: Suit
    rank: Rank

    def __str__(self) -> str:
        return f"{self.rank.symbol}{self.suit.symbol}"

    def __repr__(self) -> str:
        return self.__str__()