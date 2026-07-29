from enum import Enum


class Rank(Enum):
    """
    Represents the identity of a playing card rank.

    This class intentionally does NOT define any ordering
    or game-specific value. Those belong to the rules of
    the game (e.g. Wist, Poker, Bridge).
    """

    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"

    @property
    def symbol(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.symbol