from enum import Enum


class Suit(Enum):
    """
    Represents the identity of a playing card suit.
    """

    SPADES = ("Spades", "♠")
    HEARTS = ("Hearts", "♥")
    DIAMONDS = ("Diamonds", "♦")
    CLUBS = ("Clubs", "♣")

    @property
    def display_name(self) -> str:
        return self.value[0]

    @property
    def symbol(self) -> str:
        return self.value[1]

    def __str__(self) -> str:
        return self.display_name