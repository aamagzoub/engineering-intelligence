from random import shuffle

from intelligence.core.cards.card import Card
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit


class Deck:
    """
    Represents a standard 52-card deck.

    The deck knows how to create and shuffle cards.
    It does not know anything about Wist rules.
    """

    def __init__(self) -> None:
        self.cards: list[Card] = [
            Card(suit=suit, rank=rank)
            for suit in Suit
            for rank in Rank
        ]

    def shuffle(self) -> None:
        shuffle(self.cards)

    def deal(self, number_of_cards: int) -> list[Card]:
        if number_of_cards > len(self.cards):
            raise ValueError("Cannot deal more cards than remain in the deck.")

        dealt_cards = self.cards[:number_of_cards]
        self.cards = self.cards[number_of_cards:]
        return dealt_cards

    def __len__(self) -> int:
        return len(self.cards)