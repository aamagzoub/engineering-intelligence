from dataclasses import dataclass, field

from intelligence.core.cards.card import Card
from intelligence.core.cards.suit import Suit


@dataclass
class PlayedCard:
    """One card played in a trick, tagged with who played it."""

    player_id: int
    card: Card


@dataclass
class Trick:
    """
    Represents one trick in Hearts.

    Same structure as Wist tricks — 4 players each play one card.
    The leading suit is determined by the first card played.
    """

    leading_player_id: int
    played_cards: list[PlayedCard] = field(default_factory=list)

    @property
    def leading_suit(self) -> Suit | None:
        if not self.played_cards:
            return None
        return self.played_cards[0].card.suit

    def play_card(self, player_id: int, card: Card) -> None:
        if len(self.played_cards) >= 4:
            raise ValueError("A trick cannot contain more than 4 cards.")
        self.played_cards.append(PlayedCard(player_id=player_id, card=card))

    def is_complete(self) -> bool:
        return len(self.played_cards) == 4

    def all_cards(self) -> list[Card]:
        """Return all cards played in this trick."""
        return [pc.card for pc in self.played_cards]
