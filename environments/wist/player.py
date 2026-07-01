from dataclasses import dataclass, field

from intelligence.core.cards.card import Card


@dataclass
class Player:
    """
    Represents one Wist player.

    A Player holds cards and belongs to a team.
    The Player itself is not intelligent.
    Intelligence belongs to an Agent.
    """

    player_id: int
    team_id: int
    hand: list[Card] = field(default_factory=list)

    def receive_cards(self, cards: list[Card]) -> None:
        self.hand.extend(cards)

    def play_card(self, card: Card) -> Card:
        if card not in self.hand:
            raise ValueError("Player cannot play a card they do not have.")

        self.hand.remove(card)
        return card

    def has_suit(self, suit) -> bool:
        return any(card.suit == suit for card in self.hand)