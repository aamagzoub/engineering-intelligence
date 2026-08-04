from dataclasses import dataclass, field

from intelligence.core.cards.card import Card


@dataclass
class HeartsPlayer:
    """
    Individual Hearts player (no team).

    Holds cards and tracks tricks won this shota.
    Intelligence belongs to an Agent, not the Player.
    """

    player_id: int
    hand: list[Card] = field(default_factory=list)
    tricks_won: int = 0
    collected_cards: list[Card] = field(default_factory=list)

    def receive_cards(self, cards: list[Card]) -> None:
        """Add cards to hand."""
        self.hand.extend(cards)

    def remove_cards(self, cards: list[Card]) -> None:
        """Remove specific cards from hand (used during passing)."""
        for card in cards:
            if card not in self.hand:
                raise ValueError(
                    f"Player {self.player_id} cannot remove {card} — not in hand."
                )
            self.hand.remove(card)

    def play_card(self, card: Card) -> Card:
        """Play a card from hand."""
        if card not in self.hand:
            raise ValueError(
                f"Player {self.player_id} cannot play {card} — not in hand."
            )
        self.hand.remove(card)
        return card

    def collect_trick(self, cards: list[Card]) -> None:
        """Record winning a trick — store the cards for scoring later."""
        self.tricks_won += 1
        self.collected_cards.extend(cards)

    def reset_shota(self) -> None:
        """Reset for a new shota."""
        self.hand.clear()
        self.tricks_won = 0
        self.collected_cards.clear()

    def has_suit(self, suit) -> bool:
        """Check if player has any card of the given suit."""
        return any(card.suit == suit for card in self.hand)
