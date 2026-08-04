"""
Hearts legal move rules and trick resolution.

No trump suit in Hearts — highest card of the led suit wins.
"""

from intelligence.core.cards.card import Card
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit
from environments.hearts.trick import Trick


RANK_VALUES = {
    Rank.TWO: 2,
    Rank.THREE: 3,
    Rank.FOUR: 4,
    Rank.FIVE: 5,
    Rank.SIX: 6,
    Rank.SEVEN: 7,
    Rank.EIGHT: 8,
    Rank.NINE: 9,
    Rank.TEN: 10,
    Rank.JACK: 11,
    Rank.QUEEN: 12,
    Rank.KING: 13,
    Rank.ACE: 14,
}


def rank_value(rank: Rank) -> int:
    """Numeric value of a rank for comparison."""
    return RANK_VALUES[rank]


def legal_cards(
    hand: list[Card],
    leading_suit: Suit | None,
    is_first_trick: bool = False,
) -> list[Card]:
    """
    Return cards a player is legally allowed to play.

    Rules:
    1. If leading (leading_suit is None):
       - First trick: cannot lead with a Heart
       - After first trick: can lead anything
    2. If following:
       - Must follow the led suit if able
       - If void in led suit: can play anything
    """
    if leading_suit is None:
        # Player is leading this trick.
        if is_first_trick:
            # Cannot lead hearts on the first trick.
            non_hearts = [c for c in hand if c.suit != Suit.HEARTS]
            if non_hearts:
                return non_hearts
            # Edge case: hand is all hearts — must play one.
            return list(hand)
        # After first trick: can lead anything.
        return list(hand)

    # Player is following — must follow suit if able.
    matching = [c for c in hand if c.suit == leading_suit]
    if matching:
        return matching

    # Void in led suit — can play anything.
    return list(hand)


def trick_winner(trick: Trick) -> int:
    """
    Determine who wins a completed trick.

    In Hearts there is no trump — highest card of the LED SUIT wins.
    Off-suit cards never win regardless of rank.
    """
    if not trick.is_complete():
        raise ValueError("Cannot determine winner of an incomplete trick.")

    leading_suit = trick.leading_suit

    # Only cards of the leading suit can win.
    leading_suit_cards = [
        pc for pc in trick.played_cards
        if pc.card.suit == leading_suit
    ]

    winner = max(
        leading_suit_cards,
        key=lambda pc: rank_value(pc.card.rank),
    )

    return winner.player_id
