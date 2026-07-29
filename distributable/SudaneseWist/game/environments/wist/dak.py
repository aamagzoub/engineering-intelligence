from collections import Counter

from intelligence.core.cards.card import Card
from intelligence.core.cards.rank import Rank


PICTURE_RANKS = {
    Rank.ACE,
    Rank.KING,
    Rank.QUEEN,
    Rank.JACK,
}


def has_picture_card(hand: list[Card]) -> bool:
    return any(card.rank in PICTURE_RANKS for card in hand)


def has_eight_or_more_in_one_suit(hand: list[Card]) -> bool:
    suit_counts = Counter(card.suit for card in hand)
    return any(count >= 8 for count in suit_counts.values())


def triggers_card_based_dak(hand: list[Card]) -> bool:
    return (
        not has_picture_card(hand)
        or has_eight_or_more_in_one_suit(hand)
    )