from environments.wist.trick import Trick
from intelligence.core.cards.card import Card
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit


WIST_RANK_VALUES = {
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


def legal_cards(hand: list[Card], leading_suit: Suit | None) -> list[Card]:
    if leading_suit is None:
        return hand

    matching_suit_cards = [
        card for card in hand
        if card.suit == leading_suit
    ]

    if matching_suit_cards:
        return matching_suit_cards

    return hand


def rank_value(rank: Rank) -> int:
    return WIST_RANK_VALUES[rank]


def trick_winner(trick: Trick, trump_suit: Suit) -> int:
    if not trick.is_complete():
        raise ValueError("Cannot determine winner of an incomplete trick.")

    trump_cards = [
        played_card
        for played_card in trick.played_cards
        if played_card.card.suit == trump_suit
    ]

    if trump_cards:
        winning_card = max(
            trump_cards,
            key=lambda played_card: rank_value(played_card.card.rank),
        )
        return winning_card.player_id

    leading_suit = trick.leading_suit

    leading_suit_cards = [
        played_card
        for played_card in trick.played_cards
        if played_card.card.suit == leading_suit
    ]

    winning_card = max(
        leading_suit_cards,
        key=lambda played_card: rank_value(played_card.card.rank),
    )

    return winning_card.player_id