from environments.wist.dak import triggers_card_based_dak
from intelligence.core.cards.card import Card
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit


def test_card_based_dak_when_no_picture_cards():
    hand = [
        Card(Suit.SPADES, Rank.TWO),
        Card(Suit.HEARTS, Rank.THREE),
        Card(Suit.CLUBS, Rank.FOUR),
    ]

    assert triggers_card_based_dak(hand) is True


def test_no_card_based_dak_when_hand_has_picture_card_and_no_long_suit():
    hand = [
        Card(Suit.SPADES, Rank.ACE),
        Card(Suit.HEARTS, Rank.THREE),
        Card(Suit.CLUBS, Rank.FOUR),
    ]

    assert triggers_card_based_dak(hand) is False


def test_card_based_dak_when_eight_or_more_cards_in_one_suit():
    hand = [
        Card(Suit.SPADES, Rank.TWO),
        Card(Suit.SPADES, Rank.THREE),
        Card(Suit.SPADES, Rank.FOUR),
        Card(Suit.SPADES, Rank.FIVE),
        Card(Suit.SPADES, Rank.SIX),
        Card(Suit.SPADES, Rank.SEVEN),
        Card(Suit.SPADES, Rank.EIGHT),
        Card(Suit.SPADES, Rank.NINE),
        Card(Suit.HEARTS, Rank.ACE),
    ]

    assert triggers_card_based_dak(hand) is True