from environments.wist.rules import legal_cards
from intelligence.core.cards.card import Card
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit
from environments.wist.trick import Trick
from environments.wist.rules import rank_value, trick_winner

def test_legal_cards_allows_any_card_when_no_leading_suit():
    hand = [
        Card(Suit.SPADES, Rank.ACE),
        Card(Suit.HEARTS, Rank.KING),
    ]

    assert legal_cards(hand, leading_suit=None) == hand


def test_legal_cards_must_follow_leading_suit_if_possible():
    spade = Card(Suit.SPADES, Rank.ACE)
    heart = Card(Suit.HEARTS, Rank.KING)

    hand = [spade, heart]

    assert legal_cards(hand, leading_suit=Suit.HEARTS) == [heart]


def test_legal_cards_allows_any_card_when_void_in_leading_suit():
    hand = [
        Card(Suit.SPADES, Rank.ACE),
        Card(Suit.CLUBS, Rank.KING),
    ]

    assert legal_cards(hand, leading_suit=Suit.HEARTS) == hand

def test_rank_value_ace_is_highest():
    assert rank_value(Rank.TWO) == 2
    assert rank_value(Rank.ACE) == 14


def test_highest_leading_suit_wins_when_no_trump():
    trick = Trick(leading_player_id=0)

    trick.play_card(0, Card(Suit.HEARTS, Rank.TEN))
    trick.play_card(1, Card(Suit.HEARTS, Rank.ACE))
    trick.play_card(2, Card(Suit.HEARTS, Rank.KING))
    trick.play_card(3, Card(Suit.HEARTS, Rank.TWO))

    assert trick_winner(trick, trump_suit=Suit.SPADES) == 1


def test_highest_trump_wins():
    trick = Trick(leading_player_id=0)

    trick.play_card(0, Card(Suit.HEARTS, Rank.ACE))
    trick.play_card(1, Card(Suit.SPADES, Rank.TWO))
    trick.play_card(2, Card(Suit.SPADES, Rank.KING))
    trick.play_card(3, Card(Suit.HEARTS, Rank.KING))

    assert trick_winner(trick, trump_suit=Suit.SPADES) == 2