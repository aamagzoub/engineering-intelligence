from environments.wist.trick import Trick
from intelligence.core.cards.card import Card
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit


def test_trick_starts_empty():
    trick = Trick(leading_player_id=0)

    assert trick.leading_player_id == 0
    assert trick.played_cards == []
    assert trick.leading_suit is None
    assert trick.is_complete() is False


def test_first_card_defines_leading_suit():
    trick = Trick(leading_player_id=0)
    card = Card(suit=Suit.SPADES, rank=Rank.ACE)

    trick.play_card(player_id=0, card=card)

    assert trick.leading_suit == Suit.SPADES


def test_trick_is_complete_after_four_cards():
    trick = Trick(leading_player_id=0)

    trick.play_card(0, Card(Suit.SPADES, Rank.ACE))
    trick.play_card(1, Card(Suit.SPADES, Rank.KING))
    trick.play_card(2, Card(Suit.SPADES, Rank.QUEEN))
    trick.play_card(3, Card(Suit.SPADES, Rank.JACK))

    assert trick.is_complete() is True