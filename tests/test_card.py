from intelligence.core.cards.card import Card
from intelligence.core.cards.suit import Suit
from intelligence.core.cards.rank import Rank


def test_card_creation():
    card = Card(suit=Suit.SPADES, rank=Rank.ACE)
    
    assert card.suit == Suit.SPADES
    assert card.rank == Rank.ACE
    assert str(card) == "A♠"    