from environments.wist.actions import PlayCardAction
from intelligence.core.cards.card import Card
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit


def test_play_card_action():
    card = Card(suit=Suit.SPADES, rank=Rank.ACE)
    action = PlayCardAction(player_id=0, card=card)

    assert action.player_id == 0
    assert action.card == card