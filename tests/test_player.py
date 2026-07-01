import pytest

from environments.wist.player import Player
from intelligence.core.cards.card import Card
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit


def test_player_receives_cards():
    player = Player(player_id=0, team_id=0)
    card = Card(Suit.SPADES, Rank.ACE)

    player.receive_cards([card])

    assert player.hand == [card]


def test_player_can_play_card_from_hand():
    card = Card(Suit.SPADES, Rank.ACE)
    player = Player(player_id=0, team_id=0, hand=[card])

    played_card = player.play_card(card)

    assert played_card == card
    assert player.hand == []


def test_player_cannot_play_card_not_in_hand():
    player = Player(player_id=0, team_id=0)
    card = Card(Suit.SPADES, Rank.ACE)

    with pytest.raises(ValueError):
        player.play_card(card)


def test_player_has_suit():
    player = Player(
        player_id=0,
        team_id=0,
        hand=[Card(Suit.HEARTS, Rank.KING)],
    )

    assert player.has_suit(Suit.HEARTS) is True
    assert player.has_suit(Suit.SPADES) is False