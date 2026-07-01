import pytest

from environments.wist.actions import PlayCardAction
from environments.wist.environment import WistEnvironment
from environments.wist.player import Player
from environments.wist.round_state import RoundState
from environments.wist.trick import Trick
from intelligence.core.cards.card import Card
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit


def test_environment_observes_player_hand():
    card = Card(Suit.SPADES, Rank.ACE)
    player = Player(player_id=0, team_id=0, hand=[card])
    state = RoundState(players={0: player})
    environment = WistEnvironment(state)

    observation = environment.observe(player_id=0)

    assert observation.player_id == 0
    assert observation.hand == [card]


def test_environment_applies_play_card_action():
    card = Card(Suit.SPADES, Rank.ACE)
    player = Player(player_id=0, team_id=0, hand=[card])
    trick = Trick(leading_player_id=0)
    state = RoundState(players={0: player}, current_trick=trick)
    environment = WistEnvironment(state)

    action = PlayCardAction(player_id=0, card=card)
    environment.apply_action(action)

    assert player.hand == []
    assert state.current_trick.played_cards[0].card == card
    assert state.played_cards == [card]


def test_environment_rejects_action_when_no_current_trick():
    card = Card(Suit.SPADES, Rank.ACE)
    player = Player(player_id=0, team_id=0, hand=[card])
    state = RoundState(players={0: player})
    environment = WistEnvironment(state)

    action = PlayCardAction(player_id=0, card=card)

    with pytest.raises(ValueError):
        environment.apply_action(action)