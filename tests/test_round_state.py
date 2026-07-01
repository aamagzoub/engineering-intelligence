import pytest

from environments.wist.player import Player
from environments.wist.round_state import RoundState


def test_round_state_gets_player():
    player = Player(player_id=0, team_id=0)
    state = RoundState(players={0: player})

    assert state.get_player(0) == player


def test_round_state_rejects_unknown_player():
    state = RoundState(players={})

    with pytest.raises(ValueError):
        state.get_player(99)