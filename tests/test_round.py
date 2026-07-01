from environments.wist.player import Player
from environments.wist.round import Round


def test_round_deals_13_cards_to_each_player():
    players = [
        Player(0, 0),
        Player(1, 1),
        Player(2, 0),
        Player(3, 1),
    ]

    round_ = Round(players)
    round_.deal()

    for player in players:
        assert len(player.hand) == 13

def test_round_creates_round_state_after_dealing():
    players = [
        Player(0, 0),
        Player(1, 1),
        Player(2, 0),
        Player(3, 1),
    ]

    round_ = Round(players)
    round_.deal()

    assert round_.state is not None
    assert len(round_.state.players) == 4
    assert round_.state.get_player(0).player_id == 0