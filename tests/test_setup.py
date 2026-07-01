from environments.wist.setup import create_standard_players, create_standard_teams


def test_create_standard_players():
    players = create_standard_players()

    assert len(players) == 4
    assert players[0].team_id == 0
    assert players[1].team_id == 1
    assert players[2].team_id == 0
    assert players[3].team_id == 1


def test_create_standard_teams():
    teams = create_standard_teams()

    assert len(teams) == 2
    assert teams[0].player_ids == (0, 2)
    assert teams[1].player_ids == (1, 3)