from environments.wist.team import Team


def test_team_contains_player():
    team = Team(team_id=0, player_ids=(0, 2))

    assert team.contains_player(0) is True
    assert team.contains_player(2) is True
    assert team.contains_player(1) is False