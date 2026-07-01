from environments.wist.shota_result import ShotaResult


def test_shota_result_stores_round_outcome():
    result = ShotaResult(
        playing_team_id=0,
        defending_team_id=1,
        bid=8,
        team_tricks={0: 10, 1: 3},
    )

    assert result.playing_team_id == 0
    assert result.defending_team_id == 1
    assert result.bid == 8
    assert result.team_tricks == {0: 10, 1: 3}