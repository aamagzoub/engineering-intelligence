import pytest

from environments.wist.scoring import detect_seek, score_shota


def test_playing_team_scores_actual_tricks_when_bid_succeeds():
    result = score_shota(
        playing_team_id=0,
        defending_team_id=1,
        bid=8,
        playing_team_tricks=10,
        defending_team_tricks=3,
    )

    assert result == {0: 10, 1: 0}


def test_playing_team_loses_bid_when_bid_fails():
    result = score_shota(
        playing_team_id=0,
        defending_team_id=1,
        bid=8,
        playing_team_tricks=6,
        defending_team_tricks=7,
    )

    assert result == {0: -8, 1: 7}


def test_scores_can_go_negative():
    result = score_shota(
        playing_team_id=0,
        defending_team_id=1,
        bid=10,
        playing_team_tricks=3,
        defending_team_tricks=10,
    )

    assert result == {0: -10, 1: 10}


def test_shota_must_have_13_tricks():
    with pytest.raises(ValueError):
        score_shota(
            playing_team_id=0,
            defending_team_id=1,
            bid=8,
            playing_team_tricks=5,
            defending_team_tricks=5,
        )


def test_detect_seek():
    assert detect_seek({0: 13, 1: 0}) == 0
    assert detect_seek({0: 0, 1: 13}) == 1
    assert detect_seek({0: 8, 1: 5}) is None