import pytest

from environments.wist.game_state import DakType
from environments.wist.game_state import GameState


def test_game_state_starts_empty():
    state = GameState()

    assert state.team_scores == {0: 0, 1: 0}
    assert state.completed_shotas == 0
    assert state.winner_team_id is None
    assert state.is_finished() is False


def test_game_state_applies_shota_score():
    state = GameState()

    state.apply_shota_score({0: 10, 1: 0})

    assert state.team_scores == {0: 10, 1: 0}
    assert state.completed_shotas == 1


def test_game_state_detects_winner_at_25():
    state = GameState(team_scores={0: 20, 1: 5})

    state.apply_shota_score({0: 5, 1: 0})

    assert state.winner_team_id == 0
    assert state.is_finished() is True


def test_game_state_finishes_after_5_shotas():
    state = GameState()

    for _ in range(5):
        state.apply_shota_score({0: 1, 1: 0})

    assert state.is_finished() is True


def test_game_state_seek_ends_game_immediately():
    state = GameState()

    state.apply_seek(team_id=1)

    assert state.winner_team_id == 1
    assert state.is_finished() is True

def test_dak_always_counts_as_shota():
    state = GameState()

    state.apply_dak(DakType.CARD_BASED)

    assert state.completed_shotas == 1


def test_dak_after_first_shota_counts():
    state = GameState()
    state.apply_shota_score({0: 5, 1: 0})

    state.apply_dak(DakType.CARD_BASED)

    assert state.completed_shotas == 2


def test_pass_based_dak_can_only_happen_twice():
    state = GameState()

    state.apply_dak(DakType.PASS_BASED)
    state.apply_dak(DakType.PASS_BASED)

    with pytest.raises(ValueError):
        state.apply_dak(DakType.PASS_BASED)