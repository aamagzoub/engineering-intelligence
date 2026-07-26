"""
Tests for the full Wist game orchestrator.
"""

from agents.random.random_agent import RandomAgent
from environments.wist.game import WistGame, GameResult
from environments.wist.setup import create_standard_players


def test_game_completes_within_5_shotas():
    """
    A game should complete after at most 5 Shotas
    (plus possible Daks that count as Shotas).
    """

    for _ in range(10):
        players = create_standard_players()
        agents = [RandomAgent(), RandomAgent(), RandomAgent(), RandomAgent()]

        game = WistGame(players=players, agents=agents)
        result = game.play()

        assert result.total_shotas <= 5 + 2  # Max 2 pass-based Daks + 5 Shotas
        assert result.final_scores is not None
        assert 0 in result.final_scores
        assert 1 in result.final_scores


def test_game_has_a_winner_or_reaches_5_shotas():
    """
    The game ends either when a team reaches 25 points,
    Seek happens, or 5 Shotas are completed (higher score wins).
    """

    for _ in range(20):
        game = WistGame()
        result = game.play()

        if result.winner_team_id is not None:
            if result.ended_by_seek:
                # Seek — winner doesn't need 25.
                pass
            else:
                # Winner either reached 25 OR had higher score after 5 Shotas.
                winner_score = result.final_scores[result.winner_team_id]
                loser_id = 1 if result.winner_team_id == 0 else 0
                loser_score = result.final_scores[loser_id]
                assert winner_score >= loser_score
        else:
            # Truly tied scores after 5 Shotas (extremely rare).
            assert result.total_shotas >= 5


def test_game_rotates_qabool():
    """
    Sahib Al-Qabool should rotate counter-clockwise each Shota.
    """

    game = WistGame()
    assert game.sahib_al_qabool_id == 0

    game._rotate_qabool()
    assert game.sahib_al_qabool_id == 1

    game._rotate_qabool()
    assert game.sahib_al_qabool_id == 2

    game._rotate_qabool()
    assert game.sahib_al_qabool_id == 3

    game._rotate_qabool()
    assert game.sahib_al_qabool_id == 0


def test_game_produces_shota_outcomes():
    """
    Each played Shota should produce a ShotaOutcome.
    """

    game = WistGame()
    result = game.play()

    assert len(result.shota_outcomes) > 0

    for outcome in result.shota_outcomes:
        if not outcome.was_dak:
            assert outcome.team_tricks[0] + outcome.team_tricks[1] == 13
            assert outcome.bid >= 7
            assert outcome.trump_suit_name != ""


def test_game_scoring_is_consistent():
    """
    The final scores should equal the sum of all score deltas.
    """

    for _ in range(10):
        game = WistGame()
        result = game.play()

        total_0 = sum(o.score_delta.get(0, 0) for o in result.shota_outcomes)
        total_1 = sum(o.score_delta.get(1, 0) for o in result.shota_outcomes)

        if not result.ended_by_seek:
            assert result.final_scores[0] == total_0
            assert result.final_scores[1] == total_1


def test_game_seek_ends_immediately():
    """
    If Seek occurs, the game should end immediately
    regardless of score or Shota count.
    """

    # Run many games until we get a Seek (rare with random agents).
    # If no Seek happens in 200 games, skip (it's probabilistic).

    seek_found = False

    for _ in range(200):
        game = WistGame()
        result = game.play()

        if result.ended_by_seek:
            seek_found = True
            assert result.winner_team_id is not None
            # The winning team got all 13 tricks in some Shota.
            seek_outcome = next(
                o for o in result.shota_outcomes if o.seek_team_id is not None
            )
            assert seek_outcome.team_tricks[seek_outcome.seek_team_id] == 13
            break

    # Seek is extremely rare with random play, so we don't assert seek_found.
    # This test just validates correctness IF it happens.


def test_game_events_are_logged():
    """
    The game should produce event log entries.
    """

    game = WistGame()
    game.play()

    assert len(game.events) > 0


def test_multiple_games_produce_different_results():
    """
    Running multiple games should produce varying outcomes
    (not always the same winner/score).
    """

    results = []
    for _ in range(10):
        game = WistGame()
        result = game.play()
        results.append(result.final_scores[0])

    # At least some variation in scores (extremely unlikely to all be identical).
    assert len(set(results)) > 1
