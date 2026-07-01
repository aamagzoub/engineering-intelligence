from agents.random.random_agent import RandomAgent
from environments.wist.environment import WistEnvironment
from environments.wist.playing_engine import PlayingEngine
from environments.wist.round import Round
from environments.wist.setup import create_standard_players
from intelligence.core.cards.suit import Suit


def test_playing_engine_finishes_13_tricks():
    players = create_standard_players()

    round_ = Round(players)
    round_.deal()

    round_.state.trump_suit = Suit.SPADES

    environment = WistEnvironment(round_.state)

    agents = [
        RandomAgent(),
        RandomAgent(),
        RandomAgent(),
        RandomAgent(),
    ]

    engine = PlayingEngine()

    tricks = engine.play_shota(
        round_,
        environment,
        agents,
    )

    assert tricks[0] + tricks[1] == 13

def test_all_52_cards_are_played_by_end_of_shota():
    """
    A complete Shota must consume all 52 cards:
    13 cards × 4 players = 52 cards.
    """

    players = create_standard_players()

    round_ = Round(players)
    round_.deal()

    round_.state.trump_suit = Suit.SPADES

    environment = WistEnvironment(round_.state)

    agents = [
        RandomAgent(),
        RandomAgent(),
        RandomAgent(),
        RandomAgent(),
    ]

    engine = PlayingEngine()

    engine.play_shota(
        round_,
        environment,
        agents,
    )

    # Every player should have played all 13 cards.
    for player in players:
        assert len(player.hand) == 0

    # Exactly 52 cards should have been played.
    assert len(round_.state.played_cards) == 52

    # Exactly 13 completed tricks should exist.
    assert len(round_.state.completed_tricks) == 13