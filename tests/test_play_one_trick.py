from agents.random.random_agent import RandomAgent
from environments.wist.environment import WistEnvironment
from environments.wist.round import Round
from environments.wist.setup import create_standard_players
from intelligence.core.cards.suit import Suit


def test_round_can_play_one_complete_trick():
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

    winner = round_.play_one_trick(
        environment=environment,
        agents=agents,
    )

    assert winner in [0, 1, 2, 3]
    assert len(round_.state.completed_tricks) == 1
    assert round_.state.current_trick is None

def test_trick_winner_leads_next_trick():
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

    winner = round_.play_one_trick(environment, agents)

    assert round_.next_leading_player_id == winner