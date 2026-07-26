"""
Tests that the winning bidder must lead with a trump card on the first trick.
"""

from agents.random.random_agent import RandomAgent
from environments.wist.environment import WistEnvironment
from environments.wist.player import Player
from environments.wist.round import Round
from environments.wist.round_state import RoundState
from environments.wist.trick import Trick
from intelligence.core.cards.card import Card
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit


def test_first_trick_bidder_must_play_trump():
    """
    When the winning bidder leads the first trick,
    they must play a card from the trump suit.
    Run this 50 times to ensure it's not just luck.
    """
    from environments.wist.tasmiya_engine import determine_trump_suit

    for _ in range(50):
        players = [
            Player(player_id=0, team_id=0),
            Player(player_id=1, team_id=1),
            Player(player_id=2, team_id=0),
            Player(player_id=3, team_id=1),
        ]

        round_ = Round(players)
        round_.deal()

        winning_bidder_id = 0

        # Trump must be the bidder's longest suit (per the real rules).
        trump_suit = determine_trump_suit(players[winning_bidder_id].hand)

        round_.state.trump_suit = trump_suit
        round_.state.winning_bidder_id = winning_bidder_id
        round_.next_leading_player_id = winning_bidder_id

        environment = WistEnvironment(round_.state)
        agents = [RandomAgent(), RandomAgent(), RandomAgent(), RandomAgent()]

        # Play the first trick.
        details = round_.play_one_trick_details(
            environment=environment,
            agents=agents,
        )

        trick = details["trick"]

        # The first card played (by the bidder) must be from the trump suit.
        first_played = trick.played_cards[0]
        assert first_played.player_id == winning_bidder_id
        assert first_played.card.suit == trump_suit, (
            f"Bidder played {first_played.card} but trump is {trump_suit.name}"
        )


def test_second_trick_bidder_not_forced_to_trump():
    """
    On the second trick, even if the bidder leads,
    they are NOT forced to play trump.
    """

    players = [
        Player(player_id=0, team_id=0),
        Player(player_id=1, team_id=1),
        Player(player_id=2, team_id=0),
        Player(player_id=3, team_id=1),
    ]

    round_ = Round(players)
    round_.deal()

    trump_suit = Suit.SPADES
    winning_bidder_id = 0

    round_.state.trump_suit = trump_suit
    round_.state.winning_bidder_id = winning_bidder_id
    round_.next_leading_player_id = winning_bidder_id

    environment = WistEnvironment(round_.state)
    agents = [RandomAgent(), RandomAgent(), RandomAgent(), RandomAgent()]

    # Play first trick.
    round_.play_one_trick(environment, agents)

    # Force the bidder to lead the second trick (may not happen naturally).
    round_.next_leading_player_id = winning_bidder_id

    # Observe the bidder for the second trick.
    round_.state.current_trick = Trick(leading_player_id=winning_bidder_id)
    obs = environment.observe(winning_bidder_id)

    # must_lead_trump should be False on the second trick.
    assert obs.must_lead_trump is False


def test_non_bidder_not_forced_to_trump_on_first_trick():
    """
    A non-bidder who happens to be the leader (shouldn't happen in rules,
    but testing the logic) is not forced to play trump.
    """

    players = [
        Player(player_id=0, team_id=0),
        Player(player_id=1, team_id=1),
        Player(player_id=2, team_id=0),
        Player(player_id=3, team_id=1),
    ]

    round_ = Round(players)
    round_.deal()

    round_.state.trump_suit = Suit.SPADES
    round_.state.winning_bidder_id = 0  # Player 0 is the bidder.

    # But player 1 is observing — they are NOT the bidder.
    round_.state.current_trick = Trick(leading_player_id=1)

    environment = WistEnvironment(round_.state)
    obs = environment.observe(1)

    assert obs.must_lead_trump is False
