"""
Tests for the TasmiyaEngine — the Al-Tasmiya (bidding) orchestrator.
"""

import pytest
from collections import Counter
from unittest.mock import MagicMock

from environments.wist.actions import BidAction, PassAction
from environments.wist.observation import BiddingObservation
from environments.wist.player import Player
from environments.wist.tasmiya_engine import (
    TasmiyaEngine,
    TasmiyaResult,
    determine_trump_suit,
    max_bid_for_hand,
    tasmiya_order,
)
from intelligence.core.cards.card import Card
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit


# ---------------------------------------------------------------
# Helper: create players with specific hands
# ---------------------------------------------------------------


def make_players_with_hands(hands: list[list[Card]]) -> list[Player]:
    """Create 4 players and assign the given hands."""
    players = [
        Player(player_id=0, team_id=0),
        Player(player_id=1, team_id=1),
        Player(player_id=2, team_id=0),
        Player(player_id=3, team_id=1),
    ]
    for player, hand in zip(players, hands):
        player.hand = hand
    return players


def make_simple_hands() -> list[list[Card]]:
    """
    Create 4 hands where each player has 5 spades and 8 other cards.
    Player 0: 5 spades, 4 hearts, 4 diamonds
    Player 1: 5 hearts, 4 spades, 4 clubs
    Player 2: 5 diamonds, 4 clubs, 4 spades
    Player 3: 5 clubs, 4 diamonds, 4 hearts
    """
    ranks = list(Rank)

    hand_0 = (
        [Card(Suit.SPADES, r) for r in ranks[:5]]
        + [Card(Suit.HEARTS, r) for r in ranks[:4]]
        + [Card(Suit.DIAMONDS, r) for r in ranks[:4]]
    )
    hand_1 = (
        [Card(Suit.HEARTS, r) for r in ranks[:5]]
        + [Card(Suit.SPADES, r) for r in ranks[5:9]]
        + [Card(Suit.CLUBS, r) for r in ranks[:4]]
    )
    hand_2 = (
        [Card(Suit.DIAMONDS, r) for r in ranks[:5]]
        + [Card(Suit.CLUBS, r) for r in ranks[4:8]]
        + [Card(Suit.SPADES, r) for r in ranks[9:13]]
    )
    hand_3 = (
        [Card(Suit.CLUBS, r) for r in ranks[:5]]
        + [Card(Suit.DIAMONDS, r) for r in ranks[5:9]]
        + [Card(Suit.HEARTS, r) for r in ranks[5:9]]
    )

    return [hand_0, hand_1, hand_2, hand_3]


def make_agent_that_bids(value: int):
    """Create a mock agent that always bids the given value."""
    agent = MagicMock()
    agent.act = MagicMock(
        side_effect=lambda obs: BidAction(player_id=obs.player_id, value=value)
    )
    return agent


def make_agent_that_passes():
    """Create a mock agent that always passes."""
    agent = MagicMock()
    agent.act = MagicMock(
        side_effect=lambda obs: PassAction(player_id=obs.player_id)
    )
    return agent


# ---------------------------------------------------------------
# Tests: tasmiya_order
# ---------------------------------------------------------------


def test_tasmiya_order_from_player_0():
    order = tasmiya_order(0)
    assert order == [1, 2, 3]


def test_tasmiya_order_from_player_1():
    order = tasmiya_order(1)
    assert order == [2, 3, 0]


def test_tasmiya_order_from_player_2():
    order = tasmiya_order(2)
    assert order == [3, 0, 1]


def test_tasmiya_order_from_player_3():
    order = tasmiya_order(3)
    assert order == [0, 1, 2]


# ---------------------------------------------------------------
# Tests: determine_trump_suit
# ---------------------------------------------------------------


def test_determine_trump_suit_chooses_longest_suit():
    hand = [
        Card(Suit.SPADES, Rank.ACE),
        Card(Suit.SPADES, Rank.KING),
        Card(Suit.SPADES, Rank.QUEEN),
        Card(Suit.SPADES, Rank.JACK),
        Card(Suit.SPADES, Rank.TEN),
        Card(Suit.HEARTS, Rank.ACE),
        Card(Suit.HEARTS, Rank.KING),
        Card(Suit.DIAMONDS, Rank.ACE),
        Card(Suit.DIAMONDS, Rank.KING),
        Card(Suit.DIAMONDS, Rank.QUEEN),
        Card(Suit.CLUBS, Rank.ACE),
        Card(Suit.CLUBS, Rank.KING),
        Card(Suit.CLUBS, Rank.QUEEN),
    ]

    trump = determine_trump_suit(hand)
    assert trump == Suit.SPADES


# ---------------------------------------------------------------
# Tests: max_bid_for_hand
# ---------------------------------------------------------------


def test_max_bid_for_hand_with_5_in_longest_suit():
    hand = (
        [Card(Suit.SPADES, r) for r in [Rank.ACE, Rank.KING, Rank.QUEEN, Rank.JACK, Rank.TEN]]
        + [Card(Suit.HEARTS, r) for r in [Rank.ACE, Rank.KING, Rank.QUEEN, Rank.JACK]]
        + [Card(Suit.DIAMONDS, r) for r in [Rank.ACE, Rank.KING, Rank.QUEEN, Rank.JACK]]
    )
    assert max_bid_for_hand(hand) == 8  # 5 + 3 = 8


def test_max_bid_for_hand_with_7_in_longest_suit():
    hand = (
        [Card(Suit.SPADES, r) for r in [Rank.ACE, Rank.KING, Rank.QUEEN, Rank.JACK, Rank.TEN, Rank.NINE, Rank.EIGHT]]
        + [Card(Suit.HEARTS, r) for r in [Rank.ACE, Rank.KING, Rank.QUEEN]]
        + [Card(Suit.DIAMONDS, r) for r in [Rank.ACE, Rank.KING, Rank.QUEEN]]
    )
    assert max_bid_for_hand(hand) == 10  # 7 + 3 = 10


def test_max_bid_for_hand_with_8_plus_returns_zero():
    hand = [Card(Suit.SPADES, r) for r in Rank] + [Card(Suit.HEARTS, Rank.ACE)]
    # 13 spades → should not bid (Dak)
    assert max_bid_for_hand(hand) == 0


# ---------------------------------------------------------------
# Tests: TasmiyaEngine — normal bidding scenarios
# ---------------------------------------------------------------


def test_one_player_bids_others_pass_qabool_accepts():
    """
    Player 1 bids 8, players 2 and 3 pass.
    Sahib Al-Qabool (player 0) passes → accepts player 1's bid.
    Player 1's team plays.
    """
    hands = make_simple_hands()
    players = make_players_with_hands(hands)

    agents = [
        make_agent_that_passes(),   # Player 0 (Qabool) → pass = accept
        make_agent_that_bids(8),    # Player 1 → bid 8
        make_agent_that_passes(),   # Player 2 → pass
        make_agent_that_passes(),   # Player 3 → pass
    ]

    engine = TasmiyaEngine()
    result = engine.run(
        players=players,
        agents=agents,
        sahib_al_qabool_id=0,
    )

    assert result.is_dak is False
    assert result.winning_bidder_id == 1
    assert result.winning_bid_value == 8
    assert result.playing_team_id == 1  # Player 1 is on team 1
    assert result.defending_team_id == 0
    assert result.sahib_al_qabool_id == 0
    # Trump should be player 1's longest suit (hearts — 5 cards)
    assert result.trump_suit == Suit.HEARTS


def test_qabool_matches_bid():
    """
    Player 1 bids 8, others pass.
    Sahib Al-Qabool (player 0) matches with bid 8.
    Qabool's team plays.
    """
    hands = make_simple_hands()
    players = make_players_with_hands(hands)

    agents = [
        make_agent_that_bids(8),    # Player 0 (Qabool) → match bid 8
        make_agent_that_bids(8),    # Player 1 → bid 8
        make_agent_that_passes(),   # Player 2 → pass
        make_agent_that_passes(),   # Player 3 → pass
    ]

    engine = TasmiyaEngine()
    result = engine.run(
        players=players,
        agents=agents,
        sahib_al_qabool_id=0,
    )

    assert result.is_dak is False
    assert result.winning_bidder_id == 0
    assert result.winning_bid_value == 8
    assert result.playing_team_id == 0  # Qabool is on team 0
    assert result.defending_team_id == 1
    # Trump should be player 0's longest suit (spades — 5 cards)
    assert result.trump_suit == Suit.SPADES


def test_all_pass_results_in_dak():
    """
    All three players pass, Sahib Al-Qabool also passes → Dak.
    """
    hands = make_simple_hands()
    players = make_players_with_hands(hands)

    agents = [
        make_agent_that_passes(),  # Player 0 (Qabool)
        make_agent_that_passes(),  # Player 1
        make_agent_that_passes(),  # Player 2
        make_agent_that_passes(),  # Player 3
    ]

    engine = TasmiyaEngine()
    result = engine.run(
        players=players,
        agents=agents,
        sahib_al_qabool_id=0,
    )

    assert result.is_dak is True
    assert result.winning_bidder_id is None
    assert result.winning_bid_value is None
    assert result.trump_suit is None


def test_all_pass_qabool_bids():
    """
    All three players pass, Sahib Al-Qabool bids 7.
    Qabool's team plays.
    """
    hands = make_simple_hands()
    players = make_players_with_hands(hands)

    agents = [
        make_agent_that_bids(7),    # Player 0 (Qabool) → bids
        make_agent_that_passes(),   # Player 1
        make_agent_that_passes(),   # Player 2
        make_agent_that_passes(),   # Player 3
    ]

    engine = TasmiyaEngine()
    result = engine.run(
        players=players,
        agents=agents,
        sahib_al_qabool_id=0,
    )

    assert result.is_dak is False
    assert result.winning_bidder_id == 0
    assert result.winning_bid_value == 7
    assert result.playing_team_id == 0
    assert result.trump_suit == Suit.SPADES


def test_bid_13_stops_tasmiya_immediately():
    """
    Player 1 bids 8 (opening), Player 2 bids 13 → Al-Tasmiya stops.
    Player 3 should not be asked to bid.
    Sahib Al-Qabool still gets to decide.

    Note: opening bid cannot exceed 11, so bid 13 must be a subsequent bid.
    """
    hands = make_simple_hands()
    players = make_players_with_hands(hands)

    # Player 3 should NOT be called.
    agent_3 = MagicMock()

    agents = [
        make_agent_that_passes(),   # Player 0 (Qabool) → accept
        make_agent_that_bids(8),    # Player 1 → bid 8 (opening)
        make_agent_that_bids(13),   # Player 2 → bid 13 (stops Tasmiya)
        agent_3,
    ]

    engine = TasmiyaEngine()
    result = engine.run(
        players=players,
        agents=agents,
        sahib_al_qabool_id=0,
    )

    # Player 3 should not have been asked.
    agent_3.act.assert_not_called()

    assert result.winning_bidder_id == 2
    assert result.winning_bid_value == 13


def test_bid_history_is_recorded():
    """
    Verify that the bid history is correctly recorded.
    """
    hands = make_simple_hands()
    players = make_players_with_hands(hands)

    agents = [
        make_agent_that_passes(),   # Player 0 (Qabool) → accept
        make_agent_that_bids(8),    # Player 1 → bid 8
        make_agent_that_passes(),   # Player 2 → pass
        make_agent_that_bids(9),    # Player 3 → bid 9
    ]

    engine = TasmiyaEngine()
    result = engine.run(
        players=players,
        agents=agents,
        sahib_al_qabool_id=0,
    )

    # Bidding order from Qabool 0: [1, 2, 3], then Qabool.
    assert result.bid_history == [
        (1, 8),     # Player 1 bids 8
        (2, None),  # Player 2 passes
        (3, 9),     # Player 3 bids 9
        (0, None),  # Qabool passes (accepts)
    ]


def test_qabool_is_player_2():
    """
    When Sahib Al-Qabool is player 2, bidding order is 3, 0, 1.
    """
    hands = make_simple_hands()
    players = make_players_with_hands(hands)

    agents = [
        make_agent_that_passes(),   # Player 0 → pass
        make_agent_that_passes(),   # Player 1 → pass
        make_agent_that_bids(7),    # Player 2 (Qabool) → bid 7
        make_agent_that_passes(),   # Player 3 → pass
    ]

    engine = TasmiyaEngine()
    result = engine.run(
        players=players,
        agents=agents,
        sahib_al_qabool_id=2,
    )

    assert result.winning_bidder_id == 2
    assert result.winning_bid_value == 7
    assert result.playing_team_id == 0  # Player 2 is on team 0
    assert result.sahib_al_qabool_id == 2

    # Bid history order: [3, 0, 1, 2(qabool)]
    assert result.bid_history[0][0] == 3
    assert result.bid_history[1][0] == 0
    assert result.bid_history[2][0] == 1
    assert result.bid_history[3][0] == 2


def test_opening_bid_cannot_exceed_11():
    """
    The first actual bid cannot exceed 11.
    """
    hands = make_simple_hands()
    players = make_players_with_hands(hands)

    agents = [
        make_agent_that_passes(),   # Player 0 (Qabool)
        make_agent_that_bids(12),   # Player 1 → tries to bid 12 as opening
        make_agent_that_passes(),   # Player 2
        make_agent_that_passes(),   # Player 3
    ]

    engine = TasmiyaEngine()

    with pytest.raises(ValueError, match="Opening bid cannot exceed 11"):
        engine.run(
            players=players,
            agents=agents,
            sahib_al_qabool_id=0,
        )


def test_regular_bid_must_exceed_current_highest():
    """
    A regular player's bid must be strictly higher than the current highest.
    """
    hands = make_simple_hands()
    players = make_players_with_hands(hands)

    agents = [
        make_agent_that_passes(),   # Player 0 (Qabool)
        make_agent_that_bids(8),    # Player 1 → bid 8
        make_agent_that_bids(8),    # Player 2 → tries to bid 8 (same = invalid)
        make_agent_that_passes(),   # Player 3
    ]

    engine = TasmiyaEngine()

    with pytest.raises(ValueError, match="higher than the current highest"):
        engine.run(
            players=players,
            agents=agents,
            sahib_al_qabool_id=0,
        )
