"""
Tests for the rule-based Wist agent.
"""

from collections import Counter

from agents.wist_rule_based.rule_based_agent import RuleBasedAgent
from environments.wist.actions import BidAction, PassAction, PlayCardAction
from environments.wist.environment import WistEnvironment
from environments.wist.observation import BiddingObservation, WistObservation
from environments.wist.playing_engine import PlayingEngine
from environments.wist.round import Round
from environments.wist.setup import create_standard_players
from environments.wist.tasmiya_engine import TasmiyaEngine, determine_trump_suit
from environments.wist.trick import Trick
from intelligence.core.cards.card import Card
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit


# ---------------------------------------------------------------
# Bidding tests
# ---------------------------------------------------------------


def test_rule_based_agent_bids_with_strong_hand():
    """A hand with many trumps and high cards should bid."""
    hand = [
        Card(Suit.SPADES, Rank.ACE),
        Card(Suit.SPADES, Rank.KING),
        Card(Suit.SPADES, Rank.QUEEN),
        Card(Suit.SPADES, Rank.JACK),
        Card(Suit.SPADES, Rank.TEN),
        Card(Suit.HEARTS, Rank.ACE),
        Card(Suit.HEARTS, Rank.KING),
        Card(Suit.HEARTS, Rank.QUEEN),
        Card(Suit.DIAMONDS, Rank.ACE),
        Card(Suit.DIAMONDS, Rank.KING),
        Card(Suit.CLUBS, Rank.ACE),
        Card(Suit.CLUBS, Rank.THREE),
        Card(Suit.CLUBS, Rank.TWO),
    ]

    obs = BiddingObservation(
        player_id=0,
        hand=hand,
        is_opening_bid=True,
        is_sahib_al_qabool=False,
    )

    agent = RuleBasedAgent()
    action = agent.act(obs)

    assert isinstance(action, BidAction)
    assert action.value >= 7


def test_rule_based_agent_passes_with_weak_hand():
    """A hand with no high cards and short suits should pass."""
    hand = [
        Card(Suit.SPADES, Rank.TWO),
        Card(Suit.SPADES, Rank.THREE),
        Card(Suit.SPADES, Rank.FOUR),
        Card(Suit.HEARTS, Rank.TWO),
        Card(Suit.HEARTS, Rank.THREE),
        Card(Suit.HEARTS, Rank.FOUR),
        Card(Suit.HEARTS, Rank.FIVE),
        Card(Suit.DIAMONDS, Rank.TWO),
        Card(Suit.DIAMONDS, Rank.THREE),
        Card(Suit.CLUBS, Rank.TWO),
        Card(Suit.CLUBS, Rank.THREE),
        Card(Suit.CLUBS, Rank.FOUR),
        Card(Suit.CLUBS, Rank.FIVE),
    ]

    obs = BiddingObservation(
        player_id=1,
        hand=hand,
        is_opening_bid=True,
        is_sahib_al_qabool=False,
    )

    agent = RuleBasedAgent()
    action = agent.act(obs)

    assert isinstance(action, PassAction)


def test_rule_based_qabool_accepts_when_weak():
    """Sahib Al-Qabool with weak hand should accept (pass) opponent's bid."""
    hand = [
        Card(Suit.SPADES, Rank.TWO),
        Card(Suit.SPADES, Rank.THREE),
        Card(Suit.HEARTS, Rank.TWO),
        Card(Suit.HEARTS, Rank.THREE),
        Card(Suit.HEARTS, Rank.FOUR),
        Card(Suit.DIAMONDS, Rank.TWO),
        Card(Suit.DIAMONDS, Rank.THREE),
        Card(Suit.DIAMONDS, Rank.FOUR),
        Card(Suit.DIAMONDS, Rank.FIVE),
        Card(Suit.CLUBS, Rank.TWO),
        Card(Suit.CLUBS, Rank.THREE),
        Card(Suit.CLUBS, Rank.FOUR),
        Card(Suit.CLUBS, Rank.FIVE),
    ]

    obs = BiddingObservation(
        player_id=0,
        hand=hand,
        current_highest_bid=8,
        is_sahib_al_qabool=True,
        is_opening_bid=False,
    )

    agent = RuleBasedAgent()
    action = agent.act(obs)

    # Should accept (pass) since hand is too weak to match bid 8.
    assert isinstance(action, PassAction)


# ---------------------------------------------------------------
# Card play tests
# ---------------------------------------------------------------


def test_rule_based_leads_trump_when_must():
    """When must_lead_trump is set, should play a trump card."""
    hand = [
        Card(Suit.SPADES, Rank.ACE),
        Card(Suit.SPADES, Rank.KING),
        Card(Suit.HEARTS, Rank.ACE),
    ]

    obs = WistObservation(
        player_id=0,
        hand=hand,
        trump_suit=Suit.SPADES,
        must_lead_trump=True,
    )

    agent = RuleBasedAgent()
    action = agent.act(obs)

    assert isinstance(action, PlayCardAction)
    assert action.card.suit == Suit.SPADES


def test_rule_based_follows_suit():
    """When following suit, should play a card in the leading suit."""
    hand = [
        Card(Suit.SPADES, Rank.ACE),
        Card(Suit.HEARTS, Rank.KING),
        Card(Suit.HEARTS, Rank.TWO),
    ]

    trick = Trick(leading_player_id=1)
    trick.play_card(1, Card(Suit.HEARTS, Rank.TEN))

    obs = WistObservation(
        player_id=0,
        hand=hand,
        current_trick=trick,
        trump_suit=Suit.SPADES,
    )

    agent = RuleBasedAgent()
    action = agent.act(obs)

    assert isinstance(action, PlayCardAction)
    assert action.card.suit == Suit.HEARTS


def test_rule_based_plays_lowest_when_partner_winning():
    """When partner is winning, play lowest card to not waste."""
    hand = [
        Card(Suit.HEARTS, Rank.ACE),
        Card(Suit.HEARTS, Rank.KING),
        Card(Suit.HEARTS, Rank.TWO),
    ]

    # Player 2 (our partner) played Ace of Hearts, player 3 played a low card.
    trick = Trick(leading_player_id=2)
    trick.play_card(2, Card(Suit.HEARTS, Rank.ACE))  # Partner leads with Ace.
    trick.play_card(3, Card(Suit.HEARTS, Rank.THREE))  # Opponent plays low.

    obs = WistObservation(
        player_id=0,  # We're player 0, partner is player 2.
        hand=hand,
        current_trick=trick,
        trump_suit=Suit.SPADES,
    )

    agent = RuleBasedAgent()
    action = agent.act(obs)

    assert isinstance(action, PlayCardAction)
    # Should play lowest since partner is winning.
    assert action.card == Card(Suit.HEARTS, Rank.TWO)


def test_rule_based_trumps_when_void():
    """When void in led suit and partner not winning, should trump."""
    hand = [
        Card(Suit.SPADES, Rank.FIVE),  # Trump
        Card(Suit.CLUBS, Rank.TWO),
        Card(Suit.CLUBS, Rank.THREE),
    ]

    # Hearts led, we have no hearts.
    trick = Trick(leading_player_id=1)
    trick.play_card(1, Card(Suit.HEARTS, Rank.ACE))  # Opponent leads.

    obs = WistObservation(
        player_id=0,
        hand=hand,
        current_trick=trick,
        trump_suit=Suit.SPADES,
    )

    agent = RuleBasedAgent()
    action = agent.act(obs)

    assert isinstance(action, PlayCardAction)
    # Should trump since we're void in hearts and opponent is winning.
    assert action.card.suit == Suit.SPADES


def test_rule_based_discards_when_partner_winning_and_void():
    """When void in led suit but partner is winning, discard instead of trump."""
    hand = [
        Card(Suit.SPADES, Rank.FIVE),  # Trump
        Card(Suit.CLUBS, Rank.TWO),
        Card(Suit.CLUBS, Rank.THREE),
    ]

    # Partner (player 2) is winning with Ace of Hearts.
    trick = Trick(leading_player_id=2)
    trick.play_card(2, Card(Suit.HEARTS, Rank.ACE))  # Partner leads with Ace.
    trick.play_card(3, Card(Suit.HEARTS, Rank.KING))  # Opponent plays high but below Ace.

    obs = WistObservation(
        player_id=0,  # Partner is player 2.
        hand=hand,
        current_trick=trick,
        trump_suit=Suit.SPADES,
    )

    agent = RuleBasedAgent()
    action = agent.act(obs)

    assert isinstance(action, PlayCardAction)
    # Should discard (not waste trump) since partner is winning.
    assert action.card.suit != Suit.SPADES or action.card == Card(Suit.SPADES, Rank.FIVE)
    # More specifically, should discard the lowest non-trump.
    assert action.card in [Card(Suit.CLUBS, Rank.TWO), Card(Suit.CLUBS, Rank.THREE)]


# ---------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------


def test_rule_based_plays_full_shota():
    """The rule-based agent should be able to play a full 13-trick Shota."""
    for _ in range(10):
        players = create_standard_players()
        agents = [RuleBasedAgent(), RuleBasedAgent(), RuleBasedAgent(), RuleBasedAgent()]

        round_ = Round(players)
        round_.deal()

        if round_.has_card_based_dak():
            continue

        tasmiya = TasmiyaEngine()
        result = tasmiya.run(players=players, agents=agents, sahib_al_qabool_id=0)

        if result.is_dak:
            continue

        round_.state.trump_suit = result.trump_suit
        round_.state.winning_bidder_id = result.winning_bidder_id
        round_.next_leading_player_id = result.winning_bidder_id

        env = WistEnvironment(round_.state)
        engine = PlayingEngine()
        tricks = engine.play_shota(round_, env, agents)

        assert tricks[0] + tricks[1] == 13
        for player in players:
            assert len(player.hand) == 0


def test_rule_based_beats_random_statistically():
    """
    Rule-based agents should win more tricks than random agents
    when playing as partners against random opponents.

    Run 50 games: rule-based on team 0, random on team 1.
    The rule-based team should win more total tricks across all games.
    """
    from agents.random.random_agent import RandomAgent

    rule_based_total = 0
    random_total = 0

    for _ in range(50):
        players = create_standard_players()
        # Team 0 (players 0, 2): rule-based
        # Team 1 (players 1, 3): random
        agents = [RuleBasedAgent(), RandomAgent(), RuleBasedAgent(), RandomAgent()]

        round_ = Round(players)
        round_.deal()

        if round_.has_card_based_dak():
            continue

        trump = determine_trump_suit(players[0].hand)
        round_.state.trump_suit = trump
        round_.state.winning_bidder_id = 0
        round_.next_leading_player_id = 0

        env = WistEnvironment(round_.state)
        engine = PlayingEngine()
        tricks = engine.play_shota(round_, env, agents)

        rule_based_total += tricks[0]
        random_total += tricks[1]

    # Rule-based should win significantly more tricks than random.
    assert rule_based_total > random_total, (
        f"Rule-based ({rule_based_total}) should beat random ({random_total})"
    )
