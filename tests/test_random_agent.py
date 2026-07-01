from agents.random.random_agent import RandomAgent
from environments.wist.actions import PlayCardAction
from environments.wist.observation import WistObservation
from intelligence.core.cards.card import Card
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit
from environments.wist.trick import Trick


def test_random_agent_returns_play_card_action():
    hand = [
        Card(Suit.SPADES, Rank.ACE),
        Card(Suit.HEARTS, Rank.KING),
    ]

    observation = WistObservation(
        player_id=0,
        hand=hand,
    )

    agent = RandomAgent()
    action = agent.act(observation)

    assert isinstance(action, PlayCardAction)
    assert action.player_id == 0
    assert action.card in hand

def test_random_agent_follows_leading_suit_if_possible():
    hand = [
        Card(Suit.SPADES, Rank.ACE),
        Card(Suit.HEARTS, Rank.KING),
    ]

    trick = Trick(leading_player_id=1)
    trick.play_card(1, Card(Suit.HEARTS, Rank.TWO))

    observation = WistObservation(
        player_id=0,
        hand=hand,
        current_trick=trick,
    )

    agent = RandomAgent()

    for _ in range(20):
        action = agent.act(observation)
        assert action.card.suit == Suit.HEARTS


def test_random_agent_can_play_any_card_if_void_in_leading_suit():
    hand = [
        Card(Suit.SPADES, Rank.ACE),
        Card(Suit.CLUBS, Rank.KING),
    ]

    trick = Trick(leading_player_id=1)
    trick.play_card(1, Card(Suit.HEARTS, Rank.TWO))

    observation = WistObservation(
        player_id=0,
        hand=hand,
        current_trick=trick,
    )

    agent = RandomAgent()
    action = agent.act(observation)

    assert action.card in hand