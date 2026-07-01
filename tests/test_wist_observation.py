from environments.wist.observation import WistObservation
from intelligence.core.cards.card import Card
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit


def test_wist_observation_contains_visible_information():
    card = Card(suit=Suit.SPADES, rank=Rank.ACE)

    observation = WistObservation(
        player_id=0,
        hand=[card],
        trump_suit=Suit.SPADES,
        played_cards=[],
        team_scores={0: 0, 1: 0},
    )

    assert observation.player_id == 0
    assert observation.hand == [card]
    assert observation.trump_suit == Suit.SPADES
    assert observation.team_scores == {0: 0, 1: 0}