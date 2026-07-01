from dataclasses import dataclass, field

from intelligence.core.cards.card import Card
from intelligence.core.cards.suit import Suit
from intelligence.core.observation import Observation
from environments.wist.trick import Trick


@dataclass(frozen=True)
class WistObservation(Observation):
    """
    What one Wist player is allowed to observe.
    No hidden hands are included.
    """

    player_id: int
    hand: list[Card]
    current_trick: Trick | None = None
    trump_suit: Suit | None = None
    played_cards: list[Card] = field(default_factory=list)
    team_scores: dict[int, int] = field(default_factory=dict)