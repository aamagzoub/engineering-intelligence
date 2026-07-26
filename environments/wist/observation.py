from dataclasses import dataclass, field

from intelligence.core.cards.card import Card
from intelligence.core.cards.suit import Suit
from intelligence.core.observation import Observation
from environments.wist.trick import Trick


@dataclass(frozen=True)
class WistObservation(Observation):
    """
    What one Wist player is allowed to observe during trick play.
    No hidden hands are included.
    """

    player_id: int
    hand: list[Card]
    current_trick: Trick | None = None
    trump_suit: Suit | None = None
    played_cards: list[Card] = field(default_factory=list)
    team_scores: dict[int, int] = field(default_factory=dict)
    must_lead_trump: bool = False


@dataclass(frozen=True)
class BiddingObservation(Observation):
    """
    What one Wist player is allowed to observe during Al-Tasmiya.

    The player sees:
    - Their own hand
    - Previous bids (player_id, value) and passes (player_id)
    - The current highest bid value (or None if no one bid yet)
    - Whether they are Sahib Al-Qabool
    - Whether they are the first bidder (opening bid rule applies)
    """

    player_id: int
    hand: list[Card]
    previous_bids: list[tuple[int, int | None]] = field(default_factory=list)
    current_highest_bid: int | None = None
    is_sahib_al_qabool: bool = False
    is_opening_bid: bool = False