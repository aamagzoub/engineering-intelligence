from dataclasses import dataclass, field

from intelligence.core.cards.card import Card
from intelligence.core.observation import Observation


@dataclass(frozen=True)
class HeartsObservation(Observation):
    """
    What a Hearts player sees during trick play.

    Intentionally minimal — no scoring info, no hints about
    which cards are good or bad. The agent must discover that.
    """

    player_id: int
    hand: list[Card]
    legal_cards: list[Card]
    current_trick_cards: list[tuple[int, Card]] = field(default_factory=list)
    tricks_won_per_player: dict[int, int] = field(default_factory=dict)
    trick_number: int = 1
    cards_played_this_shota: list[Card] = field(default_factory=list)


@dataclass(frozen=True)
class PassingObservation(Observation):
    """
    What a player sees during the passing phase.
    Just their 13-card hand — pick 4 to pass left.
    """

    player_id: int
    hand: list[Card]
