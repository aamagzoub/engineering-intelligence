from dataclasses import dataclass, field

from environments.wist.player import Player
from environments.wist.trick import Trick
from intelligence.core.cards.card import Card
from intelligence.core.cards.suit import Suit


@dataclass
class RoundState:
    """
    Represents the current state of one Wist round.

    This is the environment's source of truth for a round.
    """

    players: dict[int, Player]
    trump_suit: Suit | None = None
    winning_bidder_id: int | None = None
    current_trick: Trick | None = None
    completed_tricks: list[Trick] = field(default_factory=list)
    played_cards: list[Card] = field(default_factory=list)

    def get_player(self, player_id: int) -> Player:
        if player_id not in self.players:
            raise ValueError(f"Unknown player_id: {player_id}")

        return self.players[player_id]

    @property
    def is_first_trick(self) -> bool:
        """True if no tricks have been completed yet."""
        return len(self.completed_tricks) == 0