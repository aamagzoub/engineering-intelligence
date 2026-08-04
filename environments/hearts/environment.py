"""
Hearts environment — manages game state for one shota of trick play.

Provides observations to agents and applies their actions.
Enforces legal moves but does NOT expose scoring information.
"""

from environments.hearts.actions import PlayCardAction
from environments.hearts.observation import HeartsObservation
from environments.hearts.player import HeartsPlayer
from environments.hearts.rules import legal_cards, rank_value, trick_winner
from environments.hearts.trick import Trick
from intelligence.core.action import Action
from intelligence.core.cards.card import Card
from intelligence.core.cards.suit import Suit
from intelligence.core.environment import Environment


class HeartsEnvironment(Environment):
    """
    Hearts environment for one shota of trick play.

    Owns the true game state and exposes observations to players.
    The observations intentionally do NOT include any scoring information.
    """

    def __init__(self, players: list[HeartsPlayer], first_leader_id: int) -> None:
        self.players = players
        self.current_trick: Trick | None = None
        self.trick_number: int = 1
        self.cards_played: list[Card] = []
        self.next_leader_id: int = first_leader_id

        # Start the first trick.
        self._start_new_trick()

    def _start_new_trick(self) -> None:
        """Begin a new trick with the current leader."""
        self.current_trick = Trick(leading_player_id=self.next_leader_id)

    def observe(self, player_id: int) -> HeartsObservation:
        """Build observation for one player — no scoring info exposed."""
        player = self._get_player(player_id)

        leading_suit = None
        if self.current_trick and self.current_trick.played_cards:
            leading_suit = self.current_trick.leading_suit

        is_first_trick = (self.trick_number == 1)

        # Compute legal cards.
        playable = legal_cards(
            hand=player.hand,
            leading_suit=leading_suit,
            is_first_trick=is_first_trick,
        )

        # Current trick cards as (player_id, card) tuples.
        current_trick_cards = []
        if self.current_trick:
            current_trick_cards = [
                (pc.player_id, pc.card) for pc in self.current_trick.played_cards
            ]

        # Tricks won per player.
        tricks_won = {p.player_id: p.tricks_won for p in self.players}

        return HeartsObservation(
            player_id=player_id,
            hand=list(player.hand),
            legal_cards=playable,
            current_trick_cards=current_trick_cards,
            tricks_won_per_player=tricks_won,
            trick_number=self.trick_number,
            cards_played_this_shota=list(self.cards_played),
        )

    def apply_action(self, action: Action) -> int | None:
        """
        Apply a card play action.

        Returns the trick winner's player_id if the trick is complete,
        or None if more cards are needed.
        """
        if not isinstance(action, PlayCardAction):
            raise TypeError("HeartsEnvironment only supports PlayCardAction.")

        player = self._get_player(action.player_id)
        card = player.play_card(action.card)

        if self.current_trick is None:
            raise ValueError("No current trick exists.")

        self.current_trick.play_card(player_id=player.player_id, card=card)
        self.cards_played.append(card)

        # Check if trick is complete.
        if self.current_trick.is_complete():
            winner_id = trick_winner(self.current_trick)
            winner = self._get_player(winner_id)
            winner.collect_trick(self.current_trick.all_cards())

            self.next_leader_id = winner_id
            self.trick_number += 1

            if self.trick_number <= 13:
                self._start_new_trick()

            return winner_id

        return None

    def is_shota_complete(self) -> bool:
        """Check if all 13 tricks have been played."""
        return self.trick_number > 13

    def current_player_id(self) -> int:
        """Determine whose turn it is to play."""
        if self.current_trick is None:
            raise ValueError("No active trick.")

        n_played = len(self.current_trick.played_cards)
        leader = self.current_trick.leading_player_id

        # Players play in clockwise order from the leader.
        return (leader + n_played) % 4

    def _get_player(self, player_id: int) -> HeartsPlayer:
        """Get player by ID."""
        for p in self.players:
            if p.player_id == player_id:
                return p
        raise ValueError(f"Player {player_id} not found.")
