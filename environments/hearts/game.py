"""
Full Hearts game orchestrator.

A Hearts game consists of 5 shotas. Each shota:
1. Deal 13 cards to each player
2. Pass 4 cards to the left
3. Play 13 tricks
4. Score (zero-sum)

The player with the highest total score after 5 shotas wins.
The player with the lowest total score loses.
"""

from dataclasses import dataclass, field

from environments.hearts.environment import HeartsEnvironment
from environments.hearts.observation import PassingObservation
from environments.hearts.actions import PassCardsAction
from environments.hearts.player import HeartsPlayer
from environments.hearts.playing_engine import PlayingEngine
from environments.hearts.scoring import score_shota
from intelligence.core.agent import Agent
from intelligence.core.cards.deck import Deck


@dataclass
class ShotaResult:
    """Result of one shota."""

    shota_number: int
    scores: dict[int, int]  # player_id → score
    tricks_won: dict[int, int]  # player_id → tricks


@dataclass
class GameResult:
    """Final result of a complete Hearts game."""

    winner_id: int  # Player with highest total score
    loser_id: int  # Player with lowest total score
    final_scores: dict[int, int]  # player_id → total score
    shota_results: list[ShotaResult] = field(default_factory=list)


class HeartsGame:
    """
    Orchestrates a full Hearts game (5 shotas).

    Usage:
        game = HeartsGame(agents=[agent0, agent1, agent2, agent3])
        result = game.play()
    """

    def __init__(self, agents: list[Agent]) -> None:
        if len(agents) != 4:
            raise ValueError("Hearts requires exactly 4 agents.")

        self.agents = agents
        self.players = [HeartsPlayer(player_id=i) for i in range(4)]
        self.playing_engine = PlayingEngine()

        # Dealer starts at player 0, rotates clockwise.
        self.dealer_id = 0

        # Cumulative scores.
        self.total_scores: dict[int, int] = {i: 0 for i in range(4)}
        self.shota_results: list[ShotaResult] = []

    def play(self) -> GameResult:
        """Play a full 5-shota game and return the result."""
        for shota_num in range(1, 6):
            result = self._play_one_shota(shota_num)
            self.shota_results.append(result)

            # Accumulate scores.
            for pid, score in result.scores.items():
                self.total_scores[pid] += score

            # Rotate dealer.
            self.dealer_id = (self.dealer_id + 1) % 4

        # Determine winner (highest) and loser (lowest).
        winner_id = max(self.total_scores, key=self.total_scores.get)
        loser_id = min(self.total_scores, key=self.total_scores.get)

        return GameResult(
            winner_id=winner_id,
            loser_id=loser_id,
            final_scores=dict(self.total_scores),
            shota_results=list(self.shota_results),
        )

    def _play_one_shota(self, shota_number: int) -> ShotaResult:
        """Play one shota: deal → pass → play 13 tricks → score."""

        # Reset players for new shota.
        for player in self.players:
            player.reset_shota()

        # --- Deal ---
        deck = Deck()
        deck.shuffle()
        for player in self.players:
            player.receive_cards(deck.deal(13))

        # --- Passing phase (4 cards to the left) ---
        self._passing_phase()

        # --- Playing phase (13 tricks) ---
        # Player to dealer's left leads first.
        first_leader = (self.dealer_id + 1) % 4
        environment = HeartsEnvironment(self.players, first_leader)
        tricks_won = self.playing_engine.play_shota(environment, self.agents)

        # --- Scoring ---
        collected_cards = {p.player_id: list(p.collected_cards) for p in self.players}
        scores = score_shota(collected_cards, tricks_won)

        return ShotaResult(
            shota_number=shota_number,
            scores=scores,
            tricks_won=tricks_won,
        )

    def _passing_phase(self) -> None:
        """
        Each player selects 4 cards to pass to the left.
        Cards are passed simultaneously — you pass BEFORE seeing what you receive.
        """
        # Collect all pass actions.
        cards_to_pass: dict[int, tuple] = {}

        for player in self.players:
            observation = PassingObservation(
                player_id=player.player_id,
                hand=list(player.hand),
            )
            action = self.agents[player.player_id].act(observation)

            if not isinstance(action, PassCardsAction):
                raise TypeError(
                    f"Expected PassCardsAction from agent {player.player_id}, "
                    f"got {type(action).__name__}."
                )

            if len(action.cards) != 4:
                raise ValueError(
                    f"Agent {player.player_id} must pass exactly 4 cards."
                )

            # Validate all cards are in hand.
            for card in action.cards:
                if card not in player.hand:
                    raise ValueError(
                        f"Agent {player.player_id} tried to pass {card} "
                        f"which is not in their hand."
                    )

            cards_to_pass[player.player_id] = action.cards

        # Remove passed cards from each player.
        for player in self.players:
            player.remove_cards(list(cards_to_pass[player.player_id]))

        # Give cards to the player on the left (player_id + 1) % 4.
        for player in self.players:
            receiver_id = (player.player_id + 1) % 4
            receiver = self.players[receiver_id]
            receiver.receive_cards(list(cards_to_pass[player.player_id]))
