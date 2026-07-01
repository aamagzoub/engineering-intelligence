from agents.random.random_agent import RandomAgent
from environments.wist.environment import WistEnvironment
from environments.wist.player import Player
from environments.wist.round_state import RoundState
from environments.wist.rules import trick_winner
from environments.wist.trick import Trick
from intelligence.core.agent import Agent
from intelligence.core.cards.deck import Deck


class Round:
    """
    Controls one round of Wist.
    """

    def __init__(self, players: list[Player]) -> None:
        if len(players) != 4:
            raise ValueError("A Wist round requires exactly 4 players.")

        self.players = players
        self.deck = Deck()
        self.state: RoundState | None = None
        self.next_leading_player_id = 0

    def deal(self) -> None:
        """
        Shuffle and deal 13 cards to each player.
        """

        self.deck.shuffle()

        for player in self.players:
            player.hand.clear()

        for _ in range(13):
            for player in self.players:
                player.receive_cards([self.deck.deal(1)[0]])

        self.state = RoundState(
            players={player.player_id: player for player in self.players}
        )

    def play_one_trick(
        self,
        environment: WistEnvironment,
        agents: list[Agent],
    ) -> int:
        """
        Play one complete trick and return the winning player id.
        """

        if self.state is None:
            raise ValueError("Cannot play a trick before dealing cards.")

        if self.state.trump_suit is None:
            raise ValueError("Cannot play a trick before trump suit is set.")

        self.state.current_trick = Trick(
            leading_player_id=self.next_leading_player_id
        )

        play_order = self._play_order_from_leader(
            self.next_leading_player_id
        )

        for player_id in play_order:
            observation = environment.observe(player_id)
            action = agents[player_id].act(observation)
            environment.apply_action(action)

        winner = trick_winner(
            trick=self.state.current_trick,
            trump_suit=self.state.trump_suit,
        )

        self.state.completed_tricks.append(self.state.current_trick)
        self.state.current_trick = None
        self.next_leading_player_id = winner

        return winner

    def _play_order_from_leader(self, leader_id: int) -> list[int]:
        """
        Return counter-clockwise play order starting from the leader.
        """

        return [
            (leader_id + offset) % 4
            for offset in range(4)
        ]