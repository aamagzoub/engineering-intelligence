from environments.wist.environment import WistEnvironment
from environments.wist.player import Player
from environments.wist.round_state import RoundState
from environments.wist.rules import trick_winner
from environments.wist.trick import Trick
from intelligence.core.agent import Agent
from intelligence.core.cards.deck import Deck
from environments.wist.dak import triggers_card_based_dak


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
        Play one complete trick and return only the winning player id.
        """

        trick_details = self.play_one_trick_details(
            environment=environment,
            agents=agents,
        )

        return trick_details["winner"]

    def play_one_trick_details(
        self,
        environment: WistEnvironment,
        agents: list[Agent],
    ) -> dict:
        """
        Play one complete trick and return full trick information.
        """

        if self.state is None:
            raise ValueError("Cannot play a trick before dealing cards.")

        if self.state.trump_suit is None:
            raise ValueError("Cannot play a trick before trump suit is set.")

        leader_id = self.next_leading_player_id

        self.state.current_trick = Trick(
            leading_player_id=leader_id
        )

        play_order = self._play_order_from_leader(
            leader_id
        )

        for player_id in play_order:
            observation = environment.observe(player_id)
            action = agents[player_id].act(observation)
            environment.apply_action(action)

        completed_trick = self.state.current_trick

        winner = trick_winner(
            trick=completed_trick,
            trump_suit=self.state.trump_suit,
        )

        self.state.completed_tricks.append(completed_trick)
        self.state.current_trick = None
        self.next_leading_player_id = winner

        return {
            "winner": winner,
            "leader": leader_id,
            "play_order": play_order,
            "trick": completed_trick,
        }

    def _play_order_from_leader(self, leader_id: int) -> list[int]:
        """
        Return counter-clockwise play order starting from the leader.
        """

        return [
            (leader_id + offset) % 4
            for offset in range(4)
        ]

    def has_card_based_dak(self) -> bool:
        """
        Return True if any player has a card-based Dak hand.
        """

        if self.state is None:
            raise ValueError("Cannot check Dak before dealing cards.")

        return any(
            triggers_card_based_dak(player.hand)
            for player in self.players
        )

    def first_card_based_dak_player_id(self) -> int | None:
        """
        Return the first player who triggers card-based Dak.

        For now, this checks players in player_id order.
        Later, we will adjust this to the correct Al-Tasmiya order.
        """

        if self.state is None:
            raise ValueError("Cannot check Dak before dealing cards.")

        for player in self.players:
            if triggers_card_based_dak(player.hand):
                return player.player_id

        return None