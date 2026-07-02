from environments.wist.environment import WistEnvironment
from environments.wist.round import Round
from intelligence.core.agent import Agent


class PlayingEngine:
    """
    Plays Wist tricks and full Shotas.
    """

    def play_shota(
        self,
        round_: Round,
        environment: WistEnvironment,
        agents: list[Agent],
    ) -> dict[int, int]:

        team_tricks = {
            0: 0,
            1: 0,
        }

        for _ in range(13):
            winner = round_.play_one_trick(
                environment,
                agents,
            )

            team_id = round_.players[winner].team_id
            team_tricks[team_id] += 1

        return team_tricks

    def play_trick(
        self,
        round_: Round,
        environment: WistEnvironment,
        agents: list[Agent],
    ) -> int:
        """
        Plays one trick only and returns the index of the player who won the trick.

        This keeps old GUI/controller code safe.
        """

        winner = round_.play_one_trick(
            environment,
            agents,
        )

        return winner

    def play_trick_details(
        self,
        round_: Round,
        environment: WistEnvironment,
        agents: list[Agent],
    ) -> dict:
        """
        Plays one trick only and returns full trick details.
        """

        return round_.play_one_trick_details(
            environment=environment,
            agents=agents,
        )