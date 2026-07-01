from environments.wist.environment import WistEnvironment
from environments.wist.round import Round
from intelligence.core.agent import Agent


class PlayingEngine:
    """
    Plays all 13 tricks of a Shota.
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