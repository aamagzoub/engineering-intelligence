"""
Hearts playing engine — runs 13 tricks for one shota.

Asks each agent to play in turn, applying actions to the environment.
"""

from environments.hearts.environment import HeartsEnvironment
from intelligence.core.agent import Agent


class PlayingEngine:
    """Runs the trick-play phase of one Hearts shota."""

    def play_shota(
        self,
        environment: HeartsEnvironment,
        agents: list[Agent],
    ) -> dict[int, int]:
        """
        Play all 13 tricks.

        Returns:
            dict of player_id → tricks won
        """
        while not environment.is_shota_complete():
            # Determine whose turn it is.
            current_player_id = environment.current_player_id()
            agent = agents[current_player_id]

            # Get observation and ask agent to act.
            observation = environment.observe(current_player_id)
            action = agent.act(observation)

            # Apply the action.
            environment.apply_action(action)

        # Return tricks won per player.
        return {p.player_id: p.tricks_won for p in environment.players}
