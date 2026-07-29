from dataclasses import dataclass


@dataclass(frozen=True)
class Team:
    """
    Represents one Wist team.

    In Sudanese Wist, opposite players form a team.
    """

    team_id: int
    player_ids: tuple[int, int]

    def contains_player(self, player_id: int) -> bool:
        return player_id in self.player_ids