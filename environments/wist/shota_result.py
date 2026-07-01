from dataclasses import dataclass


@dataclass(frozen=True)
class ShotaResult:
    """
    Result of one completed Shota.
    """

    playing_team_id: int
    defending_team_id: int
    bid: int
    team_tricks: dict[int, int]