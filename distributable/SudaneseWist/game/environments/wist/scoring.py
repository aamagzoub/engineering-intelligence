from dataclasses import dataclass


@dataclass(frozen=True)
class ShotaScoreResult:
    playing_team_score: int
    defending_team_score: int
    seek_team_id: int | None = None


def score_shota(
    playing_team_id: int,
    defending_team_id: int,
    bid: int,
    playing_team_tricks: int,
    defending_team_tricks: int,
) -> dict[int, int]:
    """
    Score one Shota.

    - If playing team reaches/exceeds bid:
      playing team scores actual tricks won.
    - If playing team fails:
      playing team loses bid amount.
      defending team scores actual tricks won.
    - Scores may be negative.
    """

    if playing_team_tricks + defending_team_tricks != 13:
        raise ValueError("A Shota must contain exactly 13 tricks.")

    if playing_team_tricks >= bid:
        return {
            playing_team_id: playing_team_tricks,
            defending_team_id: 0,
        }

    return {
        playing_team_id: -bid,
        defending_team_id: defending_team_tricks,
    }


def detect_seek(team_tricks: dict[int, int]) -> int | None:
    """
    Seek happens when one team wins all 13 tricks.
    """

    for team_id, tricks_won in team_tricks.items():
        if tricks_won == 13:
            return team_id

    return None