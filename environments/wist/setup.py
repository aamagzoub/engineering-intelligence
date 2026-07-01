from environments.wist.player import Player
from environments.wist.team import Team


def create_standard_players() -> list[Player]:
    """
    Create four Wist players.

    Opposite players are teammates:
    - Player 0 and Player 2
    - Player 1 and Player 3
    """

    return [
        Player(player_id=0, team_id=0),
        Player(player_id=1, team_id=1),
        Player(player_id=2, team_id=0),
        Player(player_id=3, team_id=1),
    ]


def create_standard_teams() -> list[Team]:
    """
    Create the two standard Wist teams.
    """

    return [
        Team(team_id=0, player_ids=(0, 2)),
        Team(team_id=1, player_ids=(1, 3)),
    ]