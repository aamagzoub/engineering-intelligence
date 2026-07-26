"""
Statistics collector for the Wist AI Laboratory.

Tracks game-level and shota-level metrics across multiple games
for display in the Stats dashboard.
"""

from dataclasses import dataclass, field


@dataclass
class GameStats:
    """Accumulated statistics across multiple games."""

    games_played: int = 0
    team_1_wins: int = 0
    team_2_wins: int = 0
    draws: int = 0

    total_shotas: int = 0
    total_tricks_team_1: int = 0
    total_tricks_team_2: int = 0

    bid_attempts: int = 0
    bid_successes: int = 0

    total_bids_value: int = 0
    bid_count: int = 0

    dak_count: int = 0
    seek_count: int = 0

    # Per-game history for charts.
    win_history: list[int] = field(default_factory=list)  # 0=team1, 1=team2, -1=draw
    tricks_history: list[tuple[int, int]] = field(default_factory=list)

    def record_game(self, winner_team: int | None, score_1: int, score_2: int) -> None:
        self.games_played += 1
        if winner_team == 0:
            self.team_1_wins += 1
            self.win_history.append(0)
        elif winner_team == 1:
            self.team_2_wins += 1
            self.win_history.append(1)
        else:
            self.draws += 1
            self.win_history.append(-1)

    def record_shota(self, team_1_tricks: int, team_2_tricks: int,
                     bid: int, playing_team_id: int, bid_met: bool) -> None:
        self.total_shotas += 1
        self.total_tricks_team_1 += team_1_tricks
        self.total_tricks_team_2 += team_2_tricks

        self.bid_attempts += 1
        if bid_met:
            self.bid_successes += 1

        self.total_bids_value += bid
        self.bid_count += 1

        self.tricks_history.append((team_1_tricks, team_2_tricks))

    def record_dak(self) -> None:
        self.dak_count += 1

    def record_seek(self) -> None:
        self.seek_count += 1

    # Computed properties.

    @property
    def team_1_win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.team_1_wins / self.games_played * 100

    @property
    def team_2_win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.team_2_wins / self.games_played * 100

    @property
    def avg_tricks_team_1(self) -> float:
        if self.total_shotas == 0:
            return 0.0
        return self.total_tricks_team_1 / self.total_shotas

    @property
    def avg_tricks_team_2(self) -> float:
        if self.total_shotas == 0:
            return 0.0
        return self.total_tricks_team_2 / self.total_shotas

    @property
    def bid_success_rate(self) -> float:
        if self.bid_attempts == 0:
            return 0.0
        return self.bid_successes / self.bid_attempts * 100

    @property
    def avg_bid(self) -> float:
        if self.bid_count == 0:
            return 0.0
        return self.total_bids_value / self.bid_count

    @property
    def dak_rate(self) -> float:
        if self.total_shotas == 0:
            return 0.0
        return self.dak_count / (self.total_shotas + self.dak_count) * 100

    def reset(self) -> None:
        """Clear all stats."""
        self.games_played = 0
        self.team_1_wins = 0
        self.team_2_wins = 0
        self.draws = 0
        self.total_shotas = 0
        self.total_tricks_team_1 = 0
        self.total_tricks_team_2 = 0
        self.bid_attempts = 0
        self.bid_successes = 0
        self.total_bids_value = 0
        self.bid_count = 0
        self.dak_count = 0
        self.seek_count = 0
        self.win_history.clear()
        self.tricks_history.clear()
