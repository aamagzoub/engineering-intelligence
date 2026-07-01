from dataclasses import dataclass, field
from enum import Enum


class DakType(Enum):
    CARD_BASED = "card_based"
    PASS_BASED = "pass_based"


@dataclass
class GameState:
    team_scores: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})
    completed_shotas: int = 0
    winner_team_id: int | None = None
    pass_based_dak_count: int = 0

    def apply_shota_score(self, score_delta: dict[int, int]) -> None:
        for team_id, delta in score_delta.items():
            self.team_scores[team_id] += delta

        self.completed_shotas += 1

        for team_id, score in self.team_scores.items():
            if score >= 25:
                self.winner_team_id = team_id

    def apply_dak(self, dak_type: DakType) -> None:
        if dak_type == DakType.PASS_BASED:
            if self.pass_based_dak_count >= 2:
                raise ValueError("Pass-based Dak can only happen twice per game.")
            self.pass_based_dak_count += 1

        # Any Dak counts as one Shota.
        self.completed_shotas += 1

    def apply_seek(self, team_id: int) -> None:
        self.winner_team_id = team_id

    def is_finished(self) -> bool:
        return self.winner_team_id is not None or self.completed_shotas >= 5