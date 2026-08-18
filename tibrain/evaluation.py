"""Evaluation module: Elo tracking and meta-learning for hyperparameter adaptation."""

from __future__ import annotations

from collections import deque


class EloTracker:
    """Elo rating tracker for measuring agent improvement."""

    def __init__(self, initial_elo: float = 1000.0, k_factor: float = 32.0) -> None:
        self.elo = initial_elo
        self.k_factor = k_factor
        self._history: list[tuple[int, float]] = []

    def update(self, won: bool, opponent_elo: float = 1000.0) -> None:
        """Update Elo after a game result using standard Elo formula."""
        expected = 1.0 / (1.0 + 10 ** ((opponent_elo - self.elo) / 400))
        actual = 1.0 if won else 0.0
        self.elo += self.k_factor * (actual - expected)

    def record(self, episode: int) -> None:
        """Record a (episode, elo) snapshot. Retains last 100."""
        self._history.append((episode, self.elo))
        if len(self._history) > 100:
            self._history = self._history[-100:]

    @property
    def history(self) -> list[tuple[int, float]]:
        """Return the recorded (episode, elo) snapshots."""
        return self._history

    def to_dict(self) -> dict:
        """Serialize EloTracker state to a dictionary."""
        return {"elo": self.elo, "k_factor": self.k_factor, "history": self._history[-100:]}

    @classmethod
    def from_dict(cls, data: dict) -> "EloTracker":
        """Deserialize EloTracker from a dictionary."""
        tracker = cls(
            initial_elo=data.get("elo", 1000.0),
            k_factor=data.get("k_factor", 32.0),
        )
        tracker._history = data.get("history", [])
        return tracker


class MetaLearner:
    """Tracks recent scores and suggests hyperparameter adjustments."""

    def __init__(self, window_size: int = 50, adjustment_interval: int = 200) -> None:
        self._scores: deque[float] = deque(maxlen=window_size)
        self._adjustment_interval = adjustment_interval
        self._last_adjustment: int = 0

    def record_score(self, score: float) -> None:
        """Record a score from a completed episode."""
        self._scores.append(score)

    def should_adjust(self, episode: int) -> bool:
        """Check if enough episodes have passed for adjustment."""
        return (
            episode - self._last_adjustment >= self._adjustment_interval
            and len(self._scores) >= 30
        )

    def suggest_adjustments(
        self,
        current_epsilon: float,
        episode: int,
    ) -> dict[str, float]:
        """
        Suggest hyperparameter changes based on recent performance trends.

        Compares average of last 10 scores (recent) vs overall window average (total).
        - If recent > total * 1.1 (improving >10%): suggest epsilon *= 0.95 (min 0.02)
        - If recent < total * 0.8 (declining >20%) and epsilon < 0.3: suggest epsilon *= 1.1 (max 0.3)

        Returns dict of suggested new values (empty if no change needed).
        """
        self._last_adjustment = episode

        if len(self._scores) == 0:
            return {}

        avg_score = sum(self._scores) / len(self._scores)
        recent_10 = list(self._scores)[-10:]
        avg_recent = sum(recent_10) / len(recent_10)

        adjustments: dict[str, float] = {}

        # Performance improving > 10% → reduce exploration
        if avg_recent > avg_score * 1.1:
            adjustments["epsilon"] = max(0.005, current_epsilon * 0.95)
        # Performance declining > 20% → increase exploration
        elif avg_recent < avg_score * 0.8 and current_epsilon < 0.3:
            adjustments["epsilon"] = min(0.3, current_epsilon * 1.1)

        return adjustments
