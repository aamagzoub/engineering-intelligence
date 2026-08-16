"""Reward normalization and curiosity-driven exploration bonuses."""

from __future__ import annotations

import math


class RewardNormalizer:
    """Running normalization using Welford's online algorithm.

    Maintains a running mean and variance of observed rewards and normalizes
    new rewards to zero-mean, unit-variance scale.
    """

    def __init__(self) -> None:
        self._mean: float = 0.0
        self._m2: float = 0.0
        self._count: int = 0

    def normalize(self, reward: float) -> float:
        """Normalize reward using running statistics (Welford's algorithm).

        Updates internal mean/variance estimates with the new reward, then
        returns the normalized value: (reward - mean) / std.
        """
        self._count += 1
        delta = reward - self._mean
        self._mean += delta / self._count
        delta2 = reward - self._mean
        self._m2 += delta * delta2

        std = math.sqrt(self._m2 / max(self._count, 1)) + 1e-8
        return (reward - self._mean) / std

    @property
    def mean(self) -> float:
        """Current running mean."""
        return self._mean

    @property
    def std(self) -> float:
        """Current running standard deviation."""
        return math.sqrt(self._m2 / max(self._count, 1)) + 1e-8

    @property
    def count(self) -> int:
        """Number of rewards observed."""
        return self._count

    def to_dict(self) -> dict:
        """Serialize normalizer state to a dictionary."""
        return {"mean": self._mean, "m2": self._m2, "count": self._count}

    @classmethod
    def from_dict(cls, data: dict) -> "RewardNormalizer":
        """Deserialize normalizer state from a dictionary."""
        rn = cls()
        rn._mean = data.get("mean", 0.0)
        rn._m2 = data.get("m2", 0.0)
        rn._count = data.get("count", 0)
        return rn


class CuriosityModule:
    """Exploration bonus based on state visit counts.

    Provides an intrinsic reward bonus inversely proportional to the square
    root of the visit count for a given state, encouraging exploration of
    less-visited states.
    """

    def __init__(self, scale: float = 0.1) -> None:
        self.scale = scale
        self._visit_counts: dict[str, int] = {}

    def visit(self, state_key: str) -> None:
        """Record a state visit, incrementing its count."""
        self._visit_counts[state_key] = self._visit_counts.get(state_key, 0) + 1

    def bonus(self, state_key: str) -> float:
        """Return curiosity bonus: scale / sqrt(visit_count).

        Returns full scale for unvisited states.
        """
        count = self._visit_counts.get(state_key, 0)
        if count == 0:
            return self.scale  # Max bonus for unvisited
        return self.scale / math.sqrt(count)

    def get_count(self, state_key: str) -> int:
        """Return the visit count for a given state."""
        return self._visit_counts.get(state_key, 0)
