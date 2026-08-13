"""ObservationStore — Internal store for raw observations.

This module provides an in-memory store for raw observations that is
entirely separate from insights_cache.json. Observations stored here
are used as input for pattern detection and evidence aggregation but
are never persisted to the final insight cache.

Classes:
    ObservationStore — Records, retrieves, and consumes raw observations.
"""

from __future__ import annotations

from collections import defaultdict

from agents.wist_discovery.insight_pipeline.schema import RawObservation


class ObservationStore:
    """Internal store for raw observations, separate from insights_cache.

    Raw observations are stored by category and can be queried for
    clusters of related observations sharing the same dimension_key
    and game_phase. The store enforces a maximum capacity using FIFO
    eviction per category when the total count is exceeded.

    Attributes:
        max_capacity: Maximum total number of observations across all
            categories before FIFO eviction occurs (default 10,000).
    """

    def __init__(self, max_capacity: int = 10000) -> None:
        """Initialize the observation store.

        Args:
            max_capacity: Maximum total observations before eviction.
        """
        self.max_capacity = max_capacity
        self._store: dict[str, list[RawObservation]] = defaultdict(list)
        self._total_count: int = 0

    @property
    def total_count(self) -> int:
        """Return the total number of observations across all categories."""
        return self._total_count

    def record(self, observation: RawObservation) -> None:
        """Store a raw observation for pattern detection.

        The observation is appended to the list for its category.
        If the store exceeds max_capacity after insertion, the oldest
        observation across all categories is evicted (FIFO).

        Args:
            observation: The raw observation to store.
        """
        self._store[observation.category].append(observation)
        self._total_count += 1

        if self._total_count > self.max_capacity:
            self._evict_oldest()

    def get_related(
        self,
        category: str,
        game_phase: str,
        min_count: int = 3,
    ) -> list[list[RawObservation]]:
        """Find clusters of related observations.

        Groups observations within the given category and game_phase
        by their dimension_key. Returns only clusters whose size meets
        or exceeds min_count.

        Args:
            category: The strategic category to filter by.
            game_phase: The game phase (early/mid/late) to filter by.
            min_count: Minimum cluster size to include in results.

        Returns:
            A list of observation clusters, where each cluster is a list
            of RawObservation instances sharing the same dimension_key.
        """
        observations = self._store.get(category, [])

        # Group by dimension_key, filtering by game_phase
        clusters: dict[str, list[RawObservation]] = defaultdict(list)
        for obs in observations:
            if obs.game_phase == game_phase:
                clusters[obs.dimension_key].append(obs)

        # Return only clusters meeting the minimum count threshold
        return [
            cluster
            for cluster in clusters.values()
            if len(cluster) >= min_count
        ]

    def consume(self, observations: list[RawObservation]) -> None:
        """Mark observations as consumed by removing them from the store.

        Consumed observations are removed so they are not double-counted
        in future aggregation cycles.

        Args:
            observations: List of observations to remove from the store.
        """
        # Build a set of ids for O(1) lookup
        to_remove = set(id(obs) for obs in observations)

        for category in list(self._store.keys()):
            original_len = len(self._store[category])
            self._store[category] = [
                obs
                for obs in self._store[category]
                if id(obs) not in to_remove
            ]
            removed_count = original_len - len(self._store[category])
            self._total_count -= removed_count

            # Clean up empty categories
            if not self._store[category]:
                del self._store[category]

    def _evict_oldest(self) -> None:
        """Evict the oldest observation across all categories (FIFO).

        Finds the category with the oldest first element (by timestamp)
        and removes it from the front of that category's list.
        """
        oldest_category: str | None = None
        oldest_timestamp: float = float("inf")

        for category, obs_list in self._store.items():
            if obs_list and obs_list[0].timestamp < oldest_timestamp:
                oldest_timestamp = obs_list[0].timestamp
                oldest_category = category

        if oldest_category is not None:
            self._store[oldest_category].pop(0)
            self._total_count -= 1

            # Clean up empty categories
            if not self._store[oldest_category]:
                del self._store[oldest_category]
