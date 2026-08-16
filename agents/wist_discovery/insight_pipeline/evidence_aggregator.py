"""EvidenceAggregator — Aggregates observations into pattern candidates.

This module implements the intermediate evidence layer between raw Q-value
observations and final strategic insights. It groups raw observations by
strategic dimension, computes confidence scores, and promotes patterns
that meet defined thresholds.

Classes:
    EvidenceAggregator — Groups observations, computes confidence, persists evidence.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from agents.wist_discovery.insight_pipeline.schema import (
    RawObservation,
    RepeatedPattern,
)


class EvidenceAggregator:
    """Aggregates observations into pattern candidates.

    Groups raw observations by their dimension_key, computes confidence
    as the ratio of supporting observations to total (supporting +
    contradicting), and returns patterns that meet promotion thresholds.

    Patterns below threshold are not returned but remain available in
    the raw observations for future aggregation cycles.

    Attributes:
        CONFIDENCE_THRESHOLD: Minimum confidence (0.3) for promotion.
        MIN_SNAPSHOTS: Minimum distinct snapshots (3) for promotion.
    """

    CONFIDENCE_THRESHOLD: float = 0.3
    MIN_SNAPSHOTS: int = 2

    def aggregate(
        self,
        observations: list[RawObservation],
        snapshots: dict[str, Any],
    ) -> list[RepeatedPattern]:
        """Group observations by dimension, compute confidence, return promoted patterns.

        Groups all observations by their dimension_key. For each group,
        computes observation count, distinct game states, distinct snapshots,
        contradicting observations, and confidence. Only patterns meeting
        both CONFIDENCE_THRESHOLD and MIN_SNAPSHOTS are returned.

        Args:
            observations: List of raw observations to aggregate.
            snapshots: Dictionary of strategy snapshots (keyed by snapshot id).

        Returns:
            List of RepeatedPattern objects meeting promotion thresholds.
        """
        if not observations:
            return []

        # Group observations by dimension_key
        groups: dict[str, list[RawObservation]] = defaultdict(list)
        for obs in observations:
            groups[obs.dimension_key].append(obs)

        promoted: list[RepeatedPattern] = []

        for dimension_key, obs_list in groups.items():
            # Count total observations
            observation_count = len(obs_list)

            # Count distinct snapshots (unique snapshot_id values)
            distinct_snapshots = len({obs.snapshot_id for obs in obs_list})

            # Count distinct game states using state_context as discriminator.
            # For snapshot-derived observations, each snapshot represents a
            # distinct state of the agent (Q-values evolve with training).
            # Use max of state_context diversity and snapshot diversity.
            state_diversity = len(
                {
                    self._state_context_key(obs.state_context)
                    for obs in obs_list
                }
            )
            distinct_states = max(state_diversity, distinct_snapshots)

            # Determine the majority reward direction (supporting)
            direction_counts: dict[str, int] = defaultdict(int)
            for obs in obs_list:
                direction_counts[obs.reward_direction] += 1

            # The majority direction is "supporting"; others are contradicting
            majority_direction = max(
                direction_counts, key=lambda d: direction_counts[d]
            )
            supporting_count = direction_counts[majority_direction]
            contradicting_count = observation_count - supporting_count

            # Compute confidence = supporting / (supporting + contradicting)
            confidence = supporting_count / (supporting_count + contradicting_count)

            # Only promote if meeting both thresholds
            if (
                confidence >= self.CONFIDENCE_THRESHOLD
                and distinct_snapshots >= self.MIN_SNAPSHOTS
            ):
                # Derive category from the first observation in the group
                category = obs_list[0].category

                pattern = RepeatedPattern(
                    category=category,
                    dimension_key=dimension_key,
                    observations=obs_list,
                    observation_count=observation_count,
                    distinct_states=distinct_states,
                    distinct_snapshots=distinct_snapshots,
                    confidence=confidence,
                    contradicting_count=contradicting_count,
                    stage="pattern",
                )
                promoted.append(pattern)

        return promoted

    def persist_evidence(
        self,
        patterns: list[RepeatedPattern],
        raw_observations: list[RawObservation],
        data_dir: Path,
    ) -> None:
        """Write current evidence state to strategy_evidence.json.

        Persists both promoted patterns and raw observations along with
        metadata summarizing the current evidence state.

        Args:
            patterns: List of RepeatedPattern objects (promoted).
            raw_observations: List of all raw observations (including
                below-threshold ones for future aggregation).
            data_dir: Directory path where strategy_evidence.json is written.
        """
        # Serialize patterns
        serialized_patterns: list[dict[str, Any]] = []
        for pattern in patterns:
            snapshot_ids = sorted({obs.snapshot_id for obs in pattern.observations})

            # Determine majority reward direction
            direction_counts: dict[str, int] = defaultdict(int)
            for obs in pattern.observations:
                direction_counts[obs.reward_direction] += 1
            reward_direction = max(
                direction_counts, key=lambda d: direction_counts[d]
            )

            episodes = [obs.episode for obs in pattern.observations]

            serialized_patterns.append(
                {
                    "dimension_key": pattern.dimension_key,
                    "category": pattern.category,
                    "stage": pattern.stage,
                    "observation_count": pattern.observation_count,
                    "distinct_states": pattern.distinct_states,
                    "distinct_snapshots": pattern.distinct_snapshots,
                    "confidence": pattern.confidence,
                    "contradicting_count": pattern.contradicting_count,
                    "reward_direction": reward_direction,
                    "first_observed_episode": min(episodes) if episodes else 0,
                    "last_observed_episode": max(episodes) if episodes else 0,
                    "supporting_snapshot_ids": snapshot_ids,
                }
            )

        # Serialize raw observations
        serialized_observations: list[dict[str, Any]] = []
        for obs in raw_observations:
            serialized_observations.append(
                {
                    "category": obs.category,
                    "game_phase": obs.game_phase,
                    "dimension_key": obs.dimension_key,
                    "reward_direction": obs.reward_direction,
                    "state_context": obs.state_context,
                    "episode": obs.episode,
                    "snapshot_id": obs.snapshot_id,
                }
            )

        # Compute metadata
        all_episodes = [obs.episode for obs in raw_observations]
        last_processed_episode = max(all_episodes) if all_episodes else 0

        metadata = {
            "last_processed_episode": last_processed_episode,
            "total_observations_recorded": len(raw_observations),
            "total_patterns_detected": len(serialized_patterns),
            "total_promotions": len(
                [p for p in patterns if p.confidence >= self.CONFIDENCE_THRESHOLD
                 and p.distinct_snapshots >= self.MIN_SNAPSHOTS]
            ),
        }

        evidence_data = {
            "patterns": serialized_patterns,
            "raw_observations": serialized_observations,
            "metadata": metadata,
        }

        # Write to strategy_evidence.json
        output_path = Path(data_dir) / "strategy_evidence.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(evidence_data, f, indent=2)

    @staticmethod
    def _state_context_key(state_context: dict[str, Any]) -> str:
        """Create a hashable key from a state_context dictionary.

        Produces a deterministic string representation by sorting keys
        to ensure consistent hashing regardless of dict insertion order.

        Args:
            state_context: Dictionary of abstract state features.

        Returns:
            A string key suitable for set membership testing.
        """
        return json.dumps(state_context, sort_keys=True)
