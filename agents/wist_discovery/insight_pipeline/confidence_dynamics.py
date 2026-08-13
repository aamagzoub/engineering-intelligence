"""Confidence dynamics for strategic insights.

Manages confidence growth, decay, and removal thresholds for the
insight pipeline. Also validates surprising pattern criteria per
Requirement 4.

Classes:
    ConfidenceDynamics — Growth/decay logic and surprising pattern validation.
"""

from __future__ import annotations

from agents.wist_discovery.insight_pipeline.schema import (
    RepeatedPattern,
    StrategicInsight,
)


class ConfidenceDynamics:
    """Manages confidence growth and decay over time.

    Class Constants:
        CONFIDENCE_THRESHOLD: Insights below this confidence are removed (0.3).
        SURPRISING_MIN_EVIDENCE: Minimum observation count for surprising patterns (10).
        SURPRISING_MIN_STATES: Minimum distinct game states for surprising patterns (3).
        SURPRISING_MIN_SNAPSHOTS: Minimum distinct training snapshots (2).
        SURPRISING_MIN_SUPPORT_RATIO: Minimum support:contradiction ratio (2.0).
    """

    CONFIDENCE_THRESHOLD: float = 0.3
    SURPRISING_MIN_EVIDENCE: int = 10
    SURPRISING_MIN_STATES: int = 3
    SURPRISING_MIN_SNAPSHOTS: int = 2
    SURPRISING_MIN_SUPPORT_RATIO: float = 2.0  # 2:1 supporting vs contradicting

    def apply_supporting_evidence(
        self,
        insight: StrategicInsight,
        total_observations: int,
        current_episode: int | None = None,
    ) -> None:
        """Increase confidence when a new supporting observation is found.

        Confidence increases by 1/total_observations, capped at 1.0.
        Also increments evidence_count and updates last_confirmed.

        Args:
            insight: The strategic insight to update.
            total_observations: Total number of observations recorded so far.
            current_episode: If provided, updates last_confirmed to this value.

        Raises:
            ValueError: If total_observations is less than 1.
        """
        if total_observations < 1:
            raise ValueError("total_observations must be at least 1")

        insight.confidence += 1.0 / total_observations
        insight.confidence = min(1.0, insight.confidence)
        insight.evidence_count += 1

        if current_episode is not None:
            insight.last_confirmed = current_episode

    def apply_contradicting_evidence(
        self, insight: StrategicInsight, total_observations: int
    ) -> None:
        """Recalculate confidence when contradicting evidence is found.

        Confidence is recalculated as supporting_count / total_observations,
        where supporting_count is the insight's current evidence_count.
        Result is clamped to [0.0, 1.0].

        Args:
            insight: The strategic insight to update.
            total_observations: Total number of observations recorded so far.

        Raises:
            ValueError: If total_observations is less than 1.
        """
        if total_observations < 1:
            raise ValueError("total_observations must be at least 1")

        insight.confidence = insight.evidence_count / total_observations
        insight.confidence = max(0.0, min(1.0, insight.confidence))

    def should_remove(self, insight: StrategicInsight) -> bool:
        """Determine if an insight should be removed from the cache.

        An insight is removed when its confidence drops below the
        configured threshold (default 0.3).

        Args:
            insight: The strategic insight to evaluate.

        Returns:
            True if confidence < CONFIDENCE_THRESHOLD, False otherwise.
        """
        return insight.confidence < self.CONFIDENCE_THRESHOLD

    def validate_surprising_pattern(self, pattern: RepeatedPattern) -> bool:
        """Validate a pattern against all 6 surprising pattern criteria.

        Criteria (from Requirement 4):
            a. Pattern repeats across at least 3 comparable game states
               (distinct_states >= SURPRISING_MIN_STATES).
            b. Pattern appears across at least 2 distinct training snapshots
               (distinct_snapshots >= SURPRISING_MIN_SNAPSHOTS).
            c. Pattern has evidence_count of at least 10
               (observation_count >= SURPRISING_MIN_EVIDENCE).
            d. Pattern is not obvious from rules — contradicts default
               expectation (contradicting_count > 0).
            e. Pattern is strategically reusable — references abstract
               conditions (distinct_states >= 3 ensures this).
            f. Support ratio is at least 2:1
               ((observation_count - contradicting_count) /
                max(1, contradicting_count) >= SURPRISING_MIN_SUPPORT_RATIO).

        Args:
            pattern: The repeated pattern to validate.

        Returns:
            True only if ALL 6 criteria pass, False otherwise.
        """
        # Criterion a: at least 3 distinct game states
        if pattern.distinct_states < self.SURPRISING_MIN_STATES:
            return False

        # Criterion b: at least 2 distinct training snapshots
        if pattern.distinct_snapshots < self.SURPRISING_MIN_SNAPSHOTS:
            return False

        # Criterion c: at least 10 supporting observations
        if pattern.observation_count < self.SURPRISING_MIN_EVIDENCE:
            return False

        # Criterion d: not obvious from rules (has contradicting evidence,
        # indicating it goes against the default expectation)
        if pattern.contradicting_count <= 0:
            return False

        # Criterion e: strategically reusable (distinct_states >= 3 ensures
        # the pattern references abstract conditions, not specific cards/episodes)
        # Already covered by criterion a check above.

        # Criterion f: support ratio >= 2:1
        supporting_count = pattern.observation_count - pattern.contradicting_count
        ratio = supporting_count / max(1, pattern.contradicting_count)
        if ratio < self.SURPRISING_MIN_SUPPORT_RATIO:
            return False

        return True
