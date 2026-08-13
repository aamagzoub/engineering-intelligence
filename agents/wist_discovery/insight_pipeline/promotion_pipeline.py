"""Promotion pipeline and quality gate for strategic insights.

Implements the three-stage pipeline (raw observation → repeated pattern →
strategic insight) with a quality gate that validates candidates before
promotion to insights_cache.json.

Classes:
    PromotionPipeline — Promotes observations to patterns and patterns to insights.
    QualityGate — Validates candidates against 7 quality checks before persistence.
"""

from __future__ import annotations

from agents.wist_discovery.insight_pipeline.schema import (
    RawObservation,
    RepeatedPattern,
    StrategicInsight,
    VALID_CATEGORIES,
)
from agents.wist_discovery.insight_pipeline.observation_store import ObservationStore
from agents.wist_discovery.insight_pipeline.generality_validator import GeneralityValidator


# Game mechanic keywords used to validate that "why" references a game mechanic.
_GAME_MECHANIC_KEYWORDS: frozenset[str] = frozenset([
    "trump", "lead", "follow", "position", "trick", "bid",
    "suit", "card", "hand", "void", "partner", "opponent",
    "phase", "early", "mid", "late", "rank", "high", "low",
    "win", "lose", "score", "defense", "attack", "endgame",
])

# Keywords that indicate the "why" references learned evidence.
_EVIDENCE_KEYWORDS: frozenset[str] = frozenset([
    "q-value", "observed", "states", "%", "episodes", "evidence",
    "win-rate", "ratio", "effect", "count", "times", "average",
    "mean", "across", "consistently",
])

# Game phases used to iterate during observation-to-pattern promotion.
_GAME_PHASES: list[str] = ["early", "mid", "late"]


def _token_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity of word tokens between two texts.

    Tokenizes by splitting on whitespace and computes the ratio of
    intersection size to union size of the resulting token sets.

    Args:
        text1: First text string.
        text2: Second text string.

    Returns:
        Float between 0.0 and 1.0 representing Jaccard similarity.
        Returns 0.0 if both texts are empty.
    """
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    if not tokens1 and not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    if not union:
        return 0.0

    return len(intersection) / len(union)


class QualityGate:
    """Validates candidates before promotion to insights_cache.

    Implements 7 validation checks that a candidate StrategicInsight must
    pass before being persisted to insights_cache.json. Each check maps to
    requirements from the Strategy Quality Gate (Requirement 11).

    Class Attributes:
        TEXT_SIMILARITY_THRESHOLD: Maximum allowed token overlap (80%)
            before considering two insights duplicates.
    """

    TEXT_SIMILARITY_THRESHOLD: float = 0.8

    def validate(
        self,
        candidate: StrategicInsight,
        existing: list[StrategicInsight],
        pattern: RepeatedPattern | None = None,
    ) -> tuple[bool, str | None]:
        """Validate a candidate insight against all 7 quality checks.

        Args:
            candidate: The candidate StrategicInsight to validate.
            existing: List of existing insights already in the cache.
            pattern: The source RepeatedPattern (used for distinct_states check).

        Returns:
            A tuple of (passes, rejection_reason_or_none).
            If all checks pass, returns (True, None).
            If any check fails, returns (False, reason_string).
        """
        # Check 1: Evidence from ≥3 distinct episodes (Req 11.1)
        if candidate.evidence_count < 3:
            return (False, "insufficient_episodes: evidence_count < 3")

        # Check 2: Applies to ≥2 distinct game states (Req 11.2)
        distinct_states = pattern.distinct_states if pattern else 0
        if distinct_states < 2:
            return (False, "insufficient_states: distinct_states < 2")

        # Check 3: References recurring condition, not fixed card combo (Req 11.3)
        # Use GeneralityValidator's named card pattern to detect fixed combos
        if GeneralityValidator.NAMED_CARD_PATTERN.search(candidate.strategy):
            return (False, "fixed_card_combination: strategy references named cards")

        # Check 4: Has causal reason — non-empty why referencing game mechanic (Req 11.4)
        if not candidate.why or not candidate.why.strip():
            return (False, "missing_causal_reason: why field is empty")

        why_lower = candidate.why.lower()
        has_mechanic = any(kw in why_lower for kw in _GAME_MECHANIC_KEYWORDS)
        if not has_mechanic:
            return (False, "missing_causal_reason: why does not reference a game mechanic")

        # Check 5: No >80% text similarity with existing insight in same category (Req 11.5)
        for existing_insight in existing:
            if existing_insight.category != candidate.category:
                continue
            similarity = _token_similarity(candidate.strategy, existing_insight.strategy)
            if similarity > self.TEXT_SIMILARITY_THRESHOLD:
                return (
                    False,
                    f"duplicate_strategy: {similarity:.2f} token overlap with existing insight",
                )

        # Check 6: Surprising patterns must contradict default expectation (Req 11.6)
        if candidate.category == "surprising_pattern":
            # A surprising pattern must have contradiction evidence
            if pattern and pattern.contradicting_count == 0:
                return (
                    False,
                    "surprising_not_contradicting: pattern does not contradict default expectation",
                )

        # Check 7: Reason references learned evidence (Req 11.7)
        has_evidence_ref = any(kw in why_lower for kw in _EVIDENCE_KEYWORDS)
        # Also check for numeric references (percentages, counts, etc.)
        has_numeric = any(ch.isdigit() for ch in candidate.why)
        if not has_evidence_ref and not has_numeric:
            return (
                False,
                "missing_evidence_reference: why does not reference learned evidence",
            )

        return (True, None)


class PromotionPipeline:
    """Three-stage pipeline: observation → pattern → insight.

    Manages the promotion of raw observations to repeated patterns
    and repeated patterns to strategic insights, applying quality
    gates and generality validation at each transition.
    """

    def promote_observations_to_patterns(
        self, store: ObservationStore
    ) -> list[RepeatedPattern]:
        """Promote observations to patterns when threshold is met.

        Iterates all categories and game phases in the observation store,
        finding clusters of ≥3 related observations. Only promotes clusters
        that span ≥2 distinct game states.

        Args:
            store: The ObservationStore containing raw observations.

        Returns:
            List of RepeatedPattern objects representing promoted clusters.
        """
        patterns: list[RepeatedPattern] = []

        for category in VALID_CATEGORIES:
            for game_phase in _GAME_PHASES:
                clusters = store.get_related(
                    category=category,
                    game_phase=game_phase,
                    min_count=3,
                )

                for cluster in clusters:
                    # Count distinct states using string representation of state_context
                    distinct_states = len(
                        set(str(obs.state_context) for obs in cluster)
                    )

                    # Only promote if observed across ≥2 distinct game states
                    if distinct_states < 2:
                        continue

                    # Count distinct snapshots
                    distinct_snapshots = len(
                        set(obs.snapshot_id for obs in cluster)
                    )

                    observation_count = len(cluster)

                    # Count contradicting observations (negative reward direction)
                    contradicting_count = sum(
                        1 for obs in cluster if obs.reward_direction == "negative"
                    )
                    supporting_count = observation_count - contradicting_count

                    # Compute confidence = supporting / (supporting + contradicting)
                    confidence = supporting_count / (supporting_count + contradicting_count)

                    # Determine the dimension_key from the cluster
                    dimension_key = cluster[0].dimension_key

                    pattern = RepeatedPattern(
                        category=category,
                        dimension_key=dimension_key,
                        observations=cluster,
                        observation_count=observation_count,
                        distinct_states=distinct_states,
                        distinct_snapshots=distinct_snapshots,
                        confidence=confidence,
                        contradicting_count=contradicting_count,
                        stage="pattern",
                    )
                    patterns.append(pattern)

        return patterns

    def promote_patterns_to_insights(
        self,
        patterns: list[RepeatedPattern],
        quality_gate: QualityGate,
        generality_validator: GeneralityValidator,
        existing_insights: list[StrategicInsight],
    ) -> tuple[list[StrategicInsight], list[dict]]:
        """Promote patterns to strategic insights after quality validation.

        For each pattern, creates a candidate StrategicInsight with placeholder
        text, runs generality validation and quality gate checks. Only patterns
        passing all checks are promoted.

        Args:
            patterns: List of RepeatedPattern candidates for promotion.
            quality_gate: QualityGate instance for validation.
            generality_validator: GeneralityValidator for text checks.
            existing_insights: Current insights in the cache (for dedup checks).

        Returns:
            A tuple of (promoted_insights, rejected_list) where:
            - promoted_insights: list of StrategicInsight that passed all checks.
            - rejected_list: list of dicts with 'pattern', 'candidate', and 'reason'.
        """
        promoted: list[StrategicInsight] = []
        rejected: list[dict] = []

        for pattern in patterns:
            # Create candidate insight with placeholder text
            # (TextGenerator will fill in real text later)
            strategy_text = (
                f"In {pattern.observations[0].game_phase} phase, "
                f"{pattern.dimension_key} pattern observed across "
                f"{pattern.distinct_states} states"
            )

            # Determine episode info from observations
            episodes = [obs.episode for obs in pattern.observations]
            first_episode = min(episodes) if episodes else 0
            last_episode = max(episodes) if episodes else 0

            why_text = (
                f"Observed across {pattern.observation_count} episodes in "
                f"{pattern.distinct_states} distinct game states with "
                f"{pattern.confidence:.0%} consistency"
            )

            candidate = StrategicInsight(
                strategy=strategy_text,
                category=pattern.category,
                tags=[pattern.category],
                confidence=pattern.confidence,
                evidence_count=pattern.observation_count,
                why=why_text,
                first_seen=first_episode,
                last_confirmed=last_episode,
                new=True,
            )

            # Run generality validation
            gen_passes, gen_reason = generality_validator.validate(
                text=candidate.strategy,
                evidence_count=candidate.evidence_count,
                distinct_states=pattern.distinct_states,
            )

            if not gen_passes:
                rejected.append({
                    "pattern": pattern,
                    "candidate": candidate,
                    "reason": f"generality_validation: {gen_reason}",
                })
                continue

            # Run quality gate validation
            qg_passes, qg_reason = quality_gate.validate(
                candidate=candidate,
                existing=existing_insights,
                pattern=pattern,
            )

            if not qg_passes:
                rejected.append({
                    "pattern": pattern,
                    "candidate": candidate,
                    "reason": f"quality_gate: {qg_reason}",
                })
                continue

            promoted.append(candidate)

        return (promoted, rejected)
