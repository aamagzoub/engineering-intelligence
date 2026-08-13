"""DuplicateMerger — Merges semantically equivalent strategies.

This module detects when a candidate strategic insight expresses the same
underlying advice as an existing insight in the cache, and merges them
into a single entry to prevent redundancy.

Semantic equivalence is determined by:
    1. Same category
    2. At least one overlapping tag
    3. >80% token overlap (Jaccard similarity of word tokens)

Merge rules on match:
    - confidence += candidate.confidence * 0.1 (evidence weight boost, cap at 1.0)
    - evidence_count += candidate.evidence_count
    - last_confirmed = candidate's episode

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
"""

from __future__ import annotations

from agents.wist_discovery.insight_pipeline.schema import StrategicInsight


class DuplicateMerger:
    """Merges semantically equivalent strategies."""

    TOKEN_OVERLAP_THRESHOLD = 0.8  # 80% Jaccard similarity

    def merge_if_duplicate(
        self,
        candidate: StrategicInsight,
        existing: list[StrategicInsight],
    ) -> StrategicInsight | None:
        """If candidate is semantically equivalent to an existing insight,
        merge and return the updated existing insight.

        Returns None if no duplicate found (candidate is new).

        Semantic equivalence: same category + overlapping tags +
        same core strategic advice (>80% token overlap).

        Merge rules:
        - Confidence += candidate evidence weight (cap at 1.0)
        - evidence_count += candidate.evidence_count
        - last_confirmed = candidate's episode

        Args:
            candidate: The new candidate insight to check for duplicates.
            existing: List of currently persisted insights to compare against.

        Returns:
            The updated existing insight if a duplicate was found, or None
            if the candidate is genuinely new.
        """
        for insight in existing:
            if self._is_semantically_equivalent(candidate, insight):
                # Merge: update the existing insight in place
                insight.evidence_count += candidate.evidence_count
                insight.confidence = min(
                    1.0, insight.confidence + candidate.confidence * 0.1
                )
                insight.last_confirmed = candidate.last_confirmed
                return insight

        return None

    def _is_semantically_equivalent(
        self,
        candidate: StrategicInsight,
        existing: StrategicInsight,
    ) -> bool:
        """Check if two insights express the same strategic advice.

        Criteria:
            1. Same category
            2. At least one overlapping tag
            3. Token overlap > 80% (Jaccard similarity)

        Args:
            candidate: The new candidate insight.
            existing: An existing insight to compare against.

        Returns:
            True if the insights are semantically equivalent.
        """
        # 1. Same category
        if candidate.category != existing.category:
            return False

        # 2. At least one overlapping tag
        candidate_tags = set(candidate.tags)
        existing_tags = set(existing.tags)
        if not candidate_tags.intersection(existing_tags):
            return False

        # 3. Token overlap > 80%
        similarity = self._token_similarity(candidate.strategy, existing.strategy)
        if similarity <= self.TOKEN_OVERLAP_THRESHOLD:
            return False

        return True

    def _token_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity of word tokens from two texts.

        Tokenizes by splitting on whitespace, then computes:
            |intersection| / |union| of token sets (case-insensitive)

        Args:
            text1: First text to compare.
            text2: Second text to compare.

        Returns:
            Jaccard similarity as a float in [0.0, 1.0].
            Returns 0.0 if both texts are empty.
        """
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 and not tokens2:
            return 0.0

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        return len(intersection) / len(union)
