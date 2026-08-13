"""Generality validation for candidate strategic insights.

Ensures insights don't reference specific game-state identifiers such as
literal trick numbers, specific player indices, named individual cards,
episode numbers, or hand identifiers.

Classes:
    GeneralityValidator — Validates candidate insight text for generality.
"""

from __future__ import annotations

import re


class GeneralityValidator:
    """Ensures insights don't reference specific game-state identifiers.

    A candidate insight must reference only abstract concepts (position
    category, card-strength tier, trick phase) rather than concrete
    game-state identifiers. This validator checks for patterns that
    indicate non-general content and rejects candidates accordingly.

    Class Attributes:
        TRICK_NUMBER_PATTERN: Matches literal trick numbers (e.g., "trick 3").
        PLAYER_INDEX_PATTERN: Matches specific player indices (e.g., "player 2").
        NAMED_CARD_PATTERN: Matches named individual cards (e.g., "Queen of clubs").
        EPISODE_PATTERN: Matches episode references (e.g., "episode 1000").
        HAND_ID_PATTERN: Matches hand identifiers (e.g., "hand #5", "hand 12").
    """

    # Patterns that indicate non-general content
    TRICK_NUMBER_PATTERN = re.compile(r'\btrick\s+\d+\b', re.IGNORECASE)
    PLAYER_INDEX_PATTERN = re.compile(r'\bplayer\s+\d+\b', re.IGNORECASE)
    NAMED_CARD_PATTERN = re.compile(
        r'\b(ace|king|queen|jack|[2-9]|10)\s+of\s+'
        r'(spades|hearts|clubs|diamonds)\b',
        re.IGNORECASE,
    )
    EPISODE_PATTERN = re.compile(r'\bepisode\s+\d+\b', re.IGNORECASE)
    HAND_ID_PATTERN = re.compile(r'\bhand\s+#?\d+\b', re.IGNORECASE)

    def validate(
        self,
        text: str,
        evidence_count: int,
        distinct_states: int,
    ) -> tuple[bool, str | None]:
        """Validate a candidate insight for generality.

        Checks the candidate text against multiple patterns that indicate
        non-general content, and verifies sufficient evidence support.

        Args:
            text: The candidate insight strategy text to validate.
            evidence_count: Number of distinct raw observations supporting
                the candidate.
            distinct_states: Number of distinct game states in which the
                pattern was observed.

        Returns:
            A tuple of (passes, rejection_reason_or_none).
            If the candidate passes all checks, returns (True, None).
            If rejected, returns (False, reason) where reason is one of:
                - "literal_trick_number"
                - "specific_player"
                - "named_card"
                - "single_episode"
                - "insufficient_pattern_support"
        """
        # Check 1: No literal trick number in text
        if self.TRICK_NUMBER_PATTERN.search(text):
            return (False, "literal_trick_number")

        # Check 2: No specific player index/seat in text
        if self.PLAYER_INDEX_PATTERN.search(text):
            return (False, "specific_player")

        # Check 3: No named individual card (rank + suit) in text
        if self.NAMED_CARD_PATTERN.search(text):
            return (False, "named_card")

        # Check 4: No hand ID or episode reference in text
        if self.EPISODE_PATTERN.search(text) or self.HAND_ID_PATTERN.search(text):
            return (False, "single_episode")

        # Check 5: Sufficient pattern support
        # Requirement 2.5: reject if supported by fewer than 3 distinct game states
        if evidence_count < 3 or distinct_states < 3:
            return (False, "insufficient_pattern_support")

        # All checks passed — insight is sufficiently general
        return (True, None)
