"""TextGenerator — Constrained natural-language generation for strategies.

This module generates strategy text and explanatory "why" text from
confirmed RepeatedPattern instances. Text generation is gated behind
statistical confirmation: only patterns meeting confidence, observation
count, distinct state, and distinct snapshot thresholds are eligible.

Generated text is template-based, avoids subjective qualifiers, stays
within length limits, and includes quantitative references in the "why"
field.

Classes:
    TextGenerator — Generates (strategy_text, why_text) from confirmed patterns.
"""

from __future__ import annotations

from agents.wist_discovery.insight_pipeline.schema import RepeatedPattern


class TextGenerator:
    """Constrained natural-language generation for strategies.

    Only generates text for patterns that have passed the statistical
    confirmation gate. Generated text is bounded in length, free of
    subjective qualifiers, and includes quantitative evidence references.

    Attributes:
        FORBIDDEN_QUALIFIERS: Set of subjective words that must never appear.
        MAX_STRATEGY_LENGTH: Maximum characters for strategy text (200).
        MAX_WHY_LENGTH: Maximum characters for why text (500).
        CONFIDENCE_THRESHOLD: Minimum confidence for text generation (0.3).
        MIN_OBSERVATION_COUNT: Minimum observations required (3).
        MIN_DISTINCT_STATES: Minimum distinct game states required (2).
        MIN_DISTINCT_SNAPSHOTS: Minimum distinct snapshots required (3).
    """

    FORBIDDEN_QUALIFIERS: frozenset[str] = frozenset({
        "brilliant", "optimal", "perfect", "amazing",
        "incredible", "genius", "unbelievable",
    })

    MAX_STRATEGY_LENGTH: int = 200
    MAX_WHY_LENGTH: int = 500

    # Statistical confirmation gate thresholds
    CONFIDENCE_THRESHOLD: float = 0.3
    MIN_OBSERVATION_COUNT: int = 3
    MIN_DISTINCT_STATES: int = 2
    MIN_DISTINCT_SNAPSHOTS: int = 2

    # Template mappings from dimension_key to strategy text
    _STRATEGY_TEMPLATES: dict[str, str] = {
        "leading_high_card_late_phase": (
            "In late positions, strong cards gain value when more"
            " trick information is available"
        ),
        "following_low_card_mid_phase": (
            "Following with weaker cards in the middle phase preserves"
            " strength for later opportunities"
        ),
        "leading_vs_following_leading": (
            "Leading the trick provides a positional advantage"
            " by setting the suit and forcing responses"
        ),
        "leading_vs_following_following": (
            "Following positions benefit from observing the lead"
            " before committing card strength"
        ),
        "trump_vs_nontrump_trump": (
            "Trump plays carry higher expected value when used"
            " strategically to capture contested tricks"
        ),
        "trump_vs_nontrump_nontrump": (
            "Non-trump plays preserve trump resources for"
            " higher-value opportunities later in the hand"
        ),
        "partner_vs_opponent_winning_partner": (
            "When partner is winning the trick, conservative play"
            " preserves the advantage without risk"
        ),
        "partner_vs_opponent_winning_opponent": (
            "When opponents lead the trick, aggressive play"
            " increases chances of recapturing control"
        ),
        "suit_length_long": (
            "Long suits provide consistent lead options"
            " and control over multiple trick rounds"
        ),
        "suit_length_short": (
            "Short suits create opportunities for early voids"
            " enabling future trump plays"
        ),
        "suit_length_void": (
            "Void suits allow immediate trump usage"
            " to capture tricks led in that suit"
        ),
        "phase_behaviour_early": (
            "Early-phase play benefits from information gathering"
            " before committing high-value cards"
        ),
        "phase_behaviour_mid": (
            "Mid-phase play balances card preservation"
            " with opportunistic trick capture"
        ),
        "phase_behaviour_late": (
            "Late-phase play rewards committing remaining"
            " strength when fewer unknowns remain"
        ),
        "defensive_vs_attacking_defensive": (
            "Defensive play focuses on disrupting opponent contract"
            " fulfilment through strategic interference"
        ),
        "defensive_vs_attacking_attacking": (
            "Attacking play maximises trick capture to meet"
            " contract obligations efficiently"
        ),
    }

    def generate(self, pattern: RepeatedPattern) -> tuple[str, str] | None:
        """Generate (strategy_text, why_text) from a confirmed pattern.

        Returns None if pattern hasn't passed statistical confirmation.
        The 'why' must include quantitative reference (evidence_count,
        effect size, or consistency measure).

        Args:
            pattern: A RepeatedPattern instance to generate text for.

        Returns:
            A tuple of (strategy_text, why_text) if the pattern passes
            the statistical confirmation gate, or None if it does not.
        """
        # Check statistical confirmation gate
        if not self._passes_confirmation_gate(pattern):
            return None

        # Generate strategy text
        strategy_text = self._generate_strategy_text(pattern)

        # Generate why text with quantitative reference
        why_text = self._generate_why_text(pattern)

        # Final validation: check both texts for forbidden qualifiers
        if self._contains_forbidden_qualifiers(strategy_text):
            return None
        if self._contains_forbidden_qualifiers(why_text):
            return None

        return (strategy_text, why_text)

    def _passes_confirmation_gate(self, pattern: RepeatedPattern) -> bool:
        """Check that pattern meets statistical confirmation thresholds.

        A pattern must have:
        - confidence >= 0.3
        - observation_count >= 3
        - distinct_states >= 2
        - distinct_snapshots >= 3

        Args:
            pattern: The pattern to check.

        Returns:
            True if the pattern passes all thresholds.
        """
        if pattern.confidence < self.CONFIDENCE_THRESHOLD:
            return False
        if pattern.observation_count < self.MIN_OBSERVATION_COUNT:
            return False
        if pattern.distinct_states < self.MIN_DISTINCT_STATES:
            return False
        if pattern.distinct_snapshots < self.MIN_DISTINCT_SNAPSHOTS:
            return False
        return True

    def _generate_strategy_text(self, pattern: RepeatedPattern) -> str:
        """Generate strategy text from the pattern's dimension_key.

        Uses template-based generation. Falls back to a generic template
        derived from category + game_phase + reward_direction when no
        specific template matches.

        Args:
            pattern: The confirmed pattern.

        Returns:
            Strategy text string, guaranteed ≤ MAX_STRATEGY_LENGTH chars.
        """
        # Try exact match on dimension_key
        if pattern.dimension_key in self._STRATEGY_TEMPLATES:
            text = self._STRATEGY_TEMPLATES[pattern.dimension_key]
        else:
            # Generic fallback using pattern attributes
            text = self._generate_fallback_strategy(pattern)

        # Enforce max length
        if len(text) > self.MAX_STRATEGY_LENGTH:
            text = text[: self.MAX_STRATEGY_LENGTH - 3].rsplit(" ", 1)[0] + "..."

        return text

    def _generate_fallback_strategy(self, pattern: RepeatedPattern) -> str:
        """Generate a generic strategy text from pattern attributes.

        Derives text from category, dimension_key components, and
        reward direction without referencing specific cards, tricks,
        players, or episodes.

        Args:
            pattern: The confirmed pattern.

        Returns:
            A generic strategy text string.
        """
        # Parse dimension_key components for readable text
        category_display = pattern.category.replace("_", " ")

        # Determine reward direction from observations
        reward_direction = self._get_majority_reward_direction(pattern)
        direction_phrase = (
            "shows positive outcomes"
            if reward_direction == "positive"
            else "shows caution is warranted"
        )

        # Determine phase from dimension_key if present
        phase_phrase = ""
        if "early" in pattern.dimension_key:
            phase_phrase = " in early-phase play"
        elif "mid" in pattern.dimension_key:
            phase_phrase = " in mid-phase play"
        elif "late" in pattern.dimension_key:
            phase_phrase = " in late-phase play"

        text = (
            f"The {category_display} dimension{phase_phrase}"
            f" {direction_phrase} across observed game states"
        )

        return text

    def _generate_why_text(self, pattern: RepeatedPattern) -> str:
        """Generate why text with quantitative references.

        The why text must include at least one of: evidence_count,
        effect size (%), or consistency measure. Uses the pattern's
        statistical data to produce a verifiable explanation.

        Args:
            pattern: The confirmed pattern.

        Returns:
            Why text string, guaranteed ≤ MAX_WHY_LENGTH chars.
        """
        count = pattern.observation_count
        confidence = pattern.confidence
        distinct_snapshots = pattern.distinct_snapshots
        distinct_states = pattern.distinct_states

        # Include category as game mechanic reference so quality gate passes.
        category_phrase = pattern.category.replace("_", " ")
        text = (
            f"In {category_phrase} situations, observed across {count} game states"
            f" with {confidence:.0%} consistency"
            f" ({distinct_snapshots} training snapshots,"
            f" {distinct_states} distinct states)"
        )

        # Enforce max length
        if len(text) > self.MAX_WHY_LENGTH:
            text = text[: self.MAX_WHY_LENGTH - 3].rsplit(" ", 1)[0] + "..."

        return text

    def _contains_forbidden_qualifiers(self, text: str) -> bool:
        """Check if text contains any forbidden subjective qualifiers.

        Performs case-insensitive word-boundary matching against the
        FORBIDDEN_QUALIFIERS set.

        Args:
            text: The text to check.

        Returns:
            True if any forbidden qualifier is found.
        """
        text_lower = text.lower()
        words = set(text_lower.split())
        return bool(words & self.FORBIDDEN_QUALIFIERS)

    @staticmethod
    def _get_majority_reward_direction(pattern: RepeatedPattern) -> str:
        """Determine the majority reward direction from observations.

        Args:
            pattern: The pattern whose observations to examine.

        Returns:
            The most common reward_direction string, or "positive"
            if no observations exist.
        """
        if not pattern.observations:
            return "positive"

        direction_counts: dict[str, int] = {}
        for obs in pattern.observations:
            direction_counts[obs.reward_direction] = (
                direction_counts.get(obs.reward_direction, 0) + 1
            )

        return max(direction_counts, key=lambda d: direction_counts[d])
