"""Schema definitions and validation for the insight pipeline.

Dataclasses:
    RawObservation — A single game-event record from one episode.
    RepeatedPattern — A cluster of related observations that recur across states.
    StrategicInsight — Final validated insight for persistence.

Constants:
    VALID_CATEGORIES — The 13 allowed strategic categories.
    REJECTION_REASONS — Reasons a candidate may be rejected.

Validation functions:
    validate_confidence, validate_category, validate_tags,
    validate_strategy_text, validate_why_text, validate_insight
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# Constants
# =============================================================================

VALID_CATEGORIES: frozenset[str] = frozenset([
    "leading",
    "following",
    "position",
    "card_preservation",
    "suit_management",
    "trump_management",
    "bidding",
    "defense",
    "partner_play",
    "risk",
    "information",
    "endgame",
    "surprising_pattern",
])

REJECTION_REASONS: list[str] = [
    "literal_trick_number",
    "specific_player",
    "named_card",
    "single_episode",
    "insufficient_pattern_support",
]


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class RawObservation:
    """A single game-event record from one episode."""

    category: str                    # Strategic dimension
    game_phase: str                  # early/mid/late
    dimension_key: str               # e.g., "pos_leading_high"
    reward_direction: str            # positive/negative
    state_context: dict[str, Any]    # Abstract state features
    episode: int                     # Source episode number
    snapshot_id: str                 # Which snapshot produced this
    timestamp: float                 # When recorded


@dataclass
class RepeatedPattern:
    """A cluster of related observations that recur across states."""

    category: str
    dimension_key: str
    observations: list[RawObservation] = field(default_factory=list)
    observation_count: int = 0
    distinct_states: int = 0         # Number of distinct game states
    distinct_snapshots: int = 0      # Number of training snapshots
    confidence: float = 0.0          # 0.0-1.0
    contradicting_count: int = 0     # Counter-evidence
    stage: str = "pattern"           # Pipeline stage label


@dataclass
class StrategicInsight:
    """Final validated insight for persistence."""

    strategy: str                    # Max 200 chars
    category: str                    # From VALID_CATEGORIES
    tags: list[str]                  # Max 5 tags, each max 30 chars
    confidence: float                # 0.0-1.0
    evidence_count: int              # Min 1
    why: str                         # Max 500 chars, quantitative
    first_seen: int                  # Episode number
    last_confirmed: int              # Episode number >= first_seen
    new: bool                        # True on first cycle


# =============================================================================
# Validation Functions
# =============================================================================

# Tag format: lowercase letters and underscores only.
_TAG_PATTERN = re.compile(r"^[a-z_]+$")


def validate_confidence(value: float) -> bool:
    """Check that confidence is within [0.0, 1.0].

    Args:
        value: Confidence score to validate.

    Returns:
        True if value is a finite number in [0.0, 1.0], False otherwise.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return 0.0 <= v <= 1.0


def validate_category(value: str) -> bool:
    """Check that value is one of the 13 valid categories.

    Args:
        value: Category string to validate.

    Returns:
        True if value is in VALID_CATEGORIES, False otherwise.
    """
    return value in VALID_CATEGORIES


def validate_tags(tags: list[str]) -> bool:
    """Validate tag list constraints.

    Rules:
        - Maximum 5 tags.
        - Each tag is lowercase letters and underscores only.
        - Each tag is at most 30 characters long.

    Args:
        tags: List of tag strings to validate.

    Returns:
        True if all constraints are met, False otherwise.
    """
    if not isinstance(tags, list):
        return False
    if len(tags) > 5:
        return False
    for tag in tags:
        if not isinstance(tag, str):
            return False
        if len(tag) == 0 or len(tag) > 30:
            return False
        if not _TAG_PATTERN.match(tag):
            return False
    return True


def validate_strategy_text(text: str) -> bool:
    """Check that strategy text is within 200 characters.

    Args:
        text: Strategy text to validate.

    Returns:
        True if text is a non-empty string of at most 200 characters.
    """
    if not isinstance(text, str):
        return False
    return 0 < len(text) <= 200


def validate_why_text(text: str) -> bool:
    """Check that why text is within 500 characters.

    Args:
        text: Explanation text to validate.

    Returns:
        True if text is a non-empty string of at most 500 characters.
    """
    if not isinstance(text, str):
        return False
    return 0 < len(text) <= 500


def validate_insight(insight: StrategicInsight) -> tuple[bool, list[str]]:
    """Full validation of a StrategicInsight.

    Checks all field constraints defined in Requirement 7.1:
        - strategy: non-empty string, max 200 chars
        - category: one of VALID_CATEGORIES
        - tags: max 5, each lowercase+underscores, max 30 chars
        - confidence: float 0.0–1.0
        - evidence_count: integer >= 1
        - why: non-empty string, max 500 chars
        - first_seen: integer >= 0
        - last_confirmed: integer >= first_seen
        - new: boolean

    Args:
        insight: The StrategicInsight instance to validate.

    Returns:
        Tuple of (is_valid, list_of_error_messages).
        If is_valid is True, the error list is empty.
    """
    errors: list[str] = []

    # strategy
    if not validate_strategy_text(insight.strategy):
        errors.append(
            "strategy must be a non-empty string of at most 200 characters"
        )

    # category
    if not validate_category(insight.category):
        errors.append(
            f"category '{insight.category}' is not in VALID_CATEGORIES"
        )

    # tags
    if not validate_tags(insight.tags):
        errors.append(
            "tags must be a list of at most 5 strings, each lowercase+underscores only, max 30 chars"
        )

    # confidence
    if not validate_confidence(insight.confidence):
        errors.append("confidence must be a float in [0.0, 1.0]")

    # evidence_count
    if not isinstance(insight.evidence_count, int) or insight.evidence_count < 1:
        errors.append("evidence_count must be an integer >= 1")

    # why
    if not validate_why_text(insight.why):
        errors.append("why must be a non-empty string of at most 500 characters")

    # first_seen
    if not isinstance(insight.first_seen, int) or insight.first_seen < 0:
        errors.append("first_seen must be an integer >= 0")

    # last_confirmed
    if not isinstance(insight.last_confirmed, int) or insight.last_confirmed < 0:
        errors.append("last_confirmed must be an integer >= 0")
    elif isinstance(insight.first_seen, int) and insight.last_confirmed < insight.first_seen:
        errors.append("last_confirmed must be >= first_seen")

    # new
    if not isinstance(insight.new, bool):
        errors.append("new must be a boolean")

    return (len(errors) == 0, errors)
