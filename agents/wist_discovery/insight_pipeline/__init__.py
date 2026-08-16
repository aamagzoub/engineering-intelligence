"""Insight Pipeline — Multi-stage strategic insight generation.

Public API:
    run_insight_cycle(agent, current_episode, data_dir) → list[StrategicInsight]

Pipeline order:
    Q-table → snapshot → observations → evidence aggregation
    → promotion → text generation → dedup/merge → confidence dynamics → persist

Requirements: 9.6, 7.2, 7.3, 7.4, 12.1–12.5
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from agents.wist_discovery.insight_pipeline.schema import (
    RawObservation,
    RepeatedPattern,
    StrategicInsight,
    VALID_CATEGORIES,
    REJECTION_REASONS,
    validate_insight,
)
from agents.wist_discovery.insight_pipeline.observation_store import ObservationStore
from agents.wist_discovery.insight_pipeline.snapshot_extender import SnapshotExtender
from agents.wist_discovery.insight_pipeline.evidence_aggregator import EvidenceAggregator
from agents.wist_discovery.insight_pipeline.promotion_pipeline import (
    PromotionPipeline,
    QualityGate,
)
from agents.wist_discovery.insight_pipeline.generality_validator import GeneralityValidator
from agents.wist_discovery.insight_pipeline.text_generator import TextGenerator
from agents.wist_discovery.insight_pipeline.duplicate_merger import DuplicateMerger
from agents.wist_discovery.insight_pipeline.confidence_dynamics import ConfidenceDynamics
from agents.wist_discovery.insight_pipeline.migration import migrate_legacy_insights

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Atomic File I/O Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _atomic_write_json(data: Any, path: Path) -> None:
    """Write JSON data to a file atomically using temp file + os.replace().

    Writes to a temporary file in the same directory, then uses
    os.replace() to atomically swap it into place. This ensures
    that readers never see a partially-written file.

    Args:
        data: JSON-serializable data to write.
        path: Target file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (ensures same filesystem for rename)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix=path.stem + "_",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, str(path))
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _load_json_file(path: Path) -> Any:
    """Load a JSON file, returning None if missing or corrupt.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON data, or None if file is missing/corrupt.
    """
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read %s (%s), starting fresh.", path, e)
        return None


def _load_insights_cache(path: Path) -> list[dict[str, Any]]:
    """Load insights_cache.json, handling corrupt/missing files gracefully.

    Args:
        path: Path to insights_cache.json.

    Returns:
        List of raw dict entries, or empty list if file is missing/corrupt.
    """
    data = _load_json_file(path)
    if data is None:
        return []
    if not isinstance(data, list):
        logger.warning(
            "insights_cache.json is not a list, starting fresh: %s", path
        )
        return []
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Serialization
# ─────────────────────────────────────────────────────────────────────────────


def _serialize_insight(insight: StrategicInsight) -> dict[str, Any]:
    """Serialize a StrategicInsight to a JSON-compatible dict.

    Args:
        insight: The insight to serialize.

    Returns:
        A dictionary suitable for JSON serialization.
    """
    return {
        "strategy": insight.strategy,
        "category": insight.category,
        "tags": insight.tags,
        "confidence": insight.confidence,
        "evidence_count": insight.evidence_count,
        "why": insight.why,
        "first_seen": insight.first_seen,
        "last_confirmed": insight.last_confirmed,
        "new": insight.new,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Observation Generation from Snapshot
# ─────────────────────────────────────────────────────────────────────────────

# Dimension → category mapping
_DIMENSION_CATEGORY_MAP: dict[str, str] = {
    "leading_vs_following": "position",
    "phase_behaviour": "position",
    "card_strength_by_role": "card_preservation",
    "trump_vs_nontrump": "trump_management",
    "partner_vs_opponent_winning": "partner_play",
    "suit_length": "suit_management",
    "information_available": "information",
    "bid_strength_reliability": "bidding",
    "defensive_vs_attacking": "defense",
}

# Sub-key → game phase mapping
_SUBKEY_PHASE_MAP: dict[str, str] = {
    "early": "early",
    "mid": "mid",
    "late": "late",
    "leading": "mid",
    "following": "mid",
    "leading_high": "mid",
    "leading_low": "mid",
    "following_high": "mid",
    "following_low": "mid",
    "trump": "mid",
    "nontrump": "mid",
    "partner_winning": "mid",
    "opponent_winning": "mid",
    "long": "early",
    "short": "mid",
    "defensive": "mid",
    "attacking": "mid",
}

# Threshold for significant Q-value to create an observation
_MEAN_Q_THRESHOLD: float = 0.1


def _generate_observations_from_snapshot(
    snapshot: dict[str, Any],
    current_episode: int,
    snapshot_id: str,
) -> list[RawObservation]:
    """Generate raw observations from an extended snapshot's dimensions.

    For each extended dimension in the snapshot, if the value is not None
    and count >= 5:
      - If mean_q > 0.5: create a positive observation
      - If mean_q < -0.5: create a negative observation
      - Otherwise: skip (not significant enough)

    Args:
        snapshot: The extended snapshot dict from SnapshotExtender.
        current_episode: Current training episode number.
        snapshot_id: Identifier for this snapshot.

    Returns:
        List of RawObservation objects extracted from the snapshot.
    """
    observations: list[RawObservation] = []
    timestamp = time.time()

    for dimension_key, dimension_value in snapshot.items():
        # Skip existing fields (pos_prefs, suit_prefs, etc.)
        if dimension_key in ("pos_prefs", "suit_prefs", "bid_prefs", "rank_prefs"):
            continue

        category = _DIMENSION_CATEGORY_MAP.get(dimension_key)
        if category is None:
            continue

        if not isinstance(dimension_value, dict):
            continue

        # Special handling for bid_strength_reliability
        if dimension_key == "bid_strength_reliability":
            bid_level = dimension_value.get("bid_level")
            if (
                bid_level is not None
                and isinstance(bid_level, dict)
                and bid_level.get("count", 0) >= 5
            ):
                mean_q = bid_level.get("mean_q", 0.0)
                if mean_q > _MEAN_Q_THRESHOLD:
                    reward_direction = "positive"
                elif mean_q < -_MEAN_Q_THRESHOLD:
                    reward_direction = "negative"
                else:
                    continue  # Not significant
                obs = RawObservation(
                    category=category,
                    game_phase="mid",
                    dimension_key=f"{dimension_key}_bid_level",
                    reward_direction=reward_direction,
                    state_context={
                        "dimension": dimension_key,
                        "sub_key": "bid_level",
                        "mean_q": mean_q,
                    },
                    episode=current_episode,
                    snapshot_id=snapshot_id,
                    timestamp=timestamp,
                )
                observations.append(obs)
            continue

        # Standard dimension with sub-keys
        for sub_key, sub_value in dimension_value.items():
            if sub_value is None:
                continue
            if not isinstance(sub_value, dict):
                continue

            mean_q = sub_value.get("mean_q", 0.0)
            count = sub_value.get("count", 0)

            if count < 5:
                continue

            # Determine reward direction based on threshold
            if mean_q > _MEAN_Q_THRESHOLD:
                reward_direction = "positive"
            elif mean_q < -_MEAN_Q_THRESHOLD:
                reward_direction = "negative"
            else:
                continue  # Not significant enough

            # Determine game phase from sub-key
            game_phase = _SUBKEY_PHASE_MAP.get(sub_key, "mid")

            obs = RawObservation(
                category=category,
                game_phase=game_phase,
                dimension_key=f"{dimension_key}_{sub_key}",
                reward_direction=reward_direction,
                state_context={
                    "dimension": dimension_key,
                    "sub_key": sub_key,
                    "mean_q": mean_q,
                    "count": count,
                },
                episode=current_episode,
                snapshot_id=snapshot_id,
                timestamp=timestamp,
            )
            observations.append(obs)

    return observations


# ─────────────────────────────────────────────────────────────────────────────
# Supporting/Contradicting Observation Matching
# ─────────────────────────────────────────────────────────────────────────────


def _find_supporting_observations(
    insight: StrategicInsight,
    observations: list[RawObservation],
) -> list[RawObservation]:
    """Find observations that support an existing insight.

    An observation supports an insight if it shares the same category
    and has a positive reward direction.

    Args:
        insight: The existing insight to match against.
        observations: All raw observations from this cycle.

    Returns:
        List of supporting observations.
    """
    return [
        obs
        for obs in observations
        if obs.category == insight.category and obs.reward_direction == "positive"
    ]


def _find_contradicting_observations(
    insight: StrategicInsight,
    observations: list[RawObservation],
) -> list[RawObservation]:
    """Find observations that contradict an existing insight.

    An observation contradicts an insight if it shares the same category
    but has a negative reward direction.

    Args:
        insight: The existing insight to match against.
        observations: All raw observations from this cycle.

    Returns:
        List of contradicting observations.
    """
    return [
        obs
        for obs in observations
        if obs.category == insight.category and obs.reward_direction == "negative"
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Persistence Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _persist_insights(insights: list[StrategicInsight], path: Path) -> None:
    """Persist insights to insights_cache.json atomically.

    Args:
        insights: List of validated StrategicInsight objects.
        path: Path to insights_cache.json.
    """
    serialized = [_serialize_insight(i) for i in insights]
    try:
        _atomic_write_json(serialized, path)
    except OSError as e:
        logger.error("Failed to persist insights_cache.json: %s", e)


def _persist_evidence(
    evidence_aggregator: EvidenceAggregator,
    patterns: list[RepeatedPattern],
    raw_observations: list[RawObservation],
    data_dir: Path,
) -> None:
    """Persist evidence to strategy_evidence.json atomically.

    Args:
        evidence_aggregator: The aggregator instance with persist logic.
        patterns: All detected patterns.
        raw_observations: All raw observations from this cycle.
        data_dir: Directory for the evidence file.
    """
    try:
        evidence_aggregator.persist_evidence(patterns, raw_observations, data_dir)
    except OSError as e:
        logger.warning("Failed to persist strategy_evidence.json: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def run_insight_cycle(
    agent,
    current_episode: int,
    data_dir: Path | None = None,
) -> list[StrategicInsight]:
    """Execute one full insight generation cycle.

    Pipeline:
    1. Determine data directory (default: alongside this module)
    2. Load existing insights from insights_cache.json (handle corrupt/missing)
    3. Mark all existing insights as new=False
    4. Take extended snapshot using SnapshotExtender (read-only)
    5. Generate raw observations from snapshot dimensions
    6. Store observations in ObservationStore
    7. Aggregate evidence using EvidenceAggregator
    8. Promote observations→patterns via PromotionPipeline
    9. Promote patterns→insights via QualityGate + GeneralityValidator
    10. Generate text for promoted patterns using TextGenerator
    11. Merge duplicates using DuplicateMerger
    12. Apply confidence dynamics (growth for re-confirmed, decay check)
    13. Migrate legacy counter-intuitive entries
    14. Persist results atomically to insights_cache.json
    15. Persist evidence to strategy_evidence.json

    This function is READ-ONLY on the agent — it never modifies Q-tables,
    neural network weights, experience replay buffer, or hyperparameters.

    Args:
        agent: The WistDiscoveryAgent instance (read-only access to Q-tables).
        current_episode: Current training episode number.
        data_dir: Directory for persistence files (insights_cache.json,
            strategy_evidence.json). Defaults to the wist_discovery folder
            (parent of insight_pipeline package) if None.

    Returns:
        Updated list of StrategicInsight objects persisted to insights_cache.json.
        Returns empty list if agent is None or missing Q-table attributes.
    """
    # ─────────────────────────────────────────────────────────────────────────
    # Validate agent
    # ─────────────────────────────────────────────────────────────────────────
    if agent is None:
        logger.warning("agent is None, returning empty insight list")
        return []

    if not hasattr(agent, "play_q") or not hasattr(agent, "bid_q"):
        logger.warning(
            "agent missing play_q or bid_q attributes, returning empty insight list"
        )
        return []

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Determine data directory
    # ─────────────────────────────────────────────────────────────────────────
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent
    else:
        data_dir = Path(data_dir)

    insights_path = data_dir / "insights_cache.json"
    evidence_path = data_dir / "strategy_evidence.json"

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Load existing insights from insights_cache.json
    # ─────────────────────────────────────────────────────────────────────────
    raw_entries = _load_insights_cache(insights_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Create ObservationStore + migrate legacy entries
    # ─────────────────────────────────────────────────────────────────────────
    observation_store = ObservationStore()
    migrated_entries = migrate_legacy_insights(raw_entries, observation_store)

    # Convert migrated dicts to StrategicInsight objects
    existing_insights: list[StrategicInsight] = []
    for entry in migrated_entries:
        try:
            insight = StrategicInsight(
                strategy=entry["strategy"],
                category=entry["category"],
                tags=entry.get("tags", []),
                confidence=entry["confidence"],
                evidence_count=entry.get("evidence_count", 1),
                why=entry.get("why", ""),
                first_seen=entry.get("first_seen", 0),
                last_confirmed=entry.get("last_confirmed", 0),
                new=entry.get("new", False),
            )
            existing_insights.append(insight)
        except (KeyError, TypeError) as e:
            logger.warning("Skipping unmappable migrated entry: %s", e)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Mark all existing insights as new=False (Req 7.3)
    # ─────────────────────────────────────────────────────────────────────────
    for insight in existing_insights:
        insight.new = False

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: Check for empty Q-tables (no training data yet)
    # ─────────────────────────────────────────────────────────────────────────
    if not agent.play_q and not agent.bid_q:
        logger.info("Q-tables are empty, persisting migrated insights only.")
        _persist_insights(existing_insights, insights_path)
        return existing_insights

    # ─────────────────────────────────────────────────────────────────────────
    # Step 6: Take extended snapshot (READ-ONLY on agent — Req 12.1–12.5)
    # ─────────────────────────────────────────────────────────────────────────
    snapshot_extender = SnapshotExtender()
    snapshot = snapshot_extender.take_extended_snapshot(agent)
    snapshot_id = str(current_episode)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 7: Generate raw observations from snapshot dimensions
    # ─────────────────────────────────────────────────────────────────────────
    raw_observations = _generate_observations_from_snapshot(
        snapshot, current_episode, snapshot_id
    )

    if not raw_observations:
        logger.info("No significant observations generated from snapshot.")
        _persist_insights(existing_insights, insights_path)
        return existing_insights

    # ─────────────────────────────────────────────────────────────────────────
    # Step 8: Store observations in ObservationStore
    # ─────────────────────────────────────────────────────────────────────────
    for obs in raw_observations:
        observation_store.record(obs)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 9: Aggregate evidence using EvidenceAggregator
    # ─────────────────────────────────────────────────────────────────────────
    evidence_aggregator = EvidenceAggregator()
    aggregated_patterns = evidence_aggregator.aggregate(
        raw_observations, {snapshot_id: snapshot}
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 10: Promote observations→patterns via PromotionPipeline
    # ─────────────────────────────────────────────────────────────────────────
    promotion_pipeline = PromotionPipeline()
    quality_gate = QualityGate()
    generality_validator = GeneralityValidator()

    # Also find clusters from the ObservationStore
    store_patterns = promotion_pipeline.promote_observations_to_patterns(
        observation_store
    )

    # Combine aggregated patterns + store patterns, deduplicate by dimension_key
    all_patterns = aggregated_patterns + store_patterns
    seen_keys: set[str] = set()
    unique_patterns: list[RepeatedPattern] = []
    for p in all_patterns:
        if p.dimension_key not in seen_keys:
            seen_keys.add(p.dimension_key)
            unique_patterns.append(p)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 11: Promote patterns→insights via QualityGate + GeneralityValidator
    # ─────────────────────────────────────────────────────────────────────────
    promoted_insights, rejected = promotion_pipeline.promote_patterns_to_insights(
        unique_patterns,
        quality_gate,
        generality_validator,
        existing_insights,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 12: Generate text for promoted patterns using TextGenerator
    # ─────────────────────────────────────────────────────────────────────────
    text_generator = TextGenerator()

    for insight in promoted_insights:
        # Find the source pattern for this insight to attempt text generation
        matching_pattern = _find_pattern_for_insight(insight, unique_patterns)
        if matching_pattern is not None:
            result = text_generator.generate(matching_pattern)
            if result is not None:
                strategy_text, why_text = result
                insight.strategy = strategy_text
                insight.why = why_text

    # Also try generating new insights from patterns that weren't promoted
    # but pass the text generator's statistical gate
    additional_insights: list[StrategicInsight] = []
    for pattern in unique_patterns:
        result = text_generator.generate(pattern)
        if result is None:
            continue
        strategy_text, why_text = result

        # Check if this pattern already produced a promoted insight
        already_promoted = any(
            pi.category == pattern.category
            and pi.strategy == strategy_text
            for pi in promoted_insights
        )
        if already_promoted:
            continue

        # Create new insight from text-generated content
        episodes = [obs.episode for obs in pattern.observations]
        first_ep = min(episodes) if episodes else current_episode
        last_ep = max(episodes) if episodes else current_episode

        new_insight = StrategicInsight(
            strategy=strategy_text,
            category=pattern.category,
            tags=[pattern.category],
            confidence=pattern.confidence,
            evidence_count=pattern.observation_count,
            why=why_text,
            first_seen=first_ep,
            last_confirmed=last_ep,
            new=True,
        )

        # Quality gate check for text-generated insights
        qg_passes, _ = quality_gate.validate(
            candidate=new_insight,
            existing=existing_insights + promoted_insights + additional_insights,
            pattern=pattern,
        )
        if qg_passes:
            is_valid, _ = validate_insight(new_insight)
            if is_valid:
                additional_insights.append(new_insight)

    # Combine promoted + additional text-generated insights
    all_new_candidates = promoted_insights + additional_insights

    # ─────────────────────────────────────────────────────────────────────────
    # Step 13: Merge duplicates using DuplicateMerger
    # ─────────────────────────────────────────────────────────────────────────
    duplicate_merger = DuplicateMerger()
    genuinely_new_insights: list[StrategicInsight] = []

    for candidate in all_new_candidates:
        merged = duplicate_merger.merge_if_duplicate(candidate, existing_insights)
        if merged is None:
            # No duplicate found — this is genuinely new
            candidate.new = True
            genuinely_new_insights.append(candidate)
        # If merged is not None, the existing insight was updated in-place

    # ─────────────────────────────────────────────────────────────────────────
    # Step 14: Apply confidence dynamics (growth for re-confirmed, decay)
    # ─────────────────────────────────────────────────────────────────────────
    confidence_dynamics = ConfidenceDynamics()
    total_observations = max(1, len(raw_observations))

    for insight in existing_insights:
        supporting = _find_supporting_observations(insight, raw_observations)
        contradicting = _find_contradicting_observations(insight, raw_observations)

        for _ in supporting:
            confidence_dynamics.apply_supporting_evidence(
                insight, total_observations, current_episode
            )

        for _ in contradicting:
            confidence_dynamics.apply_contradicting_evidence(
                insight, total_observations
            )

    # Remove insights below confidence threshold
    surviving_existing = [
        i for i in existing_insights if not confidence_dynamics.should_remove(i)
    ]

    # ─────────────────────────────────────────────────────────────────────────
    # Step 15: Combine and validate final insight list
    # ─────────────────────────────────────────────────────────────────────────
    final_insights = surviving_existing + genuinely_new_insights

    # Final validation pass — only persist valid insights
    validated_insights: list[StrategicInsight] = []
    for insight in final_insights:
        is_valid, errors = validate_insight(insight)
        if is_valid:
            validated_insights.append(insight)
        else:
            logger.warning(
                "Dropping invalid insight: %s — errors: %s",
                insight.strategy[:50],
                errors,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 16: Persist results atomically to insights_cache.json
    # ─────────────────────────────────────────────────────────────────────────
    _persist_insights(validated_insights, insights_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 17: Persist evidence to strategy_evidence.json
    # ─────────────────────────────────────────────────────────────────────────
    _persist_evidence(evidence_aggregator, unique_patterns, raw_observations, data_dir)

    return validated_insights


# ─────────────────────────────────────────────────────────────────────────────
# Internal Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _find_pattern_for_insight(
    insight: StrategicInsight,
    patterns: list[RepeatedPattern],
) -> RepeatedPattern | None:
    """Find the source pattern that corresponds to a promoted insight.

    Matches by category and checks if the insight's strategy text
    references the pattern's dimension_key.

    Args:
        insight: The promoted insight to find the source pattern for.
        patterns: All unique patterns from this cycle.

    Returns:
        The matching RepeatedPattern, or None if no match found.
    """
    for pattern in patterns:
        if pattern.category == insight.category:
            # Check if the insight references this pattern's dimension_key
            if pattern.dimension_key in insight.strategy:
                return pattern
    # Fall back to first category match
    for pattern in patterns:
        if pattern.category == insight.category:
            return pattern
    return None
