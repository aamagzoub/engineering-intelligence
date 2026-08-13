"""Migration module for legacy insights_cache.json entries.

Handles migration of entries from the old schema format to the new
StrategicInsight schema. Specifically:

- Detects and migrates entries with "counter-intuitive" category to the
  internal candidate store as raw observations.
- Normalizes legacy integer confidence (1-100) to float (0.0-1.0).
- Maps old schema fields to new schema where possible.
- Discards unmappable entries and logs counts.

Classes:
    InsightMigrator — Handles migration of legacy insights_cache.json to new schema.

Requirements: 5.1, 5.2, 5.3, 5.4
"""

from __future__ import annotations

import logging
import time

from agents.wist_discovery.insight_pipeline.schema import (
    VALID_CATEGORIES,
    validate_insight,
    StrategicInsight,
)
from agents.wist_discovery.insight_pipeline.observation_store import ObservationStore
from agents.wist_discovery.insight_pipeline.schema import RawObservation


logger = logging.getLogger(__name__)


# Mapping from legacy category names to valid new categories.
# Categories not in this map and not in VALID_CATEGORIES are unmappable.
_LEGACY_CATEGORY_MAP: dict[str, str] = {
    "timing": "position",
    "trump": "trump_management",
    "defense": "defense",
    "bidding": "bidding",
}


class InsightMigrator:
    """Handles migration of legacy insights_cache.json to new schema.

    The migrator detects entries with the "counter-intuitive" category and
    moves them to an internal candidate store (ObservationStore), while
    mapping all other entries to the new StrategicInsight schema format.

    Attributes:
        observation_store: Optional store for counter-intuitive candidates.
    """

    def __init__(self, observation_store: ObservationStore | None = None) -> None:
        """Initialize the migrator.

        Args:
            observation_store: Optional ObservationStore for counter-intuitive
                candidates. If None, counter-intuitive entries are discarded.
        """
        self.observation_store = observation_store

    def migrate(self, legacy_entries: list[dict]) -> tuple[list[dict], list[dict]]:
        """Migrate legacy entries to new format.

        Args:
            legacy_entries: List of dicts from old insights_cache.json format.

        Returns:
            Tuple of (migrated_new_format, counter_intuitive_candidates).
            - migrated_new_format: entries mapped to new schema (for insights_cache.json)
            - counter_intuitive_candidates: entries with category "counter-intuitive"
              moved to internal candidate store (NOT written to insights_cache.json)
        """
        migrated: list[dict] = []
        counter_intuitive_candidates: list[dict] = []
        discarded_count = 0

        for entry in legacy_entries:
            if not isinstance(entry, dict):
                discarded_count += 1
                continue

            # Check if this is a counter-intuitive entry (Req 5.1, 5.2, 5.4)
            entry_category = entry.get("category") or entry.get("type")
            if entry_category == "counter-intuitive":
                counter_intuitive_candidates.append(entry)
                self._store_counter_intuitive(entry)
                continue

            # Try to map to new schema
            new_entry = self.map_to_new_schema(entry)
            if new_entry is not None:
                migrated.append(new_entry)
            else:
                discarded_count += 1

        total = len(legacy_entries)
        logger.info(
            "Migration complete: %d total entries, %d migrated to new schema, "
            "%d counter-intuitive moved to candidate store, %d discarded",
            total,
            len(migrated),
            len(counter_intuitive_candidates),
            discarded_count,
        )

        return migrated, counter_intuitive_candidates

    def normalize_confidence(self, value) -> float:
        """Convert legacy confidence to 0.0-1.0 range.

        If integer in range 1-100, divide by 100.
        If already float in [0.0, 1.0], keep as-is.
        Otherwise default to 0.5.

        Args:
            value: The confidence value from the legacy entry.

        Returns:
            Normalized confidence as a float in [0.0, 1.0].
        """
        if value is None:
            return 0.5

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.5

        # Integer-style confidence (1-100 range): normalize by dividing by 100
        if isinstance(value, int) and 1 <= value <= 100:
            return numeric / 100.0

        # Already a float that looks like legacy integer (> 1.0 means on 1-100 scale)
        if numeric > 1.0 and numeric <= 100.0:
            return numeric / 100.0

        # Already in valid float range
        if 0.0 <= numeric <= 1.0:
            return numeric

        # Out of range or negative
        return 0.5

    def map_to_new_schema(self, entry: dict) -> dict | None:
        """Map old schema fields to new schema.

        Old fields: insight, category, confidence, episode
        New fields: strategy, category, tags, confidence, evidence_count,
                    why, first_seen, last_confirmed, new

        Returns None if the entry cannot be mapped (missing required fields
        or invalid/unmappable category).

        Args:
            entry: A dict from the legacy insights_cache.json format.

        Returns:
            A dict conforming to the new schema, or None if unmappable.
        """
        # Extract strategy text
        strategy = self._extract_strategy_text(entry)
        if strategy is None:
            return None

        # Map category — check both "type" and "category" fields
        legacy_category = entry.get("category") or entry.get("type")
        category = self._map_category(legacy_category)
        if category is None:
            return None

        # Normalize confidence
        confidence = self.normalize_confidence(entry.get("confidence"))

        # Extract episode
        episode = self._extract_episode(entry)
        if episode is None:
            episode = 0

        # Extract or default why
        why = entry.get("why", "")
        if not isinstance(why, str) or not why.strip():
            why = "Migrated from legacy system"
        else:
            why = why.strip()[:500]

        # Build tags from legacy fields
        tags: list[str] = []
        if legacy_category and legacy_category != category:
            # Add the original category name as a tag if it was remapped
            tag_candidate = (
                legacy_category.lower().replace("-", "_").replace(" ", "_")
            )
            if len(tag_candidate) <= 30 and tag_candidate.replace("_", "").isalpha():
                tags.append(tag_candidate)

        # Build new schema entry
        new_entry = {
            "strategy": strategy,
            "category": category,
            "tags": tags,
            "confidence": confidence,
            "evidence_count": 1,
            "why": why,
            "first_seen": episode,
            "last_confirmed": episode,
            "new": False,
        }

        return new_entry

    def _store_counter_intuitive(self, entry: dict) -> None:
        """Store a counter-intuitive entry in the observation store.

        If observation_store is None, the entry is simply tracked in the
        candidates list but not stored for future promotion.
        """
        if self.observation_store is None:
            return

        strategy_text = self._extract_strategy_text(entry) or "unknown"
        episode = self._extract_episode(entry) or 0

        observation = RawObservation(
            category="surprising_pattern",
            game_phase="unknown",
            dimension_key="legacy_counter_intuitive",
            reward_direction="positive",
            state_context={
                "legacy_text": strategy_text,
                "legacy_confidence": entry.get("confidence"),
                "source": "migration",
            },
            episode=episode,
            snapshot_id="legacy",
            timestamp=time.time(),
        )
        self.observation_store.record(observation)

    @staticmethod
    def _extract_strategy_text(entry: dict) -> str | None:
        """Extract strategy text from a legacy entry.

        Legacy entries may use 'insight', 'text', 'description', or 'strategy'
        as the key for the strategy content.
        """
        for key in ("strategy", "insight", "text", "description"):
            if key in entry and isinstance(entry[key], str) and entry[key].strip():
                return entry[key].strip()[:200]
        return None

    @staticmethod
    def _extract_episode(entry: dict) -> int | None:
        """Extract episode number from a legacy entry."""
        episode = entry.get("episode")
        if episode is None:
            return None
        try:
            ep = int(episode)
            return ep if ep >= 0 else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _map_category(legacy_category: str | None) -> str | None:
        """Map a legacy category string to a valid new category.

        Returns None if the category cannot be mapped.
        """
        if legacy_category is None:
            return None

        # Already a valid new category
        if legacy_category in VALID_CATEGORIES:
            return legacy_category

        # Try the legacy mapping
        return _LEGACY_CATEGORY_MAP.get(legacy_category)


# ============================================================================
# Module-level convenience function (backward-compatible API)
# ============================================================================


def migrate_legacy_insights(
    insights_data: list[dict],
    observation_store: ObservationStore | None = None,
) -> list[dict]:
    """Migrate legacy insights_cache.json entries to the new schema.

    This is a convenience wrapper around InsightMigrator for backward
    compatibility.

    1. Remove entries with category "counter-intuitive" — move them to
       observation_store as internal candidates (if provided)
    2. Normalize integer confidence (1-100) to float (0.0-1.0)
    3. Map old schema fields to new schema
    4. Remove entries that can't be mapped
    5. Return the migrated list (only valid entries in new schema)

    Args:
        insights_data: Raw list of dicts loaded from legacy insights_cache.json
        observation_store: Optional ObservationStore for counter-intuitive candidates

    Returns:
        List of dicts conforming to new insight schema
    """
    migrator = InsightMigrator(observation_store=observation_store)
    migrated, _candidates = migrator.migrate(insights_data)
    return migrated
