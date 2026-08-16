# Implementation Plan: Strategic Insight Hierarchy

## Overview

Refactor the Wist insight system into a multi-stage pipeline (`agents/wist_discovery/insight_pipeline/`) that transforms raw Q-table observations into validated, general strategic insights. The pipeline implements: raw observation → repeated pattern → strategic insight, with an intermediate evidence layer separating statistical detection from natural-language generation.

## Tasks

- [x] 1. Set up package structure, schema, and core data models
  - [x] 1.1 Create the insight_pipeline package with schema module
    - Create directory `agents/wist_discovery/insight_pipeline/`
    - Create `__init__.py` with placeholder for `run_insight_cycle()` public API
    - Create `schema.py` with dataclasses: `RawObservation`, `RepeatedPattern`, `StrategicInsight`
    - Define `VALID_CATEGORIES` frozenset (13 categories), `REJECTION_REASONS` list
    - Implement validation functions for each field (confidence range, tag format, strategy length, etc.)
    - _Requirements: 7.1, 7.6, 3.1, 3.3, 3.5_

  - [ ]* 1.2 Write property tests for schema validation (Properties 7, 8, 4, 5)
    - **Property 7: Schema Completeness Invariant** — Generate partial insight dicts, verify rejection of incomplete entries
    - **Property 8: Confidence Range Invariant** — Generate confidence values via merge/growth/decay, verify [0.0, 1.0] range
    - **Property 4: Category Validity Invariant** — Generate insights with random categories, verify only VALID_CATEGORIES accepted
    - **Property 5: Tag Constraint Invariant** — Generate random tag lists, verify max 5 tags, lowercase+underscore only, max 30 chars
    - **Validates: Requirements 7.1, 7.6, 14.5, 6.2, 6.6, 3.1, 3.2, 3.5, 3.3**

  - [x] 1.3 Create test directory structure
    - Create `tests/test_insight_pipeline/__init__.py`
    - Create `tests/test_insight_pipeline/conftest.py` with shared fixtures (mock agent, sample Q-tables, sample observations)
    - _Requirements: 15.1–15.8_

- [x] 2. Implement ObservationStore
  - [x] 2.1 Implement the ObservationStore class
    - Create `observation_store.py` with `ObservationStore` class
    - Implement `record(observation)` — stores raw observations by category
    - Implement `get_related(category, game_phase, min_count=3)` — finds clusters of related observations
    - Implement `consume(observations)` — marks observations as consumed
    - Implement FIFO eviction when max_capacity (10,000) is reached
    - Ensure raw observations are never written to `insights_cache.json`
    - _Requirements: 1.1, 1.3, 1.4_

  - [ ]* 2.2 Write property tests for ObservationStore (Properties 2, 3)
    - **Property 2: Observation Separation** — Generate random observations, store them, verify none leak to insight output
    - **Property 3: Minimum Observation Threshold** — Generate observation sets of size 0–10, verify promotion only when ≥3 observations from ≥2 states
    - **Validates: Requirements 1.1, 1.3, 1.4, 13.3**

  - [ ]* 2.3 Write unit tests for ObservationStore
    - Test FIFO eviction at max capacity
    - Test that `get_related` returns only clusters meeting min_count
    - Test that consumed observations are no longer returned
    - _Requirements: 1.1, 1.3_

- [x] 3. Implement GeneralityValidator
  - [x] 3.1 Implement the GeneralityValidator class
    - Create `generality_validator.py` with `GeneralityValidator` class
    - Implement regex patterns: `TRICK_NUMBER_PATTERN`, `PLAYER_INDEX_PATTERN`, `NAMED_CARD_PATTERN`, `EPISODE_PATTERN`, `HAND_ID_PATTERN`
    - Implement `validate(text, evidence_count, distinct_states)` returning `(passes, rejection_reason)`
    - Rejection reasons: `literal_trick_number`, `specific_player`, `named_card`, `single_episode`, `insufficient_pattern_support`
    - Verify insight references only abstract concepts (position category, card-strength tier, trick phase)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

  - [ ]* 3.2 Write property tests for GeneralityValidator (Property 1)
    - **Property 1: Generality Invariant** — Generate random strings with/without specific patterns (card names, trick numbers, player indices, episode refs), verify validator correctly accepts/rejects
    - **Validates: Requirements 1.2, 1.5, 2.1, 2.2, 2.3, 2.4, 2.6**

  - [ ]* 3.3 Write unit tests for GeneralityValidator
    - Test rejection of "Queen of clubs beats all alternatives in trick 4" (named_card)
    - Test rejection of "player 2 should always lead" (specific_player)
    - Test rejection of "in trick 7, always trump" (literal_trick_number)
    - Test acceptance of "In late-game positions, leading with trump forces opponents..."
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 15.1_

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement SnapshotExtender
  - [x] 5.1 Implement the SnapshotExtender class
    - Create `snapshot_extender.py` with `SnapshotExtender` class
    - Implement `take_extended_snapshot(agent)` — reads Q-tables (no mutation) and computes all dimensions
    - Compute 9 new aggregate dimensions: `leading_vs_following`, `phase_behaviour`, `card_strength_by_role`, `trump_vs_nontrump`, `partner_vs_opponent_winning`, `suit_length`, `information_available`, `bid_strength_reliability`, `defensive_vs_attacking`
    - Preserve existing fields (`pos_prefs`, `suit_prefs`, `bid_prefs`, `rank_prefs`) unchanged
    - Set dimension value to `null` if fewer than 5 contributing Q-table states
    - Each dimension entry contains `mean_q` (float) and `count` (int)
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 12.1, 12.2_

  - [ ]* 5.2 Write property tests for SnapshotExtender (Properties 15, 16)
    - **Property 15: Learning Preservation Invariant** — Snapshot agent state before/after snapshot extension, verify Q-tables unchanged
    - **Property 16: Snapshot Dimension Structure** — Generate Q-tables of varying sizes, verify 9–15 dimension keys, each with mean_q and count, null when count < 5
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 12.1, 12.2, 12.5**

- [x] 6. Implement EvidenceAggregator
  - [x] 6.1 Implement the EvidenceAggregator class
    - Create `evidence_aggregator.py` with `EvidenceAggregator` class
    - Implement `aggregate(observations, snapshots)` — groups observations by dimension, computes confidence, returns patterns meeting thresholds
    - Promotion thresholds: confidence ≥ 0.3 AND observed across ≥ 3 independent strategy snapshots
    - Implement `persist_evidence(patterns)` — writes to `strategy_evidence.json`
    - Aggregate raw observations into dimensions: positional behaviour, suit preference, bid behaviour, rank preference
    - Retain below-threshold evidence for future aggregation
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

  - [ ]* 6.2 Write property tests for EvidenceAggregator (Property 12)
    - **Property 12: Text Generation Gate** — Generate patterns at/below threshold, verify only patterns with confidence ≥ 0.3, evidence_count ≥ 3 across ≥ 2 states, observed across ≥ 3 snapshots are promoted
    - **Validates: Requirements 10.1, 10.4, 9.3**

  - [ ]* 6.3 Write unit tests for EvidenceAggregator
    - Test that below-threshold patterns are retained but not promoted
    - Test that `strategy_evidence.json` is correctly written with metadata
    - Test aggregation of observations into correct strategic dimensions
    - _Requirements: 9.1, 9.3, 9.4, 9.5_

- [x] 7. Implement PromotionPipeline and QualityGate
  - [x] 7.1 Implement the PromotionPipeline and QualityGate classes
    - Create `promotion_pipeline.py` with `PromotionPipeline` and `QualityGate` classes
    - Implement three-stage pipeline: raw observation → repeated pattern → strategic insight
    - Each entry carries a `stage` label indicating its pipeline position
    - `promote_observations_to_patterns()` — promotes when ≥3 related observations across ≥2 game states
    - `promote_patterns_to_insights()` — applies quality gate + generality validation
    - QualityGate implements 7 validation checks: (1) ≥3 episodes, (2) ≥2 game states, (3) recurring condition, (4) causal reason in "why", (5) no >80% text similarity, (6) surprising contradicts default, (7) reason references evidence
    - Return rejected candidates with reasons
    - _Requirements: 11.1–11.9, 13.1–13.4_

  - [ ]* 7.2 Write property tests for QualityGate (Property 17)
    - **Property 17: Quality Gate Conjunction** — Generate candidates passing/failing subsets of the 7 checks, verify correct accept/reject behavior
    - **Validates: Requirements 11.1–11.9**

  - [ ]* 7.3 Write unit tests for PromotionPipeline
    - Test pattern promotion with evidence_count=10 across 5 game states → promoted
    - Test pattern with evidence_count=2 → NOT promoted
    - Test that stage labels are correctly assigned at each pipeline phase
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 15.3, 15.4_

- [x] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement TextGenerator
  - [x] 9.1 Implement the TextGenerator class
    - Create `text_generator.py` with `TextGenerator` class
    - Implement `generate(pattern)` returning `(strategy_text, why_text)` or `None`
    - Enforce: only generates text for patterns that passed statistical confirmation gate
    - Strategy text max 200 chars, why text max 500 chars
    - Forbid subjective qualifiers: "brilliant", "optimal", "perfect", "amazing", "incredible", "genius", "unbelievable"
    - "why" must include quantitative reference (evidence_count, effect size, consistency measure)
    - Text must not extrapolate beyond confirmed pattern
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [ ]* 9.2 Write property tests for TextGenerator (Properties 13, 14)
    - **Property 13: No Subjective Qualifiers** — Generate text outputs from random patterns, verify no forbidden words appear
    - **Property 14: Quantitative "why" Field** — Generate why texts, verify each contains numeric content (evidence_count, percentage, or ratio)
    - **Validates: Requirements 10.2, 10.3**

- [x] 10. Implement DuplicateMerger
  - [x] 10.1 Implement the DuplicateMerger class
    - Create `duplicate_merger.py` with `DuplicateMerger` class
    - Implement `merge_if_duplicate(candidate, existing)` — checks semantic equivalence
    - Semantic equivalence: same category + overlapping tags + >80% token overlap
    - Merge rules: confidence += candidate evidence weight (cap at 1.0), evidence_count += candidate.evidence_count, last_confirmed = candidate's episode
    - Return `None` if no duplicate found (candidate is new)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [ ]* 10.2 Write property tests for DuplicateMerger (Property 9)
    - **Property 9: Merge Evidence Accumulation** — Generate pairs of insights to merge, verify evidence_count = sum of both, last_confirmed = candidate's episode, confidence capped at 1.0
    - **Validates: Requirements 6.3, 6.4, 6.2, 6.6**

  - [ ]* 10.3 Write unit tests for DuplicateMerger
    - Test merge of two strategies with same category + overlapping tags + equivalent text
    - Test that merged confidence is capped at 1.0
    - Test that non-duplicate candidate returns None
    - _Requirements: 6.1, 6.5, 15.2_

- [x] 11. Implement ConfidenceDynamics
  - [x] 11.1 Implement the ConfidenceDynamics class
    - Create `confidence_dynamics.py` with `ConfidenceDynamics` class
    - Implement `apply_supporting_evidence(insight, total_observations)` — increases confidence by 1/total_observations, caps at 1.0
    - Implement `apply_contradicting_evidence(insight, total_observations)` — recalculates as supporting_count / total_observations
    - Implement `should_remove(insight)` — True if confidence < 0.3 (default threshold)
    - Implement `validate_surprising_pattern(pattern)` — applies all 6 criteria from Requirement 4
    - Update `last_confirmed` on supporting evidence
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 4.1–4.7, 13.5, 13.6_

  - [ ]* 11.2 Write property tests for ConfidenceDynamics (Properties 10, 11, 6)
    - **Property 10: Confidence Growth** — Generate supporting observations, verify confidence increases by 1/total_observations (capped at 1.0)
    - **Property 11: Confidence Decay and Removal** — Generate contradicting observations, verify decay formula and removal when < 0.3
    - **Property 6: Surprising Pattern Threshold Invariant** — Generate patterns with varying evidence counts, state counts, snapshot counts, verify all 6 criteria for surprising_pattern
    - **Validates: Requirements 14.1, 14.2, 14.3, 14.4, 14.5, 13.5, 13.6, 4.1–4.7**

  - [ ]* 11.3 Write unit tests for ConfidenceDynamics
    - Test confidence growth after supporting evidence
    - Test confidence decay after contradicting evidence
    - Test removal when confidence drops below 0.3
    - Test surprising_pattern rejection with < 3 game states
    - _Requirements: 14.1, 14.2, 14.3, 15.5, 15.6, 15.7_

- [x] 12. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Implement Migration logic
  - [x] 13.1 Implement the migration module
    - Create `migration.py` with migration logic for legacy `insights_cache.json`
    - Detect and migrate entries with `"counter-intuitive"` category to internal candidate store
    - Normalize legacy integer confidence (1–100) to float (0.0–1.0)
    - Map old schema fields to new schema where possible
    - Remove unmappable entries, log count of migrated entries
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [ ]* 13.2 Write unit tests for migration
    - Test counter-intuitive entries are migrated to internal store and removed from cache
    - Test integer confidence normalization (e.g., 55 → 0.55)
    - Test old-format entries are mapped to new schema
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 14. Wire the full pipeline in __init__.py
  - [x] 14.1 Implement run_insight_cycle() public API
    - Wire all components in correct pipeline order in `__init__.py`
    - Pipeline order: Q-table → snapshot → observations → evidence aggregation → promotion → text generation → dedup/merge → confidence dynamics → persist
    - Handle `new` flag: set True on first cycle, False on subsequent cycles
    - Implement atomic file writes (write to temp file, then rename) for all JSON persistence
    - Handle all error conditions from the design (corrupt files, empty Q-tables, etc.)
    - _Requirements: 9.6, 7.2, 7.3, 7.4, 12.1–12.5_

  - [ ]* 14.2 Write integration tests for run_insight_cycle
    - Test full pipeline smoke test with mock agent and populated Q-tables
    - Test multi-cycle evolution: run 3 cycles, verify confidence grows and `new` flag transitions
    - Test output matches new schema (all fields, correct types)
    - Test schema round-trip: persist → reload → verify fields match
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 15.8_

- [x] 15. Integrate pipeline with discovery_agent.py
  - [x] 15.1 Update discovery_agent.py to use the new insight pipeline
    - Import `run_insight_cycle` from `agents.wist_discovery.insight_pipeline`
    - Replace existing insight generation calls with `run_insight_cycle(agent, current_episode)`
    - Ensure the pipeline is called at snapshot intervals (same trigger as current insight generation)
    - Verify agent Q-tables, neural network weights, and replay buffer are unchanged after invocation
    - Update any imports or references that previously pointed to the old insight system
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

  - [ ]* 15.2 Write integration test for learning preservation (Property 15)
    - **Property 15: Learning Preservation Invariant** — Capture Q-table entry count, neural network weights, replay buffer size before/after `run_insight_cycle`, verify equality
    - **Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5**

- [x] 16. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases from Requirement 15
- The pipeline package is self-contained — the only integration point is `run_insight_cycle()` called from `discovery_agent.py`
- All file I/O uses atomic writes (write to temp, then rename) to prevent corruption
- The pipeline is read-only with respect to the learning agent's internal state

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.3"] },
    { "id": 1, "tasks": ["1.2", "2.1", "3.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "3.2", "3.3", "5.1"] },
    { "id": 3, "tasks": ["5.2", "6.1"] },
    { "id": 4, "tasks": ["6.2", "6.3", "7.1"] },
    { "id": 5, "tasks": ["7.2", "7.3", "9.1"] },
    { "id": 6, "tasks": ["9.2", "10.1", "11.1"] },
    { "id": 7, "tasks": ["10.2", "10.3", "11.2", "11.3", "13.1"] },
    { "id": 8, "tasks": ["13.2", "14.1"] },
    { "id": 9, "tasks": ["14.2", "15.1"] },
    { "id": 10, "tasks": ["15.2"] }
  ]
}
```
