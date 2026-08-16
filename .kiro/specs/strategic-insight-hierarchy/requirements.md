# Requirements Document

## Introduction

Refactor the Wist learning/insight system so that `insights_cache.json` contains high-level reusable strategies instead of single-play observations. The current system promotes raw observations (specific cards, specific trick numbers, isolated comparisons) directly into the insight cache. This feature introduces a learning hierarchy — raw observation → repeated pattern → strategic insight — ensuring that only general, evidence-backed strategies reach the final insight file.

## Glossary

- **Insight_System**: The subsystem within the Wist Discovery Agent that analyses learned behaviour, detects patterns, and produces human-readable strategic insights persisted to `insights_cache.json`.
- **Raw_Observation**: A single data point from one game state, such as "Queen of clubs beats all alternatives in trick 4" or "Bid 8 beats bid 12 in this state". Not suitable as a final insight.
- **Repeated_Pattern**: A cluster of related raw observations that recur across multiple comparable game states and preferably across multiple training snapshots.
- **Strategic_Insight**: A general, reusable strategy derived from repeated patterns. It does not depend on exact trick numbers, exact player numbers, exact individual cards, one particular hand, exact training episodes, isolated Q-values, or isolated suit comparisons.
- **Evidence_Layer**: An intermediate data layer (optionally persisted as `strategy_evidence.json`) that aggregates raw observations into pattern candidates before promotion to strategic insights.
- **Strategy_Snapshot**: A periodic capture of aggregate Q-table statistics (position preferences, suit preferences, bid preferences, rank preferences) stored in `strategy_snapshots.json`.
- **Surprising_Pattern**: A strategic insight that is non-obvious from the game rules alone, repeats across multiple comparable states, has strong supporting evidence, and would make a human player react with genuine interest.
- **Confidence_Score**: A numeric value between 0.0 and 1.0 representing how strongly the evidence supports a given strategy.
- **Evidence_Count**: The number of distinct raw observations supporting a strategic insight.
- **Strategy_Category**: A classification label for a strategic insight drawn from a defined set (leading, following, position, card_preservation, suit_management, trump_management, bidding, defense, partner_play, risk, information, endgame, surprising_pattern).

## Requirements

### Requirement 1: Observation Separation

**User Story:** As a developer reviewing insights, I want raw observations separated from final strategies, so that the insight cache contains only reusable strategic knowledge.

#### Acceptance Criteria

1. THE Insight_System SHALL store raw observations in an internal data structure separate from `insights_cache.json`, where a raw observation is defined as a single game-event record referencing a specific card, trick number, bid value, or game outcome from one episode (e.g., "Queen of clubs beats all alternatives in trick 4").
2. THE Insight_System SHALL NOT write any entry into `insights_cache.json` that references a specific card by name, a specific trick number, a specific bid value from a single episode, or a specific game outcome from a single episode.
3. WHEN a raw observation is recorded, THE Insight_System SHALL retain it in the internal observation store as supporting evidence for pattern detection until it has been consumed by the combination process or until the observation store reaches its configured maximum capacity.
4. THE Insight_System SHALL require a minimum of 3 related raw observations sharing the same category and game-phase context to be combined before a strategic insight is created and written to `insights_cache.json`.
5. WHEN raw observations are combined into a strategic insight, THE Insight_System SHALL produce a generalized statement that applies across multiple game situations without referencing any single episode, specific card, or specific trick number.

### Requirement 2: Generality Validation

**User Story:** As a developer, I want each final insight validated for generality, so that no insight depends on a single game state.

#### Acceptance Criteria

1. THE Insight_System SHALL reject any candidate insight whose text contains a literal trick number (e.g. "trick 3", "trick 7").
2. THE Insight_System SHALL reject any candidate insight whose text references a specific player index or a named seat position tied to a single observed state (e.g. "player 2", "the dealer").
3. THE Insight_System SHALL reject any candidate insight that names an individual card by rank and suit together (e.g. "Queen of clubs", "5 of hearts").
4. THE Insight_System SHALL reject any candidate insight whose text references a single hand identifier or a single training episode number.
5. THE Insight_System SHALL reject any candidate insight that reports an isolated Q-value or an isolated suit comparison supported by fewer than 3 distinct game states.
6. WHEN evaluating a candidate insight for promotion, THE Insight_System SHALL apply the generality test by verifying the insight references only abstract concepts (position category, card-strength tier, trick phase) rather than concrete game-state identifiers.
7. WHEN the Insight_System rejects a candidate insight, THE Insight_System SHALL exclude the insight from the promoted insights collection and record the rejection reason as one of: literal_trick_number, specific_player, named_card, single_episode, or insufficient_pattern_support.

### Requirement 3: Strategic Categories

**User Story:** As a developer, I want insights classified into meaningful strategic categories, so that strategies are organised by their tactical domain.

#### Acceptance Criteria

1. THE Insight_System SHALL classify each strategic insight into exactly one primary category from the set: leading, following, position, card_preservation, suit_management, trump_management, bidding, defense, partner_play, risk, information, endgame, surprising_pattern.
2. IF the Insight_System cannot classify a candidate insight into any category from the defined set, THEN THE Insight_System SHALL reject the candidate and SHALL NOT write it to `insights_cache.json`.
3. THE Insight_System SHALL support optional additional tags on each insight, where each tag is a lowercase string containing only letters and underscores, each tag is at most 30 characters long, and each insight has at most 5 tags.
4. THE Insight_System SHALL allow tag values to be drawn from the primary category set or from other domain-relevant labels, enabling cross-referencing of insights that span multiple tactical domains.
5. THE Insight_System SHALL NOT use "counter-intuitive" as a final strategic category in `insights_cache.json`.

### Requirement 4: Surprising Pattern Category

**User Story:** As a player reading insights, I want genuinely surprising discoveries highlighted, so that I learn non-obvious strategic knowledge.

#### Acceptance Criteria

1. WHEN classifying an insight as surprising_pattern, THE Insight_System SHALL verify that the pattern repeats across at least 3 comparable game states that differ in at least one of: hand composition, trick number, or player position.
2. WHEN classifying an insight as surprising_pattern, THE Insight_System SHALL verify that the pattern appears across at least 2 distinct training snapshots separated by at least 1000 training episodes.
3. WHEN classifying an insight as surprising_pattern, THE Insight_System SHALL verify that the pattern has an evidence_count of at least 10 supporting observations.
4. WHEN classifying an insight as surprising_pattern, THE Insight_System SHALL verify that the pattern is not obvious directly from the Wist rules by confirming the pattern contradicts the default rank-ordering or positional expectation for its context.
5. WHEN classifying an insight as surprising_pattern, THE Insight_System SHALL verify that the pattern is strategically reusable by confirming it references abstract game conditions rather than specific cards or episodes.
6. WHEN classifying an insight as surprising_pattern, THE Insight_System SHALL verify that the ratio of supporting observations to contradicting observations is at least 2:1, indicating the result is unlikely to be noise.
7. THE Insight_System SHALL NOT classify a pattern as surprising_pattern if it is supported by observations from fewer than 3 distinct game states.

### Requirement 5: Counter-Intuitive Removal

**User Story:** As a developer, I want the counter-intuitive category removed from final insights, so that only properly validated surprising patterns appear.

#### Acceptance Criteria

1. THE Insight_System SHALL NOT include "counter-intuitive" as a category in any entry written to `insights_cache.json`.
2. THE Insight_System SHALL retain "counter-intuitive" only internally as a candidate label for raw observations that may later be promoted to surprising_pattern.
3. WHEN a counter-intuitive candidate accumulates at least 10 supporting observations across at least 3 comparable game states and at least 2 training snapshots, THE Insight_System SHALL evaluate it for promotion to surprising_pattern using the criteria defined in Requirement 4.
4. IF the insights_cache.json file already contains entries with category "counter-intuitive" from prior runs, THEN THE Insight_System SHALL migrate them to the internal candidate store on the next insight generation cycle and remove them from `insights_cache.json`.

### Requirement 6: Duplicate Strategy Merging

**User Story:** As a developer, I want duplicate strategies merged, so that the insight cache does not grow with redundant entries.

#### Acceptance Criteria

1. WHEN a new candidate strategy shares the same category and at least one overlapping tag with an existing strategy, and its semantic content expresses the same underlying advice, THE Insight_System SHALL merge them into a single entry by retaining the existing entry and discarding the candidate as a separate record.
2. WHEN merging duplicate strategies, THE Insight_System SHALL increase the confidence score of the merged entry by adding the candidate's evidence weight, capping the result at 1.0.
3. WHEN merging duplicate strategies, THE Insight_System SHALL increment the evidence_count of the merged entry by the candidate's evidence_count (minimum 1).
4. WHEN merging duplicate strategies, THE Insight_System SHALL update the last_confirmed field of the merged entry to the candidate's episode number.
5. THE Insight_System SHALL determine semantic equivalence by comparing the category, tags, and the core strategic advice expressed in the text of the candidate and existing strategies.
6. IF merging would result in a confidence score exceeding 1.0, THEN THE Insight_System SHALL cap the confidence score at 1.0.

### Requirement 7: Revised Insight Schema

**User Story:** As a developer, I want a structured insight schema, so that each entry contains the strategy text, category, tags, confidence, evidence count, explanation, temporal tracking, and novelty flag.

#### Acceptance Criteria

1. THE Insight_System SHALL persist each strategic insight in `insights_cache.json` with the following fields: strategy (string, maximum 200 characters), category (string, one of the defined categories from the set: leading, following, position, card_preservation, suit_management, trump_management, bidding, defense, partner_play, risk, information, endgame, surprising_pattern), tags (list of strings, maximum 5 tags per insight, each tag maximum 30 characters), confidence (float 0.0–1.0), evidence_count (integer, minimum 1), why (string, maximum 500 characters), first_seen (integer episode, minimum 0), last_confirmed (integer episode, minimum 0, greater than or equal to first_seen), new (boolean).
2. WHEN a strategy is first created, THE Insight_System SHALL set the "new" flag to true and set both first_seen and last_confirmed to the current episode number.
3. WHEN the next insight generation cycle executes after an insight was created, THE Insight_System SHALL set that insight's "new" flag to false.
4. WHEN an existing insight is re-confirmed by new supporting evidence during an insight generation cycle, THE Insight_System SHALL update last_confirmed to the current episode number and increment evidence_count by the number of new supporting observations found in that cycle.
5. THE Insight_System SHALL compute confidence as a value between 0.0 and 1.0, derived from the ratio of supporting observations to total observations processed for the pattern's strategic dimension, where a higher ratio yields a higher confidence value.
6. IF a persisted insight is missing any required field or contains a value outside its defined bounds, THEN THE Insight_System SHALL reject the entry and not persist it to `insights_cache.json`.

### Requirement 8: Strategy Snapshot Extension

**User Story:** As a developer, I want strategy snapshots to capture richer aggregate behaviour dimensions, so that the pattern detection layer has more data to work with.

#### Acceptance Criteria

1. WHEN a training snapshot is recorded, THE Insight_System SHALL include aggregate Q-value statistics for each of the following behaviour dimensions: leading versus following (position 0 vs positions 1-3), early phase (hand size 10-13) versus middle phase (hand size 5-9) versus late phase (hand size 1-4), card-strength tier (low/mid/upper/high) used while leading versus following, trump-suit plays versus non-trump-suit plays, partner-winning context (partner currently winning the trick) versus opponent-winning context, long-suit situations (4 or more cards in suit) versus short-suit situations (2-3 cards in suit) versus void situations (0 cards in suit), information available at decision time (number of cards already played in current trick, 0-3), bidding strength (bid level chosen: 7-13 or PASS) versus bid reliability (ratio of bids met to total bids), and defensive situations (opponent holds the contract) versus attacking situations (own team holds the contract).
2. THE Insight_System SHALL represent each aggregate dimension as a numeric value computed from Q-table entries, where each dimension entry contains at minimum a sum or mean of Q-values and a count of contributing states, producing no fewer than 9 and no more than 15 top-level dimension keys per snapshot.
3. THE Insight_System SHALL preserve all existing snapshot fields (`pos_prefs`, `suit_prefs`, `bid_prefs`, `rank_prefs`) unchanged in structure and semantics when extending the schema with new dimension keys.
4. IF a dimension has fewer than 5 contributing Q-table states at snapshot time, THEN THE Insight_System SHALL record that dimension's value as null rather than computing a statistically unreliable aggregate.
5. WHEN a new training snapshot is recorded, THE Insight_System SHALL include the extended aggregate dimensions alongside the existing fields in a single atomic write to `strategy_snapshots.json`, completing within 5 seconds for Q-tables of up to 30,000 entries.

### Requirement 9: Evidence Aggregation Layer

**User Story:** As a developer, I want an intermediate evidence layer between raw Q-values and final insights, so that pattern detection is separated from natural-language generation.

#### Acceptance Criteria

1. THE Insight_System SHALL implement a pattern aggregation step that consumes strategy snapshots and produces strategy candidates as its output, before any natural-language insight generation occurs.
2. THE Insight_System SHALL aggregate raw observations from strategy snapshots into the following strategic dimensions: positional behaviour, suit preference, bid behaviour, and rank preference.
3. WHEN aggregated evidence for a pattern reaches a confidence score of at least 0.3 AND the pattern has been observed across at least 3 independent strategy snapshots, THE Insight_System SHALL promote the pattern to a strategy candidate eligible for insight generation.
4. IF aggregated evidence for a pattern does not meet the confidence or repetition threshold, THEN THE Insight_System SHALL retain the evidence for future aggregation without passing it to the insight generation layer.
5. THE Insight_System SHALL persist pattern evidence in `strategy_evidence.json` whenever a new strategy snapshot is processed, recording per-dimension aggregated observations and their current confidence and repetition counts.
6. THE Insight_System SHALL follow the pipeline order: Q-table learned values → strategy_snapshots.json → pattern/evidence aggregation (strategy_evidence.json) → strategy candidates → confidence and repetition validation → insights_cache.json.

### Requirement 10: Text Generation Constraint

**User Story:** As a developer, I want the text generation layer restricted to explaining patterns that have been statistically established, so that no strategies are invented.

#### Acceptance Criteria

1. THE Insight_System SHALL NOT allow the natural-language layer to generate strategic text unless the statistical/pattern detection layer has first confirmed the underlying pattern with an evidence_count of at least 3 distinct observations across at least 2 comparable game states.
2. THE Insight_System SHALL ensure that the natural-language layer only explains patterns that have passed the Evidence Aggregation Layer (Requirement 9) confidence and repetition thresholds, using language that describes observable gameplay behaviour without subjective qualifiers such as "brilliant", "optimal", or "perfect".
3. WHEN producing the "why" explanation for a strategy, THE Insight_System SHALL include at least one quantitative reference from the learning data (evidence_count, effect size, or consistency measure) that a reviewer can verify against the stored evidence in `strategy_evidence.json` or `strategy_snapshots.json`.
4. IF the natural-language layer receives a pattern candidate that has not passed the statistical confirmation gate, THEN THE Insight_System SHALL discard the candidate without generating any strategy text or "why" explanation for it.
5. THE Insight_System SHALL NOT allow the natural-language layer to extrapolate, generalise beyond, or editorialize on a confirmed pattern — the generated text SHALL be limited to restating what the evidence shows.

### Requirement 11: Strategy Quality Gate

**User Story:** As a developer, I want a quality gate before any strategy is saved, so that only validated strategies reach the insight cache.

#### Acceptance Criteria

1. WHEN saving a final insight, THE Insight_System SHALL verify that the strategy is supported by evidence observed in at least 3 distinct episodes.
2. WHEN saving a final insight, THE Insight_System SHALL verify that the strategy applies to at least 2 distinct game states (differing in hand size, position, or trick count).
3. WHEN saving a final insight, THE Insight_System SHALL verify that the strategy references a condition that can recur in future hands (e.g., position, suit relationship, or trick phase) rather than a fixed card combination from a single deal.
4. WHEN saving a final insight, THE Insight_System SHALL verify that the strategy's text contains a causal reason (the "why" field is non-empty and references a game mechanic) rather than merely stating an outcome.
5. WHEN saving a final insight, THE Insight_System SHALL verify that no existing strategy in the cache shares both the same category and a text similarity above 80% (measured by token overlap) with the candidate.
6. WHEN saving a final insight marked as surprising_pattern, THE Insight_System SHALL verify that the pattern contradicts the default expectation for its category (e.g., a low card outperforming a high card in a context where high cards are generally favoured).
7. WHEN saving a final insight, THE Insight_System SHALL verify that the strategy's reason references at least one piece of learned evidence (a Q-value comparison, win-rate statistic, or episode observation) that supports the claim.
8. IF any of the 7 validation checks fails, THEN THE Insight_System SHALL reject the candidate, retain it as unpromoted evidence data, and not persist it to insights_cache.json.
9. IF all 7 validation checks pass, THEN THE Insight_System SHALL persist the candidate to insights_cache.json with a confidence score derived from the supporting evidence count.

### Requirement 12: Learning Preservation

**User Story:** As a developer, I want the refactoring to preserve the learning agent's discovery ability, so that no manual strategies are injected.

#### Acceptance Criteria

1. THE Insight_System SHALL NOT modify the learning agent's Q-tables, neural network weights, experience replay buffer, or hyperparameters.
2. THE Insight_System SHALL only read Q-values, experience data, and training statistics, and SHALL produce output solely for external display or logging.
3. THE Insight_System SHALL NOT insert predefined Wist strategy into the learner's decision path through reward shaping, curriculum modification, or action filtering.
4. WHEN generating insights, THE Insight_System SHALL derive all strategic content exclusively from learned Q-values, experience data, and training statistics without introducing externally authored rules or heuristics.
5. IF the Insight_System is invoked, THEN the learning agent's Q-table entry count, neural network weight values, and replay buffer size SHALL remain unchanged before and after the invocation.

### Requirement 13: Pattern Promotion Pipeline

**User Story:** As a developer, I want a clear promotion pipeline from observation to pattern to strategy, so that insight quality improves over time.

#### Acceptance Criteria

1. THE Insight_System SHALL implement a three-stage pipeline: raw observation → repeated pattern → strategic insight, where each entry carries a stage label indicating its current position in the pipeline.
2. WHEN an observation is first recorded, THE Insight_System SHALL store it at the raw observation stage only, with an initial observation count of 1.
3. WHEN at least 3 related observations sharing the same strategic dimension and reward direction are detected across at least 2 distinct game states, THE Insight_System SHALL promote them to a repeated pattern.
4. WHEN a repeated pattern passes the quality gate (Requirement 11), THE Insight_System SHALL promote it to a strategic insight and write it to `insights_cache.json`.
5. IF new training data contradicts a promoted strategy such that the strategy's confidence score drops below 0.1, THEN THE Insight_System SHALL remove the strategy from `insights_cache.json`.
6. IF new training data partially contradicts a promoted strategy but its confidence score remains at or above 0.1, THEN THE Insight_System SHALL reduce the confidence score of the associated strategy proportionally to the ratio of contradictory evidence and retain it in `insights_cache.json`.

### Requirement 14: Confidence Growth

**User Story:** As a developer, I want strategy confidence to grow over time as more supporting evidence appears, so that long-running patterns become more trusted.

#### Acceptance Criteria

1. WHEN a new observation matches the state-action-outcome pattern of an existing strategy, THE Insight_System SHALL increase that strategy's confidence score by an amount equal to 1 divided by the total number of observations recorded.
2. WHEN a new observation shares the state-action pattern of an existing strategy but produces a different reward outcome, THE Insight_System SHALL decrease that strategy's confidence score by recalculating it as the strategy's observation count divided by the new total number of observations.
3. IF a strategy's confidence score drops below the configured confidence_threshold (default 0.3), THEN THE Insight_System SHALL remove the strategy from `insights_cache.json` within the same observation cycle.
4. WHEN a strategy receives a new supporting observation, THE Insight_System SHALL update that strategy's last_confirmed field to the current episode number.
5. THE Insight_System SHALL constrain all strategy confidence scores to the range 0.0 to 1.0 inclusive.

### Requirement 15: Test Coverage

**User Story:** As a developer, I want automated tests validating the insight hierarchy behaviour, so that regressions are caught early.

#### Acceptance Criteria

1. THE test suite SHALL include a test verifying that a raw observation referencing an exact individual card by name is rejected from promotion to `insights_cache.json`.
2. THE test suite SHALL include a test verifying that when two candidate strategies share the same category, overlapping tags, and equivalent strategic meaning, they are merged into a single entry whose confidence is strictly greater than either original and whose evidence_count equals the sum of both originals.
3. THE test suite SHALL include a test verifying that a repeated pattern whose evidence_count meets or exceeds the system's configured promotion threshold and passes the quality gate (Requirement 11) is promoted to a strategic insight in `insights_cache.json`.
4. THE test suite SHALL include a test verifying that a pattern whose evidence_count is below the system's configured promotion threshold is not written to `insights_cache.json`.
5. THE test suite SHALL include a test verifying that a surprising_pattern candidate observed in fewer than 3 distinct comparable game states is rejected from promotion regardless of its evidence_count.
6. THE test suite SHALL include a test verifying that after new supporting evidence is added for an existing strategy, its confidence score is strictly greater than it was before the evidence was added.
7. THE test suite SHALL include a test verifying that after contradictory evidence is added for an existing strategy, its confidence score decreases, and that once its confidence drops below the system's configured minimum threshold the strategy is removed from `insights_cache.json`.
8. THE test suite SHALL include a test verifying that a strategic insight persisted to `insights_cache.json` contains all of the following fields with correct types: strategy (string), category (string), tags (list of strings), confidence (float between 0.0 and 1.0 inclusive), evidence_count (integer ≥ 1), why (string), first_seen (integer episode ≥ 0), last_confirmed (integer episode ≥ first_seen), new (boolean), and that loading the file reconstructs an equivalent data structure.
