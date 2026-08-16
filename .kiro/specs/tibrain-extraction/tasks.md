# Implementation Plan: TIBRAIN Extraction

## Overview

Extract domain-neutral reinforcement learning components from the Telecom-Native-Intelligence monorepo into a self-contained `tibrain` pip-installable package, then refactor the Wist agents to consume TIBRAIN through protocol-based interfaces. Implementation follows an incremental bottom-up approach: core data structures first, then engines, then integration.

## Tasks

- [x] 1. Create TIBRAIN package structure
  - [x] 1.1 Create `tibrain/pyproject.toml` with package metadata
    - Package name `tibrain`, Python >=3.12, numpy as sole runtime dependency
    - Add `hypothesis` as optional test dependency
    - _Requirements: 1.1_
  - [x] 1.2 Create `tibrain/__init__.py` with protocol definitions
    - Define `State = Hashable`, `Action = Hashable` type aliases
    - Define `Environment` protocol with `reset()`, `observe()`, `get_legal_actions()`, `step()` methods
    - Define `StateEncoder` and `ActionEncoder` protocols
    - Re-export all public API symbols
    - _Requirements: 1.2, 1.3, 2.1, 2.2, 2.3, 2.4_
  - [x] 1.3 Create `tibrain/discovery/__init__.py` sub-package
    - Empty init that will re-export DiscoveryEngine and Pattern
    - _Requirements: 1.3_

- [x] 2. Implement Q-Table data structure
  - [x] 2.1 Create `tibrain/q_table.py`
    - Implement `QTable` class with `get()`, `set()`, `get_best_action()`, `size`, `to_dict()`, `from_dict()`
    - Return 0.0 for unvisited state-action pairs
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  - [ ]* 2.2 Write property test for Q-Table default and retrieval
    - **Property 2: Q-Table Default and Retrieval**
    - **Validates: Requirements 5.2, 5.3**
  - [ ]* 2.3 Write property test for Q-Table size invariant
    - **Property 3: Q-Table Size Invariant**
    - **Validates: Requirements 5.4**
  - [ ]* 2.4 Write property test for Q-Table serialization round-trip
    - **Property 4: Q-Table Serialization Round-Trip**
    - **Validates: Requirements 5.5**

- [x] 3. Implement Q-Learning Engine
  - [x] 3.1 Create `tibrain/q_learning.py`
    - Implement `QLearningEngine` with Double Q-learning (two QTables)
    - Implement TD(λ) with eligibility traces
    - Implement `get_values()`, `td_update()`, `reset_episode()` methods
    - Configurable `alpha`, `gamma`, `lambda_trace` hyperparameters
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_
  - [ ]* 3.2 Write property test for TD(λ) update correctness
    - **Property 5: TD(λ) Update Correctness**
    - **Validates: Requirements 4.4, 4.5**
  - [ ]* 3.3 Write property test for Double Q-learning reduced overestimation
    - **Property 6: Double Q-Learning Reduced Overestimation**
    - **Validates: Requirements 4.1**

- [x] 4. Implement Policy module
  - [x] 4.1 Create `tibrain/policy.py`
    - Implement epsilon-greedy action selection
    - Implement UCB-inspired exploration bonus (`sqrt(log(total_visits) / action_visits)`)
    - Implement `select()`, `select_greedy()`, `decay()` methods
    - Enforce minimum epsilon floor
    - Handle unvisited actions (uniform random selection)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  - [ ]* 4.2 Write property test for epsilon decay with floor
    - **Property 7: Epsilon Decay with Floor**
    - **Validates: Requirements 6.3, 6.4**

- [x] 5. Implement Replay Buffer
  - [x] 5.1 Create `tibrain/replay_buffer.py`
    - Implement ring buffer with configurable capacity
    - Implement prioritized sampling proportional to `|td_error| + epsilon`
    - Implement `add()`, `sample()` methods
    - Return all entries when buffer is smaller than batch_size
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  - [ ]* 5.2 Write property test for replay buffer capacity invariant
    - **Property 9: Replay Buffer Capacity Invariant**
    - **Validates: Requirements 7.2**
  - [ ]* 5.3 Write property test for prioritized sampling distribution
    - **Property 10: Prioritized Sampling Distribution**
    - **Validates: Requirements 7.3**

- [x] 6. Checkpoint - Core data structures complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement Neural Network module
  - [x] 7.1 Create `tibrain/neural_net.py` with Evaluator class
    - Implement feedforward network with self-attention (residual)
    - Configurable `input_size`, `hidden_size`, `learning_rate`, `n_heads`
    - Implement `predict()`, `predict_batch()`, `update()` with batch accumulation
    - Implement gradient clipping with configurable `max_grad_norm`
    - Implement `copy()` for target network copies
    - Implement `to_dict()` and `from_dict()` for weight serialization
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.9_
  - [x] 7.2 Add QNetwork class to `tibrain/neural_net.py`
    - Implement fixed-output feedforward network for small action spaces
    - Implement `predict()` returning ndarray of Q-values and `update()` returning squared error
    - Implement `to_dict()` and `from_dict()`
    - _Requirements: 8.7, 8.8_
  - [ ]* 7.3 Write property test for Evaluator serialization round-trip
    - **Property 11: Evaluator Serialization Round-Trip**
    - **Validates: Requirements 8.6**
  - [ ]* 7.4 Write property test for Evaluator copy independence
    - **Property 12: Evaluator Copy Independence**
    - **Validates: Requirements 8.9**
  - [ ]* 7.5 Write property test for QNetwork output dimension
    - **Property 13: QNetwork Output Dimension**
    - **Validates: Requirements 8.7**
  - [ ]* 7.6 Write property test for QNetwork gradient descent
    - **Property 14: QNetwork Gradient Descent**
    - **Validates: Requirements 8.8**
  - [ ]* 7.7 Write property test for Evaluator batch consistency
    - **Property 24: Evaluator Batch Consistency**
    - **Validates: Requirements 8.3**

- [x] 8. Implement MCTS Engine
  - [x] 8.1 Create `tibrain/mcts.py`
    - Implement `MCTSEngine` with configurable `simulate_fn` callable
    - Implement `choose_action()` returning highest average simulated reward action
    - Implement `evaluate_actions()` returning normalized [0, 1] scores
    - Return immediately for single legal action (no simulation)
    - Implement `_rollout()` with random policy to terminal state
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_
  - [ ]* 8.2 Write property test for MCTS normalized scores range
    - **Property 15: MCTS Normalized Scores Range**
    - **Validates: Requirements 9.3**

- [x] 9. Implement Reward Normalization and Curiosity
  - [x] 9.1 Create `tibrain/reward.py` with RewardNormalizer and CuriosityModule
    - Implement Welford's online algorithm for running mean/variance
    - Implement `normalize(reward)` returning `(reward - mean) / std`
    - Implement CuriosityModule with visit counts and `bonus()` proportional to `1/sqrt(count)`
    - Configurable `scale` parameter for curiosity
    - Implement `to_dict()` and `from_dict()` for RewardNormalizer
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_
  - [ ]* 9.2 Write property test for reward normalizer online statistics
    - **Property 16: Reward Normalizer Online Statistics**
    - **Validates: Requirements 10.1, 10.2**
  - [ ]* 9.3 Write property test for curiosity bonus formula
    - **Property 17: Curiosity Bonus Formula**
    - **Validates: Requirements 10.3**

- [x] 10. Implement Evaluation module
  - [x] 10.1 Create `tibrain/evaluation.py` with EloTracker and MetaLearner
    - EloTracker: initial 1000, K=32, standard Elo formula
    - EloTracker: record (episode, elo) snapshots, retain last 100
    - MetaLearner: sliding window score tracking
    - MetaLearner: suggest epsilon reduction (×0.95) when performance improves >10%
    - MetaLearner: suggest epsilon increase (×1.1, max 0.3) when performance declines >20%
    - Implement `to_dict()` and `from_dict()` for EloTracker
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_
  - [ ]* 10.2 Write property test for Elo formula correctness
    - **Property 19: Elo Formula Correctness**
    - **Validates: Requirements 13.1**
  - [ ]* 10.3 Write property test for Meta-Learner adjustment conditions
    - **Property 20: Meta-Learner Adjustment Conditions**
    - **Validates: Requirements 13.4, 13.5**

- [x] 11. Implement Discovery Engine and Pattern
  - [x] 11.1 Create `tibrain/discovery/pattern.py`
    - Implement `Pattern` dataclass with state_pattern, action_pattern, reward_outcome, confidence, observations
    - _Requirements: 14.2_
  - [x] 11.2 Create `tibrain/discovery/discovery_engine.py`
    - Implement `DiscoveryEngine` with pattern registry and confidence scoring
    - Implement `observe()`, `detect_patterns()` methods
    - Remove patterns when confidence drops below threshold
    - Implement `to_dict()` and `from_dict()` for persistence
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_
  - [ ]* 11.3 Write property test for discovery engine pattern threshold
    - **Property 21: Discovery Engine Pattern Threshold**
    - **Validates: Requirements 14.3**
  - [ ]* 11.4 Write property test for discovery engine serialization round-trip
    - **Property 22: Discovery Engine Serialization Round-Trip**
    - **Validates: Requirements 14.4**

- [x] 12. Checkpoint - All TIBRAIN engine modules complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Implement Generic Agent
  - [x] 13.1 Create `tibrain/agent.py`
    - Implement `Agent` class with `choose_action()` and `learn()` methods
    - Accept `state_encoder` and `action_encoder` callables (default to `str()`)
    - Wire up QLearningEngine, Policy, ReplayBuffer, optional Evaluator
    - Implement `training` flag to disable exploration and learning
    - Implement `reset_episode()` to clear eligibility traces
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  - [ ]* 13.2 Write property test for training mode disables exploration and learning
    - **Property 8: Training Mode Disables Exploration and Learning**
    - **Validates: Requirements 3.4**

- [x] 14. Implement Persistence module
  - [x] 14.1 Create `tibrain/persistence.py`
    - Implement `save(agent, path)` serializing all state to JSON
    - Implement `load(path)` returning dict (empty dict if path missing)
    - Support incremental saves via `changed_components` parameter
    - Serialize Q-tables, neural net weights, replay buffer, reward normalizer, discovery data
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_
  - [ ]* 14.2 Write property test for persistence round-trip
    - **Property 18: Persistence Round-Trip**
    - **Validates: Requirements 11.1, 11.2, 11.3**

- [x] 15. Implement Training Loop
  - [x] 15.1 Create `tibrain/training.py`
    - Implement `TrainingPhase` dataclass and `TrainingResult` dataclass
    - Implement `train(agent, environment, episodes, ...)` function
    - Support curriculum via `phases` parameter with per-phase hyperparameters
    - Implement episode loop: reset → observe → act → step → learn
    - Invoke `on_progress` callback every `report_every` episodes
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_
  - [ ]* 15.2 Write property test for training progress callback frequency
    - **Property 25: Training Progress Callback Frequency**
    - **Validates: Requirements 12.5**

- [x] 16. Checkpoint - Full TIBRAIN library complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 17. Write unit tests for TIBRAIN components
  - [x] 17.1 Create `tibrain/tests/test_q_table.py`
    - Test get/set basic operations, default value, get_best_action, size property
    - Test edge cases: empty table, single entry, duplicate keys
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  - [x] 17.2 Create `tibrain/tests/test_q_learning.py`
    - Test td_update produces correct Q-value changes
    - Test reset_episode clears traces
    - Test Double Q random table selection
    - _Requirements: 4.1, 4.2, 4.4, 4.6_
  - [x] 17.3 Create `tibrain/tests/test_policy.py`
    - Test epsilon-greedy selects random with correct probability
    - Test UCB bonus computation
    - Test decay respects minimum epsilon
    - Test unvisited actions selected randomly
    - _Requirements: 6.1, 6.2, 6.3, 6.5_
  - [x] 17.4 Create `tibrain/tests/test_replay_buffer.py`
    - Test add/sample basics, capacity eviction, priority sampling
    - Test sample when buffer < batch_size
    - _Requirements: 7.1, 7.2, 7.4, 7.5_
  - [x] 17.5 Create `tibrain/tests/test_neural_net.py`
    - Test Evaluator predict forward pass, batch accumulation and flush
    - Test QNetwork predict shape, update reduces error
    - Test Evaluator copy produces independent weights
    - _Requirements: 8.1, 8.3, 8.4, 8.7, 8.9_
  - [x] 17.6 Create `tibrain/tests/test_mcts.py`
    - Test single action returns immediately
    - Test evaluate_actions normalizes to [0, 1]
    - Test with deterministic simulate_fn
    - _Requirements: 9.2, 9.3, 9.4_
  - [x] 17.7 Create `tibrain/tests/test_evaluation.py`
    - Test Elo update formula correctness
    - Test MetaLearner adjustment suggestions
    - Test EloTracker snapshot retention limit (100)
    - _Requirements: 13.1, 13.2, 13.4, 13.5_
  - [x] 17.8 Create `tibrain/tests/test_discovery.py`
    - Test pattern detection with observations above threshold
    - Test pattern removal when confidence drops
    - Test serialization round-trip
    - _Requirements: 14.1, 14.3, 14.4, 14.5_
  - [x] 17.9 Create `tibrain/tests/test_agent.py`
    - Test choose_action with mock encoder
    - Test learn updates Q-values
    - Test training=False disables learning
    - Test default encoders (str conversion)
    - _Requirements: 3.1, 3.2, 3.4, 3.5_
  - [x] 17.10 Create `tibrain/tests/test_training.py`
    - Test training loop with mock environment
    - Test curriculum phases apply hyperparameters
    - Test on_progress callback frequency
    - _Requirements: 12.1, 12.3, 12.4, 12.5_
  - [x] 17.11 Create `tibrain/tests/test_persistence.py`
    - Test save/load round-trip
    - Test load from nonexistent path returns empty dict
    - Test incremental save with changed_components
    - _Requirements: 11.1, 11.2, 11.4, 11.5_
  - [x] 17.12 Create `tibrain/tests/test_reward.py`
    - Test RewardNormalizer running statistics
    - Test CuriosityModule bonus formula
    - Test RewardNormalizer serialization
    - _Requirements: 10.1, 10.2, 10.3, 10.5_

- [ ] 18. Write property-based tests for correctness properties
  - [ ]* 18.1 Create `tibrain/tests/test_properties.py` with Property 1 (Domain Isolation)
    - Scan all tibrain source files for forbidden imports and domain terms
    - **Property 1: Domain Isolation**
    - **Validates: Requirements 1.4, 1.5, 16.4**
  - [ ]* 18.2 Add Property 2-4 tests (Q-Table properties) to test file
    - **Properties 2, 3, 4: Q-Table Default/Retrieval, Size Invariant, Serialization Round-Trip**
    - **Validates: Requirements 5.2, 5.3, 5.4, 5.5**
  - [ ]* 18.3 Add Property 5-6 tests (Q-Learning properties) to test file
    - **Properties 5, 6: TD(λ) Update Correctness, Double Q-Learning Reduced Overestimation**
    - **Validates: Requirements 4.1, 4.4, 4.5**
  - [ ]* 18.4 Add Property 7-8 tests (Policy and Agent properties) to test file
    - **Properties 7, 8: Epsilon Decay with Floor, Training Mode Disables Exploration**
    - **Validates: Requirements 6.3, 6.4, 3.4**
  - [ ]* 18.5 Add Property 9-10 tests (Replay Buffer properties) to test file
    - **Properties 9, 10: Capacity Invariant, Prioritized Sampling Distribution**
    - **Validates: Requirements 7.2, 7.3**
  - [ ]* 18.6 Add Property 11-14 tests (Neural Network properties) to test file
    - **Properties 11, 12, 13, 14: Evaluator Round-Trip, Copy Independence, QNetwork Dimension, QNetwork Gradient**
    - **Validates: Requirements 8.6, 8.9, 8.7, 8.8**
  - [ ]* 18.7 Add Property 15-17 tests (MCTS, Reward, Curiosity properties) to test file
    - **Properties 15, 16, 17: MCTS Normalized Scores, Reward Normalizer Stats, Curiosity Bonus**
    - **Validates: Requirements 9.3, 10.1, 10.2, 10.3**
  - [ ]* 18.8 Add Property 18-22 tests (Persistence, Evaluation, Discovery properties) to test file
    - **Properties 18, 19, 20, 21, 22: Persistence Round-Trip, Elo Formula, MetaLearner Adjustments, Discovery Threshold, Discovery Serialization**
    - **Validates: Requirements 11.1-3, 13.1, 13.4-5, 14.3, 14.4**
  - [ ]* 18.9 Add Property 24-25 tests (Evaluator batch, Training callback) to test file
    - **Properties 24, 25: Evaluator Batch Consistency, Training Progress Callback Frequency**
    - **Validates: Requirements 8.3, 12.5**

- [x] 19. Checkpoint - TIBRAIN unit and property tests complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 20. Refactor Wist LearningAgent to use TIBRAIN
  - [x] 20.1 Refactor `agents/wist_learning/learning_agent.py`
    - Import Agent, QLearningEngine, Policy, ReplayBuffer from `tibrain`
    - Remove inline Q-learning, policy, and replay buffer implementations
    - Provide Wist-specific `state_encoder` and `action_encoder` callables
    - Delegate `choose_action` and `learn` calls to TIBRAIN Agent
    - Maintain identical external API for backward compatibility
    - _Requirements: 15.1, 15.4, 15.5_

- [x] 21. Refactor Wist DiscoveryAgent to use TIBRAIN
  - [x] 21.1 Refactor `agents/wist_discovery/discovery_agent.py`
    - Import Evaluator, QNetwork, MCTSEngine, RewardNormalizer, EloTracker from `tibrain`
    - Import DiscoveryEngine from `tibrain.discovery`
    - Remove inline neural net, MCTS, reward normalization, discovery implementations
    - Provide domain-specific simulate_fn for MCTS
    - Delegate to TIBRAIN components while maintaining external API
    - _Requirements: 15.2, 15.4, 15.5_

- [x] 22. Create WistEnvironment adapter
  - [x] 22.1 Create `agents/wist_learning/wist_environment.py`
    - Implement `WistEnvironment` class conforming to TIBRAIN `Environment` protocol
    - Implement `reset()`, `observe()`, `get_legal_actions()`, `step()` methods
    - Adapt Wist game state to TIBRAIN State/Action protocol
    - _Requirements: 15.3, 2.1_

- [x] 23. Update Wist trainer to use tibrain.training
  - [x] 23.1 Refactor `agents/wist_learning/trainer.py`
    - Import `train()` and `TrainingPhase` from `tibrain.training`
    - Define 3-phase curriculum: Random → Rule-Based → Refinement
    - Wire Wist environment factory into TIBRAIN training phases
    - Implement `on_progress` callback for Wist-specific logging
    - _Requirements: 15.6, 12.4_

- [x] 24. Checkpoint - Wist refactoring complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 25. Run existing Wist tests and fix regressions
  - [x] 25.1 Run existing test suite and fix any import/API breakages
    - Execute all existing tests in the monorepo
    - Fix any broken imports due to moved code
    - Verify backward compatibility with non-learning agents
    - _Requirements: 15.5, 16.3_

- [x] 26. Update Host Project pyproject.toml for editable install
  - [x] 26.1 Update root `pyproject.toml` with TIBRAIN dependency
    - Add `tibrain` as dependency with relative path reference
    - Ensure `pip install -e .` resolves `import tibrain` to local directory
    - Verify unidirectional dependency (Host → TIBRAIN)
    - _Requirements: 16.1, 16.2, 16.4_

- [x] 27. Integration tests (TIBRAIN + Wist end-to-end)
  - [x] 27.1 Create `tests/test_integration_tibrain_wist.py`
    - Test full training loop with WistEnvironment adapter and TIBRAIN Agent
    - Test save/load cycle with all populated TIBRAIN components
    - Test behavioral equivalence: compare pre- and post-refactoring agent outputs on recorded states
    - Test curriculum training produces expected phase transitions
    - _Requirements: 15.5, 16.2, 12.4_
  - [ ]* 27.2 Write property test for behavioral equivalence
    - **Property 23: Behavioral Equivalence After Refactoring**
    - **Validates: Requirements 15.5**

- [x] 28. Final checkpoint - All integration tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The implementation language is Python (>=3.12) with numpy as the sole runtime dependency
- Property-based tests use the Hypothesis framework
- All TIBRAIN source files must remain free of domain-specific references (Property 1)

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["2.1", "11.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "2.4", "3.1", "4.1", "5.1", "11.2"] },
    { "id": 3, "tasks": ["3.2", "3.3", "4.2", "5.2", "5.3", "9.1", "10.1", "11.3", "11.4"] },
    { "id": 4, "tasks": ["7.1", "8.1", "9.2", "9.3", "10.2", "10.3"] },
    { "id": 5, "tasks": ["7.2", "7.3", "7.4", "7.5", "7.6", "7.7", "8.2"] },
    { "id": 6, "tasks": ["13.1"] },
    { "id": 7, "tasks": ["13.2", "14.1", "15.1"] },
    { "id": 8, "tasks": ["14.2", "15.2"] },
    { "id": 9, "tasks": ["17.1", "17.2", "17.3", "17.4", "17.6", "17.7", "17.8", "17.12"] },
    { "id": 10, "tasks": ["17.5", "17.9", "17.10", "17.11"] },
    { "id": 11, "tasks": ["18.1", "18.2", "18.3", "18.4", "18.5", "18.6", "18.7", "18.8", "18.9"] },
    { "id": 12, "tasks": ["20.1", "21.1", "22.1"] },
    { "id": 13, "tasks": ["23.1"] },
    { "id": 14, "tasks": ["25.1", "26.1"] },
    { "id": 15, "tasks": ["27.1", "27.2"] }
  ]
}
```
