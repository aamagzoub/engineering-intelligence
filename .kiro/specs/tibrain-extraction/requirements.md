# Requirements Document

## Introduction

TIBRAIN is a domain-neutral reinforcement learning library extracted from the Telecom-Native-Intelligence monorepo. The library encapsulates Q-learning, neural network function approximation, and Monte Carlo Tree Search into a reusable package with protocol-based interfaces. Domain-specific code (Wist card game logic) remains in the host project and consumes TIBRAIN through well-defined contracts. This spec covers the full extraction of generic RL components into a separate pip-installable package and the refactoring of Wist agents to use TIBRAIN as a dependency.

## Glossary

- **TIBRAIN**: The domain-neutral reinforcement learning package being extracted
- **Wist**: The existing Sudanese card game implementation that will consume TIBRAIN
- **Agent**: A generic learner that selects actions given states and legal actions
- **Environment**: A domain-specific system that provides observations, legal actions, and transitions
- **State**: An opaque representation of the current situation provided by the Environment
- **Action**: An opaque element from the set of legal moves at a given state
- **Reward**: A scalar signal returned by the Environment after a transition
- **Q-Table**: A mapping from (state, action) pairs to estimated value
- **Episode**: A complete sequence of steps from initial state to terminal state
- **Step**: A single state→action→reward→next_state transition
- **Policy**: A strategy for selecting actions (epsilon-greedy, UCB-inspired, adaptive)
- **Experience**: A stored tuple of (state, action, reward, next_state) for replay
- **MCTS_Engine**: The generic Monte Carlo Tree Search component
- **Evaluator**: A neural network that maps state features to a scalar Q-value
- **QNetwork**: A neural network that maps state features to Q-values for a fixed action set
- **Discovery_Engine**: A generic pattern detection system that identifies recurring structures in experience data
- **Host_Project**: The Wist monorepo that depends on TIBRAIN via editable install

## Requirements

### Requirement 1: Package Structure

**User Story:** As a developer, I want TIBRAIN to be a self-contained pip-installable package within the monorepo, so that it can be reused across projects without coupling to any domain.

#### Acceptance Criteria

1. THE TIBRAIN package SHALL provide a `pyproject.toml` at `tibrain/pyproject.toml` declaring the package name as `tibrain`, version, Python version constraint (>=3.12), and numpy as the sole runtime dependency.
2. THE TIBRAIN package SHALL expose a top-level `tibrain` module importable via `import tibrain` after installation.
3. THE TIBRAIN package SHALL contain only the modules: `agent.py`, `q_learning.py`, `q_table.py`, `policy.py`, `persistence.py`, `training.py`, `evaluation.py`, `neural_net.py`, `mcts.py`, `replay_buffer.py`, and the `discovery/` sub-package with `discovery_engine.py` and `pattern.py`.
4. THE TIBRAIN package SHALL NOT import or reference any module from the Host_Project (environments, intelligence.core, agents).
5. THE TIBRAIN package SHALL NOT contain any knowledge of cards, suits, tricks, players, Qabool, Wist rules, gNB, sector, carrier, PRB, UE, or RAN concepts.

### Requirement 2: Protocol Interfaces

**User Story:** As a domain developer, I want TIBRAIN to define protocol interfaces for Environment, State, and Action, so that I can plug any domain into the learning system with type-safety guarantees.

#### Acceptance Criteria

1. THE TIBRAIN package SHALL define an `Environment` protocol with methods: `observe() -> State`, `get_legal_actions(state: State) -> list[Action]`, and `step(action: Action) -> tuple[State, float, dict]`.
2. THE TIBRAIN package SHALL define a `State` protocol as a type alias that accepts any hashable object.
3. THE TIBRAIN package SHALL define an `Action` protocol as a type alias that accepts any hashable object.
4. THE TIBRAIN package SHALL use Python `typing.Protocol` or `abc.ABC` for all interface definitions.
5. THE TIBRAIN package SHALL enforce that domain code implements all required methods at type-check time through the protocol contracts.

### Requirement 3: Generic Agent

**User Story:** As a developer, I want a generic Agent base class that implements the core RL loop, so that domain-specific agents only need to provide encoding functions.

#### Acceptance Criteria

1. THE TIBRAIN Agent SHALL provide a `choose_action(state: State, legal_actions: list[Action]) -> Action` method that selects an action according to the current policy.
2. THE TIBRAIN Agent SHALL provide a `learn(state: State, action: Action, reward: float, next_state: State, next_legal_actions: list[Action]) -> None` method that updates internal value estimates.
3. THE TIBRAIN Agent SHALL accept a `state_encoder` callable and an `action_encoder` callable at construction time that convert domain State and Action objects to hashable string keys.
4. THE TIBRAIN Agent SHALL support a `training: bool` flag that disables exploration and learning when set to False.
5. WHEN the Agent is constructed without explicit encoders, THE TIBRAIN Agent SHALL default to using `str()` conversion on State and Action objects.

### Requirement 4: Q-Learning Engine

**User Story:** As a developer, I want a complete Q-learning engine with Double Q-learning and TD(λ), so that value estimation is accurate and sample-efficient.

#### Acceptance Criteria

1. THE TIBRAIN Q-Learning engine SHALL implement Double Q-learning with two Q-tables to reduce overestimation bias.
2. THE TIBRAIN Q-Learning engine SHALL implement TD(λ) with eligibility traces for temporal credit assignment within an episode.
3. THE TIBRAIN Q-Learning engine SHALL accept configurable hyperparameters: `alpha` (learning rate), `gamma` (discount factor), and `lambda_trace` (eligibility trace decay).
4. THE TIBRAIN Q-Learning engine SHALL provide a `td_update(state, action, reward, next_state, next_legal_actions)` method that performs a single TD step with trace propagation.
5. THE TIBRAIN Q-Learning engine SHALL decay all eligibility traces by `gamma * lambda_trace` after each update.
6. THE TIBRAIN Q-Learning engine SHALL clear eligibility traces at the start of each episode via a `reset_episode()` method.

### Requirement 5: Q-Table

**User Story:** As a developer, I want a dedicated Q-Table data structure, so that state-action values are stored efficiently with a clean interface.

#### Acceptance Criteria

1. THE TIBRAIN Q-Table SHALL store float values indexed by (state_key: str, action_key: str) pairs.
2. THE TIBRAIN Q-Table SHALL return 0.0 for unvisited state-action pairs.
3. THE TIBRAIN Q-Table SHALL provide `get(state, action) -> float`, `set(state, action, value)`, and `get_best_action(state, actions) -> Action` methods.
4. THE TIBRAIN Q-Table SHALL expose a `size` property returning the number of stored state-action pairs.
5. THE TIBRAIN Q-Table SHALL support serialization to and deserialization from a dictionary via `to_dict()` and `from_dict()` class methods.

### Requirement 6: Policy

**User Story:** As a developer, I want configurable exploration policies, so that the agent balances exploration and exploitation appropriately.

#### Acceptance Criteria

1. THE TIBRAIN Policy module SHALL implement epsilon-greedy action selection that chooses a random legal action with probability epsilon.
2. THE TIBRAIN Policy module SHALL implement UCB-inspired exploration that applies a bonus proportional to `sqrt(log(total_visits) / action_visits)`.
3. THE TIBRAIN Policy module SHALL implement adaptive epsilon decay via a `decay(factor: float)` method that multiplies epsilon by the given factor.
4. THE TIBRAIN Policy module SHALL accept a minimum epsilon bound below which decay has no effect.
5. WHEN no legal actions have been visited, THE TIBRAIN Policy module SHALL select uniformly at random.

### Requirement 7: Experience Replay Buffer

**User Story:** As a developer, I want a prioritized experience replay buffer, so that the agent learns more from surprising transitions.

#### Acceptance Criteria

1. THE TIBRAIN Replay Buffer SHALL store experience tuples of (state, action, reward, next_state, td_error).
2. THE TIBRAIN Replay Buffer SHALL accept a configurable `capacity` parameter and evict oldest entries when full.
3. THE TIBRAIN Replay Buffer SHALL sample experiences with probability proportional to `|td_error| + epsilon` where epsilon is a small constant (0.01).
4. THE TIBRAIN Replay Buffer SHALL provide an `add(state, action, reward, next_state, td_error)` method and a `sample(batch_size) -> list` method.
5. WHEN the buffer contains fewer entries than the requested batch_size, THE TIBRAIN Replay Buffer SHALL return all available entries.

### Requirement 8: Neural Network Function Approximators

**User Story:** As a developer, I want generic neural network evaluators, so that the agent generalizes across similar states without domain-specific architectures.

#### Acceptance Criteria

1. THE TIBRAIN Evaluator SHALL implement a feedforward neural network with self-attention that maps an input feature vector to a single scalar Q-value.
2. THE TIBRAIN Evaluator SHALL accept configurable `input_size`, `hidden_size`, `learning_rate`, and `n_heads` parameters at construction.
3. THE TIBRAIN Evaluator SHALL provide `predict(x: ndarray) -> float`, `predict_batch(batch: ndarray) -> ndarray`, and `update(x: ndarray, target: float)` methods.
4. THE TIBRAIN Evaluator SHALL use batch accumulation and flush training when a configurable batch size is reached.
5. THE TIBRAIN Evaluator SHALL apply gradient clipping with a configurable maximum norm to prevent exploding gradients.
6. THE TIBRAIN Evaluator SHALL provide `to_dict()` and `from_dict()` methods for weight serialization.
7. THE TIBRAIN QNetwork SHALL implement a feedforward neural network that maps an input feature vector to a fixed-size output vector of Q-values.
8. THE TIBRAIN QNetwork SHALL provide `predict(x: ndarray) -> ndarray` and `update(x: ndarray, target_idx: int, target_value: float) -> float` methods.
9. THE TIBRAIN Evaluator SHALL provide a `copy() -> Evaluator` method for creating frozen target network copies.

### Requirement 9: MCTS Engine

**User Story:** As a developer, I want a generic Monte Carlo Tree Search engine, so that any domain can use simulation-based look-ahead with legal-action enumeration.

#### Acceptance Criteria

1. THE TIBRAIN MCTS_Engine SHALL accept a `simulate_fn` callable that takes (state, action) and returns (next_state, reward, done, legal_actions).
2. THE TIBRAIN MCTS_Engine SHALL provide a `choose_action(state, legal_actions, num_simulations) -> Action` method that returns the action with highest average simulated reward.
3. THE TIBRAIN MCTS_Engine SHALL provide an `evaluate_actions(state, legal_actions, num_simulations) -> dict[Action, float]` method that returns normalized scores (0 to 1) per action.
4. WHEN only one legal action is available, THE TIBRAIN MCTS_Engine SHALL return that action without performing simulations.
5. THE TIBRAIN MCTS_Engine SHALL accept a configurable `num_simulations` parameter defaulting to 100.

### Requirement 10: Reward Normalization and Curiosity

**User Story:** As a developer, I want reward normalization and curiosity bonuses built into the library, so that learning is stable across domains with different reward scales.

#### Acceptance Criteria

1. THE TIBRAIN Reward Normalizer SHALL maintain a running mean and variance of observed rewards using Welford's online algorithm.
2. THE TIBRAIN Reward Normalizer SHALL provide a `normalize(reward: float) -> float` method returning `(reward - mean) / std`.
3. THE TIBRAIN Curiosity module SHALL maintain visit counts per state and provide a bonus proportional to `1 / sqrt(visit_count)`.
4. THE TIBRAIN Curiosity module SHALL accept a configurable `scale` parameter that multiplies the curiosity bonus.
5. THE TIBRAIN Reward Normalizer SHALL provide `to_dict()` and `from_dict()` methods for persistence.

### Requirement 11: Persistence

**User Story:** As a developer, I want to save and load all learned state (Q-tables, neural net weights, discoveries), so that training progress is not lost between sessions.

#### Acceptance Criteria

1. THE TIBRAIN Persistence module SHALL provide a `save(agent, path: Path) -> None` function that serializes all agent state to a JSON file.
2. THE TIBRAIN Persistence module SHALL provide a `load(path: Path) -> dict` function that deserializes agent state from a JSON file.
3. THE TIBRAIN Persistence module SHALL serialize Q-table data, neural network weights, replay buffer contents, reward normalizer statistics, and discovery data.
4. IF the file at the given path does not exist during load, THEN THE TIBRAIN Persistence module SHALL return an empty default state dictionary without raising an exception.
5. THE TIBRAIN Persistence module SHALL support incremental saves where only changed components are written when a `changed_components` set is provided.

### Requirement 12: Training Loop

**User Story:** As a developer, I want a generic training loop with curriculum support, so that I can train agents with progressive difficulty in any domain.

#### Acceptance Criteria

1. THE TIBRAIN Training module SHALL provide a `train(agent, environment, episodes, on_progress) -> TrainingResult` function that runs the core RL loop for the specified number of episodes.
2. THE TIBRAIN Training module SHALL call `environment.reset() -> State` at the start of each episode to obtain the initial state.
3. THE TIBRAIN Training module SHALL execute the loop: observe state, get legal actions, choose action, step environment, learn from transition, until the episode terminates.
4. THE TIBRAIN Training module SHALL support curriculum learning via a `phases: list[TrainingPhase]` parameter where each phase specifies episodes, hyperparameters, and an optional environment factory.
5. THE TIBRAIN Training module SHALL invoke the `on_progress` callback every `report_every` episodes with current metrics (episode count, win rate, epsilon, q_table_size).
6. THE TIBRAIN Training module SHALL return a `TrainingResult` dataclass containing episodes completed, cumulative rewards, and per-phase metrics.

### Requirement 13: Evaluation

**User Story:** As a developer, I want built-in Elo tracking and meta-learning, so that agent improvement is measurable and hyperparameters adapt automatically.

#### Acceptance Criteria

1. THE TIBRAIN Elo Tracker SHALL maintain a numeric Elo rating starting at 1000 and update it after each episode using standard Elo formula with K-factor of 32.
2. THE TIBRAIN Elo Tracker SHALL record (episode, elo) snapshots at configurable intervals and retain the last 100 snapshots.
3. THE TIBRAIN Meta-Learner SHALL track recent scores in a sliding window and suggest hyperparameter adjustments when performance trends are detected.
4. WHEN recent performance improves by more than 10% over the window average, THE TIBRAIN Meta-Learner SHALL suggest reducing epsilon by 5%.
5. WHEN recent performance declines by more than 20% below the window average, THE TIBRAIN Meta-Learner SHALL suggest increasing epsilon by 10% up to a maximum of 0.3.

### Requirement 14: Discovery Engine

**User Story:** As a developer, I want a generic pattern detection engine, so that the agent can identify recurring structures in its experience without domain knowledge.

#### Acceptance Criteria

1. THE TIBRAIN Discovery_Engine SHALL accept experience sequences and detect recurring (state_pattern, action_pattern, reward_outcome) triples.
2. THE TIBRAIN Discovery_Engine SHALL maintain a pattern registry with confidence scores based on frequency of observation.
3. THE TIBRAIN Discovery_Engine SHALL provide a `detect_patterns(experiences: list) -> list[Pattern]` method that returns patterns exceeding a configurable confidence threshold.
4. THE TIBRAIN Discovery_Engine SHALL provide `to_dict()` and `from_dict()` methods for persisting discovered patterns.
5. WHEN a pattern's confidence score drops below the threshold over time, THE TIBRAIN Discovery_Engine SHALL remove the pattern from the active registry.

### Requirement 15: Wist Agent Refactoring

**User Story:** As a developer, I want the existing Wist learning agents refactored to consume TIBRAIN, so that domain-specific code is separated from the generic RL engine.

#### Acceptance Criteria

1. WHEN refactored, THE Wist LearningAgent SHALL import Q-learning, policy, and replay buffer from the `tibrain` package instead of implementing them inline.
2. WHEN refactored, THE Wist DiscoveryAgent SHALL import the Evaluator, QNetwork, MCTS_Engine, reward normalizer, and Elo tracker from the `tibrain` package.
3. THE Wist agents SHALL implement a `WistEnvironment` adapter that conforms to the TIBRAIN Environment protocol, providing `observe()`, `get_legal_actions()`, and `step()` methods.
4. THE Wist agents SHALL provide domain-specific `state_encoder` and `action_encoder` callables that convert Wist observations and actions to string keys.
5. AFTER refactoring, THE Wist agents SHALL produce identical behavior (same action selection given the same random seed and Q-table state) as the pre-refactoring implementation.
6. AFTER refactoring, THE Wist trainer SHALL use `tibrain.training.train()` with a Wist-specific environment factory and curriculum phases matching the existing 3-phase curriculum (Random → Rule-Based → Refinement).

### Requirement 16: Host Project Integration

**User Story:** As a developer, I want the Host_Project to reference TIBRAIN via editable install, so that development of both packages proceeds without reinstallation friction.

#### Acceptance Criteria

1. THE Host_Project `pyproject.toml` SHALL declare TIBRAIN as a dependency using a relative path reference (e.g., `tibrain @ file:///./tibrain`).
2. WHEN installed in development mode, THE Host_Project SHALL resolve `import tibrain` to the local `tibrain/` directory in the monorepo.
3. THE Host_Project SHALL retain its existing `intelligence/core/` ABCs (Agent, Action, Observation, Environment) for backward compatibility with non-learning agents.
4. THE TIBRAIN package SHALL NOT depend on the Host_Project; the dependency is unidirectional (Host → TIBRAIN).
