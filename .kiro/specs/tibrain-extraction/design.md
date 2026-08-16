# Design Document: TIBRAIN Extraction

## Overview

TIBRAIN is a domain-neutral reinforcement learning library extracted from the Telecom-Native-Intelligence monorepo. It encapsulates Q-learning (Double Q, TD(λ)), neural network function approximation (Evaluator with self-attention, QNetwork), Monte Carlo Tree Search, and pattern discovery into a reusable, pip-installable package. Domain-specific code remains in the host project and consumes TIBRAIN through protocol-based interfaces.

The architecture enforces a **unidirectional dependency** (Host → TIBRAIN) with protocol contracts at the boundary.

## Architecture

### Component Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     Host Project (Wist)                           │
│                                                                  │
│  ┌────────────────┐   ┌─────────────────┐   ┌───────────────┐   │
│  │ WistEnvironment│   │ WistLearning    │   │ WistDiscovery │   │
│  │ (adapter)      │   │ Agent (thin)    │   │ Agent (thin)  │   │
│  └───────┬────────┘   └───────┬─────────┘   └──────┬────────┘   │
│          │                    │                     │            │
└──────────┼────────────────────┼─────────────────────┼────────────┘
           │                    │                     │
           ▼                    ▼                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                        TIBRAIN Package                            │
│                                                                  │
│  ┌───────────┐  ┌────────────┐  ┌──────────┐  ┌─────────────┐   │
│  │ Protocols │  │   Agent    │  │ Training │  │ Persistence │   │
│  │ (env,     │  │ (generic   │  │ (loop,   │  │ (save/load) │   │
│  │  state,   │  │  RL agent) │  │  curric) │  │             │   │
│  │  action)  │  │            │  │          │  │             │   │
│  └───────────┘  └─────┬──────┘  └──────────┘  └─────────────┘   │
│                        │                                         │
│         ┌──────────────┼──────────────┐                          │
│         ▼              ▼              ▼                           │
│  ┌────────────┐  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
│  │ Q-Learning │  │  Policy  │  │ Neural Net│  │    MCTS      │  │
│  │ (Double Q, │  │ (ε-greed │  │ (Evaluator│  │   Engine     │  │
│  │  TD(λ))   │  │  UCB,    │  │  QNetwork)│  │              │  │
│  │            │  │  decay)  │  │           │  │              │  │
│  └─────┬──────┘  └──────────┘  └───────────┘  └──────────────┘  │
│        │                                                         │
│  ┌─────┴──────┐  ┌──────────────┐  ┌─────────────────────────┐  │
│  │  Q-Table   │  │Replay Buffer │  │   Evaluation            │  │
│  │            │  │(prioritized) │  │ (Elo, MetaLearner)      │  │
│  └────────────┘  └──────────────┘  └─────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────┐                  │
│  │        Discovery (sub-package)             │                  │
│  │  ┌──────────────────┐  ┌───────────────┐  │                  │
│  │  │ DiscoveryEngine  │  │    Pattern    │  │                  │
│  │  └──────────────────┘  └───────────────┘  │                  │
│  └────────────────────────────────────────────┘                  │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Host Environment                    TIBRAIN Agent
─────────────────                   ─────────────
                                    
env.reset()  ──────────────────────→ state₀
                                     │
env.get_legal_actions(state) ──────→ legal_actions
                                     │
                              ┌──────┘
                              │ choose_action(state, legal_actions)
                              │   → policy.select(q_values, legal, epsilon)
                              │   → action
                              └──────┐
env.step(action) ←─────────────────── action
  │
  ├──→ (next_state, reward, info)
  │
  └──→ agent.learn(state, action, reward, next_state, next_legal)
          │
          ├─→ q_learning.td_update(...)
          ├─→ replay_buffer.add(...)
          ├─→ evaluator.update(...)  (if neural mode)
          └─→ discovery.observe(...)
```

## Module Design

### Package Layout

```
tibrain/
├── pyproject.toml          # Package metadata, numpy dependency
├── __init__.py             # Public API re-exports
├── agent.py                # Generic Agent class
├── q_learning.py           # Double Q-learning + TD(λ) engine
├── q_table.py              # Q-Table data structure
├── policy.py               # Exploration policies (ε-greedy, UCB, adaptive)
├── persistence.py          # Save/load agent state to JSON
├── training.py             # Generic training loop with curriculum
├── evaluation.py           # Elo tracker + Meta-learner
├── neural_net.py           # Evaluator (attention), QNetwork
├── mcts.py                 # Generic MCTS engine
├── replay_buffer.py        # Prioritized experience replay
└── discovery/
    ├── __init__.py
    ├── discovery_engine.py # Pattern detection engine
    └── pattern.py          # Pattern data class
```

## Components and Interfaces

### Core Type Definitions (`__init__.py`)

```python
from typing import Protocol, Any, Hashable, TypeVar, runtime_checkable

# State and Action are any hashable object — domain provides concrete types
State = Hashable
Action = Hashable


@runtime_checkable
class Environment(Protocol):
    """Protocol for any domain environment that TIBRAIN can learn in."""

    def reset(self) -> State:
        """Reset and return initial state."""
        ...

    def observe(self) -> State:
        """Return current observable state."""
        ...

    def get_legal_actions(self, state: State) -> list[Action]:
        """Return legal actions from the given state."""
        ...

    def step(self, action: Action) -> tuple[State, float, dict]:
        """Execute action, return (next_state, reward, info)."""
        ...


@runtime_checkable
class StateEncoder(Protocol):
    """Converts domain State to a hashable string key for Q-tables."""

    def __call__(self, state: State) -> str: ...


@runtime_checkable
class ActionEncoder(Protocol):
    """Converts domain Action to a hashable string key for Q-tables."""

    def __call__(self, action: Action) -> str: ...
```

### Generic Agent (`agent.py`)

```python
from __future__ import annotations
from typing import Callable
from tibrain import State, Action, StateEncoder, ActionEncoder
from tibrain.q_learning import QLearningEngine
from tibrain.policy import Policy
from tibrain.replay_buffer import ReplayBuffer
from tibrain.neural_net import Evaluator


class Agent:
    """Generic RL agent that works with any domain via encoders."""

    def __init__(
        self,
        state_encoder: StateEncoder | None = None,
        action_encoder: ActionEncoder | None = None,
        *,
        alpha: float = 0.1,
        gamma: float = 0.95,
        lambda_trace: float = 0.7,
        epsilon: float = 0.3,
        epsilon_min: float = 0.01,
        training: bool = True,
        use_neural: bool = False,
        neural_config: dict | None = None,
        replay_capacity: int = 10000,
    ) -> None:
        self.state_encoder: StateEncoder = state_encoder or str
        self.action_encoder: ActionEncoder = action_encoder or str
        self.training = training

        self.q_engine = QLearningEngine(
            alpha=alpha, gamma=gamma, lambda_trace=lambda_trace
        )
        self.policy = Policy(
            epsilon=epsilon, epsilon_min=epsilon_min
        )
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

        self._evaluator: Evaluator | None = None
        if use_neural and neural_config:
            self._evaluator = Evaluator(**neural_config)

    def choose_action(self, state: State, legal_actions: list[Action]) -> Action:
        """Select action according to current policy and Q-values."""
        if not legal_actions:
            raise ValueError("No legal actions available")

        state_key = self.state_encoder(state)
        action_keys = [self.action_encoder(a) for a in legal_actions]

        if self.training:
            chosen_key = self.policy.select(
                self.q_engine.get_values(state_key, action_keys),
                action_keys,
            )
        else:
            # Greedy: pick highest Q-value action
            chosen_key = self.policy.select_greedy(
                self.q_engine.get_values(state_key, action_keys),
                action_keys,
            )

        idx = action_keys.index(chosen_key)
        return legal_actions[idx]

    def learn(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        next_legal_actions: list[Action],
    ) -> None:
        """Update value estimates from a transition."""
        if not self.training:
            return

        state_key = self.state_encoder(state)
        action_key = self.action_encoder(action)
        next_state_key = self.state_encoder(next_state)
        next_action_keys = [self.action_encoder(a) for a in next_legal_actions]

        td_error = self.q_engine.td_update(
            state_key, action_key, reward, next_state_key, next_action_keys
        )

        self.replay_buffer.add(
            state_key, action_key, reward, next_state_key, td_error
        )

    def reset_episode(self) -> None:
        """Clear eligibility traces for a new episode."""
        self.q_engine.reset_episode()
```

### Q-Learning Engine (`q_learning.py`)

```python
from __future__ import annotations
import random
from tibrain.q_table import QTable


class QLearningEngine:
    """Double Q-learning with TD(λ) eligibility traces."""

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        lambda_trace: float = 0.7,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_trace = lambda_trace

        self.q1 = QTable()
        self.q2 = QTable()
        self._traces: dict[tuple[str, str], float] = {}

    def get_values(self, state_key: str, action_keys: list[str]) -> dict[str, float]:
        """Get combined Q-values (average of Q1 and Q2) for given actions."""
        return {
            a: (self.q1.get(state_key, a) + self.q2.get(state_key, a)) / 2.0
            for a in action_keys
        }

    def td_update(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        next_actions: list[str],
    ) -> float:
        """
        Perform a Double Q-learning TD(λ) update.

        Returns the TD error for prioritized replay.
        """
        # Randomly choose which table to update (Double Q-learning)
        if random.random() < 0.5:
            q_update, q_eval = self.q1, self.q2
        else:
            q_update, q_eval = self.q2, self.q1

        # Compute max next Q using Double Q logic
        if next_actions:
            best_next_action = q_update.get_best_action(next_state, next_actions)
            max_next_q = q_eval.get(next_state, best_next_action)
        else:
            max_next_q = 0.0

        # TD error
        current_q = q_update.get(state, action)
        td_error = reward + self.gamma * max_next_q - current_q

        # Update eligibility trace for current (state, action)
        sa = (state, action)
        self._traces[sa] = self._traces.get(sa, 0.0) + 1.0

        # Propagate update through all active traces
        to_remove = []
        for (s, a), trace in self._traces.items():
            old_val = q_update.get(s, a)
            q_update.set(s, a, old_val + self.alpha * td_error * trace)
            # Decay trace
            self._traces[(s, a)] = self.gamma * self.lambda_trace * trace
            if self._traces[(s, a)] < 0.01:
                to_remove.append((s, a))

        for key in to_remove:
            del self._traces[key]

        return td_error

    def reset_episode(self) -> None:
        """Clear all eligibility traces for a new episode."""
        self._traces.clear()
```

### Q-Table (`q_table.py`)

```python
from __future__ import annotations


class QTable:
    """Stores Q-values indexed by (state_key, action_key) pairs."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, float]] = {}

    def get(self, state: str, action: str) -> float:
        """Return Q-value, defaulting to 0.0 for unvisited pairs."""
        return self._data.get(state, {}).get(action, 0.0)

    def set(self, state: str, action: str, value: float) -> None:
        """Set Q-value for a state-action pair."""
        if state not in self._data:
            self._data[state] = {}
        self._data[state][action] = value

    def get_best_action(self, state: str, actions: list[str]) -> str:
        """Return the action with highest Q-value among given actions."""
        best_action = actions[0]
        best_value = self.get(state, actions[0])
        for a in actions[1:]:
            v = self.get(state, a)
            if v > best_value:
                best_value = v
                best_action = a
        return best_action

    @property
    def size(self) -> int:
        """Number of stored state-action pairs."""
        return sum(len(actions) for actions in self._data.values())

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {state: dict(actions) for state, actions in self._data.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "QTable":
        """Deserialize from a plain dictionary."""
        table = cls()
        table._data = {state: dict(actions) for state, actions in data.items()}
        return table
```

### Policy (`policy.py`)

```python
from __future__ import annotations
import math
import random


class Policy:
    """Exploration policies: epsilon-greedy with UCB bonus and adaptive decay."""

    def __init__(
        self,
        epsilon: float = 0.3,
        epsilon_min: float = 0.01,
    ) -> None:
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self._visit_counts: dict[str, int] = {}
        self._total_visits: int = 0

    def select(
        self,
        q_values: dict[str, float],
        action_keys: list[str],
    ) -> str:
        """Epsilon-greedy selection with UCB bonus."""
        if not action_keys:
            raise ValueError("No actions to select from")

        # Epsilon-greedy: random with probability epsilon
        if random.random() < self.epsilon:
            chosen = random.choice(action_keys)
        else:
            chosen = self._select_with_ucb(q_values, action_keys)

        # Record visit
        self._visit_counts[chosen] = self._visit_counts.get(chosen, 0) + 1
        self._total_visits += 1
        return chosen

    def select_greedy(
        self,
        q_values: dict[str, float],
        action_keys: list[str],
    ) -> str:
        """Pure greedy selection (no exploration)."""
        if not action_keys:
            raise ValueError("No actions to select from")
        return max(action_keys, key=lambda a: q_values.get(a, 0.0))

    def _select_with_ucb(
        self,
        q_values: dict[str, float],
        action_keys: list[str],
    ) -> str:
        """Select action with UCB bonus for exploration."""
        if self._total_visits == 0:
            return random.choice(action_keys)

        best_action = action_keys[0]
        best_score = float("-inf")

        for a in action_keys:
            q = q_values.get(a, 0.0)
            visits = self._visit_counts.get(a, 0)
            if visits == 0:
                # Unvisited actions get maximum priority
                return a
            ucb_bonus = math.sqrt(math.log(self._total_visits + 1) / visits)
            score = q + 0.15 * ucb_bonus
            if score > best_score:
                best_score = score
                best_action = a

        return best_action

    def decay(self, factor: float) -> None:
        """Multiply epsilon by factor, respecting minimum bound."""
        self.epsilon = max(self.epsilon_min, self.epsilon * factor)
```

### Replay Buffer (`replay_buffer.py`)

```python
from __future__ import annotations
import random
import numpy as np


class ReplayBuffer:
    """Prioritized experience replay buffer."""

    def __init__(self, capacity: int = 10000, priority_epsilon: float = 0.01) -> None:
        self._capacity = capacity
        self._priority_epsilon = priority_epsilon
        self._buffer: list[tuple[str, str, float, str, float]] = []
        self._pos: int = 0

    def add(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        td_error: float,
    ) -> None:
        """Add an experience with priority based on TD error."""
        entry = (state, action, reward, next_state, td_error)
        if len(self._buffer) < self._capacity:
            self._buffer.append(entry)
        else:
            self._buffer[self._pos] = entry
        self._pos = (self._pos + 1) % self._capacity

    def sample(self, batch_size: int) -> list[tuple[str, str, float, str]]:
        """Sample experiences proportional to |td_error| + epsilon."""
        if not self._buffer:
            return []

        actual_size = min(batch_size, len(self._buffer))
        priorities = np.array(
            [abs(entry[4]) + self._priority_epsilon for entry in self._buffer],
            dtype=np.float64,
        )
        probs = priorities / priorities.sum()

        indices = np.random.choice(
            len(self._buffer), size=actual_size, replace=False, p=probs
        )
        return [
            (self._buffer[i][0], self._buffer[i][1],
             self._buffer[i][2], self._buffer[i][3])
            for i in indices
        ]

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def capacity(self) -> int:
        return self._capacity
```

### Neural Network (`neural_net.py`)

```python
from __future__ import annotations
import numpy as np


class Evaluator:
    """
    Feedforward neural network with self-attention.
    Maps input feature vector → single scalar Q-value.
    """

    def __init__(
        self,
        input_size: int = 138,
        hidden_size: int = 256,
        learning_rate: float = 0.001,
        n_heads: int = 4,
        batch_size: int = 32,
        max_grad_norm: float = 5.0,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.n_heads = n_heads
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        # Self-attention weights
        sa = np.sqrt(2.0 / input_size)
        self.w_attn = np.random.randn(input_size, n_heads) * sa
        self.b_attn = np.zeros(n_heads)
        self.w_val = np.random.randn(input_size, input_size) * sa
        self.b_val = np.zeros(input_size)

        # Feedforward layers (3 hidden + 1 output)
        s1 = np.sqrt(2.0 / input_size)
        s2 = np.sqrt(2.0 / hidden_size)
        self.w1 = np.random.randn(input_size, hidden_size) * s1
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, hidden_size) * s2
        self.b2 = np.zeros(hidden_size)
        self.w3 = np.random.randn(hidden_size, hidden_size) * s2
        self.b3 = np.zeros(hidden_size)
        self.w4 = np.random.randn(hidden_size, 1) * s2
        self.b4 = np.zeros(1)

        # Batch accumulation
        self._batch_x: list[np.ndarray] = []
        self._batch_y: list[float] = []

    def predict(self, x: np.ndarray) -> float:
        """Forward pass → single Q-value."""
        x_att = self._attend(x)
        h1 = np.maximum(0, x_att @ self.w1 + self.b1)
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)
        h3 = np.maximum(0, h2 @ self.w3 + self.b3)
        return float((h3 @ self.w4 + self.b4)[0])

    def predict_batch(self, batch: np.ndarray) -> np.ndarray:
        """Evaluate multiple feature vectors at once."""
        attended = np.array([self._attend(x) for x in batch])
        h1 = np.maximum(0, attended @ self.w1 + self.b1)
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)
        h3 = np.maximum(0, h2 @ self.w3 + self.b3)
        return (h3 @ self.w4 + self.b4).flatten()

    def update(self, x: np.ndarray, target: float) -> None:
        """Accumulate sample; flush training when batch is full."""
        self._batch_x.append(x.copy())
        self._batch_y.append(target)
        if len(self._batch_x) >= self.batch_size:
            self._flush_batch()

    def copy(self) -> "Evaluator":
        """Create a frozen copy (for target networks)."""
        clone = Evaluator(
            self.input_size, self.hidden_size, self.lr,
            self.n_heads, self.batch_size, self.max_grad_norm,
        )
        for attr in ("w_attn", "b_attn", "w_val", "b_val",
                     "w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4"):
            setattr(clone, attr, getattr(self, attr).copy())
        return clone

    def to_dict(self) -> dict:
        """Serialize all weights to a dictionary."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "lr": self.lr,
            "n_heads": self.n_heads,
            "w_attn": self.w_attn.tolist(), "b_attn": self.b_attn.tolist(),
            "w_val": self.w_val.tolist(), "b_val": self.b_val.tolist(),
            "w1": self.w1.tolist(), "b1": self.b1.tolist(),
            "w2": self.w2.tolist(), "b2": self.b2.tolist(),
            "w3": self.w3.tolist(), "b3": self.b3.tolist(),
            "w4": self.w4.tolist(), "b4": self.b4.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Evaluator":
        """Deserialize from dictionary."""
        net = cls(
            input_size=data["input_size"],
            hidden_size=data["hidden_size"],
            learning_rate=data.get("lr", 0.001),
            n_heads=data.get("n_heads", 4),
        )
        for key in ("w_attn", "b_attn", "w_val", "b_val",
                    "w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4"):
            if key in data:
                setattr(net, key, np.array(data[key]))
        return net

    def _attend(self, x: np.ndarray) -> np.ndarray:
        """Self-attention: compute importance weights and re-weight input."""
        attn_logits = x @ self.w_attn + self.b_attn
        attn_exp = np.exp(attn_logits - attn_logits.max())
        attn_weights = attn_exp / (attn_exp.sum() + 1e-8)
        values = np.maximum(0, x @ self.w_val + self.b_val)
        chunk_size = self.input_size // self.n_heads
        attended = np.zeros(self.input_size)
        for h in range(self.n_heads):
            start = h * chunk_size
            end = start + chunk_size
            attended[start:end] = values[start:end] * attn_weights[h]
        return x + attended  # Residual connection

    def _flush_batch(self) -> None:
        """Train on accumulated batch with backpropagation and gradient clipping."""
        if not self._batch_x:
            return
        batch_x = np.array(self._batch_x)
        batch_y = np.array(self._batch_y)
        self._batch_x.clear()
        self._batch_y.clear()
        n = len(batch_x)

        # Forward
        attended = np.array([self._attend(x) for x in batch_x])
        h1 = np.maximum(0, attended @ self.w1 + self.b1)
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)
        h3 = np.maximum(0, h2 @ self.w3 + self.b3)
        preds = (h3 @ self.w4 + self.b4).flatten()

        # Backward
        errors = batch_y - preds
        d_out = (-2.0 / n) * errors.reshape(-1, 1)

        d_w4 = h3.T @ d_out
        d_b4 = d_out.sum(axis=0)
        d_h3 = d_out @ self.w4.T; d_h3[h3 <= 0] = 0
        d_w3 = h2.T @ d_h3
        d_b3 = d_h3.sum(axis=0)
        d_h2 = d_h3 @ self.w3.T; d_h2[h2 <= 0] = 0
        d_w2 = h1.T @ d_h2
        d_b2 = d_h2.sum(axis=0)
        d_h1 = d_h2 @ self.w2.T; d_h1[h1 <= 0] = 0
        d_w1 = attended.T @ d_h1
        d_b1 = d_h1.sum(axis=0)

        # Gradient clipping
        for grad in (d_w1, d_w2, d_w3, d_w4):
            norm = np.linalg.norm(grad)
            if norm > self.max_grad_norm:
                grad *= self.max_grad_norm / norm

        # SGD step
        self.w1 -= self.lr * d_w1; self.b1 -= self.lr * d_b1
        self.w2 -= self.lr * d_w2; self.b2 -= self.lr * d_b2
        self.w3 -= self.lr * d_w3; self.b3 -= self.lr * d_b3
        self.w4 -= self.lr * d_w4; self.b4 -= self.lr * d_b4


class QNetwork:
    """Fixed-output feedforward network for small discrete action spaces."""

    def __init__(
        self,
        input_size: int = 32,
        hidden_size: int = 64,
        output_size: int = 8,
        learning_rate: float = 0.001,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        s1 = np.sqrt(2.0 / input_size)
        s2 = np.sqrt(2.0 / hidden_size)
        self.w1 = np.random.randn(input_size, hidden_size) * s1
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size) * s2
        self.b2 = np.zeros(output_size)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward pass → Q-values for all actions."""
        h = np.maximum(0, x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

    def update(self, x: np.ndarray, target_idx: int, target_value: float) -> float:
        """Update Q-value for a single action index. Returns squared error."""
        h = np.maximum(0, x @ self.w1 + self.b1)
        q_values = h @ self.w2 + self.b2
        error = target_value - q_values[target_idx]

        d_out = np.zeros(self.output_size)
        d_out[target_idx] = -2 * error
        d_w2 = np.outer(h, d_out)
        d_b2 = d_out
        d_h = d_out @ self.w2.T; d_h[h <= 0] = 0
        d_w1 = np.outer(x, d_h)
        d_b1 = d_h

        self.w1 -= self.lr * d_w1; self.b1 -= self.lr * d_b1
        self.w2 -= self.lr * d_w2; self.b2 -= self.lr * d_b2
        return error ** 2

    def to_dict(self) -> dict:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "lr": self.lr,
            "w1": self.w1.tolist(), "b1": self.b1.tolist(),
            "w2": self.w2.tolist(), "b2": self.b2.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QNetwork":
        net = cls(
            input_size=data["input_size"],
            hidden_size=data["hidden_size"],
            output_size=data["output_size"],
            learning_rate=data.get("lr", 0.001),
        )
        net.w1 = np.array(data["w1"]); net.b1 = np.array(data["b1"])
        net.w2 = np.array(data["w2"]); net.b2 = np.array(data["b2"])
        return net
```

### MCTS Engine (`mcts.py`)

```python
from __future__ import annotations
from typing import Callable
from tibrain import State, Action


SimulateFn = Callable[
    [State, Action],
    tuple[State, float, bool, list[Action]],
]


class MCTSEngine:
    """Generic Monte Carlo Tree Search engine."""

    def __init__(
        self,
        simulate_fn: SimulateFn,
        num_simulations: int = 100,
    ) -> None:
        self._simulate = simulate_fn
        self.num_simulations = num_simulations

    def choose_action(
        self,
        state: State,
        legal_actions: list[Action],
        num_simulations: int | None = None,
    ) -> Action:
        """Return the action with highest average simulated reward."""
        if len(legal_actions) == 1:
            return legal_actions[0]

        scores = self._score_actions(state, legal_actions, num_simulations)
        return max(legal_actions, key=lambda a: scores.get(a, 0.0))

    def evaluate_actions(
        self,
        state: State,
        legal_actions: list[Action],
        num_simulations: int | None = None,
    ) -> dict[Action, float]:
        """Return normalized scores [0, 1] per action."""
        if len(legal_actions) <= 1:
            return {}

        scores = self._score_actions(state, legal_actions, num_simulations)
        values = list(scores.values())
        min_v, max_v = min(values), max(values)
        spread = max_v - min_v if max_v > min_v else 1.0

        return {a: (scores[a] - min_v) / spread for a in legal_actions}

    def _score_actions(
        self,
        state: State,
        legal_actions: list[Action],
        num_simulations: int | None,
    ) -> dict[Action, float]:
        """Run simulations for each action, return average rewards."""
        n_sims = num_simulations or self.num_simulations
        scores: dict[Action, float] = {}

        for action in legal_actions:
            total_reward = 0.0
            for _ in range(n_sims):
                total_reward += self._rollout(state, action)
            scores[action] = total_reward / n_sims

        return scores

    def _rollout(self, state: State, action: Action) -> float:
        """Execute a single simulation rollout from (state, action)."""
        cumulative_reward = 0.0
        next_state, reward, done, next_actions = self._simulate(state, action)
        cumulative_reward += reward

        # Continue rollout with random actions until terminal
        import random as _rnd
        current_state = next_state
        current_actions = next_actions

        while not done and current_actions:
            a = _rnd.choice(current_actions)
            current_state, reward, done, current_actions = self._simulate(
                current_state, a
            )
            cumulative_reward += reward

        return cumulative_reward
```

### Evaluation (`evaluation.py`)

```python
from __future__ import annotations
from collections import deque


class EloTracker:
    """Elo rating tracker for measuring agent improvement."""

    def __init__(self, initial_elo: float = 1000.0, k_factor: float = 32.0) -> None:
        self.elo = initial_elo
        self.k_factor = k_factor
        self._history: list[tuple[int, float]] = []

    def update(self, won: bool, opponent_elo: float = 1000.0) -> None:
        """Update Elo after a game result."""
        expected = 1.0 / (1.0 + 10 ** ((opponent_elo - self.elo) / 400))
        actual = 1.0 if won else 0.0
        self.elo += self.k_factor * (actual - expected)

    def record(self, episode: int) -> None:
        """Record a (episode, elo) snapshot. Retains last 100."""
        self._history.append((episode, self.elo))
        if len(self._history) > 100:
            self._history = self._history[-100:]

    def to_dict(self) -> dict:
        return {"elo": self.elo, "history": self._history[-100:]}

    @classmethod
    def from_dict(cls, data: dict) -> "EloTracker":
        tracker = cls(initial_elo=data.get("elo", 1000.0))
        tracker._history = data.get("history", [])
        return tracker


class MetaLearner:
    """Tracks recent scores and suggests hyperparameter adjustments."""

    def __init__(self, window_size: int = 50, adjustment_interval: int = 200) -> None:
        self._scores: deque[float] = deque(maxlen=window_size)
        self._adjustment_interval = adjustment_interval
        self._last_adjustment: int = 0

    def record_score(self, score: float) -> None:
        """Record a score from a completed episode."""
        self._scores.append(score)

    def should_adjust(self, episode: int) -> bool:
        """Check if enough episodes have passed for adjustment."""
        return (
            episode - self._last_adjustment >= self._adjustment_interval
            and len(self._scores) >= 30
        )

    def suggest_adjustments(
        self,
        current_epsilon: float,
        episode: int,
    ) -> dict[str, float]:
        """
        Suggest hyperparameter changes based on recent performance trends.

        Returns dict of suggested new values (empty if no change needed).
        """
        self._last_adjustment = episode
        avg_score = sum(self._scores) / len(self._scores)
        recent_10 = list(self._scores)[-10:]
        avg_recent = sum(recent_10) / len(recent_10)

        adjustments: dict[str, float] = {}

        # Performance improving > 10% → reduce exploration
        if avg_recent > avg_score * 1.1:
            adjustments["epsilon"] = max(0.02, current_epsilon * 0.95)
        # Performance declining > 20% → increase exploration
        elif avg_recent < avg_score * 0.8 and current_epsilon < 0.3:
            adjustments["epsilon"] = min(0.3, current_epsilon * 1.1)

        return adjustments
```

### Training (`training.py`)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
from tibrain import Environment, State
from tibrain.agent import Agent


@dataclass
class TrainingPhase:
    """Configuration for a single training phase."""
    episodes: int
    alpha: float | None = None
    gamma: float | None = None
    epsilon: float | None = None
    lambda_trace: float | None = None
    environment_factory: Callable[[], Environment] | None = None
    label: str = ""


@dataclass
class TrainingResult:
    """Results from a training session."""
    episodes_completed: int = 0
    cumulative_rewards: list[float] = field(default_factory=list)
    phase_metrics: list[dict] = field(default_factory=list)


def train(
    agent: Agent,
    environment: Environment,
    episodes: int,
    *,
    phases: list[TrainingPhase] | None = None,
    on_progress: Callable[[dict], None] | None = None,
    report_every: int = 50,
) -> TrainingResult:
    """
    Generic training loop with curriculum support.

    If `phases` is provided, training executes each phase sequentially.
    Otherwise runs `episodes` against the given environment.
    """
    result = TrainingResult()

    if phases:
        for phase in phases:
            env = phase.environment_factory() if phase.environment_factory else environment
            # Apply phase hyperparameters
            if phase.alpha is not None:
                agent.q_engine.alpha = phase.alpha
            if phase.epsilon is not None:
                agent.policy.epsilon = phase.epsilon
            if phase.lambda_trace is not None:
                agent.q_engine.lambda_trace = phase.lambda_trace

            phase_result = _run_episodes(
                agent, env, phase.episodes, on_progress, report_every
            )
            result.episodes_completed += phase_result.episodes_completed
            result.cumulative_rewards.extend(phase_result.cumulative_rewards)
            result.phase_metrics.append({
                "label": phase.label,
                "episodes": phase_result.episodes_completed,
                "final_reward": (
                    phase_result.cumulative_rewards[-1]
                    if phase_result.cumulative_rewards else 0.0
                ),
            })
    else:
        result = _run_episodes(agent, environment, episodes, on_progress, report_every)

    return result


def _run_episodes(
    agent: Agent,
    environment: Environment,
    episodes: int,
    on_progress: Callable[[dict], None] | None,
    report_every: int,
) -> TrainingResult:
    """Execute the core RL loop for N episodes."""
    result = TrainingResult()

    for ep in range(episodes):
        state = environment.reset()
        agent.reset_episode()
        episode_reward = 0.0
        done = False

        while not done:
            legal_actions = environment.get_legal_actions(state)
            if not legal_actions:
                break

            action = agent.choose_action(state, legal_actions)
            next_state, reward, info = environment.step(action)
            done = info.get("done", False)

            next_legal = (
                environment.get_legal_actions(next_state) if not done else []
            )
            agent.learn(state, action, reward, next_state, next_legal)

            state = next_state
            episode_reward += reward

        result.episodes_completed += 1
        result.cumulative_rewards.append(episode_reward)

        if on_progress and (ep + 1) % report_every == 0:
            on_progress({
                "episode": ep + 1,
                "cumulative_reward": episode_reward,
                "epsilon": agent.policy.epsilon,
                "q_table_size": agent.q_engine.q1.size + agent.q_engine.q2.size,
            })

    return result
```

### Persistence (`persistence.py`)

```python
from __future__ import annotations
import json
from pathlib import Path
from tibrain.agent import Agent


def save(agent: Agent, path: Path, changed_components: set[str] | None = None) -> None:
    """
    Serialize all agent state to a JSON file.

    If changed_components is provided, performs incremental save
    by only updating those keys in the existing file.
    """
    data = _agent_to_dict(agent)

    if changed_components and path.exists():
        existing = json.loads(path.read_text())
        for key in changed_components:
            if key in data:
                existing[key] = data[key]
        data = existing

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load(path: Path) -> dict:
    """
    Deserialize agent state from JSON file.
    Returns empty dict if file does not exist.
    """
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _agent_to_dict(agent: Agent) -> dict:
    """Convert agent internal state to serializable dictionary."""
    data: dict = {}
    data["q1"] = agent.q_engine.q1.to_dict()
    data["q2"] = agent.q_engine.q2.to_dict()
    data["policy"] = {
        "epsilon": agent.policy.epsilon,
        "epsilon_min": agent.policy.epsilon_min,
    }
    if agent.replay_buffer:
        data["replay_buffer"] = [
            list(entry) for entry in agent.replay_buffer._buffer
        ]
    if agent._evaluator:
        data["evaluator"] = agent._evaluator.to_dict()
    return data
```

### Reward Normalization & Curiosity (in `agent.py` or separate utility)

```python
import math


class RewardNormalizer:
    """Running normalization using Welford's online algorithm."""

    def __init__(self) -> None:
        self._mean: float = 0.0
        self._m2: float = 0.0
        self._count: int = 0

    def normalize(self, reward: float) -> float:
        """Normalize reward using running statistics."""
        self._count += 1
        delta = reward - self._mean
        self._mean += delta / self._count
        delta2 = reward - self._mean
        self._m2 += delta * delta2

        std = math.sqrt(self._m2 / max(self._count, 1)) + 1e-8
        return (reward - self._mean) / std

    def to_dict(self) -> dict:
        return {"mean": self._mean, "m2": self._m2, "count": self._count}

    @classmethod
    def from_dict(cls, data: dict) -> "RewardNormalizer":
        rn = cls()
        rn._mean = data.get("mean", 0.0)
        rn._m2 = data.get("m2", 0.0)
        rn._count = data.get("count", 0)
        return rn


class CuriosityModule:
    """Exploration bonus based on state visit counts."""

    def __init__(self, scale: float = 0.1) -> None:
        self.scale = scale
        self._visit_counts: dict[str, int] = {}

    def visit(self, state_key: str) -> None:
        """Record a state visit."""
        self._visit_counts[state_key] = self._visit_counts.get(state_key, 0) + 1

    def bonus(self, state_key: str) -> float:
        """Return curiosity bonus: scale / sqrt(visit_count)."""
        count = self._visit_counts.get(state_key, 0)
        if count == 0:
            return self.scale  # Max bonus for unvisited
        return self.scale / math.sqrt(count)
```

### Discovery Engine (`discovery/discovery_engine.py`)

```python
from __future__ import annotations
from tibrain.discovery.pattern import Pattern


class DiscoveryEngine:
    """Generic pattern detection in experience sequences."""

    def __init__(self, confidence_threshold: float = 0.3) -> None:
        self.confidence_threshold = confidence_threshold
        self._pattern_counts: dict[str, int] = {}
        self._total_observations: int = 0
        self._registry: dict[str, Pattern] = {}

    def observe(self, state_pattern: str, action_pattern: str, reward_outcome: str) -> None:
        """Record an observation for pattern detection."""
        key = f"{state_pattern}|{action_pattern}|{reward_outcome}"
        self._pattern_counts[key] = self._pattern_counts.get(key, 0) + 1
        self._total_observations += 1

        # Update or create pattern in registry
        confidence = self._pattern_counts[key] / self._total_observations
        if confidence >= self.confidence_threshold:
            self._registry[key] = Pattern(
                state_pattern=state_pattern,
                action_pattern=action_pattern,
                reward_outcome=reward_outcome,
                confidence=confidence,
                observations=self._pattern_counts[key],
            )
        elif key in self._registry:
            # Confidence dropped below threshold — remove
            del self._registry[key]

    def detect_patterns(self, experiences: list[tuple[str, str, str]]) -> list[Pattern]:
        """Process a batch of experiences and return patterns above threshold."""
        for state_p, action_p, reward_o in experiences:
            self.observe(state_p, action_p, reward_o)

        return [p for p in self._registry.values()
                if p.confidence >= self.confidence_threshold]

    def to_dict(self) -> dict:
        return {
            "pattern_counts": self._pattern_counts,
            "total_observations": self._total_observations,
            "threshold": self.confidence_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DiscoveryEngine":
        engine = cls(confidence_threshold=data.get("threshold", 0.3))
        engine._pattern_counts = data.get("pattern_counts", {})
        engine._total_observations = data.get("total_observations", 0)
        # Rebuild registry
        for key, count in engine._pattern_counts.items():
            confidence = count / max(engine._total_observations, 1)
            if confidence >= engine.confidence_threshold:
                parts = key.split("|")
                if len(parts) == 3:
                    engine._registry[key] = Pattern(
                        state_pattern=parts[0],
                        action_pattern=parts[1],
                        reward_outcome=parts[2],
                        confidence=confidence,
                        observations=count,
                    )
        return engine
```

### Pattern (`discovery/pattern.py`)

```python
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Pattern:
    """A detected recurring pattern in experience data."""
    state_pattern: str
    action_pattern: str
    reward_outcome: str
    confidence: float
    observations: int
```

## Data Models

### Key Data Structures

| Structure | Keys | Values | Purpose |
|-----------|------|--------|---------|
| QTable._data | state_key → action_key | float | State-action value estimates |
| ReplayBuffer._buffer | index (ring) | (state, action, reward, next_state, td_error) | Experience storage |
| Policy._visit_counts | action_key | int | UCB exploration tracking |
| EloTracker._history | list index | (episode, elo) | Performance snapshots |
| DiscoveryEngine._registry | pattern_key | Pattern | Active detected patterns |

### TrainingResult Fields

```python
@dataclass
class TrainingResult:
    episodes_completed: int       # Total episodes run
    cumulative_rewards: list[float]  # Reward per episode
    phase_metrics: list[dict]     # Per-phase summary
```

## Error Handling

| Scenario | Handling |
|----------|----------|
| No legal actions in `choose_action` | Raise `ValueError("No legal actions available")` |
| Empty Q-table lookup | Return `0.0` (default) |
| Persistence load from nonexistent path | Return empty `dict` |
| Replay buffer sample > buffer size | Return all available entries |
| Neural net batch size mismatch | Skip malformed batch silently |
| Single legal action in MCTS | Return immediately without simulation |
| Gradient norm exceeds max | Clip to `max_grad_norm` |

## Algorithms

### Double Q-Learning TD(λ) Update (per step)

```
1. Randomly select table (Q1 or Q2) for update
2. Compute best_next_action = argmax_a Q_update(next_state, a)
3. Compute max_next_q = Q_eval(next_state, best_next_action)
4. td_error = reward + gamma * max_next_q - Q_update(state, action)
5. Set e(state, action) += 1  (accumulating trace)
6. For all (s, a) with e(s, a) > 0:
     Q_update(s, a) += alpha * td_error * e(s, a)
     e(s, a) *= gamma * lambda_trace
     If e(s, a) < 0.01: remove trace
```

### MCTS Action Selection

```
1. If only one legal action: return it
2. For each legal action a:
     total_reward = 0
     For i in 1..num_simulations:
       reward = rollout(state, a, simulate_fn)
       total_reward += reward
     scores[a] = total_reward / num_simulations
3. Return argmax_a scores[a]
```

### Prioritized Experience Sampling

```
1. priorities[i] = |td_error_i| + epsilon (0.01)
2. probabilities[i] = priorities[i] / sum(priorities)
3. Sample batch_size indices without replacement using probabilities
```

### Elo Update

```
expected = 1 / (1 + 10^((opponent_elo - my_elo) / 400))
actual = 1.0 if won else 0.0
elo += K * (actual - expected)   where K = 32
```

## Wist Integration Pattern

After extraction, Wist agents become thin adapters:

```python
# agents/wist_learning/learning_agent.py (refactored)
from tibrain import Agent as TIBRAINAgent
from tibrain.training import train, TrainingPhase


class WistLearningAgent:
    """Thin wrapper: provides Wist-specific encoders to TIBRAIN."""

    def __init__(self, training: bool = True) -> None:
        self._agent = TIBRAINAgent(
            state_encoder=self._encode_state,
            action_encoder=self._encode_action,
            alpha=0.1, gamma=0.95, lambda_trace=0.7,
            epsilon=0.3, training=training,
            replay_capacity=8000,
        )

    def act(self, observation):
        state = observation  # Domain observation
        legal = self._get_legal_actions(observation)
        return self._agent.choose_action(state, legal)

    def _encode_state(self, obs) -> str:
        """Domain-specific: encode Wist observation to string key."""
        # ... existing encode_play_state logic ...

    def _encode_action(self, action) -> str:
        """Domain-specific: encode Wist action to string key."""
        # ... existing encode_play_action logic ...
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Domain Isolation

*For any* Python source file within the `tibrain/` package, the file SHALL NOT contain import statements referencing `environments`, `intelligence.core`, `agents`, or string literals containing domain-specific terms (cards, suits, tricks, Wist, gNB, sector, carrier, PRB, UE, RAN, Qabool).

**Validates: Requirements 1.4, 1.5, 16.4**

### Property 2: Q-Table Default and Retrieval

*For any* state-action pair `(s, a)` that has never been set, `QTable.get(s, a)` SHALL return `0.0`. Furthermore, for any state `s` with multiple actions set to distinct values, `QTable.get_best_action(s, actions)` SHALL return the action with the maximum stored value.

**Validates: Requirements 5.2, 5.3**

### Property 3: Q-Table Size Invariant

*For any* sequence of `N` calls to `QTable.set()` on distinct `(state, action)` pairs, the `size` property SHALL equal `N`.

**Validates: Requirements 5.4**

### Property 4: Q-Table Serialization Round-Trip

*For any* QTable containing arbitrary state-action-value entries, `QTable.from_dict(table.to_dict())` SHALL produce a table where `get(s, a)` returns the same float value for every previously set `(s, a)` pair.

**Validates: Requirements 5.5**

### Property 5: TD(λ) Update Correctness

*For any* transition `(state, action, reward, next_state, next_actions)` and existing eligibility traces, after `td_update()`: (a) the Q-value change for each traced state-action pair equals `alpha * td_error * trace_value`, and (b) all traces are subsequently decayed by `gamma * lambda_trace`.

**Validates: Requirements 4.4, 4.5**

### Property 6: Double Q-Learning Reduced Overestimation

*For any* sequence of transitions with positive rewards, the average combined Q-value estimate `(Q1 + Q2) / 2` SHALL be less than or equal to the single-table Q-learning estimate that uses `max(Q(next_state))` directly (within statistical tolerance over 100+ episodes).

**Validates: Requirements 4.1**

### Property 7: Epsilon Decay with Floor

*For any* epsilon value `e` and decay factor `f`, after calling `policy.decay(f)`, the resulting epsilon SHALL equal `max(epsilon_min, e * f)`.

**Validates: Requirements 6.3, 6.4**

### Property 8: Training Mode Disables Exploration and Learning

*For any* state and legal actions, when `agent.training = False`, calling `choose_action()` SHALL always select the greedy action (highest Q-value), and calling `learn()` SHALL not modify any Q-values.

**Validates: Requirements 3.4**

### Property 9: Replay Buffer Capacity Invariant

*For any* buffer with capacity `C`, after adding `C + K` entries, `len(buffer)` SHALL equal `C`, and the `K` oldest entries SHALL have been evicted.

**Validates: Requirements 7.2**

### Property 10: Prioritized Sampling Distribution

*For any* replay buffer containing entries with varied `|td_error|` values, over a large number of `sample()` calls, entries with higher `|td_error|` SHALL be sampled with proportionally higher frequency (chi-squared test against expected distribution).

**Validates: Requirements 7.3**

### Property 11: Evaluator Serialization Round-Trip

*For any* Evaluator with arbitrary weights, `Evaluator.from_dict(evaluator.to_dict()).predict(x)` SHALL return the same value as `evaluator.predict(x)` for all input vectors `x`.

**Validates: Requirements 8.6**

### Property 12: Evaluator Copy Independence

*For any* Evaluator, after `clone = evaluator.copy()`, updating the original with `evaluator.update(x, target)` SHALL NOT change the output of `clone.predict(x)`.

**Validates: Requirements 8.9**

### Property 13: QNetwork Output Dimension

*For any* QNetwork with `output_size = N` and any valid input vector, `predict(x)` SHALL return an ndarray of shape `(N,)`.

**Validates: Requirements 8.7**

### Property 14: QNetwork Gradient Descent

*For any* QNetwork, input `x`, target index `idx`, and target value `t`, after `update(x, idx, t)`, `|predict(x)[idx] - t|` SHALL be strictly less than the pre-update distance `|predict_before(x)[idx] - t|` (given a sufficiently small learning rate).

**Validates: Requirements 8.8**

### Property 15: MCTS Normalized Scores Range

*For any* set of legal actions (|actions| > 1) and simulate function, `evaluate_actions()` SHALL return scores where all values are in `[0.0, 1.0]` and at least one value equals `1.0` (the best action).

**Validates: Requirements 9.3**

### Property 16: Reward Normalizer Online Statistics

*For any* sequence of `N` rewards, the RewardNormalizer's internal mean SHALL equal `numpy.mean(rewards)` and its standard deviation SHALL equal `numpy.std(rewards, ddof=0)` (within floating-point tolerance).

**Validates: Requirements 10.1, 10.2**

### Property 17: Curiosity Bonus Formula

*For any* state visited `N` times with scale `s`, the curiosity bonus SHALL equal `s / sqrt(N)`.

**Validates: Requirements 10.3**

### Property 18: Persistence Round-Trip

*For any* Agent with populated Q-tables, replay buffer, and evaluator, after `save(agent, path)` followed by `data = load(path)`, the loaded dictionary SHALL contain all serialized components that reconstruct equivalent agent state.

**Validates: Requirements 11.1, 11.2, 11.3**

### Property 19: Elo Formula Correctness

*For any* sequence of `(won, opponent_elo)` results, the EloTracker's rating SHALL equal the value computed by iteratively applying the standard Elo formula: `elo += 32 * (actual - 1/(1 + 10^((opp - elo)/400)))`.

**Validates: Requirements 13.1**

### Property 20: Meta-Learner Adjustment Conditions

*For any* score window where `avg_recent > avg_total * 1.1`, the MetaLearner SHALL suggest `epsilon = max(0.02, current_epsilon * 0.95)`. For any score window where `avg_recent < avg_total * 0.8` and `epsilon < 0.3`, it SHALL suggest `epsilon = min(0.3, current_epsilon * 1.1)`.

**Validates: Requirements 13.4, 13.5**

### Property 21: Discovery Engine Pattern Threshold

*For any* set of experience observations, `detect_patterns()` SHALL only return patterns whose confidence (observation_count / total_observations) is greater than or equal to the configured threshold.

**Validates: Requirements 14.3**

### Property 22: Discovery Engine Serialization Round-Trip

*For any* DiscoveryEngine state with patterns, `DiscoveryEngine.from_dict(engine.to_dict())` SHALL produce an engine that detects the same patterns at the same confidence levels.

**Validates: Requirements 14.4**

### Property 23: Behavioral Equivalence After Refactoring

*For any* random seed and pre-existing Q-table state, the refactored Wist agent (using TIBRAIN) SHALL produce the same action sequence as the original inline implementation when presented with the same observation sequence.

**Validates: Requirements 15.5**

### Property 24: Evaluator Batch Consistency

*For any* batch of input vectors, `predict_batch(batch)[i]` SHALL equal `predict(batch[i])` for all indices `i` in the batch.

**Validates: Requirements 8.3**

### Property 25: Training Progress Callback Frequency

*For any* training run of `N` episodes with `report_every = R`, the `on_progress` callback SHALL be invoked exactly `floor(N / R)` times.

**Validates: Requirements 12.5**

## Testing Strategy

### Unit Tests

- **Protocol conformance**: Verify mock environments implement the TIBRAIN Environment protocol correctly.
- **Edge cases**: Empty legal actions, single action (MCTS shortcut), nonexistent persistence path, buffer smaller than batch_size.
- **Integration examples**: Wist adapter produces correct state/action keys from known observations.

### Property-Based Tests

All 25 correctness properties above are implemented as property-based tests using Hypothesis (Python PBT framework). Each property test generates random inputs (Q-table entries, neural net weights, reward sequences, experience tuples) and verifies the universal invariant holds across 100+ iterations.

Key generators:
- **Q-Table entries**: Random string state/action keys with float values.
- **Neural net weights**: Random numpy arrays of correct dimensions.
- **Experience tuples**: Random (state_key, action_key, reward, next_state_key, td_error) with varied distributions.
- **Episode trajectories**: Random sequences of transitions for TD(λ) trace verification.
- **Score windows**: Random float sequences for meta-learner threshold testing.

### Integration Tests

- Training loop with a simple GridWorld mock environment (deterministic, small state space).
- Full save/load cycle with all components populated.
- Wist behavioral equivalence test: compare action outputs of pre- and post-refactoring agents on recorded game states.
