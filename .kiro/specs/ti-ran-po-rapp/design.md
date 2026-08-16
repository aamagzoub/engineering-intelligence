# Design Document: TI-RAN-PO-rApp

## Overview

TI-RAN-PO-rApp is a Power Optimization rApp that uses reinforcement learning via the TIBRAIN library to learn carrier sleep/wake policies for a simulated 5G gNB. The system trains an RL agent over thousands of simulated days, generates advisory recommendations, discovers behavioural patterns, and presents results through a FastAPI + HTMX web dashboard.

The architecture follows a strict domain-separation principle: all RAN-specific knowledge lives in the domain layer, while TIBRAIN handles learning, discovery, and persistence as a domain-agnostic engine.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          TI-RAN-PO-rApp                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │              Experiment Layer (experiments/)                    │       │
│  │   train.py  │  evaluate.py  │  demo.py                        │       │
│  └──────┬──────┴───────┬───────┴──────┬──────────────────────────┘       │
│         │              │              │                                   │
│  ┌──────▼──────────────▼──────────────▼──────────────────────────┐       │
│  │              Domain Layer (src/)                                │       │
│  │                                                                │       │
│  │  ┌────────────────────────────────────────────────────┐       │       │
│  │  │  environment/                                       │       │       │
│  │  │  ┌─────────────────────────────────────────────┐   │       │       │
│  │  │  │  ran_environment.py (TIBRAIN Environment)    │   │       │       │
│  │  │  │    ├── gnb.py (1 gNB)                       │   │       │       │
│  │  │  │    │   ├── sector.py (3 sectors)            │   │       │       │
│  │  │  │    │   │   └── carrier.py (3 per sector)    │   │       │       │
│  │  │  │    ├── traffic_generator.py                 │   │       │       │
│  │  │  │    └── power_model.py                       │   │       │       │
│  │  │  └─────────────────────────────────────────────┘   │       │       │
│  │  └────────────────────────────────────────────────────┘       │       │
│  │                                                                │       │
│  │  ┌────────────┐ ┌────────────────┐ ┌──────────────────┐      │       │
│  │  │ state/     │ │ reward/        │ │ recommendation/  │      │       │
│  │  │ encoder    │ │ reward_engine  │ │ engine           │      │       │
│  │  └────────────┘ └────────────────┘ └──────────────────┘      │       │
│  │                                                                │       │
│  │  ┌────────────┐ ┌────────────────┐ ┌──────────────────┐      │       │
│  │  │ ran/       │ │ discovery/     │ │ evaluation/      │      │       │
│  │  │ measure-   │ │ adapter        │ │ evaluator        │      │       │
│  │  │ ments      │ │                │ │                  │      │       │
│  │  └────────────┘ └────────────────┘ └──────────────────┘      │       │
│  │                                                                │       │
│  └────────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │              Dashboard Layer (src/ui/)                          │       │
│  │   FastAPI app.py  →  Jinja2 + HTMX templates                  │       │
│  │   Reads: data/*.json, data/training_log.jsonl                  │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │              Data Layer (data/)                                 │       │
│  │   q_table.json │ discoveries.json │ recommendations.json       │       │
│  │   training_log.jsonl                                           │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                     TIBRAIN Library (import tibrain)                      │
│  Agent │ training.train() │ persistence │ DiscoveryEngine               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
                    Training Flow
                    ═════════════
    ┌───────────────────────────────────────────────────────┐
    │                                                       │
    │   tibrain.training.train(agent, env, episodes)        │
    │         │                                             │
    │         ▼                                             │
    │   ┌─────────────┐    observe()    ┌──────────────┐   │
    │   │  TIBRAIN     │───────────────►│ RAN_Env      │   │
    │   │  Agent       │                │              │   │
    │   │              │◄───────────────│  state dict  │   │
    │   │              │   state_key    │              │   │
    │   │  choose_     │◄──────────────┐│              │   │
    │   │  action()    │  StateEncoder ││              │   │
    │   │              │───────────────►│  step(action)│   │
    │   │  learn()     │◄───────────────│  → (s,r,info)│   │
    │   └─────────────┘                └──────┬───────┘   │
    │                                         │            │
    │                                         ▼            │
    │                               training_log.jsonl     │
    └───────────────────────────────────────────────────────┘

                   Recommendation Flow
                   ══════════════════
    ┌─────────────┐   observe()   ┌──────────────┐
    │ Recommend.  │──────────────►│ RAN_Env      │
    │ Engine      │               └──────────────┘
    │             │   policy       ┌──────────────┐
    │             │◄──────────────│ TIBRAIN Agent │
    │             │               │ (greedy mode) │
    │             │               └──────────────┘
    │             │
    │             ├──► recommendations.json
    └─────────────┘

                    Discovery Flow
                    ══════════════
    training_log.jsonl ──► DiscoveryEngine.detect_patterns()
                                  │
                                  ▼
                           RAN Discovery Adapter
                           (interpret patterns → insights)
                                  │
                                  ▼
                           discoveries.json

                    Dashboard Flow
                    ══════════════
    data/*.json ──► FastAPI endpoints ──► Jinja2+HTMX ──► Browser
```

---

## Components and Interfaces

### 1. Environment Layer (`src/environment/`)

#### `carrier.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class CarrierType(Enum):
    COVERAGE = "coverage"
    MEDIUM = "medium"
    HIGH = "high"


class CarrierStatus(Enum):
    ON = "on"
    SLEEPING = "sleeping"


@dataclass
class Carrier:
    """A single frequency layer within a sector."""
    carrier_id: str
    carrier_type: CarrierType
    status: CarrierStatus = CarrierStatus.ON
    dl_prb_utilization: float = 0.0
    ul_prb_utilization: float = 0.0
    traffic_volume: float = 0.0
    active_ue_count: int = 0
    capacity: float = 100.0  # max traffic units

    @property
    def is_controllable(self) -> bool:
        return self.carrier_type != CarrierType.COVERAGE

    @property
    def is_on(self) -> bool:
        return self.status == CarrierStatus.ON
```

#### `sector.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from src.environment.carrier import Carrier, CarrierType, CarrierStatus


@dataclass
class Sector:
    """A directional coverage area with 3 carriers."""
    sector_id: str
    carriers: list[Carrier] = field(default_factory=list)
    peak_hour: int = 12  # hour of day for peak traffic

    @property
    def coverage_carrier(self) -> Carrier:
        return next(c for c in self.carriers if c.carrier_type == CarrierType.COVERAGE)

    @property
    def controllable_carriers(self) -> list[Carrier]:
        return [c for c in self.carriers if c.is_controllable]

    @property
    def active_carriers(self) -> list[Carrier]:
        return [c for c in self.carriers if c.is_on]

    @property
    def total_traffic(self) -> float:
        return sum(c.traffic_volume for c in self.carriers)

    @property
    def total_load(self) -> float:
        active = self.active_carriers
        if not active:
            return 0.0
        return sum(c.dl_prb_utilization for c in active) / len(active)
```

#### `gnb.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from src.environment.sector import Sector


@dataclass
class GNodeB:
    """A single 5G base station containing 3 sectors."""
    gnb_id: str = "gNB-1"
    sectors: list[Sector] = field(default_factory=list)
```

#### `traffic_generator.py`

```python
from __future__ import annotations
import math
import random
from dataclasses import dataclass


@dataclass
class TrafficConfig:
    """Configuration for traffic generation."""
    base_load: float = 30.0
    peak_amplitude: float = 50.0
    noise_stddev: float = 5.0
    ue_per_traffic_unit: float = 0.5


class TrafficGenerator:
    """Generates synthetic daily traffic patterns per sector/carrier."""

    def __init__(self, config: TrafficConfig | None = None) -> None:
        self.config = config or TrafficConfig()

    def generate(
        self, time_step: int, peak_hour: int, carrier_capacity: float
    ) -> tuple[float, int]:
        """Generate traffic volume and UE count for a time step.

        Args:
            time_step: Current step (0-95) within the episode.
            peak_hour: Hour of day when traffic peaks for this sector.
            carrier_capacity: Maximum capacity of the carrier.

        Returns:
            Tuple of (traffic_volume, active_ue_count).
        """
        hour = time_step * 0.25  # 15-min steps → fractional hour
        # Sinusoidal daily pattern centered on peak_hour
        phase = 2 * math.pi * (hour - peak_hour) / 24.0
        pattern = self.config.base_load + self.config.peak_amplitude * (
            0.5 * (1 + math.cos(phase))
        )
        # Random variation
        noise = random.gauss(0, self.config.noise_stddev)
        traffic = max(0.0, min(carrier_capacity, pattern + noise))
        ue_count = max(0, int(traffic * self.config.ue_per_traffic_unit))
        return traffic, ue_count
```

#### `power_model.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from src.environment.carrier import Carrier, CarrierStatus
from src.environment.sector import Sector
from src.environment.gnb import GNodeB


@dataclass
class PowerConfig:
    """Power consumption configuration per carrier state."""
    coverage_on: float = 200.0     # Watts
    medium_on: float = 150.0       # Watts
    medium_sleeping: float = 20.0  # Watts
    high_on: float = 180.0         # Watts
    high_sleeping: float = 25.0    # Watts


class PowerModel:
    """Computes power consumption at carrier, sector, and gNB level."""

    def __init__(self, config: PowerConfig | None = None) -> None:
        self.config = config or PowerConfig()

    def carrier_power(self, carrier: Carrier) -> float:
        """Compute power for a single carrier based on its state."""
        ...

    def sector_power(self, sector: Sector) -> float:
        """Sum of carrier_power for all carriers in the sector."""
        return sum(self.carrier_power(c) for c in sector.carriers)

    def gnb_power(self, gnb: GNodeB) -> float:
        """Sum of sector_power for all sectors in the gNB."""
        return sum(self.sector_power(s) for s in gnb.sectors)

    def baseline_power(self, gnb: GNodeB) -> float:
        """Power when all carriers are ON (maximum consumption)."""
        ...

    def power_saved(self, gnb: GNodeB) -> float:
        """Difference between baseline and current power."""
        return self.baseline_power(gnb) - self.gnb_power(gnb)
```

#### `ran_environment.py`

```python
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from tibrain import Environment, State, Action
from src.environment.gnb import GNodeB
from src.environment.sector import Sector
from src.environment.carrier import Carrier, CarrierType, CarrierStatus
from src.environment.traffic_generator import TrafficGenerator, TrafficConfig
from src.environment.power_model import PowerModel, PowerConfig
from src.reward.reward_engine import RewardEngine


class CarrierAction:
    """RAN-specific action enum."""
    KEEP_ON = "KEEP_ON"
    SLEEP = "SLEEP"
    KEEP_SLEEPING = "KEEP_SLEEPING"
    WAKE = "WAKE"


class RANEnvironment:
    """Simulated RAN environment implementing the TIBRAIN Environment protocol.

    Models a single gNB with 3 sectors × 3 carriers.
    Episode = 1 day = 96 steps @ 15-minute intervals.
    """

    STEPS_PER_EPISODE: int = 96

    def __init__(
        self,
        traffic_config: TrafficConfig | None = None,
        power_config: PowerConfig | None = None,
        reward_config: dict[str, float] | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.gnb = self._build_gnb()
        self.traffic_gen = TrafficGenerator(traffic_config)
        self.power_model = PowerModel(power_config)
        self.reward_engine = RewardEngine(**(reward_config or {}))
        self._step_count: int = 0
        self._log_path = log_path
        self._log_file = None

    def reset(self) -> State:
        """Reset environment to start of a new day. Returns initial state."""
        self._step_count = 0
        self._reset_carriers()
        self._generate_traffic()
        return self.observe()

    def observe(self) -> State:
        """Return current observation as a dictionary."""
        obs: dict[str, Any] = {
            "time_of_day": self._step_count,
            "sectors": []
        }
        for sector in self.gnb.sectors:
            sector_obs = {
                "sector_id": sector.sector_id,
                "sector_total_traffic": sector.total_traffic,
                "sector_total_load": sector.total_load,
                "carriers": []
            }
            for carrier in sector.carriers:
                sector_obs["carriers"].append({
                    "carrier_id": carrier.carrier_id,
                    "carrier_status": carrier.status.value,
                    "DL_PRB_utilization": carrier.dl_prb_utilization,
                    "UL_PRB_utilization": carrier.ul_prb_utilization,
                    "traffic_volume": carrier.traffic_volume,
                    "active_UE_count": carrier.active_ue_count,
                })
            obs["sectors"].append(sector_obs)
        return obs

    def get_legal_actions(self, state: State) -> list[Action]:
        """Return legal actions for all controllable carriers."""
        actions: list[Action] = []
        for sector in self.gnb.sectors:
            for carrier in sector.controllable_carriers:
                if carrier.is_on:
                    actions.append((carrier.carrier_id, CarrierAction.KEEP_ON))
                    actions.append((carrier.carrier_id, CarrierAction.SLEEP))
                else:
                    actions.append((carrier.carrier_id, CarrierAction.KEEP_SLEEPING))
                    actions.append((carrier.carrier_id, CarrierAction.WAKE))
        return actions

    def step(self, action: Action) -> tuple[State, float, dict]:
        """Execute action, advance time, return (next_state, reward, info)."""
        carrier_id, action_type = action
        self._apply_action(carrier_id, action_type)
        self._step_count += 1
        self._generate_traffic()
        self._redistribute_traffic()
        self._compute_prb_utilization()

        next_state = self.observe()
        reward = self.reward_engine.compute(self.gnb, self.power_model)
        done = self._step_count >= self.STEPS_PER_EPISODE

        info = {"done": done, "step": self._step_count}

        if self._log_path:
            self._log_interaction(state=None, action=action, reward=reward,
                                  next_state=next_state, info=info)

        return next_state, reward, info

    def _build_gnb(self) -> GNodeB:
        """Construct the 1 gNB / 3 sector / 9 carrier topology."""
        ...

    def _reset_carriers(self) -> None:
        """Reset all carriers to ON status."""
        ...

    def _generate_traffic(self) -> None:
        """Generate traffic for current time step."""
        ...

    def _redistribute_traffic(self) -> None:
        """Redistribute sleeping carrier traffic to active carriers."""
        ...

    def _compute_prb_utilization(self) -> None:
        """Compute PRB utilization from traffic and capacity."""
        ...

    def _apply_action(self, carrier_id: str, action_type: str) -> None:
        """Apply sleep/wake action to the specified carrier."""
        ...

    def _log_interaction(self, **kwargs) -> None:
        """Append interaction to training_log.jsonl."""
        ...
```

---

### 2. State Encoding (`src/state/`)

#### `ran_state_encoder.py`

```python
from __future__ import annotations
from enum import Enum
from tibrain import State, StateEncoder


class Bucket(Enum):
    """Five discretization buckets for continuous values."""
    VERY_LOW = "VL"
    LOW = "L"
    MEDIUM = "M"
    HIGH = "H"
    VERY_HIGH = "VH"


class RANStateEncoder:
    """Discretizes RAN observations into opaque string keys.

    Implements the TIBRAIN StateEncoder protocol.
    Converts continuous metrics into 5 categorical buckets
    and produces a single hashable string key for Q-table lookup.
    """

    # Bucket boundaries (percentile-based)
    BOUNDARIES: list[float] = [20.0, 40.0, 60.0, 80.0]

    def __call__(self, state: State) -> str:
        """Encode a RAN observation dict into an opaque string key.

        The key format is intentionally opaque to TIBRAIN:
        "t{time_bucket}|s0:{load_bucket}{c1_status}{c2_status}|..."

        Args:
            state: RAN observation dictionary from RANEnvironment.observe()

        Returns:
            Opaque string key containing no RAN-specific terminology.
        """
        obs = state
        time_bucket = self._discretize(obs["time_of_day"], 0, 96)
        parts = [f"t{time_bucket.value}"]
        for i, sector in enumerate(obs["sectors"]):
            load_bucket = self._discretize(sector["sector_total_load"], 0, 100)
            carrier_states = ""
            for carrier in sector["carriers"]:
                carrier_states += "1" if carrier["carrier_status"] == "on" else "0"
            parts.append(f"s{i}:{load_bucket.value}{carrier_states}")
        return "|".join(parts)

    def _discretize(self, value: float, min_val: float, max_val: float) -> Bucket:
        """Map a continuous value to one of 5 buckets."""
        normalized = (value - min_val) / (max_val - min_val) * 100
        if normalized < self.BOUNDARIES[0]:
            return Bucket.VERY_LOW
        elif normalized < self.BOUNDARIES[1]:
            return Bucket.LOW
        elif normalized < self.BOUNDARIES[2]:
            return Bucket.MEDIUM
        elif normalized < self.BOUNDARIES[3]:
            return Bucket.HIGH
        else:
            return Bucket.VERY_HIGH
```

---

### 3. Reward Engine (`src/reward/`)

#### `reward_engine.py`

```python
from __future__ import annotations
from src.environment.gnb import GNodeB
from src.environment.power_model import PowerModel


class RewardEngine:
    """Computes scalar reward balancing power savings and service quality.

    reward = power_reduction_reward - service_degradation_penalty
    """

    def __init__(
        self,
        power_reward_scale: float = 1.0,
        penalty_magnitude: float = 5.0,
        overload_threshold: float = 80.0,
    ) -> None:
        self.power_reward_scale = power_reward_scale
        self.penalty_magnitude = penalty_magnitude
        self.overload_threshold = overload_threshold

    def compute(self, gnb: GNodeB, power_model: PowerModel) -> float:
        """Compute reward for the current gNB state.

        Args:
            gnb: Current gNB state with all sectors/carriers.
            power_model: Power model for calculating savings.

        Returns:
            Scalar reward value.
        """
        power_reward = (
            power_model.power_saved(gnb) / power_model.baseline_power(gnb)
        ) * self.power_reward_scale

        penalty = self._compute_penalty(gnb)
        return power_reward - penalty

    def _compute_penalty(self, gnb: GNodeB) -> float:
        """Apply penalty if any active carrier exceeds overload threshold."""
        penalty = 0.0
        for sector in gnb.sectors:
            for carrier in sector.active_carriers:
                if carrier.dl_prb_utilization >= self.overload_threshold:
                    penalty += self.penalty_magnitude
        return penalty
```

---

### 4. Recommendation Engine (`src/recommendation/`)

#### `recommendation_engine.py`

```python
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

from tibrain import State, Action
from tibrain.agent import Agent
from src.environment.ran_environment import RANEnvironment


@dataclass
class Recommendation:
    """An advisory carrier action recommendation."""
    carrier_id: str
    action: Literal["KEEP", "SLEEP", "WAKE"]
    expected_power_reduction: float
    service_risk: float
    confidence: float
    reason: str


class RecommendationEngine:
    """Produces advisory recommendations using the learned TIBRAIN policy.

    Operates in greedy mode (agent.training = False) and never
    auto-executes actions on the network.
    """

    def __init__(self, agent: Agent, environment: RANEnvironment) -> None:
        self.agent = agent
        self.environment = environment

    def generate(self) -> list[Recommendation]:
        """Generate recommendations for all controllable carriers.

        Sets agent to greedy mode, observes current state, queries
        the policy for each carrier, and assembles metadata.

        Returns:
            List of Recommendation objects, one per controllable carrier.
        """
        self.agent.training = False
        state = self.environment.observe()
        legal_actions = self.environment.get_legal_actions(state)
        recommendations: list[Recommendation] = []
        # Group legal actions by carrier
        # For each carrier, choose the best action and build recommendation
        ...
        return recommendations

    def persist(self, recommendations: list[Recommendation], path: Path) -> None:
        """Save recommendations to JSON file."""
        data = [asdict(r) for r in recommendations]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
```

---

### 5. Discovery Layer (`src/discovery/`)

#### `ran_discovery_adapter.py`

```python
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path

from tibrain.discovery.discovery_engine import DiscoveryEngine
from tibrain.discovery.pattern import Pattern


@dataclass
class RANInsight:
    """A human-readable insight derived from learned patterns."""
    summary: str
    time_context: str
    sector_context: str
    carrier_context: str
    action_context: str
    confidence: float
    observation_count: int


class RANDiscoveryAdapter:
    """Translates TIBRAIN generic patterns into RAN-specific insights.

    The interpretation layer maps opaque pattern keys back to
    domain concepts (time, sector, carrier, traffic, PRB, action,
    reward, Q-value) for human consumption.
    """

    # Mapping from opaque bucket codes to RAN concepts
    TIME_LABELS = {
        "VL": "night (00:00–04:45)",
        "L": "early morning (05:00–09:45)",
        "M": "midday (10:00–14:45)",
        "H": "afternoon (15:00–19:45)",
        "VH": "evening (20:00–23:45)",
    }

    LOAD_LABELS = {
        "VL": "very low load (<20%)",
        "L": "low load (20–40%)",
        "M": "moderate load (40–60%)",
        "H": "high load (60–80%)",
        "VH": "very high load (>80%)",
    }

    def __init__(self, discovery_engine: DiscoveryEngine) -> None:
        self.engine = discovery_engine

    def generate_insights(
        self, experiences: list[tuple[str, str, str]]
    ) -> list[RANInsight]:
        """Detect patterns and translate to RAN-specific insights.

        Args:
            experiences: List of (state_pattern, action_pattern, reward_outcome)
                        tuples from training log.

        Returns:
            List of RANInsight objects with human-readable context.
        """
        patterns = self.engine.detect_patterns(experiences)
        return [self._interpret(p) for p in patterns]

    def _interpret(self, pattern: Pattern) -> RANInsight:
        """Convert a generic TIBRAIN pattern to a RAN insight."""
        time_code = self._extract_time(pattern.state_pattern)
        sector_info = self._extract_sector(pattern.state_pattern)
        carrier_info = self._extract_carrier(pattern.action_pattern)
        ...
        return RANInsight(
            summary=f"During {self.TIME_LABELS.get(time_code, 'unknown')} "
                    f"in {sector_info}, {carrier_info} yields positive reward",
            time_context=self.TIME_LABELS.get(time_code, "unknown"),
            sector_context=sector_info,
            carrier_context=carrier_info,
            action_context=pattern.action_pattern,
            confidence=pattern.confidence,
            observation_count=pattern.observations,
        )

    def persist(self, insights: list[RANInsight], path: Path) -> None:
        """Save insights to JSON file."""
        data = [asdict(i) for i in insights]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    def _extract_time(self, state_pattern: str) -> str:
        """Extract time bucket code from opaque state key."""
        ...

    def _extract_sector(self, state_pattern: str) -> str:
        """Extract sector info from opaque state key."""
        ...

    def _extract_carrier(self, action_pattern: str) -> str:
        """Extract carrier action from opaque action key."""
        ...
```

---

### 6. Evaluation Engine (`src/evaluation/`)

#### `evaluator.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field

from tibrain.agent import Agent
from src.environment.ran_environment import RANEnvironment


@dataclass
class EvaluationMetrics:
    """Metrics from policy evaluation."""
    power_reduction_pct: float = 0.0
    total_sleep_duration_steps: int = 0
    service_degradation_count: int = 0
    total_reward: float = 0.0
    # Breakdowns
    by_sector: dict[str, dict] = field(default_factory=dict)
    by_carrier: dict[str, dict] = field(default_factory=dict)
    by_time_of_day: dict[int, dict] = field(default_factory=dict)


class Evaluator:
    """Compares learned policy against all-carriers-ON baseline.

    Runs complete episodes in greedy mode and collects per-element metrics.
    """

    def __init__(self, agent: Agent, environment: RANEnvironment) -> None:
        self.agent = agent
        self.environment = environment

    def evaluate(self, num_episodes: int = 10) -> EvaluationMetrics:
        """Run evaluation episodes and compute metrics.

        Sets agent to greedy mode, runs episodes, computes
        baseline vs learned policy metrics.

        Args:
            num_episodes: Number of episodes to evaluate over.

        Returns:
            Aggregated EvaluationMetrics.
        """
        self.agent.training = False  # greedy mode
        metrics = EvaluationMetrics()
        # Run episodes, collect per-step data, aggregate
        ...
        return metrics
```

---

### 7. RAN Measurements (`src/ran/`)

#### `measurements.py`

```python
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RANMeasurement:
    """A single RAN measurement sample."""
    timestamp: int
    sector_id: str
    carrier_id: str
    dl_prb_utilization: float
    ul_prb_utilization: float
    traffic_volume: float
    active_ue_count: int
```

#### `measurement_mapper.py`

```python
from __future__ import annotations
from src.ran.measurements import RANMeasurement
from src.environment.gnb import GNodeB


class MeasurementMapper:
    """Maps between environment state and RAN measurement format."""

    def extract_measurements(self, gnb: GNodeB, time_step: int) -> list[RANMeasurement]:
        """Extract current measurements from gNB state."""
        measurements = []
        for sector in gnb.sectors:
            for carrier in sector.carriers:
                measurements.append(RANMeasurement(
                    timestamp=time_step,
                    sector_id=sector.sector_id,
                    carrier_id=carrier.carrier_id,
                    dl_prb_utilization=carrier.dl_prb_utilization,
                    ul_prb_utilization=carrier.ul_prb_utilization,
                    traffic_volume=carrier.traffic_volume,
                    active_ue_count=carrier.active_ue_count,
                ))
        return measurements
```

---

### 8. Dashboard (`src/ui/`)

#### `app.py`

```python
from __future__ import annotations
import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="TI-RAN-PO-rApp Dashboard")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent.parent / "data"

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main dashboard page with all panels."""
    ...


@app.get("/api/network-overview")
async def network_overview():
    """HTMX partial: Panel A - Network Overview."""
    ...


@app.get("/api/recommendations")
async def recommendations():
    """HTMX partial: Panel B - Current Recommendations."""
    data = json.loads((DATA_DIR / "recommendations.json").read_text())
    ...


@app.get("/api/energy")
async def energy_view():
    """HTMX partial: Panel C - Energy View."""
    ...


@app.get("/api/behaviour-timeline")
async def behaviour_timeline():
    """HTMX partial: Panel D - 24-Hour Behaviour."""
    ...


@app.get("/api/insights")
async def insights():
    """HTMX partial: Panel E - Learned Insights."""
    data = json.loads((DATA_DIR / "discoveries.json").read_text())
    ...


@app.get("/api/replay/{step}")
async def replay_step(step: int):
    """HTMX partial: Panel F - Replay specific time step."""
    ...
```

---

### 9. Experiment Scripts (`experiments/`)

#### `train.py`

```python
from __future__ import annotations
from pathlib import Path

import tibrain
from tibrain.agent import Agent
from tibrain.training import train, TrainingPhase
from tibrain import persistence

from src.environment.ran_environment import RANEnvironment
from src.state.ran_state_encoder import RANStateEncoder


def run_training(config_path: Path | None = None) -> None:
    """Execute multi-phase training with curriculum."""
    env = RANEnvironment(log_path=Path("data/training_log.jsonl"))
    encoder = RANStateEncoder()
    agent = Agent(state_encoder=encoder, training=True)

    phases = [
        TrainingPhase(episodes=500, epsilon=0.5, label="exploration"),
        TrainingPhase(episodes=1500, epsilon=0.2, label="exploitation"),
        TrainingPhase(episodes=1000, epsilon=0.05, label="refinement"),
    ]

    result = train(agent, env, episodes=3000, phases=phases)

    persistence.save(agent, Path("data/q_table.json"))
    print(f"Training complete: {result.episodes_completed} episodes")
```

#### `demo.py`

```python
from __future__ import annotations
from pathlib import Path

from src.environment.ran_environment import RANEnvironment
from src.state.ran_state_encoder import RANStateEncoder
from src.discovery.ran_discovery_adapter import RANDiscoveryAdapter
from src.recommendation.recommendation_engine import RecommendationEngine
from src.evaluation.evaluator import Evaluator

from tibrain.agent import Agent
from tibrain.training import train, TrainingPhase
from tibrain.discovery.discovery_engine import DiscoveryEngine
from tibrain import persistence


def main() -> None:
    """Full pipeline: train → evaluate → discover → recommend → summarize."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # 1. Create environment
    env = RANEnvironment(log_path=data_dir / "training_log.jsonl")
    encoder = RANStateEncoder()
    agent = Agent(state_encoder=encoder, training=True)

    # 2. Train
    phases = [
        TrainingPhase(episodes=200, epsilon=0.5, label="exploration"),
        TrainingPhase(episodes=300, epsilon=0.1, label="exploitation"),
    ]
    result = train(agent, env, episodes=500, phases=phases)

    # 3. Save model
    persistence.save(agent, data_dir / "q_table.json")

    # 4. Evaluate
    evaluator = Evaluator(agent, env)
    metrics = evaluator.evaluate(num_episodes=5)

    # 5. Discover patterns
    discovery_engine = DiscoveryEngine()
    adapter = RANDiscoveryAdapter(discovery_engine)
    # Load experiences from training log
    experiences = _load_experiences(data_dir / "training_log.jsonl")
    insights = adapter.generate_insights(experiences)
    adapter.persist(insights, data_dir / "discoveries.json")

    # 6. Generate recommendations
    rec_engine = RecommendationEngine(agent, env)
    recommendations = rec_engine.generate()
    rec_engine.persist(recommendations, data_dir / "recommendations.json")

    # 7. Print summary
    print(f"Training: {result.episodes_completed} episodes")
    print(f"Power reduction: {metrics.power_reduction_pct:.1f}%")
    print(f"Insights discovered: {len(insights)}")
    print(f"Recommendations: {len(recommendations)}")
```

---

## Data Models

### Configuration (`config/default.yaml`)

```yaml
gnb:
  id: "gNB-1"
  sectors:
    - id: "sector-0"
      peak_hour: 9
      carriers:
        - {id: "s0-coverage", type: "coverage", capacity: 100}
        - {id: "s0-medium", type: "medium", capacity: 150}
        - {id: "s0-high", type: "high", capacity: 200}
    - id: "sector-1"
      peak_hour: 13
      carriers:
        - {id: "s1-coverage", type: "coverage", capacity: 100}
        - {id: "s1-medium", type: "medium", capacity: 150}
        - {id: "s1-high", type: "high", capacity: 200}
    - id: "sector-2"
      peak_hour: 18
      carriers:
        - {id: "s2-coverage", type: "coverage", capacity: 100}
        - {id: "s2-medium", type: "medium", capacity: 150}
        - {id: "s2-high", type: "high", capacity: 200}

traffic:
  base_load: 30.0
  peak_amplitude: 50.0
  noise_stddev: 5.0
  ue_per_traffic_unit: 0.5

power:
  coverage_on: 200.0
  medium_on: 150.0
  medium_sleeping: 20.0
  high_on: 180.0
  high_sleeping: 25.0

reward:
  power_reward_scale: 1.0
  penalty_magnitude: 5.0
  overload_threshold: 80.0

training:
  phases:
    - {episodes: 500, epsilon: 0.5, label: "exploration"}
    - {episodes: 1500, epsilon: 0.2, label: "exploitation"}
    - {episodes: 1000, epsilon: 0.05, label: "refinement"}
```

### Training Log Entry (`training_log.jsonl`)

```json
{"step": 42, "state_key": "tM|s0:M110|s1:L111|s2:H101", "action": "('s2-high', 'SLEEP')", "reward": 0.12, "next_state_key": "tM|s0:M110|s1:L111|s2:H100", "info": {"done": false, "step": 43}}
```

### Recommendation Entry (`recommendations.json`)

```json
[
  {
    "carrier_id": "s0-medium",
    "action": "SLEEP",
    "expected_power_reduction": 130.0,
    "service_risk": 0.15,
    "confidence": 0.87,
    "reason": "Low traffic period; sleeping medium carrier saves 130W with minimal service risk"
  }
]
```

### Discovery Entry (`discoveries.json`)

```json
[
  {
    "summary": "During night (00:00–04:45) in sector-0 with very low load, sleeping medium carrier yields positive reward",
    "time_context": "night (00:00–04:45)",
    "sector_context": "sector-0",
    "carrier_context": "medium carrier",
    "action_context": "SLEEP",
    "confidence": 0.82,
    "observation_count": 340
  }
]
```

---

## Algorithm: Training Loop Integration

The training loop delegates entirely to `tibrain.training.train()`. The sequence per step:

1. `env.reset()` → initializes gNB, generates traffic for step 0
2. `env.get_legal_actions(state)` → returns carrier action pairs
3. `agent.choose_action(state, legal_actions)` → StateEncoder encodes observation to opaque key, policy selects action
4. `env.step(action)` → applies carrier state change, advances time, regenerates traffic, redistributes load, computes PRB, calculates reward
5. `agent.learn(state, action, reward, next_state, next_legal)` → TD(λ) update
6. Repeat until `info["done"] == True` (step 96)

The `on_progress` callback logs to training_log.jsonl after each step.

---

## Error Handling

| Scenario | Handling |
|----------|----------|
| Empty legal_actions | Cannot occur — each controllable carrier always has 2 valid actions |
| Action on coverage carrier | Not possible — coverage carriers excluded from legal actions |
| PRB overload after SLEEP | Allowed — penalty applied via reward, action not blocked |
| Missing data files (dashboard) | Return empty state with informative message |
| Corrupted q_table.json | tibrain.persistence.load() returns empty dict, agent starts fresh |
| Log file I/O error | Catch and warn, training continues without logging |

---

## Testing Strategy

The TI-RAN-PO-rApp uses a dual testing approach:

**Property-Based Tests** (via Hypothesis):
- Validate universal invariants across randomized inputs (topology, power chain, reward formula, encoding, legal actions, persistence round-trips)
- Minimum 100 iterations per property test
- Focus on pure domain logic (environment, encoder, power model, reward engine)

**Unit Tests** (via pytest):
- Specific examples: protocol conformance, default configuration values, dashboard endpoint responses
- Integration points: TIBRAIN API usage, training loop execution, demo pipeline
- Edge cases: empty states, boundary PRB values, missing data files

**Smoke Tests**:
- Demo script runs end-to-end without errors
- Dashboard starts and serves HTML
- Training survives 1000+ episodes without crash

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Topology Invariant

*For any* instance of RANEnvironment, the gNB shall contain exactly 3 sectors, and each sector shall contain exactly 3 carriers: one of type COVERAGE, one of type MEDIUM, and one of type HIGH.

**Validates: Requirements 1.1, 1.2**

### Property 2: Episode Length

*For any* episode run to completion (without external interruption), the environment shall signal `done=True` at exactly step 96 and never before.

**Validates: Requirements 1.4**

### Property 3: Traffic Conservation on Sleep

*For any* sector where a controllable carrier transitions to SLEEPING, the total traffic demand for that sector (sum across all carriers including the sleeping one's reassigned load) shall be conserved — the traffic previously served by the sleeping carrier is fully redistributed to remaining active carriers in the same sector.

**Validates: Requirements 2.4**

### Property 4: Traffic Stochasticity

*For any* two episodes run with identical configuration and time steps, the generated traffic values shall differ due to random noise application.

**Validates: Requirements 2.2**

### Property 5: Sector Peak Diversity

*For any* RANEnvironment constructed from the default configuration, all three sectors shall have distinct peak traffic hours.

**Validates: Requirements 2.3**

### Property 6: Observation Completeness

*For any* observation returned by RANEnvironment.observe(), the dictionary shall contain time_of_day, and for each sector: sector_total_traffic, sector_total_load, and for each carrier: carrier_status, DL_PRB_utilization, UL_PRB_utilization, traffic_volume, and active_UE_count.

**Validates: Requirements 3.1**

### Property 7: State Encoder Bucket Validity

*For any* valid continuous value within the expected range, the RANStateEncoder shall map it to exactly one of the five buckets {VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH}, and no other value.

**Validates: Requirements 3.2**

### Property 8: Legal Action Correctness

*For any* state returned by the RANEnvironment, the legal action set shall satisfy: (a) for each controllable carrier that is ON, exactly {KEEP_ON, SLEEP} are available; (b) for each controllable carrier that is SLEEPING, exactly {KEEP_SLEEPING, WAKE} are available; (c) no actions reference a coverage carrier.

**Validates: Requirements 4.1, 4.2, 4.3**

### Property 9: Power Computation Chain

*For any* gNB state, the following arithmetic chain shall hold: (a) carrier_power equals the configured ON or SLEEPING wattage for that carrier's current state; (b) sector_power equals the sum of carrier_power for all carriers in the sector; (c) gnb_power equals the sum of all sector_power values; (d) power_saved equals baseline_power minus gnb_power.

**Validates: Requirements 5.2, 5.3, 5.4, 5.5**

### Property 10: Reward Formula Correctness

*For any* gNB state, the computed reward shall equal `power_reduction_reward - service_degradation_penalty`, where: (a) power_reduction_reward is proportional to power_saved / baseline_power scaled by the configured factor; (b) service_degradation_penalty is non-zero if and only if at least one active carrier has DL_PRB_utilization >= the configured overload threshold; (c) the environment never blocks an action that would cause overload (it applies the action and then penalizes).

**Validates: Requirements 6.1, 6.2, 6.3**

### Property 11: Domain Separation

*For any* state or action passed to the TIBRAIN Agent (via state_encoder or action_encoder), the resulting string key shall not contain any of the RAN-specific terms: "gNB", "sector", "carrier", "PRB", "UE", "power", "sleep", "wake".

**Validates: Requirements 7.6, 14.1, 14.2**

### Property 12: Recommendation Validity

*For any* set of recommendations generated by the RecommendationEngine, each recommendation shall: (a) reference a valid controllable carrier ID; (b) have an action field of exactly "KEEP", "SLEEP", or "WAKE"; (c) include non-null expected_power_reduction, service_risk, and confidence metadata fields.

**Validates: Requirements 8.1, 8.2**

### Property 13: Recommendation Persistence Round-Trip

*For any* list of Recommendation objects, serializing to recommendations.json and deserializing shall produce an equivalent list of recommendation dictionaries with all fields preserved.

**Validates: Requirements 8.3**

### Property 14: Discovery Insight Generation from Evidence

*For any* set of distinct generic TIBRAIN patterns fed to the RANDiscoveryAdapter, the generated insights shall: (a) produce distinct summary text for distinct patterns (not static/hard-coded); (b) reference at least one RAN dimension (time, sector, carrier, traffic, PRB, action, reward, or Q-value) in each insight.

**Validates: Requirements 9.2, 9.3**

### Property 15: Discovery Persistence Round-Trip

*For any* list of RANInsight objects, serializing to discoveries.json and deserializing shall produce an equivalent list of insight dictionaries with all fields preserved.

**Validates: Requirements 9.4**

### Property 16: Evaluation Completeness

*For any* evaluation result produced by the Evaluator, the metrics shall include: power_reduction_pct, total_sleep_duration_steps, service_degradation_count, and total_reward at the top level, plus breakdowns by sector, carrier, and time_of_day.

**Validates: Requirements 10.2, 10.3**

### Property 17: Training Log Completeness

*For any* training step executed with logging enabled, the training_log.jsonl file shall grow by one line containing all required fields: state (or state_key), action, reward, next_state (or next_state_key), and info.

**Validates: Requirements 11.1**

### Property 18: Agent Persistence Round-Trip

*For any* trained TIBRAIN Agent, saving via `tibrain.persistence.save()` and loading via `tibrain.persistence.load()` shall produce Q-table data from which equivalent action selections can be reproduced for all previously visited states.

**Validates: Requirements 7.3**
