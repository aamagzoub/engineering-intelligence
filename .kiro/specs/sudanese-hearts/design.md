# Sudanese Hearts — Technical Design

## Architecture Overview

```
intelligence/core/          ← Shared (already exists)
    agent.py                   Agent ABC
    environment.py             Environment ABC
    action.py                  Action base
    observation.py             Observation base
    cards/                     Card, Deck, Rank, Suit

environments/hearts/        ← NEW — Hearts game engine
    __init__.py
    actions.py                 PlayCardAction, PassCardsAction
    environment.py             HeartsEnvironment (implements Environment ABC)
    game.py                    HeartsGame orchestrator (5 shotas)
    observation.py             HeartsObservation, PassingObservation
    player.py                  HeartsPlayer (individual, no team)
    playing_engine.py          Runs 13 tricks per shota
    rules.py                   Legal moves, trick winner (no trump)
    scoring.py                 Zero-sum scoring, Gallon detection
    trick.py                   Reuses intelligence/core trick or extends

agents/discovery/           ← NEW — Discovery-based learning agent
    __init__.py
    discovery_agent.py         Agent that learns ONLY from rewards
    state_encoder.py           Minimal state encoding (no domain knowledge)
    model.py                   Q-table storage, save/load

run_hearts.py               ← NEW — Entry point for training & play
```

---

## Component Design

### 1. Actions (`environments/hearts/actions.py`)

```python
@dataclass(frozen=True)
class PlayCardAction(Action):
    """Play one card during trick phase."""
    player_id: int
    card: Card

@dataclass(frozen=True)
class PassCardsAction(Action):
    """Pass 4 cards to the left during passing phase."""
    player_id: int
    cards: tuple[Card, Card, Card, Card]
```

### 2. Observations (`environments/hearts/observation.py`)

The observation is intentionally **minimal** — no scoring info, no hints about what's good or bad.

```python
@dataclass(frozen=True)
class HeartsObservation(Observation):
    """What a Hearts player sees during trick play."""
    player_id: int
    hand: list[Card]                    # Current hand
    legal_cards: list[Card]             # Which cards can be played NOW
    current_trick_cards: list[tuple[int, Card]]  # (player_id, card) played so far
    tricks_won_per_player: dict[int, int]        # How many tricks each player won
    trick_number: int                   # Which trick we're on (1-13)
    cards_played_this_shota: list[Card] # All cards seen so far

@dataclass(frozen=True)
class PassingObservation(Observation):
    """What a player sees during the passing phase."""
    player_id: int
    hand: list[Card]                    # Full 13-card hand before passing
```

**Key design decision:** The observation includes `legal_cards` directly — the agent doesn't need to know WHY a card is legal, just WHICH cards it can play. This mirrors the telecom scenario where the agent sees available actions without understanding the underlying system constraints.

### 3. Rules (`environments/hearts/rules.py`)

```python
def legal_cards(hand: list[Card], leading_suit: Suit | None,
                is_first_trick: bool, hearts_broken: bool) -> list[Card]:
    """
    Legal move rules for Hearts:
    1. First trick: cannot lead hearts
    2. Must follow led suit if able
    3. If void in led suit: can play anything (including hearts/queen)
    4. Cannot lead hearts until hearts are "broken" (played when void)
       — EXCEPT on first trick, after first trick hearts can be led freely
    
    NOTE: Per game rules, after the first trick hearts can be led freely.
    Hearts are "broken" once any heart has been played (which happens naturally).
    """
    ...

def trick_winner(played_cards: list[tuple[int, Card]]) -> int:
    """
    Highest card of the LED SUIT wins. No trump in Hearts.
    """
    ...

RANK_VALUES = {Rank.TWO: 2, ..., Rank.ACE: 14}
```

### 4. Scoring (`environments/hearts/scoring.py`)

```python
def score_shota(tricks_per_player: dict[int, list[list[Card]]]) -> dict[int, int]:
    """
    Compute zero-sum scores for one shota.
    
    Normal scoring (everyone won at least 1 trick):
        player_score = 5 - penalties_collected
        (hearts = 1 penalty each, Queen of Spades = 7 penalties)
        Sum always = 0.
    
    Special scenarios (override normal):
        1. All tricks to one player: +18 for them, -6 each for others
        2. Full Gallon (exactly 1 with 0 tricks): +20 for them,
           others = -(their penalties) adjusted so total = -20
        3. Half Gallon (exactly 2 with 0 tricks): +10 each,
           others split -20 based on collected penalties
    
    Zero-sum invariant: sum(all scores) == 0, always.
    """
    ...
```

**Scoring detail — zero-sum enforcement:**

| Scenario | Player scores |
|---|---|
| Normal play | Each player gets their heart/queen penalties. Sum = -20. Distribute +20 proportionally? NO — just raw penalties, they naturally sum to -20 total. But wait — that's not zero-sum... |

**Zero-sum mechanism — the "+5 baseline":**

Total heart/queen penalties per shota = -20. Divided equally among 4 players = -5 expected.
Each player gets a **baseline of +5** and then subtracts their collected penalties.

**Formula: player_score = 5 - penalties_collected**

- Collected 2 hearts (penalty = 2): score = 5 - 2 = +3
- Collected 6 hearts (penalty = 6): score = 5 - 6 = -1
- Collected 3 hearts + Queen (penalty = 10): score = 5 - 10 = -5
- Collected 0 penalties: score = 5 - 0 = +5

Sum always = (4 × 5) - 20 = 0. Zero-sum enforced naturally. ✓

**Special scenarios override this formula:**
- ALL-TRICKS to one player: +18 for them, -6 for each other (sum = 0) ✓
- FULL GALLON (1 player with 0 tricks): +20 for them, others get (5 - their_penalties) but adjusted so total = 0. In practice: Gallon player gets +20, the remaining -20 is distributed among the other 3 based on their collected hearts/queen.
- HALF GALLON (2 players with 0 tricks): +10 each, the remaining -20 distributed among the other 2 based on their collected hearts/queen.

**Priority order:**
1. All-tricks (+18/-6/-6/-6) — overrides everything
2. Full Gallon (+20 / others split -20 by penalties)
3. Half Gallon (+10, +10 / others split -20 by penalties)
4. Normal play: each player = 5 - penalties_collected

### 5. HeartsPlayer (`environments/hearts/player.py`)

```python
@dataclass
class HeartsPlayer:
    """Individual Hearts player (no team)."""
    player_id: int
    hand: list[Card] = field(default_factory=list)
    
    def receive_cards(self, cards: list[Card]) -> None: ...
    def remove_cards(self, cards: list[Card]) -> None: ...
    def play_card(self, card: Card) -> Card: ...
```

### 6. HeartsEnvironment (`environments/hearts/environment.py`)

```python
class HeartsEnvironment(Environment):
    """
    Hearts environment — manages game state, provides observations.
    
    The environment enforces rules but NEVER exposes scoring logic
    or strategy to agents through observations.
    """
    
    def __init__(self, players: list[HeartsPlayer]):
        self.players = players
        self.tricks_won: dict[int, list[list[Card]]] = {i: [] for i in range(4)}
        self.current_trick: list[tuple[int, Card]] = []
        self.hearts_broken: bool = False
        self.trick_number: int = 1
        self.cards_played: list[Card] = []
    
    def observe(self, player_id: int) -> HeartsObservation: ...
    def apply_action(self, action: Action) -> None: ...
    def get_legal_cards(self, player_id: int) -> list[Card]: ...
```

### 7. HeartsGame (`environments/hearts/game.py`)

```python
class HeartsGame:
    """
    Orchestrates a full Hearts game (5 shotas).
    
    Usage:
        game = HeartsGame(agents=[agent0, agent1, agent2, agent3])
        result = game.play()
    """
    
    def play(self) -> GameResult: ...
    def _play_one_shota(self) -> ShotaResult: ...
    def _passing_phase(self) -> None: ...
    def _playing_phase(self) -> dict[int, int]: ...
```

### 8. Discovery Agent (`agents/discovery/discovery_agent.py`)

```python
class DiscoveryAgent(Agent):
    """
    Discovery-based learning agent.
    
    This agent has ZERO domain knowledge. It does not know:
    - What Hearts is
    - That hearts are bad
    - What Queen of Spades does
    - What a Gallon is
    - Any strategy
    
    It receives:
    - Observation: hand + legal moves + visible trick state
    - Reward: numeric score at end of each shota
    
    It must learn everything from the reward signal alone.
    
    Architecture:
    - State: encoded from observation (position in trick, hand composition,
      cards seen — but NO scoring-aware features)
    - Action: which legal card to play (indexed by position in legal_cards list)
    - Learning: Q-learning with experience replay
    - Passing: separate Q-table for card passing decisions
    """
    
    def __init__(self, epsilon=0.5, alpha=0.1, gamma=0.95): ...
    def act(self, observation: Observation) -> Action: ...
    def reward(self, score: float) -> None: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

**Critical constraint:** The state encoder in the discovery agent must NOT encode any Hearts-specific knowledge. It can encode:
- Number of cards in hand
- Position in trick (0/1/2/3)
- Suit distribution in hand
- What was led
- What others played in current trick
- How many tricks each player has won

It must NOT encode:
- "How many hearts in hand" as a special feature
- "Do I have Queen of Spades" as a special feature
- Any scoring-related information

The agent should treat all suits and all cards equally at initialization. If it learns that hearts are special, it discovers that from the reward signal.

### 9. State Encoder (`agents/discovery/state_encoder.py`)

```python
def encode_state(obs: HeartsObservation) -> str:
    """
    Domain-agnostic state encoding.
    
    Features (all suit-agnostic — treats all suits equally):
    - Position in trick: 0/1/2/3
    - Cards in hand: bucket (1-3/4-7/8-10/11-13)
    - Trick number: bucket (1-4/5-9/10-13)
    - Suits in hand: count of distinct suits
    - Can follow suit: Y/N
    - Highest card in led suit (if following): H/M/L
    - Number of "high cards" (A,K,Q) in hand: bucket
    - Tricks I've won so far: bucket (0/1-3/4-6/7+)
    """
    ...

def encode_passing_state(obs: PassingObservation) -> str:
    """Encode hand for passing decisions."""
    ...
```

### 10. Entry Point (`run_hearts.py`)

```python
"""
Sudanese Hearts — Training & Play

Usage:
    python run_hearts.py train --episodes 10000
    python run_hearts.py play --model saved_model.json
    python run_hearts.py stats --model saved_model.json
"""

def train(episodes: int, save_path: str): ...
def play_demo(model_path: str): ...
def show_stats(model_path: str): ...
```

---

## Data Flow

```
1. PASSING PHASE:
   Game → PassingObservation → Agent.act() → PassCardsAction → Game applies pass

2. TRICK PLAY (×13):
   Environment.observe(player_id)
       → HeartsObservation (hand + legal_cards + trick state)
       → Agent.act(observation)
       → PlayCardAction
       → Environment.apply_action(action)
       → Trick resolves → winner leads next

3. END OF SHOTA:
   Scoring engine computes zero-sum scores
       → Agent.reward(score) — the ONLY learning signal
       → Agent updates Q-tables based on episode memory

4. REPEAT × 5 SHOTAS → Game ends → Report final scores
```

---

## Key Design Decisions

1. **Legal cards in observation** — The agent sees WHICH cards it can play without knowing WHY. Same as telecom: you see available actions without understanding network constraints.

2. **Reward only at shota end** — No per-trick reward. The agent must figure out temporal credit assignment (which of its 13 plays led to a good/bad score). This is harder but more realistic for telecom (you see the outcome after a long sequence of actions).

3. **Suit-agnostic state encoding** — The discovery agent treats all suits equally. If hearts are special, it must learn that. This is the core research property.

4. **Reuse Trick from Wist** — The `Trick` class and `PlayedCard` dataclass from `environments/wist/trick.py` are game-agnostic. We reuse them directly.

5. **Separate from Wist** — Hearts has its own complete module. No imports from `environments/wist/` except potentially shared infrastructure. Clean separation proves the architecture generalizes.

---

## File Dependency Graph

```
intelligence/core/cards/card.py ──┐
intelligence/core/cards/deck.py ──┤
intelligence/core/cards/rank.py ──┤
intelligence/core/cards/suit.py ──┤
intelligence/core/agent.py ───────┤
intelligence/core/environment.py ─┤
intelligence/core/action.py ──────┤
intelligence/core/observation.py ─┘
        │
        ▼
environments/hearts/
    actions.py      → uses Action, Card
    observation.py  → uses Observation, Card
    rules.py        → uses Card, Rank, Suit
    player.py       → uses Card
    trick.py        → reuses wist Trick or own copy
    scoring.py      → standalone (just counts)
    environment.py  → uses all above
    game.py         → orchestrates everything
        │
        ▼
agents/discovery/
    state_encoder.py → uses HeartsObservation
    discovery_agent.py → uses Agent, state_encoder
    model.py         → JSON save/load for Q-tables
        │
        ▼
run_hearts.py → uses HeartsGame + DiscoveryAgent
```

---

## Testing Strategy

- Unit tests for rules (legal moves, trick winner)
- Unit tests for scoring (all scenarios: normal, Gallon, half-Gallon, all-tricks)
- Integration test: full game runs without crashing
- Learning validation: after N episodes, agent's average score improves
- Zero-sum invariant: assert sum of scores = 0 every shota (including normal play with +5 baseline)
