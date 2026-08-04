# Sudanese Hearts — Requirements

## Overview

Sudanese Hearts is a 4-player trick-avoidance card game. Unlike Wist (where you try to WIN tricks), Hearts punishes you for winning tricks that contain heart cards or the Queen of Spades. The game serves as a proving ground for a **discovery-based learning agent** — an agent that receives ONLY the environment state and legal moves, and must learn the game's mechanics, scoring, and strategy entirely from observation.

## Research Goal

The Wist agent was given rules + strategy hints and learned to optimize. The Hearts agent gets **only legal moves** — it must discover:
- What wins a trick (highest card of led suit)
- That hearts and Queen of Spades are bad
- That avoiding tricks (or specific cards) is the goal
- Strategic card passing
- When going for all tricks (Gallon) is viable

This validates the architecture for the telecom domain, where the agent will receive only logs/tickets/actions and must discover system behavior.

---

## Game Rules

### Players & Teams
- 4 individual players (no teams)
- Competitive — every player for themselves

### Card Ranks
Cards rank lowest to highest: 2, 3, 4, 5, 6, 7, 8, 9, 10, Jack, Queen, King, Ace.

### Dealing
- Full 52-card deck dealt evenly — each player gets 13 cards
- Dealer rotates clockwise each shota
- Player to dealer's left leads the first trick

### Card Passing (Pre-Play Phase)
- After dealing, each player selects 4 cards to pass face-down to the player on their left
- Players select cards to pass BEFORE seeing the cards they will receive
- Once all 4 players have passed, each receives 4 cards from the player on their right
- Passing happens every shota, always to the left

### Trick Play

**Leading:**
- Player to dealer's left leads the first trick
- Winner of each trick leads the next
- Hearts CANNOT be led on the first trick
- After the first trick, hearts can be led freely

**Following:**
- Must follow the led suit if you have cards of that suit
- If void in the led suit, you may play ANY card (including hearts or Queen of Spades — this is "whipping")

**Winning a trick:**
- No trump suit exists in Hearts
- Highest card of the LED SUIT wins the trick
- Off-suit cards (played when void) never win, regardless of rank

### Scoring (Zero-Sum)

All scores in a shota always sum to zero across all four players.

**Base penalties (distributed to trick winners):**
- Each Heart card: **-1 point** (13 hearts × -1 = -13 total)
- Queen of Spades: **-7 points**
- Total negative points per shota: **-20**
- All other cards: 0 points

**Gallon Rules (bonus for winning zero tricks):**

| Scenario | Players with 0 tricks | Their score | Others |
|---|---|---|---|
| Full Gallon | 1 player | +20 | Split the -20 from hearts/queen among themselves |
| Half Gallon | 2 players | +10 each | Split the -20 from hearts/queen among themselves |
| All tricks to one player | 3 players with 0 | -6 each | +18 for the player who took all 13 tricks |

**Important:** The "all tricks" scenario overrides normal Gallon logic. The player who won all 13 tricks gets +18 (not penalized), and the three others get -6 each.

**Zero-sum constraint:** The sum of all four players' scores MUST equal zero every shota. This is enforced by the scoring engine.

### Game Structure
- A game consists of **5 shotas**
- After 5 shotas, the player with the **highest total score wins**
- The player with the **lowest total score loses**

---

## Legal Moves (What the Agent Sees)

The environment exposes ONLY:
1. The player's current hand (13 cards, then decreasing)
2. Which cards are legal to play at this moment
3. Cards played in the current trick (who played what)
4. The result of each trick (who won it)
5. During passing phase: which cards the player holds (to choose 4 to pass)

The environment does NOT tell the agent:
- What the scoring rules are
- That hearts are bad
- That Queen of Spades is special
- What a Gallon is
- Any strategy

The agent receives a numeric reward signal at the end of each shota (its score for that shota). From this signal alone, it must learn everything.

---

## Functional Requirements

### FR-1: Hearts Environment
- Implement `environments/hearts/` as a complete game engine
- Follows the same `Environment` base class pattern as Wist
- Manages: dealing, card passing, trick play, scoring
- Enforces all legal move constraints
- Provides observations to agents (hand, legal moves, trick state)
- Computes zero-sum scores at shota end

### FR-2: Discovery Learning Agent
- New agent type under `agents/discovery/` (or extension of learning agent)
- Receives ONLY: observation (hand + legal moves + trick results) and reward (end-of-shota score)
- No hard-coded knowledge of Hearts rules, scoring, or strategy
- Must learn from repeated play what actions lead to positive/negative outcomes
- Uses the same `Agent` base class

### FR-3: Separate Entry Point
- Separate runnable script for Hearts (e.g., `run_hearts.py` or `environments/hearts/main.py`)
- Can run training sessions (agent vs agent)
- Can display learned behavior / statistics

### FR-4: Reuse Core Infrastructure
- Reuse `intelligence/core/` (Agent, Environment, Action, Observation, Cards, Deck)
- Reuse card rendering if GUI is added later
- Same project, new environment — proves the architecture generalizes

### FR-5: Training Pipeline
- Batch training: run thousands of games for the discovery agent to learn
- Track metrics: agent's average score over time, strategy emergence
- Save/load learned model (Q-tables or equivalent)

---

## Non-Functional Requirements

### NFR-1: Separation of Concerns
- Hearts environment must NOT leak scoring information to the agent through observations
- The agent's only feedback is the end-of-shota reward signal

### NFR-2: Observable Learning Progress
- Metrics should show the agent discovering patterns over time:
  - Phase 1: Random play (scores around -5 average)
  - Phase 2: Learns to avoid winning tricks with hearts
  - Phase 3: Learns Queen of Spades is especially bad
  - Phase 4: Learns strategic passing
  - Phase 5: Attempts Gallon play when hand is strong

### NFR-3: Architecture Transferability
- The discovery agent's architecture should be domain-agnostic enough that it can later be applied to `environments/telecom/` with minimal changes
