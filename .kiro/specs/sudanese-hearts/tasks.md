# Sudanese Hearts — Implementation Tasks

## Task 1: Hearts Environment Core
- [x] Create `environments/hearts/__init__.py`
- [x] Create `environments/hearts/actions.py` (PlayCardAction, PassCardsAction)
- [x] Create `environments/hearts/observation.py` (HeartsObservation, PassingObservation)
- [x] Create `environments/hearts/player.py` (HeartsPlayer — individual, no team)
- [x] Create `environments/hearts/rules.py` (legal_cards, trick_winner, rank_values)
- [x] Create `environments/hearts/trick.py` (reuse or copy from wist)

## Task 2: Scoring Engine
- [x] Create `environments/hearts/scoring.py`
- [x] Implement heart/queen penalty counting
- [x] Implement Full Gallon detection (+20)
- [x] Implement Half Gallon detection (+10 each)
- [x] Implement All-Tricks scenario (+18 / -6 each)
- [x] Enforce zero-sum constraint
- [ ] Unit tests for all scoring scenarios

## Task 3: Game Engine
- [x] Create `environments/hearts/environment.py` (HeartsEnvironment)
- [x] Create `environments/hearts/playing_engine.py` (13 tricks per shota)
- [x] Create `environments/hearts/game.py` (HeartsGame — 5 shotas, passing phase)
- [x] Implement card passing logic (4 cards to the left, before seeing received)
- [x] Implement dealer rotation
- [x] Implement first-trick lead (player to dealer's left)
- [x] Integration test: full game with random agents completes ✓ (verified via run_hearts.py)

## Task 4: Discovery Agent
- [x] Create `agents/discovery/__init__.py`
- [x] Create `agents/discovery/state_encoder.py` (domain-agnostic encoding)
- [x] Create `agents/discovery/discovery_agent.py` (Q-learning, no domain knowledge)
- [x] Create `agents/discovery/model.py` (save/load Q-tables)
- [x] Implement passing strategy learning (separate Q-table)
- [x] Implement trick-play strategy learning
- [x] End-of-shota reward propagation to episode memory

## Task 5: Training Pipeline & Entry Point
- [x] Create `run_hearts.py` (train/play/stats modes)
- [x] Batch training loop (episodes, progress tracking)
- [x] Metrics: average score per 100 episodes, win rate
- [x] Model save/load between sessions
- [x] Console output showing learning progress

## Task 6: Validation & Learning Verification
- [ ] Verify agent improves over random baseline after training
- [ ] Track if agent learns to avoid hearts (compare heart-avoidance rate over time)
- [ ] Track if agent discovers Queen of Spades is worse than a heart
- [ ] Track if agent attempts Gallon play in strong hands
- [ ] Document findings: what the agent discovered and how quickly
