# Engineering Intelligence

## Vision

This project explores how to build **general decision-making architectures** — AI systems that learn strategy, adapt to opponents, and make decisions under uncertainty. Rather than isolated solutions for individual problems, the goal is transferable intelligence that works across domains.

The long-term direction is applying these architectures to telecommunications network optimization: resource allocation, fault prediction, and adaptive routing. We start with card games because they compress the same fundamental challenges — hidden information, multi-agent competition, long-term planning, and real-time tactical choices — into environments where we can iterate rapidly and measure progress clearly.

## Current State

### Two Executables

| Application | Description |
|---|---|
| **WistDiscovery.exe** | Watch the AI learn Wist from scratch. See milestones, insights, and strategy evolution in real-time. |
| **SudaneseWist_v2.4.0.exe** | Play Wist against the AI. The AI learns from every human game. |

### Key Features

**Discovery Agent (WistDiscovery)**
- Zero domain knowledge — discovers strategy from score signal alone
- Rich insight system with counter-intuitive discoveries
- Category-filtered insights (Bidding, Trump, Timing, Partnership, Defense, Voids, Counter-Intuitive)
- Milestones that scale to billions of games (volume, streaks, seeks, accuracy)
- 5-stage curriculum (Self-play → Mixed → Adversarial → Elite → Grandmaster)
- Background training with population diversity

**Play Game (SudaneseWist)**
- Human vs AI with learned model
- AI learns from human games (online learning, alpha=0.03)
- Player evaluation: Elo rating, decision quality, category scores
- Insight-based reasoning shown in game log
- Context-aware shota-end wording
- Whip/over-trump red highlighting (only in non-trump-led tricks)

### Learning Architecture

| Component | Purpose |
|---|---|
| Double Q-Learning | Accurate value estimates without overestimation |
| Eligibility Traces (TD(λ)) | Credit assignment across 13 decisions per shota |
| N-step Returns | 3-step forward lookahead blended with backward returns |
| Per-state Adaptive Alpha | Well-learned states update slower than rare states |
| Neural Net (CardEvaluator) | Generalization to unseen states (114-dim input) |
| Prioritized Replay | Focus learning on surprising outcomes |
| MCTS + Training Targets | Forward simulation values feed back into Q-tables |
| Population Diversity | Opponents rotate styles to prevent overfitting |
| Curiosity Bonus | Prefer unvisited states during exploration |
| Curriculum (5 stages) | Progressive difficulty from self-play to grandmaster |
| Richer State Encoding | Partner bid, exact trick counts, 6-feature memory per trick |

### Insight System

Insights accumulate forever — never removed. Each represents something the brain figured out:
- **WHAT:** The lesson itself (plain language, actionable)
- **WHY:** The underlying reason
- **Version (+N):** Refined insights build on earlier ones
- **Counter-intuitive discoveries:** Things that shouldn't work but do

Categories: Bidding | Trump | Timing | Partnership | Defense | Voids | Counter-Intuitive


---

## Why Card Games?

Card games are not the destination. They are the proving ground.

A telecom network deciding how to route traffic under load shares the same decision structure as a card player deciding which card to play: limited information, multiple competing agents, immediate and delayed consequences, and the need to balance risk against reward.

By solving Sudanese Wist and Sudanese Hearts — games with rich strategic depth — we develop and validate the learning algorithms, state representations, and reward structures that will later drive real engineering decisions.

## Research Direction

1. **Card games** (current) — validate learning architectures in controlled environments
2. **Telecom simulation** — apply the same decision frameworks to network optimization
3. **Live systems** — deploy adaptive agents for real-time engineering decisions

---

## Sudanese Wist

A traditional Sudanese team trick-taking card game for 4 players (2 teams of 2). Partners sit opposite each other. A game consists of 5 Shotas (rounds), first team to 25 points wins.

### Discovery Agent — What It Knows vs. Discovers

The agent only receives:
1. **Environment** — there's a game to interact with
2. **Legal moves** — what actions are allowed right now
3. **Score signal** — a number at the end of each shota

Everything else — trump power, bid accuracy, partner cooperation, void strategy, seek pursuit — it discovers from experience.

### Game Rules Summary

- 4 players, 2 teams, 52 cards dealt equally (13 each)
- Bidding phase (Al-Tasmiya): players bid 7-13, highest bidder's team must deliver
- Trump revealed by first card played by the winning bidder
- Must follow suit; if void, play anything (including trump = "whipping")
- Seek: winning all 13 tricks ends the game immediately
- Scoring: meet bid = score tricks won; fail bid = lose bid amount as penalty

---

## Sudanese Hearts

A Sudanese variant of Hearts for 4 players (individual, no teams). Highest total score after 5 Shotas wins.

- Penalty avoidance: hearts = -1 each, Queen of Spades = -7
- Bonuses: Full Gallon (+20), Half Gallon (+10), All Tricks (+18)
- Discovery agent learns all mechanics from reward signal alone

---

## Running

```bash
# Sudanese Wist
python run_wist.py play          # PyGame interactive game
python run_wist.py lab           # Tkinter AI laboratory + training

# Sudanese Hearts
python run_hearts.py watch       # PyGame visual AI watcher

# Discovery Watcher
python gui_wist_discovery/main.py   # Watch AI learn from scratch
```

---

## License

See [LICENSE](LICENSE) file.
