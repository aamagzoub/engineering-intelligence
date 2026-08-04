# Engineering Intelligence

## Vision

This project explores how to build **general decision-making architectures** — AI systems that learn strategy, adapt to opponents, and make decisions under uncertainty. Rather than isolated solutions for individual problems, the goal is transferable intelligence that works across domains.

The long-term direction is applying these architectures to telecommunications network optimization: resource allocation, fault prediction, and adaptive routing. We start with card games because they compress the same fundamental challenges — hidden information, multi-agent competition, long-term planning, and real-time tactical choices — into environments where we can iterate rapidly and measure progress clearly.

## Why Card Games?

Card games are not the destination. They are the proving ground.

A telecom network deciding how to route traffic under load shares the same decision structure as a card player deciding which card to play: limited information, multiple competing agents, immediate and delayed consequences, and the need to balance risk against reward.

By solving Sudanese Wist and Sudanese Hearts — games with rich strategic depth — we develop and validate the learning algorithms, state representations, and reward structures that will later drive real engineering decisions.

---

## Sudanese Wist

### What It Is

A traditional Sudanese team trick-taking card game for 4 players (2 teams of 2). Partners sit opposite each other. A game consists of 5 Shotas (rounds), first team to 25 points wins.

### Why Wist?

Wist exercises the hardest class of decision problems:

- **Team cooperation with hidden information** — you must infer your partner's hand from their plays, not communication
- **Bidding under uncertainty** — commit to a target before seeing how play unfolds
- **Trump management** — when to spend, when to save, when to flush
- **Seek pressure** — the ever-present threat of an instant-win if one team takes all 13 tricks

### AI Approach

The Wist agent uses **Double Q-Learning with TD(λ) eligibility traces**:
- Learns from every trick (not just end-of-round)
- Tracks all 52 cards played for informed decisions
- Models opponent void/trump patterns
- Prioritized experience replay for rare but important events
- Trained through curriculum learning: 15,000+ games progressing from random to strategic opponents

### Game Rules

**Setup:** 52 cards dealt equally (13 each). One player holds Al-Qabool (right of acceptance) and rotates clockwise each Shota.

**Bidding (Al-Tasmiya):**
- Players bid a number representing tricks they commit to win
- Bid must equal at least (cards in chosen trump suit + 3)
- Opening bid cannot exceed 11; subsequent bids up to 13
- Qabool can match the highest bid without going higher
- A bid of 13 ends bidding immediately

**Trump (Al-Ato):**
- Never declared aloud
- Revealed by the winning bidder's first card — that suit becomes trump for the Shota

**Play:**
- Must follow suit if able
- If void in led suit, may play any card (including trump — called "whipping")
- Highest trump wins; if no trump played, highest of led suit wins
- Winner of each trick leads the next

**Dak (Re-deal):**
- Card-based: no picture cards (A,K,Q,J) or 8+ cards in one suit → mandatory re-deal
- Pass-based: all players pass → Qabool decides (limited to 2 per game)

**Scoring:**
- Playing team meets bid → scores tricks won. Defending team: 0.
- Playing team fails bid → loses bid value. Defending team: scores their tricks.
- **Seek:** Win all 13 tricks → instant game win, overrides everything.

---

## Sudanese Hearts

### What It Is

A Sudanese variant of Hearts for 4 players (individual, no teams). A game consists of 5 Shotas. Highest total score wins.

### Why Hearts?

Hearts is a fundamentally different decision problem from Wist:

- **Penalty avoidance** instead of trick-winning — you want to NOT collect hearts and the Queen of Spades
- **Individual strategy** — no partner to rely on or coordinate with
- **Passing phase** — strategic card exchange before play begins
- **Shooting the Moon** — risky all-or-nothing strategies that flip the scoring

This contrast validates that our learning architecture generalizes across game types, not just one rule set.

### AI Approach

The Hearts agent uses a **Discovery Agent** with Q-Learning:
- State abstraction over hand composition, trick position, hearts broken status, and queen tracking
- Learns passing strategy (which 3 cards to give away)
- Learns trick avoidance (dodge penalties) and shooting detection

### Game Rules

**Setup:** 52 cards dealt equally. Before play, each player passes 3 cards to the next player.

**Play:**
- Standard trick-taking: must follow suit, highest of led suit wins
- No trump suit — all suits are equal
- Hearts cannot be led until "broken" (a heart has been discarded on a previous trick)

**Scoring (per Shota):**
- Each heart collected: -1 point
- Queen of Spades collected: -7 points
- **Full Gallon:** One player takes zero tricks → +20 bonus
- **Half Gallon:** Two players take zero tricks → +10 bonus each
- **All Tricks:** One player wins all 13 → +18 bonus (shoot the moon equivalent)

**Winning:** After 5 Shotas, highest cumulative score wins.

---

## Running

```bash
# Sudanese Wist
python run_wist.py play          # PyGame interactive game
python run_wist.py lab           # Tkinter AI laboratory + training
python run_wist.py train         # CLI curriculum training (15k games)

# Sudanese Hearts
python run_hearts.py watch       # PyGame visual AI watcher
python run_hearts.py train       # CLI training
python run_hearts.py play        # Play against trained AI
```

---

## Research Direction

The progression:

1. **Card games** (current) — validate learning architectures in controlled environments
2. **Telecom simulation** — apply the same decision frameworks to network optimization
3. **Live systems** — deploy adaptive agents for real-time engineering decisions

The card game agents prove that our architectures can:
- Learn from experience without hardcoded rules
- Adapt to different opponent styles
- Balance short-term tactics with long-term strategy
- Operate under imperfect information
- Cooperate with teammates (Wist) or compete individually (Hearts)

These are exactly the capabilities needed for intelligent telecom systems.

---

## License

See [LICENSE](LICENSE) file.
