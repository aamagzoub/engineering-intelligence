# Engineering Intelligence

## Vision

This project explores how to build **general decision-making architectures** — AI systems that learn strategy, adapt to opponents, and make decisions under uncertainty. Rather than isolated solutions for individual problems, the goal is transferable intelligence that works across domains.

The long-term direction is applying these architectures to telecommunications network optimization: resource allocation, fault prediction, and adaptive routing. We start with card games because they compress the same fundamental challenges — hidden information, multi-agent competition, long-term planning, and real-time tactical choices — into environments where we can iterate rapidly and measure progress clearly.

## Why Card Games?

Card games are not the destination. They are the proving ground.

A telecom network deciding how to route traffic under load shares the same decision structure as a card player deciding which card to play: limited information, multiple competing agents, immediate and delayed consequences, and the need to balance risk against reward.

By solving Sudanese Wist and Sudanese Hearts — games with rich strategic depth — we develop and validate the learning algorithms, state representations, and reward structures that will later drive real engineering decisions.

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

## What Each Agent Is Told

This is the key insight of the project. The three environments give their agents vastly different amounts of prior knowledge:

| What the agent is TOLD | Wist | Hearts | Telecom (goal) |
|---|:---:|:---:|:---:|
| Environment exists (there's a world to interact with) | ✓ | ✓ | ✓ |
| Legal moves (what actions are allowed right now) | ✓ | ✓ | ✓ |
| Outcome signal (a score/reward after actions) | ✓ | ✓ | ✓ |
| Rules of the game (how tricks work, what wins) | ✓ | ✗ | ✗ |
| What the goal is (win tricks / avoid hearts) | ✓ | ✗ | ✗ |
| Scoring formula (how points are calculated) | ✓ | ✗ | ✗ |
| Which cards/items are special (trump, Queen♠) | ✓ | ✗ | ✗ |
| Strategy heuristics (lead trump, avoid high cards) | ✓ | ✗ | ✗ |
| Opponent behavior (what others might do) | ✓ | ✗ | ✗ |
| Game phases (bidding vs. playing) | ✓ | ✗ | ✗ |
| Team structure (who's your partner) | ✓ | ✗ | ✗ |
| Per-action feedback (this trick was good/bad) | ✓ | ✗ | ✗ |
| Domain vocabulary (trump, seek, shota) | ✓ | ✗ | ✗ |

**Summary:**
- **Wist agent** is told **13/13** things — it optimizes within fully known rules
- **Hearts agent** is told **3/13** things — it must discover the other 10 on its own
- **Telecom agent** will be told **3/13** things — same as Hearts

The three things that carry everywhere: there's an environment, here are your legal moves, here's how you did. Everything else — the agent figures out on its own.

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

### Discovery Agent — What It Knows vs. Discovers

The Wist Discovery Agent has zero domain knowledge. It only receives:
1. **Environment** — there's a game to interact with
2. **Legal moves** — what actions are allowed right now
3. **Score signal** — a number at the end of each shota

Everything else — trump power, bid accuracy, partner cooperation, seek pursuit — it discovers from experience.

#### Legal Constraints (enforced — agent cannot violate these)

**All players:**
1. Bid values: 7–13 only
2. Opening bid cannot exceed 11
3. Each bid must exceed the current highest
4. Bid ≥ max(7, chosen trump suit count + 3)
5. Trump suit can be any suit with 1–7 cards
6. 8+ cards in a suit or no pictures = can declare Dak (optional — agent learns whether to Dak or play)
7. Must follow suit if able during trick play
8. If void in led suit, may play any card

**Sahib Al-Qabool additionally:**
1. Can match the highest bid (does not need to exceed)
2. After someone else bid: exempt from trump+3 rule
3. After all passed (Qabool bids first): trump+3 applies, but cap is 13 (not 11)
4. Must lead trump on first trick (when Qabool won the bid)
5. On the 3rd pass-based Dak of a game, must play (cannot pass)

#### Strategy (discovered — agent learns purely from reward signal)
- Whether to declare Dak or play with an 8+ suit / no-picture hand
- Which suit to choose as trump
- When to bid high vs. pass
- How to play tricks (lead, follow, trump, duck)
- Partner cooperation and opponent exploitation
- When to pursue seek (all 13 tricks)
- Bid accuracy (match bid to hand strength)

#### Current Engine Limitation
Trump suit is currently auto-assigned as the longest suit by `determine_trump_suit()`. In real Wist, the player chooses any suit (1–7 cards). A future version will make this a strategic decision for the agent to learn.

### Game Rules

#### The Teams
4 players sit around a table. The two players sitting opposite each other form one team.

#### Card Ranks
Cards rank from lowest to highest: 2, 3, 4, 5, 6, 7, 8, 9, 10, Jack, Queen, King, Ace.

#### Direction of Play
Everything — dealing, Al-Tasmiya, and playing — moves clockwise.

#### Sahib Al-Qabool (The One with the Right of Acceptance)
One player each Shota holds Al-Qabool. He has the final say on Al-Tasmiya. First Shota: determined by card draw. Every Shota after: passes to the next player clockwise.

#### Dealing
52 cards dealt clockwise, one at a time, until each player holds 13 cards.

#### Dak (Re-deal)

**Card-based Dak** — declared before bidding if a player holds:
- No picture cards at all (A, K, Q, J) — must show entire hand as proof
- 8 or more cards of one suit — must show those cards as proof

When card-based Dak is declared, Al-Qabool stays with the same player and cards are re-dealt.

**Pass-based Dak** — triggered when all players pass during bidding:
- First Shota: if first two pass and third declares Dak → automatic. If all three pass → Qabool decides.
- All other Shotas: all three must pass, then Qabool decides.
- Pass-based Dak can only happen twice per game. On the third, Qabool must play.
- Al-Qabool moves to the next player clockwise.

#### Al-Tasmiya (The Bid)

Starts from the player to Qabool's left, moves clockwise. Each player bids a number only — no suit is named.

- Each bid must be higher than the previous
- Bid value must be at least (cards in chosen trump suit + 3)
- The player may choose any suit as trump (1–7 cards) — this choice is strategic, not declared
- Opening bid cannot exceed 11. Subsequent bids up to 13.
- Trump suit must have 7 or fewer cards; 8+ in any suit requires Dak declaration
- If any player bids 13, bidding stops immediately
- Qabool can match the highest bid (does not have to go higher)
- Both bid restrictions (min bid = trump+3, opening cap of 11) are lifted for Qabool
- On the 3rd pass-based Dak of a game, Qabool must play (cannot pass)

| Cards in trump suit | Standard bid | Qabool advantage |
|---|---|---|
| 1–4 | 7 (Marboota) | 7 |
| 5 | 8 | 7 or 8 |
| 6 | 9 | 8 or 9 |
| 7 | 10 | 9 or 10 |
| 8+ | Must declare Dak | — |

#### Al-Ato (The Trump Suit)

The trump suit is never declared aloud. It is revealed only when the winning bidder plays their first card — that card's suit becomes trump for the entire Shota.

#### Scoring

- Playing team meets or exceeds bid → scores actual tricks won. Defending team: 0.
- Playing team falls short → loses points equal to bid. Defending team: scores their actual tricks.

| Bid | Tricks Won | Playing Team | Defending Team | Who Won |
|---|---|---|---|---|
| 8 | 10 | +10 | 0 | Playing |
| 8 | 8 | +8 | 0 | Playing |
| 8 | 6 | -8 | +7 | Defending |
| 8 | 3 | -8 | +10 | Defending |

#### Winning
First team to reach 25 points wins. A game is 5 Shotas.

#### Seek
If a team wins all 13 tricks in a Shota, the game ends immediately and that team wins — regardless of score or Shotas played.

### Legal Moves (Card Play Rules)

These rules determine which cards a player is allowed to play:

1. **First card of a Shota (Shooter leads):** Must play a card from the trump suit. This is how trump is revealed.
2. **Following suit:** If a suit has been led, you must play a card of that suit if you have one.
3. **Void in led suit:** If you have no cards of the led suit, you may play any card — including trump (called "whipping").
4. **Trump wins:** The highest trump card played always wins the trick.
5. **No trump played:** If no trump was played, the highest card of the led suit wins.
6. **Winner leads next:** The winner of each trick leads the next one.

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
- Discovers all game mechanics purely from reward signal — no rules encoded

### Game Rules

#### Players
4 players, each playing individually (no teams). Players compete for the highest score.

#### Card Ranks
Same as Wist: 2, 3, 4, 5, 6, 7, 8, 9, 10, Jack, Queen, King, Ace (low to high).

#### Dealing
52 cards dealt equally — 13 per player. Dealer rotates clockwise each Shota.

#### Passing Phase
Before play begins, each player selects 3 cards and passes them to the next player clockwise. This is strategic: dump dangerous cards or set up avoidance.

#### Play
- The player to the dealer's left leads the first trick
- Standard trick-taking: must follow suit if able
- If void in led suit, may play any card
- **No trump suit** — all suits are equal
- Highest card of the led suit wins the trick
- **Hearts cannot be led** until "broken" (a heart has been discarded on a previous trick)
- Winner of each trick leads the next

#### Penalty Cards
- Each **heart** (♥) collected: **-1 point**
- **Queen of Spades** (Q♠) collected: **-7 points**
- All other cards: 0 points

#### Special Scoring Bonuses
- **Full Gallon:** One player takes zero tricks in a Shota → **+20 points**
- **Half Gallon:** Exactly two players take zero tricks → **+10 points each**
- **All Tricks:** One player wins all 13 tricks → **+18 points** (shoot the moon)

#### Winning
After 5 Shotas, the player with the highest cumulative score wins.

### Legal Moves (Card Play Rules)

1. **Following suit:** If a suit has been led, you must play a card of that suit if you have one.
2. **Void in led suit:** If you have no cards of the led suit, you may play any card.
3. **Hearts breaking:** Hearts cannot be led until at least one heart has been played (discarded when void).
4. **No trump:** There is no trump suit. The highest card of the led suit always wins.
5. **Winner leads next:** The winner of each trick leads the next one.

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

## License

See [LICENSE](LICENSE) file.
