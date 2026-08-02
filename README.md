# Engineering Intelligence  

## Project Vision

This repository is the first implementation of a much larger research project called **Engineering Intelligence**.

The long-term goal of Engineering Intelligence is to explore how to build **general decision-making architectures** that can be applied across completely different domains, rather than creating isolated AI solutions for individual problems.

## Why Sudanese Wist?

Sudanese Wist provides an excellent research environment because it combines many characteristics found in real-world decision-making systems:

- Hidden information
- Team cooperation
- Competitive strategy
- Long-term planning
- Short-term tactical decisions
- Imperfect information
- Learning from experience
- Explainable decision making

Rather than beginning with an abstract mathematical environment, this project starts with a game that is both challenging and familiar, allowing intelligence to emerge in a controlled setting before being transferred to more complex domains such as telecommunications.

---

## Sudanese Wist Game Description

Wist is a traditional Sudanese trick-taking card game for 4 players. This document describes the complete rules of the game exactly as it is played in Sudan. The rules below are precise and specific — do not add, assume, or substitute anything not described here.

---

## Game Rules

### The Teams
4 players sit around a table. The two players sitting opposite each other form one team.

### Card Ranks
Cards rank from lowest to highest: 2, 3, 4, 5, 6, 7, 8, 9, 10, Jack, Queen, King, Ace.

### Direction of Play
Everything — dealing, Al-Tasmiya, and playing — moves clockwise.

### Sahib Al-Qabool — The One with the Right of Acceptance
One player each Shota holds Al-Qabool. He has the final say on Al-Tasmiya. First Shota only: all four players each draw a random card from anywhere in the deck; since all 52 cards are unique there can never be a tie, and the team whose players drew the higher card between their two drawn cards wins Al-Qabool. Every Shota after: Al-Qabool passes to the next player to the left (clockwise), regardless of who won or lost.

### Seating Roles
All roles are anchored to Sahib Al-Qabool each Shota:
- The player opposite him cuts the deck
- The player to his right deals the cards
- The player to his left starts Al-Tasmiya

### Dealing
The deck is shuffled freely. The player opposite Sahib Al-Qabool cuts it. The player to his right then deals all 52 cards clockwise — one card to each player at a time — until each player holds 13 cards.

### Dak (Void)

There are two completely different situations that trigger Dak:

**Situation 1 — Card-based Dak:**
A player must declare Dak before it is their turn in Al-Tasmiya. It is triggered if they hold either:
- No picture cards at all (Ace, King, Queen, Jack are all picture cards) — must show entire hand as proof
- 8 or more cards of one suit — must show those 8+ cards as proof. This always triggers Dak with no exceptions

When card-based Dak is declared, Al-Qabool stays with the same player and cards are re-dealt from scratch.

**Situation 2 — Pass-based Dak:**

First Shota (first deal only):
- If the first two players pass and the third player declares Dak → automatic Dak, Qabool has no say
- If all three pass → Qabool decides

All other Shotas:
- All three players must pass, then Qabool decides whether to declare Dak or play

Pass-based Dak can only happen twice per game. On the third occurrence, Qabool must play.

In all pass-based Dak cases, Al-Qabool moves to the next player to the left.

**Important:** If Dak happens in the very first Shota, it does not count toward the 5 Shotas. From the second Shota onward, any Dak counts as one Shota out of five.

### Al-Tasmiya — The Bid

Al-Tasmiya starts from the player to Qabool's left and moves clockwise. Each player bids a number only — no suit is named.

**Bidding rules:**
- Each bid must be higher than the previous bid
- The bid value must be at least (cards in chosen trump suit) + 3
- The opening bid cannot exceed 11. Subsequent bids can go up to 13
- Trump suit must have 7 or fewer cards
- If any player bids 13, bidding stops immediately — goes straight to Qabool
- Qabool can match the highest bid (does not have to go higher)
- Both bid restrictions are lifted for Qabool

**Standard bid reference:**

| Cards in trump suit | Standard bid | Qabool advantage (matching only) |
|---|---|---|
| 4 | 7 (Marboota) | 7 only |
| 5 | 8 | 7 or 8 |
| 6 | 9 | 8 or 9 |
| 7 | 10 | 9 or 10 |
| 8+ | Must declare Dak | Must declare Dak |

### Qabool's Decision

After the other three have bid or passed, Qabool chooses:
1. **Accept** — that bidder's team plays, Qabool's team defends
2. **Match or outbid** — Qabool's team plays, other team defends
3. **Declare Dak** — only under pass-based Dak conditions

### Al-Ato — The Trump Suit

The trump suit is never declared out loud. It is revealed only when the winning bidder plays their very first card — that suit becomes trump for the entire Shota. The first card played must be from the trump suit.

### Scoring

- If the playing team reaches or exceeds their bid: they score actual tricks won. Defending team scores nothing.
- If the playing team falls short: they lose points equal to their bid. Defending team scores their actual tricks won.
- Scores can go negative.

**Winning a Shota:**
- The playing team wins if they won tricks **equal to or more than** their bid (met or exceeded their commitment)
- The defending team wins if the playing team won **fewer tricks than** their bid (the playing team failed their commitment)

| Playing team bid | Tricks won | Playing team score | Defending team score | Who won the Shota |
|---|---|---|---|---|
| 8 | 10 | +10 | +0 | Playing team |
| 8 | 8 | +8 | +0 | Playing team |
| 8 | 6 | -8 | +7 | Defending team |
| 8 | 3 | -8 | +10 | Defending team |

### Winning the Game
A game consists of 5 Shotas. The first team to reach 25 points wins.

### Seek
If a team wins all 13 tricks in a Shota, the game ends immediately and that team wins — regardless of score or Shotas played. Seek overrides everything.

---

## Legal Moves (Card Play Rules)

These are the rules that determine which cards a player is legally allowed to play during trick play:

1. **First card of a Shota (Shooter leads):** Must play a card from the trump suit (Al-Ato). This is how trump is revealed.

2. **Following suit:** If a suit has been led, you must play a card of that suit if you have one.

3. **Void in led suit:** If you have no cards of the led suit, you may play any card — including a trump card (this is called "whipping").

4. **Trump wins:** The highest trump card played always wins the trick, even if only one trump card was played.

5. **No trump played:** If no trump card was played, the highest card of the led suit wins.

6. **Winner leads next:** The winner of each trick leads the next one.

7. **No review:** Once a trick is placed face down it cannot be reviewed by anyone.

---

## Wist Game Lifecycle

```
START GAME
    → Determine first Sahib Al-Qabool (card draw)
    → START SHOTA
        → Shuffle + Deal 52 cards
        → Card-based Dak? → Re-deal (same Qabool)
        → Al-Tasmiya (Bidding)
        → Pass-based Dak? → Rotate Qabool → New Shota
        → Qabool Decision (Accept/Play)
        → Trump Revealed (first card)
        → Play 13 Tricks
        → Count Tricks
        → Seek (all 13)? → END GAME (winner)
        → Score Shota
        → Game Finished (5 Shotas or 25+ pts)? → END GAME
        → Rotate Qabool → Next Shota
```

---

## Project Structure

```
agents/              - AI agents (rule-based, learning)
environments/wist/   - Game engine (rules, bidding, scoring, tricks)
gui/                 - Tkinter GUI (Stats & Lab, Play for AI tabs)
gui_pygame/          - PyGame GUI (full interactive game)
intelligence/        - Core abstractions (cards, agents, environments)
```

## Running the Game

```bash
# PyGame version (recommended)
python gui_pygame/main.py

# Tkinter version (AI laboratory)
python gui/app.py
```
