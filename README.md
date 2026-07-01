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


## Sudanese Wist Game Description: 

Wist is a traditional Sudanese trick-taking card game for 4 players. This document describes the complete rules of the game exactly as it is played in Sudan. The rules below are precise and specific — do not add, assume, or substitute anything not described here.

## The Teams
4 players sit around a table. The two players sitting opposite each other form one team.

## Card Ranks
Cards rank from lowest to highest: 2, 3, 4, 5, 6, 7, 8, 9, 10, Jack, Queen, King, Ace.

## Direction of Play
Everything — dealing, Al-Tasmiya (التسمية), and playing — moves counter-clockwise.

## Sahib Al-Qabool (صاحب القبول) — The One with the Right of Acceptance
One player each Shota (شوتة) holds Al-Qabool (القبول). He has the final say on Al-Tasmiya (التسمية). First Shota (شوتة) only: all four players each draw a random card from anywhere in the deck; since all 52 cards are unique there can never be a tie, and the team whose players drew the higher card between their two drawn cards wins Al-Qabool (القبول). Every Shota (شوتة) after: Al-Qabool (القبول) passes to the next player to the left (counter-clockwise), regardless of who won or lost.

## Seating Roles
All roles are anchored to Sahib Al-Qabool (صاحب القبول) each Shota (شوتة):
- The player opposite him cuts the deck
- The player to his right deals the cards
- The player to his left starts Al-Tasmiya (التسمية)

## Dealing
The deck is shuffled freely. The player opposite Sahib Al-Qabool (صاحب القبول) cuts it — the cut can be as little as one card, anywhere in the deck. The player to his right then deals all 52 cards counter-clockwise — one card to each player at a time — until each player holds 13 cards. The last card dealt goes to Sahib Al-Qabool's (صاحب القبول) partner — the one who cut the deck.

## Dak (دك) — Void
There are two completely different situations that trigger Dak (دك):

### Situation 1 — Card-based Dak (دك)
A player must declare Dak (دك) before it is their turn in Al-Tasmiya (التسمية). They cannot declare it after they have already bid. Since play moves counter-clockwise, players declare in turn — if one player declares card-based Dak (دك), the deal stops and re-starts immediately. No other players get a chance to declare. It is triggered if they hold either:
- No picture cards at all in their hand (Ace, King, Queen, and Jack are all considered picture cards) — the player must show their entire hand as proof, or
- 8 or more cards of one suit — the player must show those 8 or more cards to all other players as proof. Holding 8 or more cards in any suit always triggers card-based Dak (دك) with no exceptions — no one, including Sahib Al-Qabool (صاحب القبول), may play when holding 8 or more cards in a suit

When card-based Dak (دك) is declared, Al-Qabool (القبول) stays with the same Sahib Al-Qabool (صاحب القبول) and the cards are simply re-dealt from scratch. No one loses Al-Qabool (القبول).

### Situation 2 — Pass-based Dak (دك)
This happens when players pass during Al-Tasmiya (التسمية).

**First Shota (شوتة) only:**
- If the first two players pass and the third player declares Dak (دك) → it is automatic Dak (دك), Sahib Al-Qabool (صاحب القبول) has no say
- If the first two players pass and the third player also passes → Sahib Al-Qabool (صاحب القبول) decides, just like any other Shota (شوتة)

**All other Shotas (شوتات):**
- All three players must pass, then Sahib Al-Qabool (صاحب القبول) decides whether to declare Dak (دك) or play

In all pass-based Dak (دك) cases, Al-Qabool (القبول) moves to the next player to the left and all roles — cut, deal, and play — reset accordingly.

Pass-based Dak (دك) can only happen twice per game. If the conditions for Dak (دك) arise a third time, Sahib Al-Qabool (صاحب القبول) must either play or accept — Dak (دك) is no longer an option.

**Important:** If Dak (دك) happens in the very first Shota (شوتة) of the game, it does not count toward the 5 Shotas (شوتات) — the game still has all 5 remaining. However, from the second Shota (شوتة) onward, any Dak (دك) — regardless of whether it is card-based or pass-based, and regardless of whether Al-Qabool (القبول) moved or not — counts as one Shota (شوتة) out of the five.

## Al-Tasmiya (التسمية) — The Bid
Al-Tasmiya (التسمية) starts from the player to Sahib Al-Qabool's (صاحب القبول) left and moves counter-clockwise: left of Sahib Al-Qabool (صاحب القبول) → opposite → right (dealer) → then Sahib Al-Qabool (صاحب القبول) decides last.

Each player bids a number only — no suit is named. A player may say pass instead of bidding. If a previous player passed, the next player can still start from 7 — Marboota (مربوطة) — passes do not impose a minimum.

**Al-Tasmiya (التسمية) rules:**
- Each bid must be higher than the previous bid — no equal bids allowed between regular players
- You cannot bid higher than the number of cards you hold in your strongest suit
- The opening bid — the first bid made by any player, not a pass — cannot exceed 11. Subsequent bids can go up to 13
- The trump suit — Al-Ato (الأتو) — cannot come from a suit with 8 or more cards. Holding 8 or more cards in a suit means card-based Dak (دك) must be declared
- Sahib Al-Qabool (صاحب القبول) is the only one who can match the highest bid — he does not have to go higher
- Both restrictions on bidding limits are lifted for Sahib Al-Qabool (صاحب القبول) — he can bid anything he wants, except he also cannot play with 8 or more cards in a suit

If any player bids 13, Al-Tasmiya (التسمية) stops immediately — no further players may bid. The remaining players in the bidding order are skipped and it goes straight to Sahib Al-Qabool (صاحب القبول) to accept or match. This applies whether it is the first player or Sahib Al-Qabool's (صاحب القبول) partner who bids 13.

The bid does not have to come from one suit. It can be based on your strongest cards across all suits. However, Al-Ato (الأتو) — the trump suit — must be the suit the bidder holds the most cards in, and that suit must have 7 or fewer cards.

**Standard Al-Tasmiya (التسمية) reference:**

| Cards in strongest suit | Bid | Notes |
|---|---|---|
| 4 | 7 — Marboota (مربوطة) | |
| 5 | 8 | |
| 6 | 9 | |
| 7 | 10 | Maximum for Al-Ato (الأتو) suit |
| 8+ | — | Must declare card-based Dak (دك) |

## Sahib Al-Qabool's (صاحب القبول) Decision
After the other three have bid or passed (or Al-Tasmiya (التسمية) stopped at 13), Sahib Al-Qabool (صاحب القبول) chooses one of the following:

**Option 1 — Accept the highest bid:** that bidder's team plays, Sahib Al-Qabool's (صاحب القبول) team defends. No further discussion. The player whose bid was accepted leads the first card.

**Option 2 — Match or outbid:** Sahib Al-Qabool (صاحب القبول) matches or exceeds the highest bid. His team plays, the other team defends. No further discussion. Sahib Al-Qabool (صاحب القبول) himself leads the first card. He declares his bid number but does not declare whether he used the extra card advantage — unless all three players before him passed, in which case he must declare whether he is using the extra card advantage or not.

When using the extra card advantage, Sahib Al-Qabool (صاحب القبول) can bid one lower than the standard formula. The trump suit must still have 7 or fewer cards — holding 8 or more cards in any suit means card-based Dak (دك) must be declared, even for Sahib Al-Qabool (صاحب القبول):

| Cards in strongest suit | Standard bid | With advantage |
|---|---|---|
| 4 | 7 — Marboota (مربوطة) | 7 — Marboota (مربوطة) only |
| 5 | 8 | 7 or 8 |
| 6 | 9 | 8 or 9 |
| 7 | 10 | 9 or 10 |
| 8+ | — | Must declare card-based Dak (دك) |

If all three passed and he does not declare the advantage, the standard Al-Tasmiya (التسمية) formula applies.

**Option 3 — Declare Dak (دك):** only available under the pass-based Dak (دك) conditions described above.

## Al-Ato (الأتو) — The Trump Suit
The trump suit — Al-Ato (الأتو) — is never declared out loud. It is revealed only when the winning bidder plays their very first card — that suit becomes Al-Ato (الأتو) for the entire Shota (شوتة). Al-Ato (الأتو) must be from the suit the bidder holds the most cards in. The first card played must be from the Al-Ato (الأتو) suit.

## Playing a Shota (شوتة)
The player whose bid was accepted leads the first card, which must be from the Al-Ato (الأتو) suit. Players must follow the suit led if they have it. If a player has no cards of the led suit, they may discard any card or play an Al-Ato (الأتو) card. The highest Al-Ato (الأتو) card played always wins the trick, even if only one Al-Ato (الأتو) card was played. If no Al-Ato (الأتو) card was played, the highest card of the led suit wins. The winner of each trick leads the next one. Once a trick is placed face down it cannot be reviewed by anyone.

## Tracking Tricks
Won tricks are placed face down on the table in front of the winning team, visible to all players in count but not in content.

## Scoring
Both teams score at the end of each Shota (شوتة) as follows:

- If the playing team **reaches or exceeds** their bid: they score the actual number of tricks they won. The defending team scores nothing.
- If the playing team **falls short** of their bid: they lose points equal to their bid number — not the number of tricks lost. The defending team scores the actual number of tricks they won.

Scores can go negative — there is no floor.

**Examples:**

| Playing team bid | Tricks won | Playing team score | Defending team score |
|---|---|---|---|
| 8 | 10 | +10 | +0 |
| 8 | 6 | -8 | +6 |
| 8 | 3 | -8 | +10 |

## Winning the Game
A game consists of 5 Shotas (شوتات). The first team to reach 25 points across all 5 Shotas (شوتات) wins the game.

## Seek (سيك)
If a team wins all 13 tricks in a Shota (شوتة), this is called Seek (سيك) — but only after it actually happens, not during Al-Tasmiya (التسمية). 13 is just a number during the bid. If Seek (سيك) happens at any point, even in the very first Shota (شوتة), the game ends immediately and the team that achieved Seek (سيك) is declared the winner of the game — regardless of the score or how many Shotas (شوتات) have been played. No points are counted.

## Wist Game Lifecycle

```text
                         ┌──────────────────────────────────────────┐
                         │                START GAME                │
                         └──────────────────────────────────────────┘
                                             │
                                             ▼
                         Determine first Sahib Al-Qabool
                              (Each player draws one card)
                                             │
                                             ▼
                         ┌──────────────────────────────────────────┐
                         │               START SHOTA                │
                         └──────────────────────────────────────────┘
                                             │
                                             ▼
                                      Shuffle Deck
                                             │
                                             ▼
                                        Cut Deck
                                             │
                                             ▼
                                      Deal 52 Cards
                                             │
                                             ▼
                                   Card-based Dak?
                                  ┌──────────┴──────────┐
                                  │                     │
                                 Yes                    No
                                  │                     │
                                  ▼                     ▼
                        Re-deal Same Shota       Start Bidding
                      (Same Sahib Al-Qabool)           │
                                                       ▼
                                                Bid / Pass
                                                       │
                                                       ▼
                                           Pass-based Dak?
                                  ┌──────────┴──────────┐
                                  │                     │
                                 Yes                    No
                                  │                     │
                                  ▼                     ▼
                         Rotate Sahib Al-Qabool    Sahib Al-Qabool
                                  │                 Decision
                                  │             (Accept / Play)
                                  │                     │
                                  ▼                     ▼
                           START NEW SHOTA      Trump Revealed
                                                      │
                                                      ▼
                                                 Play 13 Tricks
                                                      │
                                                      ▼
                                                 Count Tricks
                                                      │
                                                      ▼
                                                  Seek (13)?
                                  ┌──────────┴──────────┐
                                  │                     │
                                 Yes                    No
                                  │                     │
                                  ▼                     ▼
                              END GAME            Score Shota
                                                       │
                                                       ▼
                                                  Update Game
                                                       │
                                                       ▼
                                             Game Finished?
                                  ┌──────────┴──────────┐
                                  │                     │
                                 Yes                    No
                                  │                     │
                                  ▼                     ▼
                              END GAME        Rotate Sahib Al-Qabool
                                                       │
                                                       ▼
                                                 START NEW SHOTA
```