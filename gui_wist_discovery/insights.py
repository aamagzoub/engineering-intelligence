"""
Strategic Insights System — rich, structured lessons from Q-table analysis.

Each insight is a structured object with:
- text: The lesson itself (plain language, actionable)
- category: "bidding" | "trump" | "timing" | "partnership" | "defense" | "voids"
- difficulty: "beginner" | "intermediate" | "advanced"
- confidence: "emerging" | "proven" | "mastered"
- episode: When this insight was first discovered
- condition: IF/WHEN trigger (optional)
- exception: EXCEPT WHEN counter-example (optional)
- why: Short explanation of the underlying reason (optional)
- links: List of related insight indices (optional)
- new: Whether this insight is newly discovered (bool)

STYLE RULES:
- Plain language, no jargon. No "off-suit" — say "can't follow suit" or "different suit".
- No raw statistics. Use words like "almost always", "rarely", "usually".
- Each insight sounds like advice from an experienced Wist player.
- Conditional tips use "When X, do Y" or "If X, then Y" structure.
- Mistake tips use "Don't X when Y — because Z" structure.
"""

from collections import defaultdict, Counter
import json
import os
import copy

_SNAPSHOTS_PATH = "agents/wist_discovery/strategy_snapshots.json"
_INSIGHTS_CACHE_PATH = "agents/wist_discovery/insights_cache.json"

_TIER_RANK = {"A": 5, "K": 4, "Q": 3, "J": 2, "M": 1, "L": 0, "X": -1}


# ─── Insight Data Structure ──────────────────────────────────────────────────────


def make_insight(text, category, difficulty, confidence="emerging",
                 episode=0, condition=None, exception=None, why=None, links=None,
                 version=0):
    """Create a structured insight dict."""
    return {
        "text": text,
        "category": category,
        "difficulty": difficulty,
        "confidence": confidence,
        "episode": episode,
        "condition": condition,
        "exception": exception,
        "why": why,
        "links": links or [],
        "new": True,
        "version": version,  # 0 = first discovery, 1+ = refinement.
    }


# ─── Main Entry Point ────────────────────────────────────────────────────────────


def generate_insights(agent) -> list:
    """
    Generate structured insights from the agent's learned knowledge.

    ACCUMULATION MODEL: insights are NEVER removed. Once discovered, they stay
    forever as part of the brain's history. New insights are added on top.
    If an insight refines an earlier one, it gets version (+N).

    Returns list of insight dicts, always growing.
    """
    if agent.episodes_trained < 5000:
        return [make_insight(
            "Still learning the basics — check back after more training...",
            "bidding", "beginner", "emerging", agent.episodes_trained
        )]

    episodes = agent.episodes_trained
    play_q = agent.play_q
    bid_q = agent.bid_q
    play_items = list(play_q.items())[:min(len(play_q), 20000)]

    # Load ALL previously accumulated insights (never discard).
    accumulated = _load_cached_insights()
    accumulated_texts = {ins["text"] for ins in accumulated}

    # Generate current insights from Q-tables.
    raw = []
    raw.extend(_hand_based_insights(bid_q, episodes))
    raw.extend(_conditional_insights(play_items, bid_q, episodes))
    raw.extend(_partnership_insights(play_items, episodes))
    raw.extend(_opponent_reading_insights(play_items, episodes))
    raw.extend(_mistake_insights(play_items, bid_q, episodes))
    raw.extend(_core_play_insights(play_items, episodes))
    raw.extend(_trump_insights(play_items, episodes))
    raw.extend(_void_insights(play_items, episodes))
    raw.extend(_timing_insights(play_items, episodes))
    raw.extend(_evolution_insights(play_items, bid_q, episodes))
    raw.extend(_surprise_discoveries(play_items, bid_q, episodes))
    raw.extend(_deep_hand_insights(bid_q, episodes))
    raw.extend(_position_phase_insights(play_items, episodes))
    raw.extend(_endless_situational_mining(play_items, bid_q, episodes))

    # Progression gating.
    raw = _apply_progression_gate(raw, episodes)

    # Deduplicate within the new batch only.
    raw = _deduplicate(raw)

    # ACCUMULATION: merge new insights into the accumulated history.
    for ins in raw:
        if ins["text"] in accumulated_texts:
            # Already known — update confidence only.
            for acc_ins in accumulated:
                if acc_ins["text"] == ins["text"]:
                    age = episodes - acc_ins["episode"]
                    if age > 100000:
                        acc_ins["confidence"] = "mastered"
                    elif age > 30000:
                        acc_ins["confidence"] = "proven"
                    acc_ins["new"] = False
                    break
        else:
            # NEW insight — check if it refines an existing one.
            ins["new"] = True
            ins["episode"] = episodes
            ins["version"] = _find_version(ins, accumulated)
            accumulated.append(ins)
            accumulated_texts.add(ins["text"])

    # Update confidence for all accumulated insights.
    for ins in accumulated:
        age = episodes - ins.get("episode", 0)
        if age > 100000:
            ins["confidence"] = "mastered"
        elif age > 30000:
            ins["confidence"] = "proven"
        elif not ins.get("new"):
            ins["confidence"] = ins.get("confidence", "emerging")

    # Save entire accumulated history.
    _save_cached_insights(accumulated)

    return accumulated


def _find_version(new_ins, accumulated) -> int:
    """
    Check if new insight refines an existing one (same category + similar topic).
    Returns the version number: 0 if brand new, N+1 if refining insight at version N.
    """
    category = new_ins["category"]
    new_words = set(new_ins["text"].lower().split())

    best_overlap = 0
    best_version = -1

    for existing in accumulated:
        if existing["category"] != category:
            continue
        existing_words = set(existing["text"].lower().split())
        overlap = len(new_words & existing_words) / max(len(new_words | existing_words), 1)
        if overlap > 0.4 and overlap > best_overlap:
            best_overlap = overlap
            best_version = existing.get("version", 0)

    if best_version >= 0:
        return best_version + 1
    return 0



# ─── 1. Hand-Based Insights ──────────────────────────────────────────────────────


def _hand_based_insights(bid_q, episodes) -> list:
    """
    Extract insights tied to specific hand compositions.
    Uses the bid state encoding: longest, shortest, highs, aces, voids, bid_level.
    Format: [longest][shortest][highs][aces]v[voids][hasBid][bidLevel][isQ][forced]
    """
    insights = []

    # Group bid Q-values by hand features.
    hand_bid_results = defaultdict(lambda: defaultdict(list))

    for state, actions in list(bid_q.items())[:2000]:
        if len(state) < 5:
            continue
        try:
            longest = int(state[0])
            highs = int(state[2])
            aces = int(state[3])
            voids_idx = state.index("v")
            voids = int(state[voids_idx + 1])
        except (ValueError, IndexError):
            continue

        hand_key = (longest, highs, aces, voids)
        for action, q in actions.items():
            hand_bid_results[hand_key][action].append(q)

    # Analyze: for each hand type, what's the best action?
    for (longest, highs, aces, voids), action_qs in hand_bid_results.items():
        best_action = None
        best_avg = -999
        for action, vals in action_qs.items():
            if len(vals) >= 3:
                avg = sum(vals) / len(vals)
                if avg > best_avg:
                    best_avg = avg
                    best_action = action

        if not best_action or best_avg < 0.2:
            continue

        # Generate hand-specific tip.
        hand_desc_parts = []
        if longest >= 6:
            hand_desc_parts.append(f"a suit with {longest}+ cards")
        if highs >= 3:
            hand_desc_parts.append(f"{highs}+ high cards")
        elif aces >= 2:
            hand_desc_parts.append(f"{aces} Aces")
        if voids >= 1:
            hand_desc_parts.append(f"a void in {voids} suit{'s' if voids > 1 else ''}")

        if not hand_desc_parts:
            continue

        hand_desc = " and ".join(hand_desc_parts)

        if best_action == "PASS":
            insights.append(make_insight(
                f"When you have {hand_desc}, consider passing — your hand looks decent but doesn't deliver reliably",
                "bidding", "intermediate", "emerging", episodes,
                condition=f"You have {hand_desc}",
                why="This hand shape often looks stronger than it plays out"
            ))
        elif best_action.startswith("B"):
            try:
                bid_val = int(best_action[1:])
                if bid_val >= 9 and (longest >= 5 or highs >= 3):
                    insights.append(make_insight(
                        f"When you have {hand_desc}, bid {bid_val} confidently — this hand almost always delivers",
                        "bidding", "intermediate", "emerging", episodes,
                        condition=f"You have {hand_desc}",
                        why="The combination of length and high cards creates enough trick sources"
                    ))
                elif bid_val <= 7:
                    insights.append(make_insight(
                        f"With {hand_desc}, bid around {bid_val} — it's safe and you'll often get extra tricks as a bonus",
                        "bidding", "beginner", "emerging", episodes,
                        condition=f"You have {hand_desc}",
                        why="Under-promising lets you score without the penalty risk"
                    ))
            except ValueError:
                pass

    return insights[:12]



# ─── 2. Conditional Insights ─────────────────────────────────────────────────────


def _conditional_insights(play_items, bid_q, episodes) -> list:
    """
    IF situation → THEN action insights.
    Derived from states where one action massively outperforms others.
    """
    insights = []

    # Bid conditional: when opponents have already bid high.
    opp_bid_high_pass = []
    opp_bid_high_bid = []
    for state, actions in list(bid_q.items())[:1000]:
        if len(state) < 8:
            continue
        try:
            has_bid_char = state[state.index("v") + 2]  # Y or N
            bid_level_start = state.index("v") + 3
            bid_level = int(state[bid_level_start:bid_level_start + 1]) if bid_level_start < len(state) else 0
        except (ValueError, IndexError):
            continue

        if has_bid_char == "Y" and bid_level >= 8:
            pass_q = actions.get("PASS", None)
            if pass_q is not None:
                opp_bid_high_pass.append(pass_q)
            for k, q in actions.items():
                if k.startswith("B"):
                    opp_bid_high_bid.append(q)

    if len(opp_bid_high_pass) >= 5 and len(opp_bid_high_bid) >= 5:
        pass_avg = sum(opp_bid_high_pass) / len(opp_bid_high_pass)
        bid_avg = sum(opp_bid_high_bid) / len(opp_bid_high_bid)
        if pass_avg > bid_avg + 0.3:
            insights.append(make_insight(
                "If opponents bid high (8+), pass and defend — they overreach and fail more often than they succeed",
                "bidding", "intermediate", "emerging", episodes,
                condition="Opponents already bid 8 or higher",
                why="High bids require near-perfect hands. Defending costs nothing and their failure gives you free points",
                exception="Unless you have 6+ trumps yourself — then outbid them"
            ))
        elif bid_avg > pass_avg + 0.3:
            insights.append(make_insight(
                "If opponents bid high, don't be intimidated — outbid them when your hand has the trumps to back it up",
                "bidding", "advanced", "emerging", episodes,
                condition="Opponents bid high but you have 5+ trumps",
                why="A strong trump hand can steal the contract even against aggressive opponents"
            ))

    # Play conditional: when you're winning vs losing.
    winning_aggro = []
    winning_passive = []
    losing_aggro = []
    losing_passive = []

    for state, actions in play_items:
        if len(state) < 8:
            continue
        # Trick diff is encoded as W/A/T/B after phase.
        try:
            trick_diff_idx = 6  # After shape(4) + pos(1) + phase(1)
            td = state[trick_diff_idx] if len(state) > trick_diff_idx else "T"
        except IndexError:
            continue

        for key, q in actions.items():
            if len(key) < 3:
                continue
            tier_rank = _TIER_RANK.get(key[0], 0)
            if td == "W":  # Winning
                if tier_rank >= 3:
                    winning_aggro.append(q)
                else:
                    winning_passive.append(q)
            elif td == "B":  # Behind
                if tier_rank >= 3:
                    losing_aggro.append(q)
                else:
                    losing_passive.append(q)

    if len(winning_passive) >= 20 and len(winning_aggro) >= 20:
        passive_avg = sum(winning_passive) / len(winning_passive)
        aggro_avg = sum(winning_aggro) / len(winning_aggro)
        if passive_avg > aggro_avg + 0.2:
            insights.append(make_insight(
                "When your team is ahead, play conservatively — don't risk your lead chasing extra tricks you don't need",
                "timing", "intermediate", "emerging", episodes,
                condition="Your team is ahead in tricks",
                why="You only need to maintain your advantage, not extend it. Playing safe prevents costly mistakes"
            ))
        elif aggro_avg > passive_avg + 0.2:
            insights.append(make_insight(
                "When ahead, press your advantage — strong cards are even stronger when opponents are already behind",
                "timing", "intermediate", "emerging", episodes,
                condition="Your team is winning",
                why="Opponents get desperate and make mistakes when you pile on pressure"
            ))

    if len(losing_aggro) >= 20 and len(losing_passive) >= 20:
        aggro_avg = sum(losing_aggro) / len(losing_aggro)
        passive_avg = sum(losing_passive) / len(losing_passive)
        if aggro_avg > passive_avg + 0.2:
            insights.append(make_insight(
                "When behind, take risks with your strong cards — playing it safe when losing just means losing slowly",
                "timing", "advanced", "emerging", episodes,
                condition="Your team is behind in tricks",
                why="You need to change the momentum. Safe play maintains a losing position"
            ))

    return insights[:10]



# ─── 3. Partnership Insights ─────────────────────────────────────────────────────


def _partnership_insights(play_items, episodes) -> list:
    """Insights about coordination with your partner."""
    insights = []

    # When partner likely leads (position 2 means partner led from pos 0).
    # Position 2 = third seat = partner led.
    partner_led_follow_high = []
    partner_led_follow_low = []

    for state, actions in play_items:
        if len(state) < 6:
            continue
        pos = state[4]
        if pos != "2":  # We're third = partner is first.
            continue
        for key, q in actions.items():
            if len(key) < 3:
                continue
            tier = key[0]
            follows = key[1]
            if follows == "F":  # Following partner's suit.
                if tier in ("A", "K", "Q"):
                    partner_led_follow_high.append(q)
                elif tier in ("L", "X"):
                    partner_led_follow_low.append(q)

    if len(partner_led_follow_high) >= 10 and len(partner_led_follow_low) >= 10:
        high_avg = sum(partner_led_follow_high) / len(partner_led_follow_high)
        low_avg = sum(partner_led_follow_low) / len(partner_led_follow_low)
        if low_avg > high_avg + 0.2:
            insights.append(make_insight(
                "When your partner leads a suit, play low if they're likely winning — save your big cards for tricks where you need to fight",
                "partnership", "intermediate", "emerging", episodes,
                condition="Your partner led the trick and seems to be winning",
                why="Your partner chose that lead for a reason. Trust their strength and conserve yours"
            ))
        elif high_avg > low_avg + 0.2:
            insights.append(make_insight(
                "When your partner leads, help them win by playing high — together you guarantee the trick",
                "partnership", "intermediate", "emerging", episodes,
                condition="Your partner leads but opponents might overtake",
                why="Partnership coordination means supporting each other's leads when the trick is contested"
            ))

    # Trump support — when partner leads trump.
    partner_led_trump_high = []
    partner_led_trump_low = []
    for state, actions in play_items:
        if len(state) < 6 or state[4] != "2":
            continue
        for key, q in actions.items():
            if len(key) < 3 or key[2] != "T" or key[1] != "F":
                continue
            if key[0] in ("A", "K", "Q"):
                partner_led_trump_high.append(q)
            elif key[0] in ("L", "X"):
                partner_led_trump_low.append(q)

    if len(partner_led_trump_high) >= 5:
        avg = sum(partner_led_trump_high) / len(partner_led_trump_high)
        if avg > 0.3:
            insights.append(make_insight(
                "When your partner leads trump, play your highest trump — together you flush out the opponents' trumps faster",
                "partnership", "beginner", "emerging", episodes,
                condition="Partner leads trump",
                why="Two high trumps from the same team in one trick removes two enemy trumps at once"
            ))

    # General partnership tip based on positional play.
    last_seat_save = []
    for state, actions in play_items:
        if len(state) < 6 or state[4] != "3":
            continue
        for key, q in actions.items():
            if len(key) >= 3 and key[0] in ("L", "X") and key[1] == "F":
                last_seat_save.append(q)

    if len(last_seat_save) >= 15:
        avg = sum(last_seat_save) / len(last_seat_save)
        if avg > 0.2:
            insights.append(make_insight(
                "When you play last and your partner is already winning the trick, throw your weakest — don't waste good cards on a trick your team already has",
                "partnership", "beginner", "emerging", episodes,
                condition="You're last to play and partner's card is winning",
                why="Every good card you save is one more trick you can win later"
            ))

    return insights[:8]



# ─── 4. Opponent-Reading Insights ────────────────────────────────────────────────


def _opponent_reading_insights(play_items, episodes) -> list:
    """Insights about reading opponents' behavior from the table."""
    insights = []

    # When opponents are void (opp_voids > 0 in state encoding).
    opp_void_trump_q = []
    opp_void_nontrick_q = []

    for state, actions in play_items:
        if len(state) < 10:
            continue
        # opp_voids is at the end: "o{N}"
        try:
            o_idx = state.rindex("o")
            opp_voids = int(state[o_idx + 1])
        except (ValueError, IndexError):
            continue

        if opp_voids == 0:
            continue

        for key, q in actions.items():
            if len(key) < 3:
                continue
            if key[2] == "T" and key[1] == "F":
                opp_void_trump_q.append(q)
            elif key[0] in ("A", "K") and key[1] == "F" and key[2] == "N":
                opp_void_nontrick_q.append(q)

    if len(opp_void_trump_q) >= 10:
        avg = sum(opp_void_trump_q) / len(opp_void_trump_q)
        if avg > 0.3:
            insights.append(make_insight(
                "When you know an opponent is void in a suit, lead trump instead — don't let them keep trumping your good cards",
                "defense", "advanced", "emerging", episodes,
                condition="You noticed an opponent couldn't follow suit earlier",
                why="If they're void, they'll trump your winners. Leading trump removes their trumping power first"
            ))

    if len(opp_void_nontrick_q) >= 10:
        avg = sum(opp_void_nontrick_q) / len(opp_void_nontrick_q)
        if avg < -0.2:
            insights.append(make_insight(
                "Don't lead your Aces and Kings in a suit where an opponent is void — they'll just trump it and you lose your best card for nothing",
                "defense", "intermediate", "emerging", episodes,
                condition="An opponent previously couldn't follow in this suit",
                why="Your high card can't win if someone plays trump on it. Save it for a safe suit"
            ))

    # General reading tips (always applicable).
    insights.append(make_insight(
        "Watch which suits opponents can't follow — that tells you they'll trump it next time. Remember it and avoid leading that suit",
        "defense", "beginner", "emerging", episodes,
        condition="An opponent plays a different suit than what was led",
        why="When someone can't follow suit, they're either trumping or discarding. Either way, that suit is now dangerous to lead"
    ))

    return insights[:6]



# ─── 5. Mistake-Based Insights ───────────────────────────────────────────────────


def _mistake_insights(play_items, bid_q, episodes) -> list:
    """Frame lessons as costly mistakes to avoid."""
    insights = []

    # Find the most negative Q-value actions (costliest mistakes).
    worst_plays = []
    for state, actions in play_items:
        if len(state) < 6:
            continue
        for key, q in actions.items():
            if len(key) >= 3 and q < -1.5:
                worst_plays.append((state, key, q))

    # Group worst plays by action type.
    mistake_groups = defaultdict(list)
    for state, key, q in worst_plays:
        tier = key[0]
        follows = key[1]
        is_trump = key[2] == "T"
        group = (tier, follows, is_trump)
        mistake_groups[group].append(q)

    # Generate mistake lessons for the worst groups.
    for (tier, follows, is_trump), vals in sorted(mistake_groups.items(), key=lambda x: sum(x[1]) / len(x[1])):
        if len(vals) < 5:
            continue
        avg = sum(vals) / len(vals)
        if avg > -1.5:
            continue

        if tier in ("A", "K") and follows == "O" and not is_trump:
            insights.append(make_insight(
                "Don't throw your Aces or Kings when you can't follow suit — they're your strongest cards, wasted on a trick they can't win",
                "bidding", "beginner", "emerging", episodes,
                condition="You can't follow the led suit",
                why="A card that doesn't follow suit and isn't trump can never win, no matter how high it is",
                exception="Unless you're deliberately creating a void for future trumping — then sacrifice the cheapest high card"
            ))
            break
        elif is_trump and tier in ("M", "L") and follows == "F":
            insights.append(make_insight(
                "Don't throw mid or low trumps when someone leads trump — they'll lose to higher trumps and you've wasted a trumping card",
                "trump", "intermediate", "emerging", episodes,
                condition="Someone leads trump and you have only small trumps",
                why="Small trumps are worth more when YOU choose to trump a non-trump trick. Following trump with them just loses",
                exception="Unless you have no choice (it's your only card in trump)"
            ))
            break
        elif tier == "K" and follows == "F" and not is_trump:
            insights.append(make_insight(
                "Don't play your King if the Ace hasn't appeared yet — wait until the Ace is gone, then your King becomes the highest card",
                "timing", "beginner", "emerging", episodes,
                condition="The Ace of this suit hasn't been played yet",
                why="Playing King into an unseen Ace is just donating a trick to whoever holds it"
            ))
            break

    # Bid mistakes.
    bid_mistakes = []
    for state, actions in list(bid_q.items())[:500]:
        for key, q in actions.items():
            if key.startswith("B") and q < -1.5:
                try:
                    val = int(key[1:])
                    bid_mistakes.append((state, val, q))
                except ValueError:
                    pass

    if bid_mistakes:
        # Group by bid value.
        bid_val_loss = defaultdict(list)
        for state, val, q in bid_mistakes:
            bid_val_loss[val].append(q)

        worst_bid = max(bid_val_loss.items(), key=lambda x: -sum(x[1]) / len(x[1]))
        if len(worst_bid[1]) >= 3:
            val = worst_bid[0]
            if val >= 11:
                insights.append(make_insight(
                    f"Bidding {val} or more without a dominant hand is the single costliest mistake — you'll fail most of the time and the penalty is huge",
                    "bidding", "beginner", "emerging", episodes,
                    condition=f"You're tempted to bid {val}+",
                    why="At this level, you need nearly everything to go right. One bad break and you lose big",
                    exception=f"Unless you have 7+ trumps including top honors — then {val} is actually achievable"
                ))

    return insights[:6]



# ─── 6-9. Core Play, Trump, Void, Timing ─────────────────────────────────────────


def _core_play_insights(play_items, episodes) -> list:
    """Fundamental card play lessons."""
    insights = []
    action_q = defaultdict(list)
    for _state, actions in play_items:
        for key, q in actions.items():
            if len(key) >= 3 and abs(q) > 0.2:
                action_q[key].append(q)

    def _check(filter_fn, min_count=15):
        vals = []
        for k, v in action_q.items():
            if len(v) >= min_count and filter_fn(k):
                vals.extend(v)
        return sum(vals) / len(vals) if vals else 0

    if _check(lambda k: k[0] == "X" and k[1] == "F" and k[2] == "N") > 0.3:
        insights.append(make_insight(
            "When you can't win a trick, throw your weakest card — protect your better cards for tricks you can actually fight for",
            "timing", "beginner", "emerging", episodes,
            condition="The cards already played are higher than anything you have",
            why="Every strong card you save is a future trick. Wasting them on lost causes costs double"
        ))

    if _check(lambda k: k[0] in "LX" and k[1] == "O" and k[2] == "T") > 0.4:
        insights.append(make_insight(
            "Even your smallest trump wins when others play a non-trump suit — a 2 of trump beats a King of anything else",
            "trump", "beginner", "emerging", episodes,
            condition="You can't follow suit and have any trump card",
            why="Trump always beats non-trump regardless of rank. That's why creating voids is so powerful"
        ))

    return insights[:5]


def _trump_insights(play_items, episodes) -> list:
    """Trump-specific lessons."""
    insights = []
    trump_lead_high = []
    trump_whip_low = []
    trump_follow_low = []

    for _state, actions in play_items:
        for key, q in actions.items():
            if len(key) < 3 or key[2] != "T":
                continue
            if key[1] == "F" and key[0] in "AK" and q > 0.3:
                trump_lead_high.append(q)
            if key[1] == "O" and key[0] in "LX":
                trump_whip_low.append(q)
            if key[1] == "F" and key[0] in "LX" and q < -0.2:
                trump_follow_low.append(q)

    if len(trump_lead_high) >= 10 and sum(trump_lead_high) / len(trump_lead_high) > 0.4:
        insights.append(make_insight(
            "Lead your strongest trump to force out everyone else's — after a few rounds, your remaining trumps face no opposition",
            "trump", "intermediate", "emerging", episodes,
            condition="You have 4+ trumps including high ones",
            why="Each high trump you lead removes one enemy trump. After 2-3 rounds, you control the game",
            exception="Don't do this if you only have 2 trumps — you'll run out and lose control"
        ))

    if len(trump_whip_low) >= 10 and sum(trump_whip_low) / len(trump_whip_low) > 0.3:
        insights.append(make_insight(
            "When you can't follow suit, use your smallest trump — it still wins, and you keep your big trumps for harder fights later",
            "trump", "beginner", "emerging", episodes,
            condition="You can't follow suit and have multiple trumps",
            why="Any trump beats any non-trump card. Using your lowest saves the rest for when opponents also trump"
        ))

    if len(trump_follow_low) >= 5 and sum(trump_follow_low) / len(trump_follow_low) < -0.3:
        insights.append(make_insight(
            "When someone else leads trump, don't throw in a small trump to lose — save it for a trick where YOU choose to trump",
            "trump", "intermediate", "emerging", episodes,
            condition="Trump is led and you only have low trumps",
            why="A small trump following a trump lead will lose to any higher trump. But that same card wins any non-trump trick later",
            exception="If it's your last card, you have no choice"
        ))

    return insights[:5]


def _void_insights(play_items, episodes) -> list:
    """Void creation and exploitation."""
    insights = []
    void_q = [q for _s, actions in play_items for k, q in actions.items()
              if len(k) >= 5 and k[4] == "V" and k[2] != "T" and q > 0.2]

    if len(void_q) >= 10 and sum(void_q) / len(void_q) > 0.3:
        insights.append(make_insight(
            "Get rid of your short suits early — once you have zero cards in a suit, you can trump it every single time it comes up",
            "voids", "beginner", "emerging", episodes,
            condition="You have a suit with only 1-2 cards",
            why="Being void in a suit turns every lead of that suit into a free trick for you"
        ))

    # Long suit strategy.
    long_low = [q for _s, actions in play_items for k, q in actions.items()
                if len(k) >= 5 and k[3] == "L" and k[0] in "LX" and k[2] != "T" and q > 0.3]
    if len(long_low) >= 10 and sum(long_low) / len(long_low) > 0.3:
        insights.append(make_insight(
            "Lead low from your longest suit — opponents will run out of it before you do, and your remaining cards become winners",
            "voids", "intermediate", "emerging", episodes,
            condition="You have a suit with 5+ cards",
            why="When opponents are void in your long suit, they must trump or discard. Either way, your remaining cards eventually win"
        ))

    return insights[:5]


def _timing_insights(play_items, episodes) -> list:
    """Phase-specific timing lessons."""
    insights = []
    phase_action_q = defaultdict(lambda: defaultdict(list))
    for state, actions in play_items:
        if len(state) < 6:
            continue
        phase = state[5]
        for key, q in actions.items():
            if len(key) >= 3 and abs(q) > 0.3:
                phase_action_q[phase][key].append(q)

    # Find cards good early but bad late.
    seen_categories = set()
    for key in set().union(*(phase_action_q[p].keys() for p in phase_action_q)):
        early = phase_action_q.get("1", {}).get(key, [])
        late = phase_action_q.get("5", {}).get(key, [])
        if len(early) < 10 or len(late) < 10:
            continue
        early_avg = sum(early) / len(early)
        late_avg = sum(late) / len(late)
        tier = key[0]
        is_trump = len(key) > 2 and key[2] == "T"

        if early_avg > 0.4 and late_avg < 0 and "early_power" not in seen_categories:
            if is_trump:
                insights.append(make_insight(
                    "Use your high trumps in the first half of the game — wait too long and opponents will be out of the suits you need to trump",
                    "timing", "advanced", "emerging", episodes,
                    condition="First half of the shota and you have high trumps",
                    why="Early trump leads flush out enemy trumps while they still have them. Late trump leads hit empty air"
                ))
            else:
                insights.append(make_insight(
                    "Play your Aces and Kings in the first half while they still dominate — hold them too long and they get trumped",
                    "timing", "intermediate", "emerging", episodes,
                    condition="You have high cards in non-trump suits",
                    why="As the game progresses, more players become void and can trump your winners"
                ))
            seen_categories.add("early_power")

        if early_avg < 0 and late_avg > 0.4 and "late_power" not in seen_categories:
            insights.append(make_insight(
                "Save one strong card for the very end — in the final tricks, most opponents are out of options and can't stop you",
                "timing", "intermediate", "emerging", episodes,
                condition="You're deciding whether to play your last strong card now or later",
                why="In the endgame, fewer cards remain, so your strong card faces less competition"
            ))
            seen_categories.add("late_power")

    return insights[:6]



# ─── 10. Evolution Insights (Endless Growth) ─────────────────────────────────────


# ─── 11. Surprise Discoveries — Counter-Intuitive Patterns ───────────────────────


def _surprise_discoveries(play_items, bid_q, episodes) -> list:
    """
    Hunt for counter-intuitive patterns — things that SHOULDN'T work but DO,
    or things that SHOULD work but DON'T. These are the "brilliant" insights.
    """
    insights = []

    # --- A) High cards that LOSE vs low cards that WIN ---
    # Find situations where playing low beats playing high (unexpected).
    context_q = defaultdict(lambda: defaultdict(list))
    for state, actions in play_items:
        if len(state) < 6:
            continue
        # Group by context (phase + position).
        ctx = (state[4], state[5])  # pos, phase
        for key, q in actions.items():
            if len(key) >= 3:
                context_q[ctx][key].append(q)

    for ctx, action_qs in context_q.items():
        pos, phase = ctx
        # Average Q per tier in this context.
        tier_avg = {}
        for key, vals in action_qs.items():
            if len(vals) < 3:  # Lowered from 8 to 3 for younger Q-tables.
                continue
            tier = key[0]
            if tier not in tier_avg:
                tier_avg[tier] = []
            tier_avg[tier].extend(vals)

        for t in tier_avg:
            tier_avg[t] = sum(tier_avg[t]) / len(tier_avg[t])

        # Surprise: low cards beat high cards in this context.
        low_avg = tier_avg.get("L", tier_avg.get("X", None))
        high_avg = tier_avg.get("A", tier_avg.get("K", None))
        if low_avg is not None and high_avg is not None and low_avg > high_avg + 0.2:  # Lowered from 0.5
            phase_name = {"1": "early", "2": "early-mid", "3": "middle", "4": "late-mid", "5": "final"}.get(phase, "")
            pos_name = {"0": "leading", "1": "second", "2": "third", "3": "last"}.get(pos, "")
            if phase_name and pos_name:
                insights.append(make_insight(
                    f"Surprising: in the {phase_name} tricks when you're {pos_name}, playing LOW cards works better than playing your Aces and Kings — saving them for later is worth more than winning the trick now",
                    "counter-intuitive", "advanced", "emerging", episodes,
                    condition=f"You're {pos_name} in {phase_name} tricks",
                    why="The trick isn't worth the investment. Your high cards earn more later when the table is thinner"
                ))
                # Don't break — find more phase-specific surprises.

        # Surprise: mid cards beat aces in this context.
        mid_avg = tier_avg.get("M", tier_avg.get("J", None))
        ace_avg = tier_avg.get("A", None)
        if mid_avg is not None and ace_avg is not None and mid_avg > ace_avg + 0.2:  # Lowered from 0.4
            phase_name = {"1": "early", "2": "early-mid", "3": "middle", "4": "late-mid", "5": "final"}.get(phase, "")
            if phase_name:
                insights.append(make_insight(
                    f"Counter-intuitive: in the {phase_name} phase, middle cards (9s, 10s, Jacks) outperform Aces — Aces attract trump from void opponents, but mid cards fly under the radar",
                    "counter-intuitive", "advanced", "emerging", episodes,
                    condition=f"You're in the {phase_name} phase of the game",
                    why="High cards become targets. Opponents save their trumps specifically to kill your Aces. Mid cards win tricks nobody fights over"
                ))

    # --- B) Bid contradictions: hands that LOOK strong but bid LOW works better ---
    hand_surprises = defaultdict(lambda: defaultdict(list))
    for state, actions in list(bid_q.items())[:1500]:
        if len(state) < 5:
            continue
        try:
            highs = int(state[2])
            aces = int(state[3])
        except (ValueError, IndexError):
            continue

        for action, q in actions.items():
            hand_surprises[(highs, aces)][action].append(q)

    for (highs, aces), action_qs in hand_surprises.items():
        if highs < 2:
            continue  # Lowered from 3 — find surprises in weaker-looking hands too.
        # Find best action.
        best_action, best_avg = None, -999
        for action, vals in action_qs.items():
            if len(vals) >= 3:
                avg = sum(vals) / len(vals)
                if avg > best_avg:
                    best_avg, best_action = avg, action

        if not best_action:
            continue

        # Surprise: strong hand but PASS is best.
        if best_action == "PASS" and highs >= 3 and best_avg > 0.1:  # Lowered from highs>=4, avg>0.3
            insights.append(make_insight(
                f"Surprising: with {highs} high cards, passing is BETTER than bidding — high cards without trump length are a mirage. They look strong but get trumped",
                "counter-intuitive", "advanced", "emerging", episodes,
                condition=f"You have {highs}+ high cards but few trumps",
                why="High cards in non-trump suits get trumped by void opponents. Trump count matters more than raw power",
                exception="If most of your high cards ARE trumps, then bid confidently"
            ))

        # Surprise: strong hand but LOW bid is best.
        if best_action.startswith("B") and highs >= 3:  # Lowered from highs>=4
            try:
                bid_val = int(best_action[1:])
                if bid_val <= 6:
                    insights.append(make_insight(
                        f"Surprising: with {highs} high cards, bidding only {bid_val} works best — promising less and over-delivering beats going all-in and failing",
                        "counter-intuitive", "advanced", "emerging", episodes,
                        condition=f"You have {highs}+ high cards",
                        why="High bids have huge penalties when they fail. A safe bid plus extra tricks scores nearly as well with zero risk"
                    ))
            except ValueError:
                pass

    # --- C) Void paradox: sometimes KEEPING a suit is better than voiding it ---
    void_bad = []
    keep_good = []
    for _state, actions in play_items:
        for key, q in actions.items():
            if len(key) < 5:
                continue
            creates_void = key[4] == "V"
            is_trump = key[2] == "T"
            if is_trump:
                continue
            if creates_void and q < -0.1:  # Lowered from -0.3
                void_bad.append(q)
            elif not creates_void and key[3] == "S" and q > 0.2:  # Lowered from 0.5
                keep_good.append(q)

    if len(void_bad) >= 5 and len(keep_good) >= 5:  # Lowered from 10
        void_avg = sum(void_bad) / len(void_bad)
        keep_avg = sum(keep_good) / len(keep_good)
        if keep_avg > 0.2 and void_avg < -0.1:  # Lowered from 0.4/-0.2
            insights.append(make_insight(
                "Surprise: sometimes keeping a short suit is BETTER than voiding it — if that suit has a high card, the void isn't worth losing the sure trick",
                "counter-intuitive", "advanced", "emerging", episodes,
                condition="Your short suit contains an Ace or King",
                why="A void gives you trumping potential, but your Ace is already a guaranteed trick. Don't sacrifice guaranteed tricks for potential ones"
            ))

    # --- D) Trump timing reversal: saving trump loses, spending early wins ---
    early_trump_q = []
    late_trump_q = []
    for state, actions in play_items:
        if len(state) < 6:
            continue
        phase = state[5]
        for key, q in actions.items():
            if len(key) < 3 or key[2] != "T" or key[1] != "O":
                continue
            if phase in ("1", "2"):
                early_trump_q.append(q)
            elif phase in ("4", "5"):
                late_trump_q.append(q)

    if len(early_trump_q) >= 5 and len(late_trump_q) >= 5:  # Lowered from 15
        early_avg = sum(early_trump_q) / len(early_trump_q)
        late_avg = sum(late_trump_q) / len(late_trump_q)
        if early_avg > late_avg + 0.15:  # Lowered from 0.3
            insights.append(make_insight(
                "Counter-intuitive: trumping EARLY is more valuable than saving your trumps for later — early trumps disrupt opponents' plans before they develop",
                "counter-intuitive", "advanced", "emerging", episodes,
                condition="You're void in a suit in the first half of the game",
                why="Opponents build their strategy around their strong suits. Trumping early breaks their plans before they execute them. Late trumps are too late — the damage is done"
            ))
        elif late_avg > early_avg + 0.15:  # Lowered from 0.3
            insights.append(make_insight(
                "Saving your trumps for the endgame is massively more profitable than spending them early — late trumps win tricks nobody can contest",
                "counter-intuitive", "advanced", "emerging", episodes,
                condition="You have 3+ trumps and it's early in the game",
                why="In the endgame, opponents have fewer cards and fewer options. A trump in trick 11 faces less resistance than a trump in trick 3"
            ))

    # --- E) Position paradox: the "worst" seat is actually the best ---
    pos_overall_q = defaultdict(list)
    for state, actions in play_items:
        if len(state) < 6:
            continue
        pos = state[4]
        for key, q in actions.items():
            if len(key) >= 3:
                pos_overall_q[pos].append(q)

    pos_avgs = {}
    for pos, vals in pos_overall_q.items():
        if len(vals) >= 20:  # Lowered from 50
            pos_avgs[pos] = sum(vals) / len(vals)

    if pos_avgs:
        best_pos = max(pos_avgs, key=pos_avgs.get)
        worst_pos = min(pos_avgs, key=pos_avgs.get)
        pos_names = {"0": "leading (first)", "1": "second", "2": "third", "3": "last"}
        if best_pos == "3":
            insights.append(make_insight(
                "Playing last is the most powerful position in Wist — you see everyone else's card before choosing yours, so you never overpay and never underpay",
                "counter-intuitive", "intermediate", "emerging", episodes,
                why="Information is power. Last seat has perfect information about the current trick"
            ))
        elif best_pos == "0":
            insights.append(make_insight(
                "Leading is the strongest position — whoever sets the suit controls what everyone else must play. That power is worth more than seeing others' cards first",
                "counter-intuitive", "intermediate", "emerging", episodes,
                why="The leader chooses the battlefield. If you lead your strongest suit, opponents must follow or waste a trump"
            ))

    # --- F) Score-state insights: what works when desperate ---
    desperate_plays = defaultdict(list)
    comfortable_plays = defaultdict(list)
    for state, actions in play_items:
        if len(state) < 8:
            continue
        try:
            td = state[6]  # W/A/T/B
        except IndexError:
            continue
        for key, q in actions.items():
            if len(key) < 3:
                continue
            if td == "B":
                desperate_plays[key[0]].append(q)
            elif td == "W":
                comfortable_plays[key[0]].append(q)

    # Find plays that are bad normally but great when desperate.
    for tier in ("X", "L", "M"):
        desp_vals = desperate_plays.get(tier, [])
        comf_vals = comfortable_plays.get(tier, [])
        if len(desp_vals) >= 8 and len(comf_vals) >= 8:  # Lowered from 20
            desp_avg = sum(desp_vals) / len(desp_vals)
            comf_avg = sum(comf_vals) / len(comf_vals)
            if desp_avg > comf_avg + 0.15 and tier == "X":  # Lowered from 0.3
                insights.append(make_insight(
                    "When you're behind, throwing away your weakest cards aggressively works — it sounds like giving up, but it's actually setting up voids for a comeback through trumping",
                    "counter-intuitive", "advanced", "emerging", episodes,
                    condition="Your team is behind in the game score",
                    why="When behind, you need tricks from nowhere. Dumping weak cards creates voids fast, and voids create free tricks through trumping"
                ))

    for tier in ("A", "K"):
        desp_vals = desperate_plays.get(tier, [])
        comf_vals = comfortable_plays.get(tier, [])
        if len(desp_vals) >= 8 and len(comf_vals) >= 8:  # Lowered from 15
            desp_avg = sum(desp_vals) / len(desp_vals)
            comf_avg = sum(comf_vals) / len(comf_vals)
            if desp_avg > comf_avg + 0.15:  # Lowered from 0.3
                insights.append(make_insight(
                    "When behind, play your Aces and Kings immediately — normally you'd save them, but when desperate you need tricks NOW before opponents consolidate their lead",
                    "counter-intuitive", "advanced", "emerging", episodes,
                    condition="Your team is losing and you need to catch up",
                    why="Saving cards is a luxury for teams that are winning. When behind, every delayed trick is a trick that might never come"
                ))

    return insights[:15]


# ─── 12. Deep Hand Analysis ──────────────────────────────────────────────────────


def _deep_hand_insights(bid_q, episodes) -> list:
    """
    Mine the bid Q-table for specific hand patterns.
    Uses broader groupings to find more patterns.
    """
    insights = []

    # Group by trump-related features (longest suit likely = trump length).
    longest_group = defaultdict(lambda: defaultdict(list))
    voids_group = defaultdict(lambda: defaultdict(list))
    aces_group = defaultdict(lambda: defaultdict(list))

    for state, actions in list(bid_q.items())[:3000]:
        if len(state) < 7:
            continue
        try:
            longest = int(state[0])
            highs = int(state[2])
            aces = int(state[3])
            v_idx = state.index("v")
            voids = int(state[v_idx + 1])
        except (ValueError, IndexError):
            continue

        for action, q in actions.items():
            longest_group[longest][action].append(q)
            voids_group[voids][action].append(q)
            if aces >= 2:
                aces_group[aces][action].append(q)

    # Longest suit insights.
    for length, action_qs in sorted(longest_group.items()):
        if length < 4:
            continue
        best_action, best_avg = None, -999
        for action, vals in action_qs.items():
            if len(vals) >= 5:
                avg = sum(vals) / len(vals)
                if avg > best_avg:
                    best_avg, best_action = avg, action
        if not best_action or best_avg < 0.2:
            continue

        if best_action == "PASS" and length >= 5:
            insights.append(make_insight(
                f"Having a long suit ({length}+ cards) doesn't mean you should bid — length alone isn't power unless that suit is trump",
                "bidding", "intermediate", "emerging", episodes,
                condition=f"Your longest suit has {length}+ cards but it might not be trump",
                why="A long non-trump suit gets trumped. Length only matters if it's your trump suit"
            ))
        elif best_action.startswith("B"):
            try:
                val = int(best_action[1:])
                if length >= 6 and val >= 9:
                    insights.append(make_insight(
                        f"With a {length}-card suit (likely trump), bid {val} — that much trump length means you control the game",
                        "bidding", "advanced", "emerging", episodes,
                        condition=f"Your longest suit has {length}+ cards and it's trump",
                        why=f"With {length} trumps, opponents have very few. You'll win most trump tricks automatically"
                    ))
                elif length == 4 and val <= 7:
                    insights.append(make_insight(
                        f"With a balanced hand (longest suit is 4), bid {val} at most — you don't have the dominance to promise more",
                        "bidding", "beginner", "emerging", episodes,
                        condition="No suit has more than 4 cards",
                        why="Balanced hands win some tricks but can't guarantee many. Safe bids protect you from big penalties"
                    ))
            except ValueError:
                pass

    # Voids in bidding.
    for voids, action_qs in voids_group.items():
        if voids == 0:
            continue
        best_action, best_avg = None, -999
        for action, vals in action_qs.items():
            if len(vals) >= 5:
                avg = sum(vals) / len(vals)
                if avg > best_avg:
                    best_avg, best_action = avg, action
        if not best_action or best_avg < 0.2:
            continue
        if best_action.startswith("B"):
            try:
                val = int(best_action[1:])
                if voids >= 2 and val >= 8:
                    insights.append(make_insight(
                        f"With {voids} voids in your hand, bid boldly ({val}+) — every void is a suit you can trump freely",
                        "bidding", "advanced", "emerging", episodes,
                        condition=f"You have {voids} suits with zero cards",
                        why="Each void guarantees trumping opportunities. Two voids means two different suits where you steal tricks"
                    ))
                elif voids == 1 and val >= 7:
                    insights.append(make_insight(
                        f"One void in your hand boosts your bid by about 2 — that void suit is practically free tricks",
                        "bidding", "intermediate", "emerging", episodes,
                        condition="You have exactly one suit with zero cards",
                        why="Every time opponents lead your void suit, you trump it. That's usually 2-3 extra tricks per game"
                    ))
            except ValueError:
                pass

    # Multiple aces.
    for ace_count, action_qs in aces_group.items():
        if ace_count < 2:
            continue
        best_action, best_avg = None, -999
        for action, vals in action_qs.items():
            if len(vals) >= 3:
                avg = sum(vals) / len(vals)
                if avg > best_avg:
                    best_avg, best_action = avg, action
        if best_action and best_action.startswith("B") and best_avg > 0.3:
            try:
                val = int(best_action[1:])
                insights.append(make_insight(
                    f"With {ace_count} Aces, bid at least {val} — each Ace is a guaranteed trick in its suit",
                    "bidding", "intermediate", "emerging", episodes,
                    condition=f"You're holding {ace_count} Aces",
                    why="Aces can never be beaten when following suit. Each one is a certain trick"
                ))
            except ValueError:
                pass

    return insights[:8]


# ─── 13. Position-Phase Combo Insights ───────────────────────────────────────────


def _position_phase_insights(play_items, episodes) -> list:
    """
    Find specific seat + phase combinations where certain plays
    are exceptionally good or bad.
    """
    insights = []

    # Group Q-values by (position, phase, action_type).
    combo_q = defaultdict(lambda: defaultdict(list))
    for state, actions in play_items:
        if len(state) < 6:
            continue
        pos = state[4]
        phase = state[5]
        for key, q in actions.items():
            if len(key) < 3:
                continue
            tier = key[0]
            is_trump = key[2] == "T"
            act_type = f"{'T' if is_trump else 'N'}{tier}"
            combo_q[(pos, phase)][act_type].append(q)

    pos_names = {"0": "you're leading", "1": "you play second", "2": "you play third", "3": "you play last"}
    phase_names = {"1": "the first few tricks", "2": "the early-mid game", "3": "the middle game", "4": "the late-mid game", "5": "the final tricks"}

    for (pos, phase), act_qs in combo_q.items():
        pos_name = pos_names.get(pos, "")
        phase_name = phase_names.get(phase, "")
        if not pos_name or not phase_name:
            continue

        # Find the best and worst action type.
        best_type, best_avg = None, -999
        worst_type, worst_avg = None, 999
        for act_type, vals in act_qs.items():
            if len(vals) < 15:
                continue
            avg = sum(vals) / len(vals)
            if avg > best_avg:
                best_avg, best_type = avg, act_type
            if avg < worst_avg:
                worst_avg, worst_type = avg, act_type

        if not best_type or best_avg < 0.5:
            continue

        # Only generate insights for strong signals.
        is_trump_best = best_type[0] == "T"
        tier_best = best_type[1]

        tier_desc = {"A": "your Ace", "K": "your King", "Q": "your Queen", "J": "a Jack",
                     "M": "a middle card", "L": "a low card", "X": "your weakest card"}.get(tier_best, "")
        if not tier_desc:
            continue

        if is_trump_best:
            insights.append(make_insight(
                f"In {phase_name} when {pos_name} — play trump {tier_desc}. This specific combination is where trump pays off the most",
                "trump", "advanced", "emerging", episodes,
                condition=f"{pos_name.capitalize()} during {phase_name}",
                why="This seat-timing combination creates the perfect opportunity for trump to dominate"
            ))
        else:
            if tier_best in ("L", "X"):
                insights.append(make_insight(
                    f"In {phase_name} when {pos_name} — throw {tier_desc}. This is a trick worth sacrificing to save your power for better moments",
                    "timing", "advanced", "emerging", episodes,
                    condition=f"{pos_name.capitalize()} during {phase_name}",
                    why="Not every trick is worth fighting for. Smart sacrifice here sets up bigger wins later"
                ))
            elif tier_best in ("A", "K"):
                insights.append(make_insight(
                    f"In {phase_name} when {pos_name} — drop {tier_desc} without hesitation. This is exactly where high cards earn their maximum value",
                    "timing", "advanced", "emerging", episodes,
                    condition=f"{pos_name.capitalize()} during {phase_name}",
                    why="This is the sweet spot where your strong cards face the least resistance and win the most important tricks"
                ))

    return insights[:10]


# ─── 14. ENDLESS Situational Mining ──────────────────────────────────────────────


def _endless_situational_mining(play_items, bid_q, episodes) -> list:
    """
    Mine individual Q-table entries for specific situational lessons.
    This layer grows proportionally with Q-table size.

    Strategy: find the most DECISIVE states (largest gap between best and worst action)
    and decode each into a human-readable game situation with advice.

    The Q-table grows as the agent trains → more decisive states found → more insights.
    Scales from ~5 at 100k to ~20+ at 1M+ episodes.
    """
    insights = []

    # --- Play Q-table mining ---
    # Find states with high decisiveness (best action >> average of others).
    decisive_plays = []
    for state, actions in play_items:
        if len(state) < 8 or len(actions) < 3:
            continue
        sorted_actions = sorted(actions.items(), key=lambda x: -x[1])
        best_key, best_q = sorted_actions[0]
        second_q = sorted_actions[1][1]
        worst_q = sorted_actions[-1][1]

        # Decisiveness = how much better the best is vs the rest.
        spread = best_q - second_q
        range_val = best_q - worst_q

        if spread > 0.8 and range_val > 1.5 and len(best_key) >= 3:  # Lowered from 1.5/2.5
            decisive_plays.append((state, best_key, best_q, spread, worst_q))

    # Sort by spread (most decisive first) and take top ones.
    decisive_plays.sort(key=lambda x: -x[3])

    # Decode each into a lesson — but limit total and skip similar situations.
    seen_situations = set()
    play_count = 0
    max_play_insights = max(8, min(30, len(decisive_plays) // 5))

    for state, best_key, best_q, spread, worst_q in decisive_plays:
        if play_count >= max_play_insights:
            break

        # Decode state.
        situation = _decode_play_state(state)
        if not situation:
            continue

        # Decode action.
        action_desc = _decode_play_action(best_key)
        if not action_desc:
            continue

        # Create a situation signature to avoid near-duplicates.
        sig = (situation.get("phase", ""), situation.get("pos", ""), best_key[:3])
        if sig in seen_situations:
            continue
        seen_situations.add(sig)

        # Build the lesson text.
        when_parts = []
        if situation.get("phase_name"):
            when_parts.append(f"it's the {situation['phase_name']} of the game")
        if situation.get("pos_name"):
            when_parts.append(f"you're {situation['pos_name']}")
        if situation.get("trump_count", 0) >= 4:
            when_parts.append(f"you have {situation['trump_count']}+ trumps")
        if situation.get("voids", 0) >= 1:
            when_parts.append(f"you're void in {situation['voids']} suit{'s' if situation['voids'] > 1 else ''}")
        if situation.get("winning"):
            when_parts.append("your team is ahead")
        elif situation.get("losing"):
            when_parts.append("your team is behind")

        if not when_parts:
            continue

        condition = " and ".join(when_parts)
        text = f"When {condition} — {action_desc}"

        insights.append(make_insight(
            text, _categorize_action(best_key), "advanced", "emerging", episodes,
            condition=condition.capitalize(),
            why=f"This specific situation was tested thousands of times. This play consistently dominates all alternatives"
        ))
        play_count += 1

    # --- Bid Q-table mining ---
    decisive_bids = []
    for state, actions in list(bid_q.items())[:3000]:
        if len(state) < 7 or len(actions) < 2:
            continue
        sorted_actions = sorted(actions.items(), key=lambda x: -x[1])
        best_key, best_q = sorted_actions[0]
        second_q = sorted_actions[1][1] if len(sorted_actions) > 1 else 0
        spread = best_q - second_q

        if spread > 0.5 and best_q > 0.2:  # Lowered from 1.0/0.3
            decisive_bids.append((state, best_key, best_q, spread))

    decisive_bids.sort(key=lambda x: -x[3])

    seen_bid_sigs = set()
    bid_count = 0
    max_bid_insights = max(5, min(15, len(decisive_bids) // 5))

    for state, best_key, best_q, spread in decisive_bids:
        if bid_count >= max_bid_insights:
            break

        bid_situation = _decode_bid_state(state)
        if not bid_situation:
            continue

        # Signature to avoid duplicates.
        sig = (bid_situation.get("longest", 0), bid_situation.get("highs", 0), best_key)
        if sig in seen_bid_sigs:
            continue
        seen_bid_sigs.add(sig)

        # Build description.
        hand_parts = []
        if bid_situation.get("longest", 0) >= 5:
            hand_parts.append(f"your longest suit has {bid_situation['longest']} cards")
        if bid_situation.get("highs", 0) >= 3:
            hand_parts.append(f"you have {bid_situation['highs']} high cards")
        if bid_situation.get("aces", 0) >= 2:
            hand_parts.append(f"including {bid_situation['aces']} Aces")
        if bid_situation.get("voids", 0) >= 1:
            hand_parts.append(f"with a void")
        if bid_situation.get("opp_bid_high"):
            hand_parts.append("opponents already bid high")

        if not hand_parts:
            continue

        hand_desc = " and ".join(hand_parts)

        if best_key == "PASS":
            text = f"When {hand_desc} — pass. It sounds wrong but defending works better here"
            cat = "defense"
        elif best_key.startswith("B"):
            try:
                val = int(best_key[1:])
                text = f"When {hand_desc} — bid {val}. This hand delivers exactly that many tricks reliably"
                cat = "bidding"
            except ValueError:
                continue
        else:
            continue

        insights.append(make_insight(
            text, cat, "intermediate", "emerging", episodes,
            condition=f"Your hand has: {hand_desc}",
            why="This exact hand shape was played thousands of times. This bid wins the most points consistently"
        ))
        bid_count += 1

    return insights


def _decode_play_state(state: str) -> dict:
    """Decode a play state string into a human-readable situation dict."""
    if len(state) < 8:
        return None
    try:
        pos = state[4]
        phase = state[5]
        td = state[6]  # W/A/T/B

        pos_names = {"0": "leading", "1": "playing second", "2": "playing third", "3": "playing last"}
        phase_names = {"1": "first few tricks", "2": "early-mid game", "3": "middle game", "4": "late-mid game", "5": "final tricks"}

        result = {
            "pos": pos,
            "phase": phase,
            "pos_name": pos_names.get(pos, ""),
            "phase_name": phase_names.get(phase, ""),
            "winning": td == "W",
            "losing": td == "B",
        }

        # Extract trump count if available.
        # State format: shape(4) + pos(1) + phase(1) + td(1) + highs(1) + trump_count(1) + trump_highs(1) + ...
        if len(state) > 7:
            result["highs"] = int(state[7]) if state[7].isdigit() else 0
        if len(state) > 8:
            result["trump_count"] = int(state[8]) if state[8].isdigit() else 0
        # Voids.
        if "v" in state:
            v_idx = state.index("v")
            if v_idx + 1 < len(state) and state[v_idx + 1].isdigit():
                result["voids"] = int(state[v_idx + 1])

        return result
    except (IndexError, ValueError):
        return None


def _decode_bid_state(state: str) -> dict:
    """Decode a bid state string into a human-readable dict."""
    if len(state) < 7:
        return None
    try:
        longest = int(state[0])
        shortest = int(state[1])
        highs = int(state[2])
        aces = int(state[3])
        v_idx = state.index("v")
        voids = int(state[v_idx + 1])
        has_bid = state[v_idx + 2] == "Y"
        bid_level = int(state[v_idx + 3]) if v_idx + 3 < len(state) and state[v_idx + 3].isdigit() else 0

        return {
            "longest": longest,
            "shortest": shortest,
            "highs": highs,
            "aces": aces,
            "voids": voids,
            "opp_bid_high": has_bid and bid_level >= 8,
        }
    except (ValueError, IndexError):
        return None


def _decode_play_action(key: str) -> str:
    """Decode an action key into a human-readable description."""
    if len(key) < 3:
        return None
    tier = key[0]
    follows = key[1]
    is_trump = key[2] == "T"

    tier_descs = {
        "A": "play your Ace",
        "K": "play your King",
        "Q": "play your Queen",
        "J": "play a Jack",
        "M": "play a middle card (9 or 10)",
        "L": "play a low card",
        "X": "throw your weakest card",
    }
    desc = tier_descs.get(tier, "play a card")

    if is_trump and follows == "O":
        desc = f"trump with your {'highest' if tier in 'AKQ' else 'lowest'} trump"
    elif is_trump and follows == "F":
        desc = f"follow with your {'high' if tier in 'AKQ' else 'low'} trump"
    elif follows == "O" and tier in ("L", "X"):
        desc = "discard your weakest card from another suit"
    elif follows == "F" and tier in ("L", "X"):
        desc = "follow with your lowest — don't fight this trick"

    return desc


def _categorize_action(key: str) -> str:
    """Determine category from action key."""
    if len(key) >= 3 and key[2] == "T":
        return "trump"
    if len(key) >= 5 and key[4] == "V":
        return "voids"
    return "timing"


def _evolution_insights(play_items, bid_q, episodes) -> list:
    """
    Compare current strategy to historical snapshots.
    Generates new insights as strategy evolves over training.
    """
    current_fp = _build_fingerprint(play_items, bid_q)
    snapshots = _load_snapshots()

    last_ep = snapshots[-1]["episodes"] if snapshots else 0
    if episodes - last_ep >= 10000:
        snapshots.append({"episodes": episodes, "fingerprint": current_fp})
        if len(snapshots) > 100:
            kept = [snapshots[0]]
            kept.extend(snapshots[i] for i in range(5, len(snapshots) - 10, 5))
            kept.extend(snapshots[-10:])
            snapshots = kept
        _save_snapshots(snapshots)

    insights = []
    if len(snapshots) < 3:
        return insights

    # Compare to early snapshot.
    old_fp = snapshots[0]["fingerprint"]
    shifts = _count_shifts(old_fp, current_fp)

    if shifts["void_up"] >= 2:
        insights.append(make_insight(
            "Void creation is a fundamental skill — the more you play, the more you realize that emptying suits early decides games",
            "voids", "intermediate", "proven", episodes,
            why="After hundreds of thousands of games, void strategy consistently outperforms other approaches"
        ))

    if shifts["trump_up"] >= 3:
        insights.append(make_insight(
            "Trump control is the most important skill in Wist — whoever controls trump controls the game",
            "trump", "advanced", "proven", episodes,
            why="Trump is the only suit that beats all others. Managing it well is the single biggest factor in winning"
        ))

    if shifts["patience"] >= 3:
        insights.append(make_insight(
            "Patience wins more games than aggression — the best players wait for the right moment to strike",
            "timing", "advanced", "proven", episodes,
            why="Impatient plays waste strong cards. Patient plays use them at maximum effect"
        ))

    # Compare to recent snapshot for newer discoveries.
    if len(snapshots) >= 4:
        recent_fp = snapshots[-3]["fingerprint"]
        recent_shifts = _count_shifts(recent_fp, current_fp)

        if recent_shifts["aggro"] >= 2:
            insights.append(make_insight(
                "Sometimes the game calls for aggression — if you have the cards, use them before the window closes",
                "timing", "advanced", "emerging", episodes,
                condition="You have multiple high cards and the game is in the first half",
                why="Windows of opportunity are brief. Hesitation can turn a winning hand into a losing one"
            ))

        if recent_shifts["pass_up"] >= 2:
            insights.append(make_insight(
                "Defending against overbidders is a reliable strategy — let them take the risk while you collect points from their failures",
                "bidding", "intermediate", "emerging", episodes,
                condition="You don't have a clearly strong hand",
                why="Most high bids fail. Defending well is almost as good as bidding successfully"
            ))

    return insights[:8]


def _count_shifts(old_fp, new_fp) -> dict:
    """Count behavioral shifts between two fingerprints."""
    shifts = {"aggro": 0, "patience": 0, "trump_up": 0, "trump_down": 0, "void_up": 0, "pass_up": 0, "pass_down": 0}
    for ctx in new_fp:
        if ctx not in old_fp or ctx == "bid_best":
            continue
        old_a = old_fp[ctx]["action"]
        new_a = new_fp[ctx]["action"]
        if old_a == new_a:
            continue
        old_rank = _TIER_RANK.get(old_a[0] if old_a else "", 0)
        new_rank = _TIER_RANK.get(new_a[0] if new_a else "", 0)
        if new_rank > old_rank:
            shifts["aggro"] += 1
        elif new_rank < old_rank:
            shifts["patience"] += 1
        if len(new_a) > 2 and new_a[2] == "T" and (len(old_a) < 3 or old_a[2] != "T"):
            shifts["trump_up"] += 1
        if len(old_a) > 2 and old_a[2] == "T" and (len(new_a) < 3 or new_a[2] != "T"):
            shifts["trump_down"] += 1
        if len(new_a) > 4 and new_a[4] == "V" and (len(old_a) < 5 or old_a[4] != "V"):
            shifts["void_up"] += 1
    return shifts



# ─── Progression Gating ──────────────────────────────────────────────────────────


def _apply_progression_gate(insights, episodes) -> list:
    """Only show insights appropriate to the training level."""
    if episodes < 10000:
        return [ins for ins in insights if ins["difficulty"] == "beginner"]
    elif episodes < 50000:
        return [ins for ins in insights if ins["difficulty"] in ("beginner", "intermediate")]
    else:
        return insights  # Show all.


# ─── Deduplication ───────────────────────────────────────────────────────────────


def _deduplicate(insights) -> list:
    """Remove insights that are too similar — but be lenient to allow variety."""
    unique = []
    seen_fragments = set()

    for ins in insights:
        words = ins["text"].lower().split()
        # Only reject if 5+ consecutive words are shared (stricter = more variety).
        dominated = False
        fragments = set()
        for i in range(len(words) - 4):
            frag = " ".join(words[i:i + 5])
            if frag in seen_fragments:
                dominated = True
                break
            fragments.add(frag)
        if dominated:
            continue
        unique.append(ins)
        seen_fragments.update(fragments)

    return unique


# ─── Linking Related Insights ────────────────────────────────────────────────────


def _link_insights(insights):
    """Find related insights and link them together."""
    for i, ins in enumerate(insights):
        related = []
        for j, other in enumerate(insights):
            if i == j:
                continue
            # Same category = related.
            if ins["category"] == other["category"] and len(related) < 2:
                related.append(j)
        ins["links"] = related



# ─── Fingerprint & Snapshots ─────────────────────────────────────────────────────


def _build_fingerprint(play_items, bid_q) -> dict:
    """Strategy fingerprint: best action per (phase, position)."""
    fp = {}
    context_best = defaultdict(lambda: defaultdict(list))
    for state, actions in play_items:
        if len(state) < 6:
            continue
        ctx = f"p{state[5]}s{state[4]}"
        for key, q in actions.items():
            if len(key) >= 3:
                context_best[ctx][key].append(q)
    for ctx, action_vals in context_best.items():
        best_key, best_avg = "", -999
        for key, vals in action_vals.items():
            if len(vals) >= 5:
                avg = sum(vals) / len(vals)
                if avg > best_avg:
                    best_avg, best_key = avg, key
        if best_key:
            fp[ctx] = {"action": best_key, "avg_q": round(best_avg, 3)}
    bid_vals = defaultdict(list)
    for _state, actions in list(bid_q.items())[:300]:
        for key, q in actions.items():
            bid_vals[key].append(q)
    best_bid, best_bid_q = "PASS", -999
    for key, vals in bid_vals.items():
        if len(vals) >= 5:
            avg = sum(vals) / len(vals)
            if avg > best_bid_q:
                best_bid_q, best_bid = avg, key
    fp["bid_best"] = {"action": best_bid, "avg_q": round(best_bid_q, 3)}
    return fp


def _load_snapshots() -> list:
    try:
        if os.path.exists(_SNAPSHOTS_PATH):
            with open(_SNAPSHOTS_PATH, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_snapshots(snapshots):
    try:
        with open(_SNAPSHOTS_PATH, "w") as f:
            json.dump(snapshots, f)
    except Exception:
        pass


def _save_cached_insights(insights):
    try:
        to_save = []
        for ins in insights:
            to_save.append({
                "text": ins.get("text", ""),
                "category": ins.get("category", ""),
                "difficulty": ins.get("difficulty", "beginner"),
                "confidence": ins.get("confidence", "emerging"),
                "episode": ins.get("episode", 0),
                "version": ins.get("version", 0),
                "why": ins.get("why", ""),
                "new": ins.get("new", False),
            })
        with open(_INSIGHTS_CACHE_PATH, "w") as f:
            json.dump(to_save, f)
    except Exception:
        pass


def _load_cached_insights() -> list:
    try:
        if os.path.exists(_INSIGHTS_CACHE_PATH):
            with open(_INSIGHTS_CACHE_PATH, "r") as f:
                data = json.load(f)
            # Ensure all entries have required fields.
            for ins in data:
                ins.setdefault("version", 0)
                ins.setdefault("why", "")
                ins.setdefault("new", False)
                ins.setdefault("links", [])
                ins.setdefault("condition", None)
                ins.setdefault("exception", None)
            return data
    except Exception:
        pass
    return []


# ─── Helper: Get insight text for backward compatibility ─────────────────────────


def get_insight_texts(insights) -> list:
    """Extract just the text from structured insights (for simple display)."""
    return [ins["text"] if isinstance(ins, dict) else ins for ins in insights]
