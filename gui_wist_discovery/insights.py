"""
Strategic Insights System — AUTO-GENERATED from Q-table patterns.

Works with the MINIMAL encoding:
- State key: {n_cards_hex}{position}  (e.g., "d0" = 13 cards, leading)
- Action key: {rank_hex}{suit_index}  (e.g., "e0" = Ace of spades, "72" = 7 of clubs)
- Bid state: {has_bid}{bid_level}     (e.g., "N0" = no bid yet, "Y8" = someone bid 8)
- Bid action: "PASS" or "B7".."B13"

Insights grow as the Q-table grows. Counter-intuitive patterns detected automatically.
"""

from collections import defaultdict
import json
import os

_INSIGHTS_CACHE_PATH = "agents/wist_discovery/insights_cache.json"
_SNAPSHOTS_PATH = "agents/wist_discovery/strategy_snapshots.json"

# Snapshot intervals for comparison-based insights (recurring).
_SNAPSHOT_INTERVALS = [50000, 100000, 500000, 1000000]

# Rank hex mapping for display.
_RANK_NAMES = {
    2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
    9: "9", 10: "10", 11: "Jack", 12: "Queen", 13: "King", 14: "Ace",
}
_SUIT_NAMES = {0: "spades", 1: "hearts", 2: "clubs", 3: "diamonds"}
_POS_NAMES = {0: "first to play", 1: "second to play", 2: "third to play", 3: "last to play"}


def _parse_action_key(key: str):
    """Parse action key: {rank_hex}{suit_index}{is_highest} -> (rank_int, suit_int) or None.

    New format (v3.3.2+): e.g., "e01" = rank 14 (Ace), suit 0, is_highest=1
    Old format: "e0" = rank 14, suit 0
    """
    try:
        if len(key) < 2:
            return None
        if len(key) == 3:
            # New format: {rank_hex}{suit}{is_highest}
            rank = int(key[0], 16)
            suit = int(key[1])
            if 2 <= rank <= 14 and 0 <= suit <= 3:
                return (rank, suit)
            # Try: {rank_hex_2chars}{suit} for ranks >= 10 (a, b, c, d, e)
            rank = int(key[:2], 16)
            suit = int(key[2])
            if 2 <= rank <= 14 and 0 <= suit <= 3:
                return (rank, suit)
        elif len(key) == 2:
            # Old format: {rank_hex}{suit}
            rank = int(key[0], 16)
            suit = int(key[1])
            if 2 <= rank <= 14 and 0 <= suit <= 3:
                return (rank, suit)
        # General: try first char(s) as hex rank, last char as suit.
        # For "e01": rank=0xe=14, suit=0, ignore last char.
        rank = int(key[0], 16)
        suit = int(key[1])
        if 2 <= rank <= 14 and 0 <= suit <= 3:
            return (rank, suit)
    except (ValueError, IndexError):
        pass
    return None


def _parse_state_key(key: str):
    """Parse state key: {n_cards_hex}{position}{led}{my_tricks}{opp_tricks} -> (n_cards, position) or None.

    New format (v3.3.2+): e.g., "d0x00" = 13 cards, position 0, led=x, tricks 0/0
    Old format: "d0" = 13 cards, position 0
    """
    try:
        if len(key) < 2:
            return None
        # First char is always n_cards in hex, second is always position.
        n_cards = int(key[0], 16)
        pos = int(key[1])
        if 1 <= n_cards <= 13 and 0 <= pos <= 3:
            return (n_cards, pos)
    except (ValueError, IndexError):
        pass
    return None


def _rank_tier(rank: int) -> str:
    """Classify rank into tiers for insight grouping."""
    if rank >= 13:
        return "high"   # King, Ace
    elif rank >= 11:
        return "upper"  # Jack, Queen
    elif rank >= 8:
        return "mid"    # 8, 9, 10
    else:
        return "low"    # 2-7


# ─── Main Entry Point ────────────────────────────────────────────────────────────


def generate_insights(agent) -> list:
    """
    Auto-generate insights by mining Q-table patterns.
    Returns list of insight dicts. Accumulates over time.
    """
    if agent.episodes_trained < 3000:
        return [_make(
            "Still learning basics, building experience...",
            "timing", 1, agent.episodes_trained,
            why="Need more games to detect meaningful patterns"
        )]

    episodes = agent.episodes_trained
    play_q = agent.play_q
    bid_q = agent.bid_q
    play_items = list(play_q.items())[:min(len(play_q), 30000)]

    # Load accumulated history.
    accumulated = _load_cache()
    acc_texts = {ins["text"] for ins in accumulated}

    # Mine patterns from Q-tables.
    new_insights = []
    new_insights.extend(_mine_rank_patterns(play_items, episodes))
    new_insights.extend(_mine_suit_patterns(play_items, episodes))
    new_insights.extend(_mine_position_patterns(play_items, episodes))
    new_insights.extend(_mine_hand_size_patterns(play_items, episodes))
    new_insights.extend(_mine_bid_patterns(bid_q, episodes))
    new_insights.extend(_mine_counter_intuitive(play_items, episodes))
    new_insights.extend(_mine_deep_patterns(play_items, episodes))
    new_insights.extend(_mine_leading_vs_following(play_items, episodes))
    new_insights.extend(_mine_per_suit_rank(play_items, episodes))
    new_insights.extend(_mine_hand_size_suit_interaction(play_items, episodes))
    new_insights.extend(_mine_granular_discoveries(play_items, bid_q, episodes))

    # Snapshot comparison insights (at milestones: 50K, 100K, 500K, 1M, etc.).
    new_insights.extend(_check_and_take_snapshot(agent))

    # Dedup by core idea.
    new_insights = _dedup_merge(new_insights)

    # Merge new into accumulated.
    for ins in new_insights:
        if ins["text"] in acc_texts:
            for acc in accumulated:
                if acc["text"] == ins["text"]:
                    acc["confidence"] = acc.get("confidence", 1) + 1
                    conf = acc["confidence"]
                    if conf == 5:
                        acc["why"] = _refine_why(acc.get("why", ""), "works most of the time")
                    elif conf == 10:
                        acc["why"] = _refine_why(acc.get("why", ""), "almost always works")
                    elif conf == 20:
                        acc["why"] = _refine_why(acc.get("why", ""), "most reliable")
                    break
        else:
            accumulated.append(ins)
            acc_texts.add(ins["text"])

    # Fix capitalization after dots in all texts.
    for ins in accumulated:
        if isinstance(ins, dict):
            if ins.get("text"):
                ins["text"] = _capitalize_after_dots(ins["text"])
            if ins.get("why"):
                ins["why"] = _capitalize_after_dots(ins["why"])

    _save_cache(accumulated)
    return accumulated


def _make(text, category, confidence, episode, why=""):
    """Create insight dict."""
    return {
        "text": text,
        "category": category,
        "difficulty": "intermediate",
        "confidence": confidence,
        "episode": episode,
        "why": why,
        "version": 0,
        "new": True,
        "links": [],
        "condition": None,
        "exception": None,
    }


# ─── Pattern Miners (New Minimal Encoding) ───────────────────────────────────────


def _mine_rank_patterns(play_items, episodes) -> list:
    """Which card ranks score highest across all situations?"""
    insights = []
    rank_qs = defaultdict(list)  # rank -> list of Q-values
    tier_qs = defaultdict(list)  # tier -> list of Q-values

    for state, actions in play_items:
        if len(actions) < 2:
            continue
        for key, q in actions.items():
            parsed = _parse_action_key(key)
            if not parsed:
                continue
            rank, suit = parsed
            rank_qs[rank].append(q)
            tier_qs[_rank_tier(rank)].append(q)

    # Need minimum data.
    if not tier_qs or sum(len(v) for v in tier_qs.values()) < 50:
        return insights

    tier_avgs = {t: sum(v)/len(v) for t, v in tier_qs.items() if len(v) >= 10}
    if not tier_avgs:
        return insights

    best_tier = max(tier_avgs, key=tier_avgs.get)
    worst_tier = min(tier_avgs, key=tier_avgs.get)

    if best_tier == "high" and tier_avgs["high"] > tier_avgs.get("low", 0) + 0.2:
        insights.append(_make(
            "High cards (Kings and Aces) consistently win more than low cards, raw card power matters",
            "timing", 1, episodes,
            why="stronger cards win tricks more often regardless of situation"
        ))

    if best_tier == "low" and tier_avgs["low"] > tier_avgs.get("high", 0) + 0.1:
        insights.append(_make(
            "Low cards played at the right time outperform high cards played carelessly",
            "counter-intuitive", 1, episodes,
            why="timing matters more than raw strength"
        ))

    if best_tier == "mid":
        insights.append(_make(
            "Mid-range cards (8s, 9s, 10s) quietly outperform both high and low cards on average",
            "counter-intuitive", 1, episodes,
            why="they win tricks nobody fights over while high cards attract opposition"
        ))


    # Per-rank discoveries: which individual ranks stand out?
    rank_avgs = {r: sum(v)/len(v) for r, v in rank_qs.items() if len(v) >= 10}
    if rank_avgs:
        best_rank = max(rank_avgs, key=rank_avgs.get)
        worst_rank = min(rank_avgs, key=rank_avgs.get)
        if rank_avgs[best_rank] - rank_avgs[worst_rank] > 0.3:
            if best_rank == 14:
                insights.append(_make(
                    "Aces dominate, they win almost every trick they are played in",
                    "timing", 1, episodes,
                    why="the highest card in any suit almost always takes the trick"
                ))
            elif best_rank <= 5:
                insights.append(_make(
                    "The AI is finding value in very low cards, using them strategically rather than wasting them",
                    "counter-intuitive", 1, episodes,
                    why="low cards preserve hand shape and set up future wins"
                ))

    return insights


def _mine_suit_patterns(play_items, episodes) -> list:
    """Does one suit consistently score higher? (indicates trump discovery)."""
    insights = []
    suit_qs = defaultdict(list)  # suit_idx -> Q-values

    for state, actions in play_items:
        if len(actions) < 2:
            continue
        for key, q in actions.items():
            parsed = _parse_action_key(key)
            if not parsed:
                continue
            rank, suit = parsed
            suit_qs[suit].append(q)

    if not suit_qs or sum(len(v) for v in suit_qs.values()) < 50:
        return insights

    suit_avgs = {s: sum(v)/len(v) for s, v in suit_qs.items() if len(v) >= 10}
    if len(suit_avgs) < 2:
        return insights

    best_suit = max(suit_avgs, key=suit_avgs.get)
    worst_suit = min(suit_avgs, key=suit_avgs.get)
    spread = suit_avgs[best_suit] - suit_avgs[worst_suit]

    if spread > 0.3:
        insights.append(_make(
            f"One suit ({_SUIT_NAMES.get(best_suit, '?')}) consistently scores higher than others, the AI may be discovering trump power",
            "trump", 1, episodes,
            why="trump cards beat all other suits regardless of rank"
        ))
    elif spread > 0.15:
        insights.append(_make(
            "The AI is starting to differentiate between suits, some are more valuable than others",
            "trump", 1, episodes,
            why="not all suits are equal in Wist, one suit dominates each round"
        ))

    return insights


def _mine_position_patterns(play_items, episodes) -> list:
    """Does playing position affect performance?"""
    insights = []
    pos_qs = defaultdict(list)  # position -> list of best-action Q-values

    for state, actions in play_items:
        parsed = _parse_state_key(state)
        if not parsed or len(actions) < 2:
            continue
        n_cards, pos = parsed
        best_q = max(actions.values())
        pos_qs[pos].append(best_q)

    if not pos_qs or sum(len(v) for v in pos_qs.values()) < 30:
        return insights

    pos_avgs = {p: sum(v)/len(v) for p, v in pos_qs.items() if len(v) >= 5}
    if len(pos_avgs) < 2:
        return insights

    best_pos = max(pos_avgs, key=pos_avgs.get)
    worst_pos = min(pos_avgs, key=pos_avgs.get)
    spread = pos_avgs[best_pos] - pos_avgs[worst_pos]

    if spread > 0.2:
        pos_name = _POS_NAMES.get(best_pos, "unknown")
        if best_pos == 3:
            insights.append(_make(
                "Playing last is the strongest position, you see all other cards before choosing yours",
                "timing", 1, episodes,
                why="perfect information lets you play the minimum needed to win"
            ))
        elif best_pos == 0:
            insights.append(_make(
                "Playing first gives you control, you force everyone to follow your suit choice",
                "timing", 1, episodes,
                why="the leader dictates which suit everyone must play"
            ))
        else:
            insights.append(_make(
                f"Being the {pos_name} gives an advantage the AI is exploiting",
                "timing", 1, episodes,
                why="position determines how much information you have when choosing"
            ))

    # Position-specific card choices.
    pos_rank_qs = defaultdict(lambda: defaultdict(list))
    for state, actions in play_items:
        parsed = _parse_state_key(state)
        if not parsed:
            continue
        n_cards, pos = parsed
        for key, q in actions.items():
            a_parsed = _parse_action_key(key)
            if a_parsed:
                rank, suit = a_parsed
                pos_rank_qs[pos][_rank_tier(rank)].append(q)

    for pos in pos_rank_qs:
        tiers = pos_rank_qs[pos]
        tier_avgs = {t: sum(v)/len(v) for t, v in tiers.items() if len(v) >= 5}
        if len(tier_avgs) < 2:
            continue
        best_t = max(tier_avgs, key=tier_avgs.get)
        if best_t == "low" and pos == 0:
            insights.append(_make(
                "When playing first, starting low is more effective than leading with high cards",
                "counter-intuitive", 1, episodes,
                why="leading low probes the table safely and preserves strong cards"
            ))
        elif best_t == "high" and pos == 3:
            insights.append(_make(
                "When playing last, high cards are most effective because you know if they will win",
                "timing", 1, episodes,
                why="no risk of wasting them since you can see what beats what"
            ))

    return insights


def _mine_hand_size_patterns(play_items, episodes) -> list:
    """Does hand size (trick number) affect strategy?"""
    insights = []
    # Group by hand size (proxy for which trick we're on).
    early_qs = defaultdict(list)  # tier -> Q (hand 11-13 = tricks 1-3)
    late_qs = defaultdict(list)   # tier -> Q (hand 1-3 = tricks 11-13)

    for state, actions in play_items:
        parsed = _parse_state_key(state)
        if not parsed or len(actions) < 2:
            continue
        n_cards, pos = parsed
        for key, q in actions.items():
            a_parsed = _parse_action_key(key)
            if not a_parsed:
                continue
            rank, suit = a_parsed
            tier = _rank_tier(rank)
            if n_cards >= 11:
                early_qs[tier].append(q)
            elif n_cards <= 3:
                late_qs[tier].append(q)

    # Early game insight.
    if early_qs:
        early_avgs = {t: sum(v)/len(v) for t, v in early_qs.items() if len(v) >= 5}
        if early_avgs:
            best_early = max(early_avgs, key=early_avgs.get)
            if best_early == "low":
                insights.append(_make(
                    "In early tricks, playing low cards works better than immediately spending your best cards",
                    "timing", 1, episodes,
                    why="saving strong cards for later gives them more impact"
                ))
            elif best_early == "high":
                insights.append(_make(
                    "In early tricks, aggressive play with high cards establishes dominance immediately",
                    "timing", 1, episodes,
                    why="early wins build momentum and force opponents into weak positions"
                ))

    # Late game insight.
    if late_qs:
        late_avgs = {t: sum(v)/len(v) for t, v in late_qs.items() if len(v) >= 5}
        if late_avgs:
            best_late = max(late_avgs, key=late_avgs.get)
            if best_late == "high":
                insights.append(_make(
                    "In late tricks (final 3), high cards are almost guaranteed winners since fewer opponents can respond",
                    "timing", 1, episodes,
                    why="with fewer cards in hand, opponents have less ability to counter"
                ))

    return insights


def _mine_bid_patterns(bid_q, episodes) -> list:
    """Mine bid Q-table for bidding insights."""
    insights = []
    if not bid_q:
        return insights

    # Group by whether someone already bid.
    no_bid_actions = defaultdict(list)   # action -> Q-values when no one bid
    has_bid_actions = defaultdict(list)  # action -> Q-values when someone bid

    for state, actions in list(bid_q.items())[:5000]:
        if len(state) < 2:
            continue
        has_bid = state[0] == "Y"
        for action, q in actions.items():
            if has_bid:
                has_bid_actions[action].append(q)
            else:
                no_bid_actions[action].append(q)

    # What does the AI prefer when no one bid yet?
    if no_bid_actions:
        avgs = {a: sum(v)/len(v) for a, v in no_bid_actions.items() if len(v) >= 3}
        if avgs:
            best_action = max(avgs, key=avgs.get)
            if best_action == "PASS":
                insights.append(_make(
                    "The AI prefers to pass rather than open bidding, waiting for others to commit first",
                    "bidding", 1, episodes,
                    why="letting opponents bid first reveals information about their hands"
                ))
            elif best_action.startswith("B"):
                try:
                    val = int(best_action[1:])
                    if val <= 8:
                        insights.append(_make(
                            f"The AI favors conservative opening bids (around {val}), promising less and aiming to over-deliver",
                            "bidding", 1, episodes,
                            why="lower bids are safer and still score well when exceeded"
                        ))
                    else:
                        insights.append(_make(
                            f"The AI favors aggressive opening bids (around {val}), committing to high targets",
                            "bidding", 1, episodes,
                            why="confident bids score more when successful"
                        ))
                except ValueError:
                    pass

    # When someone already bid — compete or pass?
    if has_bid_actions:
        avgs = {a: sum(v)/len(v) for a, v in has_bid_actions.items() if len(v) >= 3}
        if avgs:
            best_action = max(avgs, key=avgs.get)
            if best_action == "PASS":
                insights.append(_make(
                    "When opponents bid first, the AI usually passes and defends rather than competing",
                    "defense", 1, episodes,
                    why="defending against an opponent's bid is often safer than overbidding"
                ))
            elif best_action.startswith("B"):
                insights.append(_make(
                    "The AI competes aggressively when opponents bid, escalating to take control",
                    "bidding", 1, episodes,
                    why="winning the bid means choosing trump suit which is a huge advantage"
                ))

    # Bid level preferences.
    bid_level_qs = defaultdict(list)
    for state, actions in list(bid_q.items())[:5000]:
        for action, q in actions.items():
            if action.startswith("B"):
                try:
                    level = int(action[1:])
                    bid_level_qs[level].append(q)
                except ValueError:
                    pass

    if bid_level_qs:
        level_avgs = {l: sum(v)/len(v) for l, v in bid_level_qs.items() if len(v) >= 3}
        if level_avgs:
            best_level = max(level_avgs, key=level_avgs.get)
            if best_level == 7:
                insights.append(_make(
                    "Bid 7 is the AI's sweet spot, low enough to make reliably but still scores well",
                    "bidding", 1, episodes,
                    why="the minimum bid maximizes the chance of success"
                ))
            elif best_level >= 10:
                insights.append(_make(
                    f"Bid {best_level} scores highest on average, the AI has learned when big bids pay off",
                    "bidding", 1, episodes,
                    why="high bids score massively when you have the hand to back them up"
                ))

    return insights


def _mine_counter_intuitive(play_items, episodes) -> list:
    """Find patterns where obvious plays lose."""
    insights = []

    # Do low cards beat high cards in specific positions?
    pos_tier_qs = defaultdict(lambda: defaultdict(list))
    for state, actions in play_items:
        parsed = _parse_state_key(state)
        if not parsed or len(actions) < 2:
            continue
        n_cards, pos = parsed
        for key, q in actions.items():
            a_parsed = _parse_action_key(key)
            if not a_parsed:
                continue
            rank, suit = a_parsed
            pos_tier_qs[pos][_rank_tier(rank)].append(q)

    for pos, tier_data in pos_tier_qs.items():
        avgs = {t: sum(v)/len(v) for t, v in tier_data.items() if len(v) >= 10}
        if "low" in avgs and "high" in avgs:
            if avgs["low"] > avgs["high"] + 0.15:
                pos_name = _POS_NAMES.get(pos, "this position")
                insights.append(_make(
                    f"When {pos_name}, low cards actually outperform high cards, saving strength for later",
                    "counter-intuitive", 1, episodes,
                    why="high cards attract opposition and get wasted in losing battles"
                ))

    # Does any single suit massively outperform? (across all positions).
    suit_performance = defaultdict(list)
    for state, actions in play_items:
        if len(actions) < 2:
            continue
        sorted_a = sorted(actions.items(), key=lambda x: -x[1])
        best_key = sorted_a[0][0]
        parsed = _parse_action_key(best_key)
        if parsed:
            rank, suit = parsed
            suit_performance[suit].append(sorted_a[0][1])

    if suit_performance:
        suit_avgs = {s: sum(v)/len(v) for s, v in suit_performance.items() if len(v) >= 10}
        if len(suit_avgs) >= 2:
            vals = sorted(suit_avgs.values())
            if vals[-1] - vals[0] > 0.4:
                dominant = max(suit_avgs, key=suit_avgs.get)
                insights.append(_make(
                    f"One suit ({_SUIT_NAMES.get(dominant, '?')}) is chosen as the winning play far more often than others",
                    "trump", 1, episodes,
                    why="this is likely the trump suit which beats all other suits"
                ))

    return insights


# ─── Dedup & Helpers ─────────────────────────────────────────────────────────────


def _dedup_merge(insights) -> list:
    """Merge insights by core idea — allow up to 20 per concept."""
    seen = {}
    concept_counts = {}  # Track how many per concept.
    MAX_PER_CONCEPT = 20

    for ins in insights:
        text = ins.get("text", "").lower()
        cat = ins.get("category", "")

        # Extract core idea.
        sig_parts = [cat]
        if "ace" in text:
            sig_parts.append("aces")
        elif "high card" in text or "kings and aces" in text:
            sig_parts.append("high_cards")
        elif "low card" in text and "outperform" in text:
            sig_parts.append("low_beats_high")
        elif "mid-range" in text or "mid card" in text:
            sig_parts.append("mid_cards")
        elif "suit" in text and ("higher" in text or "trump" in text or "dominant" in text):
            sig_parts.append("suit_dominance")
        elif "position" in text or "first to play" in text or "last to play" in text:
            sig_parts.append("position")
        elif "early" in text:
            sig_parts.append("early_game")
        elif "late" in text:
            sig_parts.append("late_game")
        elif "pass" in text and "bid" in cat:
            sig_parts.append("pass_strategy")
        elif "bid" in text and "conservative" in text:
            sig_parts.append("conservative_bid")
        elif "bid" in text and "aggressive" in text:
            sig_parts.append("aggressive_bid")
        elif "defend" in text:
            sig_parts.append("defense")
        else:
            # Use more of the text as signature — allows more diversity.
            words = [w for w in text.split() if len(w) > 4][:5]
            sig_parts.append("_".join(words))

        base_sig = "|".join(sig_parts)

        # Allow up to MAX_PER_CONCEPT per concept.
        count = concept_counts.get(base_sig, 0)
        if count < MAX_PER_CONCEPT:
            # Check exact text dedup (no two with identical text).
            unique_key = base_sig + f"_{count}"
            if text not in [s.get("text", "").lower() for s in seen.values()]:
                seen[unique_key] = ins
                concept_counts[base_sig] = count + 1
            else:
                # Exact duplicate — just bump confidence.
                for k, v in seen.items():
                    if v.get("text", "").lower() == text:
                        v["confidence"] = v.get("confidence", 1) + 1
                        break
        else:
            # Over limit — bump confidence on the first one.
            first_key = base_sig + "_0"
            if first_key in seen:
                seen[first_key]["confidence"] = seen[first_key].get("confidence", 1) + 1

    return list(seen.values())


def _refine_why(current_why: str, addition: str) -> str:
    """Refine the why text by appending a confidence qualifier."""
    if not current_why:
        return addition.capitalize()
    if addition.lower() in current_why.lower():
        return current_why
    return f"{current_why}. {addition.capitalize()}"


# ─── Cache Persistence ───────────────────────────────────────────────────────────


def _load_cache() -> list:
    try:
        if os.path.exists(_INSIGHTS_CACHE_PATH):
            with open(_INSIGHTS_CACHE_PATH, "r") as f:
                data = json.load(f)
            for ins in data:
                ins.setdefault("confidence", 1)
                ins.setdefault("why", "")
                ins.setdefault("version", 0)
                ins.setdefault("new", False)
                ins.setdefault("links", [])
                ins.setdefault("condition", None)
                ins.setdefault("exception", None)
                ins.setdefault("category", "timing")
                ins.setdefault("difficulty", "intermediate")
                ins.setdefault("episode", 0)
            return data
    except Exception:
        pass
    return []


def _capitalize_after_dots(text: str) -> str:
    """Ensure every sentence starts with a capital letter."""
    import re
    return re.sub(r'(\. )([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)


def _save_cache(insights):
    try:
        to_save = []
        for ins in insights:
            to_save.append({
                "text": ins.get("text", ""),
                "category": ins.get("category", "timing"),
                "difficulty": ins.get("difficulty", "intermediate"),
                "confidence": ins.get("confidence", 1),
                "episode": ins.get("episode", 0),
                "why": ins.get("why", ""),
                "version": ins.get("version", 0),
                "new": ins.get("new", False),
            })
        with open(_INSIGHTS_CACHE_PATH, "w") as f:
            json.dump(to_save, f)
    except Exception:
        pass


def _mine_deep_patterns(play_items, episodes) -> list:
    """Deeper cross-dimensional analysis of the Q-table."""
    insights = []

    # 1. Position × Rank: does card strength depend on when you play?
    pos_rank_avg = defaultdict(lambda: defaultdict(list))
    # 2. Suit × Position: does suit value change by position?
    pos_suit_avg = defaultdict(lambda: defaultdict(list))
    # 3. Action diversity: how many good options per state?
    positive_action_counts = []
    # 4. Hand size effect on best Q-value.
    handsize_bestq = defaultdict(list)

    for state, actions in play_items:
        parsed = _parse_state_key(state)
        if not parsed or len(actions) < 2:
            continue
        n_cards, pos = parsed

        # Count positive actions.
        pos_count = sum(1 for q in actions.values() if q > 0)
        positive_action_counts.append(pos_count)

        # Best Q per hand size.
        best_q = max(actions.values())
        handsize_bestq[n_cards].append(best_q)

        for key, q in actions.items():
            a_parsed = _parse_action_key(key)
            if not a_parsed:
                continue
            rank, suit = a_parsed
            pos_rank_avg[pos][rank].append(q)
            pos_suit_avg[pos][suit].append(q)

    # Insight 1: Position × Rank interaction.
    for pos in pos_rank_avg:
        ranks = pos_rank_avg[pos]
        rank_avgs = {r: sum(v)/len(v) for r, v in ranks.items() if len(v) >= 5}
        if len(rank_avgs) < 3:
            continue
        best_r = max(rank_avgs, key=rank_avgs.get)
        worst_r = min(rank_avgs, key=rank_avgs.get)
        spread = rank_avgs[best_r] - rank_avgs[worst_r]
        if spread < 0.3:
            continue
        pos_name = _POS_NAMES.get(pos, "this position")
        best_name = _RANK_NAMES.get(best_r, str(best_r))
        if best_r <= 7 and pos == 0:
            insights.append(_make(
                f"When leading a trick, low cards ({best_name}s) actually perform best, the agent uses them to probe safely",
                "timing", 1, episodes,
                why="leading low avoids wasting strong cards when you cannot see what opponents will play"
            ))
        elif best_r >= 13 and pos == 3:
            insights.append(_make(
                f"When playing last, high cards ({best_name}s) are most effective because you already know if they will win",
                "timing", 1, episodes,
                why="seeing all other cards before choosing means high cards are never wasted"
            ))
        elif best_r >= 13 and pos == 0:
            insights.append(_make(
                f"When leading, the agent leads with its strongest cards to seize control of the trick",
                "timing", 1, episodes,
                why="a strong opening forces opponents to spend their best cards or lose the trick"
            ))
        elif best_r <= 7 and pos == 3:
            insights.append(_make(
                f"When playing last, low cards work well because the trick is already decided and high cards are wasted",
                "counter-intuitive", 1, episodes,
                why="if your partner already won the trick, play your weakest to save strength for later"
            ))

    # Insight 2: Suit value changes by position.
    for pos in pos_suit_avg:
        suits = pos_suit_avg[pos]
        suit_avgs = {s: sum(v)/len(v) for s, v in suits.items() if len(v) >= 10}
        if len(suit_avgs) < 2:
            continue
        best_s = max(suit_avgs, key=suit_avgs.get)
        worst_s = min(suit_avgs, key=suit_avgs.get)
        spread = suit_avgs[best_s] - suit_avgs[worst_s]
        if spread < 0.25:
            continue
        pos_name = _POS_NAMES.get(pos, "this position")
        best_suit_name = _SUIT_NAMES.get(best_s, "?")
        if pos == 0:
            insights.append(_make(
                f"When leading, {best_suit_name} cards score highest, the agent prefers to lead from this suit",
                "trump", 1, episodes,
                why="leading from your strongest suit forces others into difficult positions"
            ))
        elif pos == 3:
            insights.append(_make(
                f"When playing last, {best_suit_name} is the most valuable suit to hold",
                "trump", 1, episodes,
                why="having cards in the right suit when playing last lets you control outcomes"
            ))

    # Insight 3: Action diversity — how decisive is the agent?
    if positive_action_counts and len(positive_action_counts) >= 50:
        avg_positives = sum(positive_action_counts) / len(positive_action_counts)
        if avg_positives < 2:
            insights.append(_make(
                "In most situations the agent sees only 1-2 good options, it has developed strong preferences",
                "timing", 1, episodes,
                why="fewer positive options means the agent clearly knows what works and what does not"
            ))
        elif avg_positives > 5:
            insights.append(_make(
                "The agent still sees many cards as roughly equal in most situations, strategy is still forming",
                "timing", 1, episodes,
                why="many positive options suggests the agent has not yet learned precise card selection"
            ))

    # Insight 4: Does hand size affect confidence?
    if handsize_bestq:
        early_qs = [q for n, qs in handsize_bestq.items() if n >= 10 for q in qs]
        late_qs = [q for n, qs in handsize_bestq.items() if n <= 4 for q in qs]
        if early_qs and late_qs and len(early_qs) >= 10 and len(late_qs) >= 10:
            early_avg = sum(early_qs) / len(early_qs)
            late_avg = sum(late_qs) / len(late_qs)
            if late_avg > early_avg + 0.2:
                insights.append(_make(
                    "The agent is more confident in late tricks when fewer cards remain, decisions become clearer",
                    "timing", 1, episodes,
                    why="with fewer unknowns, the agent can predict outcomes better and chooses more decisively"
                ))
            elif early_avg > late_avg + 0.2:
                insights.append(_make(
                    "The agent is more confident early when it has a full hand, suggesting strong opening strategy",
                    "timing", 1, episodes,
                    why="full hands give more options and the agent has learned which openers work best"
                ))

    # Insight 5: Suit concentration — does one suit dominate across all positions?
    overall_suit = defaultdict(list)
    for pos in pos_suit_avg:
        for suit, qs in pos_suit_avg[pos].items():
            overall_suit[suit].extend(qs)
    if overall_suit:
        suit_totals = {s: sum(v)/len(v) for s, v in overall_suit.items() if len(v) >= 20}
        if len(suit_totals) >= 3:
            sorted_suits = sorted(suit_totals.items(), key=lambda x: -x[1])
            top = sorted_suits[0]
            bottom = sorted_suits[-1]
            if top[1] - bottom[1] > 0.3:
                insights.append(_make(
                    f"{_SUIT_NAMES.get(top[0], '?').capitalize()} is the agent's strongest suit overall while {_SUIT_NAMES.get(bottom[0], '?')} is weakest, possibly reflecting trump awareness",
                    "trump", 1, episodes,
                    why="one suit consistently scoring higher across all positions suggests the agent knows which suit is trump"
                ))

    return insights[:30]  # Allow more deep insights.


def _mine_leading_vs_following(play_items, episodes) -> list:
    """Compare strategy when leading a trick vs following someone else's lead."""
    insights = []
    leading_qs = defaultdict(list)   # tier -> Q when position 0
    following_qs = defaultdict(list)  # tier -> Q when position 1-3

    for state, actions in play_items:
        parsed = _parse_state_key(state)
        if not parsed or len(actions) < 2:
            continue
        n_cards, pos = parsed
        for key, q in actions.items():
            a_parsed = _parse_action_key(key)
            if not a_parsed:
                continue
            rank, suit = a_parsed
            tier = _rank_tier(rank)
            if pos == 0:
                leading_qs[tier].append(q)
            else:
                following_qs[tier].append(q)

    if not leading_qs or not following_qs:
        return insights

    lead_avgs = {t: sum(v)/len(v) for t, v in leading_qs.items() if len(v) >= 5}
    follow_avgs = {t: sum(v)/len(v) for t, v in following_qs.items() if len(v) >= 5}

    if lead_avgs and follow_avgs:
        best_lead = max(lead_avgs, key=lead_avgs.get)
        best_follow = max(follow_avgs, key=follow_avgs.get)
        if best_lead != best_follow:
            insights.append(_make(
                f"Different cards work best in different roles: {best_lead} cards when leading, {best_follow} cards when following",
                "timing", 1, episodes,
                why="The agent has learned that leading and following require different approaches"
            ))

        # Specific: are low cards better for leading?
        if best_lead == "low" and best_follow == "high":
            insights.append(_make(
                "Lead with low cards to probe, save high cards for following when you can guarantee a win",
                "timing", 1, episodes,
                why="Leading low is safe exploration, following high is precise execution"
            ))
        elif best_lead == "high" and best_follow == "low":
            insights.append(_make(
                "Lead with strength to establish dominance, follow with low to conserve when tricks are decided",
                "timing", 1, episodes,
                why="Aggressive opening with conservative support maximizes trick count"
            ))

    return insights


def _mine_per_suit_rank(play_items, episodes) -> list:
    """Which rank performs best in each specific suit?"""
    insights = []
    suit_rank_qs = defaultdict(lambda: defaultdict(list))

    for state, actions in play_items:
        if len(actions) < 2:
            continue
        for key, q in actions.items():
            a_parsed = _parse_action_key(key)
            if not a_parsed:
                continue
            rank, suit = a_parsed
            suit_rank_qs[suit][rank].append(q)

    for suit in suit_rank_qs:
        rank_avgs = {r: sum(v)/len(v) for r, v in suit_rank_qs[suit].items() if len(v) >= 5}
        if len(rank_avgs) < 3:
            continue
        best_r = max(rank_avgs, key=rank_avgs.get)
        worst_r = min(rank_avgs, key=rank_avgs.get)
        spread = rank_avgs[best_r] - rank_avgs[worst_r]
        if spread < 0.2:
            continue
        suit_name = _SUIT_NAMES.get(suit, "?")
        best_name = _RANK_NAMES.get(best_r, str(best_r))
        worst_name = _RANK_NAMES.get(worst_r, str(worst_r))
        insights.append(_make(
            f"In {suit_name}, {best_name}s perform best while {worst_name}s perform worst",
            "trump", 1, episodes,
            why=f"The agent learned that card strength varies by suit, not all suits value the same ranks equally"
        ))

    return insights


def _mine_hand_size_suit_interaction(play_items, episodes) -> list:
    """Does suit preference change as the hand gets smaller?"""
    insights = []
    early_suit = defaultdict(list)  # suit -> Q early
    late_suit = defaultdict(list)   # suit -> Q late

    for state, actions in play_items:
        parsed = _parse_state_key(state)
        if not parsed or len(actions) < 2:
            continue
        n_cards, pos = parsed
        for key, q in actions.items():
            a_parsed = _parse_action_key(key)
            if not a_parsed:
                continue
            rank, suit = a_parsed
            if n_cards >= 10:
                early_suit[suit].append(q)
            elif n_cards <= 4:
                late_suit[suit].append(q)

    early_avgs = {s: sum(v)/len(v) for s, v in early_suit.items() if len(v) >= 10}
    late_avgs = {s: sum(v)/len(v) for s, v in late_suit.items() if len(v) >= 10}

    if early_avgs and late_avgs:
        best_early_s = max(early_avgs, key=early_avgs.get)
        best_late_s = max(late_avgs, key=late_avgs.get)
        if best_early_s != best_late_s:
            insights.append(_make(
                f"Suit preference shifts during the game: {_SUIT_NAMES.get(best_early_s, '?')} is best early but {_SUIT_NAMES.get(best_late_s, '?')} dominates late",
                "trump", 1, episodes,
                why="The agent discovers that different suits matter at different stages of each round"
            ))

    return insights


def _mine_granular_discoveries(play_items, bid_q, episodes) -> list:
    """
    Mine specific, surprising, data-backed discoveries from Q-table.

    Instead of broad categories, finds SPECIFIC contexts where one action
    dramatically outperforms others — with real numbers and explanations.
    """
    insights = []

    # 1. Find states where one action DOMINATES (best is >2x better than second best).
    dominant_plays = []
    for state, actions in play_items:
        parsed_s = _parse_state_key(state)
        if not parsed_s or len(actions) < 3:
            continue
        n_cards, pos = parsed_s
        sorted_a = sorted(actions.items(), key=lambda x: -x[1])
        best_key, best_q = sorted_a[0]
        second_key, second_q = sorted_a[1]
        worst_key, worst_q = sorted_a[-1]
        if best_q <= 0 or second_q <= 0:
            continue
        if best_q > second_q * 1.5 and best_q - worst_q > 0.3:
            best_parsed = _parse_action_key(best_key)
            if best_parsed:
                dominant_plays.append((n_cards, pos, best_parsed, best_q, second_q, worst_q, len(actions)))

    # Sort by dominance ratio and pick top discoveries.
    dominant_plays.sort(key=lambda x: x[3] / max(x[4], 0.01), reverse=True)
    for n_cards, pos, (rank, suit), best_q, second_q, worst_q, n_options in dominant_plays[:10]:
        trick_num = 14 - n_cards
        pos_name = _POS_NAMES.get(pos, "?")
        rank_name = _RANK_NAMES.get(rank, str(rank))
        suit_name = _SUIT_NAMES.get(suit, "?")
        ratio = best_q / max(second_q, 0.01)
        insights.append(_make(
            f"On trick {trick_num}, when {pos_name}, the {rank_name} of {suit_name} scores {ratio:.1f}x better than the next best option out of {n_options} choices",
            "timing", 1, episodes,
            why=f"This specific card in this exact situation is a discovered dominant play"
        ))

    # 2. Find REVERSALS: states where low cards beat high cards by a significant margin.
    reversals = []
    for state, actions in play_items:
        parsed_s = _parse_state_key(state)
        if not parsed_s or len(actions) < 4:
            continue
        n_cards, pos = parsed_s
        # Separate high vs low cards.
        high_qs = [(k, q) for k, q in actions.items() if _parse_action_key(k) and _parse_action_key(k)[0] >= 12]
        low_qs = [(k, q) for k, q in actions.items() if _parse_action_key(k) and _parse_action_key(k)[0] <= 6]
        if not high_qs or not low_qs:
            continue
        best_high = max(high_qs, key=lambda x: x[1])[1]
        best_low = max(low_qs, key=lambda x: x[1])
        if best_low[1] > best_high + 0.3:
            low_parsed = _parse_action_key(best_low[0])
            if low_parsed:
                reversals.append((n_cards, pos, low_parsed, best_low[1], best_high))

    for n_cards, pos, (rank, suit), low_q, high_q in reversals[:8]:
        trick_num = 14 - n_cards
        pos_name = _POS_NAMES.get(pos, "?")
        rank_name = _RANK_NAMES.get(rank, str(rank))
        suit_name = _SUIT_NAMES.get(suit, "?")
        margin = low_q - high_q
        insights.append(_make(
            f"Counter-intuitive: on trick {trick_num} when {pos_name}, the {rank_name} of {suit_name} outscores all high cards by {margin:.2f} points",
            "counter-intuitive", 1, episodes,
            why="The agent discovered that in this specific situation, playing low wins more than playing high"
        ))

    # 3. Find SUIT SPECIALIZATION: states where one suit dramatically outperforms.
    suit_dominance_states = []
    for state, actions in play_items:
        parsed_s = _parse_state_key(state)
        if not parsed_s or len(actions) < 4:
            continue
        n_cards, pos = parsed_s
        suit_avgs = defaultdict(list)
        for key, q in actions.items():
            parsed_a = _parse_action_key(key)
            if parsed_a:
                suit_avgs[parsed_a[1]].append(q)
        if len(suit_avgs) < 2:
            continue
        s_means = {s: sum(v)/len(v) for s, v in suit_avgs.items() if len(v) >= 2}
        if len(s_means) < 2:
            continue
        best_s = max(s_means, key=s_means.get)
        worst_s = min(s_means, key=s_means.get)
        spread = s_means[best_s] - s_means[worst_s]
        if spread > 0.4:
            suit_dominance_states.append((n_cards, pos, best_s, worst_s, spread))

    suit_dominance_states.sort(key=lambda x: x[4], reverse=True)
    for n_cards, pos, best_s, worst_s, spread in suit_dominance_states[:8]:
        trick_num = 14 - n_cards
        pos_name = _POS_NAMES.get(pos, "?")
        best_name = _SUIT_NAMES.get(best_s, "?")
        worst_name = _SUIT_NAMES.get(worst_s, "?")
        insights.append(_make(
            f"On trick {trick_num} when {pos_name}, {best_name} cards score {spread:.2f} points higher than {worst_name} cards",
            "trump", 1, episodes,
            why="The agent found that suit choice matters enormously in this context"
        ))

    # 4. Find POSITION POWER: biggest Q-value differences between positions for same hand size.
    pos_power = defaultdict(lambda: defaultdict(list))
    for state, actions in play_items:
        parsed_s = _parse_state_key(state)
        if not parsed_s or len(actions) < 2:
            continue
        n_cards, pos = parsed_s
        best_q = max(actions.values())
        pos_power[n_cards][pos].append(best_q)

    for n_cards, pos_data in pos_power.items():
        pos_means = {p: sum(v)/len(v) for p, v in pos_data.items() if len(v) >= 3}
        if len(pos_means) < 2:
            continue
        best_p = max(pos_means, key=pos_means.get)
        worst_p = min(pos_means, key=pos_means.get)
        diff = pos_means[best_p] - pos_means[worst_p]
        if diff > 0.3:
            trick_num = 14 - n_cards
            insights.append(_make(
                f"On trick {trick_num}, being {_POS_NAMES.get(best_p, '?')} gives {diff:.2f} more points than being {_POS_NAMES.get(worst_p, '?')}",
                "timing", 1, episodes,
                why="Position advantage changes throughout the game as information accumulates"
            ))

    # 5. Bid-specific findings.
    if bid_q:
        bid_items_list = list(bid_q.items())[:3000]
        bid_scores = defaultdict(list)
        for state, actions in bid_items_list:
            for action, q in actions.items():
                bid_scores[action].append(q)
        bid_means = {a: sum(v)/len(v) for a, v in bid_scores.items() if len(v) >= 5}
        if bid_means:
            sorted_bids = sorted(bid_means.items(), key=lambda x: -x[1])
            if len(sorted_bids) >= 2:
                best_bid, best_bq = sorted_bids[0]
                worst_bid, worst_bq = sorted_bids[-1]
                if best_bq - worst_bq > 0.2:
                    insights.append(_make(
                        f"The most successful bidding action is {best_bid} (avg score {best_bq:.2f}) while {worst_bid} scores worst ({worst_bq:.2f})",
                        "bidding", 1, episodes,
                        why="The agent discovered which bid levels consistently lead to better outcomes"
                    ))
            # Find bid that improved most over all others.
            for action, avg in sorted_bids[:3]:
                if action.startswith("B"):
                    val = action[1:]
                    insights.append(_make(
                        f"Bid {val} averages {avg:.2f} reward across all hand contexts the agent has seen",
                        "bidding", 1, episodes,
                        why=f"After hundreds of thousands of games, bid {val} is a learned sweet spot"
                    ))

    return insights


def _mine_granular_discoveries(play_items, bid_q, episodes) -> list:
    """
    Granular miner: one insight per SIGNIFICANT Q-table finding.
    Produces specific, data-backed, wow-worthy discoveries.
    """
    insights = []

    # 1. Decisive states: where one action dramatically outperforms.
    decisive_count = 0
    for state, actions in play_items:
        if len(actions) < 3 or decisive_count >= 30:
            continue
        parsed = _parse_state_key(state)
        if not parsed:
            continue
        n_cards, pos = parsed
        sorted_a = sorted(actions.items(), key=lambda x: -x[1])
        best_key, best_q = sorted_a[0]
        second_key, second_q = sorted_a[1]
        worst_key, worst_q = sorted_a[-1]
        spread = best_q - worst_q
        if spread < 0.5 or best_q - second_q < 0.2:
            continue
        best_p = _parse_action_key(best_key)
        worst_p = _parse_action_key(worst_key)
        if not best_p or not worst_p:
            continue
        best_rank, best_suit = best_p
        worst_rank, worst_suit = worst_p
        pos_name = _POS_NAMES.get(pos, f"position {pos}")
        best_name = _RANK_NAMES.get(best_rank, str(best_rank))
        worst_name = _RANK_NAMES.get(worst_rank, str(worst_rank))
        best_sn = _SUIT_NAMES.get(best_suit, "?")
        worst_sn = _SUIT_NAMES.get(worst_suit, "?")
        trick_num = 14 - n_cards

        if best_suit != worst_suit and best_rank <= 7 and worst_rank >= 12:
            text = (f"Trick {trick_num}, {pos_name}: a low {best_sn} ({best_name}) "
                    f"scores {spread:.1f} better than a high {worst_sn} ({worst_name})")
            cat = "counter-intuitive"
            why = "Suit choice matters more than rank in this situation"
        elif best_rank >= 13 and pos == 3:
            text = (f"Trick {trick_num}, playing last: {best_name} of {best_sn} "
                    f"wins by {spread:.1f} points with full visibility")
            cat = "timing"
            why = "Seeing all cards before choosing makes high cards safe bets"
        elif best_rank <= 5 and pos == 0:
            text = (f"Trick {trick_num}, leading: the agent leads with {best_name} "
                    f"of {best_sn}, a low card, with {spread:.1f} advantage")
            cat = "counter-intuitive"
            why = "Leading low probes safely and saves high cards for following"
        elif best_suit != worst_suit:
            text = (f"Trick {trick_num}, {pos_name}: {best_sn} cards outperform "
                    f"{worst_sn} by {spread:.1f} points")
            cat = "trump"
            why = "One suit is clearly stronger than another in this context"
        else:
            text = (f"Trick {trick_num}, {pos_name}: {best_name} of {best_sn} "
                    f"beats alternatives by {spread:.1f}")
            cat = "timing"
            why = "The agent found this to be the clear best play here"

        insights.append(_make(text, cat, 1, episodes, why=why))
        decisive_count += 1

    # 2. Bid discoveries — specific bid levels that win or lose.
    if bid_q:
        bid_count = 0
        for state, actions in list(bid_q.items())[:2000]:
            if len(actions) < 2 or bid_count >= 15:
                continue
            sorted_b = sorted(actions.items(), key=lambda x: -x[1])
            best_bid, best_bq = sorted_b[0]
            worst_bid, worst_bq = sorted_b[-1]
            spread = best_bq - worst_bq
            if spread < 0.5:
                continue
            if best_bid == "PASS" and worst_bid.startswith("B"):
                wv = worst_bid[1:]
                insights.append(_make(
                    f"Passing scores {spread:.1f} better than bidding {wv} with this hand shape. Defense wins here",
                    "defense", 1, episodes,
                    why="This hand looks biddable but consistently fails to deliver"
                ))
            elif best_bid.startswith("B") and worst_bid == "PASS":
                bv = best_bid[1:]
                insights.append(_make(
                    f"Bidding {bv} scores {spread:.1f} better than passing. This hand delivers reliably",
                    "bidding", 1, episodes,
                    why="The agent learned to commit when hand shape supports it"
                ))
            elif best_bid.startswith("B") and worst_bid.startswith("B"):
                bv = best_bid[1:]
                wv = worst_bid[1:]
                insights.append(_make(
                    f"Bid {bv} beats bid {wv} by {spread:.1f} points. Precise bid level matters",
                    "bidding", 1, episodes,
                    why="Over or under-bidding both cost points, precision is learned"
                ))
            bid_count += 1

    # 3. Position success rates.
    pos_stats = defaultdict(lambda: {"pos_total": 0, "pos_wins": 0})
    for state, actions in play_items:
        parsed = _parse_state_key(state)
        if not parsed:
            continue
        n_cards, pos = parsed
        best_q = max(actions.values()) if actions else 0
        pos_stats[pos]["pos_total"] += 1
        if best_q > 0.3:
            pos_stats[pos]["pos_wins"] += 1

    for pos, data in pos_stats.items():
        if data["pos_total"] < 20:
            continue
        rate = data["pos_wins"] / data["pos_total"] * 100
        pn = _POS_NAMES.get(pos, f"position {pos}")
        if rate > 70:
            insights.append(_make(
                f"When {pn}, the agent succeeds {rate:.0f}% of the time. Dominant position",
                "timing", 1, episodes,
                why="More information or control makes this position consistently strong"
            ))
        elif rate < 30:
            insights.append(_make(
                f"When {pn}, the agent only succeeds {rate:.0f}% of the time. Hardest position",
                "timing", 1, episodes,
                why="Limited information forces difficult choices from this seat"
            ))

    return insights


# ─── Snapshot Comparison System ──────────────────────────────────────────────────


def _take_snapshot(agent) -> dict:
    """Take a lightweight snapshot of current strategy preferences."""
    play_items = list(agent.play_q.items())[:10000]
    bid_items = list(agent.bid_q.items())[:2000]

    # Aggregate: best action per state context (position, hand size).
    pos_prefs = defaultdict(lambda: defaultdict(float))  # pos -> tier -> avg Q
    suit_prefs = defaultdict(float)  # suit -> avg Q
    bid_prefs = defaultdict(float)   # action -> avg Q
    rank_prefs = defaultdict(float)  # rank -> avg Q
    rank_counts = defaultdict(int)

    for state, actions in play_items:
        parsed = _parse_state_key(state)
        if not parsed or len(actions) < 2:
            continue
        n_cards, pos = parsed
        for key, q in actions.items():
            a_parsed = _parse_action_key(key)
            if not a_parsed:
                continue
            rank, suit = a_parsed
            tier = _rank_tier(rank)
            pos_prefs[pos][tier] += q
            suit_prefs[suit] += q
            rank_prefs[rank] += q
            rank_counts[rank] += 1

    # Normalize.
    for rank in rank_prefs:
        if rank_counts[rank] > 0:
            rank_prefs[rank] /= rank_counts[rank]

    for state, actions in bid_items:
        for action, q in actions.items():
            bid_prefs[action] += q

    return {
        "pos_prefs": {str(k): dict(v) for k, v in pos_prefs.items()},
        "suit_prefs": dict(suit_prefs),
        "bid_prefs": dict(bid_prefs),
        "rank_prefs": dict(rank_prefs),
    }


def _load_snapshots() -> dict:
    """Load saved snapshots {episode_str: snapshot_dict}."""
    try:
        if os.path.exists(_SNAPSHOTS_PATH):
            with open(_SNAPSHOTS_PATH, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_snapshots(snapshots: dict):
    """Save snapshots to disk."""
    try:
        with open(_SNAPSHOTS_PATH, "w") as f:
            json.dump(snapshots, f)
    except Exception:
        pass


def _check_and_take_snapshot(agent) -> list:
    """
    Recurring snapshot system. At every interval boundary, take a snapshot
    and compare with the previous snapshot at that same interval.

    E.g., at episode 150000:
    - 50K interval: snapshot #3 (compare with snapshot #2 at 100K)
    - 100K interval: snapshot #1 (first at this interval, no comparison yet)

    Snapshots stored as: {interval: {slot_number: snapshot_data}}
    """
    episodes = agent.episodes_trained
    snapshots = _load_snapshots()
    new_insights = []

    for interval in _SNAPSHOT_INTERVALS:
        # Which slot are we at? E.g., episodes=150000, interval=50000 → slot 3.
        current_slot = episodes // interval
        if current_slot < 1:
            continue

        interval_key = str(interval)
        if interval_key not in snapshots:
            snapshots[interval_key] = {}

        slot_key = str(current_slot)
        if slot_key in snapshots[interval_key]:
            continue  # Already have this snapshot.

        # Take snapshot for this slot.
        snap = _take_snapshot(agent)
        snapshots[interval_key][slot_key] = snap

        # Compare with previous slot if it exists.
        prev_slot_key = str(current_slot - 1)
        if prev_slot_key in snapshots[interval_key]:
            prev = snapshots[interval_key][prev_slot_key]
            # Label: e.g., "1.9M→2.0M" or "100K→150K"
            prev_ep = (current_slot - 1) * interval
            curr_ep = current_slot * interval
            prev_label = _format_ep(prev_ep)
            curr_label = _format_ep(curr_ep)
            interval_label = _format_ep(interval)

            new_insights.extend(
                _compare_two_snapshots(prev, snap, prev_label, curr_label, interval_label, episodes)
            )

        _save_snapshots(snapshots)

    return new_insights


def _format_ep(ep: int) -> str:
    """Format episode count as readable string."""
    if ep >= 1000000:
        return f"{ep / 1000000:.1f}M"
    elif ep >= 1000:
        return f"{ep // 1000}K"
    return str(ep)


def _compare_two_snapshots(prev, current, prev_label, curr_label, interval_label, episodes) -> list:
    """Compare two snapshots and generate insights about what changed. Plain English, no episode labels."""
    insights = []

    # Compare rank preferences.
    prev_ranks = prev.get("rank_prefs", {})
    curr_ranks = current.get("rank_prefs", {})
    if prev_ranks and curr_ranks:
        prev_best = max(prev_ranks, key=prev_ranks.get)
        curr_best = max(curr_ranks, key=curr_ranks.get)
        if prev_best != curr_best:
            prev_name = _RANK_NAMES.get(int(prev_best), prev_best)
            curr_name = _RANK_NAMES.get(int(curr_best), curr_best)
            insights.append(_make(
                f"The agent shifted from valuing {prev_name}s most to preferring {curr_name}s, suggesting a change in how it evaluates card strength",
                "counter-intuitive", 1, episodes,
                why=f"the highest-reward card changed, indicating the agent discovered a new pattern about which cards win"
            ))

    # Compare suit preferences.
    prev_suits = prev.get("suit_prefs", {})
    curr_suits = current.get("suit_prefs", {})
    if prev_suits and curr_suits:
        prev_best_s = max(prev_suits, key=prev_suits.get)
        curr_best_s = max(curr_suits, key=curr_suits.get)
        if prev_best_s != curr_best_s:
            prev_s = _SUIT_NAMES.get(int(prev_best_s), '?')
            curr_s = _SUIT_NAMES.get(int(curr_best_s), '?')
            insights.append(_make(
                f"The dominant suit changed from {prev_s} to {curr_s}, the agent may be adapting to different trump patterns",
                "trump", 1, episodes,
                why="suit preference shifts reveal the agent learning which suit controls each game"
            ))

    # Compare bid preferences.
    prev_bids = prev.get("bid_prefs", {})
    curr_bids = current.get("bid_prefs", {})
    if prev_bids and curr_bids:
        prev_best_b = max(prev_bids, key=prev_bids.get)
        curr_best_b = max(curr_bids, key=curr_bids.get)
        if prev_best_b != curr_best_b:
            if prev_best_b == "PASS" and curr_best_b.startswith("B"):
                bid_val = curr_best_b[1:]
                insights.append(_make(
                    f"The agent became more confident, now preferring to bid {bid_val} rather than passing and defending",
                    "bidding", 1, episodes,
                    why="shifting from passive to active bidding means the agent trusts its hand reading more"
                ))
            elif prev_best_b.startswith("B") and curr_best_b == "PASS":
                insights.append(_make(
                    "The agent became more cautious, now preferring to pass rather than commit to a bid",
                    "defense", 1, episodes,
                    why="shifting to defense suggests the agent learned that failed bids are costly"
                ))
            elif prev_best_b.startswith("B") and curr_best_b.startswith("B"):
                prev_val = prev_best_b[1:]
                curr_val = curr_best_b[1:]
                if int(curr_val) > int(prev_val):
                    insights.append(_make(
                        f"Bidding became bolder, moving from bid {prev_val} to bid {curr_val}, the agent is finding more winning potential in its hands",
                        "bidding", 1, episodes,
                        why="higher bids indicate the agent discovered it can win more tricks than it previously thought"
                    ))
                else:
                    insights.append(_make(
                        f"Bidding became more conservative, dropping from bid {prev_val} to bid {curr_val}, learning that over-promising leads to penalties",
                        "bidding", 1, episodes,
                        why="lower bids suggest the agent learned to be more realistic about hand strength"
                    ))

    # Check for convergence or divergence in rank values.
    if curr_ranks and prev_ranks and len(curr_ranks) >= 4 and len(prev_ranks) >= 4:
        curr_values = list(curr_ranks.values())
        prev_values = list(prev_ranks.values())
        curr_spread = max(curr_values) - min(curr_values)
        prev_spread = max(prev_values) - min(prev_values)
        if prev_spread > 0.1 and curr_spread < prev_spread * 0.5:
            insights.append(_make(
                "The agent now treats all card ranks more equally, meaning context and timing matter more than raw card power",
                "timing", 1, episodes,
                why="converging values show the agent learned that WHEN to play matters more than WHAT to play"
            ))
        elif curr_spread > prev_spread * 1.5 and prev_spread > 0.05:
            insights.append(_make(
                "The agent is developing stronger opinions about which cards are powerful, becoming more decisive in card selection",
                "timing", 1, episodes,
                why="widening value gaps show a clearer internal hierarchy of card strength forming"
            ))

    return insights


# ─── Public Helpers ──────────────────────────────────────────────────────────────


def get_insight_texts(insights) -> list:
    """Extract just the text from structured insights."""
    return [ins["text"] if isinstance(ins, dict) else ins for ins in insights]


def _load_cached_insights() -> list:
    """Alias for external callers."""
    return _load_cache()
