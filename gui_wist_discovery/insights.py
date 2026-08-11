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
    """Parse action key: {rank_hex}{suit_index} -> (rank_int, suit_int) or None."""
    try:
        if len(key) < 2:
            return None
        rank = int(key[:-1], 16)
        suit = int(key[-1])
        if 2 <= rank <= 14 and 0 <= suit <= 3:
            return (rank, suit)
    except (ValueError, IndexError):
        pass
    return None


def _parse_state_key(key: str):
    """Parse state key: {n_cards_hex}{position} -> (n_cards, position) or None."""
    try:
        if len(key) < 2:
            return None
        n_cards = int(key[:-1], 16)
        pos = int(key[-1])
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
    """Merge insights by core idea — max 2 per concept."""
    seen = {}

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
            words = [w for w in text.split() if len(w) > 4][:3]
            sig_parts.append("_".join(words))

        sig = "|".join(sig_parts)

        if sig in seen:
            alt_sig = sig + "_alt"
            if alt_sig not in seen:
                seen[alt_sig] = ins
            else:
                seen[sig]["confidence"] = seen[sig].get("confidence", 1) + 1
        else:
            seen[sig] = ins

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
    """Compare two snapshots and generate insights about what changed."""
    insights = []

    # Compare rank preferences.
    prev_ranks = prev.get("rank_prefs", {})
    curr_ranks = current.get("rank_prefs", {})
    if prev_ranks and curr_ranks:
        prev_best = max(prev_ranks, key=prev_ranks.get)
        curr_best = max(curr_ranks, key=curr_ranks.get)
        if prev_best != curr_best:
            insights.append(_make(
                f"Between {prev_label} and {curr_label}: strongest card shifted from {_RANK_NAMES.get(int(prev_best), prev_best)} to {_RANK_NAMES.get(int(curr_best), curr_best)}",
                "counter-intuitive", 1, episodes,
                why=f"over {interval_label} episodes the agent changed which card rank it values most"
            ))

    # Compare suit preferences.
    prev_suits = prev.get("suit_prefs", {})
    curr_suits = current.get("suit_prefs", {})
    if prev_suits and curr_suits:
        prev_best_s = max(prev_suits, key=prev_suits.get)
        curr_best_s = max(curr_suits, key=curr_suits.get)
        if prev_best_s != curr_best_s:
            insights.append(_make(
                f"Between {prev_label} and {curr_label}: dominant suit changed from {_SUIT_NAMES.get(int(prev_best_s), '?')} to {_SUIT_NAMES.get(int(curr_best_s), '?')}",
                "trump", 1, episodes,
                why=f"the agent's suit preference evolved over {interval_label} episodes"
            ))

    # Compare bid preferences.
    prev_bids = prev.get("bid_prefs", {})
    curr_bids = current.get("bid_prefs", {})
    if prev_bids and curr_bids:
        prev_best_b = max(prev_bids, key=prev_bids.get)
        curr_best_b = max(curr_bids, key=curr_bids.get)
        if prev_best_b != curr_best_b:
            insights.append(_make(
                f"Between {prev_label} and {curr_label}: bidding preference changed from {prev_best_b} to {curr_best_b}",
                "bidding", 1, episodes,
                why=f"bidding strategy evolved over {interval_label} episodes of play"
            ))

    # Check for convergence or divergence in rank values.
    if curr_ranks and prev_ranks and len(curr_ranks) >= 4 and len(prev_ranks) >= 4:
        curr_values = list(curr_ranks.values())
        prev_values = list(prev_ranks.values())
        curr_spread = max(curr_values) - min(curr_values)
        prev_spread = max(prev_values) - min(prev_values)
        if prev_spread > 0.1 and curr_spread < prev_spread * 0.5:
            insights.append(_make(
                f"Between {prev_label} and {curr_label}: card values are converging, context matters more than raw rank now",
                "timing", 1, episodes,
                why=f"the agent learned that WHEN to play matters more than WHAT to play"
            ))
        elif curr_spread > prev_spread * 1.5 and prev_spread > 0.05:
            insights.append(_make(
                f"Between {prev_label} and {curr_label}: card rank differences are widening, the agent is more decisive about which cards are best",
                "timing", 1, episodes,
                why=f"stronger opinions forming about card strength hierarchy"
            ))

    return insights


# ─── Public Helpers ──────────────────────────────────────────────────────────────


def get_insight_texts(insights) -> list:
    """Extract just the text from structured insights."""
    return [ins["text"] if isinstance(ins, dict) else ins for ins in insights]


def _load_cached_insights() -> list:
    """Alias for external callers."""
    return _load_cache()
