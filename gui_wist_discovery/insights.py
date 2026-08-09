"""
Strategic Insights System — AUTO-GENERATED from Q-table patterns.

No hardcoded insight text. Every insight is composed from detected patterns:
- WHO: position (leading, second, third, last)
- WHEN: trick range (1-3, 4-5, 6-9, 10-11, 12-13)
- WHAT: the winning action vs the losing action
- HOW MUCH: advantage magnitude
- CONTEXT: game state (ahead, behind, trumps, voids)

Insights grow proportionally with Q-table size — truly unlimited.
Counter-intuitive patterns are auto-detected when "obvious" plays lose.
"""

from collections import defaultdict
import json
import os

_INSIGHTS_CACHE_PATH = "agents/wist_discovery/insights_cache.json"
_SNAPSHOTS_PATH = "agents/wist_discovery/strategy_snapshots.json"

# ─── Decoding Maps ───────────────────────────────────────────────────────────────

_POS_DESC = {"0": "leading", "1": "playing second", "2": "playing third", "3": "playing last"}
_PHASE_DESC = {"1": "tricks 1-3", "2": "tricks 4-5", "3": "tricks 6-9", "4": "tricks 10-11", "5": "tricks 12-13"}
_TD_DESC = {"W": "ahead", "A": "slightly ahead", "T": "tied", "B": "behind"}
_TIER_RANK = {"A": 5, "K": 4, "Q": 3, "J": 2, "M": 1, "L": 0, "X": -1}

_ACTION_DESC = {
    ("A", "F", "N"): "play your Ace following suit",
    ("K", "F", "N"): "play your King following suit",
    ("Q", "F", "N"): "play your Queen following suit",
    ("J", "F", "N"): "play a Jack following suit",
    ("M", "F", "N"): "play a mid card (9-10) following suit",
    ("L", "F", "N"): "play a low card following suit",
    ("X", "F", "N"): "follow with your weakest card",
    ("A", "O", "N"): "discard your Ace (can't follow)",
    ("K", "O", "N"): "discard your King (can't follow)",
    ("L", "O", "N"): "discard a low card (can't follow)",
    ("X", "O", "N"): "throw away your weakest from another suit",
    ("A", "F", "T"): "follow with your top trump",
    ("K", "F", "T"): "follow with your King of trump",
    ("Q", "F", "T"): "follow with your Queen of trump",
    ("L", "F", "T"): "follow with a low trump",
    ("X", "F", "T"): "follow with your weakest trump",
    ("A", "O", "T"): "trump with your Ace of trump",
    ("K", "O", "T"): "trump with your King",
    ("Q", "O", "T"): "trump with your Queen",
    ("L", "O", "T"): "trump with a low trump",
    ("X", "O", "T"): "trump with your weakest trump",
    ("M", "O", "T"): "trump with a mid trump",
    ("M", "O", "N"): "discard a mid card",
    ("J", "O", "N"): "discard a Jack",
    ("J", "F", "T"): "follow with Jack of trump",
    ("M", "F", "T"): "follow with mid trump",
    ("J", "O", "T"): "trump with a Jack",
}

def _describe_action(key: str) -> str:
    """Auto-describe an action key like 'AFNLK' into human text."""
    if len(key) < 3:
        return "play a card"
    tier, follows, trump = key[0], key[1], key[2]
    desc = _ACTION_DESC.get((tier, follows, trump), None)
    if desc:
        return desc
    # Fallback auto-compose.
    tier_name = {"A": "Ace", "K": "King", "Q": "Queen", "J": "Jack", "M": "mid card", "L": "low card", "X": "weakest card"}.get(tier, "card")
    if trump == "T" and follows == "O":
        return f"trump with your {tier_name}"
    elif trump == "T":
        return f"follow with {tier_name} of trump"
    elif follows == "O":
        return f"discard your {tier_name}"
    else:
        return f"play your {tier_name}"


def _why_from_pattern(best_key, worst_key, spread, context_desc) -> str:
    """Auto-generate WHY based on what wins vs what loses."""
    best_tier = _TIER_RANK.get(best_key[0], 0) if best_key else 0
    worst_tier = _TIER_RANK.get(worst_key[0], 0) if worst_key else 0
    best_trump = best_key[2] == "T" if len(best_key) > 2 else False
    worst_trump = worst_key[2] == "T" if len(worst_key) > 2 else False
    best_follows = best_key[1] == "F" if len(best_key) > 1 else False
    best_void = len(best_key) > 4 and best_key[4] == "V"

    if best_trump and not worst_trump:
        return "trump beats everything — even a 2 of trump wins against any non-trump card"
    if best_tier < worst_tier and not best_trump:
        return "saving your strong cards for later gives them more value when fewer cards remain"
    if best_tier > worst_tier and best_follows:
        return "your high card wins the trick now — no point waiting if you can take it"
    if best_void:
        return "creating a void opens up free tricks through trumping later"
    if best_trump and best_tier < 2:
        return "even your weakest trump wins here — save the big ones for when opponents also trump"
    if spread > 3:
        return "the difference is massive — doing anything else costs heavily"
    if context_desc and "behind" in context_desc:
        return "when behind, you need tricks NOW — waiting means losing slowly"
    if context_desc and "ahead" in context_desc:
        return "when ahead, protect your lead — don't risk what you already have"
    return "this play consistently outperforms all alternatives in this situation"



def _categorize(best_key, worst_key, is_counter_intuitive) -> str:
    """Auto-categorize based on action properties."""
    if is_counter_intuitive:
        return "counter-intuitive"
    if len(best_key) >= 3 and best_key[2] == "T":
        return "trump"
    if len(best_key) >= 5 and best_key[4] == "V":
        return "voids"
    if len(best_key) >= 2 and best_key[1] == "O" and best_key[0] in "LX":
        return "voids"
    return "timing"


# ─── Main Entry Point ────────────────────────────────────────────────────────────


def generate_insights(agent) -> list:
    """
    Auto-generate insights by mining Q-table patterns.
    Returns list of insight dicts. Accumulates over time.
    """
    if agent.episodes_trained < 5000:
        return [_make(
            "Still learning basics — building Q-table...",
            "timing", "beginner", 1, agent.episodes_trained,
            why="Need more games to detect meaningful patterns"
        )]

    episodes = agent.episodes_trained
    play_q = agent.play_q
    bid_q = agent.bid_q
    play_items = list(play_q.items())[:min(len(play_q), 25000)]

    # Load accumulated history.
    accumulated = _load_cache()
    acc_texts = {ins["text"] for ins in accumulated}

    # Mine patterns from Q-tables.
    new_insights = []
    new_insights.extend(_mine_play_patterns(play_items, episodes))
    new_insights.extend(_mine_bid_patterns(bid_q, episodes))
    new_insights.extend(_mine_counter_intuitive(play_items, bid_q, episodes))

    # Merge new into accumulated.
    for ins in new_insights:
        if ins["text"] in acc_texts:
            # Re-confirmed — increment +N.
            for acc in accumulated:
                if acc["text"] == ins["text"]:
                    acc["confidence"] = acc.get("confidence", 1) + 1
                    break
        else:
            # Brand new discovery.
            accumulated.append(ins)
            acc_texts.add(ins["text"])

    # Save.
    _save_cache(accumulated)
    return accumulated


def _make(text, category, difficulty, confidence, episode, why=""):
    """Create insight dict."""
    return {
        "text": text,
        "category": category,
        "difficulty": difficulty,
        "confidence": confidence,  # Now an integer: +1, +2, +3, etc.
        "episode": episode,
        "why": why,
        "version": 0,
        "new": True,
        "links": [],
        "condition": None,
        "exception": None,
    }



# ─── Pattern Miners ──────────────────────────────────────────────────────────────


def _mine_play_patterns(play_items, episodes) -> list:
    """Mine play Q-table for decisive patterns. Auto-generates descriptions."""
    insights = []

    # Group by (position, phase, trick_diff) context.
    context_actions = defaultdict(lambda: defaultdict(list))

    for state, actions in play_items:
        if len(state) < 7 or len(actions) < 2:
            continue
        pos = state[4]
        phase = state[5]
        td = state[6] if len(state) > 6 else "T"
        ctx = (pos, phase, td)

        for key, q in actions.items():
            if len(key) >= 3:
                context_actions[ctx][key].append(q)

    # For each context, find the best and worst actions.
    for ctx, action_qs in context_actions.items():
        pos, phase, td = ctx
        pos_desc = _POS_DESC.get(pos, "")
        phase_desc = _PHASE_DESC.get(phase, "")
        td_desc = _TD_DESC.get(td, "")
        if not pos_desc or not phase_desc:
            continue

        # Compute averages per action.
        avg_by_action = {}
        for key, vals in action_qs.items():
            if len(vals) >= 3:
                avg_by_action[key] = sum(vals) / len(vals)

        if len(avg_by_action) < 2:
            continue

        # Find best and worst.
        sorted_actions = sorted(avg_by_action.items(), key=lambda x: -x[1])
        best_key, best_avg = sorted_actions[0]
        worst_key, worst_avg = sorted_actions[-1]
        spread = best_avg - worst_avg

        if spread < 0.5:
            continue  # Not decisive enough.

        # Check if counter-intuitive (low card beats high card).
        best_rank = _TIER_RANK.get(best_key[0], 0)
        worst_rank = _TIER_RANK.get(worst_key[0], 0)
        is_counter = (best_rank < worst_rank - 1)  # Low card beating much higher card.

        # Auto-generate description.
        action_desc = _describe_action(best_key)
        context_parts = [f"When {pos_desc} in {phase_desc}"]
        if td_desc and td != "T":
            context_parts.append(f"and your team is {td_desc}")
        context_str = " ".join(context_parts)

        text = f"{context_str}, {action_desc}"
        if is_counter:
            text = f"Counter-intuitive: {context_str}, {action_desc} (beats {_describe_action(worst_key)})"

        why = _why_from_pattern(best_key, worst_key, spread, td_desc)
        category = _categorize(best_key, worst_key, is_counter)
        difficulty = "advanced" if is_counter else ("intermediate" if spread > 1.5 else "beginner")

        insights.append(_make(text, category, difficulty, 1, episodes, why=why))

    # Cap per scan but keep growing across scans.
    return insights[:40]


def _mine_bid_patterns(bid_q, episodes) -> list:
    """Mine bid Q-table for hand-specific bidding advice."""
    insights = []

    # Group by hand features.
    hand_groups = defaultdict(lambda: defaultdict(list))

    for state, actions in list(bid_q.items())[:3000]:
        if len(state) < 5:
            continue
        try:
            longest = int(state[0])
            highs = int(state[2])
            aces = int(state[3])
            v_idx = state.index("v")
            voids = int(state[v_idx + 1])
        except (ValueError, IndexError):
            continue

        hand_key = (longest, highs, voids)
        for action, q in actions.items():
            hand_groups[hand_key][action].append(q)

    for (longest, highs, voids), action_qs in hand_groups.items():
        avg_by_action = {}
        for action, vals in action_qs.items():
            if len(vals) >= 2:
                avg_by_action[action] = sum(vals) / len(vals)

        if len(avg_by_action) < 2:
            continue

        sorted_actions = sorted(avg_by_action.items(), key=lambda x: -x[1])
        best_action, best_avg = sorted_actions[0]
        second_avg = sorted_actions[1][1] if len(sorted_actions) > 1 else 0
        spread = best_avg - second_avg

        if spread < 0.3:
            continue

        # Build hand description.
        hand_parts = []
        if longest >= 5:
            hand_parts.append(f"a {longest}-card suit")
        if highs >= 3:
            hand_parts.append(f"{highs}+ high cards")
        if voids >= 1:
            hand_parts.append(f"{voids} void{'s' if voids > 1 else ''}")
        if not hand_parts:
            continue
        hand_desc = ", ".join(hand_parts)

        # Action description.
        if best_action == "PASS":
            action_desc = "pass and defend"
            why = "this hand looks decent but fails to deliver reliably when bidding"
            # Counter-intuitive if hand looks strong.
            is_counter = highs >= 3
        elif best_action.startswith("B"):
            try:
                val = int(best_action[1:])
                action_desc = f"bid {val}"
                why = f"this hand shape delivers exactly {val} tricks consistently"
                is_counter = (highs >= 4 and val <= 7)  # Strong hand but low bid.
            except ValueError:
                continue
        else:
            continue

        text = f"When your hand has {hand_desc}, {action_desc}"
        if is_counter:
            text = f"Counter-intuitive: hand with {hand_desc}, {action_desc} works better than expected"

        category = "counter-intuitive" if is_counter else "bidding"
        insights.append(_make(text, category, "intermediate", 1, episodes, why=why))

    return insights[:20]



def _mine_counter_intuitive(play_items, bid_q, episodes) -> list:
    """Specifically hunt for patterns where the 'obvious' play loses."""
    insights = []

    # Group by context and find where low beats high.
    ctx_tiers = defaultdict(lambda: defaultdict(list))
    for state, actions in play_items:
        if len(state) < 6:
            continue
        pos = state[4]
        phase = state[5]
        for key, q in actions.items():
            if len(key) >= 3:
                tier = key[0]
                ctx_tiers[(pos, phase)][tier].append(q)

    for (pos, phase), tier_vals in ctx_tiers.items():
        # Compute avg per tier.
        tier_avgs = {}
        for tier, vals in tier_vals.items():
            if len(vals) >= 3:
                tier_avgs[tier] = sum(vals) / len(vals)

        if len(tier_avgs) < 3:
            continue

        # Check reversals: low > high.
        low_avg = tier_avgs.get("X", tier_avgs.get("L", None))
        high_avg = tier_avgs.get("A", tier_avgs.get("K", None))
        mid_avg = tier_avgs.get("M", tier_avgs.get("J", None))

        pos_desc = _POS_DESC.get(pos, "")
        phase_desc = _PHASE_DESC.get(phase, "")

        if low_avg is not None and high_avg is not None and low_avg > high_avg + 0.2:
            text = f"Counter-intuitive: in {phase_desc} when {pos_desc}, low cards outperform Aces and Kings, save your power for other moments"
            why = "high cards attract trumps from void opponents here. Low cards fly under the radar and preserve your hand"
            insights.append(_make(text, "counter-intuitive", "advanced", 1, episodes, why=why))

        if mid_avg is not None and high_avg is not None and mid_avg > high_avg + 0.2:
            text = f"Counter-intuitive: in {phase_desc} when {pos_desc}, mid cards (9s, 10s, Jacks) beat Aces, opponents target your high cards but ignore middle ones"
            why = "opponents save trumps to kill your Aces. Mid cards win tricks nobody fights over"
            insights.append(_make(text, "counter-intuitive", "advanced", 1, episodes, why=why))

    # Bid reversals: strong hand but pass wins.
    for state, actions in list(bid_q.items())[:1000]:
        if len(state) < 5:
            continue
        try:
            highs = int(state[2])
        except (ValueError, IndexError):
            continue
        if highs < 3:
            continue
        pass_q = actions.get("PASS", None)
        if pass_q is None:
            continue
        bid_qs = [q for k, q in actions.items() if k.startswith("B")]
        if not bid_qs:
            continue
        best_bid_q = max(bid_qs)
        if pass_q > best_bid_q + 0.3:
            text = f"Counter-intuitive: with {highs} high cards, passing beats any bid, raw card power without trump length is a trap"
            why = "high cards spread across multiple suits get trumped. Trump count matters more than face cards"
            insights.append(_make(text, "counter-intuitive", "advanced", 1, episodes, why=why))
            break  # One per scan.

    return insights[:15]


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
                ins.setdefault("difficulty", "beginner")
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
                "difficulty": ins.get("difficulty", "beginner"),
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


# ─── Helpers ─────────────────────────────────────────────────────────────────────


def get_insight_texts(insights) -> list:
    """Extract just the text from structured insights."""
    return [ins["text"] if isinstance(ins, dict) else ins for ins in insights]


def _load_cached_insights() -> list:
    """Alias for external callers."""
    return _load_cache()
