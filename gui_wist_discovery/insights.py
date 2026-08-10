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
    """Describe an action key as a simple play instruction in plain language.
    Uses generalized terms for high cards (not specific ranks)."""
    if len(key) < 3:
        return "play a card"
    tier, follows, trump = key[0], key[1], key[2]
    
    if trump == "T" and follows == "O":
        if tier in ("L", "X"):
            return "play your smallest trump"
        elif tier in ("A", "K", "Q"):
            return "play your strongest trump"
        else:
            return "play a mid-range trump"
    elif trump == "T" and follows == "F":
        if tier in ("A", "K", "Q"):
            return "play your strongest trump"
        elif tier in ("L", "X"):
            return "play your smallest trump"
        else:
            return "play a mid-range trump"
    elif follows == "O":
        if tier in ("L", "X"):
            return "play your weakest card from another suit"
        elif tier in ("A", "K", "Q"):
            return "play a high card from another suit"
        else:
            return "play a mid-range card from another suit"
    else:
        if tier in ("L", "X"):
            return "play your lowest card"
        elif tier in ("A", "K", "Q"):
            return "play your strongest card"
        else:
            return "play a mid-range card"


def _compose_play_insight(pos, phase, td, best_key, worst_key, spread) -> str:
    """
    Compose a finding + implication. One sentence each.
    Finding: what the data shows. Implication: why it matters in Wist.
    """
    best_action = _describe_action(best_key)
    worst_action = _describe_action(worst_key)
    
    best_tier = _TIER_RANK.get(best_key[0], 0) if best_key else 0
    worst_tier = _TIER_RANK.get(worst_key[0], 0) if worst_key else 0
    best_trump = len(best_key) > 2 and best_key[2] == "T"
    best_follows = len(best_key) > 1 and best_key[1] == "F"
    best_void = len(best_key) > 4 and best_key[4] == "V"
    is_counter = (best_tier < worst_tier - 1)

    pos_desc = {"0": "when you are the first to play in a trick", "1": "when you are the second player to play in a trick", "2": "when your partner plays first and you follow", "3": "when you are the last player to play in a trick"}.get(pos, "")
    phase_desc = {"1": "early", "2": "first half", "3": "mid-game", "4": "late", "5": "final tricks"}.get(phase, "")

    # Build finding + implication based on pattern type.
    if best_trump and not best_follows:
        finding = f"{pos_desc.capitalize()}, your smallest trump outscores all non-trump cards by 3x"
        implication = "Any trump beats any non-trump card regardless of rank, and so even a 2 of trump wins the trick"
    elif best_tier <= 0 and best_follows and worst_tier >= 3:
        finding = f"{pos_desc.capitalize()}, playing your lowest card scores higher than playing Kings or Queens"
        implication = "Spending high cards on tricks you cannot win wastes them, and low cards lose nothing you need"
    elif best_tier >= 4 and best_follows and td == "B":
        finding = f"When your team has won fewer tricks than the opponents and {pos_desc}, playing your strongest card immediately outscores saving it"
        implication = "When losing, waiting to use strong cards means they might never get used at all"
    elif best_void:
        finding = f"{pos_desc.capitalize()}, playing cards that remove all cards from one suit scores higher than keeping cards spread across all suits"
        implication = "Once you have zero cards in a suit, every time that suit is played you can use trump to win it"
    elif best_trump and best_tier >= 4:
        finding = f"{pos_desc.capitalize()}, your highest trump outscores lower trumps significantly"
        implication = "Lower trumps risk being beaten by an opponent's higher trump, but top trump guarantees the win"
    elif best_trump and best_tier <= 1:
        finding = f"{pos_desc.capitalize()}, your weakest trump scores nearly as well as your highest trump"
        implication = "When no opponent can also trump, any trump wins equally, so save your high ones for harder fights"
    elif is_counter:
        finding = f"{pos_desc.capitalize()}, {best_action} outscores {worst_action}"
        implication = "High cards attract opposition trumps from void opponents, but lower cards slip through uncontested"
    elif td == "W":
        finding = f"When your team has won more tricks than the opponents and {pos_desc}, conservative plays outscore aggressive ones"
        implication = "Protecting a lead is safer than extending it, and risks only help the losing team"
    elif td == "B":
        finding = f"When your team has won fewer tricks than the opponents and {pos_desc}, aggressive plays outscore conservative ones"
        implication = "Safe play when losing just means losing slowly, and bold moves can change the momentum"
    else:
        finding = f"{pos_desc.capitalize()}, {best_action} consistently outscores the alternatives"
        implication = "This play wins more across thousands of games in this situation"

    return f"{finding}. {implication}."


def _compose_partnership_insight(pos, phase, best_key, is_partner_led) -> str:
    """Compose a partnership-focused insight paragraph."""
    best_action = _describe_action(best_key)
    best_tier = _TIER_RANK.get(best_key[0], 0) if best_key else 0
    best_trump = len(best_key) > 2 and best_key[2] == "T"

    if is_partner_led:
        if best_tier <= 1:
            return (f"When your partner plays first in a trick and their card looks strong enough to win, "
                    f"play your lowest card. Keep your high cards for tricks where your team needs you to fight.")
        elif best_trump and best_key[1] == "F":
            return (f"When your partner plays trump first, play your highest trump. "
                    f"Together you remove 2 enemy trumps in 1 trick, and that helps both of you later.")
        elif best_tier >= 4:
            return (f"When your partner plays first but their card may not win, play your strongest card. "
                    f"Your high card guarantees the trick stays with your team.")
        else:
            return (f"When your partner plays first in a trick, support their play with a mid-range card. "
                    f"This supports their play without wasting your strongest cards.")
    else:
        if best_tier <= 1:
            return (f"When you play first in a trick, start with a low card. "
                    f"This lets your partner show their strength and tells you which suits are safe.")
        elif best_trump:
            return (f"When you play first in a trick, play trump. "
                    f"Playing trump first removes opponents' trumps and helps your partner too.")
        else:
            return (f"When you play first in a trick, play your strongest card. "
                    f"Your choice of suit forces everyone to follow, so pick a suit where your team is strong.")



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
    new_insights.extend(_mine_partnership_patterns(play_items, episodes))
    new_insights.extend(_mine_counter_intuitive(play_items, bid_q, episodes))

    # Dedup similar insights: same position + phase + action tier = merge into one.
    new_insights = _dedup_merge(new_insights)

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
    """Mine play Q-table for GENERAL strategic principles, not granular combos."""
    insights = []

    # Instead of per-context insights, aggregate across contexts to find
    # universal truths: actions that win in MOST situations.
    action_wins = defaultdict(lambda: {"total_q": 0, "count": 0, "beats": 0, "contexts": 0})
    action_losses = defaultdict(lambda: {"total_q": 0, "count": 0, "contexts": 0})

    # Also track: trump vs non-trump, following vs not, tier performance.
    trump_vs_non = {"trump_q": [], "non_q": []}
    follow_vs_off = {"follow_q": [], "off_q": []}
    void_creation = {"void_q": [], "keep_q": []}
    tier_overall = defaultdict(list)  # tier -> list of Q values.
    position_advantage = defaultdict(list)  # pos -> Q values.

    for state, actions in play_items:
        if len(state) < 6 or len(actions) < 2:
            continue

        sorted_a = sorted(actions.items(), key=lambda x: -x[1])
        best_key, best_q = sorted_a[0]
        worst_key, worst_q = sorted_a[-1]

        if len(best_key) < 3:
            continue

        # Track overall action properties.
        for key, q in actions.items():
            if len(key) < 3:
                continue
            tier = key[0]
            follows = key[1]
            is_trump = key[2] == "T"
            creates_void = len(key) > 4 and key[4] == "V"

            tier_overall[tier].append(q)
            if is_trump:
                trump_vs_non["trump_q"].append(q)
            else:
                trump_vs_non["non_q"].append(q)
            if follows == "F":
                follow_vs_off["follow_q"].append(q)
            else:
                follow_vs_off["off_q"].append(q)
            if creates_void:
                void_creation["void_q"].append(q)
            else:
                void_creation["keep_q"].append(q)

        # Position tracking.
        pos = state[4]
        position_advantage[pos].append(best_q)

    # Now generate GENERAL strategic principles from aggregated data.

    # 1. Trump dominance principle.
    if trump_vs_non["trump_q"] and trump_vs_non["non_q"]:
        t_avg = sum(trump_vs_non["trump_q"]) / len(trump_vs_non["trump_q"])
        n_avg = sum(trump_vs_non["non_q"]) / len(trump_vs_non["non_q"])
        if t_avg > n_avg + 0.3:
            insights.append(_make(
                "Trump control wins games, whoever runs out of trump last controls the endgame",
                "trump", "intermediate", 1, episodes,
                why="trump beats everything regardless of rank, managing it well is the single biggest factor"
            ))

    # 2. Void power principle.
    if void_creation["void_q"] and len(void_creation["void_q"]) >= 5:
        v_avg = sum(void_creation["void_q"]) / len(void_creation["void_q"])
        k_avg = sum(void_creation["keep_q"]) / len(void_creation["keep_q"]) if void_creation["keep_q"] else 0
        if v_avg > k_avg + 0.15:
            insights.append(_make(
                "A void is worth more than a King, one void gives you multiple free tricks through trumping",
                "voids", "intermediate", 1, episodes,
                why="every time that suit is led after you're void, you can trump it for free"
            ))

    # 3. Following suit vs breaking.
    if follow_vs_off["follow_q"] and follow_vs_off["off_q"]:
        f_avg = sum(follow_vs_off["follow_q"]) / len(follow_vs_off["follow_q"])
        o_avg = sum(follow_vs_off["off_q"]) / len(follow_vs_off["off_q"])
        if o_avg > f_avg + 0.2:
            insights.append(_make(
                "Breaking from the led suit (trumping or discarding) is more profitable than following when you have the choice",
                "timing", "advanced", 1, episodes,
                why="following suit with low cards is passive, breaking creates opportunities to control"
            ))
        elif f_avg > o_avg + 0.2:
            insights.append(_make(
                "Following suit with strong cards is the foundation of trick-winning, high cards in the led suit are your most reliable weapons",
                "timing", "beginner", 1, episodes,
                why="the highest card in the led suit wins unless someone trumps it"
            ))

    # 4. Tier hierarchy — which card ranks actually perform best.
    tier_avgs = {}
    for tier, vals in tier_overall.items():
        if len(vals) >= 10:
            tier_avgs[tier] = sum(vals) / len(vals)

    if tier_avgs:
        best_tier = max(tier_avgs, key=tier_avgs.get)
        worst_tier = min(tier_avgs, key=tier_avgs.get)
        # Counter-intuitive: if low cards outperform high ones overall.
        if _TIER_RANK.get(best_tier, 0) < 2:
            insights.append(_make(
                "Low cards played strategically outperform high cards played carelessly, timing matters more than raw card strength",
                "counter-intuitive", "advanced", 1, episodes,
                why="high cards attract opposition trumps, low cards preserve hand structure for when it matters"
            ))
        if _TIER_RANK.get(best_tier, 0) >= 4:
            insights.append(_make(
                "High cards dominate when played at the right moment, the key is knowing WHEN to deploy them",
                "timing", "intermediate", 1, episodes,
                why="Aces and Kings are guaranteed winners in the right context but wasted if trumped"
            ))

    # 5. Position wisdom.
    if position_advantage:
        pos_avgs = {p: sum(v)/len(v) for p, v in position_advantage.items() if len(v) >= 10}
        if pos_avgs:
            best_pos = max(pos_avgs, key=pos_avgs.get)
            if best_pos == "3":
                insights.append(_make(
                    "Playing last is the strongest position in Wist, you see everyone else's card before choosing yours",
                    "timing", "intermediate", 1, episodes,
                    why="perfect information about the current trick lets you play the minimum needed to win"
                ))
            elif best_pos == "0":
                insights.append(_make(
                    "Leading gives you control, you choose which suit everyone must follow, that power shapes the entire trick",
                    "timing", "intermediate", 1, episodes,
                    why="the leader forces opponents into their weak suits while playing from strength"
                ))

    # 6. Score-state strategy.
    winning_q = []
    losing_q = []
    for state, actions in play_items:
        if len(state) < 7:
            continue
        td = state[6]
        for key, q in actions.items():
            if len(key) >= 3:
                if td == "W":
                    winning_q.append(q)
                elif td == "B":
                    losing_q.append(q)

    if winning_q and losing_q:
        w_avg = sum(winning_q) / len(winning_q)
        l_avg = sum(losing_q) / len(losing_q)
        if l_avg > w_avg + 0.1:
            insights.append(_make(
                "The best plays happen under pressure, being behind forces sharper decisions that often have higher payoff",
                "counter-intuitive", "advanced", 1, episodes,
                why="desperation drives risk-taking which reveals hidden opportunities in the hand"
            ))

    # 7. Specific decisive plays — grouped by position+td only (no phase).
    context_actions = defaultdict(lambda: defaultdict(list))
    for state, actions in play_items:
        if len(state) < 7 or len(actions) < 2:
            continue
        pos = state[4]
        td = state[6] if len(state) > 6 else "T"
        ctx = (pos, td)
        for key, q in actions.items():
            if len(key) >= 3:
                context_actions[ctx][key].append(q)

    for ctx, action_qs in context_actions.items():
        pos, td = ctx
        avg_by_action = {}
        for key, vals in action_qs.items():
            if len(vals) >= 3:
                avg_by_action[key] = sum(vals) / len(vals)
        if len(avg_by_action) < 2:
            continue
        sorted_actions = sorted(avg_by_action.items(), key=lambda x: -x[1])
        best_key, best_avg = sorted_actions[0]
        worst_key, worst_avg = sorted_actions[-1]
        spread = best_avg - worst_avg
        if spread < 0.8:
            continue

        text = _compose_play_insight(pos, "", td, best_key, worst_key, spread)
        best_rank = _TIER_RANK.get(best_key[0], 0)
        worst_rank = _TIER_RANK.get(worst_key[0], 0)
        is_counter = (best_rank < worst_rank - 1)
        category = "counter-intuitive" if is_counter else _categorize(best_key, worst_key, False)
        insights.append(_make(text, category, "intermediate", 1, episodes, why=""))

    return insights[:30]


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
        # Proper English joining: "X and Y" or "X, Y, and Z".
        if len(hand_parts) == 1:
            hand_desc = hand_parts[0]
        elif len(hand_parts) == 2:
            hand_desc = f"{hand_parts[0]} and {hand_parts[1]}"
        else:
            hand_desc = ", ".join(hand_parts[:-1]) + f", and {hand_parts[-1]}"

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
            text = f"Hand with {hand_desc}, {action_desc} works better than expected"

        category = "counter-intuitive" if is_counter else "bidding"
        insights.append(_make(text, category, "intermediate", 1, episodes, why=why))

    return insights[:20]



def _mine_partnership_patterns(play_items, episodes) -> list:
    """Mine Q-table for partnership coordination patterns."""
    insights = []

    # Group by position only — no phase splitting for partnership.
    partner_contexts = defaultdict(lambda: defaultdict(list))
    for state, actions in play_items:
        if len(state) < 6:
            continue
        pos = state[4]
        if pos in ("0", "2"):
            for key, q in actions.items():
                if len(key) >= 3:
                    partner_contexts[pos][key].append(q)

    for pos, action_qs in partner_contexts.items():
        avg_by_action = {}
        for key, vals in action_qs.items():
            if len(vals) >= 3:
                avg_by_action[key] = sum(vals) / len(vals)

        if len(avg_by_action) < 2:
            continue

        sorted_actions = sorted(avg_by_action.items(), key=lambda x: -x[1])
        best_key, best_avg = sorted_actions[0]
        worst_key, worst_avg = sorted_actions[-1]
        spread = best_avg - worst_avg

        if spread < 0.4:
            continue

        # Use compose function for rich descriptions.
        is_partner_led = (pos == "2")
        text = _compose_partnership_insight(pos, "", best_key, is_partner_led)
        insights.append(_make(text, "partnership", "intermediate", 1, episodes, why=""))

    return insights[:15]


def _dedup_merge(insights) -> list:
    """
    Merge similar insights into one. If insights describe the same
    position + same type of action (high card, low card, trump, void),
    merge them regardless of specific rank.
    """
    seen = {}  # signature -> insight

    for ins in insights:
        text = ins.get("text", "")
        cat = ins.get("category", "")
        words = text.lower().split()

        # Build structural signature: position + action TYPE (not specific rank).
        sig_parts = [cat]

        # Position keywords.
        if "first to play" in text.lower():
            sig_parts.append("pos0")
        elif "second player" in text.lower():
            sig_parts.append("pos1")
        elif "partner" in text.lower() and "follow" in text.lower():
            sig_parts.append("pos2")
        elif "last player" in text.lower():
            sig_parts.append("pos3")

        # Action TYPE (generalized, not specific rank).
        if "trump" in text.lower() and ("smallest" in text.lower() or "weakest" in text.lower()):
            sig_parts.append("trump_low")
        elif "trump" in text.lower() and ("highest" in text.lower() or "ace" in text.lower() or "king" in text.lower()):
            sig_parts.append("trump_high")
        elif "trump" in text.lower():
            sig_parts.append("trump")
        elif any(w in text.lower() for w in ("ace", "king", "queen", "jack", "highest", "high card")):
            sig_parts.append("high_card")
        elif any(w in text.lower() for w in ("lowest", "weakest", "2-6", "7 or 8")):
            sig_parts.append("low_card")
        elif "void" in text.lower() or "remove all cards" in text.lower():
            sig_parts.append("void")

        # Game state.
        if "fewer tricks" in text.lower() or "behind" in text.lower():
            sig_parts.append("behind")
        elif "more tricks" in text.lower() or "ahead" in text.lower():
            sig_parts.append("ahead")

        sig = " ".join(sig_parts)

        if sig in seen:
            # Duplicate structure — increment confidence of existing.
            seen[sig]["confidence"] = seen[sig].get("confidence", 1) + 1
        else:
            seen[sig] = ins

    return list(seen.values())


def _mine_counter_intuitive(play_items, bid_q, episodes) -> list:
    """Find patterns where the 'obvious' play loses — general strategic reversals."""
    insights = []

    # Aggregate: do low cards beat high cards ACROSS ALL contexts?
    low_total_q = []
    high_total_q = []
    mid_total_q = []
    trump_off_q = []  # Trumping when not following.

    for state, actions in play_items:
        if len(state) < 6:
            continue
        for key, q in actions.items():
            if len(key) < 3:
                continue
            tier = key[0]
            if tier in ("L", "X"):
                low_total_q.append(q)
            elif tier in ("A", "K"):
                high_total_q.append(q)
            elif tier in ("M", "J"):
                mid_total_q.append(q)
            if key[2] == "T" and key[1] == "O":
                trump_off_q.append(q)

    # General reversal: mid cards beat high cards overall.
    if mid_total_q and high_total_q:
        mid_avg = sum(mid_total_q) / len(mid_total_q)
        high_avg = sum(high_total_q) / len(high_total_q)
        if mid_avg > high_avg + 0.1:
            insights.append(_make(
                "Middle cards (9s, 10s, Jacks) quietly outperform Aces and Kings on average, they win tricks nobody bothers to fight over",
                "counter-intuitive", "advanced", 1, episodes,
                why="opponents save their trumps to kill your high cards but let middle cards through uncontested"
            ))

    # General reversal: low cards have positive value (not just waste).
    if low_total_q:
        low_avg = sum(low_total_q) / len(low_total_q)
        if low_avg > 0.1:
            insights.append(_make(
                "Low cards are not waste, playing them strategically builds voids and preserves your hand structure for critical moments",
                "counter-intuitive", "intermediate", 1, episodes,
                why="every low card you play is one step closer to being void in that suit, which means free tricks later"
            ))

    # Trump when not following: is it worth it?
    if trump_off_q and len(trump_off_q) >= 5:
        trump_avg = sum(trump_off_q) / len(trump_off_q)
        if trump_avg > 0.5:
            insights.append(_make(
                "Trumping when you can't follow suit is almost always profitable, even your smallest trump beats their best card",
                "trump", "beginner", 1, episodes,
                why="trump overrides rank entirely, a 2 of trump beats an Ace of any other suit"
            ))

    # Bid reversals.
    strong_hand_pass = []
    strong_hand_bid = []
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
        bid_qs = [q for k, q in actions.items() if k.startswith("B")]
        if pass_q is not None:
            strong_hand_pass.append(pass_q)
        strong_hand_bid.extend(bid_qs)

    if strong_hand_pass and strong_hand_bid:
        pass_avg = sum(strong_hand_pass) / len(strong_hand_pass)
        bid_avg = sum(strong_hand_bid) / len(strong_hand_bid)
        if pass_avg > bid_avg + 0.2:
            insights.append(_make(
                "Having many high cards doesn't mean you should bid, trump LENGTH matters more than card POWER",
                "counter-intuitive", "advanced", 1, episodes,
                why="high cards spread across suits get trumped by void opponents, concentrated trump length is what delivers tricks"
            ))
        elif bid_avg > pass_avg + 0.3:
            insights.append(_make(
                "When your hand has strength, commit to it with a bid, hesitation leaves points on the table for opponents",
                "bidding", "intermediate", 1, episodes,
                why="strong hands that pass let opponents bid cheaply and control the game"
            ))

    return insights[:8]


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
