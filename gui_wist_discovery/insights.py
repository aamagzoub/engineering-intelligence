"""
Strategic insights — analyzes Q-tables to generate human-readable strategy descriptions.
"""

from collections import defaultdict, Counter


# ─── Action Encoding → Human Description ────────────────────────────────────────

TIER_NAMES_PLURAL = {
    "A": "Aces", "K": "Kings", "Q": "Queens", "J": "Jacks",
    "M": "mid cards (9-10)", "L": "low cards (5-8)", "X": "very low cards (2-4)",
}
TIER_NAMES_SINGULAR = {
    "A": "Ace", "K": "King", "Q": "Queen", "J": "Jack",
    "M": "9 or 10", "L": "low card", "X": "very low card",
}


def generate_insights(agent) -> list:
    """
    Analyze Q-tables and generate human-readable strategic insights.

    Returns a list of insight strings with category prefixes.
    """
    if agent.episodes_trained < 5000:
        return ["Still learning basics..."]

    insights = []
    play_q = agent.play_q
    bid_q = agent.bid_q

    # Limit iteration for performance.
    play_items = list(play_q.items())[:5000]

    insights.extend(_action_insights(play_items[:3000]))
    insights.extend(_phase_and_position_insights(play_items))
    insights.extend(_score_diff_insights(play_items[:3000]))
    insights.extend(_trump_insights(play_items[:3000]))
    insights.extend(_bid_insights(bid_q))
    insights.extend(_performance_insights(agent))
    insights.extend(_void_insights(play_items[:2000]))

    # Deduplicate.
    seen = set()
    return [i for i in insights if not (i in seen or seen.add(i))]


def _action_insights(play_items) -> list:
    """Insights from overall action Q-values."""
    insights = []
    action_avg = defaultdict(list)

    for _state, actions in play_items:
        for action_key, q_val in actions.items():
            if abs(q_val) > 0.1:
                action_avg[action_key].append(q_val)

    for action_key, values in action_avg.items():
        if len(values) < 10:
            continue
        avg = sum(values) / len(values)
        if abs(avg) < 0.2:
            continue
        cat, desc = categorize_insight(action_key, avg)
        if cat and desc:
            insights.append(f"{cat}: {desc}")

    return insights


def _phase_and_position_insights(play_items) -> list:
    """Insights based on game phase (early/mid/late) and trick position."""
    insights = []
    phase_action_q = defaultdict(lambda: defaultdict(list))
    pos_action_q = defaultdict(lambda: defaultdict(list))

    for state, actions in play_items:
        if len(state) < 6:
            continue
        pos = state[4] if len(state) > 4 else "0"
        phase = state[5] if len(state) > 5 else "1"

        for action_key, q_val in actions.items():
            if abs(q_val) > 0.2:
                phase_action_q[phase][action_key].append(q_val)
                pos_action_q[pos][action_key].append(q_val)

    # Phase insights.
    phase_labels = {"1": "opening", "3": "mid-game", "5": "endgame"}
    for phase, label in phase_labels.items():
        best_action, best_avg = _best_action(phase_action_q.get(phase, {}))
        if best_action and best_avg > 0.3:
            tier = TIER_NAMES_PLURAL.get(best_action[0], "cards")
            trump_flag = "trump " if len(best_action) > 2 and best_action[2] == "T" else ""
            templates = {
                "opening": f"OPENING: In the first few tricks, playing {trump_flag}{tier} tends to set up a strong position",
                "mid-game": f"TIMING: In the middle of the shota, {trump_flag}{tier} are the most effective plays",
                "endgame": f"ENDGAME: In the final tricks, {trump_flag}{tier} dominate — save them for this moment",
            }
            insights.append(templates[label])

    # Position insights.
    pos_labels = {"0": "leading", "3": "last to play"}
    for pos, label in pos_labels.items():
        best_action, best_avg = _best_action(pos_action_q.get(pos, {}))
        if best_action and best_avg > 0.3:
            tier = TIER_NAMES_SINGULAR.get(best_action[0], "card")
            if label == "leading":
                insights.append(f"LEAD: When you lead the trick, starting with a {tier} works best")
            else:
                insights.append(f"POSITION: When you play last (4th), you see everything — play the minimum {tier} needed to win")

    return insights


def _score_diff_insights(play_items) -> list:
    """Insights based on whether the agent is ahead or behind."""
    insights = []
    ahead_actions = defaultdict(list)
    behind_actions = defaultdict(list)

    for state, actions in play_items:
        if len(state) < 7:
            continue
        score_flag = state[6] if len(state) > 6 else "T"
        for action_key, q_val in actions.items():
            if abs(q_val) > 0.2:
                if score_flag == "W":
                    ahead_actions[action_key].append(q_val)
                elif score_flag == "B":
                    behind_actions[action_key].append(q_val)

    for a, vals in ahead_actions.items():
        if len(vals) >= 10:
            avg = sum(vals) / len(vals)
            if avg > 0.4:
                tier = TIER_NAMES_PLURAL.get(a[0], "cards")
                trump = "trump " if len(a) > 2 and a[2] == "T" else ""
                insights.append(f"CONTROL: When ahead in score, playing {trump}{tier} maintains your lead safely")
                break

    for a, vals in behind_actions.items():
        if len(vals) >= 10:
            avg = sum(vals) / len(vals)
            if avg > 0.4:
                tier = TIER_NAMES_PLURAL.get(a[0], "cards")
                trump = "trump " if len(a) > 2 and a[2] == "T" else ""
                insights.append(f"RECOVER: When behind in score, playing {trump}{tier} is the best way to catch up")
                break

    return insights


def _trump_insights(play_items) -> list:
    """Insights based on trump count in hand."""
    insights = []
    trump_heavy = []
    trump_light = []

    for state, actions in play_items:
        if len(state) < 10:
            continue
        try:
            trump_count = int(state[8]) if state[8].isdigit() else 0
        except (IndexError, ValueError):
            continue

        for a, q in actions.items():
            if q > 0.3:
                if trump_count >= 4:
                    trump_heavy.append(a)
                elif trump_count <= 1:
                    trump_light.append(a)

    if trump_heavy:
        most_common = Counter(trump_heavy).most_common(1)
        if most_common:
            a = most_common[0][0]
            action_desc = "trumping" if len(a) > 2 and a[2] == "T" else "playing aggressively"
            insights.append(f"TRUMP: When you hold many trumps (4+), {action_desc} dominates — you have the power to control every trick")

    if trump_light:
        most_common = Counter(trump_light).most_common(1)
        if most_common:
            a = most_common[0][0]
            tier = TIER_NAMES_PLURAL.get(a[0], "cards")
            insights.append(f"ADAPT: When you have few trumps (0-1), rely on {tier} in side suits — you can't afford to waste what little trump you have")

    return insights


def _bid_insights(bid_q) -> list:
    """Insights from bidding Q-values."""
    insights = []
    bid_pass_values = []
    bid_values_by_num = defaultdict(list)

    for _state, actions in list(bid_q.items())[:500]:
        for action_key, q_val in actions.items():
            if action_key == "PASS":
                bid_pass_values.append(q_val)
            elif action_key.startswith("B"):
                bid_values_by_num[action_key].append(q_val)

    if bid_pass_values and sum(bid_pass_values) / len(bid_pass_values) > 0.3:
        insights.append("BID: When your hand is weak, passing is smarter than overbidding — let the opponents take the risk")

    for action_key, values in sorted(bid_values_by_num.items(), key=lambda x: -sum(x[1]) / len(x[1]) if x[1] else 0):
        if len(values) < 3:
            continue
        avg = sum(values) / len(values)
        if avg <= 0.3:
            continue
        val = int(action_key[1:])
        if val == 7:
            insights.append("UNDERBID: Bidding 7 (the minimum) is the safest bet — easy to meet and hard to fail")
        elif val <= 8:
            insights.append(f"BID: Bidding {val} is a solid conservative choice — promise less, deliver more")
        elif val >= 10:
            insights.append(f"RISK: Bidding {val} requires a powerful hand — only do this with many trumps and high cards")
        break

    return insights


def _performance_insights(agent) -> list:
    """Insights from performance statistics."""
    insights = []
    # These would come from auto_stats if available; keep minimal here.
    return insights


def _void_insights(play_items) -> list:
    """Insights about exploiting opponent voids."""
    insights = []
    void_exploit_count = 0

    for state, actions in play_items:
        if "o" not in state:
            continue
        o_idx = state.index("o")
        if o_idx + 1 >= len(state) or not state[o_idx + 1].isdigit():
            continue
        opp_voids = int(state[o_idx + 1])
        if opp_voids >= 2:
            for a, q in actions.items():
                if q > 0.3 and len(a) > 2 and a[2] == "T":
                    void_exploit_count += 1

    if void_exploit_count >= 5:
        insights.append("EXPLOIT: When opponents are void in suits, they will trump your leads — switch to leading trump to neutralize their advantage")

    return insights


# ─── Helpers ────────────────────────────────────────────────────────────────────


def _best_action(actions_dict):
    """Find the action with highest average Q-value (min 5 samples)."""
    best_action = ""
    best_avg = 0
    for a, vals in actions_dict.items():
        if len(vals) >= 5:
            avg = sum(vals) / len(vals)
            if avg > best_avg:
                best_avg = avg
                best_action = a
    return best_action, best_avg


def categorize_insight(action_key: str, avg_q: float) -> tuple:
    """Map an action encoding to a (category, description) pair."""
    if len(action_key) < 4:
        return ("", "")

    tier = action_key[0]
    follows = action_key[1] if len(action_key) > 1 else ""
    is_trump = action_key[2] if len(action_key) > 2 else ""
    is_long = action_key[3] if len(action_key) > 3 else ""
    creates_void = action_key[4] if len(action_key) > 4 else ""

    positive = avg_q > 0

    if positive:
        return _positive_insight(tier, follows, is_trump, is_long, creates_void)
    return _negative_insight(tier, follows, is_trump, is_long)


def _positive_insight(tier, follows, is_trump, is_long, creates_void):
    """Generate insight for a positively-valued action."""
    # Whip patterns (trumping when void).
    if is_trump == "T" and follows == "O":
        whip_msgs = {
            "A": "When void in the led suit, trump with your Ace — it's unbeatable and guarantees the trick",
            "K": "When void, trump with King — only Ace can beat it, and it might already be played",
            "Q": "When void, Queen of trump is a strong whip — beats everything except Ace and King",
            "J": "When void, Jack of trump is a decent whip — wins unless opponents have higher trump",
        }
        if tier in whip_msgs:
            return ("WHIP", whip_msgs[tier])
        return ("WHIP", "When void in the led suit, play a small trump — you win cheaply and save your big trumps for later")

    # Void creation.
    if creates_void == "V":
        if is_trump == "N":
            return ("VOID", "Play your last card of a suit to create a void — next time that suit is led, you can trump it")
        return ("VOID", "Getting rid of your last card in a suit opens up future trumping opportunities")

    # Following suit.
    if follows == "F":
        return _following_suit_insight(tier, is_trump, is_long)

    # Off-suit dumping.
    if follows == "O" and is_trump == "N":
        dump_msgs = {
            "X": "Throw your 2, 3, or 4 when void — these cards will never win anything, get rid of them first",
            "L": "When void in the led suit and not trumping, dump your lowest non-trump — minimize loss",
            "K": "Playing King off-suit is a sacrifice — but it might help your partner win with a trump",
            "Q": "Queen off-suit is expendable when you can't follow — better than wasting your Ace",
        }
        if tier in dump_msgs:
            label = "DUMP" if tier in ("X", "L") else "SACRIFICE"
            return (label, dump_msgs[tier])

    return ("", "")


def _following_suit_insight(tier, is_trump, is_long):
    """Generate insight for following-suit actions."""
    if tier == "A":
        if is_trump == "T":
            return ("FLUSH", "Leading Ace of trump forces everyone to follow with their trumps — you thin out their trump supply")
        return ("CONTROL", "Play your Ace when following suit — it wins guaranteed and you take the lead for the next trick")

    if tier == "K":
        if is_trump == "T":
            return ("TRUMP", "King of trump following suit is extremely strong — only Ace beats it")
        return ("PRESSURE", "Play King when following — it wins unless the Ace is still out there")

    if tier == "Q":
        if is_trump == "T":
            return ("TRUMP", "Queen of trump is a solid follow — saves your King and Ace for later")
        if is_long == "L":
            return ("BLEED", "Queen from your long suit forces out opponents' higher cards — setting up your remaining cards to win")
        return ("BLOCK", "Queen following suit is competitive — it beats everything below it")

    if tier == "J":
        if is_trump == "T":
            return ("SAVING", "Jack of trump following suit is conservative — keep your royals for more critical tricks")
        return ("PROBE", "Jack following suit tests if opponents still have Queen, King, or Ace in that suit")

    if tier == "M":
        if is_trump == "T":
            return ("TIMING", "Mid-range trump (9 or 10) following suit is safe — might win without spending royals")
        return ("PROBE", "Playing 9 or 10 when following suit tests the waters safely")

    if tier == "L":
        if is_trump == "T":
            return ("SAVING", "Play low trump (5-8) when following trump — preserve your high trumps for when they really matter")
        return ("DUCK", "When you can't beat what's on the table, play a mid-low card — save your face cards for tricks you can win")

    if tier == "X":
        return ("DUCK", "Playing your smallest card (2-4) when following is the safest possible move — zero waste, pure preservation")

    return ("", "")


def _negative_insight(tier, follows, is_trump, is_long):
    """Generate insight for a negatively-valued action."""
    if follows == "O" and is_trump == "N":
        waste_msgs = {
            "A": "Don't play your Ace off-suit when you can't follow — it's your strongest card wasted on a trick you can't win",
            "K": "Playing King off-suit when void is wasteful — it could win a trick in its own suit later",
            "Q": "Queen off-suit when void is a wasted card — play something lower instead",
            "J": "Jack off-suit when void is marginal — it won't win and it's not worthless enough to dump freely",
            "M": "Mid-range cards (9-10) off-suit rarely help — they can't win the trick and they're not truly expendable",
        }
        if tier in waste_msgs:
            label = "WASTE" if tier in ("A", "K", "Q") else "RISK"
            return (label, waste_msgs[tier])

    if follows == "F" and is_trump == "N":
        if tier == "Q":
            return ("TRAP", "Playing Queen when following can be a trap — if King or Ace is behind you, your Queen is wasted and you lose a strong card")
        if tier == "K":
            return ("TRAP", "Be careful playing King when following — if the Ace hasn't appeared yet, you might lose your King for nothing")
        if tier == "A" and is_long == "S":
            return ("TIMING", "Playing Ace from a short suit early can backfire — you lose control of that suit permanently")

    if tier == "X" and follows == "F" and is_trump == "T":
        return ("RISK", "Playing your lowest trump (2-4) when following trump barely contributes — opponents' mid trumps beat it easily")

    return ("", "")
