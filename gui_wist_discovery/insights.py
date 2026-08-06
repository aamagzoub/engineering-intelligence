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
    Generates progressively more insights as the agent learns more states.
    """
    if agent.episodes_trained < 5000:
        return ["Still learning basics..."]

    insights = []
    play_q = agent.play_q
    bid_q = agent.bid_q

    # Scale analysis depth with training progress.
    q_size = len(play_q)
    sample_size = min(q_size, 10000)
    play_items = list(play_q.items())[:sample_size]

    insights.extend(_action_insights(play_items))
    insights.extend(_phase_and_position_insights(play_items))
    insights.extend(_score_diff_insights(play_items))
    insights.extend(_trump_insights(play_items))
    insights.extend(_bid_insights(bid_q))
    insights.extend(_void_insights(play_items))
    insights.extend(_combination_insights(play_items))
    insights.extend(_evolution_insights(play_items))

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

    # Phase insights — top actions per phase.
    phase_labels = {"1": "opening", "2": "early-mid", "3": "mid-game", "4": "late-mid", "5": "endgame"}
    for phase, label in phase_labels.items():
        actions_dict = phase_action_q.get(phase, {})
        top_actions = _top_n_actions(actions_dict, n=3)
        for rank_idx, (action, avg) in enumerate(top_actions):
            tier = TIER_NAMES_PLURAL.get(action[0], "cards")
            trump_flag = "trump " if len(action) > 2 and action[2] == "T" else ""
            follows = "following" if len(action) > 1 and action[1] == "F" else "off-suit"
            if rank_idx == 0:
                insights.append(f"PHASE ({label}): Best play is {trump_flag}{tier} ({follows}) — Q={avg:.2f}")
            else:
                insights.append(f"PHASE ({label}): Also strong: {trump_flag}{tier} ({follows}) — Q={avg:.2f}")

    # Worst actions per phase (things to avoid).
    for phase, label in phase_labels.items():
        actions_dict = phase_action_q.get(phase, {})
        worst_actions = _worst_n_actions(actions_dict, n=2)
        for action, avg in worst_actions:
            tier = TIER_NAMES_PLURAL.get(action[0], "cards")
            trump_flag = "trump " if len(action) > 2 and action[2] == "T" else ""
            follows = "following" if len(action) > 1 and action[1] == "F" else "off-suit"
            insights.append(f"AVOID ({label}): Don't play {trump_flag}{tier} ({follows}) — Q={avg:.2f}")

    # Position insights — all 4 positions.
    pos_labels = {"0": "leading (1st)", "1": "2nd to play", "2": "3rd to play", "3": "last (4th)"}
    for pos, label in pos_labels.items():
        top_actions = _top_n_actions(pos_action_q.get(pos, {}), n=2)
        for action, avg in top_actions:
            tier = TIER_NAMES_SINGULAR.get(action[0], "card")
            trump_flag = "trump " if len(action) > 2 and action[2] == "T" else ""
            insights.append(f"POSITION ({label}): Play {trump_flag}{tier} — Q={avg:.2f}")

    return insights


def _score_diff_insights(play_items) -> list:
    """Insights based on whether the agent is ahead or behind."""
    insights = []
    ahead_actions = defaultdict(list)
    behind_actions = defaultdict(list)
    tied_actions = defaultdict(list)

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
                elif score_flag == "A":
                    ahead_actions[action_key].append(q_val)
                elif score_flag == "T":
                    tied_actions[action_key].append(q_val)

    # Top actions when ahead.
    top_ahead = _top_n_from_action_dict(ahead_actions, n=3)
    for a, avg in top_ahead:
        tier = TIER_NAMES_PLURAL.get(a[0], "cards")
        trump = "trump " if len(a) > 2 and a[2] == "T" else ""
        insights.append(f"WHEN AHEAD: Playing {trump}{tier} maintains your lead — Q={avg:.2f}")

    # Top actions when behind.
    top_behind = _top_n_from_action_dict(behind_actions, n=3)
    for a, avg in top_behind:
        tier = TIER_NAMES_PLURAL.get(a[0], "cards")
        trump = "trump " if len(a) > 2 and a[2] == "T" else ""
        insights.append(f"WHEN BEHIND: Playing {trump}{tier} is the best way to catch up — Q={avg:.2f}")

    # Top actions when tied.
    top_tied = _top_n_from_action_dict(tied_actions, n=2)
    for a, avg in top_tied:
        tier = TIER_NAMES_PLURAL.get(a[0], "cards")
        trump = "trump " if len(a) > 2 and a[2] == "T" else ""
        insights.append(f"WHEN TIED: Playing {trump}{tier} tips the balance — Q={avg:.2f}")

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
    """Insights from performance statistics — placeholder for future use."""
    return []


def _combination_insights(play_items) -> list:
    """
    Cross-dimensional insights — combine phase × position × trump to find
    specific situational strategies.
    """
    insights = []

    # Group by (phase, position, trump_context).
    situation_q = defaultdict(lambda: defaultdict(list))

    for state, actions in play_items:
        if len(state) < 10:
            continue
        pos = state[4] if len(state) > 4 else "0"
        phase = state[5] if len(state) > 5 else "1"
        trump_str = state[7] if len(state) > 7 else "0"

        # Build situation key.
        phase_name = {"1": "early", "2": "early", "3": "mid", "4": "late", "5": "final"}.get(phase, "mid")
        pos_name = {"0": "lead", "1": "2nd", "2": "3rd", "3": "4th"}.get(pos, "?")
        trump_heavy = "many-trump" if trump_str in ("4", "5", "6", "7") else "few-trump"

        sit_key = f"{phase_name}/{pos_name}/{trump_heavy}"

        for action_key, q_val in actions.items():
            if abs(q_val) > 0.3:
                situation_q[sit_key][action_key].append(q_val)

    # Find the most decisive situations (high Q variance = strong preference).
    for sit_key, actions in situation_q.items():
        top = _top_n_actions(actions, n=1)
        if not top:
            continue
        action, avg = top[0]
        if avg < 0.5:
            continue
        tier = TIER_NAMES_PLURAL.get(action[0], "cards")
        trump_flag = "trump " if len(action) > 2 and action[2] == "T" else ""
        follows = "suit" if len(action) > 1 and action[1] == "F" else "off-suit"
        insights.append(f"SITUATION ({sit_key}): Play {trump_flag}{tier} ({follows}) — Q={avg:.2f}")

    # Cap to avoid overwhelming.
    return insights[:20]


def _evolution_insights(play_items) -> list:
    """
    Contrast insights — find where the agent's strategy differs sharply
    between contexts (e.g., same card but very different Q in different positions).
    """
    insights = []

    # Find actions that are great in one position but bad in another.
    pos_action_avgs = defaultdict(lambda: defaultdict(list))
    for state, actions in play_items:
        if len(state) < 6:
            continue
        pos = state[4]
        for action_key, q_val in actions.items():
            pos_action_avgs[action_key][pos].append(q_val)

    for action_key, pos_dict in pos_action_avgs.items():
        pos_avgs = {}
        for pos, vals in pos_dict.items():
            if len(vals) >= 10:
                pos_avgs[pos] = sum(vals) / len(vals)
        if len(pos_avgs) < 2:
            continue

        max_pos = max(pos_avgs, key=pos_avgs.get)
        min_pos = min(pos_avgs, key=pos_avgs.get)
        spread = pos_avgs[max_pos] - pos_avgs[min_pos]

        if spread > 0.5:
            tier = TIER_NAMES_PLURAL.get(action_key[0], "cards")
            trump_flag = "trump " if len(action_key) > 2 and action_key[2] == "T" else ""
            pos_names = {"0": "leading", "1": "2nd", "2": "3rd", "3": "last"}
            good_pos = pos_names.get(max_pos, max_pos)
            bad_pos = pos_names.get(min_pos, min_pos)
            insights.append(
                f"CONTEXT: {trump_flag}{tier} works well when {good_pos} "
                f"(Q={pos_avgs[max_pos]:.2f}) but poorly when {bad_pos} "
                f"(Q={pos_avgs[min_pos]:.2f})"
            )

    return insights[:15]


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


def _top_n_actions(actions_dict, n=3):
    """Return top N actions by average Q-value (min 5 samples each)."""
    averages = []
    for a, vals in actions_dict.items():
        if len(vals) >= 5:
            avg = sum(vals) / len(vals)
            if avg > 0.2:
                averages.append((a, avg))
    averages.sort(key=lambda x: -x[1])
    return averages[:n]


def _worst_n_actions(actions_dict, n=2):
    """Return N worst actions by average Q-value (min 5 samples, negative)."""
    averages = []
    for a, vals in actions_dict.items():
        if len(vals) >= 5:
            avg = sum(vals) / len(vals)
            if avg < -0.2:
                averages.append((a, avg))
    averages.sort(key=lambda x: x[1])
    return averages[:n]


def _top_n_from_action_dict(action_dict, n=3):
    """Like _top_n_actions but from a dict of action_key → [q_values]."""
    averages = []
    for a, vals in action_dict.items():
        if len(vals) >= 10:
            avg = sum(vals) / len(vals)
            if avg > 0.3:
                averages.append((a, avg))
    averages.sort(key=lambda x: -x[1])
    return averages[:n]


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
