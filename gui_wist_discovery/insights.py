"""
Strategic insights — analyzes Q-tables to extract discovered game strategies.

Generates plain-language strategy descriptions. No statistics, no Q-values.
Each insight is a lesson the agent learned from experience.
"""

from collections import defaultdict, Counter


# ─── Action Encoding Helpers ────────────────────────────────────────────────────

TIER_NAMES = {
    "A": "Aces", "K": "Kings", "Q": "Queens", "J": "Jacks",
    "M": "mid cards (9-10)", "L": "low cards (5-8)", "X": "small cards (2-4)",
}


def generate_insights(agent) -> list:
    """
    Analyze Q-tables and extract real strategic lessons the agent discovered.

    Returns plain-language insights — no stats, no noise, just strategy.
    """
    if agent.episodes_trained < 5000:
        return ["Still learning basics..."]

    insights = []
    play_q = agent.play_q
    bid_q = agent.bid_q

    q_size = len(play_q)
    play_items = list(play_q.items())[:min(q_size, 10000)]

    # Core strategy discoveries.
    insights.extend(_what_works(play_items))
    insights.extend(_what_fails(play_items))
    insights.extend(_positional_play(play_items))
    insights.extend(_trump_management(play_items))
    insights.extend(_timing_lessons(play_items))
    insights.extend(_bidding_wisdom(bid_q))
    insights.extend(_void_play(play_items))
    insights.extend(_situational_contrasts(play_items))

    # Deduplicate.
    seen = set()
    return [i for i in insights if not (i in seen or seen.add(i))]


# ─── Strategy Extractors ────────────────────────────────────────────────────────


def _what_works(play_items) -> list:
    """Discover what actions consistently lead to positive outcomes."""
    insights = []
    action_values = defaultdict(list)

    for _state, actions in play_items:
        for key, q in actions.items():
            if q > 0.3:
                action_values[key].append(q)

    # Find the strongest positive patterns (min 20 samples for reliability).
    ranked = sorted(
        ((k, sum(v) / len(v), len(v)) for k, v in action_values.items() if len(v) >= 20),
        key=lambda x: -x[1]
    )

    for key, avg, count in ranked[:8]:
        desc = _describe_action(key, positive=True)
        if desc:
            insights.append(desc)

    return insights


def _what_fails(play_items) -> list:
    """Discover what actions consistently lead to losses."""
    insights = []
    action_values = defaultdict(list)

    for _state, actions in play_items:
        for key, q in actions.items():
            if q < -0.3:
                action_values[key].append(q)

    ranked = sorted(
        ((k, sum(v) / len(v), len(v)) for k, v in action_values.items() if len(v) >= 20),
        key=lambda x: x[1]
    )

    for key, avg, count in ranked[:5]:
        desc = _describe_action(key, positive=False)
        if desc:
            insights.append(desc)

    return insights


def _positional_play(play_items) -> list:
    """Discover how position in trick affects best play."""
    insights = []
    pos_best = defaultdict(lambda: defaultdict(list))

    for state, actions in play_items:
        if len(state) < 5:
            continue
        pos = state[4]
        for key, q in actions.items():
            if q > 0.3:
                pos_best[pos][key].append(q)

    pos_names = {"0": "leading", "1": "second", "2": "third", "3": "last"}

    for pos, pos_name in pos_names.items():
        actions = pos_best.get(pos, {})
        best = _top_action(actions)
        if best:
            key, _ = best
            tier = TIER_NAMES.get(key[0], "cards")
            is_trump = len(key) > 2 and key[2] == "T"
            if pos == "0":
                if is_trump:
                    insights.append(f"When leading, start with trump {tier} to draw out opponents' trump cards")
                else:
                    insights.append(f"When leading, open with {tier} to set the pace and test opponents")
            elif pos == "3":
                if is_trump:
                    insights.append(f"When playing last, trump {tier} let you steal tricks cheaply since you see everything")
                else:
                    insights.append(f"When playing last, the minimum winning card is best — don't overspend")

    return insights


def _trump_management(play_items) -> list:
    """Discover trump-related strategies."""
    insights = []
    trump_heavy_best = defaultdict(list)
    trump_light_best = defaultdict(list)
    whip_values = []
    flush_values = []

    for state, actions in play_items:
        if len(state) < 10:
            continue
        try:
            trump_count = int(state[7]) if state[7].isdigit() else 0
        except (IndexError, ValueError):
            continue

        for key, q in actions.items():
            if len(key) < 3:
                continue
            is_trump = key[2] == "T"
            follows = key[1] if len(key) > 1 else ""

            if q > 0.3:
                if trump_count >= 4:
                    trump_heavy_best[key].append(q)
                elif trump_count <= 1:
                    trump_light_best[key].append(q)

            # Whipping (trump off-suit).
            if is_trump and follows == "O" and q > 0.3:
                whip_values.append(q)

            # Flushing (leading trump).
            if is_trump and follows == "F" and key[0] in ("A", "K") and q > 0.3:
                flush_values.append(q)

    if len(whip_values) >= 10:
        insights.append("Whipping is powerful — when void in the led suit, trumping steals tricks opponents expected to win")

    if len(flush_values) >= 10:
        insights.append("Leading high trump flushes out opponents' trump cards, leaving yours dominant for later tricks")

    # Heavy trump hand.
    if trump_heavy_best:
        best = _top_action(trump_heavy_best)
        if best:
            key, _ = best
            if key[2] == "T":
                insights.append("With many trumps (4+), play them aggressively — you control the game through sheer trump power")
            else:
                insights.append("With many trumps, even side-suit plays are safe — you can always trump back in later")

    # Light trump hand.
    if trump_light_best:
        best = _top_action(trump_light_best)
        if best:
            key, _ = best
            tier = TIER_NAMES.get(key[0], "cards")
            insights.append(f"With few trumps (0-1), rely on {tier} in side suits — every trump card is precious")

    return insights


def _timing_lessons(play_items) -> list:
    """Discover how game phase affects strategy."""
    insights = []
    phase_best = defaultdict(lambda: defaultdict(list))

    for state, actions in play_items:
        if len(state) < 6:
            continue
        phase = state[5]
        for key, q in actions.items():
            if q > 0.3:
                phase_best[phase][key].append(q)

    phase_names = {"1": "opening", "2": "early", "3": "middle", "4": "late", "5": "final"}

    for phase, name in phase_names.items():
        actions = phase_best.get(phase, {})
        best = _top_action(actions)
        if not best:
            continue
        key, _ = best
        tier = TIER_NAMES.get(key[0], "cards")
        is_trump = len(key) > 2 and key[2] == "T"

        if phase == "1" and is_trump:
            insights.append(f"In the opening tricks, leading trump establishes dominance early")
        elif phase == "1" and not is_trump:
            insights.append(f"Opening with {tier} sets your strategy without revealing trump strength")
        elif phase == "5" and is_trump:
            insights.append(f"Save trump for the final tricks — they're unbeatable when it matters most")
        elif phase == "5":
            insights.append(f"In the final tricks, {tier} clean up — by now you know who holds what")
        elif phase == "3":
            insights.append(f"Mid-game is where strategy crystallizes — {tier} are the most impactful plays here")

    return insights


def _bidding_wisdom(bid_q) -> list:
    """Discover bidding lessons."""
    insights = []
    bid_values = defaultdict(list)
    pass_values = []

    for _state, actions in list(bid_q.items())[:500]:
        for key, q in actions.items():
            if key == "PASS":
                pass_values.append(q)
            elif key.startswith("B"):
                bid_values[key].append(q)

    # Is passing generally good or bad?
    if pass_values and len(pass_values) >= 10:
        avg_pass = sum(pass_values) / len(pass_values)
        if avg_pass > 0.3:
            insights.append("Passing when your hand is weak lets the opponents overcommit — let them fail their own bid")
        elif avg_pass < -0.2:
            insights.append("Passing too often means missing scoring opportunities — bid when you have a reasonable hand")

    # Find the most successful bid level.
    best_bid = None
    best_avg = -999
    for key, values in bid_values.items():
        if len(values) >= 5:
            avg = sum(values) / len(values)
            if avg > best_avg:
                best_avg = avg
                best_bid = key

    if best_bid and best_avg > 0.2:
        val = int(best_bid[1:])
        if val <= 8:
            insights.append(f"Conservative bidding ({val}) pays off — under-promise, over-deliver wins consistently")
        elif val >= 10:
            insights.append(f"High bids ({val}+) work when your hand truly supports it — confidence backed by cards")

    return insights


def _void_play(play_items) -> list:
    """Discover void-related strategies."""
    insights = []
    void_create_count = 0
    void_exploit_count = 0

    for state, actions in play_items:
        for key, q in actions.items():
            if len(key) < 5:
                continue
            creates_void = key[4] == "V"
            is_trump_off = len(key) > 2 and key[2] == "T" and key[1] == "O"

            if creates_void and q > 0.3:
                void_create_count += 1
            if is_trump_off and q > 0.3:
                void_exploit_count += 1

    if void_create_count >= 15:
        insights.append("Creating voids is a setup move — play your last card of a suit so you can trump it next time")

    if void_exploit_count >= 10:
        insights.append("Once you're void in a suit, every time it's led you get a free trump — voids are power")

    return insights


def _situational_contrasts(play_items) -> list:
    """Discover where the same card type works in one context but fails in another."""
    insights = []
    pos_action_avgs = defaultdict(lambda: defaultdict(list))

    for state, actions in play_items:
        if len(state) < 6:
            continue
        pos = state[4]
        for key, q in actions.items():
            if abs(q) > 0.2:
                pos_action_avgs[key][pos].append(q)

    for key, pos_dict in pos_action_avgs.items():
        if len(key) < 3:
            continue
        pos_avgs = {}
        for pos, vals in pos_dict.items():
            if len(vals) >= 15:
                pos_avgs[pos] = sum(vals) / len(vals)
        if len(pos_avgs) < 2:
            continue

        max_pos = max(pos_avgs, key=pos_avgs.get)
        min_pos = min(pos_avgs, key=pos_avgs.get)
        spread = pos_avgs[max_pos] - pos_avgs[min_pos]

        if spread > 0.6:
            tier = TIER_NAMES.get(key[0], "cards")
            is_trump = key[2] == "T"
            pos_names = {"0": "leading", "1": "second", "2": "third", "3": "last"}
            good = pos_names.get(max_pos, "?")
            bad = pos_names.get(min_pos, "?")
            card_type = f"trump {tier}" if is_trump else tier
            insights.append(f"{card_type.capitalize()} work well when {good} but backfire when {bad} — position matters")

    return insights[:8]


# ─── Helpers ────────────────────────────────────────────────────────────────────


def _top_action(actions_dict, min_samples=10):
    """Find the action with highest average value."""
    best = None
    best_avg = 0
    for key, vals in actions_dict.items():
        if len(vals) >= min_samples:
            avg = sum(vals) / len(vals)
            if avg > best_avg:
                best_avg = avg
                best = (key, avg)
    return best


def _describe_action(key: str, positive: bool) -> str:
    """Convert an action encoding into a natural-language insight."""
    if len(key) < 4:
        return ""

    tier = key[0]
    follows = key[1] if len(key) > 1 else ""
    is_trump = key[2] if len(key) > 2 else ""
    is_long = key[3] if len(key) > 3 else ""
    creates_void = key[4] if len(key) > 4 else ""

    tier_name = TIER_NAMES.get(tier, "cards")

    if positive:
        # Winning strategies.
        if is_trump == "T" and follows == "O":
            return f"Trumping when void in the led suit wins tricks the opponent thought were safe"
        if tier == "A" and follows == "F":
            return f"Playing Aces when following suit guarantees the trick — decisive and efficient"
        if creates_void == "V" and is_trump == "N":
            return f"Discarding your last card of a suit creates a void for future trumping opportunities"
        if tier == "K" and follows == "F" and is_trump == "T":
            return f"King of trump following suit is nearly unbeatable — only the Ace can stop it"
        if tier in ("L", "X") and follows == "F" and is_trump == "N":
            return f"Playing small cards when you can't win saves your high cards for tricks you can take"
        if is_long == "L" and tier in ("K", "Q"):
            return f"Playing {tier_name} from your longest suit bleeds opponents dry — length creates winners"
        if tier == "A" and is_trump == "T" and follows == "F":
            return f"Ace of trump flushes everyone's trump cards — pure dominance"
        if tier in ("L", "X") and follows == "O" and is_trump == "N":
            return f"When void and not trumping, dump your weakest cards — they'll never win anything anyway"
    else:
        # Losing strategies.
        if tier == "A" and follows == "O" and is_trump == "N":
            return f"Playing your Ace off-suit when void wastes your best card on a trick you can't win"
        if tier == "K" and follows == "O" and is_trump == "N":
            return f"King off-suit when void is a wasted opportunity — it could win in its own suit later"
        if tier in ("M", "J") and follows == "O" and is_trump == "N":
            return f"Mid-range cards off-suit accomplish nothing — they can't win and aren't low enough to dump freely"
        if tier == "X" and follows == "F" and is_trump == "T":
            return f"Low trump (2-4) following trump barely helps — opponents' mid trumps crush it"

    return ""
