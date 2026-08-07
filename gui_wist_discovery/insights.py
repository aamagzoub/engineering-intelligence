"""
Strategic insights — surfaces non-obvious strategy tips from Q-table analysis.

Two layers:
1. Static insights — counter-intuitive patterns from current Q-values.
2. Evolution insights — strategy shifts detected by comparing snapshots over time.
   These are ENDLESS: every time the agent changes its mind about something,
   a new insight is generated describing the shift.

Each insight is phrased as actionable advice a player can use in their next game.
"""

from collections import defaultdict, Counter
import json
import os

# Path for evolution snapshots.
_SNAPSHOTS_PATH = "agents/wist_discovery/strategy_snapshots.json"


def generate_insights(agent) -> list:
    """
    Extract non-obvious strategy tips from what the agent learned.

    Returns actionable "do this / don't do that" tips phrased for a human player.
    Combines static tips + evolution-based discoveries (endless).
    """
    if agent.episodes_trained < 5000:
        return ["Still learning basics — check back after more training..."]

    insights = []
    play_q = agent.play_q
    bid_q = agent.bid_q

    q_size = len(play_q)
    play_items = list(play_q.items())[:min(q_size, 10000)]

    # Layer 1: Static counter-intuitive patterns.
    insights.extend(_counter_intuitive_plays(play_items))
    insights.extend(_hidden_power_moves(play_items))
    insights.extend(_timing_traps(play_items))
    insights.extend(_positional_secrets(play_items))
    insights.extend(_trump_wisdom(play_items))
    insights.extend(_bidding_tips(bid_q))
    insights.extend(_void_tactics(play_items))

    # Layer 2: Evolution insights (endless — new ones appear as strategy evolves).
    evolution = _detect_evolution(play_items, bid_q, agent.episodes_trained)
    insights.extend(evolution)

    # Deduplicate by core concept (first 6 words).
    seen_concepts = set()
    unique = []
    for insight in insights:
        words = insight.split()
        concept = " ".join(words[:6]).lower().rstrip(".,—-")
        if concept not in seen_concepts:
            seen_concepts.add(concept)
            unique.append(insight)
    return unique


# ─── Evolution Layer (Endless Insights) ─────────────────────────────────────────


def _detect_evolution(play_items, bid_q, episodes) -> list:
    """
    Compare current strategy to previous snapshots.
    Detect macro-level behavioral shifts and describe them as strategic principles.

    This produces new insights indefinitely as the strategy evolves.
    """
    current_fp = _build_fingerprint(play_items, bid_q)

    # Load previous snapshots.
    snapshots = _load_snapshots()

    # Snapshot every 10k episodes.
    last_ep = snapshots[-1]["episodes"] if snapshots else 0
    if episodes - last_ep >= 10000:
        snapshots.append({"episodes": episodes, "fingerprint": current_fp})
        if len(snapshots) > 50:
            snapshots = snapshots[-50:]
        _save_snapshots(snapshots)

    insights = []
    if len(snapshots) < 2:
        return insights

    # Compare current to an older snapshot.
    compare_idx = max(0, len(snapshots) - 3)
    old_fp = snapshots[compare_idx]["fingerprint"]

    # Detect macro-level shifts (not individual card changes).
    insights.extend(_macro_shifts(old_fp, current_fp))

    return insights


def _macro_shifts(old_fp: dict, new_fp: dict) -> list:
    """Detect high-level behavioral changes and phrase as strategic principles."""
    insights = []

    # Count how the overall strategy shifted across all contexts.
    aggression_change = 0  # Positive = became more aggressive.
    trump_change = 0       # Positive = using more trump.
    patience_change = 0    # Positive = saving more high cards.
    void_change = 0        # Positive = creating more voids.
    pass_change = 0        # Positive = passing more in bids.

    for ctx in new_fp:
        if ctx not in old_fp or ctx == "bid_best":
            continue
        old_a = old_fp[ctx]["action"]
        new_a = new_fp[ctx]["action"]
        if old_a == new_a:
            continue

        # Classify the shift direction.
        old_tier = old_a[0] if old_a else ""
        new_tier = new_a[0] if new_a else ""
        old_trump = old_a[2] == "T" if len(old_a) > 2 else False
        new_trump = new_a[2] == "T" if len(new_a) > 2 else False
        new_void = new_a[4] == "V" if len(new_a) > 4 else False
        old_void = old_a[4] == "V" if len(old_a) > 4 else False

        tier_rank = {"A": 5, "K": 4, "Q": 3, "J": 2, "M": 1, "L": 0, "X": -1}
        old_rank = tier_rank.get(old_tier, 0)
        new_rank = tier_rank.get(new_tier, 0)

        if new_rank > old_rank:
            aggression_change += 1
        elif new_rank < old_rank:
            patience_change += 1

        if new_trump and not old_trump:
            trump_change += 1
        elif old_trump and not new_trump:
            trump_change -= 1

        if new_void and not old_void:
            void_change += 1

    # Bid shift.
    if "bid_best" in old_fp and "bid_best" in new_fp:
        old_bid = old_fp["bid_best"]["action"]
        new_bid = new_fp["bid_best"]["action"]
        if old_bid != new_bid:
            if new_bid == "PASS" and old_bid != "PASS":
                pass_change += 3
            elif old_bid == "PASS" and new_bid != "PASS":
                pass_change -= 3
            elif new_bid.startswith("B") and old_bid.startswith("B"):
                try:
                    if int(new_bid[1:]) < int(old_bid[1:]):
                        pass_change += 1  # More conservative.
                    else:
                        pass_change -= 1  # More aggressive.
                except ValueError:
                    pass

    # Generate principles from macro shifts (only if significant).
    if aggression_change >= 3:
        insights.append("Start playing your high cards sooner — waiting too long lets opponents set up their trumps")
    if patience_change >= 3:
        insights.append("Don't play your Kings and Queens in the first few tricks — they win more when you save them for later")
    if trump_change >= 3:
        insights.append("Use trump cards more often — they control the game better than holding them back")
    if trump_change <= -3:
        insights.append("Stop spending trumps early — you'll need them later when opponents try to steal tricks")
    if void_change >= 2:
        insights.append("Get rid of short suits early — once you're void, you can trump that suit every time it comes up")
    if pass_change >= 2:
        insights.append("Pass more often when your hand is weak — letting opponents overbid and fail is free points")
    if pass_change <= -2:
        insights.append("Bid more when you have a decent hand — playing it too safe means missing easy points")

    return insights


def _build_fingerprint(play_items, bid_q) -> dict:
    """Build a strategy fingerprint: best action per context cluster."""
    fp = {}

    # Play fingerprint: best action per (phase, position).
    context_best = defaultdict(lambda: defaultdict(list))
    for state, actions in play_items:
        if len(state) < 6:
            continue
        pos = state[4]
        phase = state[5]
        ctx = f"p{phase}s{pos}"
        for key, q in actions.items():
            if len(key) >= 3:
                context_best[ctx][key].append(q)

    for ctx, action_vals in context_best.items():
        best_key = ""
        best_avg = -999
        for key, vals in action_vals.items():
            if len(vals) >= 5:
                avg = sum(vals) / len(vals)
                if avg > best_avg:
                    best_avg = avg
                    best_key = key
        if best_key:
            fp[ctx] = {"action": best_key, "avg_q": round(best_avg, 3)}

    # Bid fingerprint.
    bid_vals = defaultdict(list)
    for _state, actions in list(bid_q.items())[:300]:
        for key, q in actions.items():
            bid_vals[key].append(q)
    best_bid = "PASS"
    best_bid_q = -999
    for key, vals in bid_vals.items():
        if len(vals) >= 5:
            avg = sum(vals) / len(vals)
            if avg > best_bid_q:
                best_bid_q = avg
                best_bid = key
    fp["bid_best"] = {"action": best_bid, "avg_q": round(best_bid_q, 3)}

    return fp


def _load_snapshots() -> list:
    """Load strategy snapshots from disk."""
    try:
        if os.path.exists(_SNAPSHOTS_PATH):
            with open(_SNAPSHOTS_PATH, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_snapshots(snapshots: list):
    """Save strategy snapshots to disk."""
    try:
        with open(_SNAPSHOTS_PATH, "w") as f:
            json.dump(snapshots, f)
    except Exception:
        pass

def _counter_intuitive_plays(play_items) -> list:
    """Find high cards that fail and low cards that succeed — the surprises."""
    insights = []

    # Collect: action_key -> list of Q-values.
    action_q = defaultdict(list)
    for _state, actions in play_items:
        for key, q in actions.items():
            if len(key) >= 3 and abs(q) > 0.2:
                action_q[key].append(q)

    # High cards (A, K, Q) that are NEGATIVE in some context = trap.
    for key, values in action_q.items():
        if len(values) < 15:
            continue
        avg = sum(values) / len(values)
        tier = key[0]
        follows = key[1] if len(key) > 1 else ""
        is_trump = key[2] if len(key) > 2 else ""

        # Aces/Kings that backfire.
        if tier == "A" and follows == "O" and is_trump == "N" and avg < -0.3:
            insights.append("Don't throw your Ace off-suit when void — it's your strongest card wasted on a trick you can't win anyway")

        if tier == "K" and follows == "F" and is_trump == "N" and avg < -0.2:
            insights.append("Don't rush your King when following — if the Ace hasn't appeared yet, your King dies for nothing")

        if tier == "Q" and follows == "F" and is_trump == "N" and avg < -0.2:
            insights.append("Queens are traps when played too early — they lose to Kings and Aces still lurking behind you")

        # Low cards that secretly win.
        if tier == "X" and follows == "F" and is_trump == "N" and avg > 0.3:
            insights.append("Play your smallest card when you can't win — don't waste good cards on lost tricks")

        if tier == "L" and follows == "O" and is_trump == "N" and avg > 0.2:
            insights.append("Dump your weak off-suit cards when void — they'll never win anything, so get rid of them now while you can")

        # Small trump that overperforms.
        if tier in ("L", "X") and follows == "O" and is_trump == "T" and avg > 0.4:
            insights.append("Even a 2 of trump beats any off-suit King — when you're void, your worst trump is stronger than their best card")

    return insights


def _hidden_power_moves(play_items) -> list:
    """Find moves that look weak but have high Q-values — hidden strategies."""
    insights = []

    void_create_q = []
    long_suit_low_q = []

    for _state, actions in play_items:
        for key, q in actions.items():
            if len(key) < 5:
                continue
            creates_void = key[4] == "V"
            is_long = key[3] == "L" if len(key) > 3 else False
            tier = key[0]
            is_trump = key[2] == "T" if len(key) > 2 else False

            if creates_void and not is_trump and q > 0.3:
                void_create_q.append(q)

            if is_long and tier in ("L", "X") and not is_trump and q > 0.3:
                long_suit_low_q.append(q)

    if len(void_create_q) >= 10:
        avg = sum(void_create_q) / len(void_create_q)
        if avg > 0.3:
            insights.append("Play your last card of a side suit on purpose — being void lets you trump that suit every time it comes back")

    if len(long_suit_low_q) >= 10:
        avg = sum(long_suit_low_q) / len(long_suit_low_q)
        if avg > 0.3:
            insights.append("Lead low from your longest suit — opponents will run out of that suit and your remaining cards win for free")

    return insights


def _timing_traps(play_items) -> list:
    """Find strategies that only work at certain game phases — timing matters."""
    insights = []

    # Group by phase.
    phase_action_q = defaultdict(lambda: defaultdict(list))
    for state, actions in play_items:
        if len(state) < 6:
            continue
        phase = state[5]
        for key, q in actions.items():
            if len(key) >= 3 and abs(q) > 0.3:
                phase_action_q[phase][key].append(q)

    # Find actions that are GOOD early but BAD late (or vice versa).
    for key in set().union(*(phase_action_q[p].keys() for p in phase_action_q)):
        early_vals = phase_action_q.get("1", {}).get(key, [])
        late_vals = phase_action_q.get("5", {}).get(key, [])

        if len(early_vals) < 10 or len(late_vals) < 10:
            continue

        early_avg = sum(early_vals) / len(early_vals)
        late_avg = sum(late_vals) / len(late_vals)

        tier = key[0]
        is_trump = key[2] == "T" if len(key) > 2 else False
        tier_name = {"A": "Aces", "K": "Kings", "Q": "Queens", "J": "Jacks"}.get(tier, "")

        if not tier_name:
            continue

        # Good early, bad late.
        if early_avg > 0.4 and late_avg < 0:
            if is_trump:
                insights.append(f"Lead trump {tier_name} early to flush out opposition — but in the final tricks, they're wasted because opponents may already be out of trump")
            else:
                insights.append(f"Play {tier_name} early when they still dominate — holding them too long risks them being trumped in later tricks")

        # Bad early, good late.
        if early_avg < 0 and late_avg > 0.4:
            if is_trump:
                insights.append(f"Save your trump {tier_name} for the endgame — they're unstoppable when opponents have run out of options")
            else:
                insights.append(f"Hold your {tier_name} until late in the shota — by then you know exactly who holds what, and they win cleanly")

    return insights[:6]


def _positional_secrets(play_items) -> list:
    """Find position-dependent strategies — what works in one seat fails in another."""
    insights = []

    pos_action_q = defaultdict(lambda: defaultdict(list))
    for state, actions in play_items:
        if len(state) < 5:
            continue
        pos = state[4]
        for key, q in actions.items():
            if len(key) >= 3 and abs(q) > 0.3:
                pos_action_q[pos][key].append(q)

    # Compare leading (pos 0) vs last (pos 3).
    for key in set().union(*(pos_action_q[p].keys() for p in pos_action_q)):
        lead_vals = pos_action_q.get("0", {}).get(key, [])
        last_vals = pos_action_q.get("3", {}).get(key, [])

        if len(lead_vals) < 10 or len(last_vals) < 10:
            continue

        lead_avg = sum(lead_vals) / len(lead_vals)
        last_avg = sum(last_vals) / len(last_vals)
        spread = last_avg - lead_avg

        tier = key[0]
        is_trump = key[2] == "T" if len(key) > 2 else False

        if spread > 0.5:
            tier_name = {"A": "Aces", "K": "Kings", "Q": "Queens", "J": "Jacks", "M": "mid cards"}.get(tier, "")
            if tier_name:
                card_type = f"trump {tier_name}" if is_trump else tier_name
                insights.append(f"Wait to play {card_type} until you're last — you see all other cards first and can play the minimum needed to win")

        if spread < -0.5:
            tier_name = {"A": "Aces", "K": "Kings", "Q": "Queens"}.get(tier, "")
            if tier_name:
                card_type = f"trump {tier_name}" if is_trump else tier_name
                insights.append(f"Lead with {card_type} aggressively — waiting gives opponents time to set up their trumps against you")

    return insights[:6]


def _trump_wisdom(play_items) -> list:
    """Trump-specific advice derived from experience."""
    insights = []

    trump_lead_q = []
    trump_whip_high_q = []
    trump_whip_low_q = []
    trump_follow_low_q = []

    for _state, actions in play_items:
        for key, q in actions.items():
            if len(key) < 3:
                continue
            tier = key[0]
            follows = key[1]
            is_trump = key[2] == "T"

            if not is_trump:
                continue

            # Leading trump (following suit = leading since it's trump suit).
            if follows == "F" and tier in ("A", "K") and q > 0.3:
                trump_lead_q.append(q)

            # Whipping (trump off-suit) with high cards.
            if follows == "O" and tier in ("A", "K", "Q"):
                trump_whip_high_q.append(q)

            # Whipping with low cards.
            if follows == "O" and tier in ("L", "X"):
                trump_whip_low_q.append(q)

            # Following trump with low cards.
            if follows == "F" and tier in ("L", "X") and q < -0.2:
                trump_follow_low_q.append(q)

    if len(trump_lead_q) >= 10 and sum(trump_lead_q) / len(trump_lead_q) > 0.4:
        insights.append("Lead your highest trump to force everyone to spend theirs — after a few rounds of this, your remaining trumps are uncontested")

    if len(trump_whip_low_q) >= 10 and sum(trump_whip_low_q) / len(trump_whip_low_q) > 0.3:
        insights.append("Trump with your lowest trump when void — it still wins the trick, and you save your high trumps for when opponents also trump")

    if len(trump_follow_low_q) >= 5 and sum(trump_follow_low_q) / len(trump_follow_low_q) < -0.3:
        insights.append("Don't waste low trumps following a trump lead — they'll lose to everyone else's higher trump anyway. Save them for whipping")

    if len(trump_whip_high_q) >= 10 and sum(trump_whip_high_q) / len(trump_whip_high_q) > 0.4:
        insights.append("When void in a suit, trump with your highest — it guarantees the win and prevents opponents from over-trumping you")

    return insights


def _bidding_tips(bid_q) -> list:
    """Bidding wisdom — when to bid, when to pass."""
    insights = []

    pass_q = []
    bid_by_value = defaultdict(list)

    for _state, actions in list(bid_q.items())[:500]:
        for key, q in actions.items():
            if key == "PASS":
                pass_q.append(q)
            elif key.startswith("B") and len(key) > 1:
                try:
                    val = int(key[1:])
                    bid_by_value[val].append(q)
                except ValueError:
                    pass

    # Passing wisdom.
    if len(pass_q) >= 10:
        avg = sum(pass_q) / len(pass_q)
        if avg > 0.3:
            insights.append("Pass more often than you think — letting the opponent bid and fail gives your team free points without risk")
        elif avg < -0.2:
            insights.append("Don't pass when you have a playable hand — missed bids mean missed scoring opportunities your opponents will take instead")

    # Find the sweet spot bid.
    best_val = None
    best_avg = -999
    worst_val = None
    worst_avg = 999
    for val, values in bid_by_value.items():
        if len(values) >= 5:
            avg = sum(values) / len(values)
            if avg > best_avg:
                best_avg = avg
                best_val = val
            if avg < worst_avg:
                worst_avg = avg
                worst_val = val

    if best_val and best_avg > 0.2:
        if best_val <= 8:
            insights.append(f"Bid {best_val} when in doubt — under-promising and over-delivering is safer than chasing glory with a high bid")
        else:
            insights.append(f"When your hand supports it, bid {best_val} confidently — strong hands should commit fully, not play it safe")

    if worst_val and worst_avg < -0.3 and worst_val >= 10:
        insights.append(f"Avoid bidding {worst_val}+ unless you're certain — failing a high bid costs your team dearly and gifts the opponents easy points")

    return insights


def _void_tactics(play_items) -> list:
    """Void-related tactics — the hidden power of emptying a suit."""
    insights = []

    # Track void creation success.
    void_create_positive = 0
    void_create_total = 0

    # Track whipping after void.
    whip_success = 0
    whip_total = 0

    for _state, actions in play_items:
        for key, q in actions.items():
            if len(key) < 5:
                continue
            creates_void = key[4] == "V"
            is_trump_off = key[2] == "T" and key[1] == "O"

            if creates_void:
                void_create_total += 1
                if q > 0.2:
                    void_create_positive += 1

            if is_trump_off:
                whip_total += 1
                if q > 0.3:
                    whip_success += 1

    if void_create_total >= 20 and void_create_positive / max(void_create_total, 1) > 0.6:
        insights.append("Actively empty a short suit in the first tricks — once void, you can trump it every time")

    if whip_total >= 20 and whip_success / max(whip_total, 1) > 0.5:
        insights.append("Once void in a suit, every time it's led is a free trick — set up your voids early")

    return insights
