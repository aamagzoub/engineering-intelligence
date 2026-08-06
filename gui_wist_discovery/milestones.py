"""
Milestone tracking and auto-discovery system.

Detects behavioral milestones (hardcoded thresholds) and statistical
anomalies (rolling threshold crossings) as the agent learns.
"""

import json
from collections import deque, Counter


# ─── Milestone Storage ──────────────────────────────────────────────────────────

MILESTONES_PATH = "agents/wist_discovery/milestones.json"


def save_milestones(achieved, milestones_list, compute_time):
    """Persist milestones and compute time to disk."""
    try:
        data = {
            "achieved": list(achieved),
            "list": milestones_list,
            "total_compute_sec": compute_time,
        }
        with open(MILESTONES_PATH, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def load_milestones():
    """Load previously saved milestones. Returns (achieved_set, milestones_list, compute_sec)."""
    try:
        with open(MILESTONES_PATH, "r") as f:
            data = json.load(f)
        achieved = set(data.get("achieved", []))
        milestones_list = [
            tuple(x) if isinstance(x, list) else x for x in data.get("list", [])
        ]
        compute_sec = data.get("total_compute_sec", 0.0)
        return achieved, milestones_list, compute_sec
    except (FileNotFoundError, Exception):
        return set(), [], 0.0


# ─── Milestone Detection ────────────────────────────────────────────────────────


def check_milestones(context, trigger_fn):
    """
    Check all hardcoded behavioral milestones.

    Args:
        context: dict with keys:
            team_tricks, bid, playing_team, bid_met, scores,
            team_scores, shota_num, game_num, shotas_played,
            seeks_achieved, bids_met, bids_failed, wist_win_history,
            last_score, best_score, bids_met_streak, bid_history
        trigger_fn: callable(key, message) to record a milestone.
    """
    t0 = context["team_tricks"][0]
    t1 = context["team_tricks"][1]
    s0 = context["scores"].get(0, 0)
    bid = context["bid"]
    playing_team = context["playing_team"]
    bid_met = context["bid_met"]
    team_scores = context["team_scores"]
    shota_num = context["shota_num"]
    shotas_played = context["shotas_played"]
    seeks_achieved = context["seeks_achieved"]
    bids_met = context["bids_met"]
    bids_failed = context["bids_failed"]
    win_history = context["wist_win_history"]

    # === BASIC ===
    trigger_fn("first_shota", "FIRST SHOTA: Played all 13 tricks successfully.")

    if t0 > 0:
        trigger_fn("first_trick", "FIRST TRICK WON: The AI's team won at least one trick.")
    if s0 >= 0:
        trigger_fn("no_loss", "ZERO LOSS SHOTA: Scored 0 or positive -- no penalty.")
    if s0 > 0:
        trigger_fn("positive_score", "POSITIVE SCORE: Earned points this shota.")
    if t0 > t1:
        trigger_fn("won_majority", "WON MAJORITY: Won more tricks than the opponent.")

    # === BIDDING ===
    if playing_team == 0 and bid_met:
        trigger_fn("bid_met", "BID MET: Bid and delivered the promised tricks.")
    if playing_team == 0 and bid_met and bid == t0:
        trigger_fn("efficient_win", "EFFICIENT WIN: Met bid with exactly the right number of tricks.")
    if playing_team == 0 and bid <= 8 and bid_met and t0 > bid:
        trigger_fn("conservative_bid", "CONSERVATIVE BID: Bid low and exceeded it -- under-promise, over-deliver.")
    if playing_team == 0 and bid >= 10 and bid_met:
        trigger_fn("aggressive_bid", "AGGRESSIVE BID: Bid 10+ and met it -- confidence and accuracy.")
    if playing_team == 1 and not bid_met:
        trigger_fn("defended_well", "DEFENDED WELL: Opponent bid and failed -- the AI's team stopped them.")

    # === TRICK-LEVEL ===
    if t0 >= 3:
        trigger_fn("trick_streak", "TRICK STREAK: Won 3+ tricks -- building momentum.")
    if t0 >= 11:
        trigger_fn("near_seek", "NEAR SEEK: Won 11+ tricks -- almost got the seek.")
    if t0 == 13:
        trigger_fn("seek", "SEEK: Won ALL 13 tricks -- instant game win territory.")

    # === IMPROVEMENT ===
    last_score = context.get("last_score")
    if last_score is not None and s0 > last_score:
        trigger_fn("improved", "IMPROVED: Scored better than the previous shota.")

    best_score = context.get("best_score", -999)
    if s0 > best_score and s0 > 5:
        trigger_fn("best_yet", "BEST SHOTA YET: Highest score in any shota so far.")

    # === GAME ===
    if team_scores[0] >= 25:
        trigger_fn("first_win", "FIRST WIN: Reached 25 points and won a game.")
    if team_scores[0] >= 25 and team_scores[0] - team_scores[1] >= 10:
        trigger_fn("dominated", "DOMINATED: Won a game by 10+ point margin.")
    if shotas_played >= 3 and s0 > 0:
        trigger_fn("consistent", "CONSISTENCY: Multiple shotas with positive scores -- strategy forming.")

    # === WIN HISTORY ===
    if shota_num == 5 and len(win_history) >= 3 and all(win_history[-3:]):
        trigger_fn("win_streak", "WIN STREAK: Won 3 games in a row.")
    if len(win_history) >= 10:
        recent = sum(win_history[-10:])
        if recent >= 6:
            trigger_fn("mastery", "MASTERY: Win rate above 60% over 10 games.")
    if len(win_history) >= 10:
        early = sum(win_history[:5]) / 5
        late = sum(win_history[-5:]) / 5
        if late - early >= 0.2:
            trigger_fn("learning_curve", "LEARNING CURVE: Win rate jumped 20%+ from early to recent games.")

    # === PARTNERSHIP ===
    if playing_team == 0 and bid_met and t0 >= bid and t0 - bid <= 2:
        trigger_fn("team_delivery", "TEAM DELIVERY: The team met their bid with tight execution -- both partners contributed.")
    if playing_team == 1 and not bid_met and bid - t1 >= 3:
        trigger_fn("defensive_wall", "DEFENSIVE WALL: Held opponents 3+ tricks below their bid -- strong partnership defense.")

    # === OPPONENT EXPLOITATION ===
    if playing_team == 1 and bid >= 9 and not bid_met:
        trigger_fn("overbid_punish", "OVER-BID PUNISHMENT: Opponents bid 9+ and failed -- exploited their overconfidence.")
    if t1 <= 3:
        trigger_fn("opponent_starved", "OPPONENT STARVED: Opponents won 3 or fewer tricks total -- complete tactical suffocation.")
    if playing_team == 1 and bid >= 8 and t1 <= bid - 4:
        trigger_fn("opponent_crushed", "OPPONENT CRUSHED: Opponents missed their bid by 4+ tricks -- devastating defensive play.")

    # === GAME-WINNING STRATEGIES ===
    if t0 == 13 and s0 > 20:
        trigger_fn("seek_victory", "SEEK VICTORY: Won ALL 13 tricks with a huge score bonus -- the ultimate Wist achievement.")
    if playing_team == 0 and bid_met and bid == t0 and bid >= 7:
        trigger_fn("perfect_bid", "PERFECT BID: Bid 7+ and won exactly that many tricks -- precision hand evaluation.")
    if s0 >= 15:
        trigger_fn("score_explosion", "SCORE EXPLOSION: Earned 15+ points in a single shota -- maximized scoring opportunity.")

    # === COMEBACK ===
    score_at_3 = context.get("score_at_shota3")
    if shota_num == 5 and score_at_3 is not None:
        game_won = team_scores[0] > team_scores[1]
        if score_at_3 < 0 and game_won:
            trigger_fn("comeback_win", "COMEBACK WIN: Was behind at shota 3 but rallied to win the game -- adaptive strategy.")
    if shota_num == 5 and team_scores[0] - team_scores[1] >= 20:
        trigger_fn("runaway_game", "RUNAWAY GAME: Won by 20+ points -- complete strategic superiority over 5 shotas.")

    # === ADVANCED PLAY PATTERNS ===
    bid_history = context.get("bid_history", [])
    if len(bid_history) >= 5:
        recent_bids = bid_history[-5:]
        if all(recent_bids[i] <= recent_bids[i + 1] for i in range(4)) and recent_bids[-1] >= 9:
            trigger_fn("bid_escalation", "BID ESCALATION: Bids increasing over 5 shotas -- growing confidence in hand evaluation.")

    if t0 >= 9 and playing_team == 0 and bid_met:
        trigger_fn("dominant_play", "DOMINANT PLAY: Won 9+ tricks and met the bid -- overwhelming hand strength exploited perfectly.")
    if playing_team == 1 and not bid_met and t0 == (14 - bid):
        trigger_fn("defensive_precision", "DEFENSIVE PRECISION: Won exactly enough tricks to stop the opponent's bid -- surgical defense.")

    # === STREAKS ===
    bids_met_streak = context.get("bids_met_streak", 0)
    if bids_met_streak >= 3:
        trigger_fn("reliable_bidder", "RELIABLE BIDDER: Met bid in 3+ consecutive shotas -- accurate hand assessment.")

    if len(bid_history) >= 2 and playing_team == 0:
        prev_bid = bid_history[-2]
        if prev_bid <= 7 and bid >= 10 and bid_met:
            trigger_fn("high_low_strategy", "HIGH-LOW STRATEGY: Alternated between conservative and aggressive bids successfully -- adaptive bidding.")

    # === META-MASTERY ===
    if len(win_history) >= 5 and all(win_history[-5:]):
        trigger_fn("unbeatable", "UNBEATABLE: Won 5 games in a row -- opponents cannot counter the AI's strategy.")
    if len(win_history) >= 20:
        rate = sum(win_history[-20:]) / 20
        if rate >= 0.7:
            trigger_fn("grand_mastery", "GRAND MASTERY: Win rate exceeds 70% over 20 games -- deep strategic understanding achieved.")

    total_bids = bids_met + bids_failed
    if total_bids >= 10 and bids_met / total_bids >= 0.8:
        trigger_fn("bid_mastery", "BID MASTERY: Met bid in 80%+ of attempts over 10+ shotas -- expert hand evaluation.")
    if team_scores[0] >= 50:
        trigger_fn("score_accumulator", "SCORE ACCUMULATOR: Team score exceeded 50 in a single game -- sustained excellence across all shotas.")
    if seeks_achieved >= 3:
        trigger_fn("seek_hunter", "SEEK HUNTER: Achieved 3+ seeks total -- the AI actively pursues the all-13-tricks strategy when possible.")


# ─── Auto-Discovery (Statistical Threshold Crossings) ───────────────────────────


def create_auto_stats():
    """Create a fresh auto-stats tracking dict."""
    return {
        "scores": deque(maxlen=100),
        "tricks": deque(maxlen=100),
        "bids_met": deque(maxlen=50),
        "win_streaks": 0,
        "total_seeks": 0,
        "last_discovery_episode": 0,
        "thresholds_crossed": set(),
    }


def auto_discover(auto_stats, context, trigger_fn):
    """
    Detect new behaviors by watching rolling statistics.
    Triggers when metrics cross new thresholds.

    Args:
        auto_stats: mutable tracking dict (from create_auto_stats).
        context: dict with team_tricks, scores, playing_team, bid_met, episodes.
        trigger_fn: callable(key, msg).
    """
    s0 = context["scores"].get(0, 0)
    t0 = context["team_tricks"][0]
    playing_team = context["playing_team"]
    bid_met = context["bid_met"]
    episodes = context["episodes"]
    win_history = context.get("wist_win_history", [])

    stats = auto_stats
    stats["scores"].append(s0)
    stats["tricks"].append(t0)
    stats["bids_met"].append(1 if (playing_team == 0 and bid_met) else 0)

    if t0 == 13:
        stats["total_seeks"] += 1

    # Minimum gap between auto-discoveries.
    if episodes - stats["last_discovery_episode"] < 200:
        return
    if len(stats["scores"]) < 20:
        return

    recent_20 = list(stats["scores"])[-20:]
    recent_tricks = list(stats["tricks"])[-20:]
    recent_bids = list(stats["bids_met"])[-20:]

    avg_score = sum(recent_20) / 20
    avg_tricks = sum(recent_tricks) / 20
    win_rate = sum(1 for s in recent_20 if s > 0) / 20 * 100
    bid_acc = sum(recent_bids) / max(len(recent_bids), 1) * 100

    crossed = stats["thresholds_crossed"]

    # Score thresholds.
    for threshold in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]:
        key = f"avg_score_{threshold}"
        if key not in crossed and avg_score >= threshold:
            crossed.add(key)
            stats["last_discovery_episode"] = episodes
            trigger_fn(f"auto_score_{threshold}",
                       f"SCORING POWER: Average score crossed +{threshold} over 20 shotas (actual: {avg_score:.1f})")
            return

    # Win rate thresholds.
    for threshold in [45, 50, 55, 60, 65, 70, 75, 80, 85, 90]:
        key = f"win_rate_{threshold}"
        if key not in crossed and win_rate >= threshold:
            crossed.add(key)
            stats["last_discovery_episode"] = episodes
            trigger_fn(f"auto_wr_{threshold}",
                       f"WIN DOMINANCE: Win rate crossed {threshold}% over 20 shotas (actual: {win_rate:.0f}%)")
            return

    # Bid accuracy thresholds.
    for threshold in [50, 55, 60, 65, 70, 75, 80, 85, 90]:
        key = f"bid_acc_{threshold}"
        if key not in crossed and bid_acc >= threshold:
            crossed.add(key)
            stats["last_discovery_episode"] = episodes
            trigger_fn(f"auto_bid_{threshold}",
                       f"BID PRECISION: Bid accuracy crossed {threshold}% over 20 shotas (actual: {bid_acc:.0f}%)")
            return

    # Trick average thresholds.
    for threshold in [6, 7, 8, 9, 10, 11, 12]:
        key = f"avg_tricks_{threshold}"
        if key not in crossed and avg_tricks >= threshold:
            crossed.add(key)
            stats["last_discovery_episode"] = episodes
            trigger_fn(f"auto_tricks_{threshold}",
                       f"TRICK MACHINE: Averaging {threshold}+ tricks per shota (actual: {avg_tricks:.1f})")
            return

    # Seek milestones.
    for threshold in [5, 10, 20, 50, 100]:
        key = f"seeks_{threshold}"
        if key not in crossed and stats["total_seeks"] >= threshold:
            crossed.add(key)
            stats["last_discovery_episode"] = episodes
            trigger_fn(f"auto_seeks_{threshold}",
                       f"SEEK MASTER: Achieved {threshold} total seeks -- the all-13-tricks strategy is now a reliable weapon")
            return

    # Win streak records.
    if len(win_history) >= 2:
        if win_history[-1]:
            stats["win_streaks"] += 1
        else:
            stats["win_streaks"] = 0

        for threshold in [7, 10, 15, 20, 30]:
            key = f"streak_{threshold}"
            if key not in crossed and stats["win_streaks"] >= threshold:
                crossed.add(key)
                stats["last_discovery_episode"] = episodes
                trigger_fn(f"auto_streak_{threshold}",
                           f"UNSTOPPABLE: Won {threshold} games in a row -- opponents have no answer to the AI's strategy")
                return

    # Significant improvement detection.
    if len(stats["scores"]) >= 40:
        old_20 = list(stats["scores"])[-40:-20]
        new_20 = list(stats["scores"])[-20:]
        old_avg = sum(old_20) / 20
        new_avg = sum(new_20) / 20
        improvement = new_avg - old_avg

        for threshold in [3, 5, 8]:
            key = f"improvement_{threshold}"
            if key not in crossed and improvement >= threshold and new_avg > old_avg * 1.3:
                crossed.add(key)
                stats["last_discovery_episode"] = episodes
                trigger_fn(f"auto_improve_{threshold}",
                           f"BREAKTHROUGH: Average score jumped +{improvement:.1f} "
                           f"(from {old_avg:.1f} to {new_avg:.1f}) -- strategy evolved significantly")
                return
