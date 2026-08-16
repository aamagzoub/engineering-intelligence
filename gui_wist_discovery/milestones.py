"""
Milestone tracking and auto-discovery system.

Detects behavioral milestones (hardcoded thresholds) and statistical
anomalies (rolling threshold crossings) as the agent learns.
"""

import json
from collections import deque, Counter


# ─── Milestone Storage ──────────────────────────────────────────────────────────

MILESTONES_PATH = "agents/wist_discovery/milestones.json"


def save_milestones(achieved, milestones_list, compute_time, session_stats=None):
    """Persist milestones, compute time, and session stats to disk."""
    try:
        data = {
            "achieved": list(achieved),
            "list": milestones_list,
            "total_compute_sec": compute_time,
        }
        if session_stats:
            data["session_stats"] = session_stats
        with open(MILESTONES_PATH, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def load_milestones():
    """Load previously saved milestones. Returns (achieved_set, milestones_list, compute_sec, session_stats)."""
    try:
        with open(MILESTONES_PATH, "r") as f:
            data = json.load(f)
        achieved = set(data.get("achieved", []))
        milestones_list = [
            tuple(x) if isinstance(x, list) else x for x in data.get("list", [])
        ]
        compute_sec = data.get("total_compute_sec", 0.0)
        session_stats = data.get("session_stats", {})
        return achieved, milestones_list, compute_sec, session_stats
    except (FileNotFoundError, Exception):
        return set(), [], 0.0, {}


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

    # === DISCOVERY JOURNEY (pure learning from zero) ===
    trigger_fn("first_shota", "FIRST GAME: Completed 13 decisions in a row without crashing. The brain knows nothing yet.")

    if t0 > 0:
        trigger_fn("first_trick", "FIRST TRICK WON: Won a trick by accident or instinct. Something worked but the AI does not know what yet.")
    if s0 >= 0:
        trigger_fn("no_loss", "NO PENALTY: Finished a shota without losing points. Starting to avoid the worst outcomes.")
    if s0 > 0:
        trigger_fn("positive_score", "FIRST POSITIVE SCORE: Earned actual points. The reward signal is teaching something useful.")
    if t0 > t1:
        trigger_fn("won_majority", "WON MORE TRICKS: Took the majority of tricks. The AI is starting to figure out what wins.")

    # === BIDDING DISCOVERY ===
    if playing_team == 0 and bid_met:
        trigger_fn("bid_met", "BID DELIVERED: Promised a number of tricks and actually got them. Hand reading is emerging.")
    if playing_team == 0 and bid_met and bid == t0:
        trigger_fn("efficient_win", "EXACT BID: Won exactly the number promised, not one more or less. Precision hand evaluation.")
    if playing_team == 0 and bid <= 8 and bid_met and t0 > bid:
        trigger_fn("conservative_bid", "CAUTIOUS BIDDING: Bid low and got more. Learning that under-promising is safer.")
    if playing_team == 0 and bid >= 10 and bid_met:
        trigger_fn("aggressive_bid", "BOLD BID: Bid 10+ and delivered. The AI has learned to recognize very strong hands.")
    if playing_team == 1 and not bid_met:
        trigger_fn("defended_well", "DEFENSE WORKS: Stopped the opponent from making their bid. Learning to block, not just attack.")

    # === TRICK MASTERY ===
    if t0 >= 3:
        trigger_fn("trick_streak", "TRICK MOMENTUM: Won 3+ tricks in one shota. Consecutive wins suggest emerging card-play strategy.")
    if t0 >= 11:
        trigger_fn("near_seek", "NEAR SWEEP: Won 11+ tricks. The AI is dominating entire shotas now.")
    if t0 == 13:
        trigger_fn("seek", "SEEK ACHIEVED: Won ALL 13 tricks. Total domination of one shota.")

    # === LEARNING PROGRESS ===
    last_score = context.get("last_score")
    if last_score is not None and s0 > last_score:
        trigger_fn("improved", "GETTING BETTER: Scored higher than the previous shota. The learning curve is positive.")

    best_score = context.get("best_score", -999)
    if s0 > best_score and s0 > 5:
        trigger_fn("best_yet", "NEW PERSONAL BEST: Highest score ever in a single shota. The brain keeps growing.")

    # === GAME-LEVEL ===
    if team_scores[0] >= 25:
        trigger_fn("first_win", "FIRST GAME WON: Reached 25 points across 5 shotas. Sustained performance over a full game.")
    if team_scores[0] >= 25 and team_scores[0] - team_scores[1] >= 10:
        trigger_fn("dominated", "DOMINATED A GAME: Won by 10+ points. Strategy is clearly superior to the opponent.")
    if shotas_played >= 3 and s0 > 0:
        trigger_fn("consistent", "CONSISTENCY: Multiple positive scores in a row. A strategy is forming from pure trial and error.")

    # === WIN HISTORY ===
    if shota_num == 5 and len(win_history) >= 3 and all(win_history[-3:]):
        trigger_fn("win_streak", "THREE IN A ROW: Won 3 games consecutively. Not luck anymore, the AI has learned something real.")
    if len(win_history) >= 10:
        recent = sum(win_history[-10:])
        if recent >= 6:
            trigger_fn("mastery", "WINNING MAJORITY: Win rate above 60% over 10 games. Genuine strategic understanding developing.")
    if len(win_history) >= 10:
        early = sum(win_history[:5]) / 5
        late = sum(win_history[-5:]) / 5
        if late - early >= 0.2:
            trigger_fn("learning_curve", "VISIBLE LEARNING: Win rate jumped 20%+ from early games to recent. The brain is growing.")

    # === PARTNERSHIP ===
    if playing_team == 0 and bid_met and t0 >= bid and t0 - bid <= 2:
        trigger_fn("team_delivery", "TEAM COORDINATION: Met the bid with tight execution. Both team members contributed to the win.")
    if playing_team == 1 and not bid_met and bid - t1 >= 3:
        trigger_fn("defensive_wall", "STRONG DEFENSE: Held opponents 3+ tricks below their bid. Learned to block effectively.")

    # === OPPONENT EXPLOITATION ===
    if playing_team == 1 and bid >= 9 and not bid_met:
        trigger_fn("overbid_punish", "PUNISHED OVERBID: Opponents bid 9+ and failed. The AI exploited their overconfidence.")
    if t1 <= 3:
        trigger_fn("opponent_starved", "STARVED OPPONENTS: Opponents won only 3 or fewer tricks. Complete control of the table.")
    if playing_team == 1 and bid >= 8 and t1 <= bid - 4:
        trigger_fn("opponent_crushed", "CRUSHED OPPONENTS: They missed their bid by 4+ tricks. Devastating defense.")

    # === GAME-WINNING MASTERY ===
    if t0 == 13 and s0 > 20:
        trigger_fn("seek_victory", "SEEK WITH BONUS: Won all 13 tricks with a huge score bonus. The ultimate Wist play.")
    if playing_team == 0 and bid_met and bid == t0 and bid >= 7:
        trigger_fn("perfect_bid", "PERFECT READ: Bid 7+ and won exactly that many. Precision hand evaluation from zero knowledge.")
    if s0 >= 15:
        trigger_fn("score_explosion", "BIG SCORE: Earned 15+ points in one shota. Maximized a scoring opportunity.")

    # === COMEBACK ===
    score_at_3 = context.get("score_at_shota3")
    if shota_num == 5 and score_at_3 is not None:
        game_won = team_scores[0] > team_scores[1]
        if score_at_3 < 0 and game_won:
            trigger_fn("comeback_win", "COMEBACK: Was losing at shota 3 but rallied to win. Adaptive strategy under pressure.")
    if shota_num == 5 and team_scores[0] - team_scores[1] >= 20:
        trigger_fn("runaway_game", "RUNAWAY WIN: Won by 20+ points. Complete strategic superiority over all 5 shotas.")

    # === ADVANCED PATTERNS ===
    bid_history = context.get("bid_history", [])
    if len(bid_history) >= 5:
        recent_bids = bid_history[-5:]
        if all(recent_bids[i] <= recent_bids[i + 1] for i in range(4)) and recent_bids[-1] >= 9:
            trigger_fn("bid_escalation", "GROWING CONFIDENCE: Bids increasing over 5 shotas. The AI trusts its hand reading more.")

    if t0 >= 9 and playing_team == 0 and bid_met:
        trigger_fn("dominant_play", "DOMINANT SHOTA: Won 9+ tricks and met the bid. Overwhelming hand strength exploited.")
    if playing_team == 1 and not bid_met and t0 == (14 - bid):
        trigger_fn("defensive_precision", "SURGICAL DEFENSE: Won exactly enough tricks to stop the opponent. No waste.")

    # === STREAKS ===
    bids_met_streak = context.get("bids_met_streak", 0)
    if bids_met_streak >= 3:
        trigger_fn("reliable_bidder", "RELIABLE: Met bid 3 times in a row. Hand reading is becoming consistent.")
    if bids_met_streak >= 50:
        trigger_fn("bid_streak_50", "FIFTY BIDS MET: 50 consecutive successful bids. Near-perfect hand assessment.")
    if bids_met_streak >= 100:
        trigger_fn("bid_streak_100", "HUNDRED BIDS MET: 100 in a row. The AI never misjudges its hand anymore.")

    if len(bid_history) >= 2 and playing_team == 0:
        prev_bid = bid_history[-2]
        if prev_bid <= 7 and bid >= 10 and bid_met:
            trigger_fn("high_low_strategy", "ADAPTIVE BIDDING: Switches between cautious and bold bids successfully. Reads each hand differently.")

    # === SUSTAINED EXCELLENCE ===
    if len(win_history) >= 5 and all(win_history[-5:]):
        trigger_fn("unbeatable", "FIVE STRAIGHT: Won 5 games in a row. Opponents cannot counter the AI.")
    if len(win_history) >= 20:
        rate = sum(win_history[-20:]) / 20
        if rate >= 0.7:
            trigger_fn("grand_mastery", "70% WIN RATE: Over 20 games. Deep strategic understanding achieved from zero.")

    total_bids = bids_met + bids_failed
    if total_bids >= 10 and bids_met / total_bids >= 0.8:
        trigger_fn("bid_mastery", "80% BID ACCURACY: Met bid in 80%+ of attempts. Expert hand evaluation learned from scratch.")
    if team_scores[0] >= 50:
        trigger_fn("score_accumulator", "50+ POINTS: Team score exceeded 50 in one game. Sustained excellence across all shotas.")
    if seeks_achieved >= 3:
        trigger_fn("seek_hunter", "SEEK HUNTER: 3+ seeks total. The AI knows when to go for all 13 tricks.")

    # === LONG STREAKS ===
    if len(win_history) >= 50 and all(win_history[-50:]):
        trigger_fn("streak_50", "50 UNBEATEN: Won 50 games without a loss. Opponents are completely outclassed.")
    if len(win_history) >= 100 and all(win_history[-100:]):
        trigger_fn("streak_100", "100 UNBEATEN: Won 100 in a row. Total strategic dominance learned from nothing.")
    if len(win_history) >= 200 and all(win_history[-200:]):
        trigger_fn("streak_200", "200 UNBEATEN: Won 200 consecutive games. Playing at a level beyond its opponents.")
    if len(win_history) >= 500 and all(win_history[-500:]):
        trigger_fn("streak_500", "500 UNBEATEN: Won 500 in a row. Perfection sustained over extraordinary length.")
    if len(win_history) >= 1000 and all(win_history[-1000:]):
        trigger_fn("streak_1000", "1000 UNBEATEN: Won 1000 consecutive games. The brain cannot be beaten.")

    # === PERFECT GAME ===
    if shota_num == 5 and bids_met_streak >= 5 and playing_team == 0 and bid_met:
        trigger_fn("perfect_game", "PERFECT GAME: Met the bid in all 5 shotas. Flawless from start to finish.")

    # === DOMINANCE ===
    if shota_num == 5 and team_scores[1] == 0 and team_scores[0] > 0:
        trigger_fn("shutout", "SHUTOUT: Opponents scored zero all game. Complete suppression.")
    if len(win_history) >= 100 and all(win_history[-100:]):
        trigger_fn("hundred_no_loss", "HUNDRED WITHOUT LOSS: 100 games, zero losses. Untouchable.")

    # === TRICK MASTERY ===
    if t0 >= 12:
        trigger_fn("trick_12", "NEAR PERFECT: Won 12 out of 13 tricks. Only one escaped.")
    if t0 == 13 and context.get("prev_shota_tricks_0", 0) == 13:
        trigger_fn("back_to_back_seek", "BACK-TO-BACK SEEK: Won all 13 tricks in two consecutive shotas. Terrifying.")

    # === RARE EVENTS ===
    if playing_team == 0 and bid == 7 and t0 == 13:
        trigger_fn("overbid_7_seek", "BID 7 GOT 13: Bid only 7 but won all 13. Massive hidden strength in the hand.")
    if playing_team == 1 and t1 == 13:
        trigger_fn("opp_seek_survived", "OPPONENT SWEPT: Opponents won all 13 tricks. A humbling reminder.")
    if playing_team == 1 and not bid_met:
        defense_streak = context.get("defense_streak", 0)
        if defense_streak >= 20:
            trigger_fn("defense_wall_20", "IRON WALL: Opponents failed their bid 20 times in a row. Nobody gets past.")


# ─── Auto-Discovery (Statistical Threshold Crossings) ───────────────────────────


def create_auto_stats():
    """Create a fresh auto-stats tracking dict."""
    return {
        "scores": deque(maxlen=200),
        "tricks": deque(maxlen=200),
        "bids_met": deque(maxlen=200),
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
    for threshold in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20]:
        key = f"avg_score_{threshold}"
        if key not in crossed and avg_score >= threshold:
            crossed.add(key)
            stats["last_discovery_episode"] = episodes
            trigger_fn(f"auto_score_{threshold}",
                       f"SCORING POWER: Average score crossed +{threshold} over 20 shotas. Actual: {avg_score:.1f}")
            return

    # Win rate thresholds.
    for threshold in [45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 98]:
        key = f"win_rate_{threshold}"
        if key not in crossed and win_rate >= threshold:
            crossed.add(key)
            stats["last_discovery_episode"] = episodes
            trigger_fn(f"auto_wr_{threshold}",
                       f"WIN RATE: Crossed {threshold}% over 20 shotas. Actual: {win_rate:.0f}%")
            return

    # Bid accuracy thresholds.
    for threshold in [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 98]:
        key = f"bid_acc_{threshold}"
        if key not in crossed and bid_acc >= threshold:
            crossed.add(key)
            stats["last_discovery_episode"] = episodes
            trigger_fn(f"auto_bid_{threshold}",
                       f"BID ACCURACY: Crossed {threshold}% over 20 shotas. Actual: {bid_acc:.0f}%")
            return

    # Trick average thresholds.
    for threshold in [6, 7, 8, 9, 10, 11, 12]:
        key = f"avg_tricks_{threshold}"
        if key not in crossed and avg_tricks >= threshold:
            crossed.add(key)
            stats["last_discovery_episode"] = episodes
            trigger_fn(f"auto_tricks_{threshold}",
                       f"TRICK AVERAGE: Now averaging {threshold}+ tricks per shota. Actual: {avg_tricks:.1f}")
            return

    # Seek milestones.
    for threshold in [5, 10, 20, 50, 100, 200, 500, 1000, 5000]:
        key = f"seeks_{threshold}"
        if key not in crossed and stats["total_seeks"] >= threshold:
            crossed.add(key)
            stats["last_discovery_episode"] = episodes
            trigger_fn(f"auto_seeks_{threshold}",
                       f"SEEK COUNT: Achieved {threshold} total seeks. The all-13-tricks play is now a reliable weapon.")
            return

    # Win streak records.
    if len(win_history) >= 2:
        if win_history[-1]:
            stats["win_streaks"] += 1
        else:
            stats["win_streaks"] = 0

        for threshold in [7, 10, 15, 20, 30, 50, 100, 200, 500, 1000]:
            key = f"streak_{threshold}"
            if key not in crossed and stats["win_streaks"] >= threshold:
                crossed.add(key)
                stats["last_discovery_episode"] = episodes
                trigger_fn(f"auto_streak_{threshold}",
                           f"WIN STREAK: Won {threshold} games in a row. Opponents have no answer.")
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
                           f"(from {old_avg:.1f} to {new_avg:.1f}). Strategy evolved significantly.")
                return

    # Seek rate (over recent games).
    if stats["total_seeks"] > 0 and len(stats["scores"]) >= 50:
        recent_count = len(stats["scores"])
        seek_rate = stats["total_seeks"] / max(recent_count, 1) * 100
        for threshold in [5, 10]:
            key = f"seek_rate_{threshold}"
            if key not in crossed and seek_rate >= threshold:
                crossed.add(key)
                stats["last_discovery_episode"] = episodes
                trigger_fn(f"auto_seek_rate_{threshold}",
                           f"SEEK RATE: Achieving seeks in {threshold}%+ of shotas. Aggressively hunting all-13 opportunities.")
                return

    # Longer window checks (100 shotas).
    if len(stats["scores"]) >= 100:
        last_100_scores = list(stats["scores"])[-100:]
        last_100_tricks = list(stats["tricks"])[-100:]
        last_100_bids = list(stats["bids_met"])[-100:]

        avg_tricks_100 = sum(last_100_tricks) / 100
        bid_acc_100 = sum(last_100_bids) / 100 * 100

        # Trick mastery over 100 shotas.
        for threshold in [11, 12]:
            key = f"tricks_100_{threshold}"
            if key not in crossed and avg_tricks_100 >= threshold:
                crossed.add(key)
                stats["last_discovery_episode"] = episodes
                trigger_fn(f"auto_tricks100_{threshold}",
                           f"SUSTAINED TRICKS: Averaging {threshold}+ tricks over 100 shotas. Actual: {avg_tricks_100:.1f}")
                return

        # Bid accuracy over 100 shotas.
        for threshold in [95, 98]:
            key = f"bid_100_{threshold}"
            if key not in crossed and bid_acc_100 >= threshold:
                crossed.add(key)
                stats["last_discovery_episode"] = episodes
                trigger_fn(f"auto_bid100_{threshold}",
                           f"BID MASTERY: {threshold}%+ bid accuracy over 100 shotas. Near-zero hand reading error.")
                return

        # Win rate over 100 shotas.
        win_rate_100 = sum(1 for s in last_100_scores if s > 0) / 100 * 100
        for threshold in [95, 98]:
            key = f"wr_100_{threshold}"
            if key not in crossed and win_rate_100 >= threshold:
                crossed.add(key)
                stats["last_discovery_episode"] = episodes
                trigger_fn(f"auto_wr100_{threshold}",
                           f"NEAR PERFECT: {threshold}%+ win rate over 100 shotas. Losses almost non-existent.")
                return

    # Volume milestones — celebrate training progress.
    for threshold in [100000, 250000, 500000, 1000000, 2000000, 3000000, 5000000,
                      7500000, 10000000, 25000000, 50000000, 100000000]:
        key = f"volume_{threshold}"
        if key not in crossed and episodes >= threshold:
            crossed.add(key)
            stats["last_discovery_episode"] = episodes
            label = f"{threshold:,}"
            trigger_fn(f"auto_volume_{threshold}",
                       f"TRAINING VOLUME: Reached {label} shotas learned. The brain keeps growing deeper.")
            return
