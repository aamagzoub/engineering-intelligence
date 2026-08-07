"""
Player Evaluation System — tracks and evaluates human player performance.

Provides:
- Skill rating (Elo-like)
- Decision quality scoring (vs AI choices)
- Bidding accuracy tracking
- Category breakdown (trump, voids, partnership, defense)
- Post-game analysis (turning points)
- Improvement tracking over time
- Insight matching (did player follow AI-discovered lessons?)
"""

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

_SAVE_PATH = "player_stats.json"


@dataclass
class TrickDecision:
    """A single card-play decision record."""
    shota: int
    trick: int
    player_card: str        # What human played.
    ai_would_play: str      # What AI would have played.
    agreed: bool            # Did they match?
    outcome_good: bool      # Did the trick outcome benefit the team?
    position: int           # 0=lead, 1=second, 2=third, 3=last
    was_trump: bool         # Did the player play trump?
    created_void: bool      # Did this play create a void?
    context: str            # Brief description of situation.


@dataclass
class BidDecision:
    """A single bidding decision record."""
    shota: int
    player_bid: int         # What human bid (0 = pass).
    ai_would_bid: int       # What AI would have bid.
    actual_tricks: int      # How many tricks team actually won.
    bid_met: bool           # Did the team meet the bid?
    overbid: bool           # Did the human bid more than they should?
    underbid: bool          # Did the human bid less than optimal?


@dataclass
class GameResult:
    """A completed game result."""
    timestamp: float
    won: bool
    score_team1: int
    score_team2: int
    shotas_played: int
    bid_accuracy: float     # % of bids met.
    decision_agreement: float  # % agreement with AI.
    trump_efficiency: float
    void_exploitation: float
    partnership_score: float
    defense_score: float


class PlayerEvaluator:
    """Tracks and evaluates a human player's performance over time."""

    def __init__(self):
        self.elo = 1000.0
        self.games_played = 0
        self.game_history: list[GameResult] = []

        # Current game tracking.
        self._trick_decisions: list[TrickDecision] = []
        self._bid_decisions: list[BidDecision] = []
        self._current_shota = 0
        self._current_trick = 0
        self._tricks_trumped = 0
        self._tricks_total = 0
        self._voids_created = 0
        self._voids_exploited = 0
        self._partner_supported = 0
        self._partner_opportunities = 0
        self._defenses_successful = 0
        self._defense_opportunities = 0

        # Load saved stats.
        self._load()

    # ─── Recording Decisions ─────────────────────────────────────────────────

    def record_trick_decision(self, shota, trick, player_card, ai_card,
                              position, was_trump, created_void,
                              trick_won_by_team, context=""):
        """Record a human card-play decision for evaluation."""
        agreed = (player_card == ai_card)
        decision = TrickDecision(
            shota=shota,
            trick=trick,
            player_card=str(player_card),
            ai_would_play=str(ai_card),
            agreed=agreed,
            outcome_good=trick_won_by_team,
            position=position,
            was_trump=was_trump,
            created_void=created_void,
            context=context,
        )
        self._trick_decisions.append(decision)
        self._tricks_total += 1
        if was_trump and position > 0:  # Trumping (not leading trump).
            self._tricks_trumped += 1
        if created_void:
            self._voids_created += 1

    def record_bid_decision(self, shota, player_bid, ai_bid,
                            actual_tricks, bid_met):
        """Record a bidding decision."""
        overbid = player_bid > actual_tricks if player_bid > 0 else False
        underbid = (player_bid < ai_bid and player_bid > 0 and
                    actual_tricks >= ai_bid)
        decision = BidDecision(
            shota=shota,
            player_bid=player_bid,
            ai_would_bid=ai_bid,
            actual_tricks=actual_tricks,
            bid_met=bid_met,
            overbid=overbid,
            underbid=underbid,
        )
        self._bid_decisions.append(decision)

    def record_partner_support(self, supported: bool):
        """Record whether player supported partner's lead."""
        self._partner_opportunities += 1
        if supported:
            self._partner_supported += 1

    def record_defense(self, successful: bool):
        """Record a defensive play opportunity."""
        self._defense_opportunities += 1
        if successful:
            self._defenses_successful += 1

    def record_void_exploited(self):
        """Record that a void was used to trump."""
        self._voids_exploited += 1


    # ─── End of Game Analysis ────────────────────────────────────────────────

    def finish_game(self, won: bool, score_team1: int, score_team2: int,
                    shotas_played: int) -> dict:
        """
        Finalize a game. Compute all metrics and return a full analysis.
        Returns dict with all evaluation results.
        """
        # Compute metrics.
        bid_accuracy = self._compute_bid_accuracy()
        decision_agreement = self._compute_decision_agreement()
        trump_eff = self._compute_trump_efficiency()
        void_score = self._compute_void_score()
        partner_score = self._compute_partnership_score()
        defense_score = self._compute_defense_score()

        # Update Elo.
        self._update_elo(won)
        self.games_played += 1

        # Create result.
        result = GameResult(
            timestamp=time.time(),
            won=won,
            score_team1=score_team1,
            score_team2=score_team2,
            shotas_played=shotas_played,
            bid_accuracy=bid_accuracy,
            decision_agreement=decision_agreement,
            trump_efficiency=trump_eff,
            void_exploitation=void_score,
            partnership_score=partner_score,
            defense_score=defense_score,
        )
        self.game_history.append(result)

        # Build analysis.
        analysis = {
            "won": won,
            "elo": round(self.elo),
            "elo_change": round(self.elo - (self.elo - self._last_elo_change), 1),
            "games_played": self.games_played,
            "bid_accuracy": round(bid_accuracy * 100),
            "decision_agreement": round(decision_agreement * 100),
            "trump_efficiency": round(trump_eff * 100),
            "void_exploitation": round(void_score * 100),
            "partnership_score": round(partner_score * 100),
            "defense_score": round(defense_score * 100),
            "turning_points": self._find_turning_points(),
            "strengths": self._identify_strengths(),
            "weaknesses": self._identify_weaknesses(),
            "improvement": self._compute_improvement(),
        }

        # Reset current game tracking.
        self._reset_game()

        # Save.
        self._save()

        return analysis

    # ─── Metric Computation ──────────────────────────────────────────────────

    def _compute_bid_accuracy(self) -> float:
        """Fraction of bids that were met."""
        if not self._bid_decisions:
            return 0.0
        bids_made = [b for b in self._bid_decisions if b.player_bid > 0]
        if not bids_made:
            return 1.0  # All passes is "accurate" in a sense.
        met = sum(1 for b in bids_made if b.bid_met)
        return met / len(bids_made)

    def _compute_decision_agreement(self) -> float:
        """Fraction of plays that matched what AI would do."""
        if not self._trick_decisions:
            return 0.0
        agreed = sum(1 for d in self._trick_decisions if d.agreed)
        return agreed / len(self._trick_decisions)

    def _compute_trump_efficiency(self) -> float:
        """How well trump was used — trumping at good moments."""
        trump_plays = [d for d in self._trick_decisions if d.was_trump]
        if not trump_plays:
            return 0.5  # Neutral.
        good_trumps = sum(1 for d in trump_plays if d.outcome_good)
        return good_trumps / len(trump_plays)

    def _compute_void_score(self) -> float:
        """How well voids were created and exploited."""
        if self._voids_created == 0:
            return 0.0
        return min(1.0, self._voids_exploited / max(self._voids_created, 1))

    def _compute_partnership_score(self) -> float:
        """How well player supported partner."""
        if self._partner_opportunities == 0:
            return 0.5
        return self._partner_supported / self._partner_opportunities

    def _compute_defense_score(self) -> float:
        """How well player defended against opponent bids."""
        if self._defense_opportunities == 0:
            return 0.5
        return self._defenses_successful / self._defense_opportunities


    # ─── Elo Rating ──────────────────────────────────────────────────────────

    def _update_elo(self, won: bool):
        """Update Elo rating based on game result."""
        k = 32  # Standard K-factor.
        opponent_elo = 1200  # AI is assumed to be ~1200 strength.
        expected = 1.0 / (1.0 + 10 ** ((opponent_elo - self.elo) / 400))
        actual = 1.0 if won else 0.0
        self._last_elo_change = k * (actual - expected)
        self.elo += self._last_elo_change

    _last_elo_change = 0.0

    # ─── Post-Game Analysis ──────────────────────────────────────────────────

    def _find_turning_points(self) -> list:
        """Find the 2-3 most impactful decisions in the game."""
        turning_points = []
        for d in self._trick_decisions:
            if not d.agreed and not d.outcome_good:
                # Disagreed with AI AND lost the trick — potential mistake.
                turning_points.append({
                    "shota": d.shota,
                    "trick": d.trick,
                    "played": d.player_card,
                    "ai_choice": d.ai_would_play,
                    "context": d.context,
                    "type": "mistake",
                })
            elif not d.agreed and d.outcome_good:
                # Disagreed with AI but WON — player outplayed AI!
                turning_points.append({
                    "shota": d.shota,
                    "trick": d.trick,
                    "played": d.player_card,
                    "ai_choice": d.ai_would_play,
                    "context": d.context,
                    "type": "brilliance",
                })

        # Return top 3 most interesting (brilliance first, then mistakes).
        brilliance = [tp for tp in turning_points if tp["type"] == "brilliance"]
        mistakes = [tp for tp in turning_points if tp["type"] == "mistake"]
        return (brilliance[:2] + mistakes[:2])[:3]

    def _identify_strengths(self) -> list:
        """Identify what the player is good at."""
        strengths = []
        if self._compute_bid_accuracy() >= 0.7:
            strengths.append("Accurate bidding — you know your hand's limits")
        if self._compute_trump_efficiency() >= 0.7:
            strengths.append("Smart trump usage — you time your trumps well")
        if self._compute_void_score() >= 0.6:
            strengths.append("Void exploitation — you create and use voids effectively")
        if self._compute_partnership_score() >= 0.7:
            strengths.append("Team player — you support your partner's strategy")
        if self._compute_defense_score() >= 0.7:
            strengths.append("Strong defense — you shut down opponent bids")
        if self._compute_decision_agreement() >= 0.7:
            strengths.append("Expert-level play — your decisions match the AI's strategy")
        return strengths

    def _identify_weaknesses(self) -> list:
        """Identify areas for improvement."""
        weaknesses = []
        if self._compute_bid_accuracy() < 0.5:
            weaknesses.append("Overbidding — try bidding 1-2 less than you think")
        if self._compute_trump_efficiency() < 0.4:
            weaknesses.append("Trump waste — save trumps for when you can't follow suit")
        if self._compute_void_score() < 0.3:
            weaknesses.append("Missed voids — try emptying short suits early")
        if self._compute_partnership_score() < 0.4:
            weaknesses.append("Solo play — help your partner by supporting their leads")
        if self._compute_defense_score() < 0.4:
            weaknesses.append("Weak defense — focus on stopping opponent bids")
        # Overbid detection.
        overbids = sum(1 for b in self._bid_decisions if b.overbid)
        if overbids >= 2:
            weaknesses.append(f"You overbid {overbids} times — your reach exceeds your hand")
        return weaknesses

    def _compute_improvement(self) -> dict:
        """Compare recent games to earlier ones for improvement tracking."""
        if len(self.game_history) < 3:
            return {"trend": "not enough data", "details": []}

        recent = self.game_history[-3:]
        earlier = self.game_history[:-3] if len(self.game_history) > 3 else []

        if not earlier:
            return {"trend": "still early", "details": []}

        recent_win = sum(1 for g in recent if g.won) / len(recent)
        earlier_win = sum(1 for g in earlier if g.won) / len(earlier)

        recent_bid = sum(g.bid_accuracy for g in recent) / len(recent)
        earlier_bid = sum(g.bid_accuracy for g in earlier) / len(earlier)

        details = []
        if recent_win > earlier_win + 0.1:
            details.append("Win rate improving!")
        elif recent_win < earlier_win - 0.1:
            details.append("Win rate declining — opponents getting tougher?")

        if recent_bid > earlier_bid + 0.1:
            details.append("Bidding accuracy improved")
        elif recent_bid < earlier_bid - 0.1:
            details.append("Bidding getting riskier")

        trend = "improving" if recent_win > earlier_win else ("declining" if recent_win < earlier_win else "stable")
        return {"trend": trend, "details": details}


    # ─── Quick Summary ───────────────────────────────────────────────────────

    def get_live_summary(self) -> dict:
        """Get current in-game stats for live display."""
        return {
            "elo": round(self.elo),
            "games_played": self.games_played,
            "current_agreement": round(self._compute_decision_agreement() * 100),
            "current_bid_accuracy": round(self._compute_bid_accuracy() * 100),
            "tricks_decided": len(self._trick_decisions),
            "voids_created": self._voids_created,
            "win_rate": self._compute_win_rate(),
        }

    def _compute_win_rate(self) -> int:
        """Win rate over all games."""
        if not self.game_history:
            return 0
        wins = sum(1 for g in self.game_history if g.won)
        return round(wins / len(self.game_history) * 100)

    # ─── Persistence ─────────────────────────────────────────────────────────

    def _reset_game(self):
        """Reset per-game tracking."""
        self._trick_decisions.clear()
        self._bid_decisions.clear()
        self._current_shota = 0
        self._current_trick = 0
        self._tricks_trumped = 0
        self._tricks_total = 0
        self._voids_created = 0
        self._voids_exploited = 0
        self._partner_supported = 0
        self._partner_opportunities = 0
        self._defenses_successful = 0
        self._defense_opportunities = 0

    def _save(self):
        """Save player stats to disk."""
        try:
            data = {
                "elo": self.elo,
                "games_played": self.games_played,
                "game_history": [
                    {
                        "timestamp": g.timestamp,
                        "won": g.won,
                        "score_team1": g.score_team1,
                        "score_team2": g.score_team2,
                        "shotas_played": g.shotas_played,
                        "bid_accuracy": g.bid_accuracy,
                        "decision_agreement": g.decision_agreement,
                        "trump_efficiency": g.trump_efficiency,
                        "void_exploitation": g.void_exploitation,
                        "partnership_score": g.partnership_score,
                        "defense_score": g.defense_score,
                    }
                    for g in self.game_history[-50:]  # Keep last 50 games.
                ],
            }
            with open(_SAVE_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load(self):
        """Load player stats from disk."""
        try:
            if os.path.exists(_SAVE_PATH):
                with open(_SAVE_PATH, "r") as f:
                    data = json.load(f)
                self.elo = data.get("elo", 1000.0)
                self.games_played = data.get("games_played", 0)
                for g in data.get("game_history", []):
                    self.game_history.append(GameResult(
                        timestamp=g.get("timestamp", 0),
                        won=g.get("won", False),
                        score_team1=g.get("score_team1", 0),
                        score_team2=g.get("score_team2", 0),
                        shotas_played=g.get("shotas_played", 0),
                        bid_accuracy=g.get("bid_accuracy", 0),
                        decision_agreement=g.get("decision_agreement", 0),
                        trump_efficiency=g.get("trump_efficiency", 0.5),
                        void_exploitation=g.get("void_exploitation", 0),
                        partnership_score=g.get("partnership_score", 0.5),
                        defense_score=g.get("defense_score", 0.5),
                    ))
        except Exception:
            pass
