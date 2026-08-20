"""Behaviour Tracker — tracks what the agent actually does over time.

Instead of reading Q-values, this module observes real gameplay decisions
and computes behaviour frequencies that feed the insight system.

Tracks:
- How often the agent leads high vs low
- How often it plays low when partner is winning
- How often it saves trump for late tricks
- How often it bids conservatively vs aggressively
- How these behaviours change over time (early vs current)
"""

from collections import deque


class BehaviourTracker:
    """Tracks agent behaviour frequencies over time windows."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size

        # Per-trick decisions (rolling window).
        self._lead_high = deque(maxlen=window_size)  # 1=led high, 0=led low
        self._partner_winning_played_low = deque(maxlen=window_size)  # 1=yes, 0=no
        self._trump_trick_number = deque(maxlen=window_size)  # trick # when trump used
        self._follow_high_when_losing = deque(maxlen=window_size)  # 1=played high, 0=low

        # Per-shota decisions.
        self._bid_values = deque(maxlen=window_size)  # actual bid values (7-13 or 0=pass)
        self._bid_met = deque(maxlen=window_size)  # 1=met, 0=failed

        # Early training baseline (first 1000 observations).
        self._early_lead_high = []
        self._early_partner_low = []
        self._early_trump_trick = []
        self._early_captured = False

    def observe_card_play(self, rank: int, is_leading: bool, partner_winning: bool,
                          opponent_winning: bool, trick_number: int, is_trump: bool):
        """Record one card play decision."""
        high_card = rank >= 11  # J, Q, K, A

        if is_leading:
            self._lead_high.append(1 if high_card else 0)
            if not self._early_captured and len(self._early_lead_high) < 200:
                self._early_lead_high.append(1 if high_card else 0)

        if partner_winning:
            self._partner_winning_played_low.append(0 if high_card else 1)
            if not self._early_captured and len(self._early_partner_low) < 200:
                self._early_partner_low.append(0 if high_card else 1)

        if opponent_winning and not is_leading:
            self._follow_high_when_losing.append(1 if high_card else 0)

        if is_trump:
            self._trump_trick_number.append(trick_number)
            if not self._early_captured and len(self._early_trump_trick) < 100:
                self._early_trump_trick.append(trick_number)

        # Capture early baseline after enough observations.
        if not self._early_captured and len(self._early_lead_high) >= 200:
            self._early_captured = True

    def observe_bid(self, bid_value: int, bid_met: bool):
        """Record one bidding decision."""
        self._bid_values.append(bid_value)
        self._bid_met.append(1 if bid_met else 0)

    # === Computed stats ===

    def lead_high_pct(self) -> float:
        """% of times agent leads with high cards (current window)."""
        if not self._lead_high:
            return 0.0
        return sum(self._lead_high) / len(self._lead_high) * 100

    def early_lead_high_pct(self) -> float:
        """% of times agent led with high cards in early training."""
        if not self._early_lead_high:
            return 0.0
        return sum(self._early_lead_high) / len(self._early_lead_high) * 100

    def partner_low_pct(self) -> float:
        """% of times agent plays low when partner is winning."""
        if not self._partner_winning_played_low:
            return 0.0
        return sum(self._partner_winning_played_low) / len(self._partner_winning_played_low) * 100

    def early_partner_low_pct(self) -> float:
        """% from early training."""
        if not self._early_partner_low:
            return 0.0
        return sum(self._early_partner_low) / len(self._early_partner_low) * 100

    def avg_trump_trick(self) -> float:
        """Average trick number when trump is used."""
        if not self._trump_trick_number:
            return 7.0
        return sum(self._trump_trick_number) / len(self._trump_trick_number)

    def early_avg_trump_trick(self) -> float:
        """From early training."""
        if not self._early_trump_trick:
            return 7.0
        return sum(self._early_trump_trick) / len(self._early_trump_trick)

    def follow_high_when_losing_pct(self) -> float:
        """% of times agent plays high when opponent is winning."""
        if not self._follow_high_when_losing:
            return 0.0
        return sum(self._follow_high_when_losing) / len(self._follow_high_when_losing) * 100

    def avg_bid(self) -> float:
        """Average bid value (excluding passes)."""
        bids = [b for b in self._bid_values if b > 0]
        if not bids:
            return 7.0
        return sum(bids) / len(bids)

    def bid_met_pct(self) -> float:
        """% of bids successfully met."""
        if not self._bid_met:
            return 0.0
        return sum(self._bid_met) / len(self._bid_met) * 100

    def has_enough_data(self) -> bool:
        """Whether we have enough observations for meaningful insights."""
        return len(self._lead_high) >= 50 and self._early_captured

    def generate_insights(self, episodes: int) -> list[dict]:
        """Generate all insights based on tracked behaviour.

        Returns list of insight dicts with:
        - text: the main insight sentence
        - category: play/strategy/surprising
        - confidence: 0.0-1.0
        - style: question/comparison/story/stat
        """
        if not self.has_enough_data():
            return []

        insights = []

        # === Story-based: change over time ===
        lead_now = self.lead_high_pct()
        lead_early = self.early_lead_high_pct()
        if abs(lead_now - lead_early) > 15:
            if lead_now < lead_early:
                insights.append({
                    "strategy": (
                        f"The AI changed its mind about leading. Early on it led with high cards "
                        f"{lead_early:.0f}% of the time. Now it leads high only {lead_now:.0f}% — "
                        f"it learned that starting low reveals more before committing strength."
                    ),
                    "category": "leading",
                    "confidence": 0.8,
                    "tags": ["leading", "card_preservation"],
                    "why": f"Lead-high rate dropped from {lead_early:.0f}% to {lead_now:.0f}% over {episodes:,} episodes",
                    "first_seen": episodes,
                    "last_confirmed": episodes,
                    "new": True,
                })
            else:
                insights.append({
                    "strategy": (
                        f"The AI became more aggressive at leading. Early on it led high "
                        f"{lead_early:.0f}% of the time, now {lead_now:.0f}%. "
                        f"It discovered that seizing control early pays off."
                    ),
                    "category": "leading",
                    "confidence": 0.8,
                    "tags": ["leading", "risk"],
                    "why": f"Lead-high rate rose from {lead_early:.0f}% to {lead_now:.0f}% over {episodes:,} episodes",
                    "first_seen": episodes,
                    "last_confirmed": episodes,
                    "new": True,
                })

        # === Comparison: vs naive/random ===
        partner_low = self.partner_low_pct()
        if partner_low > 60:
            insights.append({
                "strategy": (
                    f"Most beginners play their best card every time. This AI plays low "
                    f"{partner_low:.0f}% of the time when its partner is already winning. "
                    f"It learned: don't waste good cards on won tricks."
                ),
                "category": "partner_play",
                "confidence": min(0.95, partner_low / 100),
                "tags": ["partner_play", "card_preservation"],
                "why": f"Plays low when partner winning {partner_low:.0f}% of the time (random would be ~50%)",
                "first_seen": episodes,
                "last_confirmed": episodes,
                "new": True,
            })

        # === Question-style ===
        trump_avg = self.avg_trump_trick()
        if trump_avg > 9:
            insights.append({
                "strategy": (
                    f"The AI waits until trick {trump_avg:.0f} on average before using trump. "
                    f"Would you have the patience to hold trump that long?"
                ),
                "category": "trump_management",
                "confidence": 0.7,
                "tags": ["trump_management", "endgame"],
                "why": f"Average trump usage at trick {trump_avg:.1f} (random would be ~7)",
                "first_seen": episodes,
                "last_confirmed": episodes,
                "new": True,
            })
        elif trump_avg < 5:
            insights.append({
                "strategy": (
                    f"The AI uses trump early — average trick {trump_avg:.0f}. "
                    f"It seems to prefer stealing tricks fast rather than saving for later."
                ),
                "category": "trump_management",
                "confidence": 0.7,
                "tags": ["trump_management", "risk"],
                "why": f"Average trump usage at trick {trump_avg:.1f}",
                "first_seen": episodes,
                "last_confirmed": episodes,
                "new": True,
            })

        # === Stat with story ===
        follow_high = self.follow_high_when_losing_pct()
        if follow_high > 70:
            insights.append({
                "strategy": (
                    f"When an opponent is winning the trick, the AI fights back with high cards "
                    f"{follow_high:.0f}% of the time. It doesn't give up tricks easily."
                ),
                "category": "defense",
                "confidence": 0.75,
                "tags": ["defense", "following"],
                "why": f"Contests losing tricks {follow_high:.0f}% of the time with rank 11+",
                "first_seen": episodes,
                "last_confirmed": episodes,
                "new": True,
            })
        elif follow_high < 30:
            insights.append({
                "strategy": (
                    f"When an opponent is winning, the AI throws low cards {100-follow_high:.0f}% "
                    f"of the time. It learned to cut losses — saving strength for tricks it can win."
                ),
                "category": "card_preservation",
                "confidence": 0.75,
                "tags": ["card_preservation", "defense"],
                "why": f"Gives up on losing tricks {100-follow_high:.0f}% of the time",
                "first_seen": episodes,
                "last_confirmed": episodes,
                "new": True,
            })

        # === Surprising: trump timing change ===
        trump_early = self.early_avg_trump_trick()
        trump_now = self.avg_trump_trick()
        if abs(trump_now - trump_early) > 2:
            if trump_now > trump_early:
                insights.append({
                    "strategy": (
                        f"The AI learned patience with trump. Early in training it used trump "
                        f"around trick {trump_early:.0f}. Now it waits until trick {trump_now:.0f}. "
                        f"Holding trump builds pressure — opponents run out of options."
                    ),
                    "category": "surprising_pattern",
                    "confidence": 0.85,
                    "tags": ["trump_management", "surprising_pattern"],
                    "why": f"Trump timing shifted from trick {trump_early:.1f} to {trump_now:.1f}",
                    "first_seen": episodes,
                    "last_confirmed": episodes,
                    "new": True,
                })

        # === Bidding insight ===
        avg_bid = self.avg_bid()
        bid_met = self.bid_met_pct()
        if bid_met > 75 and avg_bid < 8.5:
            insights.append({
                "strategy": (
                    f"The AI bids conservatively (average {avg_bid:.1f}) and delivers "
                    f"{bid_met:.0f}% of the time. It under-promises and over-delivers."
                ),
                "category": "bidding",
                "confidence": min(0.9, bid_met / 100),
                "tags": ["bidding", "risk"],
                "why": f"Average bid {avg_bid:.1f} with {bid_met:.0f}% success rate",
                "first_seen": episodes,
                "last_confirmed": episodes,
                "new": True,
            })
        elif avg_bid > 9.5 and bid_met > 60:
            insights.append({
                "strategy": (
                    f"The AI bids boldly (average {avg_bid:.1f}) and still delivers "
                    f"{bid_met:.0f}% of the time. It reads hand strength accurately."
                ),
                "category": "bidding",
                "confidence": min(0.9, bid_met / 100),
                "tags": ["bidding", "risk"],
                "why": f"Average bid {avg_bid:.1f} with {bid_met:.0f}% success rate",
                "first_seen": episodes,
                "last_confirmed": episodes,
                "new": True,
            })

        return insights
