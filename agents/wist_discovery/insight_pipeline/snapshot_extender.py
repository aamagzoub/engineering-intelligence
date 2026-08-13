"""SnapshotExtender — Extends strategy snapshots with aggregate behaviour dimensions.

Reads agent Q-tables (play_q, bid_q) without mutation and computes:
- Existing fields: pos_prefs, suit_prefs, bid_prefs, rank_prefs
- 9 new aggregate dimensions (each as {mean_q, count} or None if count < 5):
    1. leading_vs_following
    2. phase_behaviour
    3. card_strength_by_role
    4. trump_vs_nontrump
    5. partner_vs_opponent_winning
    6. suit_length
    7. information_available
    8. bid_strength_reliability
    9. defensive_vs_attacking

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 12.1, 12.2
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


class SnapshotExtender:
    """Extends strategy snapshots with new aggregate behaviour dimensions."""

    MIN_CONTRIBUTING_STATES = 5  # Below this, dimension = null

    def take_extended_snapshot(self, agent) -> dict:
        """Read Q-tables and compute all dimensions (existing + new).

        This method is READ-ONLY — it does not modify the agent's Q-tables,
        neural network weights, experience replay buffer, or hyperparameters.

        Args:
            agent: A WistDiscoveryAgent instance with play_q and bid_q dicts.

        Returns:
            A snapshot dict containing existing fields (pos_prefs, suit_prefs,
            bid_prefs, rank_prefs) and 9 new aggregate dimension keys.
        """
        play_items = list(agent.play_q.items())[:30000]
        bid_items = list(agent.bid_q.items())[:5000]

        # === Existing fields (replicate _take_snapshot logic) ===
        existing = self._compute_existing_fields(play_items, bid_items)

        # === New aggregate dimensions ===
        new_dimensions = self._compute_new_dimensions(play_items, bid_items)

        # Merge existing and new into a single snapshot dict
        snapshot = {}
        snapshot.update(existing)
        snapshot.update(new_dimensions)
        return snapshot

    # =========================================================================
    # Existing Field Computation (backward-compatible)
    # =========================================================================

    def _compute_existing_fields(
        self, play_items: list, bid_items: list
    ) -> dict:
        """Compute pos_prefs, suit_prefs, bid_prefs, rank_prefs.

        Replicates the logic from gui_wist_discovery/insights.py _take_snapshot.
        """
        pos_prefs: dict[Any, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        suit_prefs: dict[int, float] = defaultdict(float)
        bid_prefs: dict[str, float] = defaultdict(float)
        rank_prefs: dict[int, float] = defaultdict(float)
        rank_counts: dict[int, int] = defaultdict(int)

        for state, actions in play_items:
            parsed_state = self._parse_play_state(state)
            if not parsed_state or len(actions) < 2:
                continue
            _n_cards, pos, _led, _my_tricks, _opp_tricks = parsed_state
            for key, q in actions.items():
                parsed_action = self._parse_play_action(key)
                if not parsed_action:
                    continue
                rank, suit, _is_highest = parsed_action
                tier = self._rank_tier(rank)
                pos_prefs[pos][tier] += q
                suit_prefs[suit] += q
                rank_prefs[rank] += q
                rank_counts[rank] += 1

        # Normalize rank_prefs
        for rank in rank_prefs:
            if rank_counts[rank] > 0:
                rank_prefs[rank] /= rank_counts[rank]

        # Bid prefs
        for _state, actions in bid_items:
            for action, q in actions.items():
                bid_prefs[action] += q

        return {
            "pos_prefs": {str(k): dict(v) for k, v in pos_prefs.items()},
            "suit_prefs": dict(suit_prefs),
            "bid_prefs": dict(bid_prefs),
            "rank_prefs": {str(k): v for k, v in rank_prefs.items()},
        }

    # =========================================================================
    # New Aggregate Dimensions
    # =========================================================================

    def _compute_new_dimensions(
        self, play_items: list, bid_items: list
    ) -> dict:
        """Compute the 9 new aggregate behaviour dimensions."""
        # Accumulators for play_q dimensions
        # 1. leading_vs_following
        leading_qs: list[float] = []
        following_qs: list[float] = []

        # 2. phase_behaviour
        early_qs: list[float] = []
        mid_qs: list[float] = []
        late_qs: list[float] = []

        # 3. card_strength_by_role
        leading_high_qs: list[float] = []
        leading_low_qs: list[float] = []
        following_high_qs: list[float] = []
        following_low_qs: list[float] = []

        # 4. trump_vs_nontrump
        trump_qs: list[float] = []
        nontrump_qs: list[float] = []

        # 5. partner_vs_opponent_winning
        partner_winning_qs: list[float] = []
        opponent_winning_qs: list[float] = []

        # 6. suit_length (using hand_size as proxy)
        long_qs: list[float] = []   # hand_size >= 10
        short_qs: list[float] = []  # hand_size 5-9

        # 7. information_available (position = cards seen before us)
        info_0_qs: list[float] = []
        info_1_qs: list[float] = []
        info_2_qs: list[float] = []
        info_3_qs: list[float] = []

        # 9. defensive_vs_attacking
        defensive_qs: list[float] = []
        attacking_qs: list[float] = []

        for state, actions in play_items:
            parsed_state = self._parse_play_state(state)
            if not parsed_state:
                continue
            n_cards, pos, _led, my_tricks, opp_tricks = parsed_state

            for action_key, q in actions.items():
                parsed_action = self._parse_play_action(action_key)
                if not parsed_action:
                    continue
                rank, suit_idx, _is_highest = parsed_action

                # 1. leading_vs_following
                if pos == 0:
                    leading_qs.append(q)
                else:
                    following_qs.append(q)

                # 2. phase_behaviour
                if n_cards >= 10:
                    early_qs.append(q)
                elif n_cards >= 5:
                    mid_qs.append(q)
                else:
                    late_qs.append(q)

                # 3. card_strength_by_role (high rank >= 0xA = 10)
                is_high_rank = rank >= 10
                if pos == 0:
                    if is_high_rank:
                        leading_high_qs.append(q)
                    else:
                        leading_low_qs.append(q)
                else:
                    if is_high_rank:
                        following_high_qs.append(q)
                    else:
                        following_low_qs.append(q)

                # 4. trump_vs_nontrump (suit_idx 0 = trump)
                if suit_idx == 0:
                    trump_qs.append(q)
                else:
                    nontrump_qs.append(q)

                # 5. partner_vs_opponent_winning
                if my_tricks > opp_tricks:
                    partner_winning_qs.append(q)
                elif opp_tricks > my_tricks:
                    opponent_winning_qs.append(q)

                # 6. suit_length (hand_size as proxy)
                if n_cards >= 10:
                    long_qs.append(q)
                elif n_cards >= 5:
                    short_qs.append(q)
                # void won't have states (can't play from empty hand)

                # 7. information_available
                if pos == 0:
                    info_0_qs.append(q)
                elif pos == 1:
                    info_1_qs.append(q)
                elif pos == 2:
                    info_2_qs.append(q)
                elif pos == 3:
                    info_3_qs.append(q)

                # 9. defensive_vs_attacking
                if opp_tricks > my_tricks:
                    defensive_qs.append(q)
                else:
                    attacking_qs.append(q)

        # 8. bid_strength_reliability (from bid_q)
        bid_level_qs: list[float] = []
        pass_positive_count = 0
        pass_total_count = 0

        for _state, actions in bid_items:
            for action_key, q in actions.items():
                if action_key == "PASS":
                    pass_total_count += 1
                    if q > 0:
                        pass_positive_count += 1
                else:
                    # Non-PASS bid actions (B7, B8, ..., B13)
                    bid_level_qs.append(q)

        # Build dimension entries
        return {
            "leading_vs_following": {
                "leading": self._make_dimension_entry(leading_qs),
                "following": self._make_dimension_entry(following_qs),
            },
            "phase_behaviour": {
                "early": self._make_dimension_entry(early_qs),
                "mid": self._make_dimension_entry(mid_qs),
                "late": self._make_dimension_entry(late_qs),
            },
            "card_strength_by_role": {
                "leading_high": self._make_dimension_entry(leading_high_qs),
                "leading_low": self._make_dimension_entry(leading_low_qs),
                "following_high": self._make_dimension_entry(following_high_qs),
                "following_low": self._make_dimension_entry(following_low_qs),
            },
            "trump_vs_nontrump": {
                "trump": self._make_dimension_entry(trump_qs),
                "nontrump": self._make_dimension_entry(nontrump_qs),
            },
            "partner_vs_opponent_winning": {
                "partner_winning": self._make_dimension_entry(
                    partner_winning_qs
                ),
                "opponent_winning": self._make_dimension_entry(
                    opponent_winning_qs
                ),
            },
            "suit_length": {
                "long": self._make_dimension_entry(long_qs),
                "short": self._make_dimension_entry(short_qs),
            },
            "information_available": {
                "0": self._make_dimension_entry(info_0_qs),
                "1": self._make_dimension_entry(info_1_qs),
                "2": self._make_dimension_entry(info_2_qs),
                "3": self._make_dimension_entry(info_3_qs),
            },
            "bid_strength_reliability": self._make_bid_reliability_entry(
                bid_level_qs, pass_positive_count, pass_total_count
            ),
            "defensive_vs_attacking": {
                "defensive": self._make_dimension_entry(defensive_qs),
                "attacking": self._make_dimension_entry(attacking_qs),
            },
        }

    # =========================================================================
    # Dimension Entry Construction
    # =========================================================================

    def _make_dimension_entry(
        self, values: list[float]
    ) -> dict[str, Any] | None:
        """Create a dimension entry {mean_q, count} or None if count < 5.

        Args:
            values: List of Q-values contributing to this dimension.

        Returns:
            Dict with mean_q and count, or None if insufficient data.
        """
        count = len(values)
        if count < self.MIN_CONTRIBUTING_STATES:
            return None
        mean_q = sum(values) / count
        return {"mean_q": mean_q, "count": count}

    def _make_bid_reliability_entry(
        self,
        bid_level_qs: list[float],
        pass_positive_count: int,
        pass_total_count: int,
    ) -> dict[str, Any]:
        """Create the bid_strength_reliability dimension entry.

        Contains:
        - bid_level: mean Q of non-PASS bid actions (or None if < 5)
        - reliability_ratio: ratio of PASS-Q > 0 states

        Args:
            bid_level_qs: Q-values for non-PASS bid actions.
            pass_positive_count: Number of PASS actions with Q > 0.
            pass_total_count: Total number of PASS actions.

        Returns:
            Dict with bid_level entry and reliability_ratio.
        """
        bid_level_entry = self._make_dimension_entry(bid_level_qs)
        reliability_ratio = (
            pass_positive_count / pass_total_count
            if pass_total_count > 0
            else 0.0
        )
        return {
            "bid_level": bid_level_entry,
            "reliability_ratio": reliability_ratio,
        }

    # =========================================================================
    # State/Action Parsing
    # =========================================================================

    def _parse_play_state(self, key: str) -> tuple | None:
        """Parse play_q state key.

        Format: "{hand_size_hex}{position}{led_suit}{my_tricks_hex}{opp_tricks_hex}"
        - hand_size: 1-13 (encoded as single hex digit, 1-d)
        - position: 0-3
        - led_suit: "0","1","2","3" or "x" (leading)
        - my_tricks: 0-13 (hex digit)
        - opp_tricks: 0-13 (hex digit)

        Returns:
            (n_cards, position, led_suit, my_tricks, opp_tricks) or None.
        """
        try:
            if len(key) < 5:
                return None
            n_cards = int(key[0], 16)
            pos = int(key[1])
            led = key[2]
            my_tricks = int(key[3], 16)
            opp_tricks = int(key[4], 16)
            if not (1 <= n_cards <= 13 and 0 <= pos <= 3):
                return None
            if led not in ("0", "1", "2", "3", "x"):
                return None
            return (n_cards, pos, led, my_tricks, opp_tricks)
        except (ValueError, IndexError):
            return None

    def _parse_play_action(self, key: str) -> tuple | None:
        """Parse play_q action key.

        Format: "{rank_hex}{suit_idx}{is_highest}"
        - rank_hex: single hex digit for card rank (2-14 → 2-e)
        - suit_idx: 0-3
        - is_highest: "0" or "1"

        Returns:
            (rank_int, suit_int, is_highest_bool) or None.
        """
        try:
            if len(key) < 3:
                return None
            rank = int(key[0], 16)
            suit = int(key[1])
            is_highest = key[2] == "1"
            if not (2 <= rank <= 14 and 0 <= suit <= 3):
                return None
            return (rank, suit, is_highest)
        except (ValueError, IndexError):
            return None

    # =========================================================================
    # Utilities
    # =========================================================================

    @staticmethod
    def _rank_tier(rank: int) -> str:
        """Classify rank into tiers for insight grouping.

        Args:
            rank: Integer rank value (2-14).

        Returns:
            Tier string: "high", "upper", "mid", or "low".
        """
        if rank >= 13:
            return "high"  # King, Ace
        elif rank >= 11:
            return "upper"  # Jack, Queen
        elif rank >= 8:
            return "mid"  # 8, 9, 10
        else:
            return "low"  # 2-7
