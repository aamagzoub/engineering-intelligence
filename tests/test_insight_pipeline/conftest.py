"""Shared fixtures for insight pipeline tests.

Provides:
    - mock_agent: A mock discovery agent with populated Q-tables (play_q, bid_q)
    - sample_observations: Sample RawObservation instances for testing
    - sample_patterns: Sample RepeatedPattern instances
    - sample_insights: Sample StrategicInsight instances
    - tmp_json_dir: A temporary directory fixture for JSON file I/O tests
"""

import time
from collections import defaultdict
from pathlib import Path

import pytest

from agents.wist_discovery.insight_pipeline.schema import (
    RawObservation,
    RepeatedPattern,
    StrategicInsight,
    VALID_CATEGORIES,
)


# =============================================================================
# Mock Agent Fixture
# =============================================================================


class MockDiscoveryAgent:
    """Simulates the WistDiscoveryAgent for testing the insight pipeline.

    Provides the same Q-table interface as the real agent:
        - play_q: dict[str, dict[str, float]] mapping state strings to action→Q-value
        - bid_q: dict[str, dict[str, float]] mapping bid state strings to action→Q-value
    """

    def __init__(self, play_q: dict, bid_q: dict, episodes_trained: int = 100000):
        self.play_q = play_q
        self.bid_q = bid_q
        self.episodes_trained = episodes_trained


def _build_play_q(num_entries: int = 120) -> dict[str, dict[str, float]]:
    """Build a realistic play_q table with enough entries for snapshot testing.

    State encoding follows discovery_agent format:
        - hand_size (1-13)
        - position (0-3)
        - led_suit (0-3 or 'x' if leading)
        - my_tricks (0-13)
        - opp_tricks (0-13)

    Action encoding: "{rank_hex}{suit_idx}{is_highest}"
    """
    play_q: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    hand_sizes = range(1, 14)
    positions = range(4)
    led_suits = ["0", "1", "2", "3", "x"]

    entry_count = 0
    for hand_size in hand_sizes:
        for position in positions:
            for led_suit in led_suits:
                if entry_count >= num_entries:
                    break
                state = f"{hand_size}_{position}_{led_suit}_2_1"
                # Generate several actions per state
                for rank_hex in ["2", "5", "9", "c", "e"]:  # 2, 5, 9, Q, A
                    for suit_idx in range(4):
                        is_highest = "1" if rank_hex in ("c", "e") else "0"
                        action = f"{rank_hex}{suit_idx}{is_highest}"
                        # Assign Q-values that reflect some learned preferences
                        base_q = (int(rank_hex, 16) - 8) * 0.3
                        # Position bonus: leading (pos 0) gets slightly higher Q
                        if position == 0:
                            base_q += 0.5
                        # Late game bonus
                        if hand_size <= 4:
                            base_q += 0.8
                        # Trump bonus (suit 0 = spades as trump)
                        if suit_idx == 0:
                            base_q += 0.4
                        play_q[state][action] = base_q
                entry_count += 1
    return dict(play_q)


def _build_bid_q(num_entries: int = 80) -> dict[str, dict[str, float]]:
    """Build a realistic bid_q table.

    State encoding follows discovery_agent format:
        - has_bid: Y/N
        - bid_level: 0-13
        - short_suits: count of suits with 1-4 cards

    Action encoding: "PASS" or "B{level}"
    """
    bid_q: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    entry_count = 0
    for has_bid in ["Y", "N"]:
        for bid_level in range(0, 14):
            for short_suits in range(5):
                if entry_count >= num_entries:
                    break
                state = f"{has_bid}{bid_level}s{short_suits}"
                # PASS action
                bid_q[state]["PASS"] = -0.5 if short_suits >= 2 else 0.3
                # Bid actions (7 through 13)
                for bid_val in range(7, 14):
                    action = f"B{bid_val}"
                    # Higher bids are riskier
                    q_val = (13 - bid_val) * 0.2 - (0.3 if has_bid == "Y" else 0.0)
                    # More short suits = better for bidding
                    q_val += short_suits * 0.15
                    bid_q[state][action] = q_val
                entry_count += 1
    return dict(bid_q)


@pytest.fixture
def mock_agent() -> MockDiscoveryAgent:
    """A mock agent with populated Q-tables (at least 100 entries each).

    play_q: ~120 state entries, each with multiple card-play actions.
    bid_q: ~80 state entries, each with PASS + bid level actions.
    """
    return MockDiscoveryAgent(
        play_q=_build_play_q(num_entries=120),
        bid_q=_build_bid_q(num_entries=80),
        episodes_trained=100000,
    )


# =============================================================================
# Sample RawObservation Fixtures
# =============================================================================


@pytest.fixture
def sample_observations() -> list[RawObservation]:
    """A set of sample RawObservation instances spanning different categories/phases."""
    now = time.time()
    return [
        RawObservation(
            category="leading",
            game_phase="late",
            dimension_key="leading_high_card_late_phase",
            reward_direction="positive",
            state_context={"hand_size": 3, "position": 0, "trump_played": True},
            episode=50000,
            snapshot_id="50000",
            timestamp=now - 1000,
        ),
        RawObservation(
            category="leading",
            game_phase="late",
            dimension_key="leading_high_card_late_phase",
            reward_direction="positive",
            state_context={"hand_size": 2, "position": 0, "trump_played": False},
            episode=55000,
            snapshot_id="50000",
            timestamp=now - 900,
        ),
        RawObservation(
            category="leading",
            game_phase="late",
            dimension_key="leading_high_card_late_phase",
            reward_direction="positive",
            state_context={"hand_size": 4, "position": 0, "trump_played": True},
            episode=60000,
            snapshot_id="60000",
            timestamp=now - 800,
        ),
        RawObservation(
            category="following",
            game_phase="mid",
            dimension_key="following_low_card_mid_phase",
            reward_direction="negative",
            state_context={"hand_size": 7, "position": 2, "trump_played": False},
            episode=51000,
            snapshot_id="50000",
            timestamp=now - 950,
        ),
        RawObservation(
            category="bidding",
            game_phase="early",
            dimension_key="bid_conservative_short_suits",
            reward_direction="positive",
            state_context={"short_suits": 3, "hand_strength": "medium"},
            episode=52000,
            snapshot_id="50000",
            timestamp=now - 920,
        ),
        RawObservation(
            category="trump_management",
            game_phase="mid",
            dimension_key="trump_lead_forces_follow",
            reward_direction="positive",
            state_context={"hand_size": 6, "position": 0, "trumps_remaining": 3},
            episode=53000,
            snapshot_id="50000",
            timestamp=now - 880,
        ),
    ]


# =============================================================================
# Sample RepeatedPattern Fixtures
# =============================================================================


@pytest.fixture
def sample_patterns(sample_observations) -> list[RepeatedPattern]:
    """Sample RepeatedPattern instances built from observations."""
    # Pattern with enough evidence for promotion
    leading_obs = [obs for obs in sample_observations if obs.category == "leading"]
    promotable_pattern = RepeatedPattern(
        category="leading",
        dimension_key="leading_high_card_late_phase",
        observations=leading_obs,
        observation_count=len(leading_obs),
        distinct_states=3,
        distinct_snapshots=2,
        confidence=0.65,
        contradicting_count=1,
        stage="pattern",
    )

    # Pattern below promotion threshold
    below_threshold_pattern = RepeatedPattern(
        category="following",
        dimension_key="following_low_card_mid_phase",
        observations=[sample_observations[3]],
        observation_count=1,
        distinct_states=1,
        distinct_snapshots=1,
        confidence=0.15,
        contradicting_count=0,
        stage="pattern",
    )

    # Surprising pattern candidate (needs higher evidence)
    surprising_pattern = RepeatedPattern(
        category="surprising_pattern",
        dimension_key="low_card_beats_high_late",
        observations=leading_obs * 4,  # 12 observations
        observation_count=12,
        distinct_states=5,
        distinct_snapshots=3,
        confidence=0.72,
        contradicting_count=3,
        stage="pattern",
    )

    return [promotable_pattern, below_threshold_pattern, surprising_pattern]


# =============================================================================
# Sample StrategicInsight Fixtures
# =============================================================================


@pytest.fixture
def sample_insights() -> list[StrategicInsight]:
    """Sample StrategicInsight instances representing valid promoted strategies."""
    return [
        StrategicInsight(
            strategy="In late-game positions, leading with high cards forces opponents to spend their remaining trump",
            category="leading",
            tags=["leading", "endgame", "trump_management"],
            confidence=0.72,
            evidence_count=47,
            why="Observed across 47 states: leading high in late phase wins 78% vs 45% for low leads (effect: +33%)",
            first_seen=50000,
            last_confirmed=120000,
            new=False,
        ),
        StrategicInsight(
            strategy="Conservative bidding with multiple short suits reduces the risk of failing the contract",
            category="bidding",
            tags=["bidding", "risk"],
            confidence=0.58,
            evidence_count=23,
            why="Across 23 episodes: bids with 3+ short suits met contract 81% vs 54% for aggressive bids",
            first_seen=30000,
            last_confirmed=95000,
            new=False,
        ),
        StrategicInsight(
            strategy="When partner is winning the current trick, playing low preserves strong cards for future tricks",
            category="partner_play",
            tags=["partner_play", "card_preservation"],
            confidence=0.45,
            evidence_count=15,
            why="15 observations show playing low when partner wins yields +0.6 mean Q-value vs high play",
            first_seen=60000,
            last_confirmed=110000,
            new=True,
        ),
    ]


# =============================================================================
# Temporary Directory Fixture for JSON I/O Tests
# =============================================================================


@pytest.fixture
def tmp_json_dir(tmp_path) -> Path:
    """Provide a temporary directory pre-configured for JSON file I/O tests.

    Creates subdirectory structure mirroring the agent's data directory:
        tmp_path/
            insights_cache.json    (empty list)
            strategy_evidence.json (empty structure)
            strategy_snapshots.json (empty dict)
    """
    # Create empty JSON files matching expected structure
    insights_path = tmp_path / "insights_cache.json"
    insights_path.write_text("[]", encoding="utf-8")

    evidence_path = tmp_path / "strategy_evidence.json"
    evidence_path.write_text(
        '{"patterns": [], "raw_observations": [], "metadata": {}}',
        encoding="utf-8",
    )

    snapshots_path = tmp_path / "strategy_snapshots.json"
    snapshots_path.write_text("{}", encoding="utf-8")

    return tmp_path
