"""Unit tests for tibrain.policy module.

Validates: Requirements 6.1, 6.2, 6.3, 6.5
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from tibrain.policy import Policy


class TestEpsilonGreedySelection:
    """Tests for epsilon-greedy action selection (Requirement 6.1)."""

    def test_epsilon_one_picks_random_over_many_trials(self) -> None:
        """With epsilon=1.0, all actions should be selected over many trials."""
        policy = Policy(epsilon=1.0)
        actions = ["a", "b", "c", "d", "e"]
        q_values = {"a": 10.0, "b": 1.0, "c": 2.0, "d": 3.0, "e": 4.0}

        counts: Counter[str] = Counter()
        n_trials = 5000
        for _ in range(n_trials):
            chosen = policy.select(q_values, actions)
            counts[chosen] += 1

        # All actions should be picked at least once
        for action in actions:
            assert counts[action] > 0, f"Action {action} was never selected"

        # Each action should be roughly 20% — allow wide margin for randomness
        for action in actions:
            proportion = counts[action] / n_trials
            assert 0.10 < proportion < 0.35, (
                f"Action {action} proportion {proportion:.3f} outside expected range"
            )

    def test_epsilon_zero_picks_highest_q_value(self) -> None:
        """With epsilon=0.0, select() should always pick the highest Q-value action."""
        policy = Policy(epsilon=0.0)
        actions = ["a", "b", "c"]
        q_values = {"a": 1.0, "b": 5.0, "c": 3.0}

        # Need at least one total_visit to avoid the random fallback
        # Force a visit so UCB logic activates
        policy._total_visits = 1
        policy._visit_counts = {"a": 1, "b": 1, "c": 1}

        for _ in range(100):
            chosen = policy.select(q_values, actions)
            assert chosen == "b", f"Expected 'b' (highest Q), got '{chosen}'"


class TestUCBBonus:
    """Tests for UCB-inspired exploration bonus (Requirement 6.2)."""

    def test_ucb_gives_priority_to_unvisited_actions(self) -> None:
        """Unvisited actions should be selected immediately in the UCB branch."""
        policy = Policy(epsilon=0.0)
        actions = ["a", "b", "c"]
        q_values = {"a": 10.0, "b": 0.0, "c": 0.0}

        # Simulate some visits to 'a' only
        policy._total_visits = 5
        policy._visit_counts = {"a": 5}

        # With epsilon=0, select uses UCB. 'b' and 'c' have zero visits → priority
        chosen = policy.select(q_values, actions)
        assert chosen in ("b", "c"), (
            f"Expected unvisited action ('b' or 'c'), got '{chosen}'"
        )

    def test_ucb_bonus_influences_selection(self) -> None:
        """UCB bonus should allow less-visited actions to compete with high-Q actions."""
        policy = Policy(epsilon=0.0)
        actions = ["a", "b"]
        # 'a' has high Q but many visits, 'b' has low Q but few visits
        q_values = {"a": 1.0, "b": 0.9}

        # Give 'a' many visits and 'b' very few
        policy._total_visits = 1000
        policy._visit_counts = {"a": 999, "b": 1}

        # UCB bonus for b: 0.15 * sqrt(log(1001) / 1) ≈ 0.15 * 2.63 ≈ 0.39
        # Score for b: 0.9 + 0.39 = 1.29
        # UCB bonus for a: 0.15 * sqrt(log(1001) / 999) ≈ 0.15 * 0.083 ≈ 0.012
        # Score for a: 1.0 + 0.012 = 1.012
        # So 'b' should be selected
        chosen = policy.select(q_values, actions)
        assert chosen == "b", f"Expected 'b' (UCB-boosted), got '{chosen}'"

    def test_ucb_with_zero_total_visits_returns_random(self) -> None:
        """When no visits have occurred, UCB branch should pick randomly."""
        policy = Policy(epsilon=0.0)
        actions = ["a", "b", "c"]
        q_values = {"a": 1.0, "b": 2.0, "c": 3.0}

        # total_visits = 0 triggers random fallback
        counts: Counter[str] = Counter()
        for _ in range(300):
            # Reset state each time to keep total_visits = 0
            policy._total_visits = 0
            policy._visit_counts = {}
            chosen = policy.select(q_values, actions)
            counts[chosen] += 1

        # All actions should be selected at least once
        for action in actions:
            assert counts[action] > 0, f"Action {action} was never selected"


class TestDecay:
    """Tests for epsilon decay with minimum bound (Requirement 6.3)."""

    def test_decay_multiplies_epsilon(self) -> None:
        """Decay should multiply epsilon by the given factor."""
        policy = Policy(epsilon=0.5, epsilon_min=0.01)
        policy.decay(0.9)
        assert abs(policy.epsilon - 0.45) < 1e-10

    def test_decay_respects_minimum_bound(self) -> None:
        """Epsilon should never go below epsilon_min after decay."""
        policy = Policy(epsilon=0.05, epsilon_min=0.02)
        policy.decay(0.1)  # 0.05 * 0.1 = 0.005, but min is 0.02
        assert policy.epsilon == 0.02

    def test_repeated_decay_floors_at_minimum(self) -> None:
        """Many repeated decays should converge to epsilon_min."""
        policy = Policy(epsilon=1.0, epsilon_min=0.05)
        for _ in range(1000):
            policy.decay(0.99)
        assert policy.epsilon >= policy.epsilon_min
        assert abs(policy.epsilon - policy.epsilon_min) < 0.001

    def test_decay_at_minimum_stays_at_minimum(self) -> None:
        """If epsilon is already at minimum, decay should not change it."""
        policy = Policy(epsilon=0.01, epsilon_min=0.01)
        policy.decay(0.5)
        assert policy.epsilon == 0.01


class TestSelectGreedy:
    """Tests for pure greedy selection."""

    def test_select_greedy_returns_highest_q_value(self) -> None:
        """select_greedy should always return the action with highest Q-value."""
        policy = Policy(epsilon=0.5)
        actions = ["a", "b", "c", "d"]
        q_values = {"a": 1.0, "b": 3.0, "c": 2.0, "d": -1.0}

        for _ in range(50):
            chosen = policy.select_greedy(q_values, actions)
            assert chosen == "b"

    def test_select_greedy_handles_negative_values(self) -> None:
        """select_greedy should work correctly with all negative Q-values."""
        policy = Policy()
        actions = ["x", "y", "z"]
        q_values = {"x": -5.0, "y": -1.0, "z": -3.0}

        chosen = policy.select_greedy(q_values, actions)
        assert chosen == "y"

    def test_select_greedy_with_missing_q_values_defaults_to_zero(self) -> None:
        """Actions missing from q_values should be treated as 0.0."""
        policy = Policy()
        actions = ["a", "b", "c"]
        q_values = {"a": -1.0}  # b, c default to 0.0

        chosen = policy.select_greedy(q_values, actions)
        assert chosen in ("b", "c")


class TestEmptyActions:
    """Tests for empty action list raises ValueError."""

    def test_select_empty_actions_raises(self) -> None:
        """select() with empty action list should raise ValueError."""
        policy = Policy()
        with pytest.raises(ValueError, match="No actions to select from"):
            policy.select({}, [])

    def test_select_greedy_empty_actions_raises(self) -> None:
        """select_greedy() with empty action list should raise ValueError."""
        policy = Policy()
        with pytest.raises(ValueError, match="No actions to select from"):
            policy.select_greedy({}, [])


class TestUnvisitedActions:
    """Tests for unvisited action handling (Requirement 6.5)."""

    def test_all_unvisited_selects_randomly(self) -> None:
        """When all actions are unvisited and epsilon=0, random fallback is used."""
        policy = Policy(epsilon=0.0)
        actions = ["a", "b", "c"]
        q_values = {"a": 0.0, "b": 0.0, "c": 0.0}

        # No visits at all → total_visits=0 → random fallback in UCB
        counts: Counter[str] = Counter()
        for _ in range(300):
            policy._total_visits = 0
            policy._visit_counts = {}
            chosen = policy.select(q_values, actions)
            counts[chosen] += 1

        # All actions should be selected at least once
        for action in actions:
            assert counts[action] > 0, f"Action {action} was never selected"
