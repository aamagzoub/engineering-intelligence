"""Unit tests for tibrain.mcts.MCTSEngine.

Validates: Requirements 9.2, 9.3, 9.4
"""

from __future__ import annotations

import pytest

from tibrain.mcts import MCTSEngine


def deterministic_simulate(state, action):
    """A deterministic simulate_fn for testing.

    Returns a reward equal to the action value (int), terminates immediately.
    This allows predictable scoring: action with highest int value wins.
    """
    reward = float(action)
    return (state + 1, reward, True, [])


def multi_step_simulate(state, action):
    """A simulate_fn that allows one more step before terminating.

    First step returns reward = action * 2, then provides follow-up actions.
    Second step (if action > 0) terminates with reward = action.
    """
    if state == 0:
        # First step: reward is action * 2, provide follow-up actions [1, 2]
        return (1, float(action * 2), False, [1, 2])
    else:
        # Terminal step: reward equals the action
        return (2, float(action), True, [])


class TestSingleActionReturnsImmediately:
    """Test that a single legal action is returned without simulation."""

    def test_choose_action_single_action_returns_it(self) -> None:
        call_count = 0

        def counting_simulate(state, action):
            nonlocal call_count
            call_count += 1
            return (state, 0.0, True, [])

        engine = MCTSEngine(simulate_fn=counting_simulate, num_simulations=100)
        result = engine.choose_action(state=0, legal_actions=["only_action"])

        assert result == "only_action"
        assert call_count == 0, "No simulations should run for single action"

    def test_evaluate_actions_single_action_returns_empty_dict(self) -> None:
        call_count = 0

        def counting_simulate(state, action):
            nonlocal call_count
            call_count += 1
            return (state, 0.0, True, [])

        engine = MCTSEngine(simulate_fn=counting_simulate, num_simulations=50)
        result = engine.evaluate_actions(state=0, legal_actions=["only_action"])

        assert result == {}
        assert call_count == 0, "No simulations should run for single action"

    def test_evaluate_actions_empty_list_returns_empty_dict(self) -> None:
        engine = MCTSEngine(simulate_fn=deterministic_simulate, num_simulations=10)
        result = engine.evaluate_actions(state=0, legal_actions=[])

        assert result == {}


class TestEvaluateActionsNormalization:
    """Test that evaluate_actions returns scores normalized to [0, 1]."""

    def test_scores_are_in_zero_one_range(self) -> None:
        engine = MCTSEngine(simulate_fn=deterministic_simulate, num_simulations=10)
        scores = engine.evaluate_actions(state=0, legal_actions=[1, 2, 3, 4, 5])

        for action, score in scores.items():
            assert 0.0 <= score <= 1.0, (
                f"Score for action {action} is {score}, not in [0, 1]"
            )

    def test_best_action_gets_score_one(self) -> None:
        engine = MCTSEngine(simulate_fn=deterministic_simulate, num_simulations=10)
        scores = engine.evaluate_actions(state=0, legal_actions=[1, 5, 3])

        assert scores[5] == 1.0, "Best action should have normalized score of 1.0"

    def test_worst_action_gets_score_zero(self) -> None:
        engine = MCTSEngine(simulate_fn=deterministic_simulate, num_simulations=10)
        scores = engine.evaluate_actions(state=0, legal_actions=[1, 5, 3])

        assert scores[1] == 0.0, "Worst action should have normalized score of 0.0"

    def test_equal_rewards_produce_zero_scores(self) -> None:
        """When all actions have same reward, scores are all 0.0 (spread=1.0 fallback)."""

        def constant_simulate(state, action):
            return (state + 1, 5.0, True, [])

        engine = MCTSEngine(simulate_fn=constant_simulate, num_simulations=10)
        scores = engine.evaluate_actions(state=0, legal_actions=["a", "b", "c"])

        for score in scores.values():
            assert score == 0.0, "Equal rewards should produce 0.0 scores"

    def test_two_actions_normalize_correctly(self) -> None:
        engine = MCTSEngine(simulate_fn=deterministic_simulate, num_simulations=10)
        scores = engine.evaluate_actions(state=0, legal_actions=[2, 8])

        assert scores[2] == 0.0
        assert scores[8] == 1.0


class TestDeterministicSimulateFn:
    """Test choose_action with deterministic simulate_fn returns best action."""

    def test_choose_action_picks_highest_reward(self) -> None:
        engine = MCTSEngine(simulate_fn=deterministic_simulate, num_simulations=10)
        best = engine.choose_action(state=0, legal_actions=[1, 2, 3, 4, 5])

        assert best == 5, "Should pick action with highest reward"

    def test_choose_action_with_negative_rewards(self) -> None:
        def negative_simulate(state, action):
            return (state + 1, -abs(action), True, [])

        engine = MCTSEngine(simulate_fn=negative_simulate, num_simulations=10)
        best = engine.choose_action(state=0, legal_actions=[-5, -3, -1, -2, -4])

        # -abs(-1) = -1 is the highest reward
        assert best == -1

    def test_choose_action_with_two_actions(self) -> None:
        engine = MCTSEngine(simulate_fn=deterministic_simulate, num_simulations=5)
        best = engine.choose_action(state=0, legal_actions=[10, 20])

        assert best == 20


class TestNumSimulationsOverride:
    """Test num_simulations parameter override works."""

    def test_override_num_simulations_in_choose_action(self) -> None:
        call_count = 0

        def counting_simulate(state, action):
            nonlocal call_count
            call_count += 1
            return (state + 1, float(action), True, [])

        engine = MCTSEngine(simulate_fn=counting_simulate, num_simulations=100)
        # With 3 actions and num_simulations=5, expect 3 * 5 = 15 calls
        engine.choose_action(state=0, legal_actions=[1, 2, 3], num_simulations=5)

        assert call_count == 15, f"Expected 15 simulate calls, got {call_count}"

    def test_override_num_simulations_in_evaluate_actions(self) -> None:
        call_count = 0

        def counting_simulate(state, action):
            nonlocal call_count
            call_count += 1
            return (state + 1, float(action), True, [])

        engine = MCTSEngine(simulate_fn=counting_simulate, num_simulations=100)
        # With 2 actions and num_simulations=7, expect 2 * 7 = 14 calls
        engine.evaluate_actions(state=0, legal_actions=[1, 2], num_simulations=7)

        assert call_count == 14, f"Expected 14 simulate calls, got {call_count}"

    def test_default_num_simulations_used_when_not_overridden(self) -> None:
        call_count = 0

        def counting_simulate(state, action):
            nonlocal call_count
            call_count += 1
            return (state + 1, float(action), True, [])

        engine = MCTSEngine(simulate_fn=counting_simulate, num_simulations=20)
        # With 2 actions and default 20 sims, expect 2 * 20 = 40 calls
        engine.choose_action(state=0, legal_actions=[1, 2])

        assert call_count == 40, f"Expected 40 simulate calls, got {call_count}"
