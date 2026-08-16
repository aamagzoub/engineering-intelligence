"""Unit tests for tibrain.q_learning.QLearningEngine."""

import random

import pytest

from tibrain.q_learning import QLearningEngine


# --- Requirement 4.4: td_update produces correct Q-value changes ---


class TestTdUpdate:
    def test_td_update_changes_q_value_in_expected_direction(self):
        """A positive reward on a zero-initialized table should increase Q-values."""
        engine = QLearningEngine(alpha=0.1, gamma=0.9, lambda_trace=0.0)
        random.seed(42)

        engine.td_update("s1", "a1", 1.0, "s2", ["a1", "a2"])

        # The value for (s1, a1) should have increased from 0.0
        # One of the two tables will have been updated
        q1_val = engine.q1.get("s1", "a1")
        q2_val = engine.q2.get("s1", "a1")
        # At least one table should have a positive update
        assert q1_val > 0.0 or q2_val > 0.0

    def test_td_update_negative_reward_decreases_value(self):
        """A negative reward on a zero-initialized table should decrease Q-values."""
        engine = QLearningEngine(alpha=0.1, gamma=0.9, lambda_trace=0.0)
        random.seed(42)

        engine.td_update("s1", "a1", -1.0, "s2", ["a1", "a2"])

        q1_val = engine.q1.get("s1", "a1")
        q2_val = engine.q2.get("s1", "a1")
        # At least one table should have a negative update
        assert q1_val < 0.0 or q2_val < 0.0

    def test_td_update_returns_td_error(self):
        """td_update returns the TD error value."""
        engine = QLearningEngine(alpha=0.1, gamma=0.9, lambda_trace=0.0)
        random.seed(42)

        td_error = engine.td_update("s1", "a1", 1.0, "s2", ["a1"])
        # TD error = reward + gamma * max_next_q - current_q
        # = 1.0 + 0.9 * 0.0 - 0.0 = 1.0
        assert td_error == pytest.approx(1.0)

    def test_td_update_with_terminal_state(self):
        """When next_actions is empty (terminal), max_next_q is 0."""
        engine = QLearningEngine(alpha=0.5, gamma=0.9, lambda_trace=0.0)
        random.seed(0)

        td_error = engine.td_update("s1", "a1", 5.0, "terminal", [])
        # TD error = 5.0 + 0.9 * 0.0 - 0.0 = 5.0
        assert td_error == pytest.approx(5.0)

    def test_td_update_uses_alpha_as_learning_rate(self):
        """The magnitude of the update is scaled by alpha."""
        engine = QLearningEngine(alpha=0.5, gamma=0.0, lambda_trace=0.0)
        random.seed(10)

        engine.td_update("s1", "a1", 2.0, "s2", ["a1"])
        # TD error = 2.0, update = alpha * td_error * trace = 0.5 * 2.0 * 1.0 = 1.0
        q1_val = engine.q1.get("s1", "a1")
        q2_val = engine.q2.get("s1", "a1")
        updated_val = max(q1_val, q2_val)
        assert updated_val == pytest.approx(1.0)

    def test_td_update_uses_gamma_for_next_state(self):
        """Gamma discounts the next state's value."""
        engine = QLearningEngine(alpha=1.0, gamma=0.5, lambda_trace=0.0)
        # Seed q1 with a known value in next state
        engine.q1.set("s2", "a1", 4.0)
        engine.q2.set("s2", "a1", 4.0)
        random.seed(42)

        td_error = engine.td_update("s1", "a1", 0.0, "s2", ["a1"])
        # TD error = 0.0 + 0.5 * 4.0 - 0.0 = 2.0
        assert td_error == pytest.approx(2.0)


# --- Requirement 4.2: Eligibility traces propagate credit ---


class TestEligibilityTraces:
    def test_traces_propagate_to_earlier_state_action_pairs(self):
        """After multiple steps, earlier states should receive updates via traces."""
        engine = QLearningEngine(alpha=0.1, gamma=0.9, lambda_trace=0.9)
        random.seed(42)

        # Step 1: visit (s1, a1)
        engine.td_update("s1", "a1", 0.0, "s2", ["a1", "a2"])
        # Step 2: visit (s2, a2) — s1,a1 should still get credit through traces
        engine.td_update("s2", "a2", 10.0, "s3", ["a1"])

        # (s1, a1) should have received some credit from the second update
        combined_s1_a1 = engine.q1.get("s1", "a1") + engine.q2.get("s1", "a1")
        assert combined_s1_a1 != 0.0, "Earlier state-action should receive credit via traces"

    def test_trace_decay_after_update(self):
        """Traces decay by gamma * lambda_trace after each update."""
        engine = QLearningEngine(alpha=0.1, gamma=0.9, lambda_trace=0.8)
        random.seed(42)

        # First update creates trace for (s1, a1) = 1.0
        engine.td_update("s1", "a1", 0.0, "s2", ["a1"])

        # After the update, trace should have been decayed by gamma * lambda
        # Expected decayed trace: 1.0 * 0.9 * 0.8 = 0.72
        if ("s1", "a1") in engine._traces:
            assert engine._traces[("s1", "a1")] == pytest.approx(0.72)

    def test_lambda_zero_means_no_trace_propagation(self):
        """With lambda_trace=0, only the current state-action gets updated."""
        engine = QLearningEngine(alpha=0.5, gamma=0.9, lambda_trace=0.0)
        random.seed(42)

        # Visit s1 then s2
        engine.td_update("s1", "a1", 0.0, "s2", ["a1"])
        # After this step with lambda=0, trace for (s1,a1) should be decayed to 0 and removed
        # because gamma * lambda * trace = 0.9 * 0.0 * 1.0 = 0.0 < 0.01
        assert ("s1", "a1") not in engine._traces


# --- Requirement 4.6: reset_episode clears all traces ---


class TestResetEpisode:
    def test_reset_episode_clears_traces(self):
        """reset_episode() empties the eligibility trace dictionary."""
        engine = QLearningEngine(alpha=0.1, gamma=0.9, lambda_trace=0.9)
        random.seed(42)

        # Build up some traces
        engine.td_update("s1", "a1", 1.0, "s2", ["a1"])
        engine.td_update("s2", "a1", 1.0, "s3", ["a1"])
        assert len(engine._traces) > 0, "Traces should exist before reset"

        engine.reset_episode()
        assert engine._traces == {}

    def test_reset_episode_does_not_affect_q_tables(self):
        """reset_episode() only clears traces, not learned Q-values."""
        engine = QLearningEngine(alpha=0.5, gamma=0.9, lambda_trace=0.5)
        random.seed(42)

        engine.td_update("s1", "a1", 5.0, "s2", ["a1"])
        q1_before = engine.q1.to_dict()
        q2_before = engine.q2.to_dict()

        engine.reset_episode()

        assert engine.q1.to_dict() == q1_before
        assert engine.q2.to_dict() == q2_before

    def test_reset_episode_allows_fresh_trace_accumulation(self):
        """After reset, a new td_update starts traces from scratch."""
        engine = QLearningEngine(alpha=0.1, gamma=0.9, lambda_trace=0.9)
        random.seed(42)

        engine.td_update("s1", "a1", 1.0, "s2", ["a1"])
        engine.reset_episode()

        # New episode: only (s3, a1) should be tracked
        engine.td_update("s3", "a1", 1.0, "s4", ["a1"])
        # s1,a1 should NOT be in traces anymore
        assert ("s1", "a1") not in engine._traces


# --- Requirement 4.1: Double Q-learning with two Q-tables ---


class TestDoubleQLearning:
    def test_both_tables_receive_updates_over_many_calls(self):
        """Over many updates, both Q1 and Q2 should be updated."""
        engine = QLearningEngine(alpha=0.1, gamma=0.9, lambda_trace=0.0)

        # Run many updates to ensure both tables get updated
        for i in range(100):
            random.seed(i)  # Different seed each time for different table selection
            engine.td_update("s1", "a1", 1.0, "s2", ["a1"])

        # Both tables should have non-zero values
        assert engine.q1.get("s1", "a1") != 0.0, "Q1 should have received updates"
        assert engine.q2.get("s1", "a1") != 0.0, "Q2 should have received updates"

    def test_double_q_uses_different_table_for_eval(self):
        """Double Q selects best action from one table but evaluates with the other."""
        engine = QLearningEngine(alpha=1.0, gamma=1.0, lambda_trace=0.0)

        # Set up known values in tables
        engine.q1.set("s2", "a1", 10.0)
        engine.q1.set("s2", "a2", 5.0)
        engine.q2.set("s2", "a1", 2.0)
        engine.q2.set("s2", "a2", 8.0)

        # With seed(0), random.random() ≈ 0.844 (>= 0.5)
        # so q_update=q2, q_eval=q1
        random.seed(0)
        td_error = engine.td_update("s1", "a1", 0.0, "s2", ["a1", "a2"])

        # q2 selects best action in s2: a2 has 8.0 > a1 has 2.0 → selects a2
        # q1 evaluates a2: q1.get("s2", "a2") = 5.0
        # TD error = 0.0 + 1.0 * 5.0 - 0.0 = 5.0
        assert td_error == pytest.approx(5.0)

    def test_get_values_averages_both_tables(self):
        """get_values returns the average of Q1 and Q2."""
        engine = QLearningEngine()
        engine.q1.set("s1", "a1", 4.0)
        engine.q2.set("s1", "a1", 6.0)
        engine.q1.set("s1", "a2", 2.0)
        engine.q2.set("s1", "a2", 2.0)

        values = engine.get_values("s1", ["a1", "a2"])
        assert values["a1"] == pytest.approx(5.0)
        assert values["a2"] == pytest.approx(2.0)


# --- Requirement 4.3: Configurable hyperparameters ---


class TestConfigurableHyperparameters:
    def test_custom_alpha(self):
        """Alpha controls learning step size."""
        engine = QLearningEngine(alpha=0.01, gamma=0.0, lambda_trace=0.0)
        random.seed(42)

        engine.td_update("s1", "a1", 10.0, "s2", ["a1"])
        # Update = alpha * td_error * trace = 0.01 * 10.0 * 1.0 = 0.1
        q1_val = engine.q1.get("s1", "a1")
        q2_val = engine.q2.get("s1", "a1")
        updated_val = max(q1_val, q2_val)
        assert updated_val == pytest.approx(0.1)

    def test_custom_gamma(self):
        """Gamma controls discount factor for future rewards."""
        engine_low_gamma = QLearningEngine(alpha=1.0, gamma=0.1, lambda_trace=0.0)
        engine_high_gamma = QLearningEngine(alpha=1.0, gamma=0.9, lambda_trace=0.0)

        # Set a known future value
        engine_low_gamma.q1.set("s2", "a1", 10.0)
        engine_low_gamma.q2.set("s2", "a1", 10.0)
        engine_high_gamma.q1.set("s2", "a1", 10.0)
        engine_high_gamma.q2.set("s2", "a1", 10.0)

        random.seed(42)
        td_low = engine_low_gamma.td_update("s1", "a1", 0.0, "s2", ["a1"])
        random.seed(42)
        td_high = engine_high_gamma.td_update("s1", "a1", 0.0, "s2", ["a1"])

        # Higher gamma should produce larger TD error (more weight on future)
        assert abs(td_high) > abs(td_low)

    def test_custom_lambda_trace(self):
        """Lambda_trace controls eligibility trace decay speed."""
        engine_low = QLearningEngine(alpha=0.1, gamma=0.9, lambda_trace=0.1)
        engine_high = QLearningEngine(alpha=0.1, gamma=0.9, lambda_trace=0.9)
        random.seed(42)

        # First step: create a trace
        engine_low.td_update("s1", "a1", 0.0, "s2", ["a1"])
        random.seed(42)
        engine_high.td_update("s1", "a1", 0.0, "s2", ["a1"])

        # With low lambda, traces decay faster → might be removed
        # With high lambda, traces persist longer
        low_trace = engine_low._traces.get(("s1", "a1"), 0.0)
        high_trace = engine_high._traces.get(("s1", "a1"), 0.0)
        assert high_trace >= low_trace

    def test_default_hyperparameters(self):
        """Default values are alpha=0.1, gamma=0.95, lambda_trace=0.7."""
        engine = QLearningEngine()
        assert engine.alpha == 0.1
        assert engine.gamma == 0.95
        assert engine.lambda_trace == 0.7
