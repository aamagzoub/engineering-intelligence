"""Unit tests for tibrain.agent.Agent."""

import random

import pytest

from tibrain.agent import Agent


# --- Requirement 3.1: choose_action returns one of the legal actions ---


class TestChooseAction:
    def test_choose_action_returns_legal_action(self):
        """choose_action always returns an element from legal_actions."""
        agent = Agent(training=True)
        actions = ["left", "right", "up", "down"]
        chosen = agent.choose_action("state1", actions)
        assert chosen in actions

    def test_choose_action_single_action(self):
        """With only one legal action, it must be returned."""
        agent = Agent(training=True)
        chosen = agent.choose_action("state1", ["only_option"])
        assert chosen == "only_option"

    def test_choose_action_empty_raises_value_error(self):
        """Empty legal_actions raises ValueError."""
        agent = Agent(training=True)
        with pytest.raises(ValueError, match="No legal actions available"):
            agent.choose_action("state1", [])

    def test_choose_action_with_custom_encoders(self):
        """Custom encoders are applied when selecting actions."""
        state_calls = []
        action_calls = []

        def mock_state_encoder(s):
            state_calls.append(s)
            return f"encoded_{s}"

        def mock_action_encoder(a):
            action_calls.append(a)
            return f"encoded_{a}"

        agent = Agent(
            state_encoder=mock_state_encoder,
            action_encoder=mock_action_encoder,
            training=True,
        )
        actions = ["a1", "a2", "a3"]
        chosen = agent.choose_action("my_state", actions)

        # State encoder should have been called with the state
        assert "my_state" in state_calls
        # Action encoder should have been called with each legal action
        for a in actions:
            assert a in action_calls
        # Result must still be from the original actions
        assert chosen in actions

    def test_choose_action_greedy_when_not_training(self):
        """When training=False, choose_action uses greedy selection (highest Q)."""
        agent = Agent(training=False, epsilon=1.0)

        # Manually set a known Q-value so one action is clearly best
        agent.q_engine.q1.set("s1", "best", 10.0)
        agent.q_engine.q2.set("s1", "best", 10.0)
        agent.q_engine.q1.set("s1", "worse", 1.0)
        agent.q_engine.q2.set("s1", "worse", 1.0)

        # Even with epsilon=1.0, greedy should always pick "best"
        for _ in range(20):
            chosen = agent.choose_action("s1", ["best", "worse"])
            assert chosen == "best"


# --- Requirement 3.2: learn updates Q-values ---


class TestLearn:
    def test_learn_updates_q_values(self):
        """learn() should cause combined Q1+Q2 values to change."""
        agent = Agent(alpha=0.5, gamma=0.9, lambda_trace=0.0, training=True)

        # Initial Q-values are zero
        initial_values = agent.q_engine.get_values("s1", ["a1"])
        assert initial_values["a1"] == pytest.approx(0.0)

        # After learning with a positive reward, values should increase
        random.seed(42)
        agent.learn("s1", "a1", 5.0, "s2", ["a1", "a2"])

        updated_values = agent.q_engine.get_values("s1", ["a1"])
        assert updated_values["a1"] != 0.0

    def test_learn_stores_experience_in_replay_buffer(self):
        """learn() adds the experience to the replay buffer."""
        agent = Agent(training=True)
        assert len(agent.replay_buffer) == 0

        agent.learn("s1", "a1", 1.0, "s2", ["a2"])
        assert len(agent.replay_buffer) == 1

    def test_learn_multiple_transitions_accumulate(self):
        """Multiple learn() calls accumulate Q-value updates."""
        agent = Agent(alpha=0.5, gamma=0.0, lambda_trace=0.0, training=True)
        random.seed(42)

        # Multiple positive rewards should keep increasing the value
        for _ in range(10):
            agent.learn("s1", "a1", 1.0, "s2", ["a1"])

        combined = (
            agent.q_engine.q1.get("s1", "a1")
            + agent.q_engine.q2.get("s1", "a1")
        )
        assert combined > 0.0


# --- Requirement 3.4: training=False disables learning ---


class TestTrainingFlag:
    def test_training_false_learn_does_nothing(self):
        """When training=False, learn() does not update Q-values."""
        agent = Agent(training=False)

        agent.learn("s1", "a1", 10.0, "s2", ["a1", "a2"])

        # Q-values should remain at zero
        q1_val = agent.q_engine.q1.get("s1", "a1")
        q2_val = agent.q_engine.q2.get("s1", "a1")
        assert q1_val == 0.0
        assert q2_val == 0.0

    def test_training_false_replay_buffer_unchanged(self):
        """When training=False, learn() does not add to replay buffer."""
        agent = Agent(training=False)

        agent.learn("s1", "a1", 5.0, "s2", ["a1"])
        assert len(agent.replay_buffer) == 0

    def test_training_true_enables_learning(self):
        """When training=True, learn() updates state as expected."""
        agent = Agent(training=True, alpha=0.5, gamma=0.0, lambda_trace=0.0)
        random.seed(42)

        agent.learn("s1", "a1", 5.0, "s2", ["a1"])

        q1_val = agent.q_engine.q1.get("s1", "a1")
        q2_val = agent.q_engine.q2.get("s1", "a1")
        assert q1_val != 0.0 or q2_val != 0.0


# --- Requirement 3.5: Default encoders use str() conversion ---


class TestDefaultEncoders:
    def test_default_state_encoder_uses_str(self):
        """Without explicit encoder, states are converted via str()."""
        agent = Agent(training=True)

        # Use a non-string state (tuple)
        state = (1, 2, 3)
        actions = ["a", "b"]
        chosen = agent.choose_action(state, actions)
        assert chosen in actions

        # Verify that the Q-engine received the str() encoded key
        # The engine should have been queried with str((1,2,3))
        state_key = str(state)
        values = agent.q_engine.get_values(state_key, ["a", "b"])
        # Values should exist (even if 0.0) - no KeyError
        assert isinstance(values, dict)

    def test_default_action_encoder_uses_str(self):
        """Without explicit encoder, actions are converted via str()."""
        agent = Agent(training=True, alpha=0.5, gamma=0.0, lambda_trace=0.0)
        random.seed(42)

        # Use integer actions
        agent.learn(42, 7, 1.0, 43, [7, 8])

        # The Q-engine should store with str(7) as the action key
        q1_val = agent.q_engine.q1.get(str(42), str(7))
        q2_val = agent.q_engine.q2.get(str(42), str(7))
        assert q1_val != 0.0 or q2_val != 0.0

    def test_custom_encoder_overrides_default(self):
        """Explicit encoders are used instead of str()."""
        agent = Agent(
            state_encoder=lambda s: f"S:{s}",
            action_encoder=lambda a: f"A:{a}",
            training=True,
            alpha=0.5,
            gamma=0.0,
            lambda_trace=0.0,
        )
        random.seed(42)

        agent.learn("hello", "go", 1.0, "world", ["go", "stop"])

        # Should be stored with custom encoding, not plain str()
        q1_custom = agent.q_engine.q1.get("S:hello", "A:go")
        q2_custom = agent.q_engine.q2.get("S:hello", "A:go")
        assert q1_custom != 0.0 or q2_custom != 0.0

        # Plain str() keys should NOT have values
        q1_plain = agent.q_engine.q1.get("hello", "go")
        q2_plain = agent.q_engine.q2.get("hello", "go")
        assert q1_plain == 0.0 and q2_plain == 0.0


# --- Requirement 4.6 via Agent: reset_episode clears eligibility traces ---


class TestResetEpisode:
    def test_reset_episode_clears_traces(self):
        """reset_episode() delegates to q_engine and clears eligibility traces."""
        agent = Agent(
            training=True, alpha=0.1, gamma=0.9, lambda_trace=0.9
        )
        random.seed(42)

        # Build up traces via learning
        agent.learn("s1", "a1", 1.0, "s2", ["a1"])
        agent.learn("s2", "a1", 1.0, "s3", ["a1"])
        assert len(agent.q_engine._traces) > 0

        agent.reset_episode()
        assert agent.q_engine._traces == {}

    def test_reset_episode_preserves_q_values(self):
        """reset_episode() does not erase learned Q-values."""
        agent = Agent(
            training=True, alpha=0.5, gamma=0.9, lambda_trace=0.5
        )
        random.seed(42)

        agent.learn("s1", "a1", 5.0, "s2", ["a1"])
        q1_before = agent.q_engine.q1.to_dict()
        q2_before = agent.q_engine.q2.to_dict()

        agent.reset_episode()

        assert agent.q_engine.q1.to_dict() == q1_before
        assert agent.q_engine.q2.to_dict() == q2_before
