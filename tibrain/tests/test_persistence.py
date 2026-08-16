"""Unit tests for tibrain.persistence module."""

import json
import random

import pytest

from tibrain.agent import Agent
from tibrain.persistence import save, load


# --- Requirement 11.1, 11.2: save/load round-trip ---


class TestSaveLoadRoundTrip:
    def test_save_creates_json_file(self, tmp_path):
        """save() creates a JSON file at the given path."""
        agent = Agent(training=True)
        filepath = tmp_path / "agent_state.json"

        save(agent, filepath)

        assert filepath.exists()
        # Verify it's valid JSON
        data = json.loads(filepath.read_text())
        assert isinstance(data, dict)

    def test_load_returns_same_dict_data(self, tmp_path):
        """load() returns the same dict structure that was saved."""
        agent = Agent(training=True, epsilon=0.5, epsilon_min=0.02)
        filepath = tmp_path / "agent_state.json"

        save(agent, filepath)
        loaded = load(filepath)

        assert isinstance(loaded, dict)
        assert "q1" in loaded
        assert "q2" in loaded
        assert "policy" in loaded
        assert loaded["policy"]["epsilon"] == pytest.approx(0.5)
        assert loaded["policy"]["epsilon_min"] == pytest.approx(0.02)

    def test_round_trip_preserves_q_values(self, tmp_path):
        """Q-table values survive a save/load round-trip."""
        agent = Agent(training=True, alpha=0.5, gamma=0.0, lambda_trace=0.0)
        random.seed(42)

        # Train the agent so Q-values are non-zero
        agent.learn("s1", "a1", 5.0, "s2", ["a1", "a2"])
        agent.learn("s2", "a2", 3.0, "s3", ["a1"])

        filepath = tmp_path / "agent_state.json"
        save(agent, filepath)
        loaded = load(filepath)

        # Verify Q-table data round-trips correctly
        assert loaded["q1"] == agent.q_engine.q1.to_dict()
        assert loaded["q2"] == agent.q_engine.q2.to_dict()

    def test_round_trip_preserves_replay_buffer(self, tmp_path):
        """Replay buffer contents survive a save/load round-trip."""
        agent = Agent(training=True, alpha=0.1, gamma=0.9, lambda_trace=0.0)
        random.seed(42)

        agent.learn("s1", "a1", 1.0, "s2", ["a1"])
        agent.learn("s2", "a2", 2.0, "s3", ["a2"])

        filepath = tmp_path / "agent_state.json"
        save(agent, filepath)
        loaded = load(filepath)

        assert "replay_buffer" in loaded
        assert len(loaded["replay_buffer"]) == 2


# --- Requirement 11.4: load from nonexistent path returns empty dict ---


class TestLoadNonexistent:
    def test_load_nonexistent_returns_empty_dict(self, tmp_path):
        """load() from a path that does not exist returns {} without exception."""
        filepath = tmp_path / "does_not_exist.json"

        result = load(filepath)

        assert result == {}

    def test_load_nonexistent_no_exception(self, tmp_path):
        """load() does not raise any exception for a missing file."""
        filepath = tmp_path / "missing" / "nested" / "path.json"

        # Should not raise
        result = load(filepath)
        assert result == {}


# --- Requirement 11.5: incremental save with changed_components ---


class TestIncrementalSave:
    def test_incremental_save_updates_only_specified_keys(self, tmp_path):
        """Incremental save only updates keys listed in changed_components."""
        agent = Agent(training=True, epsilon=0.5, epsilon_min=0.01)
        filepath = tmp_path / "agent_state.json"

        # Initial full save
        save(agent, filepath)
        initial_data = load(filepath)

        # Modify agent state
        agent.policy.epsilon = 0.1
        random.seed(99)
        agent.learn("new_s", "new_a", 10.0, "new_s2", ["new_a"])

        # Incremental save: only update policy
        save(agent, filepath, changed_components={"policy"})
        updated_data = load(filepath)

        # Policy should be updated
        assert updated_data["policy"]["epsilon"] == pytest.approx(0.1)

        # Q-tables should remain as they were in the initial save
        # (not the new state, because we only saved "policy")
        assert updated_data["q1"] == initial_data["q1"]
        assert updated_data["q2"] == initial_data["q2"]

    def test_incremental_save_without_existing_file_writes_full(self, tmp_path):
        """When file doesn't exist, incremental save writes all data."""
        agent = Agent(training=True, epsilon=0.7)
        filepath = tmp_path / "new_file.json"

        # Even with changed_components, should write full data if no file exists
        save(agent, filepath, changed_components={"policy"})

        loaded = load(filepath)
        assert "q1" in loaded
        assert "q2" in loaded
        assert "policy" in loaded
        assert loaded["policy"]["epsilon"] == pytest.approx(0.7)

    def test_incremental_save_preserves_unspecified_keys(self, tmp_path):
        """Keys not in changed_components remain unchanged after incremental save."""
        agent = Agent(training=True, epsilon=0.3)
        filepath = tmp_path / "agent_state.json"

        # Initial save
        save(agent, filepath)
        initial_data = load(filepath)
        original_replay = initial_data.get("replay_buffer", [])

        # Learn something new (changes q-tables and replay buffer)
        random.seed(7)
        agent.learn("x", "y", 5.0, "z", ["y"])

        # Only save q1
        save(agent, filepath, changed_components={"q1"})
        updated_data = load(filepath)

        # q1 should be updated
        assert updated_data["q1"] == agent.q_engine.q1.to_dict()

        # replay_buffer should remain unchanged (from initial save)
        assert updated_data.get("replay_buffer", []) == original_replay


# --- Agent with Q-values serialization round-trip ---


class TestAgentQValuesSerialization:
    def test_agent_with_trained_q_values_serializes_correctly(self, tmp_path):
        """An agent with non-trivial Q-values serializes and loads back."""
        agent = Agent(
            training=True, alpha=0.5, gamma=0.9, lambda_trace=0.3
        )
        random.seed(123)

        # Train multiple episodes
        for _ in range(5):
            agent.learn("s1", "a1", 1.0, "s2", ["a1", "a2"])
            agent.learn("s2", "a2", -0.5, "s3", ["a1"])
            agent.reset_episode()

        filepath = tmp_path / "trained_agent.json"
        save(agent, filepath)
        loaded = load(filepath)

        # Verify all Q-value entries are present
        assert loaded["q1"] == agent.q_engine.q1.to_dict()
        assert loaded["q2"] == agent.q_engine.q2.to_dict()

        # Verify we can reconstruct Q-tables from loaded data
        from tibrain.q_table import QTable

        q1_loaded = QTable.from_dict(loaded["q1"])
        q2_loaded = QTable.from_dict(loaded["q2"])

        assert q1_loaded.to_dict() == agent.q_engine.q1.to_dict()
        assert q2_loaded.to_dict() == agent.q_engine.q2.to_dict()


# --- Agent with evaluator serialization ---


class TestAgentWithEvaluator:
    def test_agent_with_evaluator_serializes(self, tmp_path):
        """An agent with a neural evaluator serializes the evaluator weights."""
        agent = Agent(
            training=True,
            use_neural=True,
            neural_config={"input_size": 8, "hidden_size": 16, "n_heads": 2},
        )
        filepath = tmp_path / "neural_agent.json"

        save(agent, filepath)
        loaded = load(filepath)

        assert "evaluator" in loaded
        assert loaded["evaluator"]["input_size"] == 8
        assert loaded["evaluator"]["hidden_size"] == 16
        assert loaded["evaluator"]["n_heads"] == 2
        # Weights should be present
        assert "w1" in loaded["evaluator"]
        assert "b1" in loaded["evaluator"]

    def test_evaluator_can_be_reconstructed_from_loaded_data(self, tmp_path):
        """Evaluator weights loaded from JSON can reconstruct an Evaluator."""
        import numpy as np
        from tibrain.neural_net import Evaluator

        agent = Agent(
            training=True,
            use_neural=True,
            neural_config={"input_size": 8, "hidden_size": 16, "n_heads": 2},
        )
        filepath = tmp_path / "neural_agent.json"

        save(agent, filepath)
        loaded = load(filepath)

        # Reconstruct evaluator from saved data
        evaluator = Evaluator.from_dict(loaded["evaluator"])
        test_input = np.random.randn(8)

        # Both should produce the same prediction
        original_pred = agent._evaluator.predict(test_input)
        loaded_pred = evaluator.predict(test_input)
        assert loaded_pred == pytest.approx(original_pred, rel=1e-6)

    def test_agent_without_evaluator_omits_evaluator_key(self, tmp_path):
        """An agent without a neural evaluator does not include 'evaluator' in saved data."""
        agent = Agent(training=True, use_neural=False)
        filepath = tmp_path / "no_neural.json"

        save(agent, filepath)
        loaded = load(filepath)

        assert "evaluator" not in loaded
