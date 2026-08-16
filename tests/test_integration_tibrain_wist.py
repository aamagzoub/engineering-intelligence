"""
Integration tests: TIBRAIN + Wist end-to-end.

Verifies that the TIBRAIN library and Wist game work together correctly:
1. Full training loop with WistEnvironmentAdapter and TIBRAIN Agent
2. Save/load cycle with all populated TIBRAIN components
3. TIBRAIN components (Agent, Policy, QLearningEngine) properly wired into Wist LearningAgent
4. WistEnvironmentAdapter conforms to TIBRAIN Environment protocol
5. Curriculum training produces expected phase transitions

Requirements: 15.5, 16.2, 12.4
"""

import json
import tempfile
from pathlib import Path

import pytest

from tibrain import Environment as TibrainEnvironment
from tibrain.agent import Agent as TibrainAgent
from tibrain.policy import Policy
from tibrain.q_learning import QLearningEngine
from tibrain.q_table import QTable
from tibrain.training import train, TrainingPhase, TrainingResult

from agents.wist_learning.learning_agent import LearningAgent
from agents.wist_learning.wist_environment import WistEnvironmentAdapter
from agents.wist_learning.trainer import (
    train_agent,
    train_curriculum,
    _WistTIBRAINAgentAdapter,
    _make_environment_factory,
    TrainingResult as WistTrainingResult,
)


# ---------------------------------------------------------------------------
# Test 1: Full training loop with WistEnvironmentAdapter and TIBRAIN Agent
# ---------------------------------------------------------------------------


class TestFullTrainingLoop:
    """Test that TIBRAIN's train() function works with WistEnvironmentAdapter."""

    def test_training_completes_episodes(self):
        """Run a few episodes via tibrain.training.train() and verify completion."""
        env = WistEnvironmentAdapter()
        learner = LearningAgent(epsilon=0.5, training=True, alpha=0.15)
        agent_adapter = _WistTIBRAINAgentAdapter(learner)

        result = train(
            agent=agent_adapter,
            environment=env,
            episodes=3,
            report_every=1,
        )

        assert result.episodes_completed == 3
        assert len(result.cumulative_rewards) == 3

    def test_q_values_populated_after_training(self):
        """After training, Q-tables should have entries (learning occurred)."""
        env = WistEnvironmentAdapter()
        learner = LearningAgent(epsilon=0.5, training=True, alpha=0.15)
        agent_adapter = _WistTIBRAINAgentAdapter(learner)

        train(
            agent=agent_adapter,
            environment=env,
            episodes=5,
            report_every=5,
        )

        q_size = agent_adapter.q_engine.q1.size + agent_adapter.q_engine.q2.size
        assert q_size > 0, "Q-tables should be populated after training"

    def test_train_agent_helper_function(self):
        """The train_agent() helper should produce a trained agent with metrics."""
        agent, result = train_agent(
            episodes=3,
            opponent="random",
            report_every=1,
        )

        assert result.episodes > 0
        assert isinstance(agent, LearningAgent)
        assert agent.q_table_size > 0

    def test_cumulative_rewards_are_numeric(self):
        """Each episode should produce a numeric cumulative reward."""
        env = WistEnvironmentAdapter()
        learner = LearningAgent(epsilon=0.5, training=True)
        agent_adapter = _WistTIBRAINAgentAdapter(learner)

        result = train(
            agent=agent_adapter,
            environment=env,
            episodes=3,
            report_every=3,
        )

        for r in result.cumulative_rewards:
            assert isinstance(r, (int, float))


# ---------------------------------------------------------------------------
# Test 2: Save/load cycle with all populated TIBRAIN components
# ---------------------------------------------------------------------------


class TestSaveLoadCycle:
    """Test that a trained agent can be saved and reloaded with matching state."""

    def test_save_load_q_table_values_match(self):
        """Q-table values should be identical after save/load round-trip."""
        # Train briefly to populate Q-tables
        agent, _ = train_agent(episodes=3, opponent="random", report_every=3)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            save_path = f.name

        try:
            agent.save(save_path)
            loaded = LearningAgent.load(save_path, training=False)

            # Compare Q-table sizes
            assert agent.q_table_size == loaded.q_table_size

            # Compare actual Q-values
            original_q1 = agent._q_engine.q1.to_dict()
            loaded_q1 = loaded._q_engine.q1.to_dict()
            assert original_q1 == loaded_q1

            original_q2 = agent._q_engine.q2.to_dict()
            loaded_q2 = loaded._q_engine.q2.to_dict()
            assert original_q2 == loaded_q2
        finally:
            Path(save_path).unlink(missing_ok=True)

    def test_save_load_preserves_hyperparameters(self):
        """Hyperparameters should be preserved through save/load."""
        agent = LearningAgent(epsilon=0.42, alpha=0.07, gamma=0.88, lambda_=0.6)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            save_path = f.name

        try:
            agent.save(save_path)
            loaded = LearningAgent.load(save_path, training=True)

            assert loaded.epsilon == pytest.approx(0.42)
            assert loaded.alpha == pytest.approx(0.07)
            assert loaded.gamma == pytest.approx(0.88)
            assert loaded.lambda_ == pytest.approx(0.6)
        finally:
            Path(save_path).unlink(missing_ok=True)

    def test_save_load_preserves_bid_q_table(self):
        """Bid Q-table entries should survive save/load."""
        agent, _ = train_agent(episodes=5, opponent="random", report_every=5)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            save_path = f.name

        try:
            agent.save(save_path)
            loaded = LearningAgent.load(save_path, training=False)

            # Compare bid Q-tables
            for state_key in agent.bid_q:
                for action_key in agent.bid_q[state_key]:
                    original_val = agent.bid_q[state_key][action_key]
                    loaded_val = loaded.bid_q[state_key][action_key]
                    assert original_val == pytest.approx(loaded_val)
        finally:
            Path(save_path).unlink(missing_ok=True)

    def test_load_nonexistent_path_raises(self):
        """Loading from a nonexistent path should raise an error."""
        with pytest.raises((FileNotFoundError, OSError)):
            LearningAgent.load("/nonexistent/path/model.json")


# ---------------------------------------------------------------------------
# Test 3: TIBRAIN components properly wired into Wist LearningAgent
# ---------------------------------------------------------------------------


class TestComponentWiring:
    """Verify that TIBRAIN components are properly wired into the Wist LearningAgent."""

    def test_learning_agent_uses_tibrain_agent(self):
        """LearningAgent should have an internal TIBRAIN Agent instance."""
        agent = LearningAgent()
        assert hasattr(agent, "_tibrain_agent")
        assert isinstance(agent._tibrain_agent, TibrainAgent)

    def test_learning_agent_uses_tibrain_q_engine(self):
        """LearningAgent should delegate Q-learning to TIBRAIN QLearningEngine."""
        agent = LearningAgent()
        q_engine = agent._q_engine
        assert isinstance(q_engine, QLearningEngine)
        assert hasattr(q_engine, "q1")
        assert hasattr(q_engine, "q2")
        assert isinstance(q_engine.q1, QTable)
        assert isinstance(q_engine.q2, QTable)

    def test_learning_agent_uses_tibrain_policy(self):
        """LearningAgent should delegate policy to TIBRAIN Policy."""
        agent = LearningAgent(epsilon=0.25)
        policy = agent._policy
        assert isinstance(policy, Policy)
        assert policy.epsilon == pytest.approx(0.25)

    def test_epsilon_sync_between_wist_and_tibrain(self):
        """Decaying epsilon on Wist agent should sync to TIBRAIN policy."""
        agent = LearningAgent(epsilon=0.5)
        agent.decay_epsilon(min_epsilon=0.01, decay_rate=0.5)

        assert agent.epsilon == pytest.approx(0.25)
        assert agent._policy.epsilon == pytest.approx(0.25)

    def test_alpha_sync_between_wist_and_tibrain(self):
        """Decaying alpha on Wist agent should sync to TIBRAIN Q-engine."""
        agent = LearningAgent(alpha=0.2)
        agent.decay_alpha(min_alpha=0.01, decay_rate=0.5)

        assert agent.alpha == pytest.approx(0.1)
        assert agent._q_engine.alpha == pytest.approx(0.1)

    def test_reset_episode_clears_tibrain_traces(self):
        """Resetting episode should clear TIBRAIN eligibility traces."""
        agent = LearningAgent()
        # Manually add a trace entry to verify it gets cleared
        agent._q_engine._traces[("s", "a")] = 1.0
        assert len(agent._q_engine._traces) > 0

        agent.reset_episode()
        assert len(agent._q_engine._traces) == 0


# ---------------------------------------------------------------------------
# Test 4: WistEnvironmentAdapter conforms to TIBRAIN Environment protocol
# ---------------------------------------------------------------------------


class TestEnvironmentProtocolConformance:
    """Verify that WistEnvironmentAdapter conforms to the TIBRAIN Environment protocol."""

    def test_isinstance_check(self):
        """WistEnvironmentAdapter should satisfy isinstance(env, Environment)."""
        env = WistEnvironmentAdapter()
        assert isinstance(env, TibrainEnvironment)

    def test_has_reset_method(self):
        """Adapter should have reset() method returning a hashable state."""
        env = WistEnvironmentAdapter()
        state = env.reset()
        assert state is not None
        # State should be hashable (required by TIBRAIN)
        hash(state)

    def test_has_observe_method(self):
        """Adapter should have observe() method returning current state."""
        env = WistEnvironmentAdapter()
        env.reset()
        state = env.observe()
        assert state is not None
        hash(state)

    def test_has_get_legal_actions_method(self):
        """Adapter should have get_legal_actions() returning a list."""
        env = WistEnvironmentAdapter()
        state = env.reset()
        actions = env.get_legal_actions(state)
        assert isinstance(actions, list)

    def test_has_step_method(self):
        """Adapter should have step() returning (state, reward, info) tuple."""
        env = WistEnvironmentAdapter()
        state = env.reset()
        actions = env.get_legal_actions(state)

        if actions:  # Only test if there are legal actions
            next_state, reward, info = env.step(actions[0])
            assert next_state is not None
            assert isinstance(reward, (int, float))
            assert isinstance(info, dict)
            assert "done" in info

    def test_episode_lifecycle(self):
        """Full episode lifecycle: reset → act → step until done."""
        env = WistEnvironmentAdapter()
        state = env.reset()
        done = False
        steps = 0
        max_steps = 200  # Safety limit

        while not done and steps < max_steps:
            actions = env.get_legal_actions(state)
            if not actions:
                break
            import random
            action = random.choice(actions)
            state, reward, info = env.step(action)
            done = info.get("done", False)
            steps += 1

        # Episode should complete within a reasonable number of steps
        # (13 tricks = 13 steps for the learner)
        assert steps > 0
        assert steps <= max_steps


# ---------------------------------------------------------------------------
# Test 5: Curriculum training produces expected phase transitions
# ---------------------------------------------------------------------------


class TestCurriculumTraining:
    """Test that curriculum training with TrainingPhases works correctly."""

    def test_multi_phase_training_executes_all_phases(self):
        """Training with multiple phases should execute all of them."""
        env = WistEnvironmentAdapter()
        learner = LearningAgent(epsilon=0.5, training=True, alpha=0.15)
        agent_adapter = _WistTIBRAINAgentAdapter(learner)

        phases = [
            TrainingPhase(
                episodes=2,
                epsilon=0.5,
                alpha=0.15,
                environment_factory=_make_environment_factory("random"),
                label="Phase 1",
            ),
            TrainingPhase(
                episodes=2,
                epsilon=0.2,
                alpha=0.1,
                environment_factory=_make_environment_factory("random"),
                label="Phase 2",
            ),
        ]

        result = train(
            agent=agent_adapter,
            environment=env,
            episodes=0,
            phases=phases,
            report_every=1,
        )

        assert result.episodes_completed == 4
        assert len(result.phase_metrics) == 2
        assert result.phase_metrics[0]["label"] == "Phase 1"
        assert result.phase_metrics[1]["label"] == "Phase 2"

    def test_phase_hyperparameters_are_applied(self):
        """Each phase should apply its specified hyperparameters to the agent."""
        env = WistEnvironmentAdapter()
        learner = LearningAgent(epsilon=0.5, training=True, alpha=0.15)
        agent_adapter = _WistTIBRAINAgentAdapter(learner)

        # Track epsilon values seen during training
        observed_epsilons = []

        def track_epsilon(metrics):
            observed_epsilons.append(metrics["epsilon"])

        phases = [
            TrainingPhase(
                episodes=2,
                epsilon=0.8,
                alpha=0.15,
                environment_factory=_make_environment_factory("random"),
                label="High Exploration",
            ),
            TrainingPhase(
                episodes=2,
                epsilon=0.1,
                alpha=0.05,
                environment_factory=_make_environment_factory("random"),
                label="Low Exploration",
            ),
        ]

        train(
            agent=agent_adapter,
            environment=env,
            episodes=0,
            phases=phases,
            on_progress=track_epsilon,
            report_every=2,
        )

        # Phase 1 should use epsilon=0.8, Phase 2 should use epsilon=0.1
        assert len(observed_epsilons) == 2
        assert observed_epsilons[0] == pytest.approx(0.8)
        assert observed_epsilons[1] == pytest.approx(0.1)

    def test_phase_metrics_recorded(self):
        """Training result should record per-phase metrics."""
        env = WistEnvironmentAdapter()
        learner = LearningAgent(epsilon=0.5, training=True)
        agent_adapter = _WistTIBRAINAgentAdapter(learner)

        phases = [
            TrainingPhase(
                episodes=2,
                epsilon=0.5,
                environment_factory=_make_environment_factory("random"),
                label="Phase A",
            ),
            TrainingPhase(
                episodes=3,
                epsilon=0.2,
                environment_factory=_make_environment_factory("random"),
                label="Phase B",
            ),
        ]

        result = train(
            agent=agent_adapter,
            environment=env,
            episodes=0,
            phases=phases,
            report_every=100,
        )

        assert len(result.phase_metrics) == 2
        assert result.phase_metrics[0]["episodes"] == 2
        assert result.phase_metrics[1]["episodes"] == 3
        assert result.episodes_completed == 5

    def test_curriculum_training_produces_phase_transitions(self):
        """
        The full 3-phase curriculum should transition between phases correctly.
        Using reduced episode counts for test speed.
        """
        env = WistEnvironmentAdapter()
        learner = LearningAgent(epsilon=0.5, training=True, alpha=0.15)
        agent_adapter = _WistTIBRAINAgentAdapter(learner)

        # Miniature version of the 3-phase curriculum
        phases = [
            TrainingPhase(
                episodes=2,
                epsilon=0.5,
                alpha=0.15,
                environment_factory=_make_environment_factory("random"),
                label="Phase 1: Random (basic mechanics)",
            ),
            TrainingPhase(
                episodes=2,
                epsilon=0.25,
                alpha=0.1,
                environment_factory=_make_environment_factory("random"),
                label="Phase 2: Rule-Based (strategy)",
            ),
            TrainingPhase(
                episodes=2,
                epsilon=0.1,
                alpha=0.05,
                environment_factory=_make_environment_factory("random"),
                label="Phase 3: Rule-Based (refinement)",
            ),
        ]

        result = train(
            agent=agent_adapter,
            environment=env,
            episodes=0,
            phases=phases,
            report_every=100,
        )

        assert result.episodes_completed == 6
        assert len(result.phase_metrics) == 3

        # After the last phase, epsilon should be the Phase 3 value
        assert agent_adapter.policy.epsilon == pytest.approx(0.1)
        assert agent_adapter.q_engine.alpha == pytest.approx(0.05)

    def test_on_progress_callback_invoked(self):
        """Progress callback should be invoked at the configured frequency."""
        env = WistEnvironmentAdapter()
        learner = LearningAgent(epsilon=0.5, training=True)
        agent_adapter = _WistTIBRAINAgentAdapter(learner)

        progress_calls = []

        def on_progress(metrics):
            progress_calls.append(metrics)

        train(
            agent=agent_adapter,
            environment=env,
            episodes=4,
            on_progress=on_progress,
            report_every=2,
        )

        # With 4 episodes and report_every=2, expect 2 callbacks
        assert len(progress_calls) == 2
        assert progress_calls[0]["episode"] == 2
        assert progress_calls[1]["episode"] == 4
