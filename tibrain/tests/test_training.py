"""Unit tests for tibrain.training module."""

import pytest

from tibrain.agent import Agent
from tibrain.training import train, TrainingPhase, TrainingResult


# --- Mock Environment ---


class MockEnvironment:
    """Simple mock environment for testing the training loop.

    reset() returns "state_0"
    get_legal_actions(state) returns ["a", "b"]
    step(action) returns ("state_1", 1.0, {"done": True})
    """

    def __init__(self):
        self.reset_calls = 0
        self.get_legal_actions_calls = 0
        self.step_calls = 0

    def reset(self) -> str:
        self.reset_calls += 1
        return "state_0"

    def observe(self) -> str:
        return "state_0"

    def get_legal_actions(self, state) -> list[str]:
        self.get_legal_actions_calls += 1
        return ["a", "b"]

    def step(self, action) -> tuple[str, float, dict]:
        self.step_calls += 1
        return ("state_1", 1.0, {"done": True})


class MultiStepEnvironment:
    """Environment that takes multiple steps before terminating."""

    def __init__(self, steps_per_episode: int = 3):
        self._steps_per_episode = steps_per_episode
        self._current_step = 0

    def reset(self) -> str:
        self._current_step = 0
        return "state_0"

    def observe(self) -> str:
        return f"state_{self._current_step}"

    def get_legal_actions(self, state) -> list[str]:
        return ["a", "b"]

    def step(self, action) -> tuple[str, float, dict]:
        self._current_step += 1
        done = self._current_step >= self._steps_per_episode
        return (f"state_{self._current_step}", 1.0, {"done": done})


# --- Requirement 12.1: train() runs specified episodes and returns TrainingResult ---


class TestTrainBasic:
    def test_returns_training_result(self):
        agent = Agent()
        env = MockEnvironment()
        result = train(agent, env, episodes=5)
        assert isinstance(result, TrainingResult)

    def test_runs_specified_number_of_episodes(self):
        agent = Agent()
        env = MockEnvironment()
        result = train(agent, env, episodes=10)
        assert result.episodes_completed == 10

    def test_zero_episodes(self):
        agent = Agent()
        env = MockEnvironment()
        result = train(agent, env, episodes=0)
        assert result.episodes_completed == 0
        assert result.cumulative_rewards == []

    def test_single_episode(self):
        agent = Agent()
        env = MockEnvironment()
        result = train(agent, env, episodes=1)
        assert result.episodes_completed == 1


# --- Requirement 12.6: cumulative_rewards contains one entry per episode ---


class TestCumulativeRewards:
    def test_rewards_list_length_matches_episodes(self):
        agent = Agent()
        env = MockEnvironment()
        result = train(agent, env, episodes=7)
        assert len(result.cumulative_rewards) == 7

    def test_rewards_reflect_episode_total(self):
        """Mock env gives 1.0 reward per step; single-step episodes → 1.0 each."""
        agent = Agent()
        env = MockEnvironment()
        result = train(agent, env, episodes=3)
        # Each episode: one step with reward 1.0
        for r in result.cumulative_rewards:
            assert r == 1.0

    def test_multi_step_rewards_accumulate(self):
        """Multi-step env gives 1.0 per step; 3 steps → 3.0 per episode."""
        agent = Agent()
        env = MultiStepEnvironment(steps_per_episode=3)
        result = train(agent, env, episodes=2)
        for r in result.cumulative_rewards:
            assert r == 3.0


# --- Requirement 12.3: Episode loop calls reset, get_legal_actions, choose_action, step, learn ---


class TestEpisodeLoop:
    def test_reset_called_each_episode(self):
        agent = Agent()
        env = MockEnvironment()
        train(agent, env, episodes=5)
        assert env.reset_calls == 5

    def test_get_legal_actions_called(self):
        agent = Agent()
        env = MockEnvironment()
        train(agent, env, episodes=3)
        # At least once per episode (for action selection)
        assert env.get_legal_actions_calls >= 3

    def test_step_called_each_episode(self):
        agent = Agent()
        env = MockEnvironment()
        train(agent, env, episodes=4)
        # Single-step env: one step per episode
        assert env.step_calls == 4

    def test_multi_step_calls_step_multiple_times(self):
        agent = Agent()
        env = MultiStepEnvironment(steps_per_episode=3)
        train(agent, env, episodes=2)
        # 2 episodes × 3 steps = 6 total step calls
        # We check via the internal counter
        assert env._current_step == 3  # Last episode ends at step 3

    def test_agent_learns_during_training(self):
        """Verify agent Q-values are updated after training."""
        agent = Agent()
        env = MockEnvironment()
        train(agent, env, episodes=5)
        # After 5 episodes, the agent should have learned something
        total_size = agent.q_engine.q1.size + agent.q_engine.q2.size
        assert total_size > 0


# --- Requirement 12.4: Curriculum phases apply hyperparameters ---


class TestCurriculumPhases:
    def test_phases_total_episodes(self):
        agent = Agent()
        env = MockEnvironment()
        phases = [
            TrainingPhase(episodes=3, label="phase_1"),
            TrainingPhase(episodes=5, label="phase_2"),
        ]
        result = train(agent, env, episodes=0, phases=phases)
        assert result.episodes_completed == 8

    def test_phases_apply_epsilon(self):
        agent = Agent(epsilon=0.5)
        env = MockEnvironment()
        phases = [
            TrainingPhase(episodes=2, epsilon=0.1, label="low_explore"),
        ]
        train(agent, env, episodes=0, phases=phases)
        # After phase, agent epsilon should have been set to 0.1
        # (policy may have decayed further during training, but it started at 0.1)
        assert agent.policy.epsilon <= 0.1

    def test_phases_apply_alpha(self):
        agent = Agent(alpha=0.1)
        env = MockEnvironment()
        phases = [
            TrainingPhase(episodes=1, alpha=0.5, label="high_lr"),
        ]
        train(agent, env, episodes=0, phases=phases)
        assert agent.q_engine.alpha == 0.5

    def test_phases_apply_gamma(self):
        agent = Agent(gamma=0.9)
        env = MockEnvironment()
        phases = [
            TrainingPhase(episodes=1, gamma=0.99, label="long_horizon"),
        ]
        train(agent, env, episodes=0, phases=phases)
        assert agent.q_engine.gamma == 0.99

    def test_phases_apply_lambda_trace(self):
        agent = Agent(lambda_trace=0.7)
        env = MockEnvironment()
        phases = [
            TrainingPhase(episodes=1, lambda_trace=0.9, label="high_trace"),
        ]
        train(agent, env, episodes=0, phases=phases)
        assert agent.q_engine.lambda_trace == 0.9

    def test_phase_metrics_recorded(self):
        agent = Agent()
        env = MockEnvironment()
        phases = [
            TrainingPhase(episodes=3, label="first"),
            TrainingPhase(episodes=2, label="second"),
        ]
        result = train(agent, env, episodes=0, phases=phases)
        assert len(result.phase_metrics) == 2
        assert result.phase_metrics[0]["label"] == "first"
        assert result.phase_metrics[0]["episodes"] == 3
        assert result.phase_metrics[1]["label"] == "second"
        assert result.phase_metrics[1]["episodes"] == 2

    def test_phase_with_environment_factory(self):
        """Phases can swap environments via environment_factory."""
        agent = Agent()
        default_env = MockEnvironment()
        custom_env = MockEnvironment()

        phases = [
            TrainingPhase(
                episodes=2,
                environment_factory=lambda: custom_env,
                label="custom",
            ),
        ]
        train(agent, default_env, episodes=0, phases=phases)
        # The custom env should have been used, not the default
        assert custom_env.reset_calls == 2
        assert default_env.reset_calls == 0

    def test_cumulative_rewards_across_phases(self):
        agent = Agent()
        env = MockEnvironment()
        phases = [
            TrainingPhase(episodes=3, label="p1"),
            TrainingPhase(episodes=2, label="p2"),
        ]
        result = train(agent, env, episodes=0, phases=phases)
        assert len(result.cumulative_rewards) == 5


# --- Requirement 12.5: on_progress callback frequency ---


class TestOnProgressCallback:
    def test_callback_called_at_report_every(self):
        agent = Agent()
        env = MockEnvironment()
        progress_calls = []
        train(
            agent,
            env,
            episodes=10,
            on_progress=lambda metrics: progress_calls.append(metrics),
            report_every=5,
        )
        # Should be called at episode 5 and 10
        assert len(progress_calls) == 2

    def test_callback_called_every_episode_when_report_every_1(self):
        agent = Agent()
        env = MockEnvironment()
        progress_calls = []
        train(
            agent,
            env,
            episodes=4,
            on_progress=lambda metrics: progress_calls.append(metrics),
            report_every=1,
        )
        assert len(progress_calls) == 4

    def test_callback_not_called_when_episodes_less_than_report_every(self):
        agent = Agent()
        env = MockEnvironment()
        progress_calls = []
        train(
            agent,
            env,
            episodes=3,
            on_progress=lambda metrics: progress_calls.append(metrics),
            report_every=5,
        )
        assert len(progress_calls) == 0

    def test_callback_receives_expected_keys(self):
        agent = Agent()
        env = MockEnvironment()
        progress_calls = []
        train(
            agent,
            env,
            episodes=5,
            on_progress=lambda metrics: progress_calls.append(metrics),
            report_every=5,
        )
        assert len(progress_calls) == 1
        metrics = progress_calls[0]
        assert "episode" in metrics
        assert "cumulative_reward" in metrics
        assert "epsilon" in metrics
        assert "q_table_size" in metrics

    def test_callback_episode_number_is_correct(self):
        agent = Agent()
        env = MockEnvironment()
        progress_calls = []
        train(
            agent,
            env,
            episodes=10,
            on_progress=lambda metrics: progress_calls.append(metrics),
            report_every=3,
        )
        # Called at episodes 3, 6, 9
        assert len(progress_calls) == 3
        assert progress_calls[0]["episode"] == 3
        assert progress_calls[1]["episode"] == 6
        assert progress_calls[2]["episode"] == 9

    def test_no_callback_when_on_progress_is_none(self):
        """No error when on_progress is None."""
        agent = Agent()
        env = MockEnvironment()
        # Should not raise
        result = train(agent, env, episodes=5, on_progress=None, report_every=1)
        assert result.episodes_completed == 5

    def test_default_report_every_is_50(self):
        agent = Agent()
        env = MockEnvironment()
        progress_calls = []
        train(
            agent,
            env,
            episodes=100,
            on_progress=lambda metrics: progress_calls.append(metrics),
        )
        # Default report_every=50 → called at episode 50 and 100
        assert len(progress_calls) == 2
        assert progress_calls[0]["episode"] == 50
        assert progress_calls[1]["episode"] == 100
