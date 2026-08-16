"""TIBRAIN Training Loop with curriculum support.

Provides a generic training loop that runs the core RL cycle
(reset → observe → act → step → learn) for a configurable number
of episodes, with optional multi-phase curriculum support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from tibrain import Environment


@dataclass
class TrainingPhase:
    """Configuration for a single training phase."""

    episodes: int
    alpha: float | None = None
    gamma: float | None = None
    epsilon: float | None = None
    lambda_trace: float | None = None
    environment_factory: Callable[[], Any] | None = None
    label: str = ""


@dataclass
class TrainingResult:
    """Results from a training session."""

    episodes_completed: int = 0
    cumulative_rewards: list[float] = field(default_factory=list)
    phase_metrics: list[dict] = field(default_factory=list)


def train(
    agent: Any,
    environment: Any,
    episodes: int,
    *,
    phases: list[TrainingPhase] | None = None,
    on_progress: Callable[[dict], None] | None = None,
    report_every: int = 50,
) -> TrainingResult:
    """
    Generic training loop with curriculum support.

    If ``phases`` is provided, training executes each phase sequentially,
    applying per-phase hyperparameters and optionally swapping environments.
    Otherwise runs ``episodes`` against the given environment.

    Parameters
    ----------
    agent : Agent
        A TIBRAIN Agent (or duck-typed equivalent) with choose_action,
        learn, reset_episode, policy, and q_engine attributes.
    environment : Environment
        A TIBRAIN Environment (or duck-typed equivalent) with reset,
        get_legal_actions, and step methods.
    episodes : int
        Number of episodes to run (used when phases is None).
    phases : list[TrainingPhase] | None
        Optional curriculum phases. Each phase specifies its own episode
        count, hyperparameters, and optional environment factory.
    on_progress : Callable[[dict], None] | None
        Callback invoked every ``report_every`` episodes with metrics.
    report_every : int
        Frequency (in episodes) at which on_progress is called.

    Returns
    -------
    TrainingResult
        Aggregated training metrics across all episodes/phases.
    """
    result = TrainingResult()

    if phases:
        for phase in phases:
            env = (
                phase.environment_factory()
                if phase.environment_factory
                else environment
            )
            # Apply phase hyperparameters to the agent
            if phase.alpha is not None:
                agent.q_engine.alpha = phase.alpha
            if phase.gamma is not None:
                agent.q_engine.gamma = phase.gamma
            if phase.epsilon is not None:
                agent.policy.epsilon = phase.epsilon
            if phase.lambda_trace is not None:
                agent.q_engine.lambda_trace = phase.lambda_trace

            phase_result = _run_episodes(
                agent, env, phase.episodes, on_progress, report_every
            )
            result.episodes_completed += phase_result.episodes_completed
            result.cumulative_rewards.extend(phase_result.cumulative_rewards)
            result.phase_metrics.append(
                {
                    "label": phase.label,
                    "episodes": phase_result.episodes_completed,
                    "final_reward": (
                        phase_result.cumulative_rewards[-1]
                        if phase_result.cumulative_rewards
                        else 0.0
                    ),
                }
            )
    else:
        result = _run_episodes(
            agent, environment, episodes, on_progress, report_every
        )

    return result


def _run_episodes(
    agent: Any,
    environment: Any,
    episodes: int,
    on_progress: Callable[[dict], None] | None,
    report_every: int,
) -> TrainingResult:
    """Execute the core RL loop for N episodes."""
    result = TrainingResult()

    for ep in range(episodes):
        state = environment.reset()
        agent.reset_episode()
        episode_reward = 0.0
        done = False

        while not done:
            legal_actions = environment.get_legal_actions(state)
            if not legal_actions:
                break

            action = agent.choose_action(state, legal_actions)
            next_state, reward, info = environment.step(action)
            done = info.get("done", False)

            next_legal = (
                environment.get_legal_actions(next_state) if not done else []
            )
            agent.learn(state, action, reward, next_state, next_legal)

            state = next_state
            episode_reward += reward

        result.episodes_completed += 1
        result.cumulative_rewards.append(episode_reward)

        if on_progress and (ep + 1) % report_every == 0:
            on_progress(
                {
                    "episode": ep + 1,
                    "cumulative_reward": episode_reward,
                    "epsilon": agent.policy.epsilon,
                    "q_table_size": agent.q_engine.q1.size
                    + agent.q_engine.q2.size,
                }
            )

    return result
