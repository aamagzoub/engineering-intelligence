"""
Training loop for the Enhanced Learning Agent.

Uses TIBRAIN's generic training loop with curriculum phases. The Wist
layer provides domain-specific environment adapters, progress logging,
and result aggregation while TIBRAIN handles the core RL cycle.

Enhancements over basic trainer:
- Card memory integration: handled by WistEnvironmentAdapter
- Per-trick reward signals (not just end of shota)
- Curriculum phases: Random → Rule-Based → Refinement
- Learning rate annealing via TIBRAIN phase hyperparameters
- Progress tracking with Wist-specific metrics

Can be used standalone (CLI) or called from the GUI Stats tab.
"""

from pathlib import Path

from tibrain.training import train, TrainingPhase, TrainingResult as TIBRAINTrainingResult

from agents.wist_learning.learning_agent import LearningAgent
from agents.wist_learning.wist_environment import WistEnvironmentAdapter
from agents.random.random_agent import RandomAgent
from agents.wist_rule_based.rule_based_agent import RuleBasedAgent


class TrainingResult:
    """Results from a training session (Wist-specific metrics)."""

    def __init__(self) -> None:
        self.episodes: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self.win_rates: list[float] = []
        self.q_table_sizes: list[int] = []
        self.epsilon_history: list[float] = []
        self.alpha_history: list[float] = []
        self.seeks_achieved: int = 0
        self.seeks_against: int = 0
        self.bids_met: int = 0
        self.bids_failed: int = 0


class _WistTIBRAINAgentAdapter:
    """
    Adapter that bridges the TIBRAIN training loop's agent interface with
    the WistEnvironmentAdapter's WistState/WistAction objects.

    The TIBRAIN training loop calls:
      - choose_action(state, legal_actions)
      - learn(state, action, reward, next_state, next_legal_actions)
      - reset_episode()
      - policy.epsilon (for progress reporting)
      - q_engine.q1.size + q_engine.q2.size (for progress reporting)

    The WistEnvironmentAdapter provides WistState and WistAction objects.
    This adapter converts them to the string keys the underlying TIBRAIN
    agent uses, then delegates to the learner's internal TIBRAIN agent.
    """

    def __init__(self, learner: LearningAgent) -> None:
        self._learner = learner
        # Expose policy and q_engine for TIBRAIN's on_progress callback.
        self.policy = learner._tibrain_agent.policy
        self.q_engine = learner._tibrain_agent.q_engine

    def choose_action(self, state, legal_actions):
        """
        Choose an action given a WistState and list of WistActions.

        Converts to string keys, uses the Q-engine to select, and returns
        the original WistAction.
        """
        if not legal_actions:
            raise ValueError("No legal actions available")

        state_key = str(state)
        action_keys = [str(a) for a in legal_actions]

        if self._learner.training:
            chosen_key = self.policy.select(
                self.q_engine.get_values(state_key, action_keys),
                action_keys,
            )
        else:
            chosen_key = self.policy.select_greedy(
                self.q_engine.get_values(state_key, action_keys),
                action_keys,
            )

        idx = action_keys.index(chosen_key)
        return legal_actions[idx]

    def learn(self, state, action, reward, next_state, next_legal_actions):
        """Update Q-values from a transition using string keys."""
        if not self._learner.training:
            return

        state_key = str(state)
        action_key = str(action)
        next_state_key = str(next_state)
        next_action_keys = [str(a) for a in next_legal_actions]

        self.q_engine.td_update(
            state_key, action_key, reward, next_state_key, next_action_keys
        )

    def reset_episode(self):
        """Clear eligibility traces for a new episode."""
        self.q_engine.reset_episode()


def _make_opponent_factory(opponent_type: str):
    """Create an opponent factory function based on the type name."""
    if opponent_type == "rule_based":
        return RuleBasedAgent
    return RandomAgent


def _make_environment_factory(opponent_type: str):
    """
    Create a factory function that produces WistEnvironmentAdapter instances
    configured with the given opponent type.

    Used by TIBRAIN's TrainingPhase.environment_factory.
    """
    def factory() -> WistEnvironmentAdapter:
        return WistEnvironmentAdapter(
            opponent_factory=_make_opponent_factory(opponent_type),
        )
    return factory


def _make_on_progress_callback(wist_result: TrainingResult, on_progress=None):
    """
    Create a TIBRAIN on_progress callback that tracks Wist-specific metrics.

    The TIBRAIN training loop calls on_progress(dict) with:
      - episode: current episode number
      - cumulative_reward: reward from the latest episode
      - epsilon: current exploration rate
      - q_table_size: number of Q-table entries

    This wrapper accumulates Wist metrics and forwards to the user's callback.
    """
    window_wins = [0]
    window_losses = [0]

    def callback(metrics: dict) -> None:
        episode = metrics.get("episode", 0)
        reward = metrics.get("cumulative_reward", 0.0)
        epsilon = metrics.get("epsilon", 0.0)
        q_table_size = metrics.get("q_table_size", 0)

        # Infer win/loss from cumulative reward (positive = likely won).
        if reward > 0:
            wist_result.wins += 1
            window_wins[0] += 1
        else:
            wist_result.losses += 1
            window_losses[0] += 1

        wist_result.episodes += 1

        # Compute window win rate.
        total_window = window_wins[0] + window_losses[0]
        win_rate = (window_wins[0] / total_window * 100) if total_window > 0 else 0

        wist_result.win_rates.append(win_rate)
        wist_result.q_table_sizes.append(q_table_size)
        wist_result.epsilon_history.append(epsilon)

        if on_progress:
            on_progress(
                episode,
                wist_result.wins,
                wist_result.losses,
                win_rate,
                epsilon,
            )

        # Reset window periodically (every report).
        window_wins[0] = 0
        window_losses[0] = 0

    return callback


def train_agent(
    episodes: int = 1000,
    opponent: str = "random",
    save_path: str | None = None,
    report_every: int = 50,
    on_progress=None,
    learner: LearningAgent | None = None,
) -> tuple[LearningAgent, TrainingResult]:
    """
    Train a LearningAgent through self-play using TIBRAIN's training loop.

    The learning agent plays as Team 0 (Players 0 and 2).
    The opponent plays as Team 1 (Players 1 and 3).

    Args:
        episodes: Number of Shotas to play.
        opponent: "random", "rule_based", or "self_play".
        save_path: If set, save the trained agent here.
        report_every: Report progress every N episodes.
        on_progress: Callback(episode, wins, losses, win_rate, epsilon).
        learner: Existing agent to continue training (or None for new).

    Returns:
        (trained_agent, training_results)
    """
    if learner is None:
        learner = LearningAgent(epsilon=0.4, training=True)

    # Create the TIBRAIN-compatible Wist environment.
    environment = WistEnvironmentAdapter(
        opponent_factory=_make_opponent_factory(opponent),
    )

    # Build Wist-specific result tracker.
    result = TrainingResult()
    tibrain_callback = _make_on_progress_callback(result, on_progress)

    # Run the TIBRAIN training loop using a bridge adapter.
    # The WistEnvironmentAdapter produces WistState/WistAction objects;
    # the _WistTIBRAINAgentAdapter converts them to string keys for Q-learning.
    agent_adapter = _WistTIBRAINAgentAdapter(learner)

    tibrain_result = train(
        agent=agent_adapter,
        environment=environment,
        episodes=episodes,
        on_progress=tibrain_callback,
        report_every=report_every,
    )

    # Sync hyperparameters back to the Wist learner.
    learner.epsilon = agent_adapter.policy.epsilon
    learner.alpha = agent_adapter.q_engine.alpha

    result.episodes = tibrain_result.episodes_completed

    # Save if requested.
    if save_path:
        learner.save(save_path)

    return learner, result


def train_curriculum(
    save_path: str = "agents/wist_learning/wist_model.json",
    on_progress=None,
) -> tuple[LearningAgent, TrainingResult]:
    """
    Enhanced curriculum training with 3 phases using TIBRAIN's training loop.

    Phase 1: 3,000 episodes vs Random (learn basic card mechanics)
             High epsilon (0.5), high alpha (0.15)
    Phase 2: 8,000 episodes vs Rule-Based (learn strategy)
             Medium epsilon (0.25), medium alpha (0.1)
    Phase 3: 4,000 episodes vs Rule-Based (refinement)
             Low epsilon (0.1), low alpha (0.05)
    """
    learner = LearningAgent(epsilon=0.5, training=True, alpha=0.15)

    # Define the 3-phase curriculum using TIBRAIN TrainingPhase objects.
    phases = [
        TrainingPhase(
            episodes=3000,
            epsilon=0.5,
            alpha=0.15,
            environment_factory=_make_environment_factory("random"),
            label="Phase 1: Random (basic mechanics)",
        ),
        TrainingPhase(
            episodes=8000,
            epsilon=0.25,
            alpha=0.1,
            environment_factory=_make_environment_factory("rule_based"),
            label="Phase 2: Rule-Based (strategy)",
        ),
        TrainingPhase(
            episodes=4000,
            epsilon=0.1,
            alpha=0.05,
            environment_factory=_make_environment_factory("rule_based"),
            label="Phase 3: Rule-Based (refinement)",
        ),
    ]

    # Build Wist-specific result tracker.
    result = TrainingResult()
    tibrain_callback = _make_on_progress_callback(result, on_progress)

    # Use a default environment (will be overridden by phase factories).
    default_env = WistEnvironmentAdapter(
        opponent_factory=_make_opponent_factory("random"),
    )

    # Run the TIBRAIN training loop with curriculum phases.
    # Use a bridge adapter for the WistState/WistAction → string key conversion.
    agent_adapter = _WistTIBRAINAgentAdapter(learner)

    tibrain_result = train(
        agent=agent_adapter,
        environment=default_env,
        episodes=0,  # Ignored when phases are provided.
        phases=phases,
        on_progress=tibrain_callback,
        report_every=100,
    )

    # Sync hyperparameters back to the Wist learner.
    learner.epsilon = agent_adapter.policy.epsilon
    learner.alpha = agent_adapter.q_engine.alpha

    result.episodes = tibrain_result.episodes_completed

    # Save the trained model.
    learner.save(save_path)

    return learner, result


# ---------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------

if __name__ == "__main__":
    def progress(ep, wins, losses, rate, eps):
        total = wins + losses
        print(f"Episode {ep:5d} | Total {total:5d} | "
              f"Win Rate (window): {rate:.1f}% | ε: {eps:.4f} | "
              f"Overall: {wins}/{total} ({wins/total*100:.1f}%)" if total > 0 else "")

    print("=" * 70)
    print("Enhanced Curriculum Training: TD(λ) Learning Agent")
    print("Phase 1: 3,000 games vs Random (basic mechanics)")
    print("Phase 2: 8,000 games vs Rule-Based (strategy)")
    print("Phase 3: 4,000 games vs Rule-Based (refinement)")
    print("=" * 70)

    agent, results = train_curriculum(
        save_path="agents/wist_learning/wist_model.json",
        on_progress=progress,
    )

    print("=" * 70)
    total = results.wins + results.losses
    print(f"Final: {results.wins}/{total} wins ({results.wins/total*100:.1f}%)")
    print(f"Q-table size: {agent.q_table_size} state-action pairs")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Final alpha: {agent.alpha:.4f}")
    print(f"Seeks achieved: {results.seeks_achieved}")
    print(f"Seeks against: {results.seeks_against}")
    print(f"Bids met: {results.bids_met}/{results.bids_met + results.bids_failed}")
    print(f"Model saved to: agents/wist_learning/wist_model.json")
