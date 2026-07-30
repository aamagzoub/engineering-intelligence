"""
Training loop for the Learning Agent.

Runs self-play games with Monte Carlo updates after each Shota.
Supports curriculum learning: start vs Random, then graduate to Rule-Based.

Can be used standalone (CLI) or called from the GUI Stats tab.
"""

from pathlib import Path

from agents.learning.learning_agent import LearningAgent
from agents.random.random_agent import RandomAgent
from agents.rule_based.rule_based_agent import RuleBasedAgent
from environments.wist.environment import WistEnvironment
from environments.wist.round import Round
from environments.wist.rules import trick_winner
from environments.wist.scoring import detect_seek
from environments.wist.setup import create_standard_players
from environments.wist.tasmiya_engine import TasmiyaEngine
from environments.wist.trick import Trick
from intelligence.core.agent import Agent


class TrainingResult:
    """Results from a training session."""

    def __init__(self) -> None:
        self.episodes: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self.win_rates: list[float] = []
        self.q_table_sizes: list[int] = []
        self.epsilon_history: list[float] = []


def train_agent(
    episodes: int = 1000,
    opponent: str = "random",
    save_path: str | None = None,
    report_every: int = 50,
    on_progress=None,
    learner: LearningAgent | None = None,
) -> tuple[LearningAgent, TrainingResult]:
    """
    Train a LearningAgent through self-play.

    The learning agent plays as Team 0 (Players 0 and 2).
    The opponent plays as Team 1 (Players 1 and 3).

    Args:
        episodes: Number of Shotas to play.
        opponent: "random" or "rule_based".
        save_path: If set, save the trained agent here.
        report_every: Report progress every N episodes.
        on_progress: Callback(episode, wins, losses, win_rate, epsilon).
        learner: Existing agent to continue training (or None for new).

    Returns:
        (trained_agent, training_results)
    """
    if learner is None:
        learner = LearningAgent(epsilon=0.4, training=True)

    def make_opponent():
        if opponent == "rule_based":
            return RuleBasedAgent()
        return RandomAgent()

    result = TrainingResult()
    window_wins = 0
    window_losses = 0

    tasmiya_engine = TasmiyaEngine()

    for episode in range(episodes):
        players = create_standard_players()

        # Team 0 = learning agent, Team 1 = opponent.
        opp1 = make_opponent()
        opp2 = make_opponent()
        agents: list[Agent] = [learner, opp1, learner, opp2]

        round_ = Round(players)
        round_.deal()

        # Skip card-based Dak hands.
        if round_.has_card_based_dak():
            learner.reset_episode()
            continue

        # Bidding — rotate Qabool each episode.
        qabool_id = episode % 4
        tasmiya_result = tasmiya_engine.run(
            players=players,
            agents=agents,
            sahib_al_qabool_id=qabool_id,
        )

        if tasmiya_result.is_dak:
            learner.reset_episode()
            continue

        # Set up for play.
        round_.state.trump_suit = tasmiya_result.trump_suit
        round_.state.winning_bidder_id = tasmiya_result.winning_bidder_id
        round_.next_leading_player_id = tasmiya_result.winning_bidder_id

        environment = WistEnvironment(round_.state)

        # Play 13 tricks.
        team_tricks = {0: 0, 1: 0}

        for trick_num in range(13):
            leader_id = round_.next_leading_player_id
            round_.state.current_trick = Trick(leading_player_id=leader_id)

            play_order = [(leader_id + i) % 4 for i in range(4)]

            for player_id in play_order:
                obs = environment.observe(player_id)
                action = agents[player_id].act(obs)
                environment.apply_action(action)

            completed_trick = round_.state.current_trick
            winner = trick_winner(completed_trick, round_.state.trump_suit)

            round_.state.completed_tricks.append(completed_trick)
            round_.state.current_trick = None
            round_.next_leading_player_id = winner

            winner_team = players[winner].team_id
            team_tricks[winner_team] += 1

        # Determine outcomes.
        playing_team = tasmiya_result.playing_team_id
        bid = tasmiya_result.winning_bid_value
        shooter_id = tasmiya_result.winning_bidder_id
        learner_team = 0

        my_tricks = team_tricks[learner_team]
        opp_tricks = team_tricks[1]
        team_won = my_tricks > opp_tricks
        was_shooter = (playing_team == learner_team)
        bid_met = (my_tricks >= bid) if was_shooter else False

        # Seek detection.
        seek = (my_tricks == 13 or opp_tricks == 13)

        # Reward the agent.
        learner.reward_shota(
            team_won_shota=team_won,
            bid_met=bid_met,
            my_tricks=my_tricks,
            opp_tricks=opp_tricks,
            was_shooter=was_shooter,
            seek=seek,
        )
        learner.decay_epsilon()

        # Track results.
        result.episodes += 1
        if team_won:
            result.wins += 1
            window_wins += 1
        else:
            result.losses += 1
            window_losses += 1

        # Report progress.
        if (episode + 1) % report_every == 0:
            total_window = window_wins + window_losses
            win_rate = (window_wins / total_window * 100) if total_window > 0 else 0
            result.win_rates.append(win_rate)
            result.q_table_sizes.append(learner.q_table_size)
            result.epsilon_history.append(learner.epsilon)

            if on_progress:
                on_progress(episode + 1, result.wins, result.losses, win_rate, learner.epsilon)

            window_wins = 0
            window_losses = 0

    # Save if requested.
    if save_path:
        learner.save(save_path)

    return learner, result


def train_curriculum(
    save_path: str = "agents/learning/trained_model.json",
    on_progress=None,
) -> tuple[LearningAgent, TrainingResult]:
    """
    Curriculum training: Random → Rule-Based.

    Phase 1: 5000 episodes vs Random (learn basics)
    Phase 2: 10000 episodes vs Rule-Based (learn strategy)
    """
    learner = LearningAgent(epsilon=0.5, training=True)

    # Phase 1: vs Random.
    if on_progress:
        on_progress(0, 0, 0, 0, learner.epsilon)

    learner, result1 = train_agent(
        episodes=5000,
        opponent="random",
        learner=learner,
        report_every=100,
        on_progress=on_progress,
    )

    # Phase 2: vs Rule-Based (lower epsilon for more exploitation).
    learner.epsilon = 0.25

    _, result2 = train_agent(
        episodes=10000,
        opponent="rule_based",
        learner=learner,
        save_path=save_path,
        report_every=100,
        on_progress=on_progress,
    )

    # Merge results.
    result2.episodes += result1.episodes
    result2.wins += result1.wins
    result2.losses += result1.losses

    return learner, result2


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
    print("Curriculum Training: Learning Agent")
    print("Phase 1: 5,000 games vs Random")
    print("Phase 2: 10,000 games vs Rule-Based")
    print("=" * 70)

    agent, results = train_curriculum(
        save_path="agents/learning/trained_model.json",
        on_progress=progress,
    )

    print("=" * 70)
    total = results.wins + results.losses
    print(f"Final: {results.wins}/{total} wins ({results.wins/total*100:.1f}%)")
    print(f"Q-table size: {agent.q_table_size} state-action pairs")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Model saved to: agents/learning/trained_model.json")
