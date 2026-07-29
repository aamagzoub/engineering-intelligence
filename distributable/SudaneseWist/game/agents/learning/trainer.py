"""
Training loop for the Learning Agent.

Runs self-play games, updates the Q-table after each trick and Shota,
and tracks win rates over time.

Can be used standalone (CLI) or called from the GUI Stats tab.
"""

from pathlib import Path

from agents.learning.learning_agent import LearningAgent
from agents.random.random_agent import RandomAgent
from agents.rule_based.rule_based_agent import RuleBasedAgent
from environments.wist.environment import WistEnvironment
from environments.wist.playing_engine import PlayingEngine
from environments.wist.round import Round
from environments.wist.rules import trick_winner
from environments.wist.scoring import detect_seek
from environments.wist.setup import create_standard_players
from environments.wist.tasmiya_engine import TasmiyaEngine, determine_trump_suit
from environments.wist.trick import Trick
from intelligence.core.agent import Agent


class TrainingResult:
    """Results from a training session."""

    def __init__(self) -> None:
        self.episodes: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self.win_rates: list[float] = []  # Win rate per window.
        self.q_table_sizes: list[int] = []
        self.epsilon_history: list[float] = []


def train_agent(
    episodes: int = 1000,
    opponent: str = "random",
    save_path: str | None = None,
    report_every: int = 50,
    on_progress=None,
) -> tuple[LearningAgent, TrainingResult]:
    """
    Train a LearningAgent through self-play.

    The learning agent plays as Team 0 (Players 0 and 2).
    The opponent plays as Team 1 (Players 1 and 3).

    Args:
        episodes: Number of Shotas to play.
        opponent: "random" or "rule_based".
        save_path: If set, save the trained agent here.
        report_every: Print/report progress every N episodes.
        on_progress: Callback(episode, wins, losses, win_rate, epsilon)

    Returns:
        (trained_agent, training_results)
    """

    # Create agents.
    learner = LearningAgent(epsilon=0.4, alpha=0.15, gamma=0.9, training=True)

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
        agents: list[Agent] = [
            learner,              # P0 - Team 0
            make_opponent(),      # P1 - Team 1
            learner,              # P2 - Team 0 (same agent instance)
            make_opponent(),      # P3 - Team 1
        ]

        round_ = Round(players)
        round_.deal()

        # Skip card-based Dak hands.
        if round_.has_card_based_dak():
            continue

        # Bidding.
        tasmiya_result = tasmiya_engine.run(
            players=players,
            agents=agents,
            sahib_al_qabool_id=0,
        )

        if tasmiya_result.is_dak:
            learner.reset_episode()
            continue

        # Set up for play.
        round_.state.trump_suit = tasmiya_result.trump_suit
        round_.state.winning_bidder_id = tasmiya_result.winning_bidder_id
        round_.next_leading_player_id = tasmiya_result.winning_bidder_id

        environment = WistEnvironment(round_.state)

        # Play 13 tricks manually (to give per-trick rewards).
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

            # Per-trick reward to the learning agent.
            learner.reward_trick(won=(winner_team == 0))

        # Shota-level reward.
        playing_team = tasmiya_result.playing_team_id
        bid = tasmiya_result.winning_bid_value
        learner_team_tricks = team_tricks[0]
        bid_met = (learner_team_tricks >= bid) if playing_team == 0 else False
        team_won = team_tricks[0] > team_tricks[1]

        learner.reward_shota(team_won_shota=team_won, bid_met=bid_met)
        learner.decay_epsilon()

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


# ---------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------

if __name__ == "__main__":
    def progress(ep, wins, losses, rate, eps):
        print(f"Episode {ep:5d} | Wins: {wins} | Losses: {losses} | "
              f"Win Rate (window): {rate:.1f}% | ε: {eps:.3f}")

    print("Training Learning Agent vs Random...")
    print("=" * 60)

    agent, results = train_agent(
        episodes=2000,
        opponent="random",
        save_path="agents/learning/trained_model.json",
        report_every=100,
        on_progress=progress,
    )

    print("=" * 60)
    total = results.wins + results.losses
    print(f"Final: {results.wins}/{total} wins ({results.wins/total*100:.1f}%)")
    print(f"Q-table size: {agent.q_table_size} state-action pairs")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Model saved to: agents/learning/trained_model.json")
