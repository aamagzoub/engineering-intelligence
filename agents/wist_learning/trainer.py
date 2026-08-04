"""
Training loop for the Enhanced Learning Agent.

Runs self-play games with TD(λ) updates per trick and Monte Carlo
for bidding. Supports curriculum learning with progressive difficulty.

Enhancements over basic trainer:
- Card memory integration: notifies agent of every card played
- Per-trick reward signals (not just end of shota)
- Curriculum phases: Random → Rule-Based → Self-Play
- Learning rate annealing for convergence
- Progress tracking with more metrics

Can be used standalone (CLI) or called from the GUI Stats tab.
"""

from pathlib import Path

from agents.wist_learning.learning_agent import LearningAgent
from agents.random.random_agent import RandomAgent
from agents.wist_rule_based.rule_based_agent import RuleBasedAgent
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
        self.alpha_history: list[float] = []
        self.seeks_achieved: int = 0
        self.seeks_against: int = 0
        self.bids_met: int = 0
        self.bids_failed: int = 0


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

    def make_opponent():
        if opponent == "rule_based":
            return RuleBasedAgent()
        elif opponent == "self_play":
            # Self-play: use a frozen copy of the learner (no training).
            return learner  # Shares Q-table but training flag matters
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

        # Play 13 tricks with per-trick rewards.
        team_tricks = {0: 0, 1: 0}

        for trick_num in range(13):
            leader_id = round_.next_leading_player_id
            round_.state.current_trick = Trick(leading_player_id=leader_id)

            play_order = [(leader_id + i) % 4 for i in range(4)]

            for player_id in play_order:
                obs = environment.observe(player_id)
                action = agents[player_id].act(obs)
                environment.apply_action(action)

                # Notify learner of every card played (card memory).
                if hasattr(action, 'card'):
                    learner.observe_card_played(action.card)

            completed_trick = round_.state.current_trick
            winner = trick_winner(completed_trick, round_.state.trump_suit)

            round_.state.completed_tricks.append(completed_trick)
            round_.state.current_trick = None
            round_.next_leading_player_id = winner

            winner_team = players[winner].team_id
            team_tricks[winner_team] += 1

            # Per-trick reward signal to the learner.
            learner_won = (winner_team == 0)
            learner.reward_trick(won=learner_won)

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
        learner.decay_alpha()

        # Track results.
        result.episodes += 1
        if team_won:
            result.wins += 1
            window_wins += 1
        else:
            result.losses += 1
            window_losses += 1

        if seek:
            if my_tricks == 13:
                result.seeks_achieved += 1
            else:
                result.seeks_against += 1
        if was_shooter:
            if bid_met:
                result.bids_met += 1
            else:
                result.bids_failed += 1

        # Report progress.
        if (episode + 1) % report_every == 0:
            total_window = window_wins + window_losses
            win_rate = (window_wins / total_window * 100) if total_window > 0 else 0
            result.win_rates.append(win_rate)
            result.q_table_sizes.append(learner.q_table_size)
            result.epsilon_history.append(learner.epsilon)
            result.alpha_history.append(learner.alpha)

            if on_progress:
                on_progress(episode + 1, result.wins, result.losses, win_rate, learner.epsilon)

            window_wins = 0
            window_losses = 0

    # Save if requested.
    if save_path:
        learner.save(save_path)

    return learner, result


def train_curriculum(
    save_path: str = "agents/wist_learning/wist_model.json",
    on_progress=None,
) -> tuple[LearningAgent, TrainingResult]:
    """
    Enhanced curriculum training with 3 phases.

    Phase 1: 3,000 episodes vs Random (learn basic card mechanics)
             High epsilon (0.5), high alpha (0.15)
    Phase 2: 8,000 episodes vs Rule-Based (learn strategy)
             Medium epsilon (0.25), medium alpha (0.1)
    Phase 3: 4,000 episodes vs Rule-Based (refinement)
             Low epsilon (0.1), low alpha (0.05)
    """
    learner = LearningAgent(epsilon=0.5, training=True, alpha=0.15)

    # Phase 1: vs Random — learn basic card mechanics.
    if on_progress:
        on_progress(0, 0, 0, 0, learner.epsilon)

    learner, result1 = train_agent(
        episodes=3000,
        opponent="random",
        learner=learner,
        report_every=100,
        on_progress=on_progress,
    )

    # Phase 2: vs Rule-Based — learn strategic play.
    learner.epsilon = 0.25
    learner.alpha = 0.1

    learner, result2 = train_agent(
        episodes=8000,
        opponent="rule_based",
        learner=learner,
        report_every=100,
        on_progress=on_progress,
    )

    # Phase 3: Refinement — lower exploration and learning rate.
    learner.epsilon = 0.1
    learner.alpha = 0.05

    _, result3 = train_agent(
        episodes=4000,
        opponent="rule_based",
        learner=learner,
        save_path=save_path,
        report_every=100,
        on_progress=on_progress,
    )

    # Merge results.
    result3.episodes += result1.episodes + result2.episodes
    result3.wins += result1.wins + result2.wins
    result3.losses += result1.losses + result2.losses
    result3.seeks_achieved += result1.seeks_achieved + result2.seeks_achieved
    result3.seeks_against += result1.seeks_against + result2.seeks_against
    result3.bids_met += result1.bids_met + result2.bids_met
    result3.bids_failed += result1.bids_failed + result2.bids_failed

    return learner, result3


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
