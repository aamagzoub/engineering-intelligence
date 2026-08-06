"""
Background training and opponent curriculum management.
"""

import random
from collections import defaultdict

from agents.wist_discovery.discovery_agent import WistDiscoveryAgent
from environments.wist.environment import WistEnvironment
from environments.wist.round import Round
from environments.wist.rules import trick_winner
from environments.wist.scoring import score_shota
from environments.wist.setup import create_standard_players
from environments.wist.tasmiya_engine import TasmiyaEngine
from environments.wist.trick import Trick


# ─── Opponent Curriculum ────────────────────────────────────────────────────────

STAGE_STAGNATION_THRESHOLD = 5000  # Episodes without discovery to trigger graduation.


def snapshot_brain(agent):
    """Take a frozen copy of Q-tables (thread-safe)."""
    try:
        return {
            "play_q": {k: dict(v) for k, v in list(agent.play_q.items())},
            "play_q2": {k: dict(v) for k, v in list(agent.play_q2.items())},
            "bid_q": {k: dict(v) for k, v in list(agent.bid_q.items())},
            "bid_q2": {k: dict(v) for k, v in list(agent.bid_q2.items())},
        }
    except RuntimeError:
        return {"play_q": {}, "play_q2": {}, "bid_q": {}, "bid_q2": {}}


def create_opponent(agent, stage, frozen_snapshot=None, best_snapshot=None):
    """
    Create an opponent based on current curriculum stage.

    Stage 1: Same brain (self-play).
    Stage 2: Mixed (50% self + 30% weak snapshot + 20% random).
    Stage 3: Adversarial (60% frozen best + 20% current + 20% random).
    """
    opp = WistDiscoveryAgent(training=False)

    if stage == 1:
        opp.play_q = agent.play_q
        opp.play_q2 = agent.play_q2
        opp.bid_q = agent.bid_q
        opp.bid_q2 = agent.bid_q2
        opp.epsilon = agent.epsilon
        return opp

    if stage == 2:
        roll = random.random()
        if roll < 0.5:
            opp.play_q = agent.play_q
            opp.play_q2 = agent.play_q2
            opp.bid_q = agent.bid_q
            opp.bid_q2 = agent.bid_q2
            opp.epsilon = agent.epsilon
        elif roll < 0.8 and frozen_snapshot:
            _load_snapshot(opp, frozen_snapshot, epsilon=0.1)
        else:
            opp.epsilon = 1.0
        return opp

    # Stage 3: adversarial.
    roll = random.random()
    if best_snapshot and roll < 0.6:
        _load_snapshot(opp, best_snapshot, epsilon=0.05)
    elif roll < 0.8:
        opp.play_q = agent.play_q
        opp.play_q2 = agent.play_q2
        opp.bid_q = agent.bid_q
        opp.bid_q2 = agent.bid_q2
        opp.epsilon = agent.epsilon
    else:
        opp.epsilon = 0.8
    return opp


def _load_snapshot(opp, snapshot, epsilon):
    """Load a frozen Q-table snapshot into an opponent."""
    opp.play_q = defaultdict(lambda: defaultdict(float), snapshot["play_q"])
    opp.play_q2 = defaultdict(lambda: defaultdict(float), snapshot["play_q2"])
    opp.bid_q = defaultdict(lambda: defaultdict(float), snapshot["bid_q"])
    opp.bid_q2 = defaultdict(lambda: defaultdict(float), snapshot["bid_q2"])
    opp.epsilon = epsilon


# ─── Background Training ────────────────────────────────────────────────────────


def create_training_clone(discovery):
    """Create a background training agent that shares Q-tables with the main agent."""
    agent = WistDiscoveryAgent(training=True)
    agent.play_q = discovery.play_q
    agent.play_q2 = discovery.play_q2
    agent.bid_q = discovery.bid_q
    agent.bid_q2 = discovery.bid_q2
    agent.epsilon = discovery.epsilon
    agent._use_neural = discovery._use_neural
    agent._play_net = discovery._play_net
    agent._target_net = discovery._target_net
    agent._bid_net = discovery._bid_net
    agent._replay_buffer = discovery._replay_buffer
    agent._state_visit_counts = discovery._state_visit_counts
    agent._reward_normalizer = discovery._reward_normalizer
    agent.episodes_trained = discovery.episodes_trained
    agent.total_updates = discovery.total_updates
    return agent


def run_background_training(agent, opp, num_shotas=10000, milestone_callback=None):
    """
    Run silent self-play training in background.

    Args:
        agent: Training agent (shares Q-tables with main).
        opp: Opponent agent.
        num_shotas: Total shotas to train.
        milestone_callback: Optional fn(team_tricks, bid, playing_team, bid_met, scores)
            called after each shota for milestone detection.

    Returns:
        win_history: list of bools (game won or not).
    """
    win_history = []
    shotas_done = 0

    while shotas_done < num_shotas:
        team_scores = [0, 0]
        shota_count = 0

        for _ in range(5):  # 5 shotas per game.
            players = create_standard_players()
            agents_list = [agent, opp, agent, opp]
            rnd = Round(players)
            rnd.deal()

            if rnd.has_card_based_dak():
                agent.reset_episode()
                continue

            tasmiya = TasmiyaEngine()
            try:
                res = tasmiya.run(players=players, agents=agents_list, sahib_al_qabool_id=0)
            except (ValueError, Exception):
                agent.reset_episode()
                continue

            if res.is_dak:
                agent.reset_episode()
                continue

            rnd.state.trump_suit = res.trump_suit
            rnd.state.winning_bidder_id = res.winning_bidder_id
            rnd.next_leading_player_id = res.winning_bidder_id
            env = WistEnvironment(rnd.state)
            tt = {0: 0, 1: 0}

            for _ in range(13):
                lid = rnd.next_leading_player_id
                rnd.state.current_trick = Trick(leading_player_id=lid)
                for pid in [(lid + j) % 4 for j in range(4)]:
                    obs = env.observe(pid)
                    action = agents_list[pid].act(obs)
                    env.apply_action(action)
                trick = rnd.state.current_trick
                w = trick_winner(trick, rnd.state.trump_suit)
                rnd.state.completed_tricks.append(trick)
                rnd.state.current_trick = None
                rnd.next_leading_player_id = w
                tt[players[w].team_id] += 1
                agent.trick_reward(won=(players[w].team_id == 0))

            scores = score_shota(
                playing_team_id=res.playing_team_id,
                defending_team_id=1 - res.playing_team_id,
                bid=res.winning_bid_value,
                playing_team_tricks=tt[res.playing_team_id],
                defending_team_tricks=tt[1 - res.playing_team_id],
            )
            agent.reward(float(scores[0]))

            team_scores[0] += scores.get(0, 0)
            team_scores[1] += scores.get(1, 0)
            shota_count += 1
            shotas_done += 1

            if milestone_callback:
                bid_met = tt[res.playing_team_id] >= res.winning_bid_value
                milestone_callback(tt, res.winning_bid_value, res.playing_team_id, bid_met, scores)

            # Epsilon decay.
            if agent.episodes_trained % 50 == 0 and agent.epsilon > 0.03:
                agent.epsilon *= 0.98

        if shota_count > 0:
            win_history.append(team_scores[0] > team_scores[1])

    return win_history
