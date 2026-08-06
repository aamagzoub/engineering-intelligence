"""
Game engine — manages Wist game flow (deals, tricks, scoring).

Separated from display logic for clarity and testability.
"""

from environments.wist.environment import WistEnvironment
from environments.wist.round import Round
from environments.wist.rules import trick_winner
from environments.wist.scoring import score_shota
from environments.wist.setup import create_standard_players
from environments.wist.tasmiya_engine import TasmiyaEngine
from environments.wist.trick import Trick

from gui_wist_discovery.constants import SUIT_ORDER, RANK_ORDER


def sort_hand(hand):
    """Sort a hand by suit then rank for consistent display."""
    return sorted(hand, key=lambda c: (SUIT_ORDER[c.suit], RANK_ORDER[c.rank]))


def try_setup_shota(agents, shota_num, game_num, max_attempts=10):
    """
    Attempt to set up a valid shota (re-deal on dak/error).

    Returns (players, round, tasmiya_result, log_messages) on success,
    or (None, None, None, log_messages) if all attempts fail.
    """
    logs = []
    discovery = agents[0]  # Main learning agent.

    for _ in range(max_attempts):
        players = create_standard_players()
        agents_list = [agents[0], agents[1], agents[0], agents[1]]
        rnd = Round(players)
        rnd.deal()

        if rnd.has_card_based_dak():
            discovery.reset_episode()
            logs.append("  Dak — re-dealing")
            continue

        tasmiya = TasmiyaEngine()
        qabool_id = (shota_num - 1) % 4
        is_first = (game_num == 1 and shota_num == 1)
        try:
            result = tasmiya.run(
                players=players, agents=agents_list,
                sahib_al_qabool_id=qabool_id, is_first_shota=is_first,
            )
        except (ValueError, Exception):
            discovery.reset_episode()
            logs.append("  Bidding error — re-dealing")
            continue

        if result.is_dak:
            discovery.reset_episode()
            logs.append("  Pass Dak — re-dealing")
            continue

        return players, rnd, result, logs

    logs.append("  Could not deal a valid shota — scoring 0/0")
    return None, None, None, logs


def play_trick(rnd, env, agents_list, players, discovery_agent, use_mcts=True):
    """
    Play a single trick. Returns (trick_cards, winner_id, winner_team_id).

    trick_cards: list of (player_id, card) tuples.
    """
    lid = rnd.next_leading_player_id
    rnd.state.current_trick = Trick(leading_player_id=lid)
    play_order = [(lid + i) % 4 for i in range(4)]

    if use_mcts:
        discovery_agent._mcts_context = {
            "round_state": rnd.state,
            "players": players,
            "trump_suit": rnd.state.trump_suit,
            "num_simulations": 50,
        }

    trick_cards = []
    for pid in play_order:
        obs = env.observe(pid)
        action = agents_list[pid].act(obs)
        env.apply_action(action)
        trick_cards.append((pid, action.card))

    discovery_agent._mcts_context = None

    completed = rnd.state.current_trick
    winner = trick_winner(completed, rnd.state.trump_suit)
    rnd.state.completed_tricks.append(completed)
    rnd.state.current_trick = None
    rnd.next_leading_player_id = winner

    winner_team = players[winner].team_id
    return trick_cards, winner, winner_team


def score_completed_shota(tasmiya_result, team_tricks):
    """Score a completed shota. Returns scores dict {team_id: score}."""
    res = tasmiya_result
    return score_shota(
        playing_team_id=res.playing_team_id,
        defending_team_id=1 - res.playing_team_id,
        bid=res.winning_bid_value,
        playing_team_tricks=team_tricks[res.playing_team_id],
        defending_team_tricks=team_tricks[1 - res.playing_team_id],
    )
