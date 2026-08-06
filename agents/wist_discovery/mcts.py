"""
Monte Carlo Tree Search for Wist card play.

Domain-agnostic look-ahead: simulates random future tricks to determine
which card leads to the best outcome. Uses only:
- Environment (game rules for simulation)
- Legal moves (what's playable)
- Score signal (trick wins counted)

No strategy knowledge. No heuristics. Just simulation.
"""

import random
from collections import Counter
from copy import deepcopy

from environments.wist.rules import legal_cards, trick_winner, rank_value
from environments.wist.trick import Trick
from intelligence.core.cards.suit import Suit


def mcts_choose_card(obs, playable_cards, round_state, players, trump_suit,
                     num_simulations: int = 100) -> object:
    """
    Use MCTS to choose the best card from playable_cards.
    
    For each legal card, simulate num_simulations random games to completion.
    Pick the card that leads to the most trick wins on average.
    
    Args:
        obs: Current WistObservation for the player.
        playable_cards: List of legal cards to choose from.
        round_state: Current round state (completed tricks, current trick, etc.).
        players: List of players with their current hands.
        trump_suit: The trump suit for this shota.
        num_simulations: How many random simulations per card.
    
    Returns:
        The best card to play.
    """
    if len(playable_cards) == 1:
        return playable_cards[0]

    player_id = obs.player_id
    my_team = 0 if player_id in (0, 2) else 1

    # For each legal card, simulate and count wins.
    card_scores = {}

    for card in playable_cards:
        total_wins = 0

        for _ in range(num_simulations):
            wins = _simulate_from_card(
                card, player_id, my_team, obs, round_state, players, trump_suit
            )
            total_wins += wins

        card_scores[id(card)] = total_wins / num_simulations

    # Pick card with best average trick wins.
    best_card = max(playable_cards, key=lambda c: card_scores.get(id(c), 0))
    return best_card


def _simulate_from_card(card, player_id, my_team, obs, round_state, players, trump_suit) -> int:
    """
    Simulate the rest of the shota after playing `card`.
    Returns: number of tricks my_team wins from this point forward.
    """
    # Build remaining hands for all players (what we know + random for hidden).
    hands = _build_simulated_hands(player_id, card, obs, players)

    # Figure out the current trick state.
    current_trick_cards = []
    leading_suit = None
    next_player_after_me = None

    if obs.current_trick and obs.current_trick.played_cards:
        for pc in obs.current_trick.played_cards:
            current_trick_cards.append((pc.player_id, pc.card))
        leading_suit = obs.current_trick.leading_suit

    # Add our card to the current trick.
    current_trick_cards.append((player_id, card))
    if leading_suit is None:
        leading_suit = card.suit

    # Determine who still needs to play in this trick.
    players_in_trick = {pid for pid, _ in current_trick_cards}
    trick_order = []
    # Find who leads this trick.
    if obs.current_trick and obs.current_trick.leading_player_id is not None:
        leader = obs.current_trick.leading_player_id
    else:
        leader = player_id  # We're leading.

    for i in range(4):
        pid = (leader + i) % 4
        if pid not in players_in_trick:
            trick_order.append(pid)

    # Simulate remaining players in current trick.
    for pid in trick_order:
        hand = hands.get(pid, [])
        if not hand:
            continue
        playable = legal_cards(hand, leading_suit, None)
        if playable:
            chosen = random.choice(playable)
            current_trick_cards.append((pid, chosen))
            hands[pid] = [c for c in hand if c is not chosen]

    # Resolve current trick winner.
    team_wins = 0
    winner = _resolve_trick_winner(current_trick_cards, trump_suit, leading_suit)
    if winner is not None:
        winner_team = 0 if winner in (0, 2) else 1
        if winner_team == my_team:
            team_wins += 1
        next_leader = winner
    else:
        next_leader = (player_id + 1) % 4

    # Count how many cards remain (determines tricks left).
    max_remaining = max(len(h) for h in hands.values()) if hands else 0

    # Simulate remaining tricks randomly.
    for _ in range(max_remaining):
        trick_cards = []
        t_leading_suit = None

        for i in range(4):
            pid = (next_leader + i) % 4
            hand = hands.get(pid, [])
            if not hand:
                continue
            playable = legal_cards(hand, t_leading_suit, None)
            if not playable:
                continue
            chosen = random.choice(playable)
            trick_cards.append((pid, chosen))
            hands[pid] = [c for c in hand if c is not chosen]
            if t_leading_suit is None:
                t_leading_suit = chosen.suit

        if len(trick_cards) == 4:
            w = _resolve_trick_winner(trick_cards, trump_suit, t_leading_suit)
            if w is not None:
                w_team = 0 if w in (0, 2) else 1
                if w_team == my_team:
                    team_wins += 1
                next_leader = w
            else:
                next_leader = (next_leader + 1) % 4
        else:
            break

    return team_wins


def _build_simulated_hands(player_id, card_played, obs, players) -> dict:
    """
    Build hands for simulation. We know our own hand (minus the card we play).
    For opponents, use their actual remaining cards (from players list).
    """
    hands = {}
    for p in players:
        if p.player_id == player_id:
            # Our hand minus the card we're playing.
            hands[p.player_id] = [c for c in p.hand if c is not card_played]
        else:
            # Opponent's actual hand (in self-play we can see it).
            hands[p.player_id] = list(p.hand)
    return hands


def _resolve_trick_winner(trick_cards, trump_suit, leading_suit) -> int:
    """Determine winner of a completed trick from (pid, card) pairs."""
    if len(trick_cards) < 4:
        return None

    # Check for trump cards.
    trump_plays = [(pid, c) for pid, c in trick_cards if c.suit == trump_suit]
    if trump_plays:
        winner_pid, _ = max(trump_plays, key=lambda x: rank_value(x[1].rank))
        return winner_pid

    # No trump — highest of leading suit.
    leading_plays = [(pid, c) for pid, c in trick_cards if c.suit == leading_suit]
    if leading_plays:
        winner_pid, _ = max(leading_plays, key=lambda x: rank_value(x[1].rank))
        return winner_pid

    return trick_cards[0][0]  # Fallback.
