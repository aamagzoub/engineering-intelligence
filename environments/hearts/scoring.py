"""
Hearts scoring engine.

Zero-sum scoring — all four players' scores sum to 0 every shota.

Normal play: player_score = 5 - penalties_collected
    (hearts = 1 penalty each, Queen of Spades = 7 penalties)

Special scenarios (override normal):
    1. All tricks to one player: +18 for them, -6 each for others
    2. Full Gallon (exactly 1 with 0 tricks): +20 for them,
       others scored normally (5 - penalties) but adjusted so total = -20
    3. Half Gallon (exactly 2 with 0 tricks): +10 each,
       others scored normally but adjusted so total = -20
"""

from intelligence.core.cards.card import Card
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit


QUEEN_OF_SPADES = Card(suit=Suit.SPADES, rank=Rank.QUEEN)


def count_penalties(cards: list[Card]) -> int:
    """
    Count penalty points in a collection of won cards.

    - Each Heart: 1 penalty
    - Queen of Spades: 7 penalties
    """
    penalties = 0
    for card in cards:
        if card.suit == Suit.HEARTS:
            penalties += 1
        elif card == QUEEN_OF_SPADES:
            penalties += 7
    return penalties


def score_shota(collected_cards: dict[int, list[Card]],
                tricks_won: dict[int, int]) -> dict[int, int]:
    """
    Compute zero-sum scores for one shota.

    Args:
        collected_cards: player_id → list of all cards they won in tricks
        tricks_won: player_id → number of tricks won

    Returns:
        player_id → score for this shota (sum always = 0)
    """
    player_ids = sorted(collected_cards.keys())
    assert len(player_ids) == 4, "Hearts requires exactly 4 players."

    # Detect special scenarios.
    zero_trick_players = [pid for pid in player_ids if tricks_won[pid] == 0]
    all_trick_players = [pid for pid in player_ids if tricks_won[pid] == 13]

    # --- Scenario 1: One player won all 13 tricks ---
    if all_trick_players:
        winner = all_trick_players[0]
        scores = {}
        for pid in player_ids:
            if pid == winner:
                scores[pid] = 18
            else:
                scores[pid] = -6
        return scores

    # --- Scenario 2: Full Gallon (exactly 1 player with 0 tricks) ---
    if len(zero_trick_players) == 1:
        gallon_player = zero_trick_players[0]
        other_players = [pid for pid in player_ids if pid != gallon_player]

        # Gallon player gets +20.
        scores = {gallon_player: 20}

        # Others split -20 based on their collected penalties.
        penalties = {pid: count_penalties(collected_cards[pid]) for pid in other_players}
        total_penalties = sum(penalties.values())

        if total_penalties == 0:
            # Edge case: no penalties collected by others (shouldn't happen
            # since 20 penalty points exist, but handle gracefully).
            for pid in other_players:
                scores[pid] = -20 // 3  # -6 each, remainder handled below
            # Adjust for rounding.
            remainder = -20 - sum(scores[pid] for pid in other_players)
            scores[other_players[0]] += remainder
        else:
            # Distribute -20 proportionally to penalties collected.
            # Use raw penalties as scores: score = -(penalties)
            # But must sum to -20.
            # Normal formula: 5 - penalty for each, but only 3 players share.
            # Actually: each other player's score = -penalty (their share of -20).
            # Since total penalties among them = 20, their scores = -penalties.
            for pid in other_players:
                scores[pid] = -penalties[pid]

        # Verify zero-sum.
        _enforce_zero_sum(scores, player_ids)
        return scores

    # --- Scenario 3: Half Gallon (exactly 2 players with 0 tricks) ---
    if len(zero_trick_players) == 2:
        other_players = [pid for pid in player_ids if pid not in zero_trick_players]

        # Gallon players get +10 each.
        scores = {pid: 10 for pid in zero_trick_players}

        # Others split -20 based on their collected penalties.
        penalties = {pid: count_penalties(collected_cards[pid]) for pid in other_players}
        total_penalties = sum(penalties.values())

        if total_penalties == 0:
            for pid in other_players:
                scores[pid] = -10
        else:
            # Distribute -20 proportionally.
            for pid in other_players:
                scores[pid] = -penalties[pid]

        # Verify zero-sum.
        _enforce_zero_sum(scores, player_ids)
        return scores

    # --- Scenario 4: Normal play (everyone won at least 1 trick) ---
    # Formula: player_score = 5 - penalties_collected
    penalties = {pid: count_penalties(collected_cards[pid]) for pid in player_ids}
    scores = {pid: 5 - penalties[pid] for pid in player_ids}

    # This naturally sums to 0: (4 × 5) - total_penalties = 20 - 20 = 0.
    return scores


def _enforce_zero_sum(scores: dict[int, int], player_ids: list[int]) -> None:
    """
    Verify and adjust scores to ensure zero-sum.
    Rounding errors are distributed to the player with the most penalties.
    """
    total = sum(scores[pid] for pid in player_ids)
    if total != 0:
        # Find the player with the lowest score (most penalized) and adjust.
        min_player = min(player_ids, key=lambda pid: scores[pid])
        scores[min_player] -= total
