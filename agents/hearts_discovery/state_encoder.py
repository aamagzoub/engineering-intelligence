"""
Domain-agnostic state encoder for the discovery agent.

CRITICAL: This encoder has ZERO knowledge of Hearts rules or scoring.
It treats all suits equally. It does not know that hearts are special.
It does not know that Queen of Spades is special.

If the agent learns to treat certain suits/cards differently,
it discovers that entirely from the reward signal.
"""

from intelligence.core.cards.card import Card
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit
from environments.hearts.observation import HeartsObservation, PassingObservation


# Generic rank values for ordering (not Hearts-specific — same for any card game).
_RANK_ORDER = {
    Rank.TWO: 2, Rank.THREE: 3, Rank.FOUR: 4, Rank.FIVE: 5,
    Rank.SIX: 6, Rank.SEVEN: 7, Rank.EIGHT: 8, Rank.NINE: 9,
    Rank.TEN: 10, Rank.JACK: 11, Rank.QUEEN: 12, Rank.KING: 13, Rank.ACE: 14,
}

# Consistent suit indexing (arbitrary order — no preference built in).
_SUIT_INDEX = {Suit.SPADES: 0, Suit.HEARTS: 1, Suit.CLUBS: 2, Suit.DIAMONDS: 3}


def encode_play_state(obs: HeartsObservation) -> str:
    """
    Encode trick-play state into a compact string.

    Features (all domain-agnostic):
    - Position in trick: 0/1/2/3
    - Cards in hand: bucket (S=1-4, M=5-9, L=10-13)
    - Trick number: bucket (E=1-4, M=5-9, L=10-13)
    - Can follow suit: Y/N (are legal cards same suit as led?)
    - Number of legal cards: bucket (1/2-3/4-6/7+)
    - My tricks won so far: bucket (0/1-2/3-5/6+)
    - Leading player has most tricks: Y/N/T (yes/no/tied)
    - Suits in hand: count (1/2/3/4)
    - High cards (A,K,Q) in hand: bucket (0/1-2/3-4/5+)
    """
    hand = obs.hand
    n_played_in_trick = len(obs.current_trick_cards)

    # Position in trick.
    pos = str(n_played_in_trick)

    # Cards in hand bucket.
    hand_size = len(hand)
    if hand_size <= 4:
        hs = "S"
    elif hand_size <= 9:
        hs = "M"
    else:
        hs = "L"

    # Trick number bucket.
    tn = obs.trick_number
    if tn <= 4:
        tb = "E"
    elif tn <= 9:
        tb = "M"
    else:
        tb = "L"

    # Can follow suit (legal cards match a led suit).
    led_suit = None
    if obs.current_trick_cards:
        led_suit = obs.current_trick_cards[0][1].suit
    if led_suit is not None:
        can_follow = "Y" if any(c.suit == led_suit for c in hand) else "N"
    else:
        can_follow = "L"  # Leading — no suit to follow.

    # Number of legal cards.
    n_legal = len(obs.legal_cards)
    if n_legal <= 1:
        lc = "1"
    elif n_legal <= 3:
        lc = "2"
    elif n_legal <= 6:
        lc = "3"
    else:
        lc = "4"

    # My tricks won.
    my_tricks = obs.tricks_won_per_player.get(obs.player_id, 0)
    if my_tricks == 0:
        mt = "0"
    elif my_tricks <= 2:
        mt = "1"
    elif my_tricks <= 5:
        mt = "2"
    else:
        mt = "3"

    # Suits in hand.
    suits_count = len(set(c.suit for c in hand))
    sc = str(min(suits_count, 4))

    # High cards in hand (A, K, Q — domain-agnostic "high" definition).
    high_ranks = {Rank.ACE, Rank.KING, Rank.QUEEN}
    highs = sum(1 for c in hand if c.rank in high_ranks)
    if highs == 0:
        hc = "0"
    elif highs <= 2:
        hc = "1"
    elif highs <= 4:
        hc = "2"
    else:
        hc = "3"

    return f"{pos}{hs}{tb}{can_follow}{lc}{mt}{sc}{hc}"


def encode_play_action(card: Card, obs: HeartsObservation) -> str:
    """
    Encode which card was played into a context-aware action key.

    Features (domain-agnostic):
    - Rank tier: H(igh: A,K), U(pper: Q,J), M(id: 10,9,8), L(ow: 7-2)
    - Follows led suit: Y/N/L (yes/no/leading)
    - Is highest card I could play: Y/N
    - Is lowest card I could play: Y/N
    """
    rv = _RANK_ORDER[card.rank]

    # Rank tier.
    if rv >= 13:
        tier = "H"
    elif rv >= 11:
        tier = "U"
    elif rv >= 8:
        tier = "M"
    else:
        tier = "L"

    # Follows suit.
    led_suit = None
    if obs.current_trick_cards:
        led_suit = obs.current_trick_cards[0][1].suit
    if led_suit is None:
        follows = "L"
    elif card.suit == led_suit:
        follows = "Y"
    else:
        follows = "N"

    # Is highest/lowest legal card.
    legal_ranks = [_RANK_ORDER[c.rank] for c in obs.legal_cards]
    is_highest = "Y" if rv == max(legal_ranks) else "N"
    is_lowest = "Y" if rv == min(legal_ranks) else "N"

    return f"{tier}{follows}{is_highest}{is_lowest}"


def encode_passing_state(obs: PassingObservation) -> str:
    """
    Encode hand for passing decisions.

    Features:
    - High card count: bucket
    - Suit distribution: longest suit length
    - Void potential: number of suits with <=2 cards
    """
    hand = obs.hand
    suit_counts = {}
    for c in hand:
        suit_counts[c.suit] = suit_counts.get(c.suit, 0) + 1

    # High cards.
    high_ranks = {Rank.ACE, Rank.KING, Rank.QUEEN}
    highs = sum(1 for c in hand if c.rank in high_ranks)
    if highs <= 1:
        hc = "0"
    elif highs <= 3:
        hc = "1"
    else:
        hc = "2"

    # Longest suit.
    longest = max(suit_counts.values()) if suit_counts else 0
    if longest <= 4:
        ls = "S"
    elif longest <= 6:
        ls = "M"
    else:
        ls = "L"

    # Short suits (potential voids after passing).
    short_suits = sum(1 for cnt in suit_counts.values() if cnt <= 2)
    ss = str(min(short_suits, 3))

    return f"{hc}{ls}{ss}"


def encode_passing_action(cards: tuple[Card, ...], hand: list[Card]) -> str:
    """
    Encode which 4 cards were passed.

    Features:
    - Average rank tier of passed cards: H/M/L
    - Number of distinct suits passed: 1/2/3/4
    - Passed any high cards (A,K,Q): Y/N
    """
    avg_rank = sum(_RANK_ORDER[c.rank] for c in cards) / 4.0

    if avg_rank >= 11:
        tier = "H"
    elif avg_rank >= 7:
        tier = "M"
    else:
        tier = "L"

    suits_passed = len(set(c.suit for c in cards))
    sp = str(min(suits_passed, 4))

    high_ranks = {Rank.ACE, Rank.KING, Rank.QUEEN}
    has_high = "Y" if any(c.rank in high_ranks for c in cards) else "N"

    return f"{tier}{sp}{has_high}"
