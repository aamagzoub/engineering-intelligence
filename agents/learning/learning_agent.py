"""
Learning Wist Agent — Monte Carlo Policy Improvement.

Instead of Q-learning (which updates per-step and struggles with
large state spaces), this agent uses Monte Carlo methods:

1. Play a full Shota (13 tricks).
2. At the end, assign credit to every (state, action) pair
   based on whether the team WON or LOST that Shota.
3. Over thousands of games, the agent learns which actions
   in which states lead to winning.

Why this works better than Q-learning for cards:
- Card games have delayed rewards (you don't know if a play
  was good until the Shota ends).
- Monte Carlo uses the ACTUAL outcome, not estimated next-state values.
- With coarse state encoding, similar situations get grouped together
  and the agent learns general strategies.

The agent can be saved/loaded as JSON.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from environments.wist.actions import BidAction, PassAction, PlayCardAction
from environments.wist.observation import BiddingObservation, WistObservation
from environments.wist.rules import legal_cards, rank_value
from intelligence.core.action import Action
from intelligence.core.agent import Agent
from intelligence.core.cards.card import Card
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit
from intelligence.core.observation import Observation


# ---------------------------------------------------------------
# State / Action encoding
# ---------------------------------------------------------------

SUIT_INDEX = {Suit.SPADES: 0, Suit.HEARTS: 1, Suit.CLUBS: 2, Suit.DIAMONDS: 3}


def encode_state(obs: WistObservation) -> str:
    """
    Coarse state encoding for card play decisions.

    Features:
    - Position: leading / 2nd / 3rd / 4th
    - My trump strength: none(0), some(1-2), strong(3+)
    - My high cards: none, some, many
    - Can follow suit: yes/no
    - Partner winning current trick: yes/no/na
    - Game phase: early(10+), mid(6-9), late(1-5) cards left
    """
    hand = obs.hand
    trump = obs.trump_suit

    # Position in trick.
    n_played = 0
    leading_suit = None
    if obs.current_trick and obs.current_trick.played_cards:
        n_played = len(obs.current_trick.played_cards)
        leading_suit = obs.current_trick.leading_suit
    pos = str(n_played)  # "0","1","2","3"

    # Trump strength.
    trump_count = sum(1 for c in hand if c.suit == trump) if trump else 0
    ts = "0" if trump_count == 0 else ("1" if trump_count <= 2 else "2")

    # High card count (A, K, Q in hand).
    highs = sum(1 for c in hand if c.rank in (Rank.ACE, Rank.KING, Rank.QUEEN))
    hc = "0" if highs == 0 else ("1" if highs <= 2 else "2")

    # Can follow suit.
    cf = "1" if (leading_suit and any(c.suit == leading_suit for c in hand)) else "0"

    # Partner winning.
    pw = "U"
    if obs.current_trick and obs.current_trick.played_cards and trump:
        best_id = obs.current_trick.played_cards[0].player_id
        best_card = obs.current_trick.played_cards[0].card
        ls = obs.current_trick.leading_suit
        for pc in obs.current_trick.played_cards[1:]:
            c = pc.card
            if c.suit == trump and best_card.suit != trump:
                best_id, best_card = pc.player_id, c
            elif c.suit == trump and best_card.suit == trump:
                if rank_value(c.rank) > rank_value(best_card.rank):
                    best_id, best_card = pc.player_id, c
            elif c.suit == ls and best_card.suit == ls and best_card.suit != trump:
                if rank_value(c.rank) > rank_value(best_card.rank):
                    best_id, best_card = pc.player_id, c
        my_team = {obs.player_id, (obs.player_id + 2) % 4}
        pw = "Y" if best_id in my_team else "N"

    # Game phase.
    cl = len(hand)
    ph = "E" if cl >= 10 else ("M" if cl >= 6 else "L")

    return f"{pos}{ts}{hc}{cf}{pw}{ph}"


def encode_action(card: Card, trump: Suit | None, leading_suit: Suit | None) -> str:
    """
    Coarse action encoding.

    Encodes the TYPE of play, not the exact card:
    - Is it trump? T/N
    - Is it following suit? F/O (follow/off)
    - Rank tier: H(igh: A,K,Q), M(id: J,10,9), L(ow: 8-2)
    """
    is_trump = "T" if card.suit == trump else "N"
    follows = "F" if (leading_suit and card.suit == leading_suit) else "O"

    rv = rank_value(card.rank)
    if rv >= 12:  # Q, K, A
        tier = "H"
    elif rv >= 9:  # 9, 10, J
        tier = "M"
    else:
        tier = "L"

    return f"{is_trump}{follows}{tier}"


# ---------------------------------------------------------------
# Learning Agent
# ---------------------------------------------------------------


class LearningAgent(Agent):
    """
    Monte Carlo learning agent for Wist.

    Learns by playing full Shotas and updating action values
    based on whether the team won or lost.
    """

    def __init__(self, epsilon: float = 0.3, training: bool = True) -> None:
        # Action value table: state → {action: average_return}
        self.q_table: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # Visit count for averaging: state → {action: count}
        self.n_table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        self.epsilon = epsilon
        self.training = training

        # Episode memory: list of (state, action_key) for current Shota.
        self._episode: list[tuple[str, str]] = []

        self.total_updates = 0
        self.episodes_trained = 0

    def act(self, observation: Observation) -> Action:
        if isinstance(observation, BiddingObservation):
            return self._act_bidding(observation)
        if isinstance(observation, WistObservation):
            return self._act_play(observation)
        raise TypeError(f"Unsupported observation: {type(observation).__name__}")

    # ----------------------------------------------------------
    # Card play
    # ----------------------------------------------------------

    def _act_play(self, obs: WistObservation) -> Action:
        if not obs.hand:
            raise ValueError("Cannot act with empty hand.")

        leading_suit = None
        if obs.current_trick:
            leading_suit = obs.current_trick.leading_suit

        must_lead_trump = None
        if obs.must_lead_trump and obs.trump_suit:
            must_lead_trump = obs.trump_suit

        playable = legal_cards(obs.hand, leading_suit, must_lead_trump)

        if len(playable) == 1:
            card = playable[0]
        elif self.training and random.random() < self.epsilon:
            card = random.choice(playable)
        else:
            card = self._best_card(obs, playable)

        # Record state-action for this episode.
        if self.training:
            state = encode_state(obs)
            action_key = encode_action(card, obs.trump_suit, leading_suit)
            self._episode.append((state, action_key))

        return PlayCardAction(player_id=obs.player_id, card=card)

    def _best_card(self, obs: WistObservation, playable: list[Card]) -> Card:
        """Pick the card whose action-type has the best learned value."""
        state = encode_state(obs)
        q_values = self.q_table[state]

        leading_suit = None
        if obs.current_trick:
            leading_suit = obs.current_trick.leading_suit

        best_card = playable[0]
        best_q = float("-inf")

        for card in playable:
            key = encode_action(card, obs.trump_suit, leading_suit)
            q = q_values[key]
            if q > best_q:
                best_q = q
                best_card = card

        return best_card

    # ----------------------------------------------------------
    # Bidding (heuristic — learning focuses on play)
    # ----------------------------------------------------------

    def _act_bidding(self, obs: BiddingObservation) -> Action:
        hand = obs.hand
        suit_counts = Counter(card.suit for card in hand)
        longest = max(suit_counts.values()) if suit_counts else 0

        if longest >= 8:
            return PassAction(player_id=obs.player_id)

        bid_value = longest + 3
        bid_value = max(7, min(bid_value, 13))

        high_count = sum(1 for c in hand if c.rank in (Rank.ACE, Rank.KING, Rank.QUEEN))

        if obs.is_sahib_al_qabool:
            if obs.current_highest_bid is None:
                if high_count >= 2 or longest >= 5:
                    return BidAction(player_id=obs.player_id, value=max(7, bid_value - 1))
                return PassAction(player_id=obs.player_id)
            else:
                if bid_value >= obs.current_highest_bid and high_count >= 3:
                    return BidAction(player_id=obs.player_id, value=obs.current_highest_bid)
                return PassAction(player_id=obs.player_id)
        else:
            if high_count < 3:
                return PassAction(player_id=obs.player_id)
            if obs.is_opening_bid and bid_value > 11:
                return PassAction(player_id=obs.player_id)
            if obs.current_highest_bid and bid_value <= obs.current_highest_bid:
                return PassAction(player_id=obs.player_id)
            return BidAction(player_id=obs.player_id, value=bid_value)

    # ----------------------------------------------------------
    # Learning: Monte Carlo update
    # ----------------------------------------------------------

    def reward_trick(self, won: bool) -> None:
        """Per-trick reward (not used in MC, but kept for interface compat)."""
        pass

    def reward_shota(self, team_won_shota: bool, bid_met: bool) -> None:
        """
        Called at end of Shota. This is where Monte Carlo learning happens.

        Every (state, action) visited during this Shota gets updated
        based on whether we won.
        """
        if not self.training or not self._episode:
            self._episode.clear()
            return

        # Compute return for this episode.
        if team_won_shota:
            G = 1.0
        else:
            G = -1.0

        # Bonus for meeting bid (extra signal).
        if bid_met:
            G += 0.5

        # First-visit Monte Carlo: update each unique (state, action) once.
        visited = set()
        for state, action_key in self._episode:
            sa = (state, action_key)
            if sa in visited:
                continue
            visited.add(sa)

            # Incremental mean update.
            self.n_table[state][action_key] += 1
            n = self.n_table[state][action_key]
            old_q = self.q_table[state][action_key]
            # Running average: Q = Q + (G - Q) / N
            self.q_table[state][action_key] = old_q + (G - old_q) / n
            self.total_updates += 1

        self.episodes_trained += 1
        self._episode.clear()

    def reset_episode(self) -> None:
        """Clear episode memory (call on Dak / skipped Shota)."""
        self._episode.clear()

    def decay_epsilon(self, min_epsilon: float = 0.05, decay_rate: float = 0.9995) -> None:
        """Slowly reduce exploration."""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    # ----------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "q_table": {k: dict(v) for k, v in self.q_table.items()},
            "n_table": {k: dict(v) for k, v in self.n_table.items()},
            "epsilon": self.epsilon,
            "total_updates": self.total_updates,
            "episodes_trained": self.episodes_trained,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str | Path, training: bool = False) -> "LearningAgent":
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)

        agent = cls(epsilon=data.get("epsilon", 0.05), training=training)
        for state, actions in data.get("q_table", {}).items():
            for action_key, value in actions.items():
                agent.q_table[state][action_key] = value
        for state, actions in data.get("n_table", {}).items():
            for action_key, value in actions.items():
                agent.n_table[state][action_key] = value
        agent.total_updates = data.get("total_updates", 0)
        agent.episodes_trained = data.get("episodes_trained", 0)
        return agent

    @property
    def q_table_size(self) -> int:
        return sum(len(v) for v in self.q_table.values())
