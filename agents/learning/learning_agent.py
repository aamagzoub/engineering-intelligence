"""
Learning Wist Agent — Monte Carlo Policy Improvement.

Uses Monte Carlo methods for both card-play and bidding:
1. Play a full Shota (13 tricks).
2. At the end, assign credit to every (state, action) pair
   based on graduated rewards (tricks won, bid met, seek).
3. Over thousands of games, the agent learns optimal strategies.

The agent maintains two Q-tables:
- play_q: for card-play decisions (state → action → value)
- bid_q: for bidding decisions (state → action → value)
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


def encode_play_state(obs: WistObservation) -> str:
    """
    Rich state encoding for card play decisions.

    Features (11 dimensions):
    - Position in trick: 0/1/2/3
    - Trump strength: 0/1/2 (none / 1-2 / 3+)
    - High cards in hand: 0/1/2 (none / 1-2 / 3+)
    - Can follow suit: 0/1
    - Partner winning: Y/N/U (yes/no/unknown)
    - Game phase: E(arly 10+) / M(id 6-9) / L(ate 1-5)
    - Trick number bucket: A(1-4) / B(5-9) / C(10-13)
    - Team tricks differential: W(inning +3) / A(head +1-2) / T(ied) / B(ehind)
    - Am I the shooter's team: S/D (shooter/defender)
    - Leading suit is trump: Y/N
    - Bid difficulty: L(ow 7-8) / M(id 9-10) / H(igh 11-13)
    """
    hand = obs.hand
    trump = obs.trump_suit

    # Position in trick.
    n_played = 0
    leading_suit = None
    if obs.current_trick and obs.current_trick.played_cards:
        n_played = len(obs.current_trick.played_cards)
        leading_suit = obs.current_trick.leading_suit
    pos = str(n_played)

    # Trump strength.
    trump_count = sum(1 for c in hand if c.suit == trump) if trump else 0
    ts = "0" if trump_count == 0 else ("1" if trump_count <= 2 else "2")

    # High card count (A, K, Q).
    highs = sum(1 for c in hand if c.rank in (Rank.ACE, Rank.KING, Rank.QUEEN))
    hc = "0" if highs == 0 else ("1" if highs <= 2 else "2")

    # Can follow suit.
    cf = "1" if (leading_suit and any(c.suit == leading_suit for c in hand)) else "0"

    # Partner winning current trick.
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

    # Game phase (cards left in hand).
    cl = len(hand)
    ph = "E" if cl >= 10 else ("M" if cl >= 6 else "L")

    # Trick number bucket.
    trick_num = 14 - cl  # Approximate trick number.
    tn = "A" if trick_num <= 4 else ("B" if trick_num <= 9 else "C")

    # Team tricks differential.
    scores = obs.team_scores
    my_team_id = 0 if obs.player_id in (0, 2) else 1
    opp_team_id = 1 - my_team_id
    my_tricks = scores.get(my_team_id, 0)
    opp_tricks = scores.get(opp_team_id, 0)
    diff = my_tricks - opp_tricks
    if diff >= 3:
        td = "W"
    elif diff >= 1:
        td = "A"
    elif diff == 0:
        td = "T"
    else:
        td = "B"

    # Leading suit is trump.
    lt = "Y" if (leading_suit and leading_suit == trump) else "N"

    return f"{pos}{ts}{hc}{cf}{pw}{ph}{tn}{td}{lt}"


def encode_play_action(card: Card, trump: Suit | None, leading_suit: Suit | None) -> str:
    """
    Action encoding for card play.

    Encodes the TYPE of play:
    - Is it trump? T/N
    - Is it following suit? F/O (follow/off-suit)
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


def encode_bid_state(obs: BiddingObservation) -> str:
    """
    State encoding for bidding decisions.

    Features:
    - Longest suit count: 4/5/6/7
    - High cards (A,K,Q) count: 0/1/2/3+
    - Has existing bid: Y/N
    - Is Qabool: Y/N
    - Trump suit strength relative to longest
    """
    hand = obs.hand
    suit_counts = Counter(card.suit for card in hand)
    longest = max(suit_counts.values()) if suit_counts else 0
    longest_suit = max(suit_counts, key=suit_counts.get) if suit_counts else None

    # Longest suit bucket.
    ls = str(min(longest, 7))

    # High cards in longest suit.
    highs_in_longest = sum(1 for c in hand
                           if c.suit == longest_suit and c.rank in (Rank.ACE, Rank.KING, Rank.QUEEN))
    hl = str(min(highs_in_longest, 3))

    # Total high cards.
    total_highs = sum(1 for c in hand if c.rank in (Rank.ACE, Rank.KING, Rank.QUEEN))
    th = str(min(total_highs, 4))

    # Has existing bid.
    hb = "Y" if obs.current_highest_bid else "N"

    # Is Qabool.
    iq = "Y" if obs.is_sahib_al_qabool else "N"

    return f"{ls}{hl}{th}{hb}{iq}"


def encode_bid_action(action: Action) -> str:
    """Encode bidding action."""
    if isinstance(action, PassAction):
        return "PASS"
    if isinstance(action, BidAction):
        v = action.value
        if v <= 8:
            return "LOW"
        elif v <= 10:
            return "MID"
        else:
            return "HIGH"
    return "PASS"


# ---------------------------------------------------------------
# Learning Agent
# ---------------------------------------------------------------


class LearningAgent(Agent):
    """
    Monte Carlo learning agent for Wist with separate play and bid Q-tables.
    """

    def __init__(self, epsilon: float = 0.3, training: bool = True) -> None:
        # Play Q-table: state → {action: average_return}
        self.q_table: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.n_table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Bid Q-table.
        self.bid_q: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.bid_n: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        self.epsilon = epsilon
        self.training = training

        # Episode memory for current Shota.
        self._play_episode: list[tuple[str, str]] = []
        self._bid_episode: list[tuple[str, str]] = []

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
            state = encode_play_state(obs)
            action_key = encode_play_action(card, obs.trump_suit, leading_suit)
            self._play_episode.append((state, action_key))

        return PlayCardAction(player_id=obs.player_id, card=card)

    def _best_card(self, obs: WistObservation, playable: list[Card]) -> Card:
        """Pick the card whose action-type has the best learned value."""
        state = encode_play_state(obs)
        q_values = self.q_table[state]

        leading_suit = None
        if obs.current_trick:
            leading_suit = obs.current_trick.leading_suit

        best_card = playable[0]
        best_q = float("-inf")

        for card in playable:
            key = encode_play_action(card, obs.trump_suit, leading_suit)
            q = q_values[key]
            if q > best_q:
                best_q = q
                best_card = card

        return best_card

    # ----------------------------------------------------------
    # Bidding (learned)
    # ----------------------------------------------------------

    def _act_bidding(self, obs: BiddingObservation) -> Action:
        hand = obs.hand
        suit_counts = Counter(card.suit for card in hand)
        longest = max(suit_counts.values()) if suit_counts else 0

        # Can't bid with 8+ in one suit (Dak territory).
        if longest >= 8:
            action = PassAction(player_id=obs.player_id)
            if self.training:
                state = encode_bid_state(obs)
                self._bid_episode.append((state, encode_bid_action(action)))
            return action

        # Calculate feasible bid range.
        min_bid = longest + 3
        min_bid = max(7, min_bid)

        # Explore or exploit.
        if self.training and random.random() < self.epsilon:
            action = self._random_bid(obs, min_bid, longest)
        else:
            action = self._best_bid(obs, min_bid, longest)

        if self.training:
            state = encode_bid_state(obs)
            self._bid_episode.append((state, encode_bid_action(action)))

        return action

    def _random_bid(self, obs: BiddingObservation, min_bid: int, longest: int) -> Action:
        """Random exploration bid."""
        # 40% chance to pass, 60% chance to bid.
        if random.random() < 0.4:
            return PassAction(player_id=obs.player_id)

        bid_value = random.randint(max(7, min_bid), min(13, min_bid + 3))

        # Respect constraints.
        if obs.current_highest_bid and bid_value <= obs.current_highest_bid:
            if obs.is_sahib_al_qabool:
                bid_value = obs.current_highest_bid  # Qabool can match.
            else:
                return PassAction(player_id=obs.player_id)

        if not obs.is_sahib_al_qabool and obs.is_opening_bid and bid_value > 11:
            bid_value = 11

        return BidAction(player_id=obs.player_id, value=bid_value)

    def _best_bid(self, obs: BiddingObservation, min_bid: int, longest: int) -> Action:
        """Choose best bid based on learned Q-values."""
        state = encode_bid_state(obs)
        q_values = self.bid_q[state]

        # Get Q-values for pass vs bid levels.
        pass_q = q_values.get("PASS", 0.0)
        low_q = q_values.get("LOW", 0.0)
        mid_q = q_values.get("MID", 0.0)
        high_q = q_values.get("HIGH", 0.0)

        # Determine best action.
        best_action = "PASS"
        best_q = pass_q
        for key, q in [("LOW", low_q), ("MID", mid_q), ("HIGH", high_q)]:
            if q > best_q:
                best_q = q
                best_action = key

        if best_action == "PASS":
            return PassAction(player_id=obs.player_id)

        # Convert to actual bid value.
        if best_action == "LOW":
            bid_value = max(7, min_bid)
        elif best_action == "MID":
            bid_value = max(9, min_bid + 1)
        else:
            bid_value = max(11, min_bid + 2)

        bid_value = min(bid_value, 13)

        # Respect opening bid max 11 rule.
        if not obs.is_sahib_al_qabool and obs.is_opening_bid and bid_value > 11:
            bid_value = 11

        # Respect constraints.
        if obs.current_highest_bid:
            if obs.is_sahib_al_qabool:
                if bid_value < obs.current_highest_bid:
                    bid_value = obs.current_highest_bid
            else:
                if bid_value <= obs.current_highest_bid:
                    return PassAction(player_id=obs.player_id)

        return BidAction(player_id=obs.player_id, value=bid_value)

    # ----------------------------------------------------------
    # Learning: Monte Carlo update with graduated rewards
    # ----------------------------------------------------------

    def reward_trick(self, won: bool) -> None:
        """Per-trick signal (not used in MC, kept for interface)."""
        pass

    def reward_shota(self, team_won_shota: bool, bid_met: bool,
                     my_tricks: int = 0, opp_tricks: int = 0,
                     was_shooter: bool = False, seek: bool = False) -> None:
        """
        End-of-Shota Monte Carlo update with graduated rewards.

        Reward structure:
        - Base: +1.0 for winning shota, -1.0 for losing
        - Bid bonus: +0.5 if our team met the bid as shooter
        - Bid penalty: -0.5 if we were shooter and failed
        - Trick margin: +0.1 per trick above opponent
        - Seek bonus: +2.0 for achieving seek
        - Seek penalty: -1.5 for being seeked against
        """
        if not self.training:
            self._play_episode.clear()
            self._bid_episode.clear()
            return

        # Compute return.
        G = 1.0 if team_won_shota else -1.0

        # Bid bonus/penalty.
        if was_shooter:
            G += 0.5 if bid_met else -0.5

        # Trick margin bonus.
        margin = my_tricks - opp_tricks
        G += margin * 0.1

        # Seek bonus.
        if seek:
            if my_tricks == 13:
                G += 2.0
            else:
                G -= 1.5

        # Clamp to reasonable range.
        G = max(-3.0, min(3.0, G))

        # Update play Q-table (first-visit MC).
        self._update_table(self.q_table, self.n_table, self._play_episode, G)

        # Update bid Q-table with same reward.
        self._update_table(self.bid_q, self.bid_n, self._bid_episode, G)

        self.episodes_trained += 1
        self._play_episode.clear()
        self._bid_episode.clear()

    def _update_table(self, q_table, n_table, episode, G):
        """First-visit Monte Carlo update on a Q-table."""
        visited = set()
        for state, action_key in episode:
            sa = (state, action_key)
            if sa in visited:
                continue
            visited.add(sa)

            n_table[state][action_key] += 1
            n = n_table[state][action_key]
            old_q = q_table[state][action_key]
            q_table[state][action_key] = old_q + (G - old_q) / n
            self.total_updates += 1

    def reset_episode(self) -> None:
        """Clear episode memory (call on Dak / skipped Shota)."""
        self._play_episode.clear()
        self._bid_episode.clear()

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
            "bid_q": {k: dict(v) for k, v in self.bid_q.items()},
            "bid_n": {k: dict(v) for k, v in self.bid_n.items()},
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
        for state, actions in data.get("bid_q", {}).items():
            for action_key, value in actions.items():
                agent.bid_q[state][action_key] = value
        for state, actions in data.get("bid_n", {}).items():
            for action_key, value in actions.items():
                agent.bid_n[state][action_key] = value
        agent.total_updates = data.get("total_updates", 0)
        agent.episodes_trained = data.get("episodes_trained", 0)
        return agent

    @property
    def q_table_size(self) -> int:
        play_size = sum(len(v) for v in self.q_table.values())
        bid_size = sum(len(v) for v in self.bid_q.values())
        return play_size + bid_size
