"""
Learning Wist Agent — Enhanced Architecture v2.

Uses TD(λ) with eligibility traces for temporal credit assignment,
richer state representation with card memory, and finer-grained
action encoding for better policy discrimination.

Enhancements:
1. TD(λ) with eligibility traces — credit assignment within a shota
2. Card memory — tracks played cards for better state awareness
3. Finer action encoding — distinguishes specific card plays better
4. Per-trick reward shaping (immediate feedback)
5. Opponent modeling — tracks opponent void/trump patterns
6. Prioritized experience replay — revisit high-impact experiences
7. Strategic pattern detection — seek/void/trump extraction awareness
8. Double Q-learning — reduces overestimation bias

The agent maintains:
- play_q / play_q2: double Q-tables for card-play
- bid_q: bidding Q-table (state → action → value)
- card_memory: tracks played cards within a shota
- opponent_model: learned opponent tendencies
- replay_buffer: prioritized experience memory
"""

import json
import math
import random
from collections import Counter, defaultdict, deque
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
# Constants
# ---------------------------------------------------------------

SUIT_INDEX = {Suit.SPADES: 0, Suit.HEARTS: 1, Suit.CLUBS: 2, Suit.DIAMONDS: 3}

HIGH_RANKS = {Rank.ACE, Rank.KING, Rank.QUEEN, Rank.JACK}

RANK_ORDER = [
    Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX,
    Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN,
    Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE
]


# ---------------------------------------------------------------
# Opponent Model — tracks opponent tendencies
# ---------------------------------------------------------------


class OpponentModel:
    """
    Lightweight opponent model that tracks observable patterns:
    - Which suits opponents have shown void in (didn't follow)
    - How often opponents trump when void
    - High card usage patterns per game phase
    """

    def __init__(self):
        # opponent_id → set of suits they've shown void in
        self.known_voids: dict[int, set] = defaultdict(set)
        # opponent_id → count of times they trumped when void
        self.trump_tendency: dict[int, int] = defaultdict(int)
        self.void_opportunities: dict[int, int] = defaultdict(int)

    def observe_play(self, player_id: int, card: Card,
                     leading_suit: Suit | None, trump: Suit | None):
        """Record what we learn from an opponent's play."""
        if leading_suit and card.suit != leading_suit:
            # They didn't follow suit — they're void.
            self.known_voids[player_id].add(leading_suit)
            self.void_opportunities[player_id] += 1
            if trump and card.suit == trump:
                self.trump_tendency[player_id] += 1

    def is_void(self, player_id: int, suit: Suit) -> bool:
        """Check if we know a player is void in a suit."""
        return suit in self.known_voids[player_id]

    def trump_aggression(self, player_id: int) -> float:
        """How often does this opponent trump when they can? 0.0 - 1.0"""
        opps = self.void_opportunities[player_id]
        if opps == 0:
            return 0.5  # Unknown — assume moderate
        return self.trump_tendency[player_id] / opps

    def reset(self):
        self.known_voids.clear()
        self.trump_tendency.clear()
        self.void_opportunities.clear()

    def encode_context(self, my_player_id: int, trump: Suit | None) -> str:
        """
        Encode opponent model state into a compact string feature.
        Returns: opponent void pressure level (how many known voids for opponents).
        """
        my_team = {my_player_id, (my_player_id + 2) % 4}
        opp_ids = [pid for pid in range(4) if pid not in my_team]
        total_voids = sum(len(self.known_voids[pid]) for pid in opp_ids)
        if total_voids == 0:
            return "0"
        elif total_voids <= 2:
            return "1"
        else:
            return "2"


# ---------------------------------------------------------------
# Prioritized Experience Replay Buffer
# ---------------------------------------------------------------


class ReplayBuffer:
    """
    Stores high-impact experiences (state, action, reward, next_state)
    and replays them periodically for faster learning.
    Prioritized by absolute TD error — surprising outcomes get replayed more.
    """

    def __init__(self, capacity: int = 5000):
        self.buffer: deque[tuple[str, str, float, str | None, float]] = deque(maxlen=capacity)
        # Each entry: (state, action, reward, next_state, priority)

    def add(self, state: str, action: str, reward: float,
            next_state: str | None, td_error: float):
        priority = abs(td_error) + 0.01  # Never zero priority
        self.buffer.append((state, action, reward, next_state, priority))

    def sample(self, batch_size: int = 16) -> list[tuple[str, str, float, str | None]]:
        """Sample experiences weighted by priority."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        if batch_size == 0:
            return []

        priorities = [entry[4] for entry in self.buffer]
        total = sum(priorities)
        probs = [p / total for p in priorities]

        indices = random.choices(range(len(self.buffer)), weights=probs, k=batch_size)
        return [(self.buffer[i][0], self.buffer[i][1],
                 self.buffer[i][2], self.buffer[i][3]) for i in indices]

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------
# State Encoding — Rich, multi-dimensional
# ---------------------------------------------------------------


def _count_high_remaining(suit: Suit, rank: Rank, played_cards: set) -> int:
    """Count how many cards higher than `rank` in `suit` are still unplayed."""
    rv = rank_value(rank)
    count = 0
    for r in RANK_ORDER:
        if rank_value(r) > rv:
            card_key = (suit, r)
            if card_key not in played_cards:
                count += 1
    return count


def encode_play_state(obs: WistObservation, card_memory: set,
                     opponent_model: OpponentModel | None = None) -> str:
    """
    Enhanced state encoding for card play decisions.

    Features (15 dimensions):
    - Position in trick: 0/1/2/3
    - Trump strength: 0/1/2/3 (none / 1-2 / 3-4 / 5+)
    - High cards in hand: 0/1/2/3 (A,K,Q,J count buckets)
    - Can follow suit: Y/N
    - Partner winning: Y/N/U (yes/no/unknown)
    - Game phase: E(arly 10+) / M(id 6-9) / L(ate 1-5)
    - Trick number bucket: A(1-4) / B(5-9) / C(10-13)
    - Team tricks differential: W(+3) / A(+1-2) / T(ied) / B(ehind) / F(ar behind -3)
    - Leading suit is trump: Y/N
    - Trump dominance: D(ominant) / S(trong) / W(eak) / N(one)
    - Void count: 0/1/2/3
    - Cards seen ratio: L(ow <25%) / M(id 25-60%) / H(igh >60%)
    - Seek potential: S(eeking - won all so far) / P(ossible) / N(o)
    - Opponent void pressure: 0/1/2
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

    # Trump strength (finer grained).
    trump_count = sum(1 for c in hand if c.suit == trump) if trump else 0
    if trump_count == 0:
        ts = "0"
    elif trump_count <= 2:
        ts = "1"
    elif trump_count <= 4:
        ts = "2"
    else:
        ts = "3"

    # High card count (A, K, Q, J).
    highs = sum(1 for c in hand if c.rank in HIGH_RANKS)
    hc = str(min(highs, 3))

    # Can follow suit.
    cf = "Y" if (leading_suit and any(c.suit == leading_suit for c in hand)) else "N"

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
    trick_num = 14 - cl
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
    elif diff >= -2:
        td = "B"
    else:
        td = "F"

    # Leading suit is trump.
    lt = "Y" if (leading_suit and leading_suit == trump) else "N"

    # Trump dominance — do I hold the highest remaining trump?
    td_dom = "N"
    if trump and trump_count > 0:
        my_trumps = [c for c in hand if c.suit == trump]
        my_best_trump = max(my_trumps, key=lambda c: rank_value(c.rank))
        higher_unplayed = _count_high_remaining(trump, my_best_trump.rank, card_memory)
        if higher_unplayed == 0:
            td_dom = "D"  # Dominant — I hold top trump
        elif higher_unplayed <= 1:
            td_dom = "S"  # Strong
        else:
            td_dom = "W"  # Weak

    # Void count.
    suits_in_hand = set(c.suit for c in hand)
    voids = 4 - len(suits_in_hand)
    vc = str(min(voids, 3))

    # Cards seen ratio.
    seen_pct = len(card_memory) / 52.0
    cs = "L" if seen_pct < 0.25 else ("M" if seen_pct < 0.60 else "H")

    # Seek potential — are we on track for seek?
    total_tricks = my_tricks + opp_tricks
    if total_tricks > 0 and opp_tricks == 0:
        sp = "S"  # Actively seeking — won all tricks so far
    elif total_tricks > 0 and opp_tricks <= 1 and my_tricks >= 4:
        sp = "P"  # Possible — opponent barely has tricks
    else:
        sp = "N"

    # Opponent void pressure from opponent model.
    ovp = "0"
    if opponent_model:
        ovp = opponent_model.encode_context(obs.player_id, trump)

    return f"{pos}{ts}{hc}{cf}{pw}{ph}{tn}{td}{lt}{td_dom}{vc}{cs}{sp}{ovp}"


def encode_play_action(card: Card, trump: Suit | None, leading_suit: Suit | None,
                       card_memory: set) -> str:
    """
    Enhanced action encoding for card play.

    Encodes the CONTEXT of the play (7 features):
    - Is it trump? T/N
    - Is it following suit? F/O (follow/off-suit)
    - Rank tier: H(igh: A,K), U(pper-mid: Q,J), M(id: 10,9,8), L(ow: 7-2)
    - Is it the highest remaining in its suit? Y/N
    - Is it creating/maintaining a void? V/X
    """
    is_trump = "T" if card.suit == trump else "N"
    follows = "F" if (leading_suit and card.suit == leading_suit) else "O"

    rv = rank_value(card.rank)
    if rv >= 13:  # K, A
        tier = "H"
    elif rv >= 11:  # Q, J
        tier = "U"
    elif rv >= 8:  # 10, 9, 8
        tier = "M"
    else:
        tier = "L"

    # Is this the highest remaining card in its suit?
    higher_unplayed = _count_high_remaining(card.suit, card.rank, card_memory)
    top = "Y" if higher_unplayed == 0 else "N"

    return f"{is_trump}{follows}{tier}{top}"


def encode_bid_state(obs: BiddingObservation) -> str:
    """
    Enhanced state encoding for bidding decisions.

    Features (7 dimensions):
    - Longest suit count: 4/5/6/7
    - High cards in longest suit: 0/1/2/3+
    - Total high cards: 0/1/2/3/4+
    - Void suits: 0/1/2
    - Has existing bid: Y/N
    - Is Qabool: Y/N
    - Side suit strength: W(eak 0-1 high) / M(od 2) / S(trong 3+)
    """
    hand = obs.hand
    suit_counts = Counter(card.suit for card in hand)
    longest = max(suit_counts.values()) if suit_counts else 0
    longest_suit = max(suit_counts, key=suit_counts.get) if suit_counts else None

    # Longest suit bucket.
    ls = str(min(longest, 7))

    # High cards in longest suit.
    highs_in_longest = sum(1 for c in hand
                           if c.suit == longest_suit and c.rank in HIGH_RANKS)
    hl = str(min(highs_in_longest, 3))

    # Total high cards.
    total_highs = sum(1 for c in hand if c.rank in HIGH_RANKS)
    th = str(min(total_highs, 4))

    # Void suits.
    suits_present = set(c.suit for c in hand)
    voids = 4 - len(suits_present)
    vs = str(min(voids, 2))

    # Has existing bid.
    hb = "Y" if obs.current_highest_bid else "N"

    # Is Qabool.
    iq = "Y" if obs.is_sahib_al_qabool else "N"

    # Side suit strength.
    side_highs = sum(1 for c in hand
                     if c.suit != longest_suit and c.rank in HIGH_RANKS)
    ss = "W" if side_highs <= 1 else ("M" if side_highs == 2 else "S")

    return f"{ls}{hl}{th}{vs}{hb}{iq}{ss}"


def encode_bid_action(action: Action) -> str:
    """Encode bidding action into finer categories."""
    if isinstance(action, PassAction):
        return "PASS"
    if isinstance(action, BidAction):
        v = action.value
        if v <= 7:
            return "B7"
        elif v <= 8:
            return "B8"
        elif v <= 9:
            return "B9"
        elif v <= 10:
            return "B10"
        elif v <= 11:
            return "B11"
        else:
            return "B12+"
    return "PASS"


# ---------------------------------------------------------------
# Learning Agent — TD(λ) with eligibility traces
# ---------------------------------------------------------------


class LearningAgent(Agent):
    """
    Enhanced learning agent for Wist using TD(λ) with:
    - Double Q-learning (reduces overestimation)
    - Eligibility traces (temporal credit within a shota)
    - Opponent modeling (tracks void/trump patterns)
    - Prioritized experience replay (revisit surprising outcomes)
    - Card memory (tracks all played cards)
    - Adaptive exploration (UCB-inspired)
    """

    def __init__(self, epsilon: float = 0.3, training: bool = True,
                 alpha: float = 0.1, gamma: float = 0.95,
                 lambda_: float = 0.7) -> None:
        # Double Q-tables for card-play (reduces overestimation bias).
        self.q_table: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.q_table2: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.n_table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Bid Q-table.
        self.bid_q: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.bid_n: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # TD(λ) parameters.
        self.alpha = alpha        # Learning rate
        self.gamma = gamma        # Discount factor
        self.lambda_ = lambda_    # Eligibility trace decay

        self.epsilon = epsilon
        self.training = training

        # Eligibility traces for current episode.
        self._traces: dict[tuple[str, str], float] = {}

        # Episode memory.
        self._play_episode: list[tuple[str, str, float]] = []
        self._bid_episode: list[tuple[str, str]] = []
        self._last_state: str | None = None
        self._last_action: str | None = None

        # Card memory — tracks played cards within a shota.
        self._card_memory: set[tuple[Suit, Rank]] = set()
        self._tricks_won_this_shota: int = 0
        self._tricks_lost_this_shota: int = 0

        # Opponent model.
        self._opponent_model = OpponentModel()

        # Prioritized experience replay.
        self._replay_buffer = ReplayBuffer(capacity=8000)
        self._replay_interval = 4  # Replay every N tricks
        self._trick_counter = 0

        self.total_updates = 0
        self.episodes_trained = 0

    def act(self, observation: Observation) -> Action:
        if isinstance(observation, BiddingObservation):
            return self._act_bidding(observation)
        if isinstance(observation, WistObservation):
            return self._act_play(observation)
        raise TypeError(f"Unsupported observation: {type(observation).__name__}")

    # ----------------------------------------------------------
    # Card play with TD learning
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
        elif self.training and random.random() < self._explore_rate(obs):
            card = random.choice(playable)
        else:
            card = self._best_card(obs, playable)

        # Record state-action and do TD update if training.
        if self.training:
            state = encode_play_state(obs, self._card_memory, self._opponent_model)
            action_key = encode_play_action(card, obs.trump_suit, leading_suit,
                                            self._card_memory)

            # TD update from previous step.
            if self._last_state is not None:
                self._td_update(self._last_state, self._last_action, 0.0, state)

            self._last_state = state
            self._last_action = action_key
            self._play_episode.append((state, action_key, 0.0))

        # Update card memory and opponent model from current trick.
        if obs.current_trick:
            for pc in obs.current_trick.played_cards:
                self._card_memory.add((pc.card.suit, pc.card.rank))
                # Opponent modeling: record what opponents played.
                if pc.player_id != obs.player_id:
                    self._opponent_model.observe_play(
                        pc.player_id, pc.card, leading_suit, obs.trump_suit)

        return PlayCardAction(player_id=obs.player_id, card=card)

    def _explore_rate(self, obs: WistObservation) -> float:
        """
        Adaptive exploration: explore more in states we haven't seen much.
        UCB-inspired: increase epsilon for rarely-visited states.
        """
        state = encode_play_state(obs, self._card_memory, self._opponent_model)
        visits = sum(self.n_table[state].values()) if state in self.n_table else 0
        if visits < 5:
            return min(0.8, self.epsilon * 2.0)  # Explore heavily in new states
        elif visits < 20:
            return self.epsilon * 1.3
        return self.epsilon

    def _best_card(self, obs: WistObservation, playable: list[Card]) -> Card:
        """
        Pick the card whose action-type has the best learned value.
        Uses Double Q-learning: average of Q1 and Q2 for evaluation.
        """
        state = encode_play_state(obs, self._card_memory, self._opponent_model)
        q1_values = self.q_table[state]
        q2_values = self.q_table2[state]

        leading_suit = None
        if obs.current_trick:
            leading_suit = obs.current_trick.leading_suit

        best_card = playable[0]
        best_q = float("-inf")

        for card in playable:
            key = encode_play_action(card, obs.trump_suit, leading_suit,
                                     self._card_memory)
            # Double Q: use average of both tables for evaluation.
            q = (q1_values[key] + q2_values[key]) / 2.0

            # UCB bonus for less-visited actions (only during training).
            if self.training:
                n = max(1, self.n_table[state][key])
                total_visits = max(1, sum(self.n_table[state].values()))
                ucb_bonus = math.sqrt(2.0 * math.log(total_visits + 1) / n)
                score = q + 0.15 * ucb_bonus
            else:
                score = q

            if score > best_q:
                best_q = score
                best_card = card

        return best_card

    def _td_update(self, state: str, action: str, reward: float, next_state: str):
        """
        Double Q-learning TD(λ) update with eligibility traces and replay.

        With 50% probability updates Q1 or Q2:
        - Q1 update: uses Q1 to select best action, Q2 to evaluate it
        - Q2 update: uses Q2 to select best action, Q1 to evaluate it
        This reduces overestimation bias common in standard Q-learning.
        """
        # Double Q: randomly pick which table to update.
        if random.random() < 0.5:
            q_update = self.q_table
            q_eval = self.q_table2
        else:
            q_update = self.q_table2
            q_eval = self.q_table

        # Get max action from update table, evaluate with eval table.
        next_actions = q_update[next_state]
        if next_actions:
            best_next_action = max(next_actions, key=next_actions.get)
            max_next_q = q_eval[next_state][best_next_action]
        else:
            max_next_q = 0.0

        # TD error.
        current_q = q_update[state][action]
        delta = reward + self.gamma * max_next_q - current_q

        # Store in replay buffer if significant.
        if abs(delta) > 0.1:
            self._replay_buffer.add(state, action, reward, next_state, delta)

        # Update eligibility trace for current state-action.
        sa = (state, action)
        self._traces[sa] = self._traces.get(sa, 0.0) + 1.0

        # Update all traces.
        to_remove = []
        for (s, a), trace in self._traces.items():
            q_update[s][a] += self.alpha * delta * trace
            self.n_table[s][a] += 1
            self._traces[(s, a)] = self.gamma * self.lambda_ * trace
            if self._traces[(s, a)] < 0.01:
                to_remove.append((s, a))
            self.total_updates += 1

        for key in to_remove:
            del self._traces[key]

        # Periodic experience replay.
        self._trick_counter += 1
        if self._trick_counter % self._replay_interval == 0:
            self._do_replay()

    def _do_replay(self, batch_size: int = 8):
        """Replay past experiences to reinforce important lessons."""
        experiences = self._replay_buffer.sample(batch_size)
        for state, action, reward, next_state in experiences:
            if next_state is None:
                # Terminal — direct update.
                self.q_table[state][action] += self.alpha * 0.5 * (
                    reward - self.q_table[state][action])
            else:
                # Non-terminal — standard TD.
                next_q = self.q_table[next_state]
                max_next = max(next_q.values()) if next_q else 0.0
                target = reward + self.gamma * max_next
                self.q_table[state][action] += self.alpha * 0.5 * (
                    target - self.q_table[state][action])
            self.total_updates += 1

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
        # 35% chance to pass, 65% chance to bid.
        if random.random() < 0.35:
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

        # Get Q-values for each possible action.
        actions = ["PASS", "B7", "B8", "B9", "B10", "B11", "B12+"]
        best_action = "PASS"
        best_q = q_values.get("PASS", 0.0)

        for key in actions[1:]:
            q = q_values.get(key, 0.0)
            if q > best_q:
                best_q = q
                best_action = key

        if best_action == "PASS":
            return PassAction(player_id=obs.player_id)

        # Convert to actual bid value.
        action_to_value = {
            "B7": 7, "B8": 8, "B9": 9, "B10": 10, "B11": 11, "B12+": 12
        }
        bid_value = action_to_value.get(best_action, max(7, min_bid))
        bid_value = max(bid_value, min_bid)
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
    # Reward signals — per-trick and end-of-shota
    # ----------------------------------------------------------

    def reward_trick(self, won: bool) -> None:
        """
        Per-trick reward signal with strategic shaping.

        Rewards consider:
        - Basic win/loss
        - Seek potential (winning all tricks = massive bonus building)
        - Consecutive wins (momentum bonus)
        """
        if not self.training:
            return

        if won:
            self._tricks_won_this_shota += 1
            # Base reward + seek momentum bonus.
            if self._tricks_lost_this_shota == 0:
                # On seek track — escalating reward.
                seek_bonus = 0.05 * self._tricks_won_this_shota
                trick_reward = 0.3 + seek_bonus
            else:
                trick_reward = 0.25
        else:
            self._tricks_lost_this_shota += 1
            if self._tricks_lost_this_shota == 1 and self._tricks_won_this_shota >= 5:
                # Lost seek after a long streak — harsh signal.
                trick_reward = -0.5
            else:
                trick_reward = -0.15

        # TD update with the trick reward.
        if self._last_state is not None:
            current_q = self.q_table[self._last_state][self._last_action]
            delta = trick_reward - current_q * 0.1

            # Store high-impact trick results in replay buffer.
            if abs(delta) > 0.2:
                self._replay_buffer.add(
                    self._last_state, self._last_action, trick_reward, None, delta)

            sa = (self._last_state, self._last_action)
            self._traces[sa] = self._traces.get(sa, 0.0) + 1.0

            to_remove = []
            for (s, a), trace in self._traces.items():
                self.q_table[s][a] += self.alpha * 0.5 * delta * trace
                self._traces[(s, a)] = self.gamma * self.lambda_ * trace
                if self._traces[(s, a)] < 0.01:
                    to_remove.append((s, a))

            for key in to_remove:
                del self._traces[key]

    def reward_shota(self, team_won_shota: bool, bid_met: bool,
                     my_tricks: int = 0, opp_tricks: int = 0,
                     was_shooter: bool = False, seek: bool = False) -> None:
        """
        End-of-Shota reward with enhanced shaping.

        Reward structure:
        - Base: +1.5 for winning shota, -1.5 for losing
        - Bid bonus: +1.0 if our team met the bid as shooter
        - Bid penalty: -0.8 if we were shooter and failed
        - Trick margin: +0.15 per trick above opponent (diminishing)
        - Seek bonus: +3.0 for achieving seek
        - Seek penalty: -2.0 for being seeked against
        - Dominance bonus: +0.5 for winning 10+ tricks (near-seek)
        """
        if not self.training:
            self._reset_episode_state()
            return

        # Compute return.
        G = 1.5 if team_won_shota else -1.5

        # Bid bonus/penalty (stronger signal).
        if was_shooter:
            G += 1.0 if bid_met else -0.8

        # Trick margin (diminishing returns).
        margin = my_tricks - opp_tricks
        G += min(margin * 0.15, 1.5)

        # Seek signals.
        if seek:
            if my_tricks == 13:
                G += 3.0
            else:
                G -= 2.0

        # Near-seek bonus (domination without full seek).
        if my_tricks >= 10 and not seek:
            G += 0.5

        # Clamp to reasonable range.
        G = max(-4.0, min(5.0, G))

        # Final TD update — terminal state.
        if self._last_state is not None:
            current_q = self.q_table[self._last_state][self._last_action]
            delta = G - current_q
            sa = (self._last_state, self._last_action)
            self._traces[sa] = self._traces.get(sa, 0.0) + 1.0

            for (s, a), trace in self._traces.items():
                self.q_table[s][a] += self.alpha * delta * trace
                self.n_table[s][a] += 1
                self.total_updates += 1

        # Update bid Q-table with Monte Carlo (bids are one-shot).
        self._update_bid_table(G)

        self.episodes_trained += 1
        self._reset_episode_state()

    def _update_bid_table(self, G: float):
        """First-visit Monte Carlo update on bid Q-table."""
        visited = set()
        for state, action_key in self._bid_episode:
            sa = (state, action_key)
            if sa in visited:
                continue
            visited.add(sa)

            self.bid_n[state][action_key] += 1
            n = self.bid_n[state][action_key]
            old_q = self.bid_q[state][action_key]
            # Weighted average with recency bias.
            weight = min(1.0 / n, self.alpha)
            self.bid_q[state][action_key] = old_q + weight * (G - old_q)
            self.total_updates += 1

    def _reset_episode_state(self):
        """Clear all episode-specific state."""
        self._play_episode.clear()
        self._bid_episode.clear()
        self._traces.clear()
        self._last_state = None
        self._last_action = None
        self._card_memory.clear()
        self._tricks_won_this_shota = 0
        self._tricks_lost_this_shota = 0
        self._opponent_model.reset()
        self._trick_counter = 0

    def reset_episode(self) -> None:
        """Clear episode memory (call on Dak / skipped Shota)."""
        self._reset_episode_state()

    def observe_card_played(self, card: Card) -> None:
        """
        Called externally when any card is played (by any player).
        Maintains card memory for state encoding.
        """
        self._card_memory.add((card.suit, card.rank))

    def decay_epsilon(self, min_epsilon: float = 0.03, decay_rate: float = 0.9997) -> None:
        """Slowly reduce exploration. Decays slower than before for better coverage."""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    def decay_alpha(self, min_alpha: float = 0.01, decay_rate: float = 0.9999) -> None:
        """Decay learning rate for convergence."""
        self.alpha = max(min_alpha, self.alpha * decay_rate)

    # ----------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 3,  # Schema version for compatibility.
            "q_table": {k: dict(v) for k, v in self.q_table.items()},
            "q_table2": {k: dict(v) for k, v in self.q_table2.items()},
            "n_table": {k: dict(v) for k, v in self.n_table.items()},
            "bid_q": {k: dict(v) for k, v in self.bid_q.items()},
            "bid_n": {k: dict(v) for k, v in self.bid_n.items()},
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "lambda": self.lambda_,
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

        agent = cls(
            epsilon=data.get("epsilon", 0.05),
            training=training,
            alpha=data.get("alpha", 0.1),
            gamma=data.get("gamma", 0.95),
            lambda_=data.get("lambda", 0.7),
        )

        for state, actions in data.get("q_table", {}).items():
            for action_key, value in actions.items():
                agent.q_table[state][action_key] = value
        for state, actions in data.get("q_table2", {}).items():
            for action_key, value in actions.items():
                agent.q_table2[state][action_key] = value
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
        play2_size = sum(len(v) for v in self.q_table2.values())
        bid_size = sum(len(v) for v in self.bid_q.values())
        return play_size + play2_size + bid_size
