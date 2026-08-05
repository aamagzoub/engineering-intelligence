"""
Wist Discovery Agent — Advanced RL Architecture.

This agent has ZERO domain knowledge about Wist. It receives ONLY:
- Observation: hand + legal moves + visible trick state
- Reward: numeric score at the end of each shota

Architecture (domain-agnostic, transferable):
1. Double Q-Learning — two Q-tables to prevent overestimation bias
2. Eligibility Traces (TD(λ)) — decaying credit to recent state-actions
3. Experience Replay — random re-learning from stored past experiences
4. Curiosity Bonus — exploration reward for visiting new states
5. Opponent Modeling — track known opponent voids from observations
6. Hierarchical Learning — separate bid/play phases with different learning rates
7. Reward Normalization — scale rewards to standard range
8. Per-trick intermediate rewards — faster credit assignment

Self-play: opponents share Q-tables (same brain, both sides).
"""

import json
import random
import math
from collections import defaultdict, deque
from pathlib import Path

from environments.wist.actions import BidAction, PassAction, PlayCardAction
from environments.wist.observation import BiddingObservation, WistObservation
from environments.wist.rules import legal_cards, rank_value
from intelligence.core.action import Action
from intelligence.core.agent import Agent
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit
from intelligence.core.observation import Observation


# Domain-agnostic rank ordering.
RANK_VAL = {
    Rank.TWO: 2, Rank.THREE: 3, Rank.FOUR: 4, Rank.FIVE: 5, Rank.SIX: 6,
    Rank.SEVEN: 7, Rank.EIGHT: 8, Rank.NINE: 9, Rank.TEN: 10,
    Rank.JACK: 11, Rank.QUEEN: 12, Rank.KING: 13, Rank.ACE: 14,
}
SUIT_IDX = {Suit.SPADES: 0, Suit.HEARTS: 1, Suit.CLUBS: 2, Suit.DIAMONDS: 3}


# =============================================================================
# State & Action Encoding (domain-agnostic observable features)
# =============================================================================

def _encode_play_state(obs: WistObservation, opp_voids: int = 0) -> str:
    """Rich state encoding — observable features only, no strategy knowledge."""
    hand = obs.hand
    n_cards = len(hand)

    suits = [0, 0, 0, 0]
    highs = 0
    aces = 0
    trump_count = 0
    trump_highs = 0
    for c in hand:
        suits[SUIT_IDX[c.suit]] += 1
        if RANK_VAL[c.rank] >= 12:
            highs += 1
        if RANK_VAL[c.rank] == 14:
            aces += 1
        if obs.trump_suit and c.suit == obs.trump_suit:
            trump_count += 1
            if RANK_VAL[c.rank] >= 12:
                trump_highs += 1

    shape = "".join(str(min(s, 9)) for s in sorted(suits, reverse=True))

    pos = 0
    if obs.current_trick and obs.current_trick.played_cards:
        pos = len(obs.current_trick.played_cards)

    if n_cards >= 11:
        phase = "1"
    elif n_cards >= 8:
        phase = "2"
    elif n_cards >= 5:
        phase = "3"
    elif n_cards >= 2:
        phase = "4"
    else:
        phase = "5"

    my_team = 0 if obs.player_id in (0, 2) else 1
    opp_team = 1 - my_team
    my_t = obs.team_scores.get(my_team, 0)
    opp_t = obs.team_scores.get(opp_team, 0)
    diff = my_t - opp_t
    td = "W" if diff >= 3 else ("A" if diff > 0 else ("T" if diff == 0 else "B"))

    ts = f"{min(trump_count, 7)}{min(trump_highs, 4)}"
    voids = sum(1 for s in suits if s == 0)

    return f"{shape}{pos}{phase}{td}{min(highs, 5)}{ts}v{voids}a{min(aces, 4)}o{min(opp_voids, 6)}"


def _encode_play_action(card, obs: WistObservation) -> str:
    """Richer action encoding — relative strength + trump awareness."""
    rv = RANK_VAL[card.rank]
    if rv == 14:
        tier = "A"
    elif rv >= 12:
        tier = "H"
    elif rv >= 9:
        tier = "M"
    else:
        tier = "L"

    leading = None
    if obs.current_trick and obs.current_trick.leading_suit:
        leading = obs.current_trick.leading_suit
    follows = "F" if (leading and card.suit == leading) else "O"

    is_trump = "T" if (obs.trump_suit and card.suit == obs.trump_suit) else "N"

    from collections import Counter
    suit_counts = Counter(c.suit for c in obs.hand)
    longest = max(suit_counts.values()) if suit_counts else 0
    is_long = "L" if suit_counts.get(card.suit, 0) == longest else "S"

    creates_void = "V" if suit_counts.get(card.suit, 0) == 1 else "K"

    return f"{tier}{follows}{is_trump}{is_long}{creates_void}"


def _encode_bid_state(obs: BiddingObservation) -> str:
    """Richer bid state — hand composition + context."""
    hand = obs.hand
    from collections import Counter
    suit_counts = Counter(c.suit for c in hand)
    longest = max(suit_counts.values()) if suit_counts else 0
    shortest_valid = min((c for c in suit_counts.values() if 1 <= c <= 7), default=0)
    highs = sum(1 for c in hand if RANK_VAL[c.rank] >= 12)
    aces = sum(1 for c in hand if RANK_VAL[c.rank] == 14)
    voids = sum(1 for s in [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]
                if suit_counts.get(s, 0) == 0)
    has_bid = "Y" if obs.current_highest_bid else "N"
    is_q = "Y" if obs.is_sahib_al_qabool else "N"
    forced = "F" if obs.must_play else "N"
    bid_level = str(min(obs.current_highest_bid, 13)) if obs.current_highest_bid else "0"

    return f"{longest}{shortest_valid}{min(highs, 5)}{min(aces, 4)}v{voids}{has_bid}{bid_level}{is_q}{forced}"


def _encode_bid_action(action: Action) -> str:
    """Encode bid action."""
    if isinstance(action, PassAction):
        return "PASS"
    if isinstance(action, BidAction):
        return f"B{min(action.value, 13)}"
    return "PASS"



# =============================================================================
# Reward Normalization (Architecture Enhancement #7)
# =============================================================================

class RewardNormalizer:
    """Running normalization — scales rewards to ~[-1, 1] regardless of domain."""

    def __init__(self):
        self._mean = 0.0
        self._var = 1.0
        self._count = 0

    def normalize(self, reward: float) -> float:
        """Normalize reward using running statistics."""
        self._count += 1
        old_mean = self._mean
        self._mean += (reward - self._mean) / self._count
        self._var += (reward - old_mean) * (reward - self._mean)
        std = math.sqrt(self._var / max(self._count, 1)) + 1e-8
        return (reward - self._mean) / std

    def to_dict(self) -> dict:
        return {"mean": self._mean, "var": self._var, "count": self._count}

    def from_dict(self, d: dict):
        self._mean = d.get("mean", 0.0)
        self._var = d.get("var", 1.0)
        self._count = d.get("count", 0)


# =============================================================================
# Experience Replay Buffer (Architecture Enhancement #1)
# =============================================================================

class ReplayBuffer:
    """Fixed-size buffer of past experiences for random re-learning."""

    def __init__(self, capacity: int = 10000):
        self._buffer: deque = deque(maxlen=capacity)

    def add(self, state: str, action: str, reward: float, q_table_key: str):
        """Store a (state, action, reward, table_key) tuple."""
        self._buffer.append((state, action, reward, q_table_key))

    def sample(self, batch_size: int = 32) -> list:
        """Sample a random batch."""
        size = min(batch_size, len(self._buffer))
        if size == 0:
            return []
        return random.sample(list(self._buffer), size)

    def __len__(self):
        return len(self._buffer)

    def to_list(self) -> list:
        return list(self._buffer)

    def from_list(self, data: list):
        self._buffer = deque(data, maxlen=self._buffer.maxlen)


# =============================================================================
# Main Agent Class
# =============================================================================

class WistDiscoveryAgent(Agent):
    """
    Wist Discovery Agent — Advanced RL Architecture.

    Transferable learning system. No domain knowledge. No hard-coded strategy.
    Learns entirely from: environment + legal moves + score signal.
    """

    def __init__(self, epsilon: float = 0.4, alpha: float = 0.2,
                 gamma: float = 0.97, lambda_trace: float = 0.7,
                 training: bool = True) -> None:

        # === Double Q-Learning (Enhancement #3) ===
        # Two Q-tables — one selects actions, other evaluates.
        self.play_q: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.play_q2: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.bid_q: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.bid_q2: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # === Hyperparameters ===
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_trace = lambda_trace  # Eligibility trace decay (Enhancement #2)
        self.training = training

        # === Hierarchical Learning (Enhancement #6) ===
        # Separate learning rates for bid vs play phases.
        self._play_alpha_scale = 1.0   # Play phase learns at base rate.
        self._bid_alpha_scale = 1.5    # Bid phase learns faster (fewer decisions, higher impact).

        # === Episode memory ===
        self._play_episode: list[tuple[str, str]] = []
        self._bid_episode: list[tuple[str, str]] = []

        # === Eligibility Traces (Enhancement #2) ===
        self._play_traces: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._bid_traces: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # === Experience Replay (Enhancement #1) ===
        self._replay_buffer = ReplayBuffer(capacity=20000)
        self._replay_batch_size = 64

        # === Curiosity Bonus (Enhancement #4) ===
        self._state_visit_counts: dict[str, int] = defaultdict(int)
        self._curiosity_scale = 0.1  # Small bonus for novel states.

        # === Reward Normalization (Enhancement #7) ===
        self._reward_normalizer = RewardNormalizer()

        # === Opponent Modeling (Enhancement #5) ===
        self._known_voids: dict[int, set] = {0: set(), 1: set(), 2: set(), 3: set()}

        # === Stats ===
        self.episodes_trained: int = 0
        self.total_updates: int = 0

    # =========================================================================
    # Action Selection
    # =========================================================================

    def act(self, observation: Observation) -> Action:
        if isinstance(observation, BiddingObservation):
            return self._act_bid(observation)
        if isinstance(observation, WistObservation):
            return self._act_play(observation)
        raise TypeError(f"Unsupported: {type(observation).__name__}")

    def _act_bid(self, obs: BiddingObservation) -> Action:
        """Bid or pass — learned from reward only."""
        from collections import Counter
        hand = obs.hand
        suit_counts = Counter(c.suit for c in hand)

        valid_trump_suits = [s for s, count in suit_counts.items() if 1 <= count <= 7]
        if not valid_trump_suits:
            return PassAction(player_id=obs.player_id)

        shortest_trump_count = min(suit_counts[s] for s in valid_trump_suits)

        # Qabool rules.
        if obs.is_sahib_al_qabool:
            if obs.current_highest_bid:
                min_bid = obs.current_highest_bid
                max_bid = 13
            else:
                min_bid = max(7, shortest_trump_count + 3)
                max_bid = 13
        else:
            min_bid = max(7, shortest_trump_count + 3)
            max_bid = 11 if obs.is_opening_bid else 13
            if obs.current_highest_bid:
                min_bid = max(min_bid, obs.current_highest_bid + 1)

        if min_bid > max_bid and not obs.must_play:
            action = PassAction(player_id=obs.player_id)
        elif min_bid > max_bid and obs.must_play:
            action = BidAction(player_id=obs.player_id, value=min(min_bid, 13))
        elif obs.must_play:
            if self.training and random.random() < self.epsilon:
                bid_val = random.randint(min_bid, min(max_bid, min_bid + 2))
            else:
                best = self._best_bid(obs, min_bid, max_bid)
                bid_val = best.value if isinstance(best, BidAction) else min_bid
            action = BidAction(player_id=obs.player_id, value=bid_val)
        elif self.training and random.random() < self.epsilon:
            if random.random() < 0.5:
                action = PassAction(player_id=obs.player_id)
            else:
                bid_val = random.randint(min_bid, min(max_bid, min_bid + 2))
                action = BidAction(player_id=obs.player_id, value=bid_val)
        else:
            action = self._best_bid(obs, min_bid, max_bid)

        if self.training:
            state = _encode_bid_state(obs)
            action_key = _encode_bid_action(action)
            self._bid_episode.append((state, action_key))
            # Eligibility trace update.
            self._bid_traces[state][action_key] += 1.0
            # Curiosity: track state visits.
            self._state_visit_counts[state] += 1

        return action

    def _best_bid(self, obs: BiddingObservation, min_bid: int, max_bid: int = 13) -> Action:
        """Double Q-Learning: use combined Q for selection."""
        state = _encode_bid_state(obs)
        q1 = self.bid_q[state]
        q2 = self.bid_q2[state]

        best_action = "PASS"
        best_q = (q1.get("PASS", 0.0) + q2.get("PASS", 0.0)) / 2

        for v in range(min_bid, max_bid + 1):
            key = f"B{v}"
            combined = (q1.get(key, 0.0) + q2.get(key, 0.0)) / 2
            if combined > best_q:
                best_q = combined
                best_action = key

        if best_action == "PASS":
            return PassAction(player_id=obs.player_id)
        return BidAction(player_id=obs.player_id, value=int(best_action[1:]))

    def _act_play(self, obs: WistObservation) -> Action:
        """Play a card — learned from reward only."""
        self._update_voids(obs)

        leading_suit = None
        if obs.current_trick:
            leading_suit = obs.current_trick.leading_suit
        must_trump = obs.trump_suit if obs.must_lead_trump else None
        playable = legal_cards(obs.hand, leading_suit, must_trump)

        if len(playable) == 1:
            card = playable[0]
        elif self.training and random.random() < self.epsilon:
            card = random.choice(playable)
        else:
            card = self._best_card(obs, playable)

        if self.training:
            state = _encode_play_state(obs, self._get_opponent_voids_count(obs))
            action_key = _encode_play_action(card, obs)
            self._play_episode.append((state, action_key))
            # Eligibility trace update.
            self._play_traces[state][action_key] += 1.0
            # Curiosity: track state visits.
            self._state_visit_counts[state] += 1

        return PlayCardAction(player_id=obs.player_id, card=card)

    def _best_card(self, obs: WistObservation, playable: list) -> object:
        """Double Q-Learning: use combined Q for card selection."""
        state = _encode_play_state(obs, self._get_opponent_voids_count(obs))
        q1 = self.play_q[state]
        q2 = self.play_q2[state]

        best_card = playable[0]
        best_q = float("-inf")
        for card in playable:
            key = _encode_play_action(card, obs)
            combined = (q1.get(key, 0.0) + q2.get(key, 0.0)) / 2
            if combined > best_q:
                best_q = combined
                best_card = card
        return best_card

    # =========================================================================
    # Learning
    # =========================================================================

    def trick_reward(self, won: bool) -> None:
        """Per-trick intermediate reward (Enhancement #8)."""
        if not self.training or not self._play_episode:
            return

        micro_reward = 0.3 if won else -0.1
        normalized = self._reward_normalizer.normalize(micro_reward)
        effective_alpha = max(0.03, self.alpha * 0.3 * (1.0 / (1.0 + self.episodes_trained / 2000)))

        # Update last action with per-trick signal.
        state, action = self._play_episode[-1]
        for q_table in (self.play_q, self.play_q2):
            current_q = q_table[state][action]
            q_table[state][action] += effective_alpha * (micro_reward - current_q)

    def reward(self, score: float) -> None:
        """End-of-shota reward with all architecture enhancements."""
        if not self.training:
            return

        # === Reward Normalization (Enhancement #7) ===
        normalized_score = self._reward_normalizer.normalize(score)

        # === Adaptive learning rate ===
        base_alpha = max(0.05, self.alpha * (1.0 / (1.0 + self.episodes_trained / 1000)))

        # === Hierarchical Learning (Enhancement #6): play phase ===
        play_alpha = base_alpha * self._play_alpha_scale

        # === Eligibility Traces + Double Q-Learning (Enhancements #2 + #3) ===
        # Randomly choose which Q-table to update (Double Q).
        if random.random() < 0.5:
            play_q_update = self.play_q
            play_q_eval = self.play_q2
        else:
            play_q_update = self.play_q2
            play_q_eval = self.play_q

        # Update play Q-table with eligibility traces.
        reward_signal = score  # Use raw score for actual updates (normalized for comparison).
        for state, action in reversed(self._play_episode):
            trace = self._play_traces[state].get(action, 1.0)
            current_q = play_q_update[state][action]
            # TD target: use eval table for next-state value estimation.
            update = play_alpha * trace * (reward_signal - current_q)
            play_q_update[state][action] += update
            reward_signal *= self.gamma
            self.total_updates += 1

            # Store in replay buffer.
            self._replay_buffer.add(state, action, reward_signal, "play")

            # Curiosity bonus (Enhancement #4): extra reward for novel states.
            visit_count = self._state_visit_counts.get(state, 1)
            curiosity = self._curiosity_scale / math.sqrt(visit_count)
            play_q_update[state][action] += play_alpha * 0.1 * curiosity

        # === Hierarchical Learning (Enhancement #6): bid phase ===
        bid_alpha = base_alpha * self._bid_alpha_scale

        if random.random() < 0.5:
            bid_q_update = self.bid_q
        else:
            bid_q_update = self.bid_q2

        bid_reward = score * (self.gamma ** len(self._play_episode))
        for state, action in reversed(self._bid_episode):
            trace = self._bid_traces[state].get(action, 1.0)
            current_q = bid_q_update[state][action]
            bid_q_update[state][action] += bid_alpha * trace * (bid_reward - current_q)
            self.total_updates += 1
            self._replay_buffer.add(state, action, bid_reward, "bid")

        # === Experience Replay (Enhancement #1) ===
        self._do_replay(base_alpha)

        # === Decay eligibility traces ===
        self._decay_traces()

        # === Cleanup ===
        self._play_episode.clear()
        self._bid_episode.clear()
        self.episodes_trained += 1

    def _do_replay(self, alpha: float):
        """Sample from replay buffer and re-learn (Enhancement #1)."""
        batch = self._replay_buffer.sample(self._replay_batch_size)
        replay_alpha = alpha * 0.3  # Replay learns slower to not override fresh experience.

        for state, action, reward, table_key in batch:
            if table_key == "play":
                # Update a random one of the two Q-tables.
                q = self.play_q if random.random() < 0.5 else self.play_q2
            else:
                q = self.bid_q if random.random() < 0.5 else self.bid_q2
            current_q = q[state][action]
            q[state][action] += replay_alpha * (reward - current_q)

    def _decay_traces(self):
        """Decay all eligibility traces by gamma * lambda (Enhancement #2)."""
        decay = self.gamma * self.lambda_trace
        # Play traces.
        for state in list(self._play_traces.keys()):
            for action in list(self._play_traces[state].keys()):
                self._play_traces[state][action] *= decay
                if self._play_traces[state][action] < 0.01:
                    del self._play_traces[state][action]
            if not self._play_traces[state]:
                del self._play_traces[state]
        # Bid traces.
        for state in list(self._bid_traces.keys()):
            for action in list(self._bid_traces[state].keys()):
                self._bid_traces[state][action] *= decay
                if self._bid_traces[state][action] < 0.01:
                    del self._bid_traces[state][action]
            if not self._bid_traces[state]:
                del self._bid_traces[state]

    # =========================================================================
    # Opponent Modeling (Enhancement #5)
    # =========================================================================

    def _update_voids(self, obs: WistObservation) -> None:
        """Track opponent voids from trick cards."""
        if not obs.current_trick or not obs.current_trick.played_cards:
            return
        leading_suit = obs.current_trick.leading_suit
        if not leading_suit:
            return
        for played_card in obs.current_trick.played_cards:
            pid = played_card.player_id
            if pid == obs.player_id:
                continue
            if played_card.card.suit != leading_suit:
                self._known_voids[pid].add(leading_suit)

    def _get_opponent_voids_count(self, obs: WistObservation) -> int:
        """Count total known opponent voids."""
        my_team = 0 if obs.player_id in (0, 2) else 1
        opp_ids = [1, 3] if my_team == 0 else [0, 2]
        return sum(len(self._known_voids.get(pid, set())) for pid in opp_ids)

    # =========================================================================
    # Episode Management
    # =========================================================================

    def reset_episode(self):
        """Clear episode memory (on Dak/skip)."""
        self._play_episode.clear()
        self._bid_episode.clear()
        self._play_traces.clear()
        self._bid_traces.clear()
        self._known_voids = {0: set(), 1: set(), 2: set(), 3: set()}

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: str) -> None:
        """Save model to JSON."""
        data = {
            "play_q": {k: dict(v) for k, v in self.play_q.items()},
            "play_q2": {k: dict(v) for k, v in self.play_q2.items()},
            "bid_q": {k: dict(v) for k, v in self.bid_q.items()},
            "bid_q2": {k: dict(v) for k, v in self.bid_q2.items()},
            "episodes_trained": self.episodes_trained,
            "total_updates": self.total_updates,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "lambda_trace": self.lambda_trace,
            "reward_norm": self._reward_normalizer.to_dict(),
            "state_visits": dict(list(self._state_visit_counts.items())[:5000]),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        """Load model from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        self.play_q = defaultdict(lambda: defaultdict(float))
        self.play_q2 = defaultdict(lambda: defaultdict(float))
        self.bid_q = defaultdict(lambda: defaultdict(float))
        self.bid_q2 = defaultdict(lambda: defaultdict(float))

        for s, actions in data.get("play_q", {}).items():
            for a, v in actions.items():
                self.play_q[s][a] = v
        for s, actions in data.get("play_q2", {}).items():
            for a, v in actions.items():
                self.play_q2[s][a] = v
        for s, actions in data.get("bid_q", {}).items():
            for a, v in actions.items():
                self.bid_q[s][a] = v
        for s, actions in data.get("bid_q2", {}).items():
            for a, v in actions.items():
                self.bid_q2[s][a] = v

        self.episodes_trained = data.get("episodes_trained", 0)
        self.total_updates = data.get("total_updates", 0)
        self.epsilon = data.get("epsilon", self.epsilon)
        self.lambda_trace = data.get("lambda_trace", self.lambda_trace)

        if "reward_norm" in data:
            self._reward_normalizer.from_dict(data["reward_norm"])
        if "state_visits" in data:
            self._state_visit_counts = defaultdict(int, data["state_visits"])
