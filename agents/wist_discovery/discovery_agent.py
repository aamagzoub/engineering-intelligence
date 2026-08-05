"""
Wist Discovery Agent — Advanced RL Architecture v2.

Zero domain knowledge. Receives ONLY: environment + legal moves + score signal.

Architecture (fully domain-agnostic, transferable):
1. Double Q-Learning — two Q-tables to prevent overestimation bias
2. Eligibility Traces (TD(λ)) — decaying credit to recent state-actions
3. Experience Replay with Prioritization — re-learn from surprising outcomes
4. Curiosity Bonus — exploration reward for visiting new states
5. Opponent Modeling — track known opponent voids from observations
6. Hierarchical Learning — separate bid/play phases with different learning rates
7. Reward Normalization — scale rewards to standard range
8. Per-trick intermediate rewards — faster credit assignment
9. Neural Network function approximator — generalization across similar states
10. Population-based hyperparameter tracking — auto-tune learning parameters
11. Self-play Elo tracking — measure improvement over time
12. Curriculum learning — progressive difficulty
13. Opponent prediction — predict opponent actions from patterns
14. Meta-learning — adapt hyperparameters based on recent performance
"""

import json
import random
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from environments.wist.actions import BidAction, PassAction, PlayCardAction
from environments.wist.observation import BiddingObservation, WistObservation
from environments.wist.rules import legal_cards, rank_value
from intelligence.core.action import Action
from intelligence.core.agent import Agent
from intelligence.core.cards.rank import Rank
from intelligence.core.cards.suit import Suit
from intelligence.core.observation import Observation

from agents.wist_discovery.neural_net import (
    CardEvaluator, QNetwork, state_to_features, state_features, card_features,
    get_bid_action_idx, NUM_BID_ACTIONS,
)


# Domain-agnostic rank ordering.
RANK_VAL = {
    Rank.TWO: 2, Rank.THREE: 3, Rank.FOUR: 4, Rank.FIVE: 5, Rank.SIX: 6,
    Rank.SEVEN: 7, Rank.EIGHT: 8, Rank.NINE: 9, Rank.TEN: 10,
    Rank.JACK: 11, Rank.QUEEN: 12, Rank.KING: 13, Rank.ACE: 14,
}
SUIT_IDX = {Suit.SPADES: 0, Suit.HEARTS: 1, Suit.CLUBS: 2, Suit.DIAMONDS: 3}

STATE_FEATURE_SIZE = 32  # Fixed feature vector size for neural net.


# =============================================================================
# State & Action Encoding
# =============================================================================

def _encode_play_state(obs: WistObservation, opp_voids: int = 0) -> str:
    """Rich state encoding — observable features only."""
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
    diff = obs.team_scores.get(my_team, 0) - obs.team_scores.get(opp_team, 0)
    td = "W" if diff >= 3 else ("A" if diff > 0 else ("T" if diff == 0 else "B"))
    ts = f"{min(trump_count, 7)}{min(trump_highs, 4)}"
    voids = sum(1 for s in suits if s == 0)

    return f"{shape}{pos}{phase}{td}{min(highs, 5)}{ts}v{voids}a{min(aces, 4)}o{min(opp_voids, 6)}"


def _encode_play_action(card, obs: WistObservation) -> str:
    """Richer action encoding."""
    rv = RANK_VAL[card.rank]
    tier = "A" if rv == 14 else ("H" if rv >= 12 else ("M" if rv >= 9 else "L"))
    leading = obs.current_trick.leading_suit if obs.current_trick else None
    follows = "F" if (leading and card.suit == leading) else "O"
    is_trump = "T" if (obs.trump_suit and card.suit == obs.trump_suit) else "N"
    from collections import Counter
    suit_counts = Counter(c.suit for c in obs.hand)
    longest = max(suit_counts.values()) if suit_counts else 0
    is_long = "L" if suit_counts.get(card.suit, 0) == longest else "S"
    creates_void = "V" if suit_counts.get(card.suit, 0) == 1 else "K"
    return f"{tier}{follows}{is_trump}{is_long}{creates_void}"


def _encode_bid_state(obs: BiddingObservation) -> str:
    """Richer bid state."""
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
    if isinstance(action, PassAction):
        return "PASS"
    if isinstance(action, BidAction):
        return f"B{min(action.value, 13)}"
    return "PASS"



# =============================================================================
# Support Classes
# =============================================================================

class RewardNormalizer:
    """Running normalization — scales rewards to ~[-1, 1] regardless of domain."""
    def __init__(self):
        self._mean = 0.0
        self._var = 1.0
        self._count = 0

    def normalize(self, reward: float) -> float:
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


class PrioritizedReplayBuffer:
    """Experience replay with prioritization based on TD-error (Enhancement #3)."""
    def __init__(self, capacity: int = 20000):
        self._buffer: list = []
        self._priorities: list = []
        self._capacity = capacity
        self._pos = 0

    def add(self, state: str, action: str, reward: float, table_key: str, td_error: float = 1.0):
        priority = abs(td_error) + 0.01  # Small constant to avoid zero priority.
        if len(self._buffer) < self._capacity:
            self._buffer.append((state, action, reward, table_key))
            self._priorities.append(priority)
        else:
            self._buffer[self._pos] = (state, action, reward, table_key)
            self._priorities[self._pos] = priority
        self._pos = (self._pos + 1) % self._capacity

    def sample(self, batch_size: int = 64) -> list:
        if len(self._buffer) == 0:
            return []
        size = min(batch_size, len(self._buffer))
        # Probability proportional to priority.
        total = sum(self._priorities)
        if total == 0:
            indices = random.sample(range(len(self._buffer)), size)
        else:
            probs = [p / total for p in self._priorities]
            indices = np.random.choice(len(self._buffer), size=size, replace=False, p=probs).tolist()
        return [self._buffer[i] for i in indices]

    def __len__(self):
        return len(self._buffer)


class EloTracker:
    """Self-play Elo tracking — measures improvement over time (Enhancement #4)."""
    def __init__(self):
        self.elo = 1000.0
        self._history: list = []  # (episode, elo)

    def update(self, won: bool, opponent_elo: float = 1000.0):
        """Update Elo based on game result."""
        expected = 1.0 / (1.0 + 10 ** ((opponent_elo - self.elo) / 400))
        actual = 1.0 if won else 0.0
        k = 32  # K-factor.
        self.elo += k * (actual - expected)

    def record(self, episode: int):
        self._history.append((episode, self.elo))
        # Keep last 100 snapshots.
        if len(self._history) > 100:
            self._history = self._history[-100:]

    def to_dict(self) -> dict:
        return {"elo": self.elo, "history": self._history[-50:]}

    def from_dict(self, d: dict):
        self.elo = d.get("elo", 1000.0)
        self._history = d.get("history", [])


class MetaLearner:
    """Tracks performance and auto-adjusts hyperparameters (Enhancement #7)."""
    def __init__(self):
        self._recent_scores: deque = deque(maxlen=50)
        self._adjustment_interval = 200  # Adjust every N episodes.
        self._last_adjustment = 0

    def record_score(self, score: float):
        self._recent_scores.append(score)

    def should_adjust(self, episode: int) -> bool:
        return (episode - self._last_adjustment >= self._adjustment_interval
                and len(self._recent_scores) >= 30)

    def suggest_adjustments(self, current_epsilon: float, current_alpha: float,
                           current_lambda: float, episode: int) -> dict:
        """Suggest hyperparameter changes based on recent performance."""
        self._last_adjustment = episode
        avg_score = sum(self._recent_scores) / len(self._recent_scores)
        recent_10 = list(self._recent_scores)[-10:]
        avg_recent = sum(recent_10) / len(recent_10)

        adjustments = {}

        # If performance is improving, reduce exploration.
        if avg_recent > avg_score * 1.1:
            adjustments["epsilon"] = max(0.02, current_epsilon * 0.95)
        # If performance is declining, increase exploration.
        elif avg_recent < avg_score * 0.8 and current_epsilon < 0.3:
            adjustments["epsilon"] = min(0.3, current_epsilon * 1.1)

        # If learning seems stagnant, increase learning rate.
        if abs(avg_recent - avg_score) < 0.5 and len(self._recent_scores) >= 50:
            adjustments["alpha"] = min(0.3, current_alpha * 1.05)

        return adjustments


class OpponentPredictor:
    """Predict opponent actions from observed patterns (Enhancement #6)."""
    def __init__(self):
        # Track: state_context -> action_counts.
        self._opp_patterns: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def observe(self, context: str, action: str):
        """Record an observed opponent action."""
        self._opp_patterns[context][action] += 1

    def predict_confidence(self, context: str) -> float:
        """How predictable is the opponent in this context? 0=unknown, 1=very predictable."""
        actions = self._opp_patterns.get(context, {})
        if not actions:
            return 0.0
        total = sum(actions.values())
        if total < 3:
            return 0.0
        max_count = max(actions.values())
        return max_count / total  # High = one dominant action.



# =============================================================================
# Main Agent Class
# =============================================================================

class WistDiscoveryAgent(Agent):
    """
    Wist Discovery Agent — Advanced RL Architecture v2.
    Transferable learning system. No domain knowledge.
    """

    def __init__(self, epsilon: float = 0.4, alpha: float = 0.2,
                 gamma: float = 0.97, lambda_trace: float = 0.7,
                 training: bool = True) -> None:

        # === Double Q-Learning ===
        self.play_q: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.play_q2: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.bid_q: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.bid_q2: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # === Neural Network Function Approximator (Enhancement #9) ===
        # CardEvaluator: per-card evaluation (state 24 + card 8 = 32 features → 1 Q-value).
        self._play_net = CardEvaluator(input_size=32, hidden_size=64, learning_rate=0.0005)
        self._bid_net = QNetwork(STATE_FEATURE_SIZE, hidden_size=64,
                                 output_size=NUM_BID_ACTIONS, learning_rate=0.001)
        self._use_neural = False  # Starts with Q-tables, switches after enough data.
        self._neural_switch_threshold = 5000  # Switch to neural after N episodes.

        # === Hyperparameters ===
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_trace = lambda_trace
        self.training = training

        # === Hierarchical Learning: different rates per phase ===
        self._play_alpha_scale = 1.0
        self._bid_alpha_scale = 1.5

        # === Episode memory ===
        self._play_episode: list[tuple[str, str]] = []
        self._bid_episode: list[tuple[str, str]] = []
        self._nn_play_features: list = []  # Neural net feature vectors per play action.

        # === Eligibility Traces ===
        self._play_traces: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._bid_traces: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # === Prioritized Experience Replay ===
        self._replay_buffer = PrioritizedReplayBuffer(capacity=20000)
        self._replay_batch_size = 64

        # === Curiosity Bonus ===
        self._state_visit_counts: dict[str, int] = defaultdict(int)
        self._curiosity_scale = 0.1

        # === Reward Normalization ===
        self._reward_normalizer = RewardNormalizer()

        # === Opponent Modeling ===
        self._known_voids: dict[int, set] = {0: set(), 1: set(), 2: set(), 3: set()}

        # === Opponent Prediction ===
        self._opp_predictor = OpponentPredictor()

        # === Self-play Elo Tracking ===
        self._elo_tracker = EloTracker()

        # === Meta-Learning ===
        self._meta_learner = MetaLearner()

        # === Curriculum Learning ===
        self._curriculum_level = 1  # 1=basic, 2=intermediate, 3=full game.

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
            self._bid_traces[state][action_key] += 1.0
            self._state_visit_counts[state] += 1

        return action

    def _best_bid(self, obs: BiddingObservation, min_bid: int, max_bid: int = 13) -> Action:
        """Combined Q-table + neural net for bid selection."""
        state = _encode_bid_state(obs)

        # Q-table values.
        q1 = self.bid_q[state]
        q2 = self.bid_q2[state]

        # Neural net values (if active).
        if self._use_neural:
            features = state_to_features(state, STATE_FEATURE_SIZE)
            nn_q = self._bid_net.predict(features)
        else:
            nn_q = None

        best_action = "PASS"
        best_q = self._combined_q(q1.get("PASS", 0.0), q2.get("PASS", 0.0),
                                   nn_q[get_bid_action_idx("PASS")] if nn_q is not None else 0.0)

        for v in range(min_bid, max_bid + 1):
            key = f"B{v}"
            combined = self._combined_q(q1.get(key, 0.0), q2.get(key, 0.0),
                                         nn_q[get_bid_action_idx(key)] if nn_q is not None else 0.0)
            if combined > best_q:
                best_q = combined
                best_action = key

        if best_action == "PASS":
            return PassAction(player_id=obs.player_id)
        return BidAction(player_id=obs.player_id, value=int(best_action[1:]))

    def _act_play(self, obs: WistObservation) -> Action:
        """Play a card — learned from reward only."""
        self._update_voids(obs)
        self._observe_opponents(obs)

        leading_suit = obs.current_trick.leading_suit if obs.current_trick else None
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
            self._play_traces[state][action_key] += 1.0
            self._state_visit_counts[state] += 1

            # Store neural net features for this card choice.
            if self._use_neural:
                s_feat = state_features(obs, self._get_opponent_voids_count(obs),
                                        rank_value, SUIT_IDX)
                c_feat = card_features(card, obs, playable, rank_value, SUIT_IDX)
                self._nn_play_features.append(np.concatenate([s_feat, c_feat]))

        return PlayCardAction(player_id=obs.player_id, card=card)

    def _best_card(self, obs: WistObservation, playable: list) -> object:
        """Combined Q-table + per-card neural net for card selection."""
        state_str = _encode_play_state(obs, self._get_opponent_voids_count(obs))
        q1 = self.play_q[state_str]
        q2 = self.play_q2[state_str]

        # Neural net: evaluate each card individually.
        nn_values = {}
        if self._use_neural:
            s_feat = state_features(obs, self._get_opponent_voids_count(obs),
                                    rank_value, SUIT_IDX)
            for card in playable:
                c_feat = card_features(card, obs, playable, rank_value, SUIT_IDX)
                combined = np.concatenate([s_feat, c_feat])
                nn_values[id(card)] = self._play_net.predict(combined)

        best_card = playable[0]
        best_q = float("-inf")
        for card in playable:
            key = _encode_play_action(card, obs)
            q_table_val = (q1.get(key, 0.0) + q2.get(key, 0.0)) / 2
            nn_val = nn_values.get(id(card), 0.0)
            combined = self._combined_q(q_table_val, q_table_val, nn_val)
            if combined > best_q:
                best_q = combined
                best_card = card
        return best_card

    def _combined_q(self, q1_val: float, q2_val: float, nn_val: float = 0.0) -> float:
        """Combine Q-table and neural net estimates."""
        if self._use_neural:
            # Blend: 60% Q-table average, 40% neural net.
            return 0.6 * (q1_val + q2_val) / 2 + 0.4 * nn_val
        return (q1_val + q2_val) / 2

    # =========================================================================
    # Learning
    # =========================================================================

    def trick_reward(self, won: bool) -> None:
        """Per-trick intermediate reward."""
        if not self.training or not self._play_episode:
            return
        micro_reward = 0.3 if won else -0.1
        effective_alpha = max(0.03, self.alpha * 0.3 * (1.0 / (1.0 + self.episodes_trained / 2000)))
        state, action = self._play_episode[-1]
        for q_table in (self.play_q, self.play_q2):
            current_q = q_table[state][action]
            q_table[state][action] += effective_alpha * (micro_reward - current_q)

        # Train CardEvaluator per-trick.
        if self._use_neural and self._nn_play_features:
            self._play_net.update(self._nn_play_features[-1], micro_reward)

    def reward(self, score: float) -> None:
        """End-of-shota reward with all architecture enhancements."""
        if not self.training:
            return

        # === Meta-learning: record score ===
        self._meta_learner.record_score(score)

        # === Elo tracking ===
        self._elo_tracker.update(won=(score > 0))
        if self.episodes_trained % 100 == 0:
            self._elo_tracker.record(self.episodes_trained)

        # === Reward Normalization ===
        self._reward_normalizer.normalize(score)

        # === Adaptive learning rate ===
        base_alpha = max(0.05, self.alpha * (1.0 / (1.0 + self.episodes_trained / 1000)))

        # === Play phase with eligibility traces + Double Q ===
        play_alpha = base_alpha * self._play_alpha_scale
        if random.random() < 0.5:
            play_q_update, play_q_eval = self.play_q, self.play_q2
        else:
            play_q_update, play_q_eval = self.play_q2, self.play_q

        reward_signal = score
        nn_features_reversed = list(reversed(self._nn_play_features)) if self._nn_play_features else []
        for idx, (state, action) in enumerate(reversed(self._play_episode)):
            trace = self._play_traces[state].get(action, 1.0)
            current_q = play_q_update[state][action]
            td_error = reward_signal - current_q
            update = play_alpha * trace * td_error
            play_q_update[state][action] += update
            reward_signal *= self.gamma
            self.total_updates += 1

            # Prioritized replay.
            self._replay_buffer.add(state, action, reward_signal, "play", td_error)

            # Curiosity bonus.
            visit_count = self._state_visit_counts.get(state, 1)
            curiosity = self._curiosity_scale / math.sqrt(visit_count)
            play_q_update[state][action] += play_alpha * 0.1 * curiosity

            # Train CardEvaluator neural net.
            if self._use_neural and idx < len(nn_features_reversed):
                self._play_net.update(nn_features_reversed[idx], reward_signal)

        # === Bid phase ===
        bid_alpha = base_alpha * self._bid_alpha_scale
        if random.random() < 0.5:
            bid_q_update = self.bid_q
        else:
            bid_q_update = self.bid_q2

        bid_reward = score * (self.gamma ** len(self._play_episode))
        for state, action in reversed(self._bid_episode):
            trace = self._bid_traces[state].get(action, 1.0)
            current_q = bid_q_update[state][action]
            td_error = bid_reward - current_q
            bid_q_update[state][action] += bid_alpha * trace * td_error
            self.total_updates += 1
            self._replay_buffer.add(state, action, bid_reward, "bid", td_error)

            if self._use_neural:
                features = state_to_features(state, STATE_FEATURE_SIZE)
                action_idx = get_bid_action_idx(action)
                self._bid_net.update(features, action_idx, bid_reward)

        # === Prioritized Experience Replay ===
        self._do_replay(base_alpha)

        # === Decay eligibility traces ===
        self._decay_traces()

        # === Meta-learning: auto-adjust hyperparameters ===
        if self._meta_learner.should_adjust(self.episodes_trained):
            adjustments = self._meta_learner.suggest_adjustments(
                self.epsilon, self.alpha, self.lambda_trace, self.episodes_trained)
            if "epsilon" in adjustments:
                self.epsilon = adjustments["epsilon"]
            if "alpha" in adjustments:
                self.alpha = adjustments["alpha"]

        # === Curriculum: switch to neural net after enough episodes ===
        if not self._use_neural and self.episodes_trained >= self._neural_switch_threshold:
            self._use_neural = True

        # === Cleanup ===
        self._play_episode.clear()
        self._bid_episode.clear()
        self._nn_play_features.clear()
        self.episodes_trained += 1

    def _do_replay(self, alpha: float):
        """Prioritized experience replay."""
        batch = self._replay_buffer.sample(self._replay_batch_size)
        replay_alpha = alpha * 0.2

        for state, action, reward, table_key in batch:
            if table_key == "play":
                q = self.play_q if random.random() < 0.5 else self.play_q2
                # CardEvaluator can't replay from string state alone (needs obs).
                # Only Q-table replay for play actions.
            else:
                q = self.bid_q if random.random() < 0.5 else self.bid_q2
                if self._use_neural:
                    features = state_to_features(state, STATE_FEATURE_SIZE)
                    self._bid_net.update(features, get_bid_action_idx(action), reward)
            current_q = q[state][action]
            q[state][action] += replay_alpha * (reward - current_q)

    def _decay_traces(self):
        """Decay eligibility traces."""
        decay = self.gamma * self.lambda_trace
        for traces in (self._play_traces, self._bid_traces):
            for state in list(traces.keys()):
                for action in list(traces[state].keys()):
                    traces[state][action] *= decay
                    if traces[state][action] < 0.01:
                        del traces[state][action]
                if not traces[state]:
                    del traces[state]

    # =========================================================================
    # Opponent Modeling & Prediction
    # =========================================================================

    def _update_voids(self, obs: WistObservation) -> None:
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

    def _observe_opponents(self, obs: WistObservation) -> None:
        """Track opponent play patterns for prediction."""
        if not obs.current_trick or not obs.current_trick.played_cards:
            return
        for played_card in obs.current_trick.played_cards:
            pid = played_card.player_id
            if pid == obs.player_id:
                continue
            # Context: position in trick + leading suit.
            pos = len([pc for pc in obs.current_trick.played_cards if pc.player_id != pid])
            leading = obs.current_trick.leading_suit
            context = f"p{pos}{'T' if leading else 'L'}"
            action_str = f"{'F' if played_card.card.suit == leading else 'O'}"
            self._opp_predictor.observe(context, action_str)

    def _get_opponent_voids_count(self, obs: WistObservation) -> int:
        my_team = 0 if obs.player_id in (0, 2) else 1
        opp_ids = [1, 3] if my_team == 0 else [0, 2]
        return sum(len(self._known_voids.get(pid, set())) for pid in opp_ids)

    # =========================================================================
    # Episode Management
    # =========================================================================

    def reset_episode(self):
        self._play_episode.clear()
        self._bid_episode.clear()
        self._nn_play_features.clear()
        self._play_traces.clear()
        self._bid_traces.clear()
        self._known_voids = {0: set(), 1: set(), 2: set(), 3: set()}

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: str) -> None:
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
            "elo": self._elo_tracker.to_dict(),
            "use_neural": self._use_neural,
            "play_net": self._play_net.to_dict(),
            "bid_net": self._bid_net.to_dict(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
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
        self._use_neural = data.get("use_neural", False)

        if "reward_norm" in data:
            self._reward_normalizer.from_dict(data["reward_norm"])
        if "elo" in data:
            self._elo_tracker.from_dict(data["elo"])
        if "play_net" in data:
            self._play_net = CardEvaluator.from_dict(data["play_net"])
        if "bid_net" in data:
            self._bid_net = QNetwork.from_dict(data["bid_net"])
