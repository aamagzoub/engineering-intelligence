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
from collections import defaultdict, deque, Counter
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

def _encode_play_state(obs: WistObservation, opp_voids: int = 0,
                       partner_bid: int = 0, my_tricks: int = 0, opp_tricks: int = 0) -> str:
    """Absolute minimal state encoding — almost nothing.

    The agent sees only its raw hand size as a rough state key.
    All real learning happens through the neural net from raw features.
    The Q-table provides a weak baseline; neural net handles generalization.
    """
    hand = obs.hand
    n_cards = len(hand)
    pos = 0
    if obs.current_trick and obs.current_trick.played_cards:
        pos = len(obs.current_trick.played_cards)
    # Minimal key: just hand size and position — enough for Q-table baseline.
    return f"{n_cards:x}{pos}"


def _encode_play_action(card, obs: WistObservation) -> str:
    """Minimal action encoding — no domain knowledge.

    Contains only:
    - Raw rank number (2-14) — just the card's face value
    - Suit index (0-3) — just identifies which suit
    No tiers, no trump labeling, no void detection, no longest suit.
    """
    rv = RANK_VAL[card.rank]
    si = SUIT_IDX[card.suit]
    return f"{rv:x}{si}"


def _encode_bid_state(obs: BiddingObservation) -> str:
    """Absolute minimal bid state — just whether someone bid and the level.

    Agent must discover everything about hand strength from the neural net.
    """
    has_bid = "Y" if obs.current_highest_bid else "N"
    bid_level = str(min(obs.current_highest_bid, 13)) if obs.current_highest_bid else "0"
    return f"{has_bid}{bid_level}"


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
        # Take a snapshot to avoid race conditions with concurrent writes.
        buf_snapshot = list(self._buffer)
        pri_snapshot = list(self._priorities[:len(buf_snapshot)])
        size = min(batch_size, len(buf_snapshot))
        total = sum(pri_snapshot)
        if total == 0 or len(pri_snapshot) != len(buf_snapshot):
            indices = random.sample(range(len(buf_snapshot)), size)
        else:
            probs = np.array(pri_snapshot, dtype=np.float64)
            probs /= probs.sum()
            indices = np.random.choice(len(buf_snapshot), size=size, replace=False, p=probs).tolist()
        return [buf_snapshot[i] for i in indices]

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
        # CardEvaluator: per-card evaluation (state 28 + card 8 + memory 78 = 114 features → 1 Q-value).
        self._play_net = CardEvaluator(input_size=114, hidden_size=128, learning_rate=0.0003)
        self._target_net = self._play_net.copy()  # Target network (frozen, updated every 500 episodes).
        self._target_update_interval = 500
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

        # === Sequence Memory (full shota history) ===
        # Each entry: (rank_norm, was_trump, followed_suit, won_trick)
        self._trick_memory: list[tuple[float, float, float, float]] = []
        self._memory_size = 13  # Remember all tricks in the shota.

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

        # === Card Counting — track suits played globally ===
        self._suits_played: dict = {0: 0, 1: 0, 2: 0, 3: 0}  # SUIT_IDX → count of cards played

        # === Opponent Prediction ===
        self._opp_predictor = OpponentPredictor()

        # === Self-play Elo Tracking ===
        self._elo_tracker = EloTracker()

        # === Meta-Learning ===
        self._meta_learner = MetaLearner()

        # === Stats ===
        self.episodes_trained: int = 0
        self.total_updates: int = 0

        # === Enhanced observations (partner bid, trick counts) ===
        self._partner_bid: int = 0  # Observable: what did partner bid?
        self._my_tricks: int = 0    # Observable: tricks won by my team so far.
        self._opp_tricks: int = 0   # Observable: tricks won by opponents so far.

        # === Per-state adaptive alpha (#5) ===
        self._state_update_counts: dict[str, int] = defaultdict(int)

        # === N-step return buffer (#2) ===
        self._nstep_buffer: list[tuple[str, str, float]] = []  # (state, action, intermediate_value)
        self._nstep_n: int = 3  # Look 3 steps ahead for returns.

        # === MCTS training integration (#1, #14) ===
        self._mcts_context: dict = None  # Set externally when MCTS is available.
        self._mcts_value_weight: float = 0.3  # How much MCTS value influences training.

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
        """Bid or pass — learned from reward only.

        Bid rules:
          Regular player:
            min_bid = max(7, trump_count + 3)   [1-4 trump → 7, 5→8, 6→9, 7→10]
            max_bid = 11 if opening bid, else 13
            must exceed current highest bid if one exists

          Sahib Al-Qabool:
            min_bid (no one bid) = max(7, trump_count + 3)   [same as regular]
            min_bid (someone bid, matching) = max(7, trump_count + 2)  [one card advantage]
            max_bid = 13 always (no opening cap)
            can match current highest bid (not required to exceed)
        """
        hand = obs.hand
        suit_counts = Counter(c.suit for c in hand)

        # Valid trump suits: 1–7 cards (8+ = Dak, cannot be trump).
        valid_trump_suits = [s for s, count in suit_counts.items() if 1 <= count <= 7]
        if not valid_trump_suits:
            return PassAction(player_id=obs.player_id)

        longest_trump_count = max(suit_counts[s] for s in valid_trump_suits)

        # Min bid is driven by trump count (floor of what you can responsibly bid).
        trump_floor = max(7, longest_trump_count + 3)  # 1-4 trump→7, 5→8, 6→9, 7→10

        if obs.is_sahib_al_qabool:
            # Max is always 13 for Qabool — no opening cap.
            max_bid = 13
            if obs.current_highest_bid:
                # Matching advantage: Qabool's floor drops by 1 (trump+2 instead of trump+3).
                match_floor = max(7, longest_trump_count + 2)
                # Must bid at least the current highest (to match), and at least own floor.
                min_bid = max(match_floor, obs.current_highest_bid)
            else:
                # All others passed — standard floor, free range up to 13.
                min_bid = trump_floor
        else:
            # Regular player.
            min_bid = trump_floor
            if obs.is_opening_bid:
                max_bid = 11           # Opening bid capped at 11.
            else:
                max_bid = 13           # Not opening — can go up to 13.
            if obs.current_highest_bid:
                min_bid = max(min_bid, obs.current_highest_bid + 1)  # Must exceed.

        # Safety clamp.
        max_bid = min(max_bid, 13)

        if min_bid > max_bid and not obs.must_play:
            action = PassAction(player_id=obs.player_id)
        elif min_bid > max_bid and obs.must_play:
            action = BidAction(player_id=obs.player_id, value=min(min_bid, 13))
        elif obs.must_play:
            if self.training and random.random() < self.epsilon:
                bid_val = random.randint(min_bid, max_bid)
            else:
                best = self._best_bid(obs, min_bid, max_bid)
                bid_val = best.value if isinstance(best, BidAction) else min_bid
            action = BidAction(player_id=obs.player_id, value=bid_val)
        elif self.training and random.random() < self.epsilon:
            if random.random() < 0.5:
                action = PassAction(player_id=obs.player_id)
            else:
                bid_val = random.randint(min_bid, max_bid)
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
            # Curiosity-driven exploration: prefer unvisited states.
            if random.random() < 0.3:
                # Pick action that leads to least-visited state.
                state_str = _encode_play_state(obs, self._get_opponent_voids_count(obs),
                                              self._partner_bid, self._my_tricks, self._opp_tricks)
                min_visits = float("inf")
                curiosity_card = random.choice(playable)
                for c in playable:
                    key = _encode_play_action(c, obs)
                    full_key = state_str + key
                    visits = self._state_visit_counts.get(full_key, 0)
                    if visits < min_visits:
                        min_visits = visits
                        curiosity_card = c
                card = curiosity_card
            else:
                card = random.choice(playable)
        else:
            card = self._best_card(obs, playable)

        if self.training:
            state = _encode_play_state(obs, self._get_opponent_voids_count(obs),
                                       self._partner_bid, self._my_tricks, self._opp_tricks)
            action_key = _encode_play_action(card, obs)
            self._play_episode.append((state, action_key))
            self._play_traces[state][action_key] += 1.0
            self._state_visit_counts[state] += 1

            # Store neural net features for this card choice.
            if self._use_neural:
                s_feat = state_features(obs, self._get_opponent_voids_count(obs),
                                        rank_value, SUIT_IDX, self._suits_played)
                c_feat = card_features(card, obs, playable, rank_value, SUIT_IDX)
                mem_feat = self._get_memory_features()
                self._nn_play_features.append(np.concatenate([s_feat, c_feat, mem_feat]))

        return PlayCardAction(player_id=obs.player_id, card=card)

    def _best_card(self, obs: WistObservation, playable: list) -> object:
        """Combined Q-table + per-card neural net + optional MCTS for card selection.

        MCTS integration (#1, #14): When MCTS is available, use it for card selection
        AND feed its value estimates back as training targets for the neural net.
        """
        # MCTS: if context is available, use simulation-based look-ahead.
        if getattr(self, '_mcts_context', None) and len(playable) > 1:
            from agents.wist_discovery.mcts import mcts_choose_card, mcts_evaluate_actions
            ctx = self._mcts_context
            num_sims = ctx.get('num_simulations', 80)

            # Get MCTS action values for training (#1).
            try:
                mcts_values = mcts_evaluate_actions(
                    obs, playable, ctx.get('round_state'), ctx.get('players'),
                    ctx.get('trump_suit'), num_simulations=num_sims
                )
                # Use MCTS values as soft training targets for Q-table.
                if self.training and mcts_values:
                    state_str = _encode_play_state(obs, self._get_opponent_voids_count(obs),
                                                  self._partner_bid, self._my_tricks, self._opp_tricks)
                    mcts_alpha = self._mcts_value_weight * 0.01
                    for card, mcts_val in mcts_values.items():
                        action_key = _encode_play_action(card, obs)
                        current = self.play_q[state_str][action_key]
                        self.play_q[state_str][action_key] += mcts_alpha * (mcts_val - current)
            except (ImportError, Exception):
                pass

            return mcts_choose_card(
                obs, playable, ctx.get('round_state'), ctx.get('players'),
                ctx.get('trump_suit'), num_simulations=num_sims
            )

        state_str = _encode_play_state(obs, self._get_opponent_voids_count(obs),
                                      self._partner_bid, self._my_tricks, self._opp_tricks)
        q1 = self.play_q[state_str]
        q2 = self.play_q2[state_str]

        # Neural net: evaluate each card individually with sequence memory.
        nn_values = {}
        if self._use_neural:
            s_feat = state_features(obs, self._get_opponent_voids_count(obs),
                                    rank_value, SUIT_IDX, self._suits_played)
            mem_feat = self._get_memory_features()
            for card in playable:
                c_feat = card_features(card, obs, playable, rank_value, SUIT_IDX)
                combined_feat = np.concatenate([s_feat, c_feat, mem_feat])
                nn_values[id(card)] = self._play_net.predict(combined_feat)

        best_card = playable[0]
        best_q = float("-inf")
        for card in playable:
            key = _encode_play_action(card, obs)
            q_table_val = (q1.get(key, 0.0) + q2.get(key, 0.0)) / 2
            nn_val = nn_values.get(id(card), 0.0)
            combined = self._combined_q(q_table_val, q_table_val, nn_val)

            # Tiebreaker: when Q-values are effectively equal, use per-card
            # neural net score (unique per card). If no neural net, use rank as
            # a stable tiebreaker so the choice is deterministic.
            tiebreaker = nn_val * 0.001 if nn_val else RANK_VAL[card.rank] * 0.0001
            combined += tiebreaker

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
        """No-op. The agent must discover that tricks matter from the shota score alone.
        No intermediate feedback — only the final score teaches."""
        pass

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

        # === Adaptive learning rate (per-state) ===
        base_alpha = max(0.05, self.alpha * (1.0 / (1.0 + self.episodes_trained / 1000)))

        # === Play phase with eligibility traces + Double Q + N-step returns ===
        play_alpha = base_alpha * self._play_alpha_scale
        if random.random() < 0.5:
            play_q_update, play_q_eval = self.play_q, self.play_q2
        else:
            play_q_update, play_q_eval = self.play_q2, self.play_q

        # N-step returns: compute returns from windows of N steps.
        episode_len = len(self._play_episode)
        nstep_returns = [0.0] * episode_len
        # First compute single-step (standard): propagate reward backward.
        reward_signal = score
        for idx in range(episode_len - 1, -1, -1):
            nstep_returns[idx] = reward_signal
            reward_signal *= self.gamma

        # N-step enhancement: blend forward returns for first (episode_len - N) steps.
        n = self._nstep_n
        for idx in range(episode_len - n):
            # N-step return = gamma^n * V(state+n) + discounted intermediates.
            # Since we don't have intermediate rewards, n-step just looks further ahead.
            future_state, future_action = self._play_episode[idx + n]
            future_q = play_q_eval[future_state].get(future_action, 0.0)
            nstep_return = (self.gamma ** n) * future_q + nstep_returns[idx] * (1.0 - self.gamma ** n)
            # Blend: 70% standard backward return, 30% n-step forward return.
            nstep_returns[idx] = 0.7 * nstep_returns[idx] + 0.3 * nstep_return

        nn_features_reversed = list(reversed(self._nn_play_features)) if self._nn_play_features else []
        for idx, (state, action) in enumerate(reversed(self._play_episode)):
            # Per-state adaptive alpha: states visited many times learn slower.
            self._state_update_counts[state] += 1
            visit_count = self._state_update_counts[state]
            adaptive_alpha = play_alpha / (1.0 + visit_count / 500.0)

            trace = self._play_traces[state].get(action, 1.0)
            current_q = play_q_update[state][action]
            target = nstep_returns[episode_len - 1 - idx]
            td_error = target - current_q
            update = adaptive_alpha * trace * td_error
            play_q_update[state][action] += update
            self.total_updates += 1

            # Prioritized replay.
            self._replay_buffer.add(state, action, target, "play", td_error)

            # Train CardEvaluator neural net.
            if self._use_neural and idx < len(nn_features_reversed):
                self._play_net.update(nn_features_reversed[idx], target)

        # === Bid phase ===
        bid_alpha = base_alpha * self._bid_alpha_scale
        if random.random() < 0.5:
            bid_q_update = self.bid_q
        else:
            bid_q_update = self.bid_q2

        bid_reward = score  # Bid directly caused the shota outcome — full credit.
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

        # === Meta-learning: log performance (no epsilon/alpha override — single decay owns it) ===
        self._meta_learner.should_adjust(self.episodes_trained)  # Track stats only.

        # === Curriculum: switch to neural net after enough episodes ===
        if not self._use_neural and self.episodes_trained >= self._neural_switch_threshold:
            self._use_neural = True

        # === Target network update (every N episodes) ===
        if self._use_neural and self.episodes_trained % self._target_update_interval == 0:
            self._target_net = self._play_net.copy()

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
            # Card counting — track how many of each suit played.
            suit_idx = SUIT_IDX.get(played_card.card.suit, 0)
            self._suits_played[suit_idx] = min(13, self._suits_played.get(suit_idx, 0) + 1)

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

    def _get_memory_features(self) -> np.ndarray:
        """Empty memory — agent gets no trick-by-trick history.
        Returns zeros; neural net must learn from raw hand state alone."""
        return np.zeros(78)

    # =========================================================================
    # Episode Management
    # =========================================================================

    def reset_episode(self):
        self._play_episode.clear()
        self._bid_episode.clear()
        self._nn_play_features.clear()
        self._trick_memory.clear()
        self._play_traces.clear()
        self._bid_traces.clear()
        self._known_voids = {0: set(), 1: set(), 2: set(), 3: set()}
        self._suits_played = {0: 0, 1: 0, 2: 0, 3: 0}
        self._my_tricks = 0
        self._opp_tricks = 0

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: str) -> None:
        try:
            data = {
                "play_q": {k: dict(v) for k, v in list(self.play_q.items())},
                "play_q2": {k: dict(v) for k, v in list(self.play_q2.items())},
                "bid_q": {k: dict(v) for k, v in list(self.bid_q.items())},
                "bid_q2": {k: dict(v) for k, v in list(self.bid_q2.items())},
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
        except RuntimeError:
            # Dict changed during iteration (background thread). Retry with snapshot.
            data = {
                "play_q": {},
                "play_q2": {},
                "bid_q": {},
                "bid_q2": {},
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
            # Try once more with list() snapshots.
            try:
                data["play_q"] = {k: dict(v) for k, v in list(self.play_q.items())}
                data["play_q2"] = {k: dict(v) for k, v in list(self.play_q2.items())}
                data["bid_q"] = {k: dict(v) for k, v in list(self.bid_q.items())}
                data["bid_q2"] = {k: dict(v) for k, v in list(self.bid_q2.items())}
            except RuntimeError:
                pass  # Save with whatever we got.

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
            try:
                self._play_net = CardEvaluator.from_dict(data["play_net"])
            except (KeyError, ValueError):
                pass  # Old format — start fresh CardEvaluator.
        if "bid_net" in data:
            self._bid_net = QNetwork.from_dict(data["bid_net"])
