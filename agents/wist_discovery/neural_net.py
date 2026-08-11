"""
Neural Network for Q-value approximation.

Pure numpy implementation — no torch/tensorflow dependency.

Two architectures:
1. CardEvaluator — takes (state_features + card_features) → single Q-value.
   Evaluates each legal card independently. Picks highest Q.
   No information loss — every card is distinct.

2. QNetwork — fixed output size, used for bidding (small action space).
"""

import numpy as np
from collections import Counter


class CardEvaluator:
    """
    Neural network that evaluates a single card in context.
    
    3 hidden layers × 128 neurons. Larger capacity for complex patterns.
    Supports batch training for stable gradients.
    """

    def __init__(self, input_size: int = 84, hidden_size: int = 128,
                 learning_rate: float = 0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lr = learning_rate

        # Xavier initialization — 3 hidden layers.
        s1 = np.sqrt(2.0 / input_size)
        s2 = np.sqrt(2.0 / hidden_size)
        s3 = np.sqrt(2.0 / hidden_size)
        s4 = np.sqrt(2.0 / hidden_size)

        self.w1 = np.random.randn(input_size, hidden_size) * s1
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, hidden_size) * s2
        self.b2 = np.zeros(hidden_size)
        self.w3 = np.random.randn(hidden_size, hidden_size) * s3
        self.b3 = np.zeros(hidden_size)
        self.w4 = np.random.randn(hidden_size, 1) * s4
        self.b4 = np.zeros(1)

        # Batch training buffer.
        self._batch_x: list = []
        self._batch_y: list = []
        self._batch_size = 32

    def predict(self, x: np.ndarray) -> float:
        """Forward pass — returns single Q-value."""
        h1 = np.maximum(0, x @ self.w1 + self.b1)
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)
        h3 = np.maximum(0, h2 @ self.w3 + self.b3)
        return float((h3 @ self.w4 + self.b4)[0])

    def predict_batch(self, batch: np.ndarray) -> np.ndarray:
        """Evaluate multiple feature vectors at once."""
        h1 = np.maximum(0, batch @ self.w1 + self.b1)
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)
        h3 = np.maximum(0, h2 @ self.w3 + self.b3)
        return (h3 @ self.w4 + self.b4).flatten()

    def update(self, x: np.ndarray, target_value: float):
        """Accumulate sample for batch update. Flushes when batch is full."""
        self._batch_x.append(x.copy())
        self._batch_y.append(target_value)
        if len(self._batch_x) >= self._batch_size:
            self._flush_batch()

    def _flush_batch(self):
        """Train on accumulated batch — more stable gradients."""
        if not self._batch_x:
            return

        # Snapshot and clear atomically to avoid race conditions with concurrent writes.
        batch_x_list = self._batch_x
        batch_y_list = self._batch_y
        self._batch_x = []
        self._batch_y = []

        batch_x = np.array(batch_x_list)
        batch_y = np.array(batch_y_list)
        n = len(batch_x)

        # Validate shapes match (defensive against concurrent modification).
        if len(batch_y) != n:
            return

        # Forward pass (batch).
        h1 = np.maximum(0, batch_x @ self.w1 + self.b1)
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)
        h3 = np.maximum(0, h2 @ self.w3 + self.b3)
        predictions = (h3 @ self.w4 + self.b4).flatten()

        # Validate prediction shape matches batch.
        if predictions.shape[0] != n:
            return

        # Error.
        errors = batch_y - predictions  # (n,)

        # Backprop (averaged over batch).
        d_out = (-2.0 / n) * errors.reshape(-1, 1)  # (n, 1)

        # Layer 4.
        d_w4 = h3.T @ d_out  # (hidden, 1)
        d_b4 = d_out.sum(axis=0)
        d_h3 = d_out @ self.w4.T  # (n, hidden)
        d_h3[h3 <= 0] = 0

        # Layer 3.
        d_w3 = h2.T @ d_h3
        d_b3 = d_h3.sum(axis=0)
        d_h2 = d_h3 @ self.w3.T
        d_h2[h2 <= 0] = 0

        # Layer 2.
        d_w2 = h1.T @ d_h2
        d_b2 = d_h2.sum(axis=0)
        d_h1 = d_h2 @ self.w2.T
        d_h1[h1 <= 0] = 0

        # Layer 1.
        d_w1 = batch_x.T @ d_h1
        d_b1 = d_h1.sum(axis=0)

        # Gradient clipping.
        max_norm = 5.0
        for grad in [d_w1, d_w2, d_w3, d_w4]:
            norm = np.linalg.norm(grad)
            if norm > max_norm:
                grad *= max_norm / norm

        # SGD step.
        self.w1 -= self.lr * d_w1
        self.b1 -= self.lr * d_b1
        self.w2 -= self.lr * d_w2
        self.b2 -= self.lr * d_b2
        self.w3 -= self.lr * d_w3
        self.b3 -= self.lr * d_b3
        self.w4 -= self.lr * d_w4
        self.b4 -= self.lr * d_b4

        # Clear batch.
        self._batch_x.clear()
        self._batch_y.clear()

    def copy(self) -> "CardEvaluator":
        """Create a frozen copy (for target network)."""
        clone = CardEvaluator(self.input_size, self.hidden_size, self.lr)
        clone.w1 = self.w1.copy()
        clone.b1 = self.b1.copy()
        clone.w2 = self.w2.copy()
        clone.b2 = self.b2.copy()
        clone.w3 = self.w3.copy()
        clone.b3 = self.b3.copy()
        clone.w4 = self.w4.copy()
        clone.b4 = self.b4.copy()
        return clone

        return error ** 2

    def to_dict(self) -> dict:
        return {
            "w1": self.w1.tolist(), "b1": self.b1.tolist(),
            "w2": self.w2.tolist(), "b2": self.b2.tolist(),
            "w3": self.w3.tolist(), "b3": self.b3.tolist(),
            "w4": self.w4.tolist(), "b4": self.b4.tolist(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size, "lr": self.lr,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CardEvaluator":
        net = cls(input_size=data["input_size"],
                  hidden_size=data["hidden_size"],
                  learning_rate=data.get("lr", 0.001))
        net.w1 = np.array(data["w1"])
        net.b1 = np.array(data["b1"])
        net.w2 = np.array(data["w2"])
        net.b2 = np.array(data["b2"])
        net.w3 = np.array(data["w3"])
        net.b3 = np.array(data["b3"])
        net.w4 = np.array(data["w4"])
        net.b4 = np.array(data["b4"])
        return net


class QNetwork:
    """Simple network for bidding (small fixed action space)."""

    def __init__(self, input_size: int = 32, hidden_size: int = 64,
                 output_size: int = 8, learning_rate: float = 0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        scale1 = np.sqrt(2.0 / input_size)
        scale2 = np.sqrt(2.0 / hidden_size)
        self.w1 = np.random.randn(input_size, hidden_size) * scale1
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size) * scale2
        self.b2 = np.zeros(output_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0, x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def update(self, x: np.ndarray, target_idx: int, target_value: float):
        h = np.maximum(0, x @ self.w1 + self.b1)
        q_values = h @ self.w2 + self.b2
        error = target_value - q_values[target_idx]
        d_out = np.zeros(self.output_size)
        d_out[target_idx] = -2 * error
        d_w2 = np.outer(h, d_out)
        d_b2 = d_out
        d_h = d_out @ self.w2.T
        d_h[h <= 0] = 0
        d_w1 = np.outer(x, d_h)
        d_b1 = d_h
        self.w1 -= self.lr * d_w1
        self.b1 -= self.lr * d_b1
        self.w2 -= self.lr * d_w2
        self.b2 -= self.lr * d_b2
        return error ** 2

    def to_dict(self) -> dict:
        return {
            "w1": self.w1.tolist(), "b1": self.b1.tolist(),
            "w2": self.w2.tolist(), "b2": self.b2.tolist(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size, "lr": self.lr,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QNetwork":
        net = cls(input_size=data["input_size"], hidden_size=data["hidden_size"],
                  output_size=data["output_size"], learning_rate=data.get("lr", 0.001))
        net.w1 = np.array(data["w1"])
        net.b1 = np.array(data["b1"])
        net.w2 = np.array(data["w2"])
        net.b2 = np.array(data["b2"])
        return net


# =============================================================================
# Feature extraction — per-card features (no information loss)
# =============================================================================

RANK_NORM = {2: 0.0, 3: 0.08, 4: 0.17, 5: 0.25, 6: 0.33, 7: 0.42,
             8: 0.5, 9: 0.58, 10: 0.67, 11: 0.75, 12: 0.83, 13: 0.92, 14: 1.0}


def card_features(card, obs, playable_cards, rank_val_func, suit_idx_map) -> np.ndarray:
    """
    Raw card features — NO domain knowledge.

    Only observable facts about this card:
    1. Rank (normalized 0-1) — just the number on the card
    2. Suit index (normalized 0-1) — which suit it is
    3. How many cards of this suit in hand (normalized)
    4. Is this the only card of its suit in hand (0/1)
    5. Rank position within same-suit cards in hand (relative)
    6. Number of cards played in this trick so far (normalized)
    7. Number of legal options available (normalized)
    8. How many cards total in hand (normalized)

    NO trump labeling, NO "can win", NO "follows suit" labeling.
    """
    rv = rank_val_func(card.rank)
    features = np.zeros(8)

    # 1. Raw rank (normalized).
    features[0] = rv / 14.0

    # 2. Suit index (normalized).
    si = suit_idx_map.get(card.suit, 0) if suit_idx_map else 0
    features[1] = (si + 1) / 4.0

    # 3. How many cards of this suit in hand.
    suit_count = sum(1 for c in obs.hand if c.suit == card.suit)
    features[2] = suit_count / 13.0

    # 4. Is this the only card of its suit (would create void if played).
    features[3] = 1.0 if suit_count == 1 else 0.0

    # 5. Relative rank within same-suit cards.
    same_suit_ranks = sorted([rank_val_func(c.rank) for c in obs.hand if c.suit == card.suit])
    if len(same_suit_ranks) > 1:
        pos = same_suit_ranks.index(rv)
        features[4] = pos / (len(same_suit_ranks) - 1)
    else:
        features[4] = 1.0

    # 6. Cards already played in this trick.
    if obs.current_trick and obs.current_trick.played_cards:
        features[5] = len(obs.current_trick.played_cards) / 4.0

    # 7. Number of legal options (normalized).
    features[6] = len(playable_cards) / 13.0

    # 8. Total cards in hand.
    features[7] = len(obs.hand) / 13.0

    return features


def state_features(obs, opp_voids: int = 0, rank_val_func=None, suit_idx_map=None,
                   suits_played: dict = None) -> np.ndarray:
    """
    Raw state features — NO domain knowledge.

    Just raw observable facts:
    - What cards are in the hand (normalized rank per suit slot)
    - How many cards on the table in this trick
    - What cards are on the table (rank + suit, normalized)
    """
    hand = obs.hand
    n_cards = len(hand)
    features = np.zeros(28)

    if rank_val_func is None:
        return features

    # 1-4: How many cards per suit (raw counts, not sorted — preserves suit identity).
    suit_counts = [0, 0, 0, 0]
    for c in hand:
        si = suit_idx_map.get(c.suit, 0) if suit_idx_map else 0
        suit_counts[si] += 1
    for i in range(4):
        features[i] = suit_counts[i] / 13.0

    # 5: Total cards in hand (normalized).
    features[4] = n_cards / 13.0

    # 6-9: Average rank per suit (what the hand looks like strength-wise per suit).
    suit_rank_sums = [0.0, 0.0, 0.0, 0.0]
    for c in hand:
        si = suit_idx_map.get(c.suit, 0) if suit_idx_map else 0
        suit_rank_sums[si] += rank_val_func(c.rank)
    for i in range(4):
        if suit_counts[i] > 0:
            features[5 + i] = (suit_rank_sums[i] / suit_counts[i]) / 14.0
        else:
            features[5 + i] = 0.0

    # 10-13: Current trick — cards played (rank normalized, per slot).
    if obs.current_trick and obs.current_trick.played_cards:
        for idx, pc in enumerate(obs.current_trick.played_cards[:4]):
            features[9 + idx] = rank_val_func(pc.card.rank) / 14.0

    # 14-17: Current trick — suits of played cards (normalized suit index).
    if obs.current_trick and obs.current_trick.played_cards:
        for idx, pc in enumerate(obs.current_trick.played_cards[:4]):
            si = suit_idx_map.get(pc.card.suit, 0) if suit_idx_map else 0
            features[13 + idx] = (si + 1) / 4.0

    # 18: Number of cards played in this trick (position proxy).
    if obs.current_trick and obs.current_trick.played_cards:
        features[17] = len(obs.current_trick.played_cards) / 4.0

    # 19-22: Cards played per suit globally (card counting — raw observable).
    if suits_played:
        for suit_idx in range(4):
            features[18 + suit_idx] = suits_played.get(suit_idx, 0) / 13.0

    # 23: Number of known opponent voids (observed: they didn't follow suit).
    features[22] = min(opp_voids, 8) / 8.0

    # 24-27: Which player played each card in current trick (partner vs opponent).
    # Encode: 0.25=partner, 0.75=opponent, 0=not yet played.
    if obs.current_trick and obs.current_trick.played_cards:
        my_pid = getattr(obs, 'player_id', 0)
        partner_pid = (my_pid + 2) % 4
        for idx, pc in enumerate(obs.current_trick.played_cards[:4]):
            if pc.player_id == partner_pid:
                features[23 + idx] = 0.25  # Partner.
            else:
                features[23 + idx] = 0.75  # Opponent.

    # 27: Remaining (pad to 28).

    return features


def state_to_features(state_str: str, max_len: int = 32) -> np.ndarray:
    """Legacy: convert state string to feature vector (for bid network)."""
    features = np.zeros(max_len)
    for i, ch in enumerate(state_str[:max_len]):
        features[i] = ord(ch) / 127.0
    return features


# Bid actions: PASS + B7..B13 = 8 actions.
BID_ACTIONS = ["PASS"] + [f"B{v}" for v in range(7, 14)]
BID_ACTION_TO_IDX = {a: i for i, a in enumerate(BID_ACTIONS)}
NUM_BID_ACTIONS = len(BID_ACTIONS)


def get_bid_action_idx(action_str: str) -> int:
    return BID_ACTION_TO_IDX.get(action_str, 0)
