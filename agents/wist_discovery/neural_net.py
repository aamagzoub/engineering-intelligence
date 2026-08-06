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
    Extract 8 numeric features for a single card in the current game context.
    
    Features (all normalized 0-1):
    1. Rank value (normalized)
    2. Is trump (0/1)
    3. Follows leading suit (0/1)
    4. Can win this trick (0/1) — observable fact
    5. Creates void if played (0/1)
    6. Is from longest suit in hand (0/1)
    7. Relative rank position (highest in suit in hand = 1, lowest = 0)
    8. Cards remaining in this suit (normalized)
    """
    rv = rank_val_func(card.rank)
    features = np.zeros(8)

    # 1. Rank (normalized).
    features[0] = RANK_NORM.get(rv, 0.5)

    # 2. Is trump.
    features[1] = 1.0 if (obs.trump_suit and card.suit == obs.trump_suit) else 0.0

    # 3. Follows leading suit.
    leading = obs.current_trick.leading_suit if obs.current_trick else None
    features[2] = 1.0 if (leading and card.suit == leading) else 0.0

    # 4. Can win this trick (observable — look at cards already played).
    can_win = 0.0
    if obs.current_trick and obs.current_trick.played_cards:
        # Find current highest.
        highest_rank = 0
        highest_is_trump = False
        for pc in obs.current_trick.played_cards:
            pc_is_trump = (obs.trump_suit and pc.card.suit == obs.trump_suit)
            pc_rv = rank_val_func(pc.card.rank)
            if pc_is_trump and not highest_is_trump:
                highest_rank = pc_rv
                highest_is_trump = True
            elif pc_is_trump and highest_is_trump:
                highest_rank = max(highest_rank, pc_rv)
            elif not pc_is_trump and not highest_is_trump:
                if pc.card.suit == (leading or pc.card.suit):
                    highest_rank = max(highest_rank, pc_rv)

        card_is_trump = (obs.trump_suit and card.suit == obs.trump_suit)
        if card_is_trump and not highest_is_trump:
            can_win = 1.0  # Any trump beats non-trump.
        elif card_is_trump and highest_is_trump:
            can_win = 1.0 if rv > highest_rank else 0.0
        elif not card_is_trump and not highest_is_trump:
            if leading and card.suit == leading:
                can_win = 1.0 if rv > highest_rank else 0.0
            else:
                can_win = 0.0  # Off-suit non-trump can't win.
        else:
            can_win = 0.0  # Non-trump can't beat trump.
    else:
        # Leading the trick — always "wins" by default.
        can_win = 1.0
    features[3] = can_win

    # 5. Creates void.
    suit_counts = Counter(c.suit for c in obs.hand)
    features[4] = 1.0 if suit_counts.get(card.suit, 0) == 1 else 0.0

    # 6. Is from longest suit.
    longest = max(suit_counts.values()) if suit_counts else 0
    features[5] = 1.0 if suit_counts.get(card.suit, 0) == longest else 0.0

    # 7. Relative rank position in this suit in hand.
    same_suit_ranks = sorted([rank_val_func(c.rank) for c in obs.hand if c.suit == card.suit])
    if len(same_suit_ranks) > 1:
        pos = same_suit_ranks.index(rv)
        features[6] = pos / (len(same_suit_ranks) - 1)
    else:
        features[6] = 1.0  # Only card = highest.

    # 8. Cards remaining in this suit (normalized by 13).
    features[7] = suit_counts.get(card.suit, 0) / 13.0

    return features


def state_features(obs, opp_voids: int = 0, rank_val_func=None, suit_idx_map=None,
                   suits_played: dict = None) -> np.ndarray:
    """
    Extract 28 numeric state features from the observation.
    
    Combined with 8 card features + 52 memory = 88 total input to CardEvaluator.
    """
    hand = obs.hand
    n_cards = len(hand)
    features = np.zeros(28)

    if rank_val_func is None:
        return features

    suit_counts = Counter(c.suit for c in hand)

    # 1-4. Suit distribution (sorted descending, normalized).
    sorted_counts = sorted(suit_counts.values(), reverse=True)
    for i in range(min(4, len(sorted_counts))):
        features[i] = sorted_counts[i] / 13.0

    # 5. Cards remaining (normalized).
    features[4] = n_cards / 13.0

    # 6. Position in trick (0-3, normalized).
    pos = 0
    if obs.current_trick and obs.current_trick.played_cards:
        pos = len(obs.current_trick.played_cards)
    features[5] = pos / 3.0

    # 7. Trump count (normalized).
    trump_count = sum(1 for c in hand if obs.trump_suit and c.suit == obs.trump_suit)
    features[6] = trump_count / 13.0

    # 8. High trump count (K, Q, A of trump).
    trump_highs = sum(1 for c in hand if obs.trump_suit and c.suit == obs.trump_suit
                      and rank_val_func(c.rank) >= 12)
    features[7] = trump_highs / 4.0

    # 9. Total high cards (normalized).
    highs = sum(1 for c in hand if rank_val_func(c.rank) >= 12)
    features[8] = highs / 13.0

    # 10. Aces count.
    aces = sum(1 for c in hand if rank_val_func(c.rank) == 14)
    features[9] = aces / 4.0

    # 11. Void count.
    voids = 4 - len(suit_counts)
    features[10] = voids / 4.0

    # 12. Team score difference (normalized).
    my_team = 0 if obs.player_id in (0, 2) else 1
    opp_team = 1 - my_team
    diff = obs.team_scores.get(my_team, 0) - obs.team_scores.get(opp_team, 0)
    features[11] = max(-1.0, min(1.0, diff / 15.0))

    # 13. Opponent voids known (normalized).
    features[12] = min(opp_voids / 6.0, 1.0)

    # 14. Game phase (0=early, 1=late).
    features[13] = 1.0 - (n_cards / 13.0)

    # 15-18. Current trick cards info (if any played).
    if obs.current_trick and obs.current_trick.played_cards:
        played = obs.current_trick.played_cards
        # Highest rank on table.
        max_rank = max(rank_val_func(pc.card.rank) for pc in played)
        features[14] = RANK_NORM.get(max_rank, 0.5)
        # Has trump been played in this trick?
        features[15] = 1.0 if any(obs.trump_suit and pc.card.suit == obs.trump_suit
                                   for pc in played) else 0.0
        # Number of cards played.
        features[16] = len(played) / 4.0
        # Is leading suit same as trump?
        features[17] = 1.0 if (obs.current_trick.leading_suit == obs.trump_suit) else 0.0

    # 19. Must lead trump.
    features[18] = 1.0 if obs.must_lead_trump else 0.0

    # 20-23. Cards played per suit (card counting — normalized by 13).
    if suits_played:
        for suit_idx in range(4):
            features[19 + suit_idx] = suits_played.get(suit_idx, 0) / 13.0

    # 24-27. Cards remaining per suit in the game (13 - played - in_hand).
    if suits_played and suit_idx_map:
        for suit, idx in suit_idx_map.items():
            in_hand = sum(1 for c in hand if c.suit == suit)
            played = suits_played.get(idx, 0)
            remaining = max(0, 13 - played - in_hand)
            features[23 + idx] = remaining / 13.0

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
