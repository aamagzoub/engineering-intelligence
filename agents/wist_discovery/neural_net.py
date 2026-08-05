"""
Minimal Neural Network for Q-value approximation.

Pure numpy implementation — no torch/tensorflow dependency.
Domain-agnostic: takes a fixed-size feature vector, outputs Q-values for actions.

Architecture: 2-layer MLP with ReLU activation.
- Input: state feature vector (variable size based on encoding)
- Hidden: 128 neurons
- Output: Q-value estimates for each possible action
"""

import numpy as np


class QNetwork:
    """Simple 2-layer neural network for Q-value approximation."""

    def __init__(self, input_size: int, hidden_size: int = 128, output_size: int = 20,
                 learning_rate: float = 0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        # Xavier initialization.
        scale1 = np.sqrt(2.0 / input_size)
        scale2 = np.sqrt(2.0 / hidden_size)
        self.w1 = np.random.randn(input_size, hidden_size) * scale1
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size) * scale2
        self.b2 = np.zeros(output_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. Returns Q-values for all actions."""
        self._x = x
        self._h = np.maximum(0, x @ self.w1 + self.b1)  # ReLU
        return self._h @ self.w2 + self.b2

    def update(self, x: np.ndarray, target_idx: int, target_value: float):
        """Single-sample SGD update for one action's Q-value."""
        # Forward.
        h = np.maximum(0, x @ self.w1 + self.b1)
        q_values = h @ self.w2 + self.b2

        # Compute error only for the target action.
        error = target_value - q_values[target_idx]

        # Backward pass (only for target_idx output neuron).
        d_out = np.zeros(self.output_size)
        d_out[target_idx] = -2 * error  # MSE gradient

        # Gradient for w2, b2.
        d_w2 = np.outer(h, d_out)
        d_b2 = d_out

        # Gradient for hidden layer.
        d_h = d_out @ self.w2.T
        d_h[h <= 0] = 0  # ReLU gradient

        # Gradient for w1, b1.
        d_w1 = np.outer(x, d_h)
        d_b1 = d_h

        # SGD step.
        self.w1 -= self.lr * d_w1
        self.b1 -= self.lr * d_b1
        self.w2 -= self.lr * d_w2
        self.b2 -= self.lr * d_b2

        return error ** 2  # Return loss for monitoring.

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict Q-values (same as forward, but cleaner name)."""
        return self.forward(x)

    def to_dict(self) -> dict:
        """Serialize network weights."""
        return {
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "lr": self.lr,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QNetwork":
        """Deserialize network weights."""
        net = cls(
            input_size=data["input_size"],
            hidden_size=data["hidden_size"],
            output_size=data["output_size"],
            learning_rate=data.get("lr", 0.001),
        )
        net.w1 = np.array(data["w1"])
        net.b1 = np.array(data["b1"])
        net.w2 = np.array(data["w2"])
        net.b2 = np.array(data["b2"])
        return net


def state_to_features(state_str: str, max_len: int = 32) -> np.ndarray:
    """
    Convert a state string to a fixed-size numeric feature vector.
    
    Domain-agnostic: works by converting each character to its ASCII value
    and padding/truncating to fixed length. The neural net learns what
    these positions mean through training.
    """
    # Convert each char to a normalized numeric value.
    features = np.zeros(max_len)
    for i, ch in enumerate(state_str[:max_len]):
        features[i] = ord(ch) / 127.0  # Normalize to ~[0, 1]
    return features


# Fixed action space for play decisions.
PLAY_ACTIONS = [
    "AFNLK", "AFNLV", "AFNSK", "AFNSV",  # Ace, follow, non-trump
    "AFTNLK", "AFTNLV", "AFTNSK", "AFTNSV",  # Ace, follow, trump (not real but pad)
    "HFNLK", "HFNSK", "HFTNLK", "HFTNSK",  # High, follow
    "MFNLK", "MFNSK", "MFTNLK", "MFTNSK",  # Mid, follow
    "LFNLK", "LFNSK", "LFTNLK", "LFTNSK",  # Low, follow
    "AONLK", "AONSK", "AOTNLK", "AOTNSK",  # Ace, off-suit
    "HONLK", "HONSK", "HOTNLK", "HOTNSK",  # High, off-suit
    "MONLK", "MONSK", "MOTNLK", "MOTNSK",  # Mid, off-suit
    "LONLK", "LONSK", "LOTNLK", "LOTNSK",  # Low, off-suit
]

# Build action-to-index mapping.
PLAY_ACTION_TO_IDX = {a: i for i, a in enumerate(PLAY_ACTIONS)}
NUM_PLAY_ACTIONS = len(PLAY_ACTIONS)

# Bid actions: PASS + B7..B13 = 8 actions.
BID_ACTIONS = ["PASS"] + [f"B{v}" for v in range(7, 14)]
BID_ACTION_TO_IDX = {a: i for i, a in enumerate(BID_ACTIONS)}
NUM_BID_ACTIONS = len(BID_ACTIONS)


def get_play_action_idx(action_str: str) -> int:
    """Get index for a play action string. Returns closest match if exact not found."""
    if action_str in PLAY_ACTION_TO_IDX:
        return PLAY_ACTION_TO_IDX[action_str]
    # Fuzzy match: find closest by prefix.
    for key, idx in PLAY_ACTION_TO_IDX.items():
        if key[:3] == action_str[:3]:
            return idx
    return 0  # Default to first action.


def get_bid_action_idx(action_str: str) -> int:
    """Get index for a bid action string."""
    return BID_ACTION_TO_IDX.get(action_str, 0)
