"""
Neural Network function approximators for TIBRAIN.

Pure numpy implementation — no torch/tensorflow dependency.

Provides:
- Evaluator: Feedforward network with self-attention (residual connection)
  that maps input feature vector → single scalar Q-value.
"""

from __future__ import annotations

import numpy as np


class Evaluator:
    """
    Feedforward neural network with self-attention.
    Maps an input feature vector to a single scalar Q-value.

    Architecture:
    1. Self-attention layer with n_heads — learns which input features
       matter for this decision via importance weighting.
    2. Residual connection — attended features added to original input.
    3. 3 hidden layers (hidden_size neurons each) with ReLU activation.
    4. Single linear output — Q-value scalar.

    Training uses batch accumulation: samples are collected until batch_size
    is reached, then _flush_batch() performs backpropagation with gradient
    clipping to prevent exploding gradients.
    """

    def __init__(
        self,
        input_size: int = 138,
        hidden_size: int = 256,
        learning_rate: float = 0.001,
        n_heads: int = 4,
        batch_size: int = 32,
        max_grad_norm: float = 5.0,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.n_heads = n_heads
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        # === Self-Attention Layer ===
        # Attention weights: project input to attention scores per head.
        sa = np.sqrt(2.0 / input_size)
        self.w_attn = np.random.randn(input_size, n_heads) * sa
        self.b_attn = np.zeros(n_heads)
        # Value projection: transform input for attention weighting.
        self.w_val = np.random.randn(input_size, input_size) * sa
        self.b_val = np.zeros(input_size)

        # === Feedforward Layers (3 hidden + 1 output) ===
        s1 = np.sqrt(2.0 / input_size)
        s2 = np.sqrt(2.0 / hidden_size)

        self.w1 = np.random.randn(input_size, hidden_size) * s1
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, hidden_size) * s2
        self.b2 = np.zeros(hidden_size)
        self.w3 = np.random.randn(hidden_size, hidden_size) * s2
        self.b3 = np.zeros(hidden_size)
        self.w4 = np.random.randn(hidden_size, 1) * s2
        self.b4 = np.zeros(1)

        # Batch accumulation buffer.
        self._batch_x: list[np.ndarray] = []
        self._batch_y: list[float] = []

    def predict(self, x: np.ndarray) -> float:
        """Forward pass through attention + feedforward → single Q-value."""
        x_att = self._attend(x)
        h1 = np.maximum(0, x_att @ self.w1 + self.b1)
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)
        h3 = np.maximum(0, h2 @ self.w3 + self.b3)
        return float((h3 @ self.w4 + self.b4)[0])

    def predict_batch(self, batch: np.ndarray) -> np.ndarray:
        """Evaluate multiple feature vectors at once.

        Args:
            batch: 2D array of shape (n_samples, input_size).

        Returns:
            1D array of Q-values, one per sample.
        """
        # Vectorized attention for the whole batch.
        attended = self._attend_batch(batch)
        h1 = np.maximum(0, attended @ self.w1 + self.b1)
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)
        h3 = np.maximum(0, h2 @ self.w3 + self.b3)
        return (h3 @ self.w4 + self.b4).flatten()

    def _attend_batch(self, batch: np.ndarray) -> np.ndarray:
        """Vectorized self-attention for a batch of inputs.

        Args:
            batch: 2D array of shape (n_samples, input_size).

        Returns:
            2D array of shape (n_samples, input_size) with residual attention applied.
        """
        # Attention logits: (n_samples, n_heads)
        attn_logits = batch @ self.w_attn + self.b_attn
        # Stable softmax per sample.
        attn_exp = np.exp(attn_logits - attn_logits.max(axis=1, keepdims=True))
        attn_weights = attn_exp / (attn_exp.sum(axis=1, keepdims=True) + 1e-8)
        # Value projection with ReLU: (n_samples, input_size)
        values = np.maximum(0, batch @ self.w_val + self.b_val)
        # Scale chunks by head weights.
        chunk_size = self.input_size // self.n_heads
        attended = np.zeros_like(batch)
        for h in range(self.n_heads):
            start = h * chunk_size
            end = start + chunk_size
            attended[:, start:end] = values[:, start:end] * attn_weights[:, h:h+1]
        # Residual connection.
        return batch + attended

    def update(self, x: np.ndarray, target: float) -> None:
        """Accumulate a training sample; flush when batch is full.

        Args:
            x: Input feature vector.
            target: Target Q-value to train toward.
        """
        self._batch_x.append(x.copy())
        self._batch_y.append(target)
        if len(self._batch_x) >= self.batch_size:
            self._flush_batch()

    def copy(self) -> "Evaluator":
        """Create a frozen copy for use as a target network.

        The copy has independent weight arrays so updates to the original
        do not affect the copy.
        """
        clone = Evaluator(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            learning_rate=self.lr,
            n_heads=self.n_heads,
            batch_size=self.batch_size,
            max_grad_norm=self.max_grad_norm,
        )
        for attr in (
            "w_attn", "b_attn", "w_val", "b_val",
            "w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4",
        ):
            setattr(clone, attr, getattr(self, attr).copy())
        return clone

    def to_dict(self) -> dict:
        """Serialize all weights and configuration to a dictionary."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "lr": self.lr,
            "n_heads": self.n_heads,
            "batch_size": self.batch_size,
            "max_grad_norm": self.max_grad_norm,
            "w_attn": self.w_attn.tolist(),
            "b_attn": self.b_attn.tolist(),
            "w_val": self.w_val.tolist(),
            "b_val": self.b_val.tolist(),
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
            "w3": self.w3.tolist(),
            "b3": self.b3.tolist(),
            "w4": self.w4.tolist(),
            "b4": self.b4.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Evaluator":
        """Deserialize from a dictionary previously produced by to_dict().

        Args:
            data: Dictionary containing configuration and weight arrays.

        Returns:
            A fully initialized Evaluator with restored weights.
        """
        net = cls(
            input_size=data["input_size"],
            hidden_size=data["hidden_size"],
            learning_rate=data.get("lr", 0.001),
            n_heads=data.get("n_heads", 4),
            batch_size=data.get("batch_size", 32),
            max_grad_norm=data.get("max_grad_norm", 5.0),
        )
        for key in (
            "w_attn", "b_attn", "w_val", "b_val",
            "w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4",
        ):
            if key in data:
                setattr(net, key, np.array(data[key]))
        return net

    def _attend(self, x: np.ndarray) -> np.ndarray:
        """Self-attention: compute importance weights and re-weight input.

        Groups the input into n_heads chunks. Each head gets an attention
        weight via softmax over attention logits. The value-projected input
        is scaled per chunk by its head's attention weight. A residual
        connection adds the attended result back to the original input.
        """
        # Compute attention scores (one per head).
        attn_logits = x @ self.w_attn + self.b_attn  # shape: (n_heads,)
        # Stable softmax over heads.
        attn_exp = np.exp(attn_logits - attn_logits.max())
        attn_weights = attn_exp / (attn_exp.sum() + 1e-8)  # shape: (n_heads,)
        # Value projection with ReLU.
        values = np.maximum(0, x @ self.w_val + self.b_val)  # shape: (input_size,)
        # Scale input chunks by their head's attention weight.
        chunk_size = self.input_size // self.n_heads
        attended = np.zeros(self.input_size)
        for h in range(self.n_heads):
            start = h * chunk_size
            end = start + chunk_size
            attended[start:end] = values[start:end] * attn_weights[h]
        # Residual connection: original input + attended features.
        return x + attended

    def _flush_batch(self) -> None:
        """Train on accumulated batch with backpropagation and gradient clipping.

        Performs a full forward pass, computes MSE loss gradients, applies
        gradient clipping per weight matrix, then updates all weights via SGD.
        Also backpropagates through the attention layer for end-to-end training.
        """
        if not self._batch_x:
            return

        batch_x = np.array(self._batch_x)
        batch_y = np.array(self._batch_y)
        self._batch_x.clear()
        self._batch_y.clear()
        n = len(batch_x)

        # Forward pass.
        attended = np.array([self._attend(x) for x in batch_x])
        h1 = np.maximum(0, attended @ self.w1 + self.b1)
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)
        h3 = np.maximum(0, h2 @ self.w3 + self.b3)
        preds = (h3 @ self.w4 + self.b4).flatten()

        # Backward pass — MSE loss: L = (1/n) * sum((y - pred)^2)
        # dL/dpred = (-2/n) * (y - pred)
        errors = batch_y - preds
        d_out = (-2.0 / n) * errors.reshape(-1, 1)

        # Layer 4 (output).
        d_w4 = h3.T @ d_out
        d_b4 = d_out.sum(axis=0)
        d_h3 = d_out @ self.w4.T
        d_h3[h3 <= 0] = 0  # ReLU derivative

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
        d_w1 = attended.T @ d_h1
        d_b1 = d_h1.sum(axis=0)

        # Backprop through attention layer.
        d_attended = d_h1 @ self.w1.T
        d_w_val = batch_x.T @ d_attended
        d_b_val = d_attended.sum(axis=0)
        # Attention weight gradients (scaled for stability).
        d_w_attn = batch_x.T @ (d_attended @ self.w_attn) * 0.1

        # Gradient clipping — clip each weight gradient independently.
        for grad in (d_w1, d_w2, d_w3, d_w4, d_w_val, d_w_attn):
            norm = np.linalg.norm(grad)
            if norm > self.max_grad_norm:
                grad *= self.max_grad_norm / norm

        # SGD update — feedforward layers.
        self.w1 -= self.lr * d_w1
        self.b1 -= self.lr * d_b1
        self.w2 -= self.lr * d_w2
        self.b2 -= self.lr * d_b2
        self.w3 -= self.lr * d_w3
        self.b3 -= self.lr * d_b3
        self.w4 -= self.lr * d_w4
        self.b4 -= self.lr * d_b4

        # SGD update — attention layers (lower effective LR for stability).
        self.w_val -= self.lr * d_w_val
        self.b_val -= self.lr * d_b_val
        self.w_attn -= self.lr * 0.1 * d_w_attn


class QNetwork:
    """
    Fixed-output feedforward network for small discrete action spaces.

    Architecture:
    1. Input layer → hidden layer (ReLU activation)
    2. Hidden layer → output layer (linear, one output per action)

    Unlike Evaluator which produces a single scalar Q-value, QNetwork
    outputs Q-values for ALL actions simultaneously, making it suitable
    for environments with small, fixed action spaces.
    """

    def __init__(
        self,
        input_size: int = 32,
        hidden_size: int = 64,
        output_size: int = 8,
        learning_rate: float = 0.001,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        # He initialization for ReLU layers.
        s1 = np.sqrt(2.0 / input_size)
        s2 = np.sqrt(2.0 / hidden_size)
        self.w1 = np.random.randn(input_size, hidden_size) * s1
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size) * s2
        self.b2 = np.zeros(output_size)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward pass → Q-values for all actions.

        Args:
            x: Input feature vector of shape (input_size,).

        Returns:
            1D array of Q-values, one per action (shape: output_size).
        """
        h = np.maximum(0, x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

    def update(self, x: np.ndarray, target_idx: int, target_value: float) -> float:
        """Update Q-value for a single action index. Returns squared error.

        Performs a single-sample gradient descent step to move the Q-value
        at target_idx toward target_value.

        Args:
            x: Input feature vector of shape (input_size,).
            target_idx: Index of the action to update.
            target_value: Desired Q-value for that action.

        Returns:
            Squared error before the update: (target_value - predicted)^2.
        """
        # Forward pass.
        h = np.maximum(0, x @ self.w1 + self.b1)
        q_values = h @ self.w2 + self.b2

        # Compute error for the target action only.
        error = target_value - q_values[target_idx]

        # Backward pass — gradient only flows through target_idx output.
        d_out = np.zeros(self.output_size)
        d_out[target_idx] = -2.0 * error

        # Layer 2 gradients.
        d_w2 = np.outer(h, d_out)
        d_b2 = d_out

        # Backprop through ReLU to layer 1.
        d_h = d_out @ self.w2.T
        d_h[h <= 0] = 0  # ReLU derivative
        d_w1 = np.outer(x, d_h)
        d_b1 = d_h

        # SGD update.
        self.w1 -= self.lr * d_w1
        self.b1 -= self.lr * d_b1
        self.w2 -= self.lr * d_w2
        self.b2 -= self.lr * d_b2

        return error ** 2

    def to_dict(self) -> dict:
        """Serialize all weights and configuration to a dictionary."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "lr": self.lr,
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QNetwork":
        """Deserialize from a dictionary previously produced by to_dict().

        Args:
            data: Dictionary containing configuration and weight arrays.

        Returns:
            A fully initialized QNetwork with restored weights.
        """
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
