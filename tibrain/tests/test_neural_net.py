"""Unit tests for tibrain.neural_net (Evaluator and QNetwork).

Validates Requirements: 8.1, 8.3, 8.4, 8.7, 8.9
"""

import numpy as np
import pytest

from tibrain.neural_net import Evaluator, QNetwork


# ---------------------------------------------------------------------------
# Evaluator Tests
# ---------------------------------------------------------------------------


class TestEvaluatorPredict:
    """Test Evaluator.predict(x) returns a float (Req 8.1)."""

    def test_predict_returns_float(self):
        np.random.seed(42)
        ev = Evaluator(input_size=16, hidden_size=32, n_heads=4)
        x = np.random.randn(16)
        result = ev.predict(x)
        assert isinstance(result, float)

    def test_predict_deterministic_same_input(self):
        np.random.seed(42)
        ev = Evaluator(input_size=16, hidden_size=32, n_heads=4)
        x = np.ones(16) * 0.5
        r1 = ev.predict(x)
        r2 = ev.predict(x)
        assert r1 == r2

    def test_predict_different_inputs_may_differ(self):
        np.random.seed(42)
        ev = Evaluator(input_size=16, hidden_size=32, n_heads=4)
        x1 = np.ones(16)
        x2 = -np.ones(16)
        r1 = ev.predict(x1)
        r2 = ev.predict(x2)
        # Different inputs should generally produce different outputs
        # (with random weights, identical output is astronomically unlikely)
        assert r1 != r2


class TestEvaluatorPredictBatch:
    """Test Evaluator.predict_batch(batch) returns array of correct shape (Req 8.3)."""

    def test_batch_returns_correct_shape(self):
        np.random.seed(42)
        ev = Evaluator(input_size=16, hidden_size=32, n_heads=4)
        batch = np.random.randn(5, 16)
        result = ev.predict_batch(batch)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)

    def test_batch_single_item_matches_predict(self):
        np.random.seed(42)
        ev = Evaluator(input_size=16, hidden_size=32, n_heads=4)
        x = np.random.randn(16)
        single = ev.predict(x)
        batch_result = ev.predict_batch(x.reshape(1, -1))
        assert np.isclose(single, batch_result[0], atol=1e-10)

    def test_batch_multiple_items_independent(self):
        np.random.seed(42)
        ev = Evaluator(input_size=16, hidden_size=32, n_heads=4)
        batch = np.random.randn(3, 16)
        results = ev.predict_batch(batch)
        # Each result should match individual predict calls
        for i in range(3):
            expected = ev.predict(batch[i])
            assert np.isclose(results[i], expected, atol=1e-10)


class TestEvaluatorUpdate:
    """Test Evaluator.update() accumulates samples and flushes at batch_size (Req 8.4)."""

    def test_update_accumulates_without_immediate_change(self):
        np.random.seed(42)
        ev = Evaluator(input_size=16, hidden_size=32, n_heads=4, batch_size=8)
        x = np.random.randn(16)
        pred_before = ev.predict(x)
        # Add fewer than batch_size samples — no flush yet
        for _ in range(7):
            ev.update(np.random.randn(16), 1.0)
        pred_after = ev.predict(x)
        # Weights should not have changed (no flush happened)
        assert pred_before == pred_after

    def test_update_flushes_at_batch_size(self):
        np.random.seed(42)
        ev = Evaluator(input_size=16, hidden_size=32, n_heads=4, batch_size=8)
        x = np.random.randn(16)
        pred_before = ev.predict(x)
        # Add exactly batch_size samples — triggers flush
        for _ in range(8):
            ev.update(np.random.randn(16), 1.0)
        pred_after = ev.predict(x)
        # After flush, predictions should change
        assert pred_before != pred_after

    def test_update_multiple_flushes_continue_changing(self):
        np.random.seed(42)
        ev = Evaluator(input_size=16, hidden_size=32, n_heads=4, batch_size=4)
        x = np.random.randn(16)
        predictions = [ev.predict(x)]
        # Perform two flushes
        for i in range(8):
            ev.update(np.random.randn(16), 1.0)
            if (i + 1) % 4 == 0:
                predictions.append(ev.predict(x))
        # Predictions should change after each flush
        assert predictions[0] != predictions[1]
        assert predictions[1] != predictions[2]


class TestEvaluatorCopy:
    """Test Evaluator.copy() produces independent weights (Req 8.9)."""

    def test_copy_produces_same_predictions(self):
        np.random.seed(42)
        ev = Evaluator(input_size=16, hidden_size=32, n_heads=4)
        copy = ev.copy()
        x = np.random.randn(16)
        assert ev.predict(x) == copy.predict(x)

    def test_copy_independent_after_original_update(self):
        np.random.seed(42)
        ev = Evaluator(input_size=16, hidden_size=32, n_heads=4, batch_size=4)
        copy = ev.copy()
        x = np.random.randn(16)
        pred_copy_before = copy.predict(x)
        # Update the original enough to trigger a flush
        for _ in range(4):
            ev.update(np.random.randn(16), 5.0)
        # Original should change, copy should not
        pred_copy_after = copy.predict(x)
        assert pred_copy_before == pred_copy_after

    def test_copy_independent_after_copy_update(self):
        np.random.seed(42)
        ev = Evaluator(input_size=16, hidden_size=32, n_heads=4, batch_size=4)
        copy = ev.copy()
        x = np.random.randn(16)
        pred_orig_before = ev.predict(x)
        # Update the copy enough to trigger a flush
        for _ in range(4):
            copy.update(np.random.randn(16), 5.0)
        # Copy should change, original should not
        pred_orig_after = ev.predict(x)
        assert pred_orig_before == pred_orig_after


class TestEvaluatorSerialization:
    """Test Evaluator.to_dict() / from_dict() round-trip preserves predictions."""

    def test_round_trip_preserves_predictions(self):
        np.random.seed(42)
        ev = Evaluator(input_size=16, hidden_size=32, n_heads=4)
        x = np.random.randn(16)
        pred_before = ev.predict(x)
        data = ev.to_dict()
        restored = Evaluator.from_dict(data)
        pred_after = restored.predict(x)
        assert np.isclose(pred_before, pred_after, atol=1e-10)

    def test_round_trip_preserves_config(self):
        np.random.seed(42)
        ev = Evaluator(input_size=20, hidden_size=64, learning_rate=0.01, n_heads=2)
        data = ev.to_dict()
        restored = Evaluator.from_dict(data)
        assert restored.input_size == 20
        assert restored.hidden_size == 64
        assert restored.lr == 0.01
        assert restored.n_heads == 2

    def test_round_trip_batch_predictions(self):
        np.random.seed(42)
        ev = Evaluator(input_size=16, hidden_size=32, n_heads=4)
        batch = np.random.randn(5, 16)
        preds_before = ev.predict_batch(batch)
        data = ev.to_dict()
        restored = Evaluator.from_dict(data)
        preds_after = restored.predict_batch(batch)
        np.testing.assert_allclose(preds_before, preds_after, atol=1e-10)


# ---------------------------------------------------------------------------
# QNetwork Tests
# ---------------------------------------------------------------------------


class TestQNetworkPredict:
    """Test QNetwork.predict(x) returns array of shape (output_size,) (Req 8.7)."""

    def test_predict_returns_correct_shape(self):
        np.random.seed(42)
        qn = QNetwork(input_size=16, hidden_size=32, output_size=5)
        x = np.random.randn(16)
        result = qn.predict(x)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)

    def test_predict_deterministic(self):
        np.random.seed(42)
        qn = QNetwork(input_size=16, hidden_size=32, output_size=5)
        x = np.random.randn(16)
        r1 = qn.predict(x)
        r2 = qn.predict(x)
        np.testing.assert_array_equal(r1, r2)

    def test_predict_different_output_sizes(self):
        np.random.seed(42)
        for out_size in [1, 4, 10]:
            qn = QNetwork(input_size=8, hidden_size=16, output_size=out_size)
            x = np.random.randn(8)
            result = qn.predict(x)
            assert result.shape == (out_size,)


class TestQNetworkUpdate:
    """Test QNetwork.update() reduces squared error after multiple updates (Req 8.7, 8.8)."""

    def test_update_returns_squared_error(self):
        np.random.seed(42)
        qn = QNetwork(input_size=8, hidden_size=16, output_size=4)
        x = np.random.randn(8)
        sq_err = qn.update(x, target_idx=2, target_value=1.0)
        assert isinstance(sq_err, float)
        assert sq_err >= 0.0

    def test_update_reduces_error_over_iterations(self):
        np.random.seed(42)
        qn = QNetwork(input_size=8, hidden_size=16, output_size=4, learning_rate=0.01)
        x = np.random.randn(8)
        target_idx = 1
        target_value = 2.0
        initial_error = qn.update(x, target_idx, target_value)
        # Perform many updates on same sample
        final_error = initial_error
        for _ in range(100):
            final_error = qn.update(x, target_idx, target_value)
        assert final_error < initial_error

    def test_update_moves_prediction_toward_target(self):
        np.random.seed(42)
        qn = QNetwork(input_size=8, hidden_size=16, output_size=4, learning_rate=0.01)
        x = np.random.randn(8)
        target_idx = 0
        target_value = 5.0
        pred_before = qn.predict(x)[target_idx]
        for _ in range(50):
            qn.update(x, target_idx, target_value)
        pred_after = qn.predict(x)[target_idx]
        # Prediction should move closer to target
        assert abs(target_value - pred_after) < abs(target_value - pred_before)


class TestQNetworkSerialization:
    """Test QNetwork.to_dict() / from_dict() round-trip."""

    def test_round_trip_preserves_predictions(self):
        np.random.seed(42)
        qn = QNetwork(input_size=8, hidden_size=16, output_size=4)
        x = np.random.randn(8)
        preds_before = qn.predict(x)
        data = qn.to_dict()
        restored = QNetwork.from_dict(data)
        preds_after = restored.predict(x)
        np.testing.assert_allclose(preds_before, preds_after, atol=1e-10)

    def test_round_trip_preserves_config(self):
        np.random.seed(42)
        qn = QNetwork(input_size=12, hidden_size=24, output_size=6, learning_rate=0.005)
        data = qn.to_dict()
        restored = QNetwork.from_dict(data)
        assert restored.input_size == 12
        assert restored.hidden_size == 24
        assert restored.output_size == 6
        assert restored.lr == 0.005

    def test_round_trip_after_updates(self):
        np.random.seed(42)
        qn = QNetwork(input_size=8, hidden_size=16, output_size=4, learning_rate=0.01)
        x = np.random.randn(8)
        # Train for a few steps
        for _ in range(10):
            qn.update(x, target_idx=2, target_value=3.0)
        preds_before = qn.predict(x)
        data = qn.to_dict()
        restored = QNetwork.from_dict(data)
        preds_after = restored.predict(x)
        np.testing.assert_allclose(preds_before, preds_after, atol=1e-10)
