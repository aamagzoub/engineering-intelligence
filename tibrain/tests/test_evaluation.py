"""Unit tests for tibrain.evaluation (EloTracker and MetaLearner)."""

import pytest

from tibrain.evaluation import EloTracker, MetaLearner


# --- Requirement 13.1: Elo initial value and K-factor ---


class TestEloTrackerInit:
    def test_initial_elo_is_1000(self):
        tracker = EloTracker()
        assert tracker.elo == 1000.0

    def test_default_k_factor_is_32(self):
        tracker = EloTracker()
        assert tracker.k_factor == 32.0

    def test_custom_initial_elo(self):
        tracker = EloTracker(initial_elo=1500.0)
        assert tracker.elo == 1500.0

    def test_custom_k_factor(self):
        tracker = EloTracker(k_factor=16.0)
        assert tracker.k_factor == 16.0


# --- Requirement 13.1: Elo update formula correctness ---


class TestEloUpdate:
    def test_win_against_equal_opponent_increases_elo(self):
        tracker = EloTracker()
        tracker.update(won=True, opponent_elo=1000.0)
        # Expected score is 0.5 when ratings are equal
        # Elo change = 32 * (1.0 - 0.5) = 16
        assert tracker.elo == pytest.approx(1016.0)

    def test_loss_against_equal_opponent_decreases_elo(self):
        tracker = EloTracker()
        tracker.update(won=False, opponent_elo=1000.0)
        # Elo change = 32 * (0.0 - 0.5) = -16
        assert tracker.elo == pytest.approx(984.0)

    def test_win_against_stronger_opponent_gives_larger_gain(self):
        tracker = EloTracker()
        tracker.update(won=True, opponent_elo=1400.0)
        # Expected = 1 / (1 + 10^((1400 - 1000)/400)) = 1 / (1 + 10^1) ≈ 0.0909
        # Gain = 32 * (1.0 - 0.0909) ≈ 29.09
        assert tracker.elo > 1025.0  # Large gain

    def test_loss_against_weaker_opponent_gives_larger_penalty(self):
        tracker = EloTracker()
        tracker.update(won=False, opponent_elo=600.0)
        # Expected = 1 / (1 + 10^((600 - 1000)/400)) = 1 / (1 + 10^(-1)) ≈ 0.9091
        # Loss = 32 * (0.0 - 0.9091) ≈ -29.09
        assert tracker.elo < 975.0  # Large penalty

    def test_standard_elo_formula_exact(self):
        tracker = EloTracker(initial_elo=1200.0, k_factor=32.0)
        tracker.update(won=True, opponent_elo=1000.0)
        # Expected = 1 / (1 + 10^((1000-1200)/400)) = 1 / (1 + 10^(-0.5))
        expected_score = 1.0 / (1.0 + 10 ** (-0.5))
        new_elo = 1200.0 + 32.0 * (1.0 - expected_score)
        assert tracker.elo == pytest.approx(new_elo)

    def test_consecutive_updates_accumulate(self):
        tracker = EloTracker()
        tracker.update(won=True, opponent_elo=1000.0)
        tracker.update(won=True, opponent_elo=1000.0)
        # After first win: 1016; second win uses new elo for expected
        assert tracker.elo > 1030.0


# --- Requirement 13.2: Snapshot recording and retention ---


class TestEloTrackerRecord:
    def test_record_stores_snapshot(self):
        tracker = EloTracker()
        tracker.record(episode=1)
        assert len(tracker.history) == 1
        assert tracker.history[0] == (1, 1000.0)

    def test_record_multiple_snapshots(self):
        tracker = EloTracker()
        tracker.record(episode=10)
        tracker.update(won=True, opponent_elo=1000.0)
        tracker.record(episode=20)
        assert len(tracker.history) == 2
        assert tracker.history[0] == (10, 1000.0)
        assert tracker.history[1][0] == 20
        assert tracker.history[1][1] == pytest.approx(1016.0)

    def test_retains_last_100_snapshots(self):
        tracker = EloTracker()
        for i in range(150):
            tracker.record(episode=i)
        assert len(tracker.history) == 100
        # Should have episodes 50–149 retained
        assert tracker.history[0] == (50, 1000.0)
        assert tracker.history[-1] == (149, 1000.0)

    def test_exactly_100_snapshots_retained(self):
        tracker = EloTracker()
        for i in range(100):
            tracker.record(episode=i)
        assert len(tracker.history) == 100

    def test_101_snapshots_trimmed_to_100(self):
        tracker = EloTracker()
        for i in range(101):
            tracker.record(episode=i)
        assert len(tracker.history) == 100
        assert tracker.history[0] == (1, 1000.0)


# --- EloTracker serialization ---


class TestEloTrackerSerialization:
    def test_to_dict_round_trip(self):
        tracker = EloTracker(initial_elo=1200.0, k_factor=16.0)
        tracker.update(won=True, opponent_elo=1100.0)
        tracker.record(episode=5)

        data = tracker.to_dict()
        restored = EloTracker.from_dict(data)

        assert restored.elo == pytest.approx(tracker.elo)
        assert restored.k_factor == tracker.k_factor
        assert len(restored.history) == len(tracker.history)
        assert restored.history[0][0] == tracker.history[0][0]

    def test_from_dict_defaults(self):
        tracker = EloTracker.from_dict({})
        assert tracker.elo == 1000.0
        assert tracker.k_factor == 32.0
        assert tracker.history == []


# --- Requirement 13.4: MetaLearner suggests reducing epsilon when improving >10% ---


class TestMetaLearnerImproving:
    def test_suggests_epsilon_reduction_when_improving(self):
        ml = MetaLearner(window_size=50)
        # Fill window with low scores first (average ~5.0)
        for _ in range(40):
            ml.record_score(5.0)
        # Add high recent scores (average ~7.0, which is >10% above 5.0)
        for _ in range(10):
            ml.record_score(7.0)

        adjustments = ml.suggest_adjustments(current_epsilon=0.2, episode=200)
        assert "epsilon" in adjustments
        assert adjustments["epsilon"] == pytest.approx(0.2 * 0.95)

    def test_epsilon_reduction_respects_minimum(self):
        ml = MetaLearner(window_size=50)
        for _ in range(40):
            ml.record_score(5.0)
        for _ in range(10):
            ml.record_score(7.0)

        # With very low current epsilon, should clamp at 0.02
        adjustments = ml.suggest_adjustments(current_epsilon=0.015, episode=200)
        assert "epsilon" in adjustments
        assert adjustments["epsilon"] == 0.02


# --- Requirement 13.5: MetaLearner suggests increasing epsilon when declining >20% ---


class TestMetaLearnerDeclining:
    def test_suggests_epsilon_increase_when_declining(self):
        ml = MetaLearner(window_size=50)
        # Fill window with high scores first (average ~10.0)
        for _ in range(40):
            ml.record_score(10.0)
        # Add low recent scores (average ~6.0, which is <80% of ~9.2 overall avg)
        for _ in range(10):
            ml.record_score(6.0)

        adjustments = ml.suggest_adjustments(current_epsilon=0.1, episode=200)
        assert "epsilon" in adjustments
        assert adjustments["epsilon"] == pytest.approx(0.1 * 1.1)

    def test_epsilon_increase_capped_at_0_3(self):
        ml = MetaLearner(window_size=50)
        for _ in range(40):
            ml.record_score(10.0)
        for _ in range(10):
            ml.record_score(6.0)

        # With epsilon already at 0.3, should not increase further
        adjustments = ml.suggest_adjustments(current_epsilon=0.3, episode=200)
        # When epsilon is already >= 0.3, no increase is suggested
        assert adjustments.get("epsilon", 0.3) <= 0.3

    def test_epsilon_increase_near_cap(self):
        ml = MetaLearner(window_size=50)
        for _ in range(40):
            ml.record_score(10.0)
        for _ in range(10):
            ml.record_score(6.0)

        adjustments = ml.suggest_adjustments(current_epsilon=0.28, episode=200)
        assert "epsilon" in adjustments
        # 0.28 * 1.1 = 0.308, capped to 0.3
        assert adjustments["epsilon"] == pytest.approx(0.3)


# --- MetaLearner no adjustment needed ---


class TestMetaLearnerNoAdjustment:
    def test_returns_empty_dict_when_no_trend(self):
        ml = MetaLearner(window_size=50)
        # Fill with stable scores — recent ~= average
        for _ in range(50):
            ml.record_score(5.0)

        adjustments = ml.suggest_adjustments(current_epsilon=0.2, episode=200)
        assert adjustments == {}

    def test_returns_empty_dict_when_empty(self):
        ml = MetaLearner(window_size=50)
        adjustments = ml.suggest_adjustments(current_epsilon=0.2, episode=200)
        assert adjustments == {}


# --- MetaLearner should_adjust ---


class TestMetaLearnerShouldAdjust:
    def test_should_adjust_after_interval(self):
        ml = MetaLearner(window_size=50, adjustment_interval=200)
        for _ in range(30):
            ml.record_score(5.0)
        assert ml.should_adjust(episode=200)

    def test_should_not_adjust_before_interval(self):
        ml = MetaLearner(window_size=50, adjustment_interval=200)
        for _ in range(30):
            ml.record_score(5.0)
        assert not ml.should_adjust(episode=100)

    def test_should_not_adjust_with_too_few_scores(self):
        ml = MetaLearner(window_size=50, adjustment_interval=200)
        for _ in range(10):
            ml.record_score(5.0)
        assert not ml.should_adjust(episode=200)

    def test_should_adjust_requires_minimum_30_scores(self):
        ml = MetaLearner(window_size=50, adjustment_interval=200)
        for _ in range(29):
            ml.record_score(5.0)
        assert not ml.should_adjust(episode=200)
        ml.record_score(5.0)
        assert ml.should_adjust(episode=200)

    def test_should_adjust_resets_after_suggest(self):
        ml = MetaLearner(window_size=50, adjustment_interval=200)
        for _ in range(50):
            ml.record_score(5.0)
        assert ml.should_adjust(episode=200)
        # Calling suggest_adjustments updates _last_adjustment
        ml.suggest_adjustments(current_epsilon=0.2, episode=200)
        # Now shouldn't adjust until episode 400
        assert not ml.should_adjust(episode=300)
        assert ml.should_adjust(episode=400)
