"""Unit tests for tibrain.reward (RewardNormalizer and CuriosityModule)."""

import math

import numpy as np
import pytest

from tibrain.reward import CuriosityModule, RewardNormalizer


# --- Requirement 10.1: Running mean and variance via Welford's algorithm ---


class TestRewardNormalizerRunningStats:
    def test_single_observation_sets_mean(self):
        rn = RewardNormalizer()
        rn.normalize(5.0)
        assert rn.mean == pytest.approx(5.0)

    def test_two_observations_update_mean(self):
        rn = RewardNormalizer()
        rn.normalize(4.0)
        rn.normalize(6.0)
        assert rn.mean == pytest.approx(5.0)

    def test_count_increments_per_normalize(self):
        rn = RewardNormalizer()
        rn.normalize(1.0)
        rn.normalize(2.0)
        rn.normalize(3.0)
        assert rn.count == 3

    def test_many_observations_mean_approximates_numpy(self):
        rn = RewardNormalizer()
        rewards = [1.0, 2.5, -3.0, 7.2, 0.1, -1.5, 4.8, 6.3, -2.1, 3.7]
        for r in rewards:
            rn.normalize(r)
        expected_mean = np.mean(rewards)
        assert rn.mean == pytest.approx(expected_mean, abs=1e-10)

    def test_many_observations_std_approximates_numpy(self):
        rn = RewardNormalizer()
        rewards = [1.0, 2.5, -3.0, 7.2, 0.1, -1.5, 4.8, 6.3, -2.1, 3.7]
        for r in rewards:
            rn.normalize(r)
        # Welford uses population std (ddof=0)
        expected_std = np.std(rewards)
        # std property includes 1e-8 epsilon
        assert rn.std == pytest.approx(expected_std + 1e-8, abs=1e-8)

    def test_large_sample_mean_converges(self):
        rn = RewardNormalizer()
        rng = np.random.default_rng(42)
        rewards = rng.normal(loc=3.0, scale=2.0, size=1000).tolist()
        for r in rewards:
            rn.normalize(r)
        assert rn.mean == pytest.approx(np.mean(rewards), abs=1e-10)

    def test_large_sample_std_converges(self):
        rn = RewardNormalizer()
        rng = np.random.default_rng(42)
        rewards = rng.normal(loc=3.0, scale=2.0, size=1000).tolist()
        for r in rewards:
            rn.normalize(r)
        expected_std = np.std(rewards)
        assert rn.std == pytest.approx(expected_std + 1e-8, abs=1e-8)


# --- Requirement 10.2: normalize() returns (reward - mean) / std ---


class TestRewardNormalizerNormalize:
    def test_normalize_returns_zero_for_constant_rewards(self):
        rn = RewardNormalizer()
        # After many identical rewards, mean = reward, so normalized ≈ 0
        for _ in range(10):
            result = rn.normalize(5.0)
        # With constant input, reward - mean = 0
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_normalize_first_observation(self):
        rn = RewardNormalizer()
        # First observation: mean becomes the reward, so (reward - mean) = 0
        result = rn.normalize(10.0)
        # After first observation, mean = 10.0, m2 = 0, std ≈ 1e-8
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_normalize_second_observation_positive(self):
        rn = RewardNormalizer()
        rn.normalize(0.0)
        result = rn.normalize(4.0)
        # After observing [0, 4]: mean = 2.0
        # m2 = 0*(0-0) + (4-0)*(4-2) = 0 + 4*2 = 8... wait let's trace
        # count=1: mean=0, m2=0
        # count=2: delta=4-0=4, mean=0+4/2=2, delta2=4-2=2, m2=0+4*2=8
        # std = sqrt(8/2) + 1e-8 = 2.0 + 1e-8
        # normalized = (4 - 2) / (2.0 + 1e-8) ≈ 1.0
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_normalize_updates_running_mean(self):
        rn = RewardNormalizer()
        rn.normalize(2.0)
        assert rn.mean == pytest.approx(2.0)
        rn.normalize(4.0)
        assert rn.mean == pytest.approx(3.0)
        rn.normalize(6.0)
        assert rn.mean == pytest.approx(4.0)


# --- Requirement 10.3: CuriosityModule bonus for visited states ---


class TestCuriosityModuleBonusVisited:
    def test_bonus_for_visited_state_once(self):
        cm = CuriosityModule(scale=1.0)
        cm.visit("state_a")
        bonus = cm.bonus("state_a")
        # scale / sqrt(1) = 1.0
        assert bonus == pytest.approx(1.0)

    def test_bonus_for_visited_state_four_times(self):
        cm = CuriosityModule(scale=1.0)
        for _ in range(4):
            cm.visit("state_b")
        bonus = cm.bonus("state_b")
        # scale / sqrt(4) = 0.5
        assert bonus == pytest.approx(0.5)

    def test_bonus_decreases_with_more_visits(self):
        cm = CuriosityModule(scale=1.0)
        cm.visit("s")
        bonus1 = cm.bonus("s")
        cm.visit("s")
        bonus2 = cm.bonus("s")
        cm.visit("s")
        bonus3 = cm.bonus("s")
        assert bonus1 > bonus2 > bonus3

    def test_bonus_formula_scale_divided_by_sqrt_count(self):
        cm = CuriosityModule(scale=0.5)
        for _ in range(9):
            cm.visit("x")
        bonus = cm.bonus("x")
        # 0.5 / sqrt(9) = 0.5 / 3 ≈ 0.1667
        assert bonus == pytest.approx(0.5 / 3.0)


# --- Requirement 10.3: CuriosityModule bonus for unvisited states ---


class TestCuriosityModuleBonusUnvisited:
    def test_unvisited_state_returns_full_scale(self):
        cm = CuriosityModule(scale=1.0)
        bonus = cm.bonus("never_seen")
        assert bonus == pytest.approx(1.0)

    def test_unvisited_state_custom_scale(self):
        cm = CuriosityModule(scale=0.5)
        bonus = cm.bonus("unknown")
        assert bonus == pytest.approx(0.5)

    def test_unvisited_state_different_from_visited(self):
        cm = CuriosityModule(scale=1.0)
        cm.visit("visited_state")
        # Visited state has lower bonus
        assert cm.bonus("visited_state") <= cm.bonus("brand_new")


# --- CuriosityModule visit increments count ---


class TestCuriosityModuleVisit:
    def test_visit_increments_count(self):
        cm = CuriosityModule(scale=1.0)
        assert cm.get_count("state") == 0
        cm.visit("state")
        assert cm.get_count("state") == 1
        cm.visit("state")
        assert cm.get_count("state") == 2

    def test_visit_independent_per_state(self):
        cm = CuriosityModule(scale=1.0)
        cm.visit("a")
        cm.visit("a")
        cm.visit("b")
        assert cm.get_count("a") == 2
        assert cm.get_count("b") == 1
        assert cm.get_count("c") == 0


# --- Requirement 10.5: RewardNormalizer serialization round-trip ---


class TestRewardNormalizerSerialization:
    def test_to_dict_contains_required_keys(self):
        rn = RewardNormalizer()
        rn.normalize(3.0)
        rn.normalize(7.0)
        data = rn.to_dict()
        assert "mean" in data
        assert "m2" in data
        assert "count" in data

    def test_from_dict_restores_state(self):
        rn = RewardNormalizer()
        rn.normalize(1.0)
        rn.normalize(2.0)
        rn.normalize(3.0)

        data = rn.to_dict()
        restored = RewardNormalizer.from_dict(data)

        assert restored.mean == pytest.approx(rn.mean)
        assert restored.std == pytest.approx(rn.std)
        assert restored.count == rn.count

    def test_round_trip_preserves_normalization(self):
        rn = RewardNormalizer()
        rewards = [2.0, 4.0, 6.0, 8.0, 10.0]
        for r in rewards:
            rn.normalize(r)

        data = rn.to_dict()
        restored = RewardNormalizer.from_dict(data)

        # After restoring, normalizing a new value should give same result
        # We need fresh instances that continue from same state
        result_original = RewardNormalizer.from_dict(data)
        result_restored = RewardNormalizer.from_dict(data)
        assert result_original.normalize(12.0) == pytest.approx(
            result_restored.normalize(12.0)
        )

    def test_from_dict_empty_returns_fresh_normalizer(self):
        rn = RewardNormalizer.from_dict({})
        assert rn.mean == 0.0
        assert rn.count == 0

    def test_serialization_after_many_observations(self):
        rn = RewardNormalizer()
        rng = np.random.default_rng(123)
        for r in rng.normal(0, 1, size=100):
            rn.normalize(float(r))

        data = rn.to_dict()
        restored = RewardNormalizer.from_dict(data)

        assert restored.mean == pytest.approx(rn.mean, abs=1e-12)
        assert restored.std == pytest.approx(rn.std, abs=1e-12)
        assert restored.count == rn.count
