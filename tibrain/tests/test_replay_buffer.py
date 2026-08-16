"""Unit tests for tibrain.replay_buffer.ReplayBuffer.

Validates: Requirements 7.1, 7.2, 7.4, 7.5
"""

from __future__ import annotations

import numpy as np
import pytest

from tibrain.replay_buffer import ReplayBuffer


class TestAddAndLen:
    """Test add() stores entries and len() increases."""

    def test_empty_buffer_has_length_zero(self) -> None:
        buf = ReplayBuffer(capacity=10)
        assert len(buf) == 0

    def test_add_single_entry_increases_len(self) -> None:
        buf = ReplayBuffer(capacity=10)
        buf.add("s1", "a1", 1.0, "s2", 0.5)
        assert len(buf) == 1

    def test_add_multiple_entries_increases_len(self) -> None:
        buf = ReplayBuffer(capacity=10)
        for i in range(5):
            buf.add(f"s{i}", f"a{i}", float(i), f"s{i+1}", 0.1 * i)
        assert len(buf) == 5


class TestCapacityEviction:
    """Test capacity is enforced via ring buffer eviction."""

    def test_buffer_does_not_exceed_capacity(self) -> None:
        capacity = 5
        buf = ReplayBuffer(capacity=capacity)
        for i in range(10):
            buf.add(f"s{i}", f"a{i}", float(i), f"s{i+1}", 0.1)
        assert len(buf) == capacity

    def test_oldest_entries_are_evicted(self) -> None:
        capacity = 3
        buf = ReplayBuffer(capacity=capacity)
        # Add 5 entries: s0,s1,s2,s3,s4
        for i in range(5):
            buf.add(f"s{i}", f"a{i}", float(i), f"s{i+1}", 1.0)
        # After eviction, only s2, s3, s4 should remain (oldest s0, s1 evicted)
        states_in_buffer = {entry[0] for entry in buf._buffer}
        assert "s0" not in states_in_buffer
        assert "s1" not in states_in_buffer
        assert "s2" in states_in_buffer
        assert "s3" in states_in_buffer
        assert "s4" in states_in_buffer

    def test_capacity_property_returns_configured_value(self) -> None:
        buf = ReplayBuffer(capacity=42)
        assert buf.capacity == 42


class TestSampleBasics:
    """Test sample() returns tuples of (state, action, reward, next_state) — no td_error."""

    def test_sample_returns_four_element_tuples(self) -> None:
        buf = ReplayBuffer(capacity=10)
        buf.add("s1", "a1", 1.0, "s2", 0.5)
        buf.add("s2", "a2", 2.0, "s3", 0.3)
        buf.add("s3", "a3", 3.0, "s4", 0.7)

        samples = buf.sample(2)
        assert len(samples) == 2
        for s in samples:
            assert len(s) == 4  # (state, action, reward, next_state)

    def test_sample_does_not_include_td_error(self) -> None:
        buf = ReplayBuffer(capacity=10)
        buf.add("state_A", "action_X", 5.0, "state_B", 99.9)

        samples = buf.sample(1)
        assert len(samples) == 1
        state, action, reward, next_state = samples[0]
        assert state == "state_A"
        assert action == "action_X"
        assert reward == 5.0
        assert next_state == "state_B"

    def test_sample_returns_valid_entries_from_buffer(self) -> None:
        buf = ReplayBuffer(capacity=10)
        added_states = set()
        for i in range(5):
            buf.add(f"s{i}", f"a{i}", float(i), f"ns{i}", 0.1 * i)
            added_states.add(f"s{i}")

        samples = buf.sample(3)
        for state, action, reward, next_state in samples:
            assert state in added_states


class TestSampleWhenBufferSmall:
    """Test sample() when buffer is smaller than batch_size returns all entries."""

    def test_sample_returns_all_when_fewer_than_batch_size(self) -> None:
        buf = ReplayBuffer(capacity=10)
        buf.add("s1", "a1", 1.0, "s2", 0.5)
        buf.add("s2", "a2", 2.0, "s3", 0.3)

        samples = buf.sample(batch_size=10)
        assert len(samples) == 2  # Only 2 entries available

    def test_sample_exact_size_returns_all(self) -> None:
        buf = ReplayBuffer(capacity=5)
        for i in range(3):
            buf.add(f"s{i}", f"a{i}", float(i), f"s{i+1}", 0.1)

        samples = buf.sample(batch_size=3)
        assert len(samples) == 3


class TestPrioritizedSampling:
    """Test prioritized sampling: entries with high |td_error| get sampled more often."""

    def test_high_td_error_sampled_more_often(self) -> None:
        """Statistical test: high priority entries appear more frequently."""
        np.random.seed(42)
        buf = ReplayBuffer(capacity=100, priority_epsilon=0.01)

        # Add one high-priority entry and many low-priority entries
        buf.add("high", "a_high", 10.0, "high_next", 100.0)  # |td_error| = 100
        for i in range(9):
            buf.add(f"low{i}", f"a_low{i}", 0.0, f"low_next{i}", 0.001)  # |td_error| ~ 0

        # Sample many times and count how often "high" appears
        high_count = 0
        n_trials = 1000
        for _ in range(n_trials):
            samples = buf.sample(batch_size=1)
            if samples[0][0] == "high":
                high_count += 1

        # With |td_error| = 100 vs 0.001, "high" should be sampled much more
        # than uniform (uniform would give ~10% = 100 out of 1000).
        # With priorities: high gets 100.01, each low gets ~0.011
        # high fraction ≈ 100.01 / (100.01 + 9*0.011) ≈ 0.999
        assert high_count > 500, f"Expected high_count > 500, got {high_count}"

    def test_all_entries_have_nonzero_sampling_probability(self) -> None:
        """Even low-priority entries can be sampled due to epsilon."""
        np.random.seed(123)
        buf = ReplayBuffer(capacity=10, priority_epsilon=0.01)

        # Use moderate td_error difference so low-priority entry still gets sampled
        buf.add("s0", "a0", 0.0, "ns0", 0.0)  # priority = 0.01
        buf.add("s1", "a1", 1.0, "ns1", 0.5)  # priority = 0.51

        # Over many samples, both entries should appear at least once
        # s0 has probability 0.01 / 0.52 ≈ 1.9%, so 2000 trials should suffice
        seen_states: set[str] = set()
        for _ in range(2000):
            samples = buf.sample(batch_size=1)
            seen_states.add(samples[0][0])

        assert "s0" in seen_states, "Low-priority entry should still be sampled"
        assert "s1" in seen_states, "High-priority entry should be sampled"


class TestEmptyBufferSample:
    """Test empty buffer sample() returns empty list."""

    def test_sample_empty_buffer_returns_empty_list(self) -> None:
        buf = ReplayBuffer(capacity=10)
        result = buf.sample(batch_size=5)
        assert result == []

    def test_sample_empty_buffer_with_zero_batch_size(self) -> None:
        buf = ReplayBuffer(capacity=10)
        result = buf.sample(batch_size=0)
        assert result == []
