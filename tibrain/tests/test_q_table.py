"""Unit tests for tibrain.q_table.QTable."""

import pytest

from tibrain.q_table import QTable


# --- Requirement 5.2: Default value for unvisited pairs ---


class TestGetDefault:
    def test_returns_zero_for_unvisited_state_action(self):
        table = QTable()
        assert table.get("s1", "a1") == 0.0

    def test_returns_zero_for_unvisited_action_in_known_state(self):
        table = QTable()
        table.set("s1", "a1", 1.5)
        assert table.get("s1", "a2") == 0.0

    def test_returns_zero_for_unknown_state(self):
        table = QTable()
        table.set("s1", "a1", 1.0)
        assert table.get("s999", "a1") == 0.0


# --- Requirements 5.1, 5.3: get/set round-trip and get_best_action ---


class TestSetAndGet:
    def test_set_then_get_returns_value(self):
        table = QTable()
        table.set("s1", "a1", 3.14)
        assert table.get("s1", "a1") == 3.14

    def test_set_negative_value(self):
        table = QTable()
        table.set("s1", "a1", -2.5)
        assert table.get("s1", "a1") == -2.5

    def test_overwrite_value(self):
        table = QTable()
        table.set("s1", "a1", 1.0)
        table.set("s1", "a1", 9.9)
        assert table.get("s1", "a1") == 9.9

    def test_multiple_actions_same_state(self):
        table = QTable()
        table.set("s1", "a1", 1.0)
        table.set("s1", "a2", 2.0)
        table.set("s1", "a3", 3.0)
        assert table.get("s1", "a1") == 1.0
        assert table.get("s1", "a2") == 2.0
        assert table.get("s1", "a3") == 3.0

    def test_multiple_states_same_action(self):
        table = QTable()
        table.set("s1", "a1", 10.0)
        table.set("s2", "a1", 20.0)
        assert table.get("s1", "a1") == 10.0
        assert table.get("s2", "a1") == 20.0


class TestGetBestAction:
    def test_returns_highest_value_action(self):
        table = QTable()
        table.set("s1", "a1", 1.0)
        table.set("s1", "a2", 5.0)
        table.set("s1", "a3", 3.0)
        assert table.get_best_action("s1", ["a1", "a2", "a3"]) == "a2"

    def test_returns_first_action_on_tie(self):
        table = QTable()
        table.set("s1", "a1", 2.0)
        table.set("s1", "a2", 2.0)
        # When tied, returns first in list (no strict > means first stays best)
        assert table.get_best_action("s1", ["a1", "a2"]) == "a1"

    def test_unvisited_actions_all_zero(self):
        table = QTable()
        # All unvisited → all 0.0, should return first action
        result = table.get_best_action("s1", ["a1", "a2", "a3"])
        assert result == "a1"

    def test_single_action(self):
        table = QTable()
        table.set("s1", "a1", 5.0)
        assert table.get_best_action("s1", ["a1"]) == "a1"

    def test_best_among_subset(self):
        table = QTable()
        table.set("s1", "a1", 10.0)
        table.set("s1", "a2", 5.0)
        table.set("s1", "a3", 8.0)
        # Only considering a2, a3 → a3 wins
        assert table.get_best_action("s1", ["a2", "a3"]) == "a3"


# --- Requirement 5.4: size property ---


class TestSize:
    def test_empty_table_size_is_zero(self):
        table = QTable()
        assert table.size == 0

    def test_single_entry(self):
        table = QTable()
        table.set("s1", "a1", 1.0)
        assert table.size == 1

    def test_multiple_entries(self):
        table = QTable()
        table.set("s1", "a1", 1.0)
        table.set("s1", "a2", 2.0)
        table.set("s2", "a1", 3.0)
        assert table.size == 3

    def test_overwrite_does_not_increase_size(self):
        table = QTable()
        table.set("s1", "a1", 1.0)
        table.set("s1", "a1", 2.0)
        assert table.size == 1


# --- Requirement 5.5: Serialization round-trip ---


class TestSerialization:
    def test_to_dict_empty(self):
        table = QTable()
        assert table.to_dict() == {}

    def test_to_dict_with_entries(self):
        table = QTable()
        table.set("s1", "a1", 1.0)
        table.set("s1", "a2", 2.0)
        table.set("s2", "a3", 3.0)
        expected = {"s1": {"a1": 1.0, "a2": 2.0}, "s2": {"a3": 3.0}}
        assert table.to_dict() == expected

    def test_from_dict_creates_equivalent_table(self):
        data = {"s1": {"a1": 1.5, "a2": -0.5}, "s2": {"a1": 3.0}}
        table = QTable.from_dict(data)
        assert table.get("s1", "a1") == 1.5
        assert table.get("s1", "a2") == -0.5
        assert table.get("s2", "a1") == 3.0
        assert table.size == 3

    def test_round_trip(self):
        table = QTable()
        table.set("state_x", "action_a", 42.0)
        table.set("state_x", "action_b", -1.0)
        table.set("state_y", "action_c", 0.001)

        serialized = table.to_dict()
        restored = QTable.from_dict(serialized)

        assert restored.get("state_x", "action_a") == 42.0
        assert restored.get("state_x", "action_b") == -1.0
        assert restored.get("state_y", "action_c") == 0.001
        assert restored.size == table.size

    def test_from_dict_empty(self):
        table = QTable.from_dict({})
        assert table.size == 0
        assert table.get("any", "thing") == 0.0


# --- Edge cases ---


class TestEdgeCases:
    def test_empty_table_get_best_action_returns_first(self):
        table = QTable()
        assert table.get_best_action("s1", ["x", "y", "z"]) == "x"

    def test_duplicate_set_overwrites(self):
        table = QTable()
        table.set("s", "a", 1.0)
        table.set("s", "a", 99.0)
        assert table.get("s", "a") == 99.0
        assert table.size == 1

    def test_zero_value_is_stored(self):
        table = QTable()
        table.set("s1", "a1", 0.0)
        # Size should still increase — explicitly stored 0.0
        assert table.size == 1
        assert table.get("s1", "a1") == 0.0

    def test_large_number_of_entries(self):
        table = QTable()
        for i in range(100):
            table.set(f"s{i}", f"a{i}", float(i))
        assert table.size == 100
        assert table.get("s50", "a50") == 50.0
