"""Unit tests for webapp.edits — bulk find/replace across table cells."""
from webapp import edits


def test_replace_in_grid_counts_all_occurrences():
    grid = [["a៛b", "៛"], ["x", "y៛z៛"]]
    new, count = edits.replace_in_grid(grid, "៛", "R")
    assert count == 4
    assert new == [["aRb", "R"], ["x", "yRzR"]]


def test_replace_empty_find_is_noop():
    grid = [["a"]]
    new, count = edits.replace_in_grid(grid, "", "x")
    assert count == 0 and new is grid


def test_replace_across_returns_only_changed_tables():
    tabs = [("t1", [["foo", "bar"]]), ("t2", [["baz"]])]
    changed, total = edits.replace_across(tabs, "foo", "FOO")
    assert total == 1
    assert set(changed) == {"t1"}
    assert changed["t1"] == [["FOO", "bar"]]


def test_replace_across_no_match():
    changed, total = edits.replace_across([("t1", [["a"]])], "zzz", "x")
    assert total == 0 and changed == {}
