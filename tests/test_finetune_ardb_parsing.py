"""Tests for the shared ARDB fine-tune parsing helpers (JSON repair + markdown-table
grid parsing) — no subprocess, no model, no network. Both `gemma_ardb_engine.py` and
`qwen_ardb_engine.py` build on this module; it is tested in isolation first."""
from __future__ import annotations

import json

from khmer_pipeline.engines.finetune_ardb.parsing import (
    parse_regions,
    markdown_table_to_grid,
)


# --- parse_regions: valid JSON ---

def test_parse_regions_valid_json():
    text = json.dumps([{"box_2d": [0, 0, 10, 10], "label": "Text", "text": "hi"}])
    assert parse_regions(text) == [{"box_2d": [0, 0, 10, 10], "label": "Text", "text": "hi"}]


def test_parse_regions_not_a_list_returns_none():
    assert parse_regions(json.dumps({"not": "a list"})) is None


def test_parse_regions_garbage_returns_none():
    assert parse_regions("not json at all") is None


# --- parse_regions: repair fallback (mirrors scripts/eval_finetune_real.py's
# _try_repair_json, kept in sync per the reconnaissance findings) ---

def test_parse_regions_repairs_missing_closing_bracket():
    # Missing only the outer list's closing ']' — a stray end-of-turn token follows.
    text = '[{"box_2d": [0,0,1,1], "label": "Text", "text": "a"}\n<|im_end|>'
    regions = parse_regions(text)
    assert regions == [{"box_2d": [0, 0, 1, 1], "label": "Text", "text": "a"}]


def test_parse_regions_repairs_missing_both_brackets():
    # Bare comma-separated objects, no wrapping list at all.
    text = '{"box_2d": [0,0,1,1], "label": "Text", "text": "a"},{"box_2d": [1,1,2,2], "label": "Text", "text": "b"}'
    regions = parse_regions(text)
    assert regions == [
        {"box_2d": [0, 0, 1, 1], "label": "Text", "text": "a"},
        {"box_2d": [1, 1, 2, 2], "label": "Text", "text": "b"},
    ]


def test_parse_regions_mid_field_truncation_not_guessed():
    # Cut off with no trailing '}' at all — nothing safe to repair, must stay None.
    text = '[{"box_2d": [0,0,1,1], "label": "Text", "tex'
    assert parse_regions(text) is None


# --- markdown_table_to_grid ---

def test_markdown_table_clean_grid():
    md = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
    grid = markdown_table_to_grid(md)
    assert grid == {
        (0, 0): "a", (0, 1): "b",
        (1, 0): "1", (1, 1): "2",
        (2, 0): "3", (2, 1): "4",
    }


def test_markdown_table_ragged_rows_pad_missing_cells():
    # A short row must not crash or silently drop columns — missing cells become "".
    md = "| a | b | c |\n|---|---|---|\n| 1 | 2 |"
    grid = markdown_table_to_grid(md)
    assert grid[(1, 0)] == "1"
    assert grid[(1, 1)] == "2"
    assert grid[(1, 2)] == ""


def test_markdown_table_empty_cells_are_empty_string():
    md = "| a | b |\n|---|---|\n|  | 2 |"
    grid = markdown_table_to_grid(md)
    assert grid[(1, 0)] == ""
    assert grid[(1, 1)] == "2"


def test_markdown_table_khmer_text_preserved():
    md = "| មុខទំនិញ | ថ្លៃ |\n|---|---|\n| សាច់គោ | ១២០០០ |"
    grid = markdown_table_to_grid(md)
    assert grid[(1, 0)] == "សាច់គោ"
    assert grid[(1, 1)] == "១២០០០"


def test_markdown_table_not_a_table_returns_empty_grid():
    assert markdown_table_to_grid("just plain prose, no pipes here") == {}


def test_markdown_table_empty_string_returns_empty_grid():
    assert markdown_table_to_grid("") == {}
