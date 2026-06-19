from __future__ import annotations
import unicodedata
import pytest

from khmer_pipeline.evaluate_structure import (
    _norm,
    _strip_title_row,
    _levenshtein,
    cer,
    gt_table_grid,
    gt_paragraph_lines,
    pred_table_grid,
    evaluate_table,
    evaluate_text,
)

# --- _norm ---

def test_norm_nfc_khmer():
    # Khmer character decomposed via NFD should equal NFC form
    s = "ក"
    nfd = unicodedata.normalize("NFD", s)
    assert _norm(nfd) == _norm(s)

def test_norm_whitespace_collapse():
    assert _norm("a  b\t\nc") == "a b c"

def test_norm_strip():
    assert _norm("  hello  ") == "hello"

def test_norm_empty():
    assert _norm("") == ""

# --- _strip_title_row ---

def test_strip_title_row_strips():
    grid = [["ចំណងជើង", "", ""], ["ក", "ខ", "គ"], ["d", "e", "f"]]
    result = _strip_title_row(grid)
    assert result == [["ក", "ខ", "គ"], ["d", "e", "f"]]

def test_strip_title_row_leaves_normal():
    grid = [["ក", "ខ", "គ"], ["d", "e", "f"]]
    result = _strip_title_row(grid)
    assert result == [["ក", "ខ", "គ"], ["d", "e", "f"]]

def test_strip_title_row_empty_grid():
    assert _strip_title_row([]) == []

def test_strip_title_row_not_stripped_when_rest_nonempty():
    # row[0][1] is non-empty so it's not a title row
    grid = [["Title", "Other", ""], ["a", "b", "c"]]
    result = _strip_title_row(grid)
    assert result == grid

def test_strip_title_row_single_row_with_title():
    # single row that looks like a title should still be stripped
    grid = [["ចំណងជើង", "", ""]]
    result = _strip_title_row(grid)
    assert result == []

# --- _levenshtein / cer ---

def test_levenshtein_identical():
    assert _levenshtein("abc", "abc") == 0

def test_levenshtein_empty():
    assert _levenshtein("", "") == 0

def test_levenshtein_insert():
    assert _levenshtein("", "a") == 1

def test_levenshtein_delete():
    assert _levenshtein("a", "") == 1

def test_levenshtein_substitute():
    assert _levenshtein("a", "b") == 1

def test_levenshtein_known():
    # kitten -> sitting: 3 edits
    assert _levenshtein("kitten", "sitting") == 3

def test_cer_identical():
    assert cer("abc", "abc") == 0.0

def test_cer_known():
    # 3 edits over 6 chars = 0.5
    assert cer("kitten", "sitting") == pytest.approx(3 / 6)

def test_cer_empty_ref_empty_hyp():
    assert cer("", "") == 0.0

def test_cer_empty_ref_nonempty_hyp():
    assert cer("", "something") == 1.0

def test_cer_khmer():
    ref = "ក"
    hyp = "ក"
    assert cer(ref, hyp) == 0.0

# --- gt_table_grid ---

ISOLATED_GT = {
    "font_family": "Khmer OS",
    "table_index": 0,
    "template": "ទំហំ",
    "data": [
        ["ចំណងជើង", "", ""],
        ["ក", "ខ", "គ"],
        ["1", "2", "3"],
    ],
}

DOCS_GT = {
    "font_family": "Khmer OS",
    "template": "report",
    "document_type": "financial",
    "paragraphs": ["line one", "line two"],
    "tables": [{"data": [["ក", "ខ"], ["1", "2"]]}],
    "footer": "footer text",
}

def test_gt_table_grid_isolated():
    grid = gt_table_grid(ISOLATED_GT)
    assert grid == ISOLATED_GT["data"]

def test_gt_table_grid_docs():
    grid = gt_table_grid(DOCS_GT)
    assert grid == [["ក", "ខ"], ["1", "2"]]

def test_gt_table_grid_neither():
    assert gt_table_grid({"font_family": "x"}) is None

# --- gt_paragraph_lines ---

def test_gt_paragraph_lines_docs():
    lines = gt_paragraph_lines(DOCS_GT)
    assert "line one" in lines
    assert "line two" in lines
    assert "footer text" in lines

def test_gt_paragraph_lines_docs_multiline():
    gt = {
        "paragraphs": ["line A\nline B"],
        "tables": [{"data": []}],
        "footer": "foot\nfoot2",
    }
    lines = gt_paragraph_lines(gt)
    assert "line A" in lines
    assert "line B" in lines
    assert "foot" in lines
    assert "foot2" in lines

def test_gt_paragraph_lines_isolated():
    assert gt_paragraph_lines(ISOLATED_GT) == []

def test_gt_paragraph_lines_no_footer():
    gt = {"paragraphs": ["hello"], "tables": [{"data": []}]}
    lines = gt_paragraph_lines(gt)
    assert lines == ["hello"]

# --- pred_table_grid ---

def _make_cell(row_id, col_id, text):
    return {"row_id": row_id, "col_id": col_id, "text_lines": [{"text": text}]}

def test_pred_table_grid_basic():
    table = {
        "cells": [
            _make_cell(0, 0, "ក"),
            _make_cell(0, 1, "ខ"),
            _make_cell(1, 0, "1"),
            _make_cell(1, 1, "2"),
        ]
    }
    grid = pred_table_grid(table)
    assert grid == [["ក", "ខ"], ["1", "2"]]

def test_pred_table_grid_missing_cell():
    # col 1 in row 0 is missing — should be ""
    table = {
        "cells": [
            _make_cell(0, 0, "ក"),
            _make_cell(1, 0, "1"),
            _make_cell(1, 1, "2"),
        ]
    }
    grid = pred_table_grid(table)
    assert grid[0] == ["ក", ""]
    assert grid[1] == ["1", "2"]

def test_pred_table_grid_empty():
    grid = pred_table_grid({"cells": []})
    assert grid == []

def test_pred_table_grid_multiple_text_lines():
    table = {
        "cells": [
            {"row_id": 0, "col_id": 0, "text_lines": [{"text": "hello"}, {"text": "world"}]},
        ]
    }
    grid = pred_table_grid(table)
    assert grid[0][0] == "hello world"

# --- evaluate_table ---

def _make_table_from_grid(grid):
    cells = []
    for r, row in enumerate(grid):
        for c, text in enumerate(row):
            cells.append(_make_cell(r, c, text))
    return {"cells": cells}

def test_evaluate_table_exact_match():
    gt_grid = [["ក", "ខ"], ["1", "2"]]
    pred_table = _make_table_from_grid(gt_grid)
    result = evaluate_table([pred_table], gt_grid)
    assert result["cell_accuracy"] == pytest.approx(1.0)
    assert result["table_cer"] == pytest.approx(0.0)
    assert result["tables_found"] == 1
    assert result["cells_total"] == 4
    assert result["cells_correct"] == 4

def test_evaluate_table_no_pred_tables():
    gt_grid = [["ក", "ខ"], ["1", "2"]]
    result = evaluate_table([], gt_grid)
    assert result["cell_accuracy"] == pytest.approx(0.0)
    assert result["tables_found"] == 0
    assert result["cells_total"] == 4

def test_evaluate_table_extra_leading_row_shifts():
    # pred has an extra non-title row prepended — causes a positional shift
    # because it has content in both cells it won't be stripped as a title row
    gt_grid = [["ក", "ខ"], ["1", "2"]]
    pred_grid = [["extra", "row"], ["ក", "ខ"], ["1", "2"]]
    pred_table = _make_table_from_grid(pred_grid)
    result = evaluate_table([pred_table], gt_grid)
    # positional accuracy should be low (shifted: "extra"!="ក", "row"!="ខ", "ក"!="1", "ខ"!="2")
    assert result["cell_accuracy"] == pytest.approx(0.0)
    # content recall should be high (all gt values are in pred somewhere)
    assert result["cell_content_recall"] == pytest.approx(1.0)

def test_evaluate_table_none_gt_grid():
    result = evaluate_table([], None)
    assert result["cell_accuracy"] == 0.0
    assert result["gt_rows"] == 0
    assert result["gt_cols"] == 0

def test_evaluate_table_dim_mismatch():
    gt_grid = [["ក", "ខ", "គ"], ["1", "2", "3"]]
    pred_grid = [["ក", "ខ"], ["1", "2"]]
    pred_table = _make_table_from_grid(pred_grid)
    result = evaluate_table([pred_table], gt_grid)
    assert result["gt_cols"] == 3
    assert result["pred_cols"] == 2
    # col 2 always mismatches
    assert result["cell_accuracy"] < 1.0

def test_evaluate_table_title_strip_applied():
    # GT has a title row (isolated schema), pred also has it — after stripping both, should match
    gt_grid = [["ចំណងជើង", "", ""], ["ក", "ខ", "គ"]]
    pred_grid = [["ចំណងជើង", "", ""], ["ក", "ខ", "គ"]]
    pred_table = _make_table_from_grid(pred_grid)
    result = evaluate_table([pred_table], gt_grid)
    assert result["cell_accuracy"] == pytest.approx(1.0)

# --- evaluate_text ---

def test_evaluate_text_isolated_returns_none():
    result = evaluate_text("some text", [], ISOLATED_GT)
    assert result["text_cer"] is None
    assert result["paragraph_recall"] is None
    assert result["paragraph_leak"] is None

def test_evaluate_text_recall_and_cer():
    gt = {
        "paragraphs": ["hello world"],
        "tables": [{"data": []}],
        "footer": "",
    }
    ocr_text = "hello world"
    result = evaluate_text(ocr_text, [], gt)
    assert result["paragraph_recall"] == pytest.approx(1.0)
    assert result["text_cer"] == pytest.approx(0.0)

def test_evaluate_text_partial_recall():
    gt = {
        "paragraphs": ["hello world", "missing line"],
        "tables": [{"data": []}],
        "footer": "",
    }
    ocr_text = "hello world"
    result = evaluate_text(ocr_text, [], gt)
    assert result["paragraph_recall"] == pytest.approx(0.5)

def test_evaluate_text_paragraph_leak():
    # A paragraph line appears inside a predicted table cell
    gt = {
        "paragraphs": ["this is body text"],
        "tables": [{"data": []}],
        "footer": "",
    }
    leak_cell = {"row_id": 0, "col_id": 0, "text_lines": [{"text": "this is body text"}]}
    pred_tables = [{"cells": [leak_cell]}]
    result = evaluate_text("", pred_tables, gt)
    assert result["paragraph_leak"] == 1

def test_evaluate_text_no_leak():
    gt = {
        "paragraphs": ["body text"],
        "tables": [{"data": []}],
        "footer": "",
    }
    # table cell has different text
    cell = {"row_id": 0, "col_id": 0, "text_lines": [{"text": "unrelated content"}]}
    pred_tables = [{"cells": [cell]}]
    result = evaluate_text("body text", pred_tables, gt)
    assert result["paragraph_leak"] == 0
