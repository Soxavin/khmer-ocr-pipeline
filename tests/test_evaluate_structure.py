from __future__ import annotations
import unicodedata
import pytest

from khmer_pipeline.evaluate_structure import (
    _norm,
    _strip_title_row,
    _align_rows,
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

def test_gt_table_grid_empty_tables_list():
    # harvested GT has tables=[] — must return None, not IndexError
    assert gt_table_grid({"tables": []}) is None

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
    # pred has an extra non-title row prepended; _strip_title_row won't remove it
    # because it has content in both cells. Row alignment now recovers accuracy.
    gt_grid = [["ក", "ខ"], ["1", "2"]]
    pred_grid = [["extra", "row"], ["ក", "ខ"], ["1", "2"]]
    pred_table = _make_table_from_grid(pred_grid)
    result = evaluate_table([pred_table], gt_grid)
    # row alignment maps GT rows to the matching pred rows → accuracy == 1.0
    assert result["cell_accuracy"] == pytest.approx(1.0)
    # content recall should also be high
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

# --- _align_rows ---

def test_align_rows_identical():
    sigs = [("a",), ("b",), ("c",)]
    pairs = _align_rows(sigs, sigs)
    assert pairs == [(0, 0), (1, 1), (2, 2)]

def test_align_rows_extra_leading_pred_row():
    # gt=[A,B,C], pred=[X,A,B,C] — X is extra; GT rows map to later pred rows
    A, B, C, X = ("a",), ("b",), ("c",), ("x",)
    pairs = _align_rows([A, B, C], [X, A, B, C])
    # A→1, B→2, C→3 (X at index 0 is unmatched insert)
    assert (0, 1) in pairs
    assert (1, 2) in pairs
    assert (2, 3) in pairs
    assert len(pairs) == 3

def test_align_rows_deleted_gt_row():
    # gt=[A,B,C], pred=[A,C] — B is deleted (no pair for GT index 1)
    A, B, C = ("a",), ("b",), ("c",)
    pairs = _align_rows([A, B, C], [A, C])
    gt_indices = [gi for gi, _ in pairs]
    assert 1 not in gt_indices  # B (GT index 1) is unmatched
    assert (0, 0) in pairs
    assert (2, 1) in pairs

# --- evaluate_table alignment cases ---

def test_evaluate_table_extra_leading_title_row_accuracy_one():
    # pred identical to GT but with an extra title-like first row that isn't stripped
    gt_grid = [["ក", "ខ", "គ"], ["1", "2", "3"], ["4", "5", "6"]]
    pred_grid = [["TITLE", "TITLE", "TITLE"], ["ក", "ខ", "គ"], ["1", "2", "3"], ["4", "5", "6"]]
    pred_table = _make_table_from_grid(pred_grid)
    result = evaluate_table([pred_table], gt_grid)
    assert result["cell_accuracy"] == pytest.approx(1.0)

def test_evaluate_table_hallucinated_middle_row():
    # pred has an extra row inserted in the middle; surrounding rows still align
    gt_grid = [["ក", "ខ"], ["1", "2"], ["3", "4"]]
    pred_grid = [["ក", "ខ"], ["HALLUC", "ROW"], ["1", "2"], ["3", "4"]]
    pred_table = _make_table_from_grid(pred_grid)
    result = evaluate_table([pred_table], gt_grid)
    # rows ក/ខ and 3/4 align perfectly; row 1/2 also aligns → all GT rows covered
    assert result["cell_accuracy"] == pytest.approx(1.0)

def test_evaluate_table_missing_middle_row():
    # pred is missing the middle GT row → those GT cells count as misses
    gt_grid = [["ក", "ខ"], ["1", "2"], ["3", "4"]]
    pred_grid = [["ក", "ខ"], ["3", "4"]]
    pred_table = _make_table_from_grid(pred_grid)
    result = evaluate_table([pred_table], gt_grid)
    # GT row ["1","2"] has no paired pred row → 2 misses out of 6 total cells
    assert result["cell_accuracy"] < 1.0
    # content recall may stay high since "1","2" are not in pred at all here
    assert result["cells_total"] == 6

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
