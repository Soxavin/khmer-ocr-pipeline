from __future__ import annotations
import unicodedata
import pytest

from khmer_pipeline.evaluation.evaluate_structure import (
    _norm,
    _strip_title_row,
    _align_rows,
    _levenshtein,
    _fold_numeric,
    _is_numeric,
    _has_khmer_digit,
    cer,
    gt_table_grid,
    gt_paragraph_lines,
    pred_table_grid,
    evaluate_table,
    evaluate_text,
    evaluate_document,
    evaluate_recognition,
    pool_gt_text,
    pool_gt_recognition_text,
    pool_pred_text,
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

def test_evaluate_table_combines_fragmented_tables():
    # real docs: Surya fragments one logical table into many regions;
    # evaluate_table must score the union of all detected tables, not just [0].
    gt_grid = [["ក", "ខ"], ["1", "2"], ["3", "4"]]
    t1 = _make_table_from_grid([["ក", "ខ"]])
    t2 = _make_table_from_grid([["1", "2"], ["3", "4"]])
    result = evaluate_table([t1, t2], gt_grid)
    assert result["cell_accuracy"] == pytest.approx(1.0)
    assert result["table_cer"] == pytest.approx(0.0)
    assert result["tables_found"] == 2  # fragmentation signal preserved
    assert result["cells_total"] == 6
    assert result["cells_correct"] == 6

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

def test_evaluate_table_none_gt_grid_still_reports_detection():
    # real docs labelled paragraphs-only have no GT grid, but we still want to
    # know how many tables the OCR actually detected
    pred_table = _make_table_from_grid([["ក", "ខ"], ["1", "2"]])
    result = evaluate_table([pred_table], None)
    assert result["tables_found"] == 1
    assert result["cell_accuracy"] == 0.0  # no GT grid → no cell scoring

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

def test_align_rows_extra_leading_row_with_garbled_rows():
    # The real-document case (§2.42): OCR garbles every row, so NO row is an
    # exact match, and a leading title row is detected on top. Exact-match
    # opcodes see one big "replace" and pair positionally — shifting every row
    # by one and collapsing position-sensitive metrics. Alignment must key on
    # row *similarity*, not equality.
    gt = [("29,199.60", "31.16%"), ("28,836.48", "31.15%"), ("25,759.06", "31.86%")]
    pred = [("", "ម"),                      # garbled title row, extra
            ("29,199.6", "31.16%"),              # ~gt[0]
            ("28.83648", "31.15%"),              # ~gt[1]
            ("25,752.0%", "31.86%")]             # ~gt[2]
    pairs = _align_rows(gt, pred)
    assert (0, 1) in pairs
    assert (1, 2) in pairs
    assert (2, 3) in pairs

def test_align_rows_is_monotonic():
    # Alignment must never cross (a later GT row pairing to an earlier pred row),
    # otherwise rows could be scored against the wrong table region.
    gt = [("alpha", "1"), ("beta", "2"), ("gamma", "3")]
    pred = [("beta", "2"), ("alpha", "1"), ("gamma", "3")]
    pairs = _align_rows(gt, pred)
    assert pairs == sorted(pairs)
    assert [pj for _, pj in pairs] == sorted(pj for _, pj in pairs)

def test_align_rows_dissimilar_rows_unmatched():
    # A pred row bearing no resemblance to the GT row must NOT be paired — that
    # would score unrelated content as if it were the analyst's row.
    gt = [("alpha", "1"), ("beta", "2")]
    pred = [("alpha", "1"), ("zzzzzzzz", "999999")]
    pairs = _align_rows(gt, pred)
    assert (0, 0) in pairs
    assert 1 not in [gi for gi, _ in pairs]

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

# --- pool_gt_text ---

def test_pool_gt_text_paragraphs_and_footer():
    gt = {
        "paragraphs": ["para one", "para two"],
        "tables": [{"data": [["cell A", "cell B"]]}],
        "footer": "foot",
    }
    result = pool_gt_text(gt)
    lines = result.split("\n")
    assert "para one" in lines
    assert "para two" in lines
    assert "cell A" in lines
    assert "cell B" in lines
    assert "foot" in lines

def test_pool_gt_text_isolated_data_schema():
    gt = {"data": [["ក", "ខ"], ["1", "2"]]}
    result = pool_gt_text(gt)
    lines = result.split("\n")
    assert "ក" in lines
    assert "ខ" in lines
    assert "1" in lines
    assert "2" in lines

def test_pool_gt_text_empty_gt():
    assert pool_gt_text({}) == ""

def test_pool_gt_text_missing_keys():
    # missing paragraphs/tables/footer should not raise
    gt = {"paragraphs": ["hello"]}
    result = pool_gt_text(gt)
    assert "hello" in result

def test_pool_gt_text_empty_footer_omitted():
    gt = {"paragraphs": ["text"], "tables": [], "footer": ""}
    result = pool_gt_text(gt)
    assert result == "text"

def test_pool_gt_text_skips_empty_cells():
    gt = {"data": [["", "hello", ""]]}
    result = pool_gt_text(gt)
    assert result == "hello"

# --- pool_pred_text ---

def test_pool_pred_text_ocr_and_cells():
    cell = {"row_id": 0, "col_id": 0, "text_lines": [{"text": "cell content"}]}
    result = pool_pred_text("ocr text", [{"cells": [cell]}])
    lines = result.split("\n")
    assert "ocr text" in lines
    assert "cell content" in lines

def test_pool_pred_text_empty_ocr_omitted():
    cell = {"row_id": 0, "col_id": 0, "text_lines": [{"text": "cell only"}]}
    result = pool_pred_text("", [{"cells": [cell]}])
    assert result == "cell only"
    # empty string not included as a blank line
    assert not result.startswith("\n")

def test_pool_pred_text_no_tables():
    result = pool_pred_text("just ocr", [])
    assert result == "just ocr"

def test_pool_pred_text_empty_cells_skipped():
    cell = {"row_id": 0, "col_id": 0, "text_lines": [{"text": ""}]}
    result = pool_pred_text("ocr", [{"cells": [cell]}])
    assert result == "ocr"

# --- evaluate_document ---

def test_evaluate_document_identical():
    # (a) gt paragraphs == ocr_text → document_cer == 0.0
    gt = {
        "paragraphs": ["hello world"],
        "tables": [],
        "footer": "",
    }
    result = evaluate_document("hello world", [], gt)
    assert result["document_cer"] == pytest.approx(0.0)

def test_evaluate_document_placement_agnostic():
    # (b) GT content is in paragraphs; pred side has same content as table cells
    # with ocr_text="" — document_cer should be near 0
    gt = {
        "paragraphs": ["hello world"],
        "tables": [],
        "footer": "",
    }
    cell = {"row_id": 0, "col_id": 0, "text_lines": [{"text": "hello world"}]}
    pred_tables = [{"cells": [cell]}]
    result = evaluate_document("", pred_tables, gt)
    assert result["document_cer"] == pytest.approx(0.0)

def test_evaluate_document_wrong_prediction():
    # (c) completely wrong prediction → high CER (≈ 1.0)
    gt = {
        "paragraphs": ["hello world"],
        "tables": [],
        "footer": "",
    }
    result = evaluate_document("zzzzzzzzzzz", [], gt)
    assert result["document_cer"] > 0.5

def test_evaluate_document_empty_tables_and_missing_keys():
    # (d) tables: [] / missing keys → no error
    gt = {"paragraphs": ["some text"], "tables": []}
    result = evaluate_document("some text", [], gt)
    assert "document_cer" in result
    assert result["document_cer"] == pytest.approx(0.0)

def test_evaluate_document_isolated_schema():
    # isolated table GT (has "data", no "paragraphs"/"tables") → no error
    gt = {"data": [["ក", "ខ"], ["1", "2"]]}
    result = evaluate_document("", [], gt)
    assert "document_cer" in result

def test_evaluate_document_pools_gt_table_cells():
    # GT has content in both paragraphs and table cells; all pooled for comparison
    gt = {
        "paragraphs": ["intro text"],
        "tables": [{"data": [["cell one", "cell two"]]}],
        "footer": "foot",
    }
    # pred pools same text in ocr_text
    result = evaluate_document("intro text cell one cell two foot", [], gt)
    assert result["document_cer"] == pytest.approx(0.0)

# --- pool_gt_recognition_text (single-source, no double-count) ---

def test_pool_gt_recognition_text_table_page_pools_cells_once():
    # ARDB-style GT: paragraphs restate the table rows. pool_gt_text double-counts;
    # the recognition pooler must use the table cells ONCE and ignore the paragraphs.
    gt = {
        "paragraphs": ["cell one cell two"],
        "tables": [{"data": [["cell one", "cell two"]]}],
        "footer": "",
    }
    result = pool_gt_recognition_text(gt)
    assert result.count("cell one") == 1
    assert result.count("cell two") == 1

def test_pool_gt_recognition_text_text_page_uses_paragraphs():
    # no table grid → fall back to paragraphs + footer
    gt = {"paragraphs": ["hello world"], "tables": [], "footer": "foot"}
    result = pool_gt_recognition_text(gt)
    lines = result.split("\n")
    assert "hello world" in lines
    assert "foot" in lines

def test_pool_gt_recognition_text_isolated_data_schema():
    gt = {"data": [["ក", "ខ"], ["1", "2"]]}
    result = pool_gt_recognition_text(gt)
    lines = result.split("\n")
    assert "ក" in lines and "ខ" in lines and "1" in lines and "2" in lines

def test_pool_gt_recognition_text_skips_empty_cells():
    gt = {"data": [["", "hello", ""]]}
    assert pool_gt_recognition_text(gt) == "hello"

# --- evaluate_recognition ---

def test_evaluate_recognition_table_page_no_double_count():
    # perfect prediction on a table page scores ~0 (would be ~0.5 under evaluate_document)
    gt = {
        "paragraphs": ["cell one cell two"],
        "tables": [{"data": [["cell one", "cell two"]]}],
        "footer": "",
    }
    assert evaluate_recognition("cell one cell two", [], gt)["recognition_cer"] == pytest.approx(0.0)

def test_evaluate_recognition_flat_external_pred():
    # a flat external model passes its text as ocr_text with pred_tables=[]
    gt = {"paragraphs": ["hello world"], "tables": [], "footer": ""}
    assert evaluate_recognition("hello world", [], gt)["recognition_cer"] == pytest.approx(0.0)

def test_evaluate_recognition_wrong_prediction():
    gt = {"paragraphs": ["hello world"], "tables": [], "footer": ""}
    assert evaluate_recognition("zzzzzzzzzzz", [], gt)["recognition_cer"] > 0.5

# --- numeric-cell detection (_fold_numeric / _is_numeric / _has_khmer_digit) ---

def test_fold_numeric_folds_khmer_digits():
    assert _fold_numeric("១២៣") == "123"

def test_fold_numeric_strips_internal_spaces():
    # a split number "7 800" folds to the joined form for comparison
    assert _fold_numeric("7 800") == "7800"
    assert _fold_numeric(" -3.85 % ") == "-3.85%"

def test_is_numeric_plain_integer():
    assert _is_numeric("12") is True

def test_is_numeric_thousands_comma():
    assert _is_numeric("7,800") is True
    assert _is_numeric("1,234,567.89") is True

def test_is_numeric_negative_percent():
    assert _is_numeric("-3.85%") is True

def test_is_numeric_positive_sign():
    assert _is_numeric("+5") is True

def test_is_numeric_khmer_digits():
    # Khmer numerals fold to Arabic → numeric
    assert _is_numeric("១២៣") is True
    assert _is_numeric("០.00%") is True
    assert _is_numeric("២៣") is True  # row-index column style

def test_is_numeric_decimal_and_percent():
    assert _is_numeric("12.5") is True
    assert _is_numeric("0.00%") is True

def test_is_numeric_space_separated_number():
    # internal space stripped before matching
    assert _is_numeric("7 800") is True

def test_is_numeric_riel_unit_not_numeric():
    # unit strings must NOT count as numeric cells
    assert _is_numeric("៛/គ.ក") is False
    assert _is_numeric("៛/គ្រាប់") is False

def test_is_numeric_rejects_non_numbers():
    assert _is_numeric("") is False
    assert _is_numeric("abc") is False
    assert _is_numeric("N/A") is False
    assert _is_numeric("-") is False
    assert _is_numeric("ចេក") is False

def test_is_numeric_rejects_digit_duplication_artifact():
    # "7,800" -> "7,8000" is the known Kiri duplication artifact; the malformed
    # thousands grouping means it is NOT a well-formed number. Bare ASCII digits
    # carry no locale signal, so the comma-decimal reading (under which "7,8000"
    # would be valid) stays disabled here — this guard must survive that widening.
    assert _is_numeric("7,8000") is False

# --- Cambodian financial number forms: unit affixes + comma decimal separator ---

def test_is_numeric_unit_suffix_khmer():
    # MoC gas bulletin money cells: Khmer digits, comma decimal, unit word
    assert _is_numeric("០,៧១១៧ ដុល្លារ") is True
    assert _is_numeric("០,២០ ដុល្លារ") is True
    assert _is_numeric("១,១៤ ដុល្លារ") is True

def test_is_numeric_unit_suffix_with_space_grouped_thousands():
    # "៤ ៦០០ រៀល" — space-grouped thousands AND a unit token
    assert _is_numeric("៤ ៦០០ រៀល") is True

def test_is_numeric_unit_suffix_latin():
    assert _is_numeric("1.14 USD") is True
    assert _is_numeric("12.5 kg") is True

def test_is_numeric_currency_symbol_prefix():
    # Unicode currency-symbol category (Sc): "$" and the Khmer riel sign "៛"
    assert _is_numeric("$1.14") is True
    assert _is_numeric("៛4,600") is True

def test_is_numeric_parenthesised_negative():
    # Accounting convention for negatives, common in budget/TOFE tables
    assert _is_numeric("(1,234)") is True
    assert _is_numeric("(3.85%)") is True

def test_is_numeric_comma_decimal_needs_locale_signal():
    # Khmer digits are a locale signal → comma reads as a decimal separator
    assert _is_numeric("០,៧១១៧") is True
    # A unit affix is also a locale signal
    assert _is_numeric("0,7117 ដុល្លារ") is True
    # Bare ASCII with no signal keeps the strict period-decimal grammar
    assert _is_numeric("0,7117") is False

def test_is_numeric_comma_decimal_with_period_thousands():
    assert _is_numeric("១.២៣៤,៥៦") is True

def test_is_numeric_rejects_alphabetic_prefix_label():
    # "Gasoline 92" is a column label, not a numeric cell: the leading token is
    # alphabetic, so it is not a strippable affix and the core fails to parse.
    assert _is_numeric("Gasoline 92") is False
    assert _is_numeric("Gasoil 10ppm") is False

def test_is_numeric_rejects_multiple_number_cores():
    # A Khmer sentence quoting several percentages stays a label cell.
    assert _is_numeric("៣០% មក ១៥%") is False
    # A merged cell holding two values must not be scored as one number.
    assert _is_numeric("០,០៧៨៥ ដុល្លារ ០,០០០០ ដុល្លារ") is False

def test_is_numeric_rejects_long_trailing_token():
    # The unit affix is one SHORT token; a sentence that happens to end in a word
    # after a number is not a numeric cell.
    assert _is_numeric("១ ថ្លៃប្រេងអន្តរជាតិជាមធ្យម") is False

def test_is_numeric_unit_affix_alone_is_not_numeric():
    # No number core at all — the existing riel-unit header case, restated for
    # the affix path (regression guard for the widened classifier).
    assert _is_numeric("៛/គ.ក") is False
    assert _is_numeric("ដុល្លារ") is False

def test_khmer_and_numeric_classes_stay_disjoint_for_unit_suffixed_money():
    # A unit-suffixed money cell is Khmer-heavy by character ratio, but it is a
    # NUMBER: it must move to the numeric class, not be counted twice.
    from khmer_pipeline.evaluation.evaluate_structure import _is_khmer_text
    assert _is_numeric("០,៧១១៧ ដុល្លារ") is True
    assert _is_khmer_text("០,៧១១៧ ដុល្លារ") is False
    # A genuine Khmer label is unaffected.
    assert _is_khmer_text("ថ្លៃប្រេងអន្តរជាតិជាមធ្យម") is True

def test_has_khmer_digit():
    assert _has_khmer_digit("១") is True
    assert _has_khmer_digit("៧,៨០០") is True
    assert _has_khmer_digit("7,800") is False
    assert _has_khmer_digit("៛/គ.ក") is False  # Riel sign is not a digit

# --- Numeric_Cell_Accuracy scoring in evaluate_table ---

# 4 numeric GT cells ("1", "7,800", "2", "-3.85%"); 2 non-numeric ("ចេក", "៛/គ.ក")
_NUMERIC_GT = [
    ["1", "ចេក", "7,800"],
    ["2", "៛/គ.ក", "-3.85%"],
]

def test_numeric_cells_exact_match():
    pred_table = _make_table_from_grid(_NUMERIC_GT)
    result = evaluate_table([pred_table], _NUMERIC_GT)
    assert result["numeric_cells_total"] == 4
    assert result["numeric_cells_correct"] == 4
    assert result["numeric_cell_accuracy"] == pytest.approx(1.0)
    assert result["numeric_cells_khmer_digit_slips"] == 0

def test_numeric_cells_khmer_digit_rendering_still_value_correct():
    # pred renders the SAME values in Khmer digits → value-correct after folding,
    # but flagged as khmer-digit slips; plain cell_accuracy would MISS these.
    pred_grid = [
        ["១", "ចេក", "៧,៨០០"],
        ["២", "៛/គ.ក", "-៣.៨៥%"],
    ]
    pred_table = _make_table_from_grid(pred_grid)
    result = evaluate_table([pred_table], _NUMERIC_GT)
    assert result["numeric_cells_total"] == 4
    assert result["numeric_cells_correct"] == 4
    assert result["numeric_cell_accuracy"] == pytest.approx(1.0)
    assert result["numeric_cells_khmer_digit_slips"] == 4
    # plain exact-match cell_accuracy does NOT credit the Khmer-digit numerics
    assert result["cell_accuracy"] < 1.0

def test_numeric_cells_wrong_value():
    pred_grid = [
        ["1", "ចេក", "7,900"],   # 7,900 != 7,800
        ["2", "៛/គ.ក", "-3.85%"],
    ]
    pred_table = _make_table_from_grid(pred_grid)
    result = evaluate_table([pred_table], _NUMERIC_GT)
    assert result["numeric_cells_total"] == 4
    assert result["numeric_cells_correct"] == 3
    assert result["numeric_cell_accuracy"] == pytest.approx(0.75)
    assert result["numeric_cells_khmer_digit_slips"] == 0

def test_numeric_cells_dropped_row_counts_against_denominator():
    # GT row ["2","៛/គ.ក","-3.85%"] has no pred pair → its 2 numeric cells miss
    pred_grid = [["1", "ចេក", "7,800"]]
    pred_table = _make_table_from_grid(pred_grid)
    result = evaluate_table([pred_table], _NUMERIC_GT)
    assert result["numeric_cells_total"] == 4       # all numeric GT cells
    assert result["numeric_cells_correct"] == 2      # only the aligned row's
    assert result["numeric_cell_accuracy"] == pytest.approx(0.5)

def test_numeric_cells_none_gt_grid():
    result = evaluate_table([], None)
    assert result["numeric_cells_total"] == 0
    assert result["numeric_cells_correct"] == 0
    assert result["numeric_cell_accuracy"] == 0.0
    assert result["numeric_cells_khmer_digit_slips"] == 0

def test_numeric_additions_do_not_perturb_existing_metrics():
    # PIN: adding the numeric metric must not change cell_accuracy / recall /
    # table_cer / cells_correct on a fixture that contains numeric cells.
    pred_table = _make_table_from_grid(_NUMERIC_GT)
    result = evaluate_table([pred_table], _NUMERIC_GT)
    assert result["cell_accuracy"] == pytest.approx(1.0)
    assert result["cell_content_recall"] == pytest.approx(1.0)
    assert result["table_cer"] == pytest.approx(0.0)
    assert result["cells_total"] == 6
    assert result["cells_correct"] == 6


# --- Khmer_Cell_Accuracy scoring in evaluate_table ---
# Mirrors the numeric metric for the other half of a financial table: numbers vs
# Khmer labels degrade independently per engine (Kiri is strong on Khmer text and
# weak on numerals), and a single aggregate accuracy hides which one moved.
# The two classes are deliberately DISJOINT — see the Khmer-digit test below.

# 2 Khmer-heavy GT cells ("ចេក", "៛/គ.ក"); the other 4 are numeric.
def test_khmer_cells_exact_match():
    pred_table = _make_table_from_grid(_NUMERIC_GT)
    result = evaluate_table([pred_table], _NUMERIC_GT)
    assert result["khmer_cells_total"] == 2
    assert result["khmer_cells_correct"] == 2
    assert result["khmer_cell_accuracy"] == pytest.approx(1.0)


def test_khmer_cells_wrong_text():
    # Swap the two Khmer labels between rows: a realistic misread, and it uses
    # only Khmer strings already present in this fixture.
    pred_grid = [
        ["1", "៛/គ.ក", "7,800"],
        ["2", "ចេក", "-3.85%"],
    ]
    pred_table = _make_table_from_grid(pred_grid)
    result = evaluate_table([pred_table], _NUMERIC_GT)
    assert result["khmer_cells_total"] == 2
    assert result["khmer_cells_correct"] == 0
    assert result["khmer_cell_accuracy"] == pytest.approx(0.0)
    # The numerals were untouched — the split is what proves it.
    assert result["numeric_cell_accuracy"] == pytest.approx(1.0)


def test_khmer_digit_numerals_count_as_numeric_not_khmer():
    # "១២៣" is Khmer script but semantically a number: it must land in the
    # numeric class only, so the two classes never double-count a cell.
    gt_grid = [["h1", "h2"], ["១២៣", "ចេក"]]
    pred_table = _make_table_from_grid(gt_grid)
    result = evaluate_table([pred_table], gt_grid)
    assert result["numeric_cells_total"] == 1
    assert result["khmer_cells_total"] == 1  # only "ចេក", not "១២៣"


def test_khmer_cells_dropped_row_counts_against_denominator():
    # GT row 2 has no pred pair → its Khmer cell misses, mirroring the numeric rule.
    pred_grid = [["1", "ចេក", "7,800"]]
    pred_table = _make_table_from_grid(pred_grid)
    result = evaluate_table([pred_table], _NUMERIC_GT)
    assert result["khmer_cells_total"] == 2
    assert result["khmer_cells_correct"] == 1
    assert result["khmer_cell_accuracy"] == pytest.approx(0.5)


def test_khmer_cells_none_gt_grid():
    result = evaluate_table([], None)
    assert result["khmer_cells_total"] == 0
    assert result["khmer_cells_correct"] == 0
    assert result["khmer_cell_accuracy"] == 0.0


def test_khmer_additions_do_not_perturb_existing_metrics():
    # PIN: the Khmer metric must not move cell_accuracy / recall / table_cer / numeric.
    pred_table = _make_table_from_grid(_NUMERIC_GT)
    result = evaluate_table([pred_table], _NUMERIC_GT)
    assert result["cell_accuracy"] == pytest.approx(1.0)
    assert result["cell_content_recall"] == pytest.approx(1.0)
    assert result["table_cer"] == pytest.approx(0.0)
    assert result["cells_total"] == 6
    assert result["numeric_cell_accuracy"] == pytest.approx(1.0)


# --- Empty-cell precision (phantom text in empty GT cells, e.g. Kiri's "|") ---

def test_empty_cell_precision_counts_phantom_text():
    # First row fully non-empty so _strip_title_row leaves the grid alone.
    gt_grid = [["h1", "h2"], ["b", ""], ["c", ""]]
    pred_table = _make_table_from_grid([["h1", "h2"], ["b", "|"], ["c", ""]])
    m = evaluate_table([pred_table], gt_grid)
    assert m["empty_gt_cells_total"] == 2
    assert m["empty_gt_cells_clean"] == 1
    assert m["empty_cell_precision"] == 0.5


def test_empty_cell_precision_perfect_when_empties_stay_empty():
    gt_grid = [["h1", "h2"], ["b", ""]]
    pred_table = _make_table_from_grid([["h1", "h2"], ["b", ""]])
    m = evaluate_table([pred_table], gt_grid)
    assert m["empty_cell_precision"] == 1.0


def test_empty_cell_precision_none_when_gt_has_no_empty_cells():
    gt_grid = [["a", "b"]]
    pred_table = _make_table_from_grid([["a", "b"]])
    m = evaluate_table([pred_table], gt_grid)
    assert m["empty_gt_cells_total"] == 0
    assert m["empty_cell_precision"] is None

