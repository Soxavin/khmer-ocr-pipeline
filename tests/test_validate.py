from __future__ import annotations

from khmer_pipeline.models import CorrectedPageResult
from khmer_pipeline.validate import validate_pages, _parse_number, _sequence_illegal


def _cell(row, col, text="", conf=None):
    c: dict = {"row_id": row, "col_id": col,
               "text_lines": ([{"text": text}] if text else [])}
    if conf is not None:
        c["confidence"] = conf
    return c


def _table(cells):
    return {"rows": [], "cols": [], "cells": cells}


def _page(tables, page_index=0):
    return CorrectedPageResult(
        page_index=page_index, text_blocks=[], tables=tables,
        raw_ocr_text="", corrected_text="", correction_diff="", qwen_used=False,
    )


def _flags(cell):
    return cell.get("flags", [])


# --- number parsing ---------------------------------------------------------

def test_parse_plain_and_separators():
    assert _parse_number("1,234")[0] == 1234.0
    assert _parse_number("12.5")[0] == 12.5


def test_parse_khmer_digits():
    assert _parse_number("១២,០០០")[0] == 12000.0


def test_parse_negatives_and_percent():
    assert _parse_number("(123)")[0] == -123.0
    assert _parse_number("-5")[0] == -5.0
    val, pct = _parse_number("31.16%")
    assert val == 31.16 and pct is True


def test_parse_rejects_text():
    assert _parse_number("hello") is None
    assert _parse_number("") is None


# --- sequence_illegal -------------------------------------------------------

def test_sequence_illegal_trailing_coeng():
    assert _sequence_illegal(chr(0x179F) + chr(0x17D2)) is True  # base + coeng at end


def test_sequence_illegal_leading_mark():
    assert _sequence_illegal(chr(0x17B6)) is True  # dependent vowel with no base


def test_sequence_legal_normal_cluster():
    # base + coeng + base is a valid subscript cluster
    assert _sequence_illegal(chr(0x179F) + chr(0x17D2) + chr(0x1780)) is False


def test_sequence_illegal_flag_attached():
    cell = _cell(0, 0, chr(0x179F) + chr(0x17D2))
    validate_pages([_page([_table([cell])])])
    assert "sequence_illegal" in _flags(cell)


# --- digit_mixed ------------------------------------------------------------

def test_digit_mixed_flagged():
    cell = _cell(0, 0, "12៣")  # ASCII 1,2 + Khmer ៣
    validate_pages([_page([_table([cell])])])
    assert "digit_mixed" in _flags(cell)


def test_digit_pure_not_flagged():
    cell = _cell(0, 0, "123")
    validate_pages([_page([_table([cell])])])
    assert "digit_mixed" not in _flags(cell)


# --- low_conf ---------------------------------------------------------------

def test_low_conf_flagged_below_threshold():
    cell = _cell(0, 0, "x", conf=0.4)
    validate_pages([_page([_table([cell])])])
    assert "low_conf" in _flags(cell)


def test_high_conf_not_flagged():
    cell = _cell(0, 0, "x", conf=0.95)
    validate_pages([_page([_table([cell])])])
    assert "flags" not in cell


def test_blank_cell_never_flagged_low_conf():
    cell = _cell(0, 0, "", conf=0.0)
    validate_pages([_page([_table([cell])])])
    assert "flags" not in cell


# --- numeric column + numeric_unparseable -----------------------------------

def _numeric_column_table(body_col_texts, header="amount"):
    # column 1 is the numeric column; column 0 is labels. row 0 is header.
    cells = [_cell(0, 0, "label"), _cell(0, 1, header)]
    for i, txt in enumerate(body_col_texts, start=1):
        cells.append(_cell(i, 0, f"item{i}"))
        cells.append(_cell(i, 1, txt))
    return _table(cells)


def test_numeric_unparseable_flagged_in_numeric_column():
    # 4 numeric + 1 junk => 80% >= 70% => numeric column; junk flagged.
    table = _numeric_column_table(["10", "20", "30", "40", "oops"])
    validate_pages([_page([table])])
    cells = {(c["row_id"], c["col_id"]): c for c in table["cells"]}
    assert "numeric_unparseable" in _flags(cells[(5, 1)])
    assert "flags" not in cells[(1, 1)]


def test_below_threshold_column_not_numeric():
    # 2 numeric + 3 junk => 40% < 70% => not numeric => nothing flagged.
    table = _numeric_column_table(["10", "20", "a", "b", "c"])
    validate_pages([_page([table])])
    cells = {(c["row_id"], c["col_id"]): c for c in table["cells"]}
    assert all("flags" not in c for c in cells.values())


def test_header_not_flagged_unparseable():
    table = _numeric_column_table(["10", "20", "30"])
    validate_pages([_page([table])])
    cells = {(c["row_id"], c["col_id"]): c for c in table["cells"]}
    assert "flags" not in cells[(0, 1)]  # header text cell not flagged


# --- structure_ragged -------------------------------------------------------

def test_structure_ragged_flags_odd_row():
    cells = [
        _cell(0, 0, "a"), _cell(0, 1, "b"),
        _cell(1, 0, "c"), _cell(1, 1, "d"),
        _cell(2, 0, "e"),  # short row
    ]
    table = _table(cells)
    validate_pages([_page([table])])
    by = {(c["row_id"], c["col_id"]): c for c in cells}
    assert "structure_ragged" in _flags(by[(2, 0)])
    assert "structure_ragged" not in _flags(by[(0, 0)])


def test_structure_ragged_skipped_under_three_rows():
    cells = [_cell(0, 0, "a"), _cell(0, 1, "b"), _cell(1, 0, "c")]
    table = _table(cells)
    validate_pages([_page([table])])
    assert all("structure_ragged" not in _flags(c) for c in cells)


# --- numeric_mismatch (dormant: no harvested keywords) ----------------------

def test_numeric_mismatch_dormant():
    # A "total"-looking row whose value is wrong is NOT flagged, because the
    # harvested total-row keyword list is empty (see validate._TOTAL_ROW_LABELS).
    from khmer_pipeline.validate import _TOTAL_ROW_LABELS
    assert _TOTAL_ROW_LABELS == ()
    cells = [
        _cell(0, 0, "label"), _cell(0, 1, "amount"),
        _cell(1, 0, "a"), _cell(1, 1, "10"),
        _cell(2, 0, "b"), _cell(2, 1, "20"),
        _cell(3, 0, "total"), _cell(3, 1, "999"),  # wrong total
    ]
    table = _table(cells)
    validate_pages([_page([table])])
    by = {(c["row_id"], c["col_id"]): c for c in cells}
    assert "numeric_mismatch" not in _flags(by[(3, 1)])


def test_numeric_mismatch_fires_with_injected_keyword(monkeypatch):
    # Exercise the (dormant) mismatch logic by injecting a keyword.
    import khmer_pipeline.validate as v
    monkeypatch.setattr(v, "_TOTAL_ROW_LABELS", ("TOTAL",))
    cells = [
        _cell(0, 0, "label"), _cell(0, 1, "amount"),
        _cell(1, 0, "a"), _cell(1, 1, "10"),
        _cell(2, 0, "b"), _cell(2, 1, "20"),
        _cell(3, 0, "TOTAL"), _cell(3, 1, "999"),  # should be 30
    ]
    table = _table(cells)
    v.validate_pages([_page([table])])
    by = {(c["row_id"], c["col_id"]): c for c in cells}
    assert "numeric_mismatch" in _flags(by[(3, 1)])


def test_numeric_match_no_flag_with_injected_keyword(monkeypatch):
    import khmer_pipeline.validate as v
    monkeypatch.setattr(v, "_TOTAL_ROW_LABELS", ("TOTAL",))
    cells = [
        _cell(0, 0, "label"), _cell(0, 1, "amount"),
        _cell(1, 0, "a"), _cell(1, 1, "10"),
        _cell(2, 0, "b"), _cell(2, 1, "20"),
        _cell(3, 0, "TOTAL"), _cell(3, 1, "30"),  # correct
    ]
    table = _table(cells)
    v.validate_pages([_page([table])])
    by = {(c["row_id"], c["col_id"]): c for c in cells}
    assert "numeric_mismatch" not in _flags(by[(3, 1)])


# --- multi-flag + clean cells -----------------------------------------------

def test_multi_flag_cell():
    # low conf + mixed digits on one cell.
    cell = _cell(0, 0, "12៣", conf=0.3)
    validate_pages([_page([_table([cell])])])
    assert set(_flags(cell)) >= {"low_conf", "digit_mixed"}


def test_clean_cells_get_no_flags_key():
    cells = [_cell(0, 0, "hello", conf=0.99), _cell(0, 1, "world", conf=0.99)]
    validate_pages([_page([_table(cells)])])
    assert all("flags" not in c for c in cells)


def test_no_duplicate_flags():
    cell = _cell(0, 0, "12៣", conf=0.3)
    validate_pages([_page([_table([cell])])])
    # run again — flags must not duplicate
    validate_pages([_page([_table([cell])])])
    assert _flags(cell).count("digit_mixed") == 1


# --- summary warnings -------------------------------------------------------

def test_returns_per_page_summary_warnings():
    cell = _cell(0, 0, "12៣", conf=0.3)
    warnings = validate_pages([_page([_table([cell])], page_index=1)])
    assert any("page 2" in w and "digit_mixed" in w for w in warnings)
