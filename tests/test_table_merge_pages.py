from __future__ import annotations
from types import SimpleNamespace
from khmer_pipeline.table_merge_pages import merge_document_tables
from khmer_pipeline.evaluate_structure import pred_table_grid

_H = ["ល.រ", "មុខទំនិញ", "តម្លៃ"]


def _table(grid):
    cells = []
    for r, row in enumerate(grid):
        for c, txt in enumerate(row):
            cells.append({"row_id": r, "col_id": c,
                          "text_lines": ([{"text": txt}] if txt else []), "bbox": []})
    n_cols = max((len(r) for r in grid), default=0)
    return {"rows": [{"row_id": i} for i in range(len(grid))],
            "cols": [{"col_id": j} for j in range(n_cols)],
            "cells": cells}


def _page(idx, *grids):
    return SimpleNamespace(page_index=idx, tables=[_table(g) for g in grids])


def test_continuation_joins_same_column_tables_across_pages():
    pages = [_page(0, [_H, ["1", "aa", "10"]]), _page(1, [_H, ["2", "bb", "20"]])]
    merged = merge_document_tables(pages)
    assert len(merged) == 1
    assert pred_table_grid(merged[0]) == [_H, ["1", "aa", "10"], ["2", "bb", "20"]]
    assert merged[0]["source_pages"] == [0, 1]


def test_repeated_header_deduped_at_page_break():
    pages = [_page(0, [_H, ["1", "aa", "10"]]), _page(1, [_H, ["2", "bb", "20"]])]
    grid = pred_table_grid(merge_document_tables(pages)[0])
    assert grid.count(_H) == 1  # header appears once, not once per page


def test_non_header_first_row_not_dropped():
    pages = [_page(0, [_H, ["1", "aa", "10"]]), _page(1, [["2", "bb", "20"], ["3", "cc", "30"]])]
    grid = pred_table_grid(merge_document_tables(pages)[0])
    assert grid == [_H, ["1", "aa", "10"], ["2", "bb", "20"], ["3", "cc", "30"]]


def test_column_count_change_starts_new_logical_table():
    pages = [_page(0, [["a", "b"]]), _page(1, [["c", "d", "e", "f"]])]  # 2 vs 4 cols, > tolerance
    merged = merge_document_tables(pages)
    assert len(merged) == 2


def test_within_tolerance_columns_still_merge():
    pages = [_page(0, [["a", "b", "c"]]), _page(1, [["d", "e"]])]  # 3 vs 2, diff 1 == tolerance
    assert len(merge_document_tables(pages)) == 1


def test_rows_renumbered_contiguously():
    pages = [_page(0, [_H, ["1", "aa", "10"]]), _page(1, [["2", "bb", "20"]])]
    merged = merge_document_tables(pages)[0]
    row_ids = sorted({c["row_id"] for c in merged["cells"]})
    assert row_ids == [0, 1, 2]


def test_merged_row_count_equals_sum_minus_deduped_headers():
    pages = [_page(0, [_H, ["1", "aa", "10"], ["2", "bb", "20"]]),
             _page(1, [_H, ["3", "cc", "30"]])]
    merged = merge_document_tables(pages)[0]
    n_rows = len(merged["rows"])
    assert n_rows == 4  # 3 + 2 = 5 input rows, minus 1 duplicated header


def test_empty_and_cellless_tables_ignored():
    pages = [_page(0), SimpleNamespace(page_index=1, tables=[{"cells": []}]),
             _page(2, [_H, ["1", "aa", "10"]])]
    merged = merge_document_tables(pages)
    assert len(merged) == 1
    assert merged[0]["source_pages"] == [2]


def test_blank_rows_dropped_in_merge():
    pages = [_page(0, [_H, ["1", "aa", "10"], ["", "", ""], ["2", "bb", "20"], ["", "", ""]])]
    grid = pred_table_grid(merge_document_tables(pages)[0])
    assert grid == [_H, ["1", "aa", "10"], ["2", "bb", "20"]]


def test_no_tables_returns_empty():
    assert merge_document_tables([_page(0), _page(1)]) == []
