"""Unit tests for webapp.tables — the pure table-shaping helpers ported from app.py."""
from webapp import tables


def _block(table_id, cells):
    return {"table_id": table_id, "cells": cells}


def test_block_to_table_builds_rectangular_grid_and_conf():
    block = _block("t1", [
        {"row": 0, "col": 0, "text": "a", "confidence": 0.9},
        {"row": 0, "col": 1, "text": "b"},
        {"row": 1, "col": 0, "text": "c", "confidence": 0.2},
    ])
    tid, grid, conf = tables.block_to_table(block)
    assert tid == "t1"
    assert grid == [["a", "b"], ["c", ""]]
    assert conf[0][0] == 0.9 and conf[0][1] is None and conf[1][0] == 0.2


def test_empty_block_yields_empty_grid():
    _, grid, conf = tables.block_to_table(_block("t", []))
    assert grid == [] and conf == []


def test_stitched_detection_and_scopes():
    doc = {"document_tables": [_block("d1", [{"row": 0, "col": 0, "text": "x"}])]}
    assert tables.is_stitched(doc)
    # stitched: every page shows all doc tables
    assert len(tables.page_table_blocks(doc, 0)) == 1
    assert len(tables.all_table_blocks(doc)) == 1


def test_unstitched_page_scope_isolates_pages():
    doc = {"pages": [
        {"tables": [_block("p1t1", [{"row": 0, "col": 0, "text": "a"}])]},
        {"tables": [_block("p2t1", [{"row": 0, "col": 0, "text": "b"}]),
                    _block("p2t2", [{"row": 0, "col": 0, "text": "c"}])]},
    ]}
    assert not tables.is_stitched(doc)
    # page 0 sees only its own table; downloads see all 3
    assert [b["table_id"] for b in tables.page_table_blocks(doc, 0)] == ["p1t1"]
    assert [b["table_id"] for b in tables.page_table_blocks(doc, 1)] == ["p2t1", "p2t2"]
    assert len(tables.all_table_blocks(doc)) == 3


def test_page_scope_out_of_range_is_empty():
    doc = {"pages": [{"tables": []}]}
    assert tables.page_table_blocks(doc, 5) == []


def test_patch_table_block_rewrites_cells_from_edit():
    block = _block("t1", [{"row": 0, "col": 0, "text": "old"}])
    patched = tables.patch_table_block(block, {"t1": [["new", "x"]]})
    assert patched["cells"] == [
        {"row": 0, "col": 0, "text": "new"},
        {"row": 0, "col": 1, "text": "x"},
    ]


def test_patch_table_block_untouched_without_edit():
    block = _block("t1", [{"row": 0, "col": 0, "text": "old"}])
    assert tables.patch_table_block(block, {}) is block


# ---------------------------------------------------------------------------
# stitch_grids — export-time joining of per-page tables (§2.43)
# ---------------------------------------------------------------------------
# Mirrors engines/table_merge_pages.merge_document_tables at the GRID level:
# review always runs unstitched (so page↔image linking survives), and the
# analyst's EDITS live as grids — so the combine must happen where the edits are.


def test_stitch_grids_joins_pages_and_drops_repeated_header():
    ft = [
        ("doc_page1_table1", [["ID", "Item"], ["1", "rice"]]),
        ("doc_page2_table1", [["ID", "Item"], ["2", "beef"]]),
    ]
    out = tables.stitch_grids(ft, "doc")
    assert out == [("doc_table1", [["ID", "Item"], ["1", "rice"], ["2", "beef"]])]


def test_stitch_grids_keeps_a_differing_first_row():
    """A continuation page whose first row is data, not a repeated header, keeps it."""
    ft = [
        ("doc_page1_table1", [["ID", "Item"], ["1", "rice"]]),
        ("doc_page2_table1", [["2", "beef"], ["3", "pork"]]),
    ]
    out = tables.stitch_grids(ft, "doc")
    assert out[0][1] == [["ID", "Item"], ["1", "rice"], ["2", "beef"], ["3", "pork"]]


def test_stitch_grids_splits_on_column_structure_change():
    """Genuinely different sections stay separate tables (tolerance ±1 col)."""
    ft = [
        ("doc_page1_table1", [["a", "b", "c"], ["1", "2", "3"]]),
        ("doc_page2_table1", [["x", "y", "z", "w", "v"], ["4", "5", "6", "7", "8"]]),
    ]
    out = tables.stitch_grids(ft, "doc")
    assert len(out) == 2
    assert [tid for tid, _ in out] == ["doc_table1", "doc_table2"]


def test_stitch_grids_drops_fully_empty_rows_and_pads_ragged():
    ft = [
        ("doc_page1_table1", [["ID", "Item"], ["", "  "], ["1", "rice"]]),
        ("doc_page2_table1", [["2", "beef", "extra"]]),  # within ±1 col tolerance
    ]
    out = tables.stitch_grids(ft, "doc")
    grid = out[0][1]
    assert ["", "  "] not in grid
    assert all(len(r) == 3 for r in grid)  # padded to the widest row
    assert grid[-1] == ["2", "beef", "extra"]


def test_stitch_grids_normalises_header_before_comparing():
    """Whitespace/unicode differences must not defeat header de-duplication."""
    ft = [
        ("doc_page1_table1", [["ID", "Item"], ["1", "rice"]]),
        ("doc_page2_table1", [["ID ", " Item"], ["2", "beef"]]),
    ]
    out = tables.stitch_grids(ft, "doc")
    assert out[0][1] == [["ID", "Item"], ["1", "rice"], ["2", "beef"]]


def test_stitch_grids_single_and_empty_inputs():
    assert tables.stitch_grids([], "doc") == []
    ft = [("doc_page1_table1", [["a"], ["1"]])]
    assert tables.stitch_grids(ft, "doc") == [("doc_table1", [["a"], ["1"]])]


def test_stitch_grids_preserves_edits():
    """The whole point: combining happens after edits are folded in."""
    ft = [
        ("doc_page1_table1", [["ID", "Item"], ["1", "EDITED"]]),
        ("doc_page2_table1", [["ID", "Item"], ["2", "beef"]]),
    ]
    out = tables.stitch_grids(ft, "doc")
    assert "EDITED" in out[0][1][1]
