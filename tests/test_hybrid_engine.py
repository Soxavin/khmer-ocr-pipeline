from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import khmer_pipeline.hybrid_engine as he
from khmer_pipeline.models import PreprocessResult, SuryaResult, SuryaPageResult
from khmer_pipeline.evaluate_structure import pred_table_grid


def _preprocess(n_pages: int = 1) -> PreprocessResult:
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    return PreprocessResult(source_name="t.pdf", page_images=[img.copy() for _ in range(n_pages)],
                            dpi=200, page_count=n_pages)


def _base_with_table() -> SuryaResult:
    # a page Surya returns: paragraph text_blocks + ocr_text + one (fragmented) table bbox
    page = SuryaPageResult(
        page_index=0,
        text_blocks=[{"text": "para", "bbox": [0, 0, 10, 10]}],
        tables=[{"bbox": [10, 10, 200, 200], "cells": []}],
        ocr_text="para",
    )
    return SuryaResult(source_name="t.pdf", pages=[page], warnings=[])


# fake SLANet: a 2x2 grid
_FAKE_CELLS = [
    {"row_id": 0, "col_id": 0, "row_span": 1, "col_span": 1, "bbox": [0, 0, 50, 20]},
    {"row_id": 0, "col_id": 1, "row_span": 1, "col_span": 1, "bbox": [50, 0, 100, 20]},
    {"row_id": 1, "col_id": 0, "row_span": 1, "col_span": 1, "bbox": [0, 20, 50, 40]},
    {"row_id": 1, "col_id": 1, "row_span": 1, "col_span": 1, "bbox": [50, 20, 100, 40]},
]


# --- _build_table (pure) ---

def test_build_table_shape_and_grid():
    texts = ["A", "B", "C", "D"]
    tbl = _build = he._build_table(_FAKE_CELLS, texts, (10, 10, 200, 200))
    assert len(tbl["rows"]) == 2 and len(tbl["cols"]) == 2
    assert tbl["image_bbox"] == [10, 10, 200, 200]
    grid = pred_table_grid(tbl)  # the shape evaluate_structure consumes
    assert grid == [["A", "B"], ["C", "D"]]


def test_build_table_empty_cell_has_no_text_lines():
    tbl = he._build_table(_FAKE_CELLS, ["A", "", "C", "D"], (0, 0, 1, 1))
    empty = [c for c in tbl["cells"] if c["row_id"] == 0 and c["col_id"] == 1][0]
    assert empty["text_lines"] == []


# --- run_hybrid wiring (Surya + SLANet mocked) ---

def _run_hybrid_mocked(base: SuryaResult, cells, texts):
    # these tests exercise the cell-mode wiring specifically
    with patch.object(he, "run_surya", return_value=base), \
         patch.object(he, "_get_predictors", return_value=(None, object())), \
         patch.object(he, "_hybrid_mode", return_value="cell"), \
         patch.object(he, "predict_cells", return_value=cells), \
         patch.object(he, "_ocr_cells", return_value=texts):
        return he.run_hybrid(_preprocess())


def test_run_hybrid_returns_surya_result_with_rebuilt_table():
    r = _run_hybrid_mocked(_base_with_table(), _FAKE_CELLS, ["A", "B", "C", "D"])
    assert isinstance(r, SuryaResult)
    assert len(r.pages[0].tables) == 1
    assert pred_table_grid(r.pages[0].tables[0]) == [["A", "B"], ["C", "D"]]


def test_run_hybrid_preserves_text_blocks_and_ocr_text():
    r = _run_hybrid_mocked(_base_with_table(), _FAKE_CELLS, ["A", "B", "C", "D"])
    assert r.pages[0].text_blocks == [{"text": "para", "bbox": [0, 0, 10, 10]}]
    assert r.pages[0].ocr_text == "para"


def test_run_hybrid_page_without_tables_passthrough():
    page = SuryaPageResult(page_index=0, text_blocks=[], tables=[], ocr_text="x")
    base = SuryaResult(source_name="t.pdf", pages=[page], warnings=[])
    r = _run_hybrid_mocked(base, _FAKE_CELLS, ["A"])
    assert r.pages[0] is page  # untouched


def test_run_hybrid_slanet_empty_keeps_original_page():
    base = _base_with_table()
    r = _run_hybrid_mocked(base, [], [])
    assert r.pages[0] is base.pages[0]  # no cells → original page kept


# --- rowband mode ---

def test_hybrid_mode_defaults_to_rowband(monkeypatch):
    monkeypatch.delenv("KHMER_HYBRID_MODE", raising=False)
    assert he._hybrid_mode() == "rowband"


# --- layout detector selection ---

def test__layout_detector_defaults_to_surya(monkeypatch):
    monkeypatch.delenv("KHMER_LAYOUT_DETECTOR", raising=False)
    assert he._layout_detector() == "surya"


def test_run_hybrid_doclayout_sources_region_from_detector_not_page_tables(monkeypatch):
    monkeypatch.setenv("KHMER_LAYOUT_DETECTOR", "doclayout")
    page = SuryaPageResult(
        page_index=0,
        text_blocks=[{"text": "para", "bbox": [0, 0, 10, 10]}],
        tables=[],
        ocr_text="para",
    )
    base = SuryaResult(source_name="t.pdf", pages=[page], warnings=[])
    grid = {(0, 0): "A", (0, 1): "B", (1, 0): "C", (1, 1): "D"}
    with patch.object(he, "run_surya", return_value=base), \
         patch.object(he, "detect_table_boxes", return_value=[[10, 10, 200, 200]]), \
         patch.object(he, "_get_predictors", return_value=(None, object())), \
         patch.object(he, "predict_cells", return_value=_FAKE_CELLS), \
         patch.object(he, "_ocr_rowbands", return_value=grid):
        r = he.run_hybrid(_preprocess())
    assert len(r.pages[0].tables) == 1
    assert pred_table_grid(r.pages[0].tables[0]) == [["A", "B"], ["C", "D"]]


def test_run_hybrid_doclayout_no_boxes_passthrough(monkeypatch):
    monkeypatch.setenv("KHMER_LAYOUT_DETECTOR", "doclayout")
    page = SuryaPageResult(page_index=0, text_blocks=[], tables=[], ocr_text="x")
    base = SuryaResult(source_name="t.pdf", pages=[page], warnings=[])
    with patch.object(he, "run_surya", return_value=base), \
         patch.object(he, "detect_table_boxes", return_value=[]), \
         patch.object(he, "_get_predictors", return_value=(None, object())):
        r = he.run_hybrid(_preprocess())
    assert r.pages[0] is page  # untouched


def test_row_bands_full_width_and_padding():
    # a row whose cells do NOT span the full width must still produce a full-width strip
    cells = [
        {"row_id": 0, "col_id": 1, "row_span": 1, "col_span": 1, "bbox": [40, 4, 90, 24]},
        {"row_id": 1, "col_id": 0, "row_span": 1, "col_span": 1, "bbox": [0, 60, 50, 95]},
    ]
    bands = he._row_bands(cells, crop_w=200, crop_h=100)
    assert [b["row_id"] for b in bands] == [0, 1]
    assert all(b["bbox"][0] == 0 and b["bbox"][2] == 200 for b in bands)  # full-width x
    assert bands[0]["bbox"][1] == 0 and bands[0]["bbox"][3] == 32          # y0 clamped to 0, 24+8
    assert bands[1]["bbox"][1] == 52 and bands[1]["bbox"][3] == 100        # 60-8, y1 clamped to crop_h


def _fake_rec_seq(*calls):
    # each call returns one block per html in the next list (in order)
    seq = list(calls)

    def rec(imgs, layout_results, full_page):
        htmls = seq.pop(0)
        return [SimpleNamespace(blocks=[SimpleNamespace(html=h) for h in htmls])]
    return rec


def test_best_row_keeps_row_with_most_nonempty_cells():
    g = {(0, 0): "x", (0, 1): "", (1, 0): "A", (1, 1): "B"}  # row 1 is fuller
    assert he._best_row(g) == {(0, 0): "A", (0, 1): "B"}


def test_ocr_rowbands_offsets_rows_and_parses_columns():
    crop = np.zeros((100, 200, 3), dtype=np.uint8)
    bands = [{"row_id": 0, "bbox": [0, 0, 200, 30]}, {"row_id": 1, "bbox": [0, 30, 200, 60]}]
    htmls = ["<table><tr><td>A</td><td>B</td></tr></table>",
             "<table><tr><td>C</td><td>D</td></tr></table>"]
    grid = he._ocr_rowbands(_fake_rec_seq(htmls), crop, bands)  # no blanks → single pass
    assert grid == {(0, 0): "A", (0, 1): "B", (1, 0): "C", (1, 1): "D"}


def test_ocr_rowbands_retries_blank_band_and_fills_it():
    crop = np.zeros((100, 200, 3), dtype=np.uint8)
    bands = [{"row_id": 0, "bbox": [0, 0, 200, 30]},
             {"row_id": 1, "bbox": [0, 30, 200, 60]},
             {"row_id": 2, "bbox": [0, 60, 200, 90]}]
    pass1 = ["<table><tr><td>A</td></tr></table>", "", "<table><tr><td>C</td></tr></table>"]
    retry = ["<table><tr><td>B</td></tr></table>"]  # one box for the single blank band
    grid = he._ocr_rowbands(_fake_rec_seq(pass1, retry), crop, bands)
    assert grid == {(0, 0): "A", (1, 0): "B", (2, 0): "C"}


def test_ocr_rowbands_clamps_trailing_column_to_n_cols():
    # Surya sometimes emits a spurious trailing empty <td>; n_cols (SLANet count) drops it.
    crop = np.zeros((100, 200, 3), dtype=np.uint8)
    bands = [{"row_id": 0, "bbox": [0, 0, 200, 30]}]
    htmls = ["<table><tr><td>A</td><td>B</td><td></td></tr></table>"]  # 3 tds, last empty
    grid = he._ocr_rowbands(_fake_rec_seq(htmls), crop, bands, n_cols=2)
    assert grid == {(0, 0): "A", (0, 1): "B"}


def test_ocr_rowbands_still_blank_after_retry_reserves_a_row():
    crop = np.zeros((100, 200, 3), dtype=np.uint8)
    bands = [{"row_id": 0, "bbox": [0, 0, 200, 30]},
             {"row_id": 1, "bbox": [0, 30, 200, 60]},
             {"row_id": 2, "bbox": [0, 60, 200, 90]}]
    pass1 = ["<table><tr><td>A</td></tr></table>", "", "<table><tr><td>C</td></tr></table>"]
    retry = [""]  # retry also blank → band 1 stays empty but keeps its row slot
    grid = he._ocr_rowbands(_fake_rec_seq(pass1, retry), crop, bands)
    assert grid == {(0, 0): "A", (2, 0): "C"}


def test_run_hybrid_rowband_builds_table_from_grid(monkeypatch):
    monkeypatch.setenv("KHMER_HYBRID_MODE", "rowband")
    grid = {(0, 0): "A", (0, 1): "B", (1, 0): "C", (1, 1): "D"}
    with patch.object(he, "run_surya", return_value=_base_with_table()), \
         patch.object(he, "_get_predictors", return_value=(None, object())), \
         patch.object(he, "predict_cells", return_value=_FAKE_CELLS), \
         patch.object(he, "_ocr_rowbands", return_value=grid):
        r = he.run_hybrid(_preprocess())
    assert len(r.pages[0].tables) == 1
    assert pred_table_grid(r.pages[0].tables[0]) == [["A", "B"], ["C", "D"]]
