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
    with patch.object(he, "run_surya", return_value=base), \
         patch.object(he, "_get_predictors", return_value=(None, object())), \
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
