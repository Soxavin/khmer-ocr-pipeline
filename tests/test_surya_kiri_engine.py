"""Tests for the Surya + Kiri hybrid engine (surya_kiri_engine.py).

Covers registry wiring, contract compliance, raw-image recognition, and
fallthrough paths for pages with no tables. The layout predictor, TableRecPredictor,
and the Kiri recognizer are stubbed to avoid model downloads in CI.
"""
from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch
import numpy as np

from khmer_pipeline.models import PreprocessResult, SuryaResult, SuryaPageResult
from khmer_pipeline.engines.engine_registry import _OCR_ENGINES
from khmer_pipeline.engines.surya_kiri_engine import run_surya_kiri


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _preprocess(n_pages: int = 1, with_raw: bool = True) -> PreprocessResult:
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    pages = [img.copy() for _ in range(n_pages)]
    return PreprocessResult(
        source_name="test.pdf",
        page_images=pages,
        dpi=200,
        page_count=n_pages,
        recognition_page_images=[p.copy() for p in pages] if with_raw else None,
    )


def _base_page() -> SuryaPageResult:
    return SuryaPageResult(
        page_index=0,
        text_blocks=[{"text": "Header", "bbox": [10, 10, 100, 30]}],
        tables=[{"bbox": [50.0, 50.0, 350.0, 350.0], "cells": []}],
        ocr_text="Header",
    )


def _base_result(warnings=None) -> SuryaResult:
    return SuryaResult(source_name="test.pdf", pages=[_base_page()], warnings=warnings or [])


class _FakeBox:
    def __init__(self, bbox, label):
        self.bbox = bbox
        self.label = label


class _FakeLayout:
    def __init__(self, boxes):
        self.bboxes = boxes


def _fake_layout_pred(with_table: bool = True):
    boxes = ([_FakeBox([50.0, 50.0, 350.0, 350.0], "Table")] if with_table
             else [_FakeBox([0.0, 0.0, 20.0, 20.0], "Text")])
    return lambda imgs: [_FakeLayout(boxes)]


def _fake_table_rec_cells(n_rows: int = 2, n_cols: int = 2):
    """Mock cells matching surya.table_rec.schema.TableCell (polygon/row_id/col_id)."""
    cells = []
    for r in range(n_rows):
        for c in range(n_cols):
            cell = MagicMock()
            cell.row_id = r
            cell.col_id = c
            x0, y0 = c * 50 + 5, r * 30 + 5
            x1, y1 = x0 + 45, y0 + 25
            cell.polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            cells.append(cell)
    return cells


def _run(base=None, *, with_table=True, cells=None, recognize="42",
         preprocess=None):
    """Run the engine with all model-loading dependencies stubbed."""
    base = base or _base_result()
    cells = _fake_table_rec_cells(2, 2) if cells is None else cells
    with ExitStack() as stack:
        p = lambda tgt, **kw: stack.enter_context(patch(f"khmer_pipeline.engines.surya_kiri_engine.{tgt}", **kw))
        p("run_surya", return_value=base)
        p("get_manager")
        p("_get_predictors", return_value=(_fake_layout_pred(with_table), None))
        if isinstance(recognize, list):
            p("recognize_cells", return_value=recognize)
        else:
            p("recognize_cells", side_effect=lambda crops: [recognize] * len(crops))
        mock_tbl = MagicMock()
        mock_result = MagicMock()
        mock_result.cells = cells
        mock_tbl.return_value = [mock_result]
        stack.enter_context(patch("surya.table_rec.TableRecPredictor", return_value=mock_tbl))
        return run_surya_kiri(preprocess or _preprocess())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_contains_surya_kiri():
    assert "surya_kiri" in _OCR_ENGINES
    assert _OCR_ENGINES["surya_kiri"] is run_surya_kiri


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def test_run_surya_kiri_returns_surya_result():
    r = _run()
    assert isinstance(r, SuryaResult)


def test_run_surya_kiri_preserves_source_name():
    assert _run().source_name == "test.pdf"


def test_run_surya_kiri_preserves_warnings():
    r = _run(base=SuryaResult(source_name="t", pages=[_base_page()], warnings=["test warning"]))
    assert "test warning" in r.warnings


# ---------------------------------------------------------------------------
# Fallthrough
# ---------------------------------------------------------------------------

def test_no_table_pages_fall_through():
    """Pages whose raw-image layout has no Table region keep the base page."""
    base = _base_result()
    r = _run(base=base, with_table=False)
    assert len(r.pages) == 1
    assert r.pages[0].tables == base.pages[0].tables


def test_empty_cells_skips_table():
    """When TableRecPredictor returns zero cells, the base page is kept."""
    base = _base_result()
    r = _run(base=base, cells=[])
    assert r.pages[0].tables == base.pages[0].tables


# ---------------------------------------------------------------------------
# Raw-image usage
# ---------------------------------------------------------------------------

def test_falls_back_to_page_images_without_raw():
    """When recognition_page_images is None, the engine still runs (uses page_images)."""
    r = _run(preprocess=_preprocess(with_raw=False))
    assert isinstance(r, SuryaResult)
    assert len(r.pages[0].tables) == 1


# ---------------------------------------------------------------------------
# Table structure
# ---------------------------------------------------------------------------

def test_grid_produces_correct_table_shape():
    r = _run(recognize=["A", "B", "C", "D"])
    table = r.pages[0].tables[0]
    assert len(table["rows"]) == 2
    assert len(table["cols"]) == 2
    from khmer_pipeline.evaluation.evaluate_structure import pred_table_grid
    assert pred_table_grid(table) == [["A", "B"], ["C", "D"]]


def test_table_has_top_level_bbox_for_layout_overlay():
    """app.py's layout overlay reads t["bbox"] on every table; the engine must set
    it (run_surya does so separately from _build_table_from_grid)."""
    table = _run().pages[0].tables[0]
    assert "bbox" in table
    assert len(table["bbox"]) == 4
