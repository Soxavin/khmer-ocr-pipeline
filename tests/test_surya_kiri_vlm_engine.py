"""Tests for the Surya-VLM + Kiri engine (surya_kiri_vlm_engine.py).

The engine's contract: its FLOOR is plain Surya — every fallback path (grid
mismatch, TableRec failure, low Kiri confidence) keeps Surya's text untouched.
Kiri only replaces the text of Khmer-heavy cells, and only when the TableRec
grid shape exactly matches the VLM grid shape (a discrete gate — §2.40's
pixel-threshold lesson). Models are stubbed as in test_surya_kiri_engine.py.
"""
from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch
import numpy as np

from khmer_pipeline.models import PreprocessResult, SuryaResult, SuryaPageResult
from khmer_pipeline.engines.engine_registry import _OCR_ENGINES
from khmer_pipeline.engines.surya_kiri_vlm_engine import (
    run_surya_kiri_vlm, _khmer_ratio, _KIRI_REPLACE_MIN_CONF,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _preprocess(with_raw: bool = True) -> PreprocessResult:
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    return PreprocessResult(
        source_name="test.pdf", page_images=[img.copy()], dpi=200, page_count=1,
        recognition_page_images=[img.copy()] if with_raw else None,
    )


def _cell(r, c, text, col_span=None):
    cell = {"row_id": r, "col_id": c, "cell_id": r * 10 + c, "bbox": [], "polygon": [],
            "text_lines": [{"text": text, "bbox": []}] if text else []}
    if col_span:
        cell["col_span"] = col_span
    return cell


def _vlm_table(cells, n_rows, n_cols, bbox=(50.0, 50.0, 350.0, 350.0)):
    return {"rows": [{"row_id": i} for i in range(n_rows)],
            "cols": [{"col_id": j} for j in range(n_cols)],
            "cells": cells, "image_bbox": list(bbox), "bbox": list(bbox)}


def _base(tables) -> SuryaResult:
    page = SuryaPageResult(page_index=0, text_blocks=[{"text": "para", "bbox": [0, 0, 9, 9]}],
                           tables=tables, ocr_text="para")
    return SuryaResult(source_name="test.pdf", pages=[page], warnings=[])


def _tr_cells(n_rows, n_cols):
    """TableRec unit-cell mocks: 100x60-px cells tiled from (0,0) in crop coords."""
    out = []
    for r in range(n_rows):
        for c in range(n_cols):
            m = MagicMock()
            m.row_id, m.col_id = r, c
            x0, y0 = c * 100, r * 60
            m.polygon = [[x0, y0], [x0 + 100, y0], [x0 + 100, y0 + 60], [x0, y0 + 60]]
            out.append(m)
    return out


_KH = "សាច់គោ"          # Khmer-heavy text
_NUM = "13,000"          # numeric text


def _run(base, *, tr_cells=None, tr_raises=False, recognize="ខ្មែរ",
         confidence=0.99, preprocess=None, crop_sink=None):
    with ExitStack() as stack:
        p = lambda tgt, **kw: stack.enter_context(
            patch(f"khmer_pipeline.engines.surya_kiri_vlm_engine.{tgt}", **kw))
        p("run_surya", return_value=base)
        p("get_manager")

        def _rec(crops, warning_sink=None):
            if crop_sink is not None:
                crop_sink.extend(crops)
            texts = recognize if isinstance(recognize, list) else [recognize] * len(crops)
            confs = confidence if isinstance(confidence, list) else [confidence] * len(crops)
            return list(zip(texts, confs))

        p("recognize_cells_conf", side_effect=_rec)
        mock_tbl = MagicMock()
        if tr_raises:
            mock_tbl.side_effect = RuntimeError("tablerec boom")
        else:
            res = MagicMock()
            res.cells = tr_cells if tr_cells is not None else _tr_cells(2, 2)
            mock_tbl.return_value = [res]
        stack.enter_context(patch("surya.table_rec.TableRecPredictor", return_value=mock_tbl))
        return run_surya_kiri_vlm(preprocess or _preprocess())


def _text(table, r, c):
    cell = next(x for x in table["cells"] if x["row_id"] == r and x["col_id"] == c)
    return " ".join(l["text"] for l in cell.get("text_lines", []) if l.get("text"))


# ---------------------------------------------------------------------------
# Registry + classification
# ---------------------------------------------------------------------------

def test_registry_contains_surya_kiri_vlm():
    assert "surya_kiri_vlm" in _OCR_ENGINES
    assert _OCR_ENGINES["surya_kiri_vlm"] is run_surya_kiri_vlm


def test_khmer_ratio_classification():
    assert _khmer_ratio("សាច់គោ") == 1.0
    assert _khmer_ratio("13,000") == 0.0
    assert _khmer_ratio("") == 0.0
    assert 0.4 < _khmer_ratio("សាច់ CP1") < 0.7  # mixed


# ---------------------------------------------------------------------------
# The re-read path
# ---------------------------------------------------------------------------

def test_khmer_cell_replaced_numeric_untouched():
    table = _vlm_table([_cell(0, 0, _KH), _cell(0, 1, _NUM),
                        _cell(1, 0, _KH), _cell(1, 1, _NUM)], 2, 2)
    r = _run(_base([table]), recognize="ខ្មែរថ្មី", confidence=0.99)
    t = r.pages[0].tables[0]
    assert _text(t, 0, 0) == "ខ្មែរថ្មី"          # Khmer-heavy → Kiri text
    assert _text(t, 0, 1) == _NUM               # numeric → Surya text kept
    kh_cell = next(x for x in t["cells"] if x["row_id"] == 0 and x["col_id"] == 0)
    num_cell = next(x for x in t["cells"] if x["row_id"] == 0 and x["col_id"] == 1)
    assert kh_cell["confidence"] == 0.99        # re-read cells carry confidence
    assert "confidence" not in num_cell          # untouched cells stay plain-Surya


def test_low_confidence_keeps_surya_text():
    table = _vlm_table([_cell(0, 0, _KH), _cell(0, 1, _NUM),
                        _cell(1, 0, _KH), _cell(1, 1, _NUM)], 2, 2)
    r = _run(_base([table]), recognize="x", confidence=_KIRI_REPLACE_MIN_CONF - 0.1)
    assert _text(r.pages[0].tables[0], 0, 0) == _KH


def test_empty_kiri_read_keeps_surya_text():
    table = _vlm_table([_cell(0, 0, _KH), _cell(0, 1, _NUM),
                        _cell(1, 0, _KH), _cell(1, 1, _NUM)], 2, 2)
    r = _run(_base([table]), recognize="  ", confidence=0.99)
    assert _text(r.pages[0].tables[0], 0, 0) == _KH


def test_spanning_khmer_anchor_gets_union_crop_with_pad():
    # VLM 2x3 grid; anchor (0,0) spans cols 0-1. Union crop = TableRec units
    # (0,0)+(0,1) → width 200 plus _CROP_PAD_PX on the free right edge (left/top
    # clamp at 0). The pad keeps Khmer ascenders/descenders from being clipped
    # by the cell-tight boxes.
    from khmer_pipeline.engines.surya_kiri_vlm_engine import _CROP_PAD_PX
    table = _vlm_table([_cell(0, 0, _KH, col_span=2), _cell(0, 2, _NUM),
                        _cell(1, 0, _NUM), _cell(1, 1, _NUM), _cell(1, 2, _NUM)], 2, 3)
    crops: list = []
    r = _run(_base([table]), tr_cells=_tr_cells(2, 3), recognize="ថ្មី",
             confidence=0.99, crop_sink=crops)
    assert len(crops) == 1                              # only the one Khmer cell re-read
    assert crops[0].shape[1] == 200 + _CROP_PAD_PX      # union + right pad (left clamped)
    assert crops[0].shape[0] == 60 + _CROP_PAD_PX       # unit height + bottom pad (top clamped)
    assert _text(r.pages[0].tables[0], 0, 0) == "ថ្មី"


# ---------------------------------------------------------------------------
# The gate + fallbacks — the floor is plain Surya
# ---------------------------------------------------------------------------

def test_grid_shape_mismatch_keeps_table_and_warns():
    table = _vlm_table([_cell(0, 0, _KH), _cell(0, 1, _NUM)], 1, 2)  # VLM 1x2
    r = _run(_base([table]), tr_cells=_tr_cells(2, 2))               # TableRec 2x2
    assert r.pages[0].tables[0] == table  # byte-identical passthrough
    assert any("grid mismatch" in w and "1x2" in w and "2x2" in w for w in r.warnings)


def test_tablerec_failure_keeps_table_and_warns():
    table = _vlm_table([_cell(0, 0, _KH), _cell(0, 1, _NUM),
                        _cell(1, 0, _KH), _cell(1, 1, _NUM)], 2, 2)
    r = _run(_base([table]), tr_raises=True)
    assert r.pages[0].tables[0] == table
    assert any("Khmer re-read skipped" in w for w in r.warnings)


def test_no_khmer_cells_skips_tablerec_entirely():
    table = _vlm_table([_cell(0, 0, _NUM), _cell(0, 1, _NUM),
                        _cell(1, 0, _NUM), _cell(1, 1, _NUM)], 2, 2)
    r = _run(_base([table]))
    assert r.pages[0].tables[0] == table
    assert not any("re-read" in w for w in r.warnings)  # nothing to do, no noise


def test_base_result_never_mutated():
    table = _vlm_table([_cell(0, 0, _KH), _cell(0, 1, _NUM),
                        _cell(1, 0, _KH), _cell(1, 1, _NUM)], 2, 2)
    base = _base([table])
    _run(base, recognize="ថ្មី", confidence=0.99)
    # the ORIGINAL table object still holds Surya's text
    assert _text(base.pages[0].tables[0], 0, 0) == _KH


def test_text_blocks_and_ocr_text_passthrough():
    table = _vlm_table([_cell(0, 0, _KH), _cell(0, 1, _NUM),
                        _cell(1, 0, _KH), _cell(1, 1, _NUM)], 2, 2)
    r = _run(_base([table]))
    assert r.pages[0].text_blocks == [{"text": "para", "bbox": [0, 0, 9, 9]}]
    assert r.pages[0].ocr_text == "para"


def test_missing_recognition_images_warns_and_still_runs():
    table = _vlm_table([_cell(0, 0, _KH), _cell(0, 1, _NUM),
                        _cell(1, 0, _KH), _cell(1, 1, _NUM)], 2, 2)
    r = _run(_base([table]), preprocess=_preprocess(with_raw=False))
    assert any("recognition images unavailable" in w for w in r.warnings)
    assert isinstance(r, SuryaResult)
