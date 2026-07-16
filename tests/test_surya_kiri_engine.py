"""Tests for the Surya + Kiri hybrid engine (surya_kiri_engine.py).

Covers registry wiring, contract compliance, raw-image recognition, and
fallthrough paths for pages with no tables. The layout predictor, TableRecPredictor,
and the Kiri recognizer are stubbed to avoid model downloads in CI.
"""
from __future__ import annotations

import os
from contextlib import ExitStack
from unittest.mock import MagicMock, patch
import numpy as np

from khmer_pipeline.models import PreprocessResult, SuryaResult, SuryaPageResult
from khmer_pipeline.engines.engine_registry import _OCR_ENGINES
from khmer_pipeline.engines.surya_kiri_engine import run_surya_kiri, _LOW_CONF_THRESHOLD


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
         confidence=1.0, preprocess=None, table_raises=False,
         structure=None, slanet_cells=None, crop_sink=None, slanet_raises=False):
    """Run the engine with all model-loading dependencies stubbed.

    `recognize` and `confidence` are either a scalar (applied to every pending
    cell) or a list (one entry per pending cell, in structure-cell order) —
    mirroring what `recognize_cells_conf` returns as (text, conf) pairs.
    `table_raises=True` makes the stubbed structure predictor raise, exercising
    the structure-prediction failure path.

    `structure` sets KHMER_KIRI_STRUCTURE for the run (None = unset → default).
    `slanet_cells` stubs `predict_cells` (the slanet path). `crop_sink`, when a
    list, collects the crops handed to the recognizer (to assert span crops).
    """
    base = base or _base_result()
    cells = _fake_table_rec_cells(2, 2) if cells is None else cells
    env = dict(os.environ)
    env.pop("KHMER_KIRI_STRUCTURE", None)
    if structure is not None:
        env["KHMER_KIRI_STRUCTURE"] = structure
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, env, clear=True))
        p = lambda tgt, **kw: stack.enter_context(patch(f"khmer_pipeline.engines.surya_kiri_engine.{tgt}", **kw))
        p("run_surya", return_value=base)
        p("get_manager")
        p("_get_predictors", return_value=(_fake_layout_pred(with_table), None))

        def _rec(crops, warning_sink=None):
            if crop_sink is not None:
                crop_sink.extend(crops)
            if isinstance(recognize, list):
                confs = confidence if isinstance(confidence, list) else [confidence] * len(recognize)
                return list(zip(recognize, confs))
            return [(recognize, confidence)] * len(crops)

        p("recognize_cells_conf", side_effect=_rec)
        if table_raises or slanet_raises:
            p("predict_cells", side_effect=RuntimeError("slanet boom"))
        else:
            p("predict_cells", return_value=slanet_cells if slanet_cells is not None else [])
        mock_tbl = MagicMock()
        if table_raises:
            mock_tbl.side_effect = RuntimeError("tablerec boom")
        else:
            mock_result = MagicMock()
            mock_result.cells = cells
            mock_tbl.return_value = [mock_result]
        tbl_cls = stack.enter_context(patch("surya.table_rec.TableRecPredictor", return_value=mock_tbl))
        result = run_surya_kiri(preprocess or _preprocess())
        result._tablerec_instantiated = tbl_cls.called  # test-only probe
        return result


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
# A2 (warning half): dropping a table region must surface a warning, never be
# silent — the analyst otherwise sees "no table" on a page that has one.
# ---------------------------------------------------------------------------

def test_table_structure_failure_warns():
    """A TableRecPredictor exception drops the region but adds a warning."""
    r = _run(table_raises=True)
    assert any("table omitted" in w and "region" in w for w in r.warnings)


def test_empty_cells_warns():
    """Zero cells for a region drops it but adds a warning (not silent)."""
    r = _run(cells=[])
    assert any("table omitted" in w and "region" in w for w in r.warnings)


# ---------------------------------------------------------------------------
# Raw-image usage
# ---------------------------------------------------------------------------

def test_falls_back_to_page_images_without_raw():
    """When recognition_page_images is None, the engine still runs (uses page_images)."""
    r = _run(preprocess=_preprocess(with_raw=False))
    assert isinstance(r, SuryaResult)
    assert len(r.pages[0].tables) == 1


def test_recognition_images_none_adds_fallback_warning():
    """B5: the geometric-image fallback is a measured accuracy loss — it must warn."""
    r = _run(preprocess=_preprocess(with_raw=False))
    assert any("recognition images unavailable" in w for w in r.warnings)


def test_recognition_images_present_no_fallback_warning():
    r = _run(preprocess=_preprocess(with_raw=True))
    assert not any("recognition images unavailable" in w for w in r.warnings)


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


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def test_cells_carry_confidence():
    """Every table cell gets a `confidence` key sourced from recognize_cells_conf."""
    table = _run(recognize=["A", "B", "C", "D"], confidence=0.95).pages[0].tables[0]
    assert len(table["cells"]) == 4
    assert all(cell["confidence"] == 0.95 for cell in table["cells"])


def test_low_confidence_cells_trigger_warning():
    """When confidences are below _LOW_CONF_THRESHOLD, a page-level warning is added."""
    r = _run(recognize=["A", "B", "C", "D"], confidence=_LOW_CONF_THRESHOLD - 0.1)
    assert any("below" in w and "confidence" in w for w in r.warnings)


def test_high_confidence_cells_no_warning():
    """When all confidences are above _LOW_CONF_THRESHOLD, no confidence warning is added."""
    r = _run(recognize=["A", "B", "C", "D"], confidence=_LOW_CONF_THRESHOLD + 0.1)
    assert not any("confidence" in w for w in r.warnings)


# ---------------------------------------------------------------------------
# SLANet structure path (KHMER_KIRI_STRUCTURE=slanet): column-spanning cells
# must be recognized as ONE crop covering the full span, anchored at their
# (row_id, col_id) — TableRec's row×col intersections split spanned text.
# ---------------------------------------------------------------------------

# crop-relative SLANet cells: row 0 = one cell spanning cols 0-1 (wide bbox),
# row 1 = two unit cells.
_SLANET_SPAN_CELLS = [
    {"row_id": 0, "col_id": 0, "row_span": 1, "col_span": 2, "bbox": [5, 5, 140, 25]},
    {"row_id": 1, "col_id": 0, "row_span": 1, "col_span": 1, "bbox": [5, 35, 60, 55]},
    {"row_id": 1, "col_id": 1, "row_span": 1, "col_span": 1, "bbox": [70, 35, 140, 55]},
]


def test_slanet_spanning_cell_is_one_crop_at_anchor():
    crops: list = []
    r = _run(structure="slanet", slanet_cells=_SLANET_SPAN_CELLS,
             recognize=["14-06-26", "A", "B"], crop_sink=crops)
    # one crop per SLANet cell — the spanning cell is NOT split into unit crops
    assert len(crops) == 3
    assert crops[0].shape[1] == 135  # full span width (140-5), not a unit slice
    from khmer_pipeline.evaluation.evaluate_structure import pred_table_grid
    grid = pred_table_grid(r.pages[0].tables[0])
    assert grid[0][0] == "14-06-26"   # complete text at the anchor
    assert grid[0][1] == ""           # spanned-over column padded empty
    assert grid[1] == ["A", "B"]


def test_slanet_path_never_instantiates_tablerec():
    r = _run(structure="slanet", slanet_cells=_SLANET_SPAN_CELLS, recognize="x")
    assert r._tablerec_instantiated is False


def test_default_structure_is_tablerec():
    # §2.40 postscript: `merged` passed the eval gate but produced a data-cell
    # false merge in production UI use (ID merged into product name), so the
    # default reverted to pure TableRec — data integrity > header cosmetics.
    # `merged` stays opt-in via KHMER_KIRI_STRUCTURE.
    import khmer_pipeline.engines.surya_kiri_engine as ske
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("KHMER_KIRI_STRUCTURE", None)
        assert ske._kiri_structure() == "tablerec"
    r = _run(recognize=["A", "B", "C", "D"])  # no flag → pure TableRec grid
    assert r._tablerec_instantiated is True
    from khmer_pipeline.evaluation.evaluate_structure import pred_table_grid
    assert pred_table_grid(r.pages[0].tables[0]) == [["A", "B"], ["C", "D"]]


def test_tablerec_optout_skips_span_merging():
    # Explicit tablerec = pre-§2.40 behaviour: SLANet proposals ignored entirely.
    slanet = [{"row_id": 0, "col_id": 0, "row_span": 1, "col_span": 2, "bbox": [0, 0, 145, 32]}]
    r = _run(structure="tablerec", slanet_cells=slanet, recognize=["A", "B", "C", "D"])
    from khmer_pipeline.evaluation.evaluate_structure import pred_table_grid
    assert pred_table_grid(r.pages[0].tables[0]) == [["A", "B"], ["C", "D"]]


def test_slanet_cells_carry_span_metadata():
    r = _run(structure="slanet", slanet_cells=_SLANET_SPAN_CELLS, recognize="x")
    cells = r.pages[0].tables[0]["cells"]
    anchor = next(c for c in cells if c["row_id"] == 0 and c["col_id"] == 0)
    unit = next(c for c in cells if c["row_id"] == 1 and c["col_id"] == 0)
    assert anchor["col_span"] == 2
    assert "col_span" not in unit  # unit cells stay shape-unchanged


def test_slanet_size_guard_blanks_tiny_cells():
    tiny = [{"row_id": 0, "col_id": 0, "row_span": 1, "col_span": 1, "bbox": [5, 5, 7, 7]},
            {"row_id": 0, "col_id": 1, "row_span": 1, "col_span": 1, "bbox": [10, 5, 60, 25]}]
    crops: list = []
    r = _run(structure="slanet", slanet_cells=tiny, recognize="X", crop_sink=crops)
    assert len(crops) == 1  # tiny cell never reaches the recognizer
    cells = r.pages[0].tables[0]["cells"]
    tiny_cell = next(c for c in cells if c["col_id"] == 0)
    assert tiny_cell["text_lines"] == [] and tiny_cell["confidence"] == 1.0


def test_slanet_empty_cells_warns_and_keeps_page():
    base = _base_result()
    r = _run(base=base, structure="slanet", slanet_cells=[])
    assert r.pages[0].tables == base.pages[0].tables
    assert any("table omitted" in w for w in r.warnings)


def test_slanet_structure_failure_warns():
    r = _run(structure="slanet", table_raises=True)
    assert any("table omitted" in w and "region" in w for w in r.warnings)


# ---------------------------------------------------------------------------
# Merged mode (KHMER_KIRI_STRUCTURE=merged): TableRec's trusted unit grid +
# SLANet used ONLY as a span detector. Unit cells whose centers fall inside a
# SLANet span are merged into one union crop; everything else is pure TableRec.
# (Probe finding: pure slanet fixes spans but degrades the data grid — merged
# keeps TableRec data fidelity AND unsplit span text.)
# ---------------------------------------------------------------------------

def test_merge_spans_units_inside_span_become_one_record():
    from khmer_pipeline.engines.surya_kiri_engine import _merge_spans
    units = [
        (0, 0, [5, 5, 60, 25], 1, 1),
        (0, 1, [70, 5, 140, 25], 1, 1),
        (1, 0, [5, 35, 60, 55], 1, 1),
        (1, 1, [70, 35, 140, 55], 1, 1),
    ]
    # SLANet span covering the two row-0 unit centers
    out = _merge_spans(units, [[0, 0, 145, 30]])
    assert len(out) == 3
    merged = [r for r in out if r[3] > 1 or r[4] > 1]
    assert merged == [(0, 0, [5, 5, 140, 25], 1, 2)]  # union bbox, anchored, col_span=2
    assert (1, 0, [5, 35, 60, 55], 1, 1) in out and (1, 1, [70, 35, 140, 55], 1, 1) in out


def test_merge_spans_single_covered_unit_is_no_op():
    from khmer_pipeline.engines.surya_kiri_engine import _merge_spans
    units = [(0, 0, [5, 5, 60, 25], 1, 1), (0, 1, [70, 5, 140, 25], 1, 1)]
    out = _merge_spans(units, [[0, 0, 65, 30]])  # covers only unit (0,0)'s center
    assert sorted(out) == sorted(units)


def test_merge_spans_unit_consumed_by_one_span_only():
    from khmer_pipeline.engines.surya_kiri_engine import _merge_spans
    units = [(0, 0, [0, 0, 40, 20], 1, 1), (0, 1, [50, 0, 90, 20], 1, 1),
             (0, 2, [100, 0, 140, 20], 1, 1)]
    # two overlapping spans both covering unit (0,1); first wins, second then
    # covers only one free unit → no second merge
    out = _merge_spans(units, [[0, 0, 95, 25], [45, 0, 145, 25]])
    merged = [r for r in out if r[4] > 1]
    assert merged == [(0, 0, [0, 0, 90, 20], 1, 2)]
    assert (0, 2, [100, 0, 140, 20], 1, 1) in out


def test_merge_spans_rejects_multi_row_block_spans():
    # Measured on ARDB p1: SLANet block spans (row_span×col_span) over the sparse
    # data region consumed real data cells (col-4 digit rows 22→16). Only
    # same-row merges are legitimate here — block spans must be ignored.
    from khmer_pipeline.engines.surya_kiri_engine import _merge_spans
    units = [(0, 0, [0, 0, 40, 20], 1, 1), (0, 1, [50, 0, 90, 20], 1, 1),
             (1, 0, [0, 30, 40, 50], 1, 1), (1, 1, [50, 30, 90, 50], 1, 1)]
    out = _merge_spans(units, [[0, 0, 95, 55]])  # covers the whole 2x2 block
    assert sorted(out) == sorted(units)  # untouched


def test_merge_spans_rejects_non_consecutive_columns():
    from khmer_pipeline.engines.surya_kiri_engine import _merge_spans
    units = [(0, 0, [0, 0, 40, 20], 1, 1), (0, 2, [100, 0, 140, 20], 1, 1)]
    out = _merge_spans(units, [[0, 0, 145, 25]])  # covers cols 0 and 2, gap at 1
    assert sorted(out) == sorted(units)


def test_merge_spans_requires_substantial_coverage_of_each_unit():
    # Each covered unit needs ≥60% of its width inside the candidate box
    # (measured on ARDB p1: real spans cover their units 92–105%; a drifted
    # box covering a neighbour 33% must NOT capture it).
    from khmer_pipeline.engines.surya_kiri_engine import _merge_spans
    units = [(0, 0, [0, 0, 40, 20], 1, 1), (0, 1, [40, 0, 100, 20], 1, 1)]
    # box ends at x=79: unit (0,1) x-overlap = 39/60 = 65% ≥ 60% → merge
    out = _merge_spans(units, [[0, 0, 79, 25]])
    assert [r for r in out if r[4] > 1] == [(0, 0, [0, 0, 100, 20], 1, 2)]
    # box ends at x=64: unit (0,1) x-overlap = 40% < 60% → no merge
    out = _merge_spans(units, [[0, 0, 64, 25]])
    assert sorted(out) == sorted(units)


def test_has_vertical_separator_detects_full_height_line():
    from khmer_pipeline.engines.surya_kiri_engine import _has_vertical_separator
    crop = np.full((100, 200, 3), 255, dtype=np.uint8)  # white
    crop[:, 99:101] = 0  # full-height dark rule at x≈100
    a, b = [10, 10, 100, 90], [100, 10, 190, 90]
    assert _has_vertical_separator(crop, a, b) is True


def test_has_vertical_separator_ignores_sparse_text_strokes():
    from khmer_pipeline.engines.surya_kiri_engine import _has_vertical_separator
    crop = np.full((100, 200, 3), 255, dtype=np.uint8)
    crop[40:55, 99:101] = 0  # short stroke crossing the boundary (15% of height)
    a, b = [10, 10, 100, 90], [100, 10, 190, 90]
    assert _has_vertical_separator(crop, a, b) is False


def test_has_vertical_separator_blank_boundary_is_open():
    from khmer_pipeline.engines.surya_kiri_engine import _has_vertical_separator
    crop = np.full((100, 200, 3), 255, dtype=np.uint8)
    a, b = [10, 10, 100, 90], [100, 10, 190, 90]
    assert _has_vertical_separator(crop, a, b) is False


def test_has_vertical_separator_unmeasurable_band_blocks_merge():
    # Too little shared height to measure → no positive evidence of openness →
    # treat as separated (conservative: false merges eat data cells).
    from khmer_pipeline.engines.surya_kiri_engine import _has_vertical_separator
    crop = np.full((100, 200, 3), 255, dtype=np.uint8)
    a, b = [10, 10, 100, 15], [100, 10, 190, 15]  # 5px band < 8px minimum
    assert _has_vertical_separator(crop, a, b) is True


def test_has_vertical_separator_scans_whole_gap_between_text_tight_units():
    # TableRec data-row bboxes hug the text, so the real gridline sits anywhere
    # in the (wide) whitespace between them — a thin midpoint strip misses it.
    from khmer_pipeline.engines.surya_kiri_engine import _has_vertical_separator
    crop = np.full((100, 200, 3), 255, dtype=np.uint8)
    crop[:, 89:91] = 0  # rule at x≈90, far from the midpoint (105) of the gap
    a, b = [10, 10, 80, 90], [130, 10, 190, 90]
    assert _has_vertical_separator(crop, a, b) is True


def test_merge_spans_rejects_pairs_split_by_a_gridline():
    # SLANet grid drift makes its unit boxes straddle two TableRec columns —
    # geometrically identical to a real span. The pixel evidence (a vertical
    # rule/gap between the units) is what tells them apart, document-generally.
    from khmer_pipeline.engines.surya_kiri_engine import _merge_spans
    crop = np.full((60, 200, 3), 255, dtype=np.uint8)
    crop[:, 99:101] = 0  # rule between the two units
    units = [(0, 0, [10, 5, 100, 55], 1, 1), (0, 1, [100, 5, 190, 55], 1, 1)]
    out = _merge_spans(units, [[0, 0, 195, 60]], crop=crop)
    assert sorted(out) == sorted(units)  # separator present → no merge
    crop_open = np.full((60, 200, 3), 255, dtype=np.uint8)
    out = _merge_spans(units, [[0, 0, 195, 60]], crop=crop_open)
    assert [r for r in out if r[4] > 1] == [(0, 0, [10, 5, 190, 55], 1, 2)]


def test_merged_mode_uses_physical_boxes_not_logical_spans():
    # ARDB p1 measured case: SLANet lost a column, so the 2nd date header came
    # back as a LOGICAL unit (col_span=1) whose PHYSICAL box still covers two
    # TableRec columns (100%/92%). Merging must key on the box geometry, not on
    # SLANet's col_span flag.
    slanet_unit = [{"row_id": 0, "col_id": 0, "row_span": 1, "col_span": 1,
                    "bbox": [0, 0, 145, 32]}]  # covers both fake row-0 units fully
    crops: list = []
    r = _run(structure="merged", slanet_cells=slanet_unit,
             recognize=["A", "B", "15-06-26"], crop_sink=crops)
    assert len(crops) == 3
    from khmer_pipeline.evaluation.evaluate_structure import pred_table_grid
    grid = pred_table_grid(r.pages[0].tables[0])
    assert grid[0] == ["15-06-26", ""]
    assert grid[1] == ["A", "B"]


def test_merged_mode_span_text_at_anchor_data_grid_intact():
    # TableRec 2x2 units; SLANet reports a span over row-0 cols 0-1. The union
    # crop is recognized once; row-1 data cells stay pure TableRec.
    slanet = [{"row_id": 0, "col_id": 0, "row_span": 1, "col_span": 2, "bbox": [0, 0, 145, 32]}]
    crops: list = []
    r = _run(structure="merged", slanet_cells=slanet,
             recognize=["A", "B", "14-06-26"],  # pending order: kept units then merged
             crop_sink=crops)
    assert len(crops) == 3  # 2 kept units + 1 union (not 4 unit crops)
    from khmer_pipeline.evaluation.evaluate_structure import pred_table_grid
    grid = pred_table_grid(r.pages[0].tables[0])
    assert grid[0] == ["14-06-26", ""]
    assert grid[1] == ["A", "B"]
    anchor = next(c for c in r.pages[0].tables[0]["cells"] if c["row_id"] == 0 and c["col_id"] == 0)
    assert anchor["col_span"] == 2


def test_merged_mode_slanet_failure_falls_back_to_tablerec_with_warning():
    # SLANet breaking must NOT drop the table (TableRec worked) — proceed
    # unmerged and surface a warning.
    r = _run(structure="merged", recognize=["A", "B", "C", "D"], slanet_raises=True)
    from khmer_pipeline.evaluation.evaluate_structure import pred_table_grid
    assert pred_table_grid(r.pages[0].tables[0]) == [["A", "B"], ["C", "D"]]
    assert any("span detection failed" in w for w in r.warnings)


# ---------------------------------------------------------------------------
# A3: Kiri recognizer failure visibility + per-run latch reset
# ---------------------------------------------------------------------------

def test_run_surya_kiri_resets_failure_latch():
    """Each run clears the within-run load-failure latch so a transient first
    failure doesn't disable Kiri for the whole (long-lived) process."""
    with patch("khmer_pipeline.engines.surya_kiri_engine.reset_kiri_failure") as mock_reset:
        _run()
    mock_reset.assert_called_once()


def test_kiri_unavailable_warning_reaches_result():
    """When the recognizer is unavailable it appends to the sink; the engine must
    merge that into SuryaResult.warnings (deduped)."""
    def _fail(crops, warning_sink=None):
        if warning_sink is not None:
            warning_sink.append("Kiri recognizer unavailable: boom — table cells left empty")
        return [("", 0.0)] * len(crops)

    base = _base_result()
    with ExitStack() as stack:
        p = lambda tgt, **kw: stack.enter_context(patch(f"khmer_pipeline.engines.surya_kiri_engine.{tgt}", **kw))
        p("run_surya", return_value=base)
        p("get_manager")
        p("_get_predictors", return_value=(_fake_layout_pred(True), None))
        p("recognize_cells_conf", side_effect=_fail)
        mock_tbl = MagicMock()
        mock_result = MagicMock()
        mock_result.cells = _fake_table_rec_cells(2, 2)
        mock_tbl.return_value = [mock_result]
        stack.enter_context(patch("surya.table_rec.TableRecPredictor", return_value=mock_tbl))
        r = run_surya_kiri(_preprocess())

    unavailable = [w for w in r.warnings if "unavailable" in w]
    assert len(unavailable) == 1  # merged/deduped to a single warning
