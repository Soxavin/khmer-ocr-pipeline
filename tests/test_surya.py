from __future__ import annotations
from unittest.mock import MagicMock, patch
import numpy as np
from khmer_pipeline.models import PreprocessResult, SuryaResult, SuryaPageResult
from khmer_pipeline.surya import run_surya


def _make_preprocess_result(n_pages: int = 2) -> PreprocessResult:
    row = np.arange(100, dtype=np.uint8).reshape(1, 100)
    channel = np.tile(row, (100, 1))
    img = np.stack([channel, channel, channel], axis=2)
    return PreprocessResult(
        source_name="ardb.pdf",
        page_images=[img.copy() for _ in range(n_pages)],
        dpi=200,
        page_count=n_pages,
    )


def _make_text_line_mock(idx: int = 0) -> MagicMock:
    """Matches surya.recognition.schema.TextLine in the new Surya API."""
    line = MagicMock()
    line.text = f"ខ្មែរ {idx}"
    line.bbox = [10.0, 10.0, 200.0, 50.0]
    line.polygon = [[10.0, 10.0], [200.0, 10.0], [200.0, 50.0], [10.0, 50.0]]
    line.confidence = 0.95
    return line


def _make_layout_bbox_mock(label: str = "Text") -> MagicMock:
    """Matches surya.layout.schema.LayoutBox in the new Surya API."""
    b = MagicMock()
    b.label = label
    b.bbox = [10.0, 10.0, 200.0, 50.0]
    b.polygon = [[10.0, 10.0], [200.0, 10.0], [200.0, 50.0], [10.0, 50.0]]
    b.position = 0
    return b


def _make_cell_mock(bbox: list) -> MagicMock:
    cell = MagicMock()
    cell.bbox = bbox
    cell.model_dump.return_value = {
        "polygon": [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]],
        "confidence": 0.9,
        "bbox": bbox,
        "row_id": 0,
        "col_id": 0,
        "cell_id": 0,
        "text_lines": None,
    }
    return cell


def _make_predictors(with_table: bool = False):
    """Returns (layout_pred, rec_pred, table_pred) mocks for the new Surya API."""
    layout_bboxes = [_make_layout_bbox_mock("Text")]
    if with_table:
        table_bbox = _make_layout_bbox_mock("Table")
        table_bbox.bbox = [10.0, 60.0, 200.0, 150.0]
        layout_bboxes.append(table_bbox)

    layout_result = MagicMock()
    layout_result.bboxes = layout_bboxes
    layout_pred = MagicMock(return_value=[layout_result])

    ocr_result = MagicMock()
    ocr_result.text_lines = [_make_text_line_mock(0)]
    rec_pred = MagicMock(return_value=[ocr_result])

    if with_table:
        table_result = MagicMock()
        table_result.rows = []
        table_result.cols = []
        table_result.cells = []
        table_result.image_bbox = [0.0, 0.0, 190.0, 90.0]
        table_pred = MagicMock(return_value=[table_result])
    else:
        table_pred = MagicMock(return_value=[])

    return layout_pred, rec_pred, table_pred


# --- Contract tests ---

def test_run_surya_returns_surya_result():
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result())
    assert isinstance(r, SuryaResult)


def test_run_surya_preserves_source_name():
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result())
    assert r.source_name == "ardb.pdf"


def test_run_surya_page_count_matches():
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result(n_pages=3))
    assert len(r.pages) == 3


def test_run_surya_pages_are_surya_page_result():
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result())
    for page in r.pages:
        assert isinstance(page, SuryaPageResult)


def test_run_surya_page_index_is_zero_based():
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result(n_pages=2))
    assert r.pages[0].page_index == 0
    assert r.pages[1].page_index == 1


def test_run_surya_text_blocks_is_list_of_dicts():
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result())
    assert isinstance(r.pages[0].text_blocks, list)
    assert all(isinstance(b, dict) for b in r.pages[0].text_blocks)


def test_run_surya_block_has_required_keys():
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result())
    block = r.pages[0].text_blocks[0]
    for key in ("label", "bbox", "polygon", "reading_order"):
        assert key in block, f"Missing key: {key}"


def test_run_surya_ocr_text_is_str():
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result())
    assert isinstance(r.pages[0].ocr_text, str)


def test_run_surya_ocr_text_contains_khmer():
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result())
    assert "ខ្មែរ" in r.pages[0].ocr_text


def test_run_surya_no_tables_gives_empty_list():
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors(with_table=False)):
        r = run_surya(_make_preprocess_result())
    assert r.pages[0].tables == []


def test_run_surya_with_table_gives_non_empty_list():
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors(with_table=True)):
        r = run_surya(_make_preprocess_result())
    assert len(r.pages[0].tables) == 1


def test_run_surya_table_dict_has_required_keys():
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors(with_table=True)):
        r = run_surya(_make_preprocess_result())
    table = r.pages[0].tables[0]
    for key in ("rows", "cols", "cells", "image_bbox"):
        assert key in table, f"Missing table key: {key}"


def test_phantom_cells_outside_bbox_are_discarded():
    """Cells whose bbox falls entirely outside image_bbox are removed."""
    layout_bboxes = [_make_layout_bbox_mock("Text")]
    table_bbox = _make_layout_bbox_mock("Table")
    table_bbox.bbox = [10.0, 60.0, 200.0, 150.0]
    layout_bboxes.append(table_bbox)

    layout_result = MagicMock()
    layout_result.bboxes = layout_bboxes
    layout_pred = MagicMock(return_value=[layout_result])

    ocr_result = MagicMock()
    ocr_result.text_lines = [_make_text_line_mock(0)]
    rec_pred = MagicMock(return_value=[ocr_result])

    inside_cell = _make_cell_mock([10.0, 10.0, 50.0, 40.0])
    outside_cell = _make_cell_mock([-50.0, -50.0, -10.0, -10.0])

    table_result = MagicMock()
    table_result.rows = []
    table_result.cols = []
    table_result.cells = [inside_cell, outside_cell]
    table_result.image_bbox = [0.0, 0.0, 190.0, 90.0]
    table_pred = MagicMock(return_value=[table_result])

    with patch("khmer_pipeline.surya._get_predictors", return_value=(layout_pred, rec_pred, table_pred)):
        r = run_surya(_make_preprocess_result(n_pages=1))

    cells = r.pages[0].tables[0]["cells"]
    assert len(cells) == 1
    assert cells[0]["bbox"] == [10.0, 10.0, 50.0, 40.0]


def test_table_cells_get_ocr_text():
    """rec_pred must be called with cell bboxes and text_lines set on each cell."""
    layout_bboxes = [_make_layout_bbox_mock("Table")]
    layout_bboxes[0].bbox = [10.0, 60.0, 200.0, 150.0]
    layout_result = MagicMock()
    layout_result.bboxes = layout_bboxes
    layout_pred = MagicMock(return_value=[layout_result])

    cell = _make_cell_mock([5.0, 5.0, 40.0, 20.0])

    table_result = MagicMock()
    table_result.rows = []
    table_result.cols = []
    table_result.cells = [cell]
    table_result.image_bbox = [0.0, 0.0, 190.0, 90.0]
    table_pred = MagicMock(return_value=[table_result])

    ocr_result = MagicMock()
    ocr_result.text_lines = [_make_text_line_mock(0)]
    rec_pred = MagicMock(return_value=[ocr_result])

    with patch("khmer_pipeline.surya._get_predictors",
               return_value=(layout_pred, rec_pred, table_pred)):
        run_surya(_make_preprocess_result(n_pages=1))

    # rec_pred called once for cell OCR (no non-table bboxes on this page)
    assert rec_pred.call_count == 1
    assert cell.text_lines is not None
    assert cell.text_lines[0]["text"] == "ខ្មែរ 0"


def test_small_region_skipped():
    """Layout bbox smaller than 50×20 pixels produces no text blocks."""
    tiny_bbox = _make_layout_bbox_mock("Text")
    tiny_bbox.bbox = [10.0, 10.0, 40.0, 25.0]   # 30×15 — below both thresholds

    layout_result = MagicMock()
    layout_result.bboxes = [tiny_bbox]
    layout_pred = MagicMock(return_value=[layout_result])

    ocr_result = MagicMock()
    ocr_result.text_lines = [_make_text_line_mock(0)]
    rec_pred = MagicMock(return_value=[ocr_result])

    table_pred = MagicMock(return_value=[])

    with patch("khmer_pipeline.surya._get_predictors",
               return_value=(layout_pred, rec_pred, table_pred)):
        r = run_surya(_make_preprocess_result(n_pages=1))

    assert r.pages[0].text_blocks == []
    rec_pred.assert_not_called()


def test_region_label_in_text_blocks():
    """Every text block must have a 'region_label' key."""
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result(n_pages=1))
    for block in r.pages[0].text_blocks:
        assert "region_label" in block, f"Block missing region_label: {block}"


def test_ocr_text_has_no_region_labels():
    """ocr_text must be plain text — layout label names must not appear as prefixes."""
    with patch("khmer_pipeline.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result(n_pages=1))
    ocr_text = r.pages[0].ocr_text
    for label in ("Text:", "Table:", "Title:", "Figure:", "Caption:", "Picture:"):
        assert label not in ocr_text, f"ocr_text contains label prefix '{label}'"
