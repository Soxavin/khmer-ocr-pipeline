from __future__ import annotations
from unittest.mock import MagicMock, patch
import numpy as np
from khmer_pipeline.models import PreprocessResult, SuryaResult, SuryaPageResult
from khmer_pipeline.engines.surya import run_surya, _process_page, _parse_html_table, _find_matching_html


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


def _make_block_mock(idx: int = 0, label: str = "Text") -> MagicMock:
    """Matches surya.recognition.schema.BlockOCRResult in Surya 0.20."""
    block = MagicMock()
    block.skipped = False
    block.error = False
    block.html = f"<p>ខ្មែរ {idx}</p>"
    block.bbox = [10.0, 10.0, 200.0, 50.0]
    block.polygon = [[10.0, 10.0], [200.0, 10.0], [200.0, 50.0], [10.0, 50.0]]
    block.confidence = 0.95
    block.label = label
    block.raw_label = label
    block.reading_order = idx
    return block


def _make_layout_bbox_mock(label: str = "Text") -> MagicMock:
    """Matches surya.layout.schema.LayoutBox in Surya 0.20."""
    b = MagicMock()
    b.label = label
    b.bbox = [10.0, 10.0, 200.0, 50.0]
    b.polygon = [[10.0, 10.0], [200.0, 10.0], [200.0, 50.0], [10.0, 50.0]]
    b.position = 0

    def _model_copy(update=None):
        # Mimic pydantic's model_copy(update=...): a fresh box with fields applied,
        # so label/bbox reflect the Track-A replacement rather than staying MagicMocks.
        nb = _make_layout_bbox_mock(b.label)
        nb.bbox, nb.polygon, nb.position = b.bbox, b.polygon, b.position
        for k, v in (update or {}).items():
            setattr(nb, k, v)
        return nb

    b.model_copy = _model_copy
    return b


def _make_predictors(with_table: bool = False):
    # Returns (layout_pred, rec_pred) mocks for Surya 0.20 API.
    layout_bboxes = [_make_layout_bbox_mock("Text")]
    if with_table:
        table_bbox = _make_layout_bbox_mock("Table")
        table_bbox.bbox = [10.0, 60.0, 200.0, 150.0]
        layout_bboxes.append(table_bbox)

    layout_result = MagicMock()
    layout_result.error = False
    layout_result.bboxes = layout_bboxes
    layout_pred = MagicMock(return_value=[layout_result])

    page_ocr = MagicMock()
    blocks = [_make_block_mock(0)]
    if with_table:
        # Table HTML block: label="Table", bbox matches layout table bbox
        table_block = _make_block_mock(0, label="Table")
        table_block.bbox = [10.0, 60.0, 200.0, 150.0]
        table_block.html = "<table><tr><td>ខ្មែរ cell</td></tr></table>"
        blocks.append(table_block)
    page_ocr.blocks = blocks
    rec_pred = MagicMock(return_value=[page_ocr])

    return layout_pred, rec_pred


# --- Contract tests ---

def test_run_surya_returns_surya_result():
    with patch("khmer_pipeline.engines.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result())
    assert isinstance(r, SuryaResult)


def test_run_surya_preserves_source_name():
    with patch("khmer_pipeline.engines.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result())
    assert r.source_name == "ardb.pdf"


def test_run_surya_page_count_matches():
    with patch("khmer_pipeline.engines.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result(n_pages=3))
    assert len(r.pages) == 3


def test_run_surya_pages_are_surya_page_result():
    with patch("khmer_pipeline.engines.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result())
    for page in r.pages:
        assert isinstance(page, SuryaPageResult)


def test_run_surya_page_index_is_zero_based():
    with patch("khmer_pipeline.engines.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result(n_pages=2))
    assert r.pages[0].page_index == 0
    assert r.pages[1].page_index == 1


def test_run_surya_text_blocks_is_list_of_dicts():
    with patch("khmer_pipeline.engines.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result())
    assert isinstance(r.pages[0].text_blocks, list)
    assert all(isinstance(b, dict) for b in r.pages[0].text_blocks)


def test_run_surya_block_has_required_keys():
    with patch("khmer_pipeline.engines.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result())
    block = r.pages[0].text_blocks[0]
    for key in ("label", "bbox", "polygon", "reading_order"):
        assert key in block, f"Missing key: {key}"


def test_run_surya_ocr_text_is_str():
    with patch("khmer_pipeline.engines.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result())
    assert isinstance(r.pages[0].ocr_text, str)


def test_run_surya_ocr_text_contains_khmer():
    with patch("khmer_pipeline.engines.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result())
    assert "ខ្មែរ" in r.pages[0].ocr_text


def test_run_surya_no_tables_gives_empty_list():
    with patch("khmer_pipeline.engines.surya._get_predictors", return_value=_make_predictors(with_table=False)):
        r = run_surya(_make_preprocess_result())
    assert r.pages[0].tables == []


def test_run_surya_with_table_gives_non_empty_list():
    with patch("khmer_pipeline.engines.surya._get_predictors", return_value=_make_predictors(with_table=True)):
        r = run_surya(_make_preprocess_result())
    assert len(r.pages[0].tables) == 1


def test_run_surya_table_dict_has_required_keys():
    with patch("khmer_pipeline.engines.surya._get_predictors", return_value=_make_predictors(with_table=True)):
        r = run_surya(_make_preprocess_result())
    table = r.pages[0].tables[0]
    for key in ("rows", "cols", "cells", "image_bbox", "bbox"):
        assert key in table, f"Missing table key: {key}"


def test_table_bbox_is_page_space_layout_bbox():
    # tbl['bbox'] must be the page-space bbox of the Table layout region.
    with patch("khmer_pipeline.engines.surya._get_predictors", return_value=_make_predictors(with_table=True)):
        r = run_surya(_make_preprocess_result(n_pages=1))
    table = r.pages[0].tables[0]
    assert table["bbox"] == [10.0, 60.0, 200.0, 150.0]


def test_table_cells_get_ocr_text():
    # Cell text is parsed from the Table block's HTML output — no table_pred.
    TABLE_BBOX = [10.0, 60.0, 200.0, 150.0]

    layout_bboxes = [_make_layout_bbox_mock("Table")]
    layout_bboxes[0].bbox = TABLE_BBOX
    layout_result = MagicMock()
    layout_result.error = False
    layout_result.bboxes = layout_bboxes
    layout_pred = MagicMock(return_value=[layout_result])

    table_block = _make_block_mock(0, label="Table")
    table_block.bbox = TABLE_BBOX
    table_block.html = "<table><tr><td>ខ្មែរ</td></tr></table>"
    page_ocr = MagicMock()
    page_ocr.blocks = [table_block]

    rec_pred = MagicMock(return_value=[page_ocr])

    with patch("khmer_pipeline.engines.surya._get_predictors",
               return_value=(layout_pred, rec_pred)):
        r = run_surya(_make_preprocess_result(n_pages=1))

    table = r.pages[0].tables[0]
    assert table["cells"][0]["text_lines"]
    assert "ខ្មែរ" in table["cells"][0]["text_lines"][0]["text"]


def test_region_label_in_text_blocks():
    """Every text block must have a 'region_label' key."""
    with patch("khmer_pipeline.engines.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result(n_pages=1))
    assert r.pages[0].text_blocks, "Expected at least one text block"
    for block in r.pages[0].text_blocks:
        assert "region_label" in block, f"Block missing region_label: {block}"


def test_ocr_text_has_no_region_labels():
    """ocr_text must be plain text — layout label names must not appear as prefixes."""
    with patch("khmer_pipeline.engines.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result(n_pages=1))
    ocr_text = r.pages[0].ocr_text
    for label in ("Text:", "Table:", "Title:", "Figure:", "Caption:", "Picture:"):
        assert label not in ocr_text, f"ocr_text contains label prefix '{label}'"


def test_per_region_ocr_batched_in_single_call():
    """All text regions on a page are OCR'd in a single rec_pred call via layout_results."""
    bbox1 = _make_layout_bbox_mock("Text")
    bbox1.bbox = [10.0, 10.0, 200.0, 50.0]
    bbox1.position = 0

    bbox2 = _make_layout_bbox_mock("Text")
    bbox2.bbox = [10.0, 100.0, 200.0, 150.0]
    bbox2.position = 1

    layout_result = MagicMock()
    layout_result.error = False
    layout_result.bboxes = [bbox1, bbox2]
    layout_pred = MagicMock(return_value=[layout_result])

    block1 = _make_block_mock(0)
    block1.html = "<p>ខ្មែរ first</p>"
    block1.reading_order = 0

    block2 = _make_block_mock(1)
    block2.html = "<p>ខ្មែរ second</p>"
    block2.reading_order = 1

    page_ocr = MagicMock()
    page_ocr.blocks = [block1, block2]
    rec_pred = MagicMock(return_value=[page_ocr])

    with patch("khmer_pipeline.engines.surya._get_predictors",
               return_value=(layout_pred, rec_pred)):
        r = run_surya(_make_preprocess_result(n_pages=1))

    # One call with layout_results (not per-region bbox batching)
    assert rec_pred.call_count == 1
    call_kwargs = rec_pred.call_args[1]
    assert "layout_results" in call_kwargs

    texts = [b["text"] for b in r.pages[0].text_blocks]
    assert "ខ្មែរ first" in texts
    assert "ខ្មែរ second" in texts


def test_multiple_tables_built_from_html():
    # Two Table layout regions + two matching Table HTML blocks → two tables in output.
    table_bbox1 = _make_layout_bbox_mock("Table")
    table_bbox1.bbox = [10.0, 60.0, 200.0, 150.0]

    table_bbox2 = _make_layout_bbox_mock("Table")
    table_bbox2.bbox = [10.0, 200.0, 300.0, 400.0]

    layout_result = MagicMock()
    layout_result.error = False
    layout_result.bboxes = [table_bbox1, table_bbox2]
    layout_pred = MagicMock(return_value=[layout_result])

    table_block1 = _make_block_mock(0, label="Table")
    table_block1.bbox = [10.0, 60.0, 200.0, 150.0]
    table_block1.html = "<table><tr><td>first</td></tr></table>"

    table_block2 = _make_block_mock(1, label="Table")
    table_block2.bbox = [10.0, 200.0, 300.0, 400.0]
    table_block2.html = "<table><tr><td>second</td></tr></table>"

    page_ocr = MagicMock()
    page_ocr.blocks = [table_block1, table_block2]
    rec_pred = MagicMock(return_value=[page_ocr])

    # Disable table stitching: this test exercises HTML→table building, not the
    # layout-region merge (these two stacked regions would otherwise be merged).
    with patch("khmer_pipeline.engines.surya._get_predictors",
               return_value=(layout_pred, rec_pred)), \
         patch("khmer_pipeline.engines.surya._stitch_enabled", return_value=False):
        r = run_surya(_make_preprocess_result(n_pages=1))

    assert len(r.pages[0].tables) == 2


def test_duplicate_html_block_assignment_warns():
    """One recognition-HTML block matched to two layout tables emits a warning
    (duplicated-rows risk); the assignment algorithm itself is unchanged."""
    b1 = _make_layout_bbox_mock("Table")
    b1.bbox = [10.0, 60.0, 200.0, 150.0]
    b2 = _make_layout_bbox_mock("Table")
    b2.bbox = [12.0, 61.0, 201.0, 151.0]  # within tolerance of the same HTML key
    layout_result = MagicMock()
    layout_result.error = False
    layout_result.bboxes = [b1, b2]
    layout_pred = MagicMock(return_value=[layout_result])

    table_block = _make_block_mock(0, label="Table")
    table_block.bbox = [10.0, 60.0, 200.0, 150.0]
    table_block.html = "<table><tr><td>x</td></tr></table>"
    page_ocr = MagicMock()
    page_ocr.blocks = [table_block]
    rec_pred = MagicMock(return_value=[page_ocr])

    with patch("khmer_pipeline.engines.surya._get_predictors",
               return_value=(layout_pred, rec_pred)), \
         patch("khmer_pipeline.engines.surya._stitch_enabled", return_value=False):
        r = run_surya(_make_preprocess_result(n_pages=1))

    assert any("reused" in w for w in r.warnings)


def test_run_surya_warnings_empty_when_no_issues():
    with patch("khmer_pipeline.engines.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result(n_pages=1))
    assert r.warnings == []


def _make_low_confidence_block_mock(idx: int = 0) -> MagicMock:
    block = _make_block_mock(idx)
    block.confidence = 0.3  # below CONFIDENCE_LOW (0.5)
    return block


def test_low_confidence_block_emits_warning():
    """A text block with confidence below CONFIDENCE_LOW adds a warning to SuryaResult.warnings."""
    layout_bboxes = [_make_layout_bbox_mock("Text")]
    layout_result = MagicMock()
    layout_result.error = False
    layout_result.bboxes = layout_bboxes
    layout_pred = MagicMock(return_value=[layout_result])

    page_ocr = MagicMock()
    page_ocr.blocks = [_make_low_confidence_block_mock(0)]
    rec_pred = MagicMock(return_value=[page_ocr])

    with patch("khmer_pipeline.engines.surya._get_predictors",
               return_value=(layout_pred, rec_pred)):
        r = run_surya(_make_preprocess_result(n_pages=1))

    assert any("low OCR confidence" in w for w in r.warnings)


def test_high_confidence_blocks_emit_no_warning():
    with patch("khmer_pipeline.engines.surya._get_predictors", return_value=_make_predictors()):
        r = run_surya(_make_preprocess_result(n_pages=1))
    assert not any("low OCR confidence" in w for w in r.warnings)


def test_table_html_parser_colspan_padding():
    """colspan cells pad the row so col_id indices stay aligned."""
    html = (
        '<table>'
        '<tr><th colspan="2">Header</th><th>C</th></tr>'
        '<tr><td>A</td><td>B</td><td>C</td></tr>'
        '</table>'
    )
    grid = _parse_html_table(html)
    assert grid[(0, 0)] == "Header"
    assert grid[(0, 1)] == ""       # padded spanned slot
    assert grid[(0, 2)] == "C"
    assert grid[(1, 0)] == "A"
    assert grid[(1, 1)] == "B"
    assert grid[(1, 2)] == "C"


def test_parse_html_table_with_spans_exposes_colspan_anchors():
    """The spans variant returns the same padded grid PLUS {(row,col): colspan}
    for spanning anchors — needed by surya_kiri_vlm to crop a spanned cell as
    one union region."""
    from khmer_pipeline.engines.surya import _parse_html_table_with_spans
    html = (
        '<table>'
        '<tr><th colspan="2">Header</th><th>C</th></tr>'
        '<tr><td>A</td><td>B</td><td>C</td></tr>'
        '</table>'
    )
    grid, spans = _parse_html_table_with_spans(html)
    assert grid == _parse_html_table(html)  # grid identical to the plain parse
    assert spans == {(0, 0): 2}


def test_parse_html_table_with_spans_empty_when_no_colspan():
    from khmer_pipeline.engines.surya import _parse_html_table_with_spans
    html = '<table><tr><td>A</td><td>B</td></tr></table>'
    grid, spans = _parse_html_table_with_spans(html)
    assert spans == {}
    assert grid[(0, 1)] == "B"


def test_vlm_tables_carry_col_span_on_spanning_anchors():
    """Plain Surya's built tables carry optional col_span on spanning anchors
    (from the VLM's colspan attribute) so span-aware consumers need no re-parse."""
    TABLE_BBOX = [10.0, 60.0, 200.0, 150.0]
    layout_bboxes = [_make_layout_bbox_mock("Table")]
    layout_bboxes[0].bbox = TABLE_BBOX
    layout_result = MagicMock()
    layout_result.error = False
    layout_result.bboxes = layout_bboxes
    layout_pred = MagicMock(return_value=[layout_result])

    table_block = _make_block_mock(0, label="Table")
    table_block.bbox = TABLE_BBOX
    table_block.html = ('<table><tr><td colspan="2">14-06-26</td><td>x</td></tr>'
                        '<tr><td>a</td><td>b</td><td>c</td></tr></table>')
    page_ocr = MagicMock()
    page_ocr.blocks = [table_block]
    rec_pred = MagicMock(return_value=[page_ocr])

    with patch("khmer_pipeline.engines.surya._get_predictors",
               return_value=(layout_pred, rec_pred)):
        r = run_surya(_make_preprocess_result(n_pages=1))

    cells = r.pages[0].tables[0]["cells"]
    anchor = next(c for c in cells if c["row_id"] == 0 and c["col_id"] == 0)
    unit = next(c for c in cells if c["row_id"] == 1 and c["col_id"] == 0)
    assert anchor["col_span"] == 2
    assert "col_span" not in unit


def test_flat_text_fallback_when_vlm_omits_table_tag():
    # Table HTML block with flat <p> text → warning + flat text in first cell.
    TABLE_BBOX = [10.0, 60.0, 200.0, 150.0]

    layout_bboxes = [_make_layout_bbox_mock("Table")]
    layout_bboxes[0].bbox = TABLE_BBOX
    layout_result = MagicMock()
    layout_result.error = False
    layout_result.bboxes = layout_bboxes
    layout_pred = MagicMock(return_value=[layout_result])

    table_block = _make_block_mock(0, label="Table")
    table_block.bbox = TABLE_BBOX
    table_block.html = "<p>flat text only</p>"
    page_ocr = MagicMock()
    page_ocr.blocks = [table_block]
    rec_pred = MagicMock(return_value=[page_ocr])

    with patch("khmer_pipeline.engines.surya._get_predictors",
               return_value=(layout_pred, rec_pred)):
        r = run_surya(_make_preprocess_result(n_pages=1))

    assert any("flat text" in w for w in r.warnings)
    cells = r.pages[0].tables[0]["cells"]
    assert cells[0]["text_lines"][0]["text"] == "flat text only"


def test_bbox_tolerance_matches_offset_bbox():
    """_find_matching_html matches a slightly offset bbox but rejects a far one."""
    table_html_map = {(10, 60, 200, 150): "<table><tr><td>x</td></tr></table>"}
    assert _find_matching_html([10.4, 60.1, 200.2, 150.1], table_html_map) == (
        "<table><tr><td>x</td></tr></table>"
    )
    assert _find_matching_html([100.0, 60.0, 290.0, 150.0], table_html_map) == ""


def test_table_cells_not_shifted_by_extra_html_row():
    # Regression: colspan title row must not shift data rows' col indices.
    TABLE_BBOX = [10.0, 60.0, 200.0, 150.0]

    layout_bboxes = [_make_layout_bbox_mock("Table")]
    layout_bboxes[0].bbox = TABLE_BBOX
    layout_result = MagicMock()
    layout_result.error = False
    layout_result.bboxes = layout_bboxes
    layout_pred = MagicMock(return_value=[layout_result])

    table_block = _make_block_mock(0, label="Table")
    table_block.bbox = TABLE_BBOX
    table_block.html = (
        '<table>'
        '<tr><td colspan="3">Title</td></tr>'
        '<tr><td>A</td><td>B</td><td>C</td></tr>'
        '</table>'
    )
    page_ocr = MagicMock()
    page_ocr.blocks = [table_block]
    rec_pred = MagicMock(return_value=[page_ocr])

    with patch("khmer_pipeline.engines.surya._get_predictors",
               return_value=(layout_pred, rec_pred)):
        r = run_surya(_make_preprocess_result(n_pages=1))

    cells = r.pages[0].tables[0]["cells"]
    by_pos = {(c["row_id"], c["col_id"]): c for c in cells}

    # Title in row 0 col 0 with colspan padding in cols 1, 2
    assert by_pos[(0, 0)]["text_lines"][0]["text"] == "Title"
    assert by_pos[(0, 1)]["text_lines"] == []  # empty padded slot
    assert by_pos[(0, 2)]["text_lines"] == []  # empty padded slot

    # Data row at row 1
    assert by_pos[(1, 0)]["text_lines"][0]["text"] == "A"
    assert by_pos[(1, 1)]["text_lines"][0]["text"] == "B"
    assert by_pos[(1, 2)]["text_lines"][0]["text"] == "C"


def test_skip_tables_drops_table_regions_before_recognition():
    """skip_tables=True strips Table bboxes before rec_pred so no table HTML is
    produced (surya_kiri rebuilds tables itself and would otherwise pay for
    Surya's table VLM pass for nothing)."""
    layout_pred, rec_pred = _make_predictors(with_table=True)
    layout_result = layout_pred.return_value[0]
    pil_img = MagicMock()

    page = _process_page(0, pil_img, layout_pred, rec_pred, skip_tables=True)

    call_kwargs = rec_pred.call_args[1]
    passed_layout_results = call_kwargs["layout_results"]
    assert all(
        b.label != "Table"
        for lr in passed_layout_results
        for b in lr.bboxes
    )
    assert page.tables == []


def test_skip_tables_false_passes_table_region_through():
    """Companion check: with skip_tables=False (default), a Table region is
    still passed to rec_pred so table HTML continues to be produced."""
    layout_pred, rec_pred = _make_predictors(with_table=True)

    page = _process_page(0, MagicMock(), layout_pred, rec_pred, skip_tables=False)

    call_kwargs = rec_pred.call_args[1]
    passed_layout_results = call_kwargs["layout_results"]
    assert any(
        b.label == "Table"
        for lr in passed_layout_results
        for b in lr.bboxes
    )
    assert len(page.tables) == 1


# --- Track A: fine-tuned layout detector replaces Surya's Table boxes (§2.43) ---
# When KHMER_LAYOUT_WEIGHTS is set, _process_page swaps Surya's Table-labelled
# layout boxes for detect_table_boxes() output (deterministic; §2.37 wants this),
# leaving every other label untouched. One integration point serves all 3 engines.

def _table_labels(rec_pred):
    lr = rec_pred.call_args[1]["layout_results"][0]
    return [b for b in lr.bboxes if b.label == "Table"]


def test_yolo_layout_replaces_surya_table_boxes(monkeypatch):
    monkeypatch.setenv("KHMER_LAYOUT_DETECTOR", "rapid")
    layout_pred, rec_pred = _make_predictors(with_table=True)
    with patch("khmer_pipeline.engines.surya.detect_table_boxes",
               return_value=[[1.0, 2.0, 3.0, 4.0]]) as dtb:
        _process_page(0, MagicMock(), layout_pred, rec_pred, skip_tables=False)
    dtb.assert_called_once()
    tables = _table_labels(rec_pred)
    assert len(tables) == 1
    assert list(tables[0].bbox) == [1.0, 2.0, 3.0, 4.0]


def test_yolo_layout_leaves_non_table_boxes_untouched(monkeypatch):
    monkeypatch.setenv("KHMER_LAYOUT_DETECTOR", "rapid")
    layout_pred, rec_pred = _make_predictors(with_table=True)
    with patch("khmer_pipeline.engines.surya.detect_table_boxes",
               return_value=[[1.0, 2.0, 3.0, 4.0]]):
        _process_page(0, MagicMock(), layout_pred, rec_pred, skip_tables=False)
    lr = rec_pred.call_args[1]["layout_results"][0]
    texts = [b for b in lr.bboxes if b.label == "Text"]
    assert len(texts) == 1
    assert list(texts[0].bbox) == [10.0, 10.0, 200.0, 50.0]


def test_yolo_layout_empty_result_keeps_surya_boxes(monkeypatch):
    # Empty YOLO detection must NOT drop the table — fall back to Surya's box + warn.
    monkeypatch.setenv("KHMER_LAYOUT_DETECTOR", "rapid")
    layout_pred, rec_pred = _make_predictors(with_table=True)
    with patch("khmer_pipeline.engines.surya.detect_table_boxes", return_value=[]):
        _process_page(0, MagicMock(), layout_pred, rec_pred, skip_tables=False)
    assert len(_table_labels(rec_pred)) == 1


def test_no_layout_env_does_not_call_detector(monkeypatch):
    monkeypatch.delenv("KHMER_LAYOUT_DETECTOR", raising=False)
    layout_pred, rec_pred = _make_predictors(with_table=True)
    with patch("khmer_pipeline.engines.surya.detect_table_boxes") as dtb:
        _process_page(0, MagicMock(), layout_pred, rec_pred, skip_tables=False)
    dtb.assert_not_called()


def test_run_surya_reports_sub_steps_per_page(monkeypatch):
    """The OCR stage is the long one: `on_step` reports what it is doing inside a
    page so the UI does not sit frozen on a single label for minutes."""
    import numpy as np
    from types import SimpleNamespace
    from khmer_pipeline.engines import surya as surya_mod
    from khmer_pipeline.models import PreprocessResult

    steps: list[str] = []

    def fake_process_page(idx, pil_img, layout_pred, rec_pred, skip_tables=False, on_step=None):
        if on_step is not None:
            on_step("layout")
            on_step("tables")
        return SimpleNamespace(page_index=idx, text_blocks=[], tables=[], ocr_text="")

    monkeypatch.setattr(surya_mod, "_get_predictors", lambda: (object(), object()))
    monkeypatch.setattr(surya_mod, "_process_page", fake_process_page)

    page = np.full((20, 20, 3), 255, dtype=np.uint8)
    result = PreprocessResult(source_name="x.pdf", page_images=[page], dpi=200, page_count=1)
    surya_mod.run_surya(result, on_step=steps.append)

    assert steps == ["layout", "tables"]
