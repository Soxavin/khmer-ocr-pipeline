"""Tests for the shared ARDB fine-tune region -> SuryaPageResult transform.

Both models emit {"box_2d": [y1,x1,y2,x2] (0-1000 grid), "label", "text"} per
region; Table regions carry their text as HTML (Gemma) or markdown (Qwen). No
subprocess/model here — pure data-shape tests."""
from __future__ import annotations

from khmer_pipeline.engines.finetune_ardb.transform import regions_to_page


def test_table_html_becomes_a_table():
    regions = [
        {"box_2d": [0, 0, 100, 1000], "label": "Table",
         "text": "<table><tr><td>a</td><td>1</td></tr></table>"},
    ]
    page = regions_to_page(regions, page_index=0, table_text_format="html",
                           page_w=200, page_h=100)
    assert len(page.tables) == 1
    cells = page.tables[0]["cells"]
    texts = {(c["row_id"], c["col_id"]): c["text_lines"][0]["text"] for c in cells if c["text_lines"]}
    assert texts == {(0, 0): "a", (0, 1): "1"}


def test_table_markdown_becomes_a_table():
    md = "| a | b |\n|---|---|\n| 1 | 2 |"
    regions = [{"box_2d": [0, 0, 100, 1000], "label": "Table", "text": md}]
    page = regions_to_page(regions, page_index=0, table_text_format="markdown",
                           page_w=200, page_h=100)
    assert len(page.tables) == 1
    cells = page.tables[0]["cells"]
    texts = {(c["row_id"], c["col_id"]): c["text_lines"][0]["text"] for c in cells if c["text_lines"]}
    assert texts[(1, 0)] == "1"
    assert texts[(1, 1)] == "2"


def test_non_table_regions_become_text_blocks():
    regions = [
        {"box_2d": [0, 0, 50, 500], "label": "Section-Header", "text": "Title"},
        {"box_2d": [50, 0, 100, 500], "label": "Text", "text": "body"},
    ]
    page = regions_to_page(regions, page_index=0, table_text_format="html",
                           page_w=1000, page_h=1000)
    assert page.tables == []
    assert {b["text"] for b in page.text_blocks} == {"Title", "body"}


def test_picture_regions_carry_no_text():
    regions = [{"box_2d": [0, 0, 100, 100], "label": "Picture", "text": ""}]
    page = regions_to_page(regions, page_index=0, table_text_format="html",
                           page_w=100, page_h=100)
    assert page.text_blocks == []
    assert page.tables == []


def test_ocr_text_pools_prose_not_table_cells():
    regions = [
        {"box_2d": [0, 0, 10, 10], "label": "Text", "text": "prose"},
        {"box_2d": [10, 0, 20, 10], "label": "Table",
         "text": "<table><tr><td>cell</td></tr></table>"},
    ]
    page = regions_to_page(regions, page_index=0, table_text_format="html",
                           page_w=100, page_h=100)
    assert "prose" in page.ocr_text


def test_box_2d_rescaled_from_1000_grid_to_pixels():
    # box_2d is [y1, x1, y2, x2] on a 0-1000 grid; must rescale to actual pixel bbox.
    regions = [{"box_2d": [0, 0, 500, 1000], "label": "Text", "text": "half"}]
    page = regions_to_page(regions, page_index=0, table_text_format="html",
                           page_w=200, page_h=100)
    block = page.text_blocks[0]
    # y: 0-500/1000 * 100 = 0-50; x: 0-1000/1000 * 200 = 0-200
    assert block["bbox"] == [0.0, 0.0, 200.0, 50.0]  # [x1, y1, x2, y2]


def test_malformed_box_2d_degrades_to_empty_bbox():
    regions = [{"box_2d": "garbage", "label": "Text", "text": "x"}]
    page = regions_to_page(regions, page_index=0, table_text_format="html",
                           page_w=100, page_h=100)
    assert page.text_blocks[0]["bbox"] == []


def test_missing_text_field_treated_as_empty():
    regions = [{"box_2d": [0, 0, 10, 10], "label": "Text"}]
    page = regions_to_page(regions, page_index=0, table_text_format="html",
                           page_w=100, page_h=100)
    assert page.text_blocks == []  # empty text -> not pooled as a block
