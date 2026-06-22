from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from khmer_pipeline.models import PreprocessResult, SuryaResult
from khmer_pipeline.tesseract_engine import (
    _import_pytesseract,
    _dicts_to_words,
    _build_text_blocks,
    run_tesseract,
)


# --- helpers ---

def _make_preprocess_result(n_pages: int = 1) -> PreprocessResult:
    img = np.zeros((50, 100, 3), dtype=np.uint8)
    return PreprocessResult(
        source_name="fake.pdf",
        page_images=[img.copy() for _ in range(n_pages)],
        dpi=200,
        page_count=n_pages,
    )


# Two lines, two words each. Mixed Khmer + Latin.
TWO_LINE_DATA: dict = {
    "block_num": [1, 1, 1, 1],
    "par_num":   [1, 1, 1, 1],
    "line_num":  [1, 1, 2, 2],
    "left":      [10, 50, 10, 60],
    "top":       [10, 10, 30, 30],
    "width":     [30, 20, 40, 30],
    "height":    [15, 15, 15, 15],
    "conf":      [90, 85, 88, 92],
    "text":      ["ខ្មែរ", "ស", "hello", "world"],
}


def _make_fake_pytesseract(words_data: dict) -> SimpleNamespace:
    return SimpleNamespace(
        image_to_data=Mock(return_value=words_data),
        Output=SimpleNamespace(DICT="dict"),
    )


@pytest.fixture
def fake_pytesseract(monkeypatch):
    """Replace _import_pytesseract with one that returns a fake module."""
    fake = _make_fake_pytesseract(TWO_LINE_DATA)
    monkeypatch.setattr(
        "khmer_pipeline.tesseract_engine._import_pytesseract",
        lambda: fake,
    )
    return fake


# --- _import_pytesseract ---

def test_import_pytesseract_raises_clear_error(monkeypatch):
    # Force the helper to raise the documented ImportError
    def fake_import():
        raise ImportError(
            "pytesseract is required for the Tesseract engine. "
            "Install with: brew install tesseract tesseract-lang"
        )
    monkeypatch.setattr(
        "khmer_pipeline.tesseract_engine._import_pytesseract", fake_import
    )
    with pytest.raises(ImportError, match="brew install tesseract"):
        run_tesseract(_make_preprocess_result())


# --- _dicts_to_words ---

def test_dicts_to_words_repacks_parallel_lists():
    out = _dicts_to_words(TWO_LINE_DATA)
    assert len(out) == 4
    assert out[0]["text"] == "ខ្មែរ"
    assert out[0]["block_num"] == 1
    assert out[2]["line_num"] == 2


def test_dicts_to_words_handles_none_text():
    data = {
        "text":      ["a", None, "c"],
        "conf":      [80, 80, 80],
        "left":      [0, 10, 20],
        "top":       [0, 0, 0],
        "width":     [10, 10, 10],
        "height":    [10, 10, 10],
        "block_num": [1, 1, 1],
        "par_num":   [1, 1, 1],
        "line_num":  [1, 1, 1],
    }
    out = _dicts_to_words(data)
    assert out[1]["text"] == ""


# --- _build_text_blocks ---

def test_build_text_blocks_groups_words_into_lines():
    blocks = _build_text_blocks(_dicts_to_words(TWO_LINE_DATA))
    assert len(blocks) == 2
    assert blocks[0]["text"] == "ខ្មែរ ស"
    assert blocks[1]["text"] == "hello world"


def test_build_text_blocks_emits_required_keys():
    blocks = _build_text_blocks(_dicts_to_words(TWO_LINE_DATA))
    required = {"text", "bbox", "polygon", "confidence", "label", "region_label", "reading_order"}
    for b in blocks:
        assert required.issubset(b.keys())


def test_build_text_blocks_drops_empty_lines():
    data = {
        "text":      ["", "", "x"],
        "conf":      [0, 0, 90],
        "left":      [0, 0, 0],
        "top":       [0, 0, 10],
        "width":     [10, 10, 10],
        "height":    [10, 10, 10],
        "block_num": [1, 1, 1],
        "par_num":   [1, 1, 1],
        "line_num":  [1, 1, 2],
    }
    blocks = _build_text_blocks(_dicts_to_words(data))
    assert len(blocks) == 1
    assert blocks[0]["text"] == "x"


def test_build_text_blocks_bbox_geometry():
    blocks = _build_text_blocks(_dicts_to_words(TWO_LINE_DATA))
    # Line 1 words: left=10/50, top=10, width 30/20, height 15
    # → min_left=10, min_top=10, max_right=50+20=70, max_bottom=10+15=25
    assert blocks[0]["bbox"] == [10, 10, 70, 25]


def test_build_text_blocks_polygon_is_bbox_corners():
    blocks = _build_text_blocks(_dicts_to_words(TWO_LINE_DATA))
    l, t, r, b = blocks[0]["bbox"]
    assert blocks[0]["polygon"] == [[l, t], [r, t], [r, b], [l, b]]


def test_build_text_blocks_confidence_normalized_to_0_1():
    blocks = _build_text_blocks(_dicts_to_words(TWO_LINE_DATA))
    for b in blocks:
        assert 0.0 <= b["confidence"] <= 1.0


def test_build_text_blocks_ignores_negative_conf():
    # Tesseract emits conf=-1 for non-text regions — must be excluded from the mean.
    data = dict(TWO_LINE_DATA)
    data["conf"] = [-1, 90, -1, 95]
    blocks = _build_text_blocks(_dicts_to_words(data))
    # Line 1: only the 90 word contributes → 0.9
    assert blocks[0]["confidence"] == pytest.approx(0.9)


def test_build_text_blocks_reading_order_assigned():
    blocks = _build_text_blocks(_dicts_to_words(TWO_LINE_DATA))
    assert [b["reading_order"] for b in blocks] == [0, 1]


def test_build_text_blocks_label_is_text():
    blocks = _build_text_blocks(_dicts_to_words(TWO_LINE_DATA))
    for b in blocks:
        assert b["label"] == "Text"
        assert b["region_label"] == "Text"


# --- run_tesseract (integration) ---

def test_run_tesseract_returns_surya_result(fake_pytesseract):
    r = run_tesseract(_make_preprocess_result())
    assert isinstance(r, SuryaResult)


def test_run_tesseract_source_name_preserved(fake_pytesseract):
    r = run_tesseract(_make_preprocess_result())
    assert r.source_name == "fake.pdf"


def test_run_tesseract_page_count_matches(fake_pytesseract):
    r = run_tesseract(_make_preprocess_result(n_pages=3))
    assert len(r.pages) == 3
    assert [p.page_index for p in r.pages] == [0, 1, 2]


def test_run_tesseract_text_blocks_contain_words(fake_pytesseract):
    r = run_tesseract(_make_preprocess_result())
    blocks = r.pages[0].text_blocks
    assert len(blocks) == 2
    all_text = " ".join(b["text"] for b in blocks)
    assert "ខ្មែរ" in all_text
    assert "hello" in all_text


def test_run_tesseract_text_blocks_have_required_keys(fake_pytesseract):
    r = run_tesseract(_make_preprocess_result())
    required = {"text", "bbox", "polygon", "confidence", "label", "region_label", "reading_order"}
    for b in r.pages[0].text_blocks:
        assert required.issubset(b.keys())


def test_run_tesseract_tables_empty(fake_pytesseract):
    # Tesseract yields no table structure — documented outcome.
    r = run_tesseract(_make_preprocess_result())
    assert r.pages[0].tables == []


def test_run_tesseract_ocr_text_contains_words(fake_pytesseract):
    r = run_tesseract(_make_preprocess_result())
    text = r.pages[0].ocr_text
    assert "ខ្មែរ" in text
    assert "hello" in text


def test_run_tesseract_on_page_callback_invoked(fake_pytesseract):
    calls: list[tuple[int, int]] = []
    run_tesseract(_make_preprocess_result(n_pages=3),
                  on_page=lambda i, n: calls.append((i, n)))
    assert calls == [(0, 3), (1, 3), (2, 3)]


def test_run_tesseract_swallows_page_errors(monkeypatch):
    # image_to_data raises on the first page, succeeds on the second.
    call_count = [0]

    def flaky(img, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("simulated tesseract failure")
        return TWO_LINE_DATA

    fake = SimpleNamespace(
        image_to_data=flaky,
        Output=SimpleNamespace(DICT="dict"),
    )
    monkeypatch.setattr(
        "khmer_pipeline.tesseract_engine._import_pytesseract",
        lambda: fake,
    )
    r = run_tesseract(_make_preprocess_result(n_pages=2))
    assert len(r.pages) == 2
    assert r.pages[0].text_blocks == []  # errored → empty
    assert r.pages[0].ocr_text == ""
    assert len(r.pages[1].text_blocks) == 2  # succeeded
    assert any("Page 1 failed" in w for w in r.warnings)
