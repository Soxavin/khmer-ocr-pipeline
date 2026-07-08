from __future__ import annotations
import io
import numpy as np
import pytest
import fitz
from PIL import Image

from khmer_pipeline.models import IngestResult
from khmer_pipeline.ingest import ingest, MAX_PAGES


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_pdf(n_pages: int = 1) -> bytes:
    doc = fitz.open()
    for _ in range(n_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), "Test page", fontsize=12)
    return doc.tobytes()


def _make_png(width: int = 100, height: int = 80) -> bytes:
    img = Image.new("RGB", (width, height), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── tests ─────────────────────────────────────────────────────────────────────

def test_pdf_returns_ingest_result():
    result = ingest(_make_pdf(), "test.pdf")
    assert isinstance(result, IngestResult)


def test_pdf_page_count_single():
    result = ingest(_make_pdf(1), "test.pdf")
    assert result.page_count == 1
    assert len(result.page_images) == 1


def test_pdf_page_count_multi():
    result = ingest(_make_pdf(3), "three.pdf")
    assert result.page_count == 3
    assert len(result.page_images) == 3


def test_pdf_images_are_rgb_uint8():
    result = ingest(_make_pdf(), "test.pdf")
    arr = result.page_images[0]
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint8
    assert arr.ndim == 3
    assert arr.shape[2] == 3


def test_pdf_dpi_stored():
    result = ingest(_make_pdf(), "test.pdf", dpi=300)
    assert result.dpi == 300


def test_pdf_default_dpi_is_200():
    result = ingest(_make_pdf(), "test.pdf")
    assert result.dpi == 200


def test_pdf_page_limit_raises():
    data = _make_pdf(MAX_PAGES + 1)
    with pytest.raises(ValueError, match="limit is"):
        ingest(data, "big.pdf")


def test_pdf_at_exact_limit_passes():
    data = _make_pdf(MAX_PAGES)
    result = ingest(data, "edge.pdf")
    assert result.page_count == MAX_PAGES


def test_image_png_wraps_to_single_page():
    result = ingest(_make_png(100, 80), "scan.png")
    assert result.page_count == 1
    assert len(result.page_images) == 1


def test_image_dimensions_preserved():
    result = ingest(_make_png(100, 80), "scan.png")
    arr = result.page_images[0]
    assert arr.shape == (80, 100, 3)


def test_image_dpi_is_zero():
    result = ingest(_make_png(), "scan.png")
    assert result.dpi == 0


def test_source_name_stored():
    result = ingest(_make_pdf(), "ardb_sample.pdf")
    assert result.source_name == "ardb_sample.pdf"


def test_unsupported_format_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        ingest(b"fake", "document.docx")


# ── page-selective ingest (B4) ─────────────────────────────────────────────────

def test_page_indices_renders_only_selected():
    result = ingest(_make_pdf(5), "five.pdf", page_indices=[1, 3])
    assert result.page_count == 2
    assert len(result.page_images) == 2


def test_page_indices_none_renders_all():
    result = ingest(_make_pdf(4), "four.pdf", page_indices=None)
    assert result.page_count == 4


def test_page_indices_single_page():
    result = ingest(_make_pdf(10), "ten.pdf", page_indices=[7])
    assert result.page_count == 1


def test_page_indices_out_of_range_raises():
    with pytest.raises(ValueError, match="out of range"):
        ingest(_make_pdf(3), "three.pdf", page_indices=[5])


def test_page_indices_negative_raises():
    with pytest.raises(ValueError, match="out of range"):
        ingest(_make_pdf(3), "three.pdf", page_indices=[-1])


def test_page_indices_limit_applies_to_rendered_count_not_doc_length():
    # A doc longer than MAX_PAGES is fine when only a few pages are selected.
    data = _make_pdf(MAX_PAGES + 5)
    result = ingest(data, "big.pdf", page_indices=[0, 1, 2])
    assert result.page_count == 3


def test_page_indices_selection_is_reindexed_zero_based():
    # Selecting page index 2 yields a single-page result whose only image is that page.
    full = ingest(_make_pdf(4), "four.pdf")
    sel = ingest(_make_pdf(4), "four.pdf", page_indices=[2])
    assert np.array_equal(sel.page_images[0], full.page_images[2])


def test_page_indices_ignored_for_images():
    # Image inputs are single-page; page_indices is ignored, not an error.
    result = ingest(_make_png(100, 80), "scan.png", page_indices=[3, 4])
    assert result.page_count == 1


# ── multi-frame TIFF (B7) ──────────────────────────────────────────────────────

def _make_multiframe_tiff(n_frames: int = 3, size: tuple = (40, 30)) -> bytes:
    frames = [Image.new("RGB", size, color=(i * 40, i * 20, i * 10)) for i in range(n_frames)]
    buf = io.BytesIO()
    frames[0].save(buf, format="TIFF", save_all=True, append_images=frames[1:])
    return buf.getvalue()


def test_multiframe_tiff_returns_all_frames_as_pages():
    result = ingest(_make_multiframe_tiff(3), "scan.tiff")
    assert result.page_count == 3
    assert len(result.page_images) == 3


def test_multiframe_tiff_frames_are_distinct_rgb():
    result = ingest(_make_multiframe_tiff(3, size=(40, 30)), "scan.tiff")
    assert result.page_images[0].shape == (30, 40, 3)
    assert result.page_images[0].dtype == np.uint8
    # distinct fills → frames must differ (no silent page loss/dup)
    assert not np.array_equal(result.page_images[0], result.page_images[1])
    assert not np.array_equal(result.page_images[1], result.page_images[2])


def test_single_frame_tiff_is_one_page():
    single = _make_multiframe_tiff(1, size=(20, 10))
    result = ingest(single, "one.tiff")
    assert result.page_count == 1
    assert len(result.page_images) == 1


def test_single_frame_png_byte_identical_to_direct_convert():
    # Single-frame images must be unchanged by the multi-frame iteration.
    data = _make_png(100, 80)
    result = ingest(data, "scan.png")
    expected = np.array(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)
    assert np.array_equal(result.page_images[0], expected)


def test_multiframe_tiff_over_limit_raises():
    data = _make_multiframe_tiff(MAX_PAGES + 1, size=(4, 4))
    with pytest.raises(ValueError, match="limit is"):
        ingest(data, "big.tiff")
