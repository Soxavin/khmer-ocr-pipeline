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


# ── Auto-DPI resolution (§2.68) ──────────────────────────────────────────────
# "auto" picks a render DPI by inspecting the PDF's embedded-image density:
# high-density/vector → 200 (enough, faster); faint/low-res scans → 300 (more
# pixels per Khmer glyph, prioritising accuracy over speed).

def _make_scanned_pdf(img_w: int, img_h: int, page_w: int = 595, page_h: int = 842) -> bytes:
    """A one-page PDF whose full-page content is a raster image of img_w×img_h px.
    Native density ≈ img_w / (page_w/72) DPI."""
    doc = fitz.open()
    page = doc.new_page(width=page_w, height=page_h)
    pil = Image.new("RGB", (img_w, img_h), color=(180, 180, 180))
    buf = io.BytesIO(); pil.save(buf, format="PNG")
    page.insert_image(page.rect, stream=buf.getvalue())
    return doc.tobytes()


def test_auto_dpi_high_density_scan_uses_200():
    from khmer_pipeline.ingest import resolve_auto_dpi
    # 595pt ≈ 8.26in; 2480px / 8.26in ≈ 300 DPI native → clean, 200 suffices.
    pdf = _make_scanned_pdf(2480, 3508)
    assert resolve_auto_dpi(pdf, "clean.pdf") == 200


def test_auto_dpi_low_density_scan_falls_back_to_300():
    from khmer_pipeline.ingest import resolve_auto_dpi
    # 850px / 8.26in ≈ 103 DPI native → faint/low-res, upscale to 300 for OCR.
    pdf = _make_scanned_pdf(850, 1100)
    assert resolve_auto_dpi(pdf, "faint.pdf") == 300


def test_auto_dpi_born_digital_pdf_uses_200():
    from khmer_pipeline.ingest import resolve_auto_dpi
    # Vector text, no embedded raster → nothing to upscale.
    assert resolve_auto_dpi(_make_pdf(), "vector.pdf") == 200


def test_auto_dpi_image_input_uses_200():
    from khmer_pipeline.ingest import resolve_auto_dpi
    # Images ingest at native pixels regardless; dpi is not a render knob for them.
    assert resolve_auto_dpi(_make_png(), "scan.png") == 200


def test_auto_dpi_worst_page_drives_decision():
    from khmer_pipeline.ingest import resolve_auto_dpi
    # One faint page among clean ones is enough to warrant the higher DPI.
    doc = fitz.open()
    for w, h in ((2480, 3508), (850, 1100)):
        page = doc.new_page(width=595, height=842)
        pil = Image.new("RGB", (w, h), color=(180, 180, 180))
        buf = io.BytesIO(); pil.save(buf, format="PNG")
        page.insert_image(page.rect, stream=buf.getvalue())
    assert resolve_auto_dpi(doc.tobytes(), "mixed.pdf") == 300


# Real-world regression: ARDB bulletins are born-digital text pages that embed a
# small masthead logo. Reading the density of that LOGO (≈50 DPI) instead of a
# page scan made "auto" return 300 for every document — the setting never once
# chose 200. A page is only a scan when a raster actually covers it.

def _make_pdf_with_logo(logo_w: int = 499, logo_h: int = 142) -> bytes:
    """Born-digital text page carrying a small low-resolution logo, as ARDB does."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 200), "Born-digital Khmer table content", fontsize=12)
    pil = Image.new("RGB", (logo_w, logo_h), color=(120, 120, 120))
    buf = io.BytesIO(); pil.save(buf, format="PNG")
    # Placed small, near the top — a masthead, not a page scan.
    page.insert_image(fitz.Rect(72, 40, 272, 97), stream=buf.getvalue())
    return doc.tobytes()


def test_auto_dpi_ignores_a_small_logo_on_a_born_digital_page():
    from khmer_pipeline.ingest import resolve_auto_dpi
    # The logo is low-density, but the PAGE is vector text — upscaling buys nothing.
    assert resolve_auto_dpi(_make_pdf_with_logo(), "ardb.pdf") == 200


def test_page_is_scanned_true_for_full_page_raster():
    import fitz as _f
    from khmer_pipeline.ingest import page_is_scanned
    doc = _f.open(stream=_make_scanned_pdf(850, 1100), filetype="pdf")
    assert page_is_scanned(doc[0]) is True
    doc.close()


def test_page_is_scanned_false_for_logo_page():
    import fitz as _f
    from khmer_pipeline.ingest import page_is_scanned
    doc = _f.open(stream=_make_pdf_with_logo(), filetype="pdf")
    assert page_is_scanned(doc[0]) is False
    doc.close()


def test_page_is_scanned_false_for_pure_vector():
    import fitz as _f
    from khmer_pipeline.ingest import page_is_scanned
    doc = _f.open(stream=_make_pdf(), filetype="pdf")
    assert page_is_scanned(doc[0]) is False
    doc.close()
