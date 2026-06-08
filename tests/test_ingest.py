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
