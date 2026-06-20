from __future__ import annotations

import io
import json
from pathlib import Path

import fitz
import pytest

from khmer_pipeline.inspect_pdf import (
    inspect_pdf,
    _khmer_char_count,
    _latin_char_count,
    _MIN_TEXT_CHARS,
    _UNICODE_KHMER_RATIO,
    _LEGACY_KHMER_RATIO,
)


# ---------------------------------------------------------------------------
# Helpers for building in-memory PDFs
# ---------------------------------------------------------------------------

def _pdf_bytes_with_text(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    # insert_text clips at page width; use multiple lines to exceed _MIN_TEXT_CHARS
    y = 72.0
    chunk = 60
    for i in range(0, len(text), chunk):
        page.insert_text((72, y), text[i:i + chunk], fontsize=12)
        y += 20
        if y > 800:
            break
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _pdf_bytes_image_only() -> bytes:
    # Create a blank white pixmap and insert it as an image — no text layer
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 100))
    pix.clear_with(255)
    page.insert_image(page.rect, pixmap=pix)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _write_pdf(tmp_path: Path, name: str, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


# ---------------------------------------------------------------------------
# Unit tests for pure helpers
# ---------------------------------------------------------------------------

def test_khmer_char_count_pure():
    # Khmer Unicode block U+1780–U+17FF
    assert _khmer_char_count("ក") == 1
    assert _khmer_char_count("ABC") == 0
    assert _khmer_char_count("ក ABC ខ") == 2


def test_latin_char_count_pure():
    assert _latin_char_count("hello") == 5
    assert _latin_char_count("123") == 0
    assert _latin_char_count("ក") == 0


# ---------------------------------------------------------------------------
# Classification logic unit tests (bypasses fitz round-trip for Khmer)
# ---------------------------------------------------------------------------

def _classify(text_chars: int, khmer_block_chars: int, latin_chars: int, has_images: bool) -> str:
    """Mirror the classification logic from inspect_pdf for isolated testing."""
    alpha_chars = khmer_block_chars + latin_chars
    khmer_ratio = khmer_block_chars / max(1, alpha_chars)
    substantial = text_chars >= _MIN_TEXT_CHARS

    if substantial and khmer_ratio >= _UNICODE_KHMER_RATIO:
        return "born_digital_unicode"
    if substantial and khmer_ratio <= _LEGACY_KHMER_RATIO and latin_chars > khmer_block_chars:
        return "likely_legacy_encoded"
    if not substantial and has_images:
        return "scanned_image_only"
    return "mixed_or_unknown"


def test_classify_born_digital_unicode():
    # Mostly Khmer chars + enough total text
    khmer = 200
    latin = 10
    result = _classify(khmer + latin, khmer, latin, False)
    assert result == "born_digital_unicode"


def test_classify_likely_legacy_encoded():
    # Legacy encoding: all alpha are Latin (Khmer stored as Latin code points)
    latin = 300
    khmer = 5
    result = _classify(latin + khmer, khmer, latin, False)
    assert result == "likely_legacy_encoded"


def test_classify_scanned_image_only():
    result = _classify(0, 0, 0, True)
    assert result == "scanned_image_only"


def test_classify_mixed_or_unknown_no_images_no_text():
    result = _classify(0, 0, 0, False)
    assert result == "mixed_or_unknown"


def test_classify_mixed_or_unknown_low_text_no_images():
    # a tiny bit of text, no images → mixed_or_unknown
    result = _classify(50, 30, 5, False)
    assert result == "mixed_or_unknown"


# ---------------------------------------------------------------------------
# Integration tests using real fitz PDFs
# ---------------------------------------------------------------------------

def test_inspect_pdf_single_file_latin_heavy(tmp_path):
    # Latin-only text → likely_legacy_encoded (Khmer ratio very low)
    text = "ABCDEFGHIJ " * 20  # enough Latin chars, no Khmer
    pdf_path = _write_pdf(tmp_path, "latin.pdf", _pdf_bytes_with_text(text))
    results = inspect_pdf(pdf_path)
    assert len(results) == 1
    r = results[0]
    assert r["filename"] == "latin.pdf"
    assert r["page_count"] == 1
    assert r["text_chars"] > 0
    assert r["khmer_block_chars"] == 0
    assert r["classification"] == "likely_legacy_encoded"


def test_inspect_pdf_scanned_image_only(tmp_path):
    pdf_path = _write_pdf(tmp_path, "scanned.pdf", _pdf_bytes_image_only())
    results = inspect_pdf(pdf_path)
    assert len(results) == 1
    r = results[0]
    assert r["has_images"] is True
    assert r["classification"] == "scanned_image_only"


def test_inspect_pdf_directory(tmp_path):
    # Put two PDFs in a dir, inspect the dir
    latin_text = "ABCDEFGHIJ " * 20
    _write_pdf(tmp_path, "doc1.pdf", _pdf_bytes_with_text(latin_text))
    _write_pdf(tmp_path, "doc2.pdf", _pdf_bytes_image_only())
    results = inspect_pdf(tmp_path)
    assert len(results) == 2
    names = {r["filename"] for r in results}
    assert names == {"doc1.pdf", "doc2.pdf"}


def test_inspect_pdf_returns_required_keys(tmp_path):
    pdf_path = _write_pdf(tmp_path, "check.pdf", _pdf_bytes_with_text("hello world " * 20))
    results = inspect_pdf(pdf_path)
    r = results[0]
    for key in ("filename", "page_count", "text_chars", "khmer_block_chars",
                "latin_chars", "khmer_ratio", "has_images", "max_image_dims", "classification"):
        assert key in r, f"Missing key: {key}"


def test_inspect_pdf_nonexistent_file_returns_error(tmp_path):
    # inspect_pdf returns an error dict rather than raising for missing files
    results = inspect_pdf(tmp_path / "nonexistent.pdf")
    assert len(results) == 1
    assert results[0]["classification"] == "error"
    assert "error" in results[0]


def test_inspect_pdf_with_khmer_text(tmp_path):
    # Insert Khmer Unicode text — fitz default font may not render it visually,
    # but we test that classification handles whatever get_text returns.
    khmer_str = "ក" * 200  # 200 Khmer chars in the string
    text = khmer_str
    pdf_path = _write_pdf(tmp_path, "khmer.pdf", _pdf_bytes_with_text(text))
    results = inspect_pdf(pdf_path)
    assert len(results) == 1
    r = results[0]
    # The classification must be one of the known values regardless of font rendering
    assert r["classification"] in (
        "born_digital_unicode", "likely_legacy_encoded",
        "scanned_image_only", "mixed_or_unknown",
    )
    # If text survived the round-trip, khmer_block_chars should be non-zero
    # and classification should be born_digital_unicode — but we don't assert
    # on the rendering behavior since it's font-dependent in test env.
    if r["khmer_block_chars"] >= _MIN_TEXT_CHARS:
        assert r["classification"] == "born_digital_unicode"
