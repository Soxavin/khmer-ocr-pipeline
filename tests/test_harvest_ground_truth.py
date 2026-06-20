from __future__ import annotations

import io
import json
from pathlib import Path

import fitz
import pytest

from khmer_pipeline.harvest_ground_truth import harvest


# ---------------------------------------------------------------------------
# Helper: build a tiny born-digital PDF with real text blocks
# ---------------------------------------------------------------------------

def _make_text_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    # Use insert_textbox for better block structure in get_text("blocks")
    rect = fitz.Rect(72, 72, 500, 200)
    page.insert_textbox(rect, text, fontsize=12)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _make_multi_page_pdf(texts: list[str]) -> bytes:
    doc = fitz.open()
    for text in texts:
        page = doc.new_page()
        rect = fitz.Rect(72, 72, 500, 200)
        page.insert_textbox(rect, text, fontsize=12)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_harvest_produces_png_and_json(tmp_path):
    pdf_path = tmp_path / "invoice.pdf"
    pdf_path.write_bytes(_make_text_pdf("Hello world paragraph text"))

    out_dir = tmp_path / "out"
    written = harvest(pdf_path, out_dir, dpi=72)

    written_names = {p.name for p in written}
    assert "invoice_p1.png" in written_names
    assert "invoice_p1_ground_truth.json" in written_names


def test_harvest_png_exists_and_nonzero(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(_make_text_pdf("Some text content here"))

    out_dir = tmp_path / "out"
    harvest(pdf_path, out_dir, dpi=72)

    png = out_dir / "doc_p1.png"
    assert png.exists()
    assert png.stat().st_size > 0


def test_harvest_ground_truth_json_schema(tmp_path):
    pdf_path = tmp_path / "report.pdf"
    pdf_path.write_bytes(_make_text_pdf("Paragraph one content"))

    out_dir = tmp_path / "out"
    harvest(pdf_path, out_dir, dpi=72)

    gt_path = out_dir / "report_p1_ground_truth.json"
    assert gt_path.exists()
    gt = json.loads(gt_path.read_text())

    # Required keys
    for key in ("font_family", "template", "document_type", "paragraphs", "tables", "footer"):
        assert key in gt, f"Missing required key: {key}"

    assert gt["font_family"] == "real"
    assert gt["template"] == "report"
    assert gt["document_type"] == "real"
    assert isinstance(gt["paragraphs"], list)
    assert gt["tables"] == []
    assert gt["footer"] == ""


def test_harvest_paragraphs_nonempty(tmp_path):
    pdf_path = tmp_path / "filled.pdf"
    pdf_path.write_bytes(_make_text_pdf("This is a real paragraph with enough text content"))

    out_dir = tmp_path / "out"
    harvest(pdf_path, out_dir, dpi=72)

    gt = json.loads((out_dir / "filled_p1_ground_truth.json").read_text())
    assert len(gt["paragraphs"]) > 0


def test_harvest_multi_page(tmp_path):
    pdf_path = tmp_path / "multi.pdf"
    pdf_path.write_bytes(_make_multi_page_pdf(["Page one text", "Page two text", "Page three text"]))

    out_dir = tmp_path / "out"
    written = harvest(pdf_path, out_dir, dpi=72)

    written_names = {p.name for p in written}
    # 3 pages × 2 files each = 6 total
    assert len(written) == 6
    assert "multi_p1.png" in written_names
    assert "multi_p2.png" in written_names
    assert "multi_p3.png" in written_names
    assert "multi_p1_ground_truth.json" in written_names
    assert "multi_p2_ground_truth.json" in written_names
    assert "multi_p3_ground_truth.json" in written_names


def test_harvest_creates_output_dir(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(_make_text_pdf("hello world"))

    out_dir = tmp_path / "nested" / "deep" / "out"
    assert not out_dir.exists()
    harvest(pdf_path, out_dir, dpi=72)
    assert out_dir.exists()


def test_harvest_returns_path_list(tmp_path):
    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(_make_text_pdf("some text content here"))

    out_dir = tmp_path / "out"
    result = harvest(pdf_path, out_dir, dpi=72)

    assert isinstance(result, list)
    assert all(isinstance(p, Path) for p in result)


def test_harvest_json_nfc_normalized(tmp_path):
    # NFC normalization: the text is already ASCII here but the code path runs
    import unicodedata
    pdf_path = tmp_path / "nfc.pdf"
    pdf_path.write_bytes(_make_text_pdf("Normalize this text content"))

    out_dir = tmp_path / "out"
    harvest(pdf_path, out_dir, dpi=72)

    gt = json.loads((out_dir / "nfc_p1_ground_truth.json").read_text())
    for para in gt["paragraphs"]:
        assert para == unicodedata.normalize("NFC", para)
