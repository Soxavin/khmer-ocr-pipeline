"""Tests for HITL correction capture (corrections.py).

Turns an analyst's verified table edits into (cell crop, corrected text) training
pairs for the Kiri fine-tune path. The curation rules matter as much as the
capture: a pair that is not a genuine recognition error is training noise.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from khmer_pipeline import corrections as co


def _cell(row: int, col: int, text: str, bbox=None, conf: float = 0.9,
          flags=None) -> dict:
    cell = {
        "row_id": row, "col_id": col, "cell_id": row * 10 + col,
        "bbox": bbox if bbox is not None else [10.0, 10.0, 60.0, 40.0],
        "polygon": [],
        "text_lines": [{"text": text, "bbox": []}] if text else [],
        "confidence": conf,
    }
    if flags:
        cell["flags"] = flags
    return cell


def _table(cells: list[dict]) -> dict:
    return {"rows": [], "cols": [], "cells": cells,
            "bbox": [0.0, 0.0, 200.0, 200.0], "image_bbox": []}


def _page_image() -> np.ndarray:
    # Distinctive gradient so a crop is visibly non-uniform if written out.
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:, :, 0] = np.arange(200, dtype=np.uint8)[None, :]
    return img


def _capture(tables, edited, tmp_path, **kw):
    return co.capture_corrections(
        tables=tables, edited_grids=edited, page_images=[_page_image()],
        source_name="ardb.pdf", out_dir=tmp_path, **kw)


# --- diff: only genuine changes become pairs ---

def test_unchanged_cells_produce_no_pairs(tmp_path):
    tables = [_table([_cell(0, 0, "៛/គ.ក")])]
    recs = _capture(tables, {"0": [["៛/គ.ក"]]}, tmp_path)
    assert recs == []


def test_real_glyph_fix_produces_one_pair(tmp_path):
    # The measured §2.33 error class: ៛ misread as អ.
    tables = [_table([_cell(0, 0, "អ/គ.ក", conf=0.41, flags=["low_conf"])])]
    recs = _capture(tables, {"0": [["៛/គ.ក"]]}, tmp_path)
    assert len(recs) == 1
    assert recs[0]["text"] == "៛/គ.ក"
    assert recs[0]["origin"] == "correction"


# --- curation: cosmetic edits must NOT become training pairs ---

def test_whitespace_only_edit_is_dropped(tmp_path):
    tables = [_table([_cell(0, 0, "7 800")])]
    recs = _capture(tables, {"0": [["7  800"]]}, tmp_path)
    assert recs == []


def test_zero_width_joiner_only_difference_is_dropped(tmp_path):
    """ZWJ/ZWNJ are deliberately PRESERVED by khmer_normalize (they affect Khmer
    shaping), so two visually identical strings differing only by a joiner would
    otherwise slip through the cosmetic filter and become bogus training data."""
    base = "ស​ម"          # with ZWSP
    edited = "ស‌ម"        # ZWNJ instead
    tables = [_table([_cell(0, 0, base)])]
    recs = _capture(tables, {"0": [[edited]]}, tmp_path)
    assert recs == [], "joiner-only differences are not recognition errors"


def test_unicode_normalization_only_difference_is_dropped(tmp_path):
    # NFD vs NFC of the same text is not a recognition error.
    import unicodedata
    text = "ខ្មែរ"
    tables = [_table([_cell(0, 0, unicodedata.normalize("NFC", text))])]
    recs = _capture(tables, {"0": [[unicodedata.normalize("NFD", text)]]}, tmp_path)
    assert recs == []


# --- gold-standard rule + robustness ---

def test_cell_without_geometry_is_skipped_not_crashed(tmp_path):
    # surya (VLM) cells have no per-cell bbox; a save must never crash on them.
    tables = [_table([_cell(0, 0, "wrong", bbox=[])])]
    recs = _capture(tables, {"0": [["right"]]}, tmp_path)
    assert recs == []


def test_degenerate_bbox_is_skipped(tmp_path):
    tables = [_table([_cell(0, 0, "wrong", bbox=[10.0, 10.0, 10.0, 10.0])])]
    recs = _capture(tables, {"0": [["right"]]}, tmp_path)
    assert recs == []


# --- outputs: crop written, provenance nested ---

def test_crop_png_is_written_and_non_empty(tmp_path):
    tables = [_table([_cell(0, 0, "អ/គ.ក")])]
    recs = _capture(tables, {"0": [["៛/គ.ក"]]}, tmp_path)
    img_path = tmp_path / recs[0]["image"]
    assert img_path.exists() and img_path.stat().st_size > 0


def test_provenance_is_a_nested_object_with_filterable_fields(tmp_path):
    """Nested (not flat) so a future retrain can filter by flags without a
    data-cleaning pass — e.g. 'train only on sequence_illegal'."""
    tables = [_table([_cell(2, 3, "អ/គ.ក", conf=0.41,
                            flags=["sequence_illegal", "low_conf"])])]
    recs = _capture(tables, {"0": [[""] * 4] * 2 + [["", "", "", "៛/គ.ក"]]}, tmp_path)
    assert len(recs) == 1
    prov = recs[0]["provenance"]
    assert isinstance(prov, dict)
    assert prov["prediction"] == "អ/គ.ក"
    assert prov["flags"] == ["sequence_illegal", "low_conf"]
    assert prov["confidence"] == pytest.approx(0.41)
    assert prov["source"] == "ardb.pdf"
    assert prov["row"] == 2 and prov["col"] == 3
    assert prov["engine"] == "surya_kiri"
    assert "timestamp" in prov


def test_jsonl_is_appended_and_build_trainset_shaped(tmp_path):
    # Top-level keys must match build_trainset.py's schema so it ingests unchanged.
    tables = [_table([_cell(0, 0, "អ/គ.ក")])]
    _capture(tables, {"0": [["៛/គ.ក"]]}, tmp_path)
    _capture(tables, {"0": [["៛/ផ្លែ"]]}, tmp_path)
    lines = (tmp_path / "corrections.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, "records append across sessions, never overwrite"
    rec = json.loads(lines[0])
    assert set(["image", "text", "origin"]).issubset(rec)


def test_capture_is_noop_when_no_edits_supplied(tmp_path):
    tables = [_table([_cell(0, 0, "៛/គ.ក")])]
    assert _capture(tables, {}, tmp_path) == []
    assert not (tmp_path / "corrections.jsonl").exists()
