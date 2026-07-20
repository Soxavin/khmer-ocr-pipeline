"""Tests for the auto-routing engine (auto_engine.py).

The router is a deterministic circuit-breaker: run surya_kiri (the ARDB specialist),
and only if its OWN per-cell confidence says it is struggling on this document
(low-confidence cell fraction over a conservative cutoff) fall back to surya. The
delegate engines are monkeypatched so these tests never load a model.
"""
from __future__ import annotations

from unittest.mock import patch
import pytest

from khmer_pipeline.models import SuryaResult, SuryaPageResult
import khmer_pipeline.engines.auto_engine as ae


def _cell(conf: float) -> dict:
    return {"row_id": 0, "col_id": 0, "cell_id": 0, "bbox": [], "polygon": [],
            "text_lines": [{"text": "x", "bbox": []}], "confidence": conf}


def _result(confs: list[float], source: str = "doc.pdf") -> SuryaResult:
    """A SuryaResult with one page whose table cells carry the given confidences."""
    table = {"rows": [{"row_id": 0}], "cols": [{"col_id": 0}],
             "cells": [_cell(c) for c in confs], "image_bbox": []}
    page = SuryaPageResult(page_index=0, text_blocks=[], tables=[table], ocr_text="")
    return SuryaResult(source_name=source, pages=[page])


def _preprocess_stub():
    # auto_engine never inspects the PreprocessResult itself (it delegates), so a
    # bare object suffices as the argument.
    return object()


# --- routing behaviour ---

def test_keeps_surya_kiri_when_confident():
    kiri = _result([0.99] * 10)  # 0% low-conf → well under cutoff
    with patch.object(ae, "run_surya_kiri", return_value=kiri) as rk, \
         patch.object(ae, "run_surya") as rs:
        out = ae.run_auto(_preprocess_stub())
    assert out is kiri
    rk.assert_called_once()
    # Latency penalty: surya must NOT run when surya_kiri is confident (common case).
    rs.assert_not_called()


def test_falls_back_to_surya_when_low_confidence():
    kiri = _result([0.1] * 10)   # 100% low-conf → over cutoff
    surya = _result([0.99] * 10, source="doc.pdf")
    with patch.object(ae, "run_surya_kiri", return_value=kiri) as rk, \
         patch.object(ae, "run_surya", return_value=surya) as rs:
        out = ae.run_auto(_preprocess_stub())
    assert out is surya
    rk.assert_called_once()
    rs.assert_called_once()


def test_fallback_emits_machine_readable_warning():
    kiri = _result([0.1] * 10)
    surya = _result([0.99] * 10)
    with patch.object(ae, "run_surya_kiri", return_value=kiri), \
         patch.object(ae, "run_surya", return_value=surya):
        out = ae.run_auto(_preprocess_stub())
    joined = " ".join(out.warnings)
    assert "[AutoRouter] fallback surya_kiri->surya" in joined
    assert "frac=" in joined and "cutoff=" in joined


def test_kept_decision_emits_machine_readable_warning():
    kiri = _result([0.99] * 10)
    with patch.object(ae, "run_surya_kiri", return_value=kiri), \
         patch.object(ae, "run_surya"):
        out = ae.run_auto(_preprocess_stub())
    joined = " ".join(out.warnings)
    assert "[AutoRouter] kept surya_kiri" in joined
    assert "frac=" in joined and "cutoff=" in joined


def test_low_conf_fraction_pools_all_cells_across_pages():
    # Fraction is computed over ALL table cells in the result (document-level),
    # so a doc that is half-failing still crosses the cutoff.
    frac = ae._low_conf_fraction(_result([0.99, 0.99, 0.1, 0.1, 0.1]))  # 3/5 low
    assert frac == pytest.approx(0.6)


def test_low_conf_fraction_no_cells_is_zero():
    # No table cells (e.g. a text-only page) → 0.0, never a fallback trigger.
    empty = SuryaResult(source_name="d", pages=[
        SuryaPageResult(page_index=0, text_blocks=[], tables=[], ocr_text="")])
    assert ae._low_conf_fraction(empty) == 0.0


def test_cutoff_is_between_measured_ardb_and_budget():
    # Step-1 measurement: worst ARDB page 0.222, budget p3 0.539. The cutoff must
    # sit strictly between so ARDB never falls back and budget always does.
    assert 0.222 < ae._FALLBACK_LOW_CONF_FRACTION < 0.539
