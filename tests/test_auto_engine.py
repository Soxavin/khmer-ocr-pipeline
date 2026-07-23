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


class _PreStub:
    """Minimal stand-in for PreprocessResult carrying only what the router reads."""
    def __init__(self, low_res_scan: bool = False):
        self.low_res_scan = low_res_scan


def _preprocess_stub(low_res_scan: bool = False):
    return _PreStub(low_res_scan=low_res_scan)


# --- pre-flight routing: low-res scans go straight to Surya ---
# §2.75 proved Kiri's self-confidence cannot detect a low-res scan (frac 0.222
# where it wins vs 0.231 where it fails — 0.009 apart). But the scan is knowable
# from the source BEFORE running anything, so the router shortcuts on it: skip
# Kiri entirely, run Surya, at zero extra inference.

def test_low_res_scan_routes_straight_to_surya_without_running_kiri():
    surya = _result([0.99] * 10)
    with patch.object(ae, "run_surya_kiri") as rk, \
         patch.object(ae, "run_surya", return_value=surya) as rs:
        out = ae.run_auto(_preprocess_stub(low_res_scan=True))
    assert out is surya
    rk.assert_not_called()   # the whole point: no wasted per-cell pass
    rs.assert_called_once()


def test_low_res_scan_decision_is_machine_readable():
    surya = _result([0.99] * 10)
    with patch.object(ae, "run_surya_kiri"), \
         patch.object(ae, "run_surya", return_value=surya):
        out = ae.run_auto(_preprocess_stub(low_res_scan=True))
    joined = " ".join(out.warnings)
    assert "[AutoRouter] pre-flight surya (low-res scan)" in joined


def test_non_scan_still_runs_the_confidence_path():
    # A born-digital doc (low_res_scan False) must reach the existing Kiri path.
    kiri = _result([0.99] * 10)
    with patch.object(ae, "run_surya_kiri", return_value=kiri) as rk, \
         patch.object(ae, "run_surya") as rs:
        out = ae.run_auto(_preprocess_stub(low_res_scan=False))
    assert out is kiri
    rk.assert_called_once()
    rs.assert_not_called()


def test_router_tolerates_result_without_the_flag():
    # Older PreprocessResults / bare stubs lacking low_res_scan must not crash —
    # absence means "unknown", i.e. take the confidence path.
    kiri = _result([0.99] * 10)
    with patch.object(ae, "run_surya_kiri", return_value=kiri), \
         patch.object(ae, "run_surya"):
        out = ae.run_auto(object())
    assert out is kiri


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
