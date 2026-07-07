from __future__ import annotations
from khmer_pipeline.engines.engine_registry import get_ocr_engine
from khmer_pipeline.engines.surya import run_surya
from khmer_pipeline.engines.surya_kiri_engine import run_surya_kiri


def test_get_ocr_engine_returns_surya_kiri():
    assert get_ocr_engine("surya_kiri") is run_surya_kiri


def test_get_ocr_engine_returns_surya():
    assert get_ocr_engine("surya") is run_surya


def test_get_ocr_engine_unknown_falls_back_to_surya():
    assert get_ocr_engine("not_a_real_engine") is run_surya
